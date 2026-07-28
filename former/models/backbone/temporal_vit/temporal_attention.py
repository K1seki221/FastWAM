"""Causal per-patch temporal attention for the Short-Term Memory (STM) module.

The layer is wrapped in a *warm-start residual* — at initialization it is
numerically the identity, and a per-channel learnable scale grows the
contribution as the model finds it useful. Concretely:

* ``self.temporal_pos_enc`` enters via ``normed`` (the LayerNorm output that
  feeds Q/K/V), **not** via the residual stream. Without this, history
  frames' V-projection picks up a sinusoidal pattern whose norm is
  comparable to the ViT hidden state (~32 for dim=1024) and leaks into the
  current-frame residual through the attention's weighted sum. PE is
  there to bias attention scores by relative time — not to overwrite
  feature content.
* ``self.gamma_attn`` / ``self.gamma_ffn`` (LayerScale, CaiT / DeiT-III style)
  are per-channel learnable diagonal scales replacing the scalar gates.
  Init at ``layer_scale_init`` (default 1e-6) → residual contribution is
  numerically zero at step 0, so the STM block strictly equals the base
  ViT block at init regardless of K, regardless of the rest of the layer
  weights. Per-channel resolution lets the model open STM only on the
  channels it actually benefits.

Train-time regularizers still extend the base layer:

* ``gate_dropout`` — DDP-safe scale-based dropout of the whole layer residual.
* ``use_cmtd``    — Content-Modulated Temporal Decay: per-sample scaler on a
  shared learnable distance-decay curve, added as an additive bias to scores.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTemporalEncoding(nn.Module):
    """Fixed sinusoidal encoding for temporal frame offsets."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, time_offsets: torch.Tensor) -> torch.Tensor:
        t = time_offsets.float().unsqueeze(1)
        # cos(x)-1 (not cos(x)) so e(0)=0, making K=1 numerically identical to base ViT.
        return torch.cat([
            torch.sin(t * self.inv_freq),
            torch.cos(t * self.inv_freq) - 1.0,
        ], dim=-1)


class TemporalAttentionLayer(nn.Module):
    """Causal temporal attention with a LayerScale-gated residual.

    Warm-start invariant: with ``layer_scale_init=0`` (or any sufficiently
    small value, default 1e-6) the layer's output is numerically equal to
    its input for any sequence length K. The base ViT's behaviour is
    therefore unaffected at step 0, and the model only diverges from
    baseline as ``gamma_attn`` / ``gamma_ffn`` grow during training.
    """

    def __init__(
        self,
        dim: int = 1152,
        num_heads: int = 16,
        dropout: float = 0.1,
        layer_scale_init: float = 1e-6,
        gate_dropout: float = 0.0,
        use_cmtd: bool = False,
        cmtd_max_distance: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.gate_dropout = gate_dropout
        self.use_cmtd = use_cmtd
        self.cmtd_max_distance = cmtd_max_distance
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        # Per-channel LayerScale (CaiT / DeiT-III). Init at ``layer_scale_init``
        # gives a numerically-zero residual at step 0 while the gradient
        # through ``gamma * attn_out`` stays non-zero, so the parameter
        # can grow during training.
        self.gamma_attn = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.gamma_ffn = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.temporal_pos_enc = SinusoidalTemporalEncoding(dim)

        if use_cmtd:
            # Linear 0..max_distance: the untrained ALiBi-style recency prior.
            self.distance_decay = nn.Parameter(
                torch.arange(cmtd_max_distance + 1, dtype=torch.float32),
            )
            # softplus(log(e-1)) = 1, so the per-sample λ is exactly 1 at init
            # ⇒ behaves like a plain shared decay curve until lambda_proj trains away.
            self.lambda_proj = nn.Linear(dim, 1)
            nn.init.zeros_(self.lambda_proj.weight)
            nn.init.constant_(self.lambda_proj.bias, math.log(math.e - 1.0))

    def _build_cmtd_bias(
        self, time_offsets: torch.Tensor, content: torch.Tensor, num_patches: int,
    ) -> torch.Tensor:
        """(B*N*num_heads, K, K) additive bias = -λ(content) · decay(|Δt|)."""
        K = time_offsets.shape[0]
        BN = content.shape[0]
        B = BN // num_patches

        d = (time_offsets.unsqueeze(0) - time_offsets.unsqueeze(1)).abs()
        d_clamped = d.clamp(max=self.cmtd_max_distance).long()
        base = self.distance_decay[d_clamped]  # (K, K)

        # Current-frame pooled-over-patches features → per-sample λ.
        current = content.reshape(B, num_patches, K, -1)[:, :, -1, :].mean(dim=1)  # (B, D)
        lam = F.softplus(self.lambda_proj(current))  # (B, 1)

        bias = -lam.unsqueeze(-1) * base.unsqueeze(0)  # (B, K, K)
        bias = bias.unsqueeze(1).expand(B, num_patches, K, K).reshape(BN, K, K)
        return bias.repeat_interleave(self.num_heads, dim=0)

    def _combine_attn_mask(
        self, cmtd_bias: torch.Tensor, causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        K = causal_mask.shape[-1]
        float_causal = torch.zeros(
            K, K, device=cmtd_bias.device, dtype=cmtd_bias.dtype,
        ).masked_fill(causal_mask, float('-inf'))
        return cmtd_bias + float_causal

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        time_offsets: torch.Tensor | None = None,
        num_patches: int | None = None,
    ) -> torch.Tensor:
        """x: (B*N, K, D). Returns same shape with LayerScale-gated temporal residual.

        Output = x + γ_attn · attn(norm(x) + PE(t)) + γ_ffn · ffn(x)

        ``causal_mask`` is required: cached single-frame inference appends
        history frames before this layer, so a non-causal call here would let
        the current frame attend to future cached entries during training but
        not at inference (or vice-versa), silently breaking train/test
        consistency. Pass ``build_causal_mask(K, device)`` from
        :class:`TemporalViTEncoder` and let downstream code stay honest.
        """
        assert causal_mask is not None, (
            'causal_mask is required for temporal attention to keep the '
            'training path and the cache-based inference path consistent.'
        )
        if self.use_cmtd and (num_patches is None or time_offsets is None):
            raise ValueError(
                'num_patches and time_offsets are required when use_cmtd=True '
                '(needed to pool content features per sample for λ and to compute |Δt|).',
            )
        normed = self.norm(x)
        # PE enters only via the attention-input route. Adding it to ``x``
        # directly would leak a sinusoid-shaped vector through V into the
        # residual stream regardless of gating — exactly the failure that
        # made K>1 disrupt the pretrained ViT distribution at step 0.
        if time_offsets is not None:
            normed = normed + self.temporal_pos_enc(time_offsets)

        if self.use_cmtd:
            assert time_offsets is not None and num_patches is not None  # narrowed by the check above
            attn_mask = self._combine_attn_mask(
                self._build_cmtd_bias(time_offsets, x, num_patches), causal_mask,
            )
        else:
            attn_mask = causal_mask
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)

        # DDP-safe scale-based dropout of the entire layer residual.
        # Sampled once per forward call; same across ranks is unnecessary
        # because each rank processes disjoint samples.
        if self.training and self.gate_dropout > 0.0:
            drop = torch.rand((), device=x.device).item() < self.gate_dropout
            scale = 0.0 if drop else 1.0 / (1.0 - self.gate_dropout)
        else:
            scale = 1.0
        x = x + self.gamma_attn * attn_out * scale
        x = x + self.gamma_ffn * self.ffn(x) * scale
        return x
