# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.normalization import RMSNorm
from torch import nn


class RawKVAttention(nn.Module):
    """
    SDPA-based attention that directly reuses pre-computed VLM KV cache.

    Avoids the wasteful pipeline: VLM KV -> hidden states -> re-project to KV.
    Uses F.scaled_dot_product_attention for Flash Attention / Memory Efficient dispatch.

    Two modes controlled by ``concat_dit_kv`` (determined by ``vlm_fusion_mode``):

    - True (context fusion — DiT tokens see both VLM and each other):
            Q: to_q(dit)                      [B, H, seq_dit, D]
            K: cat(vlm_key,  to_k(dit))       [B, H, seq_vlm + seq_dit, D]
            V: cat(vlm_val,  to_v(dit))       [B, H, seq_vlm + seq_dit, D]

    - False (cross fusion — DiT tokens only see VLM):
            Q: to_q(dit)    [B, H, seq_dit, D]
            K: vlm_key      [B, H, seq_vlm, D]
            V: vlm_val      [B, H, seq_vlm, D]

    Requirements:
        VLM num_kv_heads must equal DiT num_heads, and head_dim must match.
        For Qwen3-VL-2B (8 KV heads, head_dim=128) with DiT (8 heads, head_dim=128): exact match.
    """

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        concat_dit_kv: bool = True,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.concat_dit_kv = concat_dit_kv
        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        if concat_dit_kv:
            self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
            self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)

        self.to_out = nn.Linear(inner_dim, query_dim, bias=out_bias)
        self.dropout_p = dropout

        # QK-Norm: normalize Q and K before computing attention scores
        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == 'rms_norm':
            self.norm_q = RMSNorm(head_dim, eps=1e-6)
            self.norm_k = RMSNorm(head_dim, eps=1e-6)
        elif qk_norm == 'layer_norm':
            self.norm_q = nn.LayerNorm(head_dim)
            self.norm_k = nn.LayerNorm(head_dim)
        else:
            raise ValueError(f'RawKVAttention does not support qk_norm={qk_norm}')

    def _reshape_to_head(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_key_cache: Optional[torch.Tensor] = None,
        vlm_value_cache: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: DiT tokens [B, seq_dit, dim]
            vlm_key_cache: VLM key [B, num_kv_heads, seq_vlm, head_dim] or None
            vlm_value_cache: VLM value [B, num_kv_heads, seq_vlm, head_dim] or None
            attention_mask: SDPA mask [B, 1, 1, seq_kv] where True=attend, False=mask

        Returns:
            Output tensor [B, seq_dit, dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        query = self._reshape_to_head(self.to_q(hidden_states), batch_size, seq_len)

        # Apply QK-Norm to query
        if self.norm_q is not None:
            query = self.norm_q(query)

        key: Optional[torch.Tensor]
        value: Optional[torch.Tensor]
        if self.concat_dit_kv:
            # Context fusion: K/V = cat(VLM_KV, DiT_KV) → [B, H, seq_vlm + seq_dit, D]
            dit_key = self._reshape_to_head(self.to_k(hidden_states), batch_size, seq_len)
            dit_value = self._reshape_to_head(self.to_v(hidden_states), batch_size, seq_len)
            # Apply QK-Norm to DiT key before concatenation
            if self.norm_k is not None:
                dit_key = self.norm_k(dit_key)
            if vlm_key_cache is not None:
                assert vlm_value_cache is not None
                # VLM keys are already QK-Normed by Qwen3-VL backbone; skip norm to avoid double-normalization
                key = torch.cat([vlm_key_cache, dit_key], dim=2)
                value = torch.cat([vlm_value_cache, dit_value], dim=2)
            else:
                key = dit_key
                value = dit_value
        else:
            # Cross fusion: K/V = VLM_KV only → [B, H, seq_vlm, D]
            # VLM keys are already QK-Normed by Qwen3-VL backbone; skip norm to avoid double-normalization
            key = vlm_key_cache
            value = vlm_value_cache

        dropout_p = self.dropout_p if self.training else 0.0
        assert key is not None
        assert value is not None
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.to_out(attn_output)


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


def _build_raw_kv_mask(
    encoder_attention_mask: torch.Tensor,
    vlm_key_cache: torch.Tensor,
    hidden_states: torch.Tensor,
    concat_dit_kv: bool,
) -> Optional[torch.Tensor]:
    """
    Build SDPA-compatible attention mask for RawKVAttention.

    Args:
        encoder_attention_mask: [B, seq_vlm] mask for VLM tokens
        vlm_key_cache: [B, H, seq_vlm, D] VLM key cache (used for shape)
        hidden_states: [B, seq_dit, dim] DiT hidden states (used for shape)
        concat_dit_kv: whether DiT KV is concatenated with VLM KV

    Returns:
        SDPA mask [B, 1, 1, seq_kv] or None
    """
    vlm_mask = encoder_attention_mask.bool()
    if concat_dit_kv:
        # Context: mask covers [VLM, DiT] → [B, seq_vlm + seq_dit]
        dit_seq_len = hidden_states.shape[1]
        dit_mask = torch.ones(
            hidden_states.shape[0], dit_seq_len,
            device=hidden_states.device, dtype=torch.bool,
        )
        kv_mask = torch.cat([vlm_mask, dit_mask], dim=1)
    else:
        # Cross: mask covers VLM only → [B, seq_vlm]
        kv_mask = vlm_mask
    return kv_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, seq_kv]


class SeparateTransformerBlock(nn.Module):
    """
    Separate basic transformer block that supports single attention type per block.
    This is a cleaner alternative to CombinedTransformerBlock where each block handles
    either self-attention or cross-attention, not both.

    Structure:
    - Self mode: Self-Attention + AdaLayerNorm + skip [+ optional FFN + LayerNorm + skip]
    - Cross mode: Cross-Attention + AdaLayerNorm + skip + FFN + LayerNorm + skip

    When ``use_raw_kv=True`` and ``attention_mode='cross'``, the cross-attention layer
    is replaced with :class:`RawKVAttention` that directly consumes VLM KV cache.

    Args:
        attention_mode: 'self' or 'cross'
        has_ffn: Whether to include FFN after attention (only applies to 'self' mode,
                 cross mode always has FFN)
        use_raw_kv: If True, cross-attention blocks use RawKVAttention (SDPA)
                    instead of diffusers Attention.
        concat_dit_kv: If True, concatenate DiT KV with VLM KV (context fusion).
                       Only used when use_raw_kv=True.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        attention_mode: str = 'self',  # 'self' or 'cross'
        dropout=0.0,
        activation_fn: str = 'geglu',
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = 'ada_norm',
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        has_ffn: bool = True,
        use_raw_kv: bool = False,
        concat_dit_kv: bool = False,
        qk_norm: Optional[str] = None,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.norm_type = norm_type
        self.attention_mode = attention_mode
        self.has_ffn = has_ffn
        self.use_raw_kv = use_raw_kv and (attention_mode == 'cross')

        if attention_mode not in ['self', 'cross']:
            raise ValueError(f'attention_mode must be "self" or "cross", got {attention_mode}')

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                'If `positional_embedding` type is defined, `num_positional_embeddings` must also be defined.'
            )

        if positional_embeddings == 'sinusoidal':
            assert num_positional_embeddings is not None
            self.pos_embed: nn.Module | None = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Attention with AdaLayerNorm
        if norm_type == 'ada_norm':
            self.norm_attn: nn.Module = AdaLayerNorm(
                dim, norm_elementwise_affine=norm_elementwise_affine, norm_eps=norm_eps
            )
        else:
            self.norm_attn = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        # Create attention layer (self or cross)
        if self.use_raw_kv:
            self.attn: nn.Module = RawKVAttention(
                query_dim=dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                out_bias=attention_out_bias,
                concat_dit_kv=concat_dit_kv,
                qk_norm=qk_norm,
            )
        else:
            self.attn = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=(cross_attention_dim if cross_attention_dim is not None else dim)
                if attention_mode == 'cross' else None,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                qk_norm=qk_norm,
            )

        self.dropout_attn: Callable[[torch.Tensor], torch.Tensor]
        if final_dropout:
            self.dropout_attn = nn.Dropout(dropout)
        else:
            self.dropout_attn = lambda x: x

        # FFN with LayerNorm (always present for cross mode, optional for self mode)
        if attention_mode == 'cross' or has_ffn:
            self.norm_ffn = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
            self.ff = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        vlm_key_cache: Optional[torch.Tensor] = None,
        vlm_value_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention (self or cross)
        norm_hidden_states = (
            self.norm_attn(hidden_states, temb) if self.norm_type == 'ada_norm' else self.norm_attn(hidden_states)
        )

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        if self.use_raw_kv:
            # Raw KV cross-attention: Q from DiT, K/V depends on concat_dit_kv
            # kv_mask: [B, 1, 1, S_vlm] (cross) or [B, 1, 1, S_vlm + S_dit] (context)
            kv_mask = None
            if encoder_attention_mask is not None and vlm_key_cache is not None:
                kv_mask = _build_raw_kv_mask(
                    encoder_attention_mask, vlm_key_cache, hidden_states, self.attn.concat_dit_kv,
                )
            # attn_output: [B, S_dit, dim]
            attn_output = self.attn(
                norm_hidden_states,
                vlm_key_cache=vlm_key_cache,
                vlm_value_cache=vlm_value_cache,
                attention_mask=kv_mask,
            )
        elif self.attention_mode == 'self':
            attn_output = self.attn(
                norm_hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
            )
        else:
            attn_mask = encoder_attention_mask.bool() if encoder_attention_mask is not None else None
            attn_output = self.attn(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attn_mask,
            )

        attn_output = self.dropout_attn(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Feed-forward (always present for cross mode, optional for self mode)
        if self.attention_mode == 'cross' or self.has_ffn:
            norm_hidden_states = self.norm_ffn(hidden_states)
            ff_output = self.ff(norm_hidden_states)
            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        return hidden_states


def _top_k_renorm(w: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the ``k`` largest weights per row, zero the rest, renormalize to sum 1.

    ``k <= 0`` or ``k >= K`` is a no-op (dense mixing). Differentiable w.r.t. the
    kept logits; dropped experts receive zero weight (hence zero gradient) that
    step — standard top-k router behavior.
    """
    n_sources = w.shape[-1]
    if k <= 0 or k >= n_sources:
        return w
    _, top_idx = w.topk(k, dim=-1)
    keep = torch.zeros_like(w).scatter_(-1, top_idx, 1.0)
    w = w * keep
    return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)


class CrossOnlyTransformerBlock(nn.Module):
    """
    Single attention + FFN block without a dedicated self-attention layer.

    Used when ``dit_block_type='cross_only'``. Whether DiT tokens can see each
    other depends on ``concat_dit_kv`` (in raw KV mode) or ``vlm_fusion_mode``
    (in hidden state mode), not on this block itself.

    Architecture:
        hidden_states -> norm1 -> attention -> residual -> norm2 -> FFN -> residual -> output

    Token-level condition routing (v2, ``install_token_router``): each query
    token picks its own mix of the K VLM condition sources. Five input taps:
    ``post_adaln`` (v2.0) reads the post-AdaLN hidden ``x_hat`` DIRECTLY (no
    extra norm — a LayerNorm here would re-normalize away the AdaLN scale/shift
    and strip the timestep signal), so routing is timestep-aware but its
    gradient couples into AdaLN; ``pre_norm`` (v2.1) reads the raw pre-AdaLN
    residual through a router-private LayerNorm, fully decoupled from AdaLN;
    ``pre_norm_temb`` (v2.2) adds an explicit zero-init ``W_t*temb`` logit term
    (``logits_u = W_x*LN_r(x_u) + W_t*temb + b``) — content- and timestep-aware
    without touching AdaLN; ``post_adaln_norm`` (v2.3) reads ``x_hat`` through a
    router-private LayerNorm — completes the tap-point/router-LN 2x2 (still
    gradient-coupled to AdaLN, timestep modulation largely stripped);
    ``post_adaln_norm_temb`` (v2.4) is v2.3 plus the ``W_t*temb`` term. All K
    experts share the query, the prev-context and the mask; outputs are mixed
    per token before the residual.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = 'geglu',
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        use_raw_kv: bool = False,
        concat_dit_kv: bool = True,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim or dim
        self.norm_type = norm_type
        self.use_raw_kv = use_raw_kv

        # Single attention layer
        if norm_type == 'ada_norm':
            self.norm1: nn.Module = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        if use_raw_kv:
            self.attn: nn.Module = RawKVAttention(
                query_dim=dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                out_bias=attention_out_bias,
                concat_dit_kv=concat_dit_kv,
                qk_norm=qk_norm,
            )
        else:
            self.attn = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=self.cross_attention_dim,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                qk_norm=qk_norm,
            )

        # Feed-forward
        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # Dropout
        self.dropout_layer: Callable[[torch.Tensor], torch.Tensor]
        if final_dropout:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x

        # Token router (installed post-construction by DiT when token_router=True)
        self.token_router: Optional[nn.Linear] = None
        self.token_router_tap = 'post_adaln'
        # pre_norm / post_adaln_norm taps: router-private LayerNorm
        self.token_router_norm: Optional[nn.LayerNorm] = None
        # pre_norm_temb tap only: explicit timestep logit term (W_t * temb)
        self.token_router_temb: Optional[nn.Linear] = None
        self.token_router_temb_stopgrad = False
        self.token_router_top_k = 0
        self.token_router_temperature = 1.0
        # detached running mean of the last routing weights, for TB logging
        self.last_router_weights: Optional[torch.Tensor] = None
        self.last_router_weights_by_pos: Optional[torch.Tensor] = None

    def install_token_router(
        self, num_sources: int, incumbent_col: int, top_k: int, temperature: float, init_bias: float,
        tap: str = 'post_adaln', temb_stopgrad: bool = False,
    ) -> None:
        """Attach a per-token condition router. Zero-weight + identity-bias init
        reproduces this block's static (v1) incumbent distribution at step 0.

        tap='post_adaln' (v2.0) routes on the timestep-modulated x_hat;
        tap='pre_norm' (v2.1) routes on the raw pre-AdaLN residual through a
        router-private LayerNorm (inert at init: it feeds a zero matrix);
        tap='pre_norm_temb' (v2.2) adds an explicit zero-init W_t*temb logit
        term — timestep-aware routing without touching AdaLN;
        tap='post_adaln_norm' (v2.3) routes on x_hat through a router-private
        LayerNorm — the pre_norm/post_adaln 2x2 completer (still gradient-
        coupled to AdaLN; timestep modulation largely stripped by the LN);
        tap='post_adaln_norm_temb' (v2.4) is v2.3 plus the W_t*temb term."""
        assert tap in (
            'post_adaln', 'pre_norm', 'pre_norm_temb', 'post_adaln_norm', 'post_adaln_norm_temb'
        )
        router = nn.Linear(self.dim, num_sources, bias=True)
        nn.init.zeros_(router.weight)
        nn.init.zeros_(router.bias)
        with torch.no_grad():
            router.bias[incumbent_col] = init_bias
        self.token_router = router
        self.token_router_tap = tap
        # every tap except the raw post_adaln read gets the private LayerNorm
        self.token_router_norm = nn.LayerNorm(self.dim) if tap != 'post_adaln' else None
        if tap.endswith('_temb'):
            temb_proj = nn.Linear(self.dim, num_sources, bias=False)
            nn.init.zeros_(temb_proj.weight)
            self.token_router_temb = temb_proj
        else:
            self.token_router_temb = None
        self.token_router_temb_stopgrad = temb_stopgrad
        self.token_router_top_k = top_k
        self.token_router_temperature = temperature

    def forward_token_routed(
        self,
        hidden_states: torch.Tensor,       # [B, H, D] DiT tokens
        candidate_stack: torch.Tensor,     # [K, B, S_cond, D] the K condition sources (shared across blocks)
        candidate_mask: Optional[torch.Tensor],  # [B, S_cond] VLM validity mask (shared across experts)
        previous_output: Optional[torch.Tensor],  # [B, H, D] prev block output for context fusion
        temb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert self.token_router is not None and self.norm_type == 'ada_norm'
        x_hat = self.norm1(hidden_states, temb)  # [B, H, D] — the shared query

        if self.token_router_norm is not None:
            if self.token_router_tap.startswith('post_adaln_norm'):
                # post_adaln_norm taps: x_hat through the router's private
                # LayerNorm — scale-stabilized, timestep modulation largely
                # stripped (still gradient-coupled to AdaLN through x_hat)
                router_in = self.token_router_norm(x_hat)
            else:
                # pre_norm taps: route on the raw residual stream through the
                # router's private LayerNorm — no read of (or gradient into) AdaLN
                router_in = self.token_router_norm(hidden_states)
            logits = self.token_router(router_in)  # [B, H, K]
            if self.token_router_temb is not None:
                # *_temb taps: explicit timestep term, one logit offset for every
                # query token. temb_stopgrad=True detaches e_t so the routing loss
                # cannot reshape the shared TimestepEncoder (which feeds every
                # AdaLN); default False lets that gradient flow. W_t itself learns
                # either way.
                assert temb is not None
                t_in = temb.detach() if self.token_router_temb_stopgrad else temb
                logits = logits + self.token_router_temb(t_in).unsqueeze(1)  # [B, 1, K]
        else:
            # post_adaln tap: route on raw x_hat (no extra norm — it would strip
            # AdaLN's timestep scale/shift): timestep-aware, per token, per sample
            logits = self.token_router(x_hat)  # [B, H, K]
        w = torch.softmax(logits / self.token_router_temperature, dim=-1)
        w = _top_k_renorm(w, self.token_router_top_k)

        # build the shared prev-context prefix once (same across experts)
        b, h, _ = x_hat.shape
        if candidate_mask is not None:
            base_mask = candidate_mask.bool()
        else:
            base_mask = None
        if previous_output is not None:
            prev_ones = torch.ones(b, previous_output.shape[1], device=x_hat.device, dtype=torch.bool)
            expert_mask = (
                torch.cat([prev_ones, base_mask], dim=1) if base_mask is not None else None
            )
        else:
            expert_mask = base_mask

        expert_outputs = []
        for k in range(candidate_stack.shape[0]):
            enc = candidate_stack[k]  # [B, S_cond, D]
            if previous_output is not None:
                enc = torch.cat([previous_output, enc], dim=1)
            o_k = self.attn(x_hat, encoder_hidden_states=enc, attention_mask=expert_mask)
            if o_k.ndim == 4:
                o_k = o_k.squeeze(1)
            expert_outputs.append(o_k)
        stacked = torch.stack(expert_outputs, dim=2)  # [B, H, K, D]
        attn_output = (w.unsqueeze(-1) * stacked).sum(dim=2)  # [B, H, D]
        attn_output = self.dropout_layer(attn_output)

        if self.training:
            with torch.no_grad():
                self.last_router_weights = w.mean(dim=(0, 1)).detach()      # [K]
                self.last_router_weights_by_pos = w.mean(dim=0).detach()    # [H, K]

        hidden_states = attn_output + hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        vlm_key_cache: Optional[torch.Tensor] = None,
        vlm_value_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Current DiT tokens [B, seq_dit, dim]
            encoder_hidden_states: Context hidden states [B, seq_total, dim] (hidden state mode only)
            encoder_attention_mask: Mask for encoder_hidden_states
            attention_mask: Mask for hidden_states
            temb: Time embedding (for adaptive norm)
            vlm_key_cache: Raw VLM key [B, num_heads, seq_vlm, head_dim] (raw KV mode only)
            vlm_value_cache: Raw VLM value [B, num_heads, seq_vlm, head_dim] (raw KV mode only)

        Returns:
            Updated hidden_states [B, seq_dit, dim]
        """
        norm_hidden_states = (
            self.norm1(hidden_states, temb) if self.norm_type == 'ada_norm' else self.norm1(hidden_states)
        )

        if self.use_raw_kv:
            # Raw KV mode: build SDPA-compatible mask
            # kv_mask: [B, 1, 1, S_vlm] (cross) or [B, 1, 1, S_vlm + S_dit] (context)
            kv_mask = None
            if encoder_attention_mask is not None and vlm_key_cache is not None:
                vlm_seq_len = vlm_key_cache.shape[2]
                if encoder_attention_mask.shape[-1] == vlm_seq_len:
                    kv_mask = _build_raw_kv_mask(
                        encoder_attention_mask, vlm_key_cache, hidden_states, self.attn.concat_dit_kv,
                    )

            # attn_output: [B, S_dit, dim]
            attn_output = self.attn(
                norm_hidden_states,
                vlm_key_cache=vlm_key_cache,
                vlm_value_cache=vlm_value_cache,
                attention_mask=kv_mask,
            )
        else:
            # Hidden state mode: use diffusers Attention
            attn_mask = encoder_attention_mask if encoder_hidden_states is not None else attention_mask
            if attn_mask is not None:
                attn_mask = attn_mask.bool()

            attn_output = self.attn(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attn_mask,
            )

        attn_output = self.dropout_layer(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class FullAttentionBlock(nn.Module):
    """
    Joint self-attention block over concatenated [VLM, action] tokens.

    Used when ``dit_block_type='full_attention'``. Unlike other block types where
    VLM tokens are injected via cross-attention (every layer), this block receives
    the already-concatenated sequence and performs standard self-attention. VLM tokens
    are concatenated once before the block stack; individual blocks never receive
    separate encoder_hidden_states.

    Architecture:
        [B, S+T, D] -> AdaNorm -> Self-Attention -> residual -> LayerNorm -> FFN -> residual -> [B, S+T, D]
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        activation_fn: str = 'geglu',
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = 'ada_norm',
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                'If `positional_embedding` type is defined, `num_positional_embeddings` must also be defined.'
            )

        if positional_embeddings == 'sinusoidal':
            assert num_positional_embeddings is not None
            self.pos_embed: nn.Module | None = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Self-attention with adaptive layer norm
        if norm_type == 'ada_norm':
            self.norm1: nn.Module = AdaLayerNorm(
                dim, norm_elementwise_affine=norm_elementwise_affine, norm_eps=norm_eps
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,  # Self-attention only
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
        )

        # Feed-forward
        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.dropout: Callable[[torch.Tensor], torch.Tensor]
        if final_dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        # Unused — kept for unified block signature in DiT.forward()
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        vlm_key_cache: Optional[torch.Tensor] = None,
        vlm_value_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Concatenated [VLM, action] tokens [B, S+T, D]
            attention_mask: Joint mask [B, S+T] (True=attend, False=mask), or None
            temb: Timestep embedding [B, D]

        Returns:
            Updated tokens [B, S+T, D]
        """
        # Self-attention over all tokens
        norm_hidden_states = (
            self.norm1(hidden_states, temb) if self.norm_type == 'ada_norm' else self.norm1(hidden_states)
        )

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )
        attn_output = self.dropout(attn_output)
        hidden_states = attn_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class CombinedTransformerBlock(nn.Module):
    """
    Legacy combined block: Self + Cross + FFN in one block.
    Kept for backward compatibility.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = 'geglu',
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = 'default',
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                'If `positional_embedding` type is defined, `num_positional_embeddings` must also be defined.'
            )

        if positional_embeddings == 'sinusoidal':
            assert num_positional_embeddings is not None
            self.pos_embed: nn.Module | None = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == 'ada_norm':
            self.norm1: nn.Module = AdaLayerNorm(dim)
            self.norm2: nn.Module = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
        )
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.dropout1: Callable[[torch.Tensor], torch.Tensor]
        self.dropout2: Callable[[torch.Tensor], torch.Tensor]
        if final_dropout:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.dropout1 = lambda x: x
            self.dropout2 = lambda x: x

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        vlm_key_cache: Optional[torch.Tensor] = None,
        vlm_value_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 0. Self-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, temb) if self.norm_type == 'ada_norm' else self.norm1(hidden_states)
        )

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )
        attn_output = self.dropout1(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = (
            self.norm2(hidden_states, temb) if self.norm_type == 'ada_norm' else self.norm2(hidden_states)
        )

        # Use encoder_attention_mask for cross-attention over encoder hidden states
        # When doing cross-attention, we want to mask padding tokens in the encoder
        cross_attn_mask = encoder_attention_mask if encoder_hidden_states is not None else attention_mask
        # Convert attention mask to bool type if provided (0=mask, 1=attend -> False=mask, True=attend)
        if cross_attn_mask is not None:
            cross_attn_mask = cross_attn_mask.bool()

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=cross_attn_mask,
        )
        attn_output = self.dropout2(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states
