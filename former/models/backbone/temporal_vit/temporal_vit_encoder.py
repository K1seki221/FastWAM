"""Qwen3-VL vision encoder wrapped with per-patch temporal attention at DeepStack points.

The encoder reuses the underlying vision model's blocks by reference — only
the ``TemporalAttentionLayer`` instances introduce new parameters. Two public
code paths:

* ``_forward_multi_frame`` — training / batch inference: accepts ``B*K`` stacked
  frames, applies temporal attention per DeepStack point, returns only the
  current-frame features.
* ``_forward_with_cache`` — sequential single-sample inference. The cache
  stores pre-temporal-attention features so behaviour matches the training path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.temporal_sampling import build_history_offsets, compute_exponential_gaps

from .temporal_attention import TemporalAttentionLayer
from .temporal_feature_cache import TemporalFeatureCache


@dataclass
class TemporalViTOutput:
    """Output compatible with ``BaseModelOutputWithDeepstackFeatures``."""

    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    deepstack_features: list[torch.Tensor]


class TemporalViTEncoder(nn.Module):
    """Qwen3-VL ViT wrapper that inserts causal temporal attention at DeepStack points.

    Requires the underlying vision model to expose ``deepstack_visual_indexes``
    (the list of layer indices where per-patch temporal attention is inserted)
    and a Qwen3-VL-shaped ViT block interface.  Other backbones need their own
    adapter; a minimal sanity check on construction fails loudly instead of
    crashing deep in the forward pass.
    """

    def __init__(self, base_vision_model: nn.Module, stm_config):
        super().__init__()
        if not hasattr(base_vision_model, 'deepstack_visual_indexes'):
            raise TypeError(
                f'TemporalViTEncoder currently supports Qwen3-VL-shaped backbones with '
                f'``deepstack_visual_indexes``; got {type(base_vision_model).__name__}. '
                f'Disable STM (stm=null) or add a backbone-specific adapter.'
            )
        # Reference, not copy — weights are shared with the backbone.
        self.base_vision_model = base_vision_model
        self.stm_config = stm_config

        self.deepstack_indexes = list(base_vision_model.deepstack_visual_indexes)
        self._deepstack_map: dict[int, int] = {
            layer: idx for idx, layer in enumerate(self.deepstack_indexes)
        }

        vit_dim = base_vision_model.blocks[0].norm1.weight.shape[0]
        self.vit_dim = vit_dim

        self.temporal_attn_layers = nn.ModuleList([
            TemporalAttentionLayer(
                dim=vit_dim, num_heads=16, dropout=0.1,
                gate_dropout=getattr(stm_config, 'gate_dropout', 0.0),
                use_cmtd=getattr(stm_config, 'use_cmtd', False),
            )
            for _ in range(len(self.deepstack_indexes))
        ])

        self.cache: TemporalFeatureCache | None = None

        # Fallback offsets for the multi-frame training path when the caller
        # does not pass per-sample ``time_offsets`` explicitly.
        num_hist = getattr(stm_config, 'num_history_frames', 0)
        if num_hist > 0:
            gaps = compute_exponential_gaps(
                num_hist,
                getattr(stm_config, 'frame_interval', 1),
                getattr(stm_config, 'exp_ratio', 1.0),
            )
            default_offsets = build_history_offsets(gaps) + [0]
        else:
            default_offsets = [0]
        self.register_buffer(
            'default_time_offsets',
            torch.tensor(default_offsets, dtype=torch.long),
            persistent=False,
        )

    def enable_cache(
        self,
        max_frames: int | None = None,
        inference_freq: float | None = None,
        training_image_freq: float | None = None,
        slow_play_factor: float = 1.0,
    ) -> None:
        """Enable the inference cache.

        Builds two paired lists:

        * ``time_offsets`` — canonical training-rate labels, the integers
          ``SinusoidalTemporalEncoding`` was trained against. Always derived
          from ``stm_config.frame_interval`` / ``exp_ratio`` regardless of
          deploy rate or slow-play.
        * ``slot_offsets`` — FIFO positions to read; scaled by
          ``inference_freq / training_image_freq / slow_play_factor`` so the
          picked frames cover the same physical time window as training even
          when the deploy loop runs at a different rate or under slow-play.

        K=0 short-circuit: when the policy has no video history
        (``num_history_frames=0``, e.g. state-only configs), leave the cache
        off entirely. The forward path routes K=1 through
        ``_forward_multi_frame`` already, and materialising a 1-slot cache
        would silently start blending the previous frame as ``-1`` history
        on subsequent calls (K_total=2), diverging from the training-time
        single-frame self-attention.
        """
        K = max_frames if max_frames is not None else self.stm_config.num_history_frames
        if K <= 0:
            self.cache = None
            return
        exp_ratio = getattr(self.stm_config, 'exp_ratio', 1.0)
        base_interval = getattr(self.stm_config, 'frame_interval', 1)

        canonical_gaps = compute_exponential_gaps(K, base_interval, exp_ratio)
        time_offsets = build_history_offsets(canonical_gaps)

        freq_ratio = 1.0
        if inference_freq is not None and training_image_freq is not None:
            freq_ratio = inference_freq / training_image_freq
        scale = freq_ratio / slow_play_factor
        slot_gaps = (
            canonical_gaps if scale == 1.0
            else [max(1, round(g * scale)) for g in canonical_gaps]
        )
        slot_offsets = build_history_offsets(slot_gaps)

        # Clamp to >=1 so the deque stays usable in the degenerate K=0 case.
        cache_size = max(abs(slot_offsets[0]) if slot_offsets else K, 1)
        self.cache = TemporalFeatureCache(
            cache_size, len(self.deepstack_indexes),
            slot_offsets=slot_offsets or None,
            time_offsets=time_offsets or None,
        )

    def disable_cache(self) -> None:
        self.cache = None

    def reset_cache(self) -> None:
        """Clear cache contents. Call at episode boundaries."""
        if self.cache is not None:
            self.cache.reset()

    @staticmethod
    def build_causal_mask(K: int, device: torch.device) -> torch.Tensor:
        """(K, K) bool mask, True above the diagonal (frame t blocks frames >t)."""
        return torch.triu(torch.ones(K, K, device=device, dtype=torch.bool), diagonal=1)

    def _prepare_vit_inputs(self, pixel_values, grid_thw):
        base_vit = self.base_vision_model
        hidden_states = base_vit.patch_embed(pixel_values)
        pos_embeds = base_vit.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = base_vit.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        return hidden_states, position_embeddings, cu_seqlens

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        num_frames: int = 1,
        **kwargs,
    ):
        """Route to cached single-frame, or the (degenerate-K=1-tolerant) multi-frame path.

        K=1 is a degenerate but valid case for ``_forward_multi_frame``: the
        causal mask is (1, 1), attention is self-only, and SinusoidalTemporalEncoding
        at offset 0 returns 0 by design. Routing K=1 through the multi-frame
        path (rather than bypassing to ``base_vision_model``) keeps the STM
        parameters in the computation graph, so DDP with
        ``find_unused_parameters=False`` is happy even when the operator
        constructs STM but feeds it a single frame (e.g. state-history-only
        experiments where ``num_history_frames=0``).
        """
        K = num_frames

        if K <= 1 and self.cache is not None:
            return self._forward_with_cache(pixel_values, grid_thw, **kwargs)

        return self._forward_multi_frame(pixel_values, grid_thw, K, **kwargs)

    def _forward_multi_frame(self, pixel_values, grid_thw, K, **kwargs):
        base_vit = self.base_vision_model
        time_offsets = kwargs.pop('time_offsets', None)
        if time_offsets is None and self.default_time_offsets.numel() == K:
            time_offsets = self.default_time_offsets

        hidden_states, position_embeddings, cu_seqlens = self._prepare_vit_inputs(pixel_values, grid_thw)

        num_images = grid_thw.shape[0]
        patches_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        N = patches_per_image[0]
        assert all(n == N for n in patches_per_image), (
            f'V1 STM requires all frames to have identical grid_thw. '
            f'Got varying patch counts: {set(patches_per_image)}'
        )
        B = num_images // K
        assert num_images == B * K, (
            f'num_images ({num_images}) must be divisible by num_frames ({K})'
        )
        D = hidden_states.shape[-1]

        causal_mask = self.build_causal_mask(K, hidden_states.device)

        deepstack_feature_lists = []

        for layer_num, blk in enumerate(base_vit.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            temporal_idx = self._deepstack_map.get(layer_num, -1)
            if temporal_idx >= 0:
                hs = hidden_states.reshape(B, K, N, D)
                hs = hs.permute(0, 2, 1, 3).reshape(B * N, K, D)
                hs = self.temporal_attn_layers[temporal_idx](
                    hs, causal_mask, time_offsets=time_offsets, num_patches=N,
                )
                hidden_states = (
                    hs.reshape(B, N, K, D)
                    .permute(0, 2, 1, 3)
                    .reshape(B * K * N, D)
                )

                current_hs = hidden_states.reshape(B, K, N, D)[:, -1].reshape(B * N, D)
                deepstack_feature = base_vit.deepstack_merger_list[temporal_idx](current_hs)
                deepstack_feature_lists.append(deepstack_feature)

        current_hs = hidden_states.reshape(B, K, N, D)[:, -1].reshape(B * N, D)
        merged_hidden_states = base_vit.merger(current_hs)

        return TemporalViTOutput(
            last_hidden_state=current_hs,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )

    def _forward_with_cache(self, pixel_values, grid_thw, **kwargs):
        """Single-frame inference with cached history. V1: B=1 only.

        Cache stores pre-temporal-attention features so this path is
        mathematically equivalent to the multi-frame training path.
        """
        base_vit = self.base_vision_model

        hidden_states, position_embeddings, cu_seqlens = self._prepare_vit_inputs(pixel_values, grid_thw)

        B = grid_thw.shape[0]
        assert B == 1, (
            f'V1 STM cache inference only supports B=1 (single sample). Got B={B}. '
            f'For batch inference, disable cache and use the default path.'
        )
        N = (grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2]).item()
        D = hidden_states.shape[-1]

        deepstack_feature_lists = []

        hist_offsets = self.cache.get_history_time_offsets(0)
        all_offsets = hist_offsets + [0]
        K_total = len(all_offsets)
        causal_mask = self.build_causal_mask(K_total, hidden_states.device)
        time_offsets = torch.tensor(all_offsets, device=hidden_states.device)

        for layer_num, blk in enumerate(base_vit.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            temporal_idx = self._deepstack_map.get(layer_num, -1)
            if temporal_idx >= 0:
                current_features = hidden_states[:N]
                history = self.cache.get_history(temporal_idx)

                # Cache BEFORE temporal attention so history matches training input.
                self.cache.push(temporal_idx, current_features)

                if history is not None:
                    temporal_seq = torch.cat(
                        [history.to(current_features.device), current_features.unsqueeze(0)],
                        dim=0,
                    ).permute(1, 0, 2)
                else:
                    temporal_seq = current_features.unsqueeze(1)

                temporal_out = self.temporal_attn_layers[temporal_idx](
                    temporal_seq, causal_mask, time_offsets=time_offsets, num_patches=N,
                )

                current_out = temporal_out[:, -1, :]
                hidden_states = current_out.reshape(B * N, D)

                deepstack_feature = base_vit.deepstack_merger_list[temporal_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = base_vit.merger(hidden_states)

        return TemporalViTOutput(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )
