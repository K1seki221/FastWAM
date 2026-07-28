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

from typing import Optional, Union

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from models.head.transformer.blocks import (
    CombinedTransformerBlock,
    CrossOnlyTransformerBlock,
    FullAttentionBlock,
    SeparateTransformerBlock,
    TimestepEncoder,
)
from torch import nn


class DiT(ModelMixin, ConfigMixin):
    """
    Diffusion Transformer (DiT) for action prediction.

    Three independent configuration dimensions control how VLM features are
    consumed by DiT blocks:

    1. ``dit_block_type`` — DiT internal architecture
        - ``'combined'``:       Legacy GR00T block. Each block contains self-attention +
                                cross-attention + FFN (CombinedTransformerBlock).
        - ``'separate'``:       Self-attention and cross-attention live in separate blocks,
                                interleaved as [self, cross, self, cross, ...].
        - ``'cross_only'``:     Single attention + FFN per block. No dedicated self-attention
                                layer (Pi-style architecture).
        - ``'full_attention'``: Joint self-attention over concatenated [VLM, action] tokens.
                                VLM tokens are concatenated once before the block stack.
                                Blocks perform standard self-attention (no cross-attention).
                                Enables bidirectional information flow between VLM and action.

    2. ``vlm_feature_format`` — the form of information coming from the VLM
        - ``'hidden'``:     VLM outputs hidden states [B, S, D]. DiT projects them
                            into K/V via learned linear layers.
        - ``'kv'``:         VLM outputs raw KV cache [B, H, S, head_dim] directly.
                            DiT uses them as K/V without re-projection (RawKVAttention).
                            Only supported by ``separate`` and ``cross_only``.

    3. ``vlm_fusion_mode`` — how VLM information enters the attention computation
        - ``'cross'``:      Pure cross-attention. Q = DiT tokens, K/V = VLM only.
                            DiT tokens do NOT see each other through this attention.
        - ``'context'``:    Context-style attention. Q = DiT tokens,
                            K/V = [DiT tokens, VLM tokens] concatenated.
                            DiT tokens CAN see each other and VLM tokens simultaneously.

    Combination matrix (excluding ``combined`` which is the legacy GR00T baseline):

    +---------------+----------------+------------------------------------------+
    | feature_format | fusion_mode   | Attention behavior                       |
    +===============+================+==========================================+
    | hidden        | cross          | Q=DiT, K/V=proj(VLM_hidden)              |
    +---------------+----------------+------------------------------------------+
    | hidden        | context        | Q=DiT, K/V=proj([DiT_tokens, VLM_hidden])|
    +---------------+----------------+------------------------------------------+
    | kv            | cross          | Q=DiT, K/V=VLM_KV  (concat_dit_kv=False) |
    +---------------+----------------+------------------------------------------+
    | kv            | context        | Q=DiT, K/V=cat(VLM_KV, proj_kv(DiT))    |
    |               |                |                       (concat_dit_kv=True)|
    +---------------+----------------+------------------------------------------+

    The above applies to both ``separate`` (cross-attention blocks only) and
    ``cross_only`` block types. For ``separate``, self-attention blocks always
    handle DiT-internal interaction independently.

    Reference architectures:
        - GR00T:  combined + hidden + cross   (default)
        - Pi:     cross_only + kv + context
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = 'gelu-approximate',
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = 'ada_norm',
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = 'sinusoidal',
        interleave_self_attention=False,
        dit_block_type: str = 'combined',  # 'combined', 'separate', 'cross_only', or 'full_attention'
        vlm_fusion_mode: str = 'cross',  # 'cross' or 'context'
        vlm_feature_format: str = 'hidden',  # 'hidden' or 'kv'
        self_has_ffn: bool = True,  # Only used when dit_block_type='separate'
        cross_layer_mapping: Optional[list] = None,  # For multi-layer cross attention
        qk_norm: Optional[str] = None,  # QK-Norm for attention stability: 'rms_norm', 'layer_norm', etc.
        # GR00T-N1.7-style alternation knobs (only honored when dit_block_type='separate'):
        flip_self_cross: bool = False,  # False: [self, cross, ...] (default); True: [cross, self, ...] (N1.7)
        cross_attention_dim: Optional[int] = None,  # K/V proj input dim for cross blocks; None = same as inner_dim
        attend_text_every_n_blocks: int = 2,  # AlternateVL: cross blocks alternate text/image masks every 2N blocks
        # v2 token router (cross_only only): each query token routes over the K condition sources
        token_router: bool = False,
        token_router_top_k: int = 0,
        token_router_temperature: float = 1.0,
        token_router_init_bias: float = 3.0,
        token_router_tap: str = 'post_adaln',  # 'post_adaln' (timestep-aware) | 'pre_norm' (private LN, decoupled)
        token_router_temb_stopgrad: bool = False,  # *_temb taps: detach e_t before W_t
    ):
        super().__init__()

        self.output_dim = output_dim
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.gradient_checkpointing = False
        self.dit_block_type = dit_block_type
        self.vlm_fusion_mode = vlm_fusion_mode
        self.vlm_feature_format = vlm_feature_format
        # Raw KV reuse: directly use VLM KV cache in attention, skip re-projection.
        # Supported by separate and cross_only block types; combined is kept as legacy baseline.
        self.use_raw_kv = (dit_block_type != 'combined' and vlm_feature_format == 'kv')
        # concat_dit_kv: whether to concatenate DiT's own KV with VLM KV in RawKVAttention.
        # Determined by vlm_fusion_mode: 'context' → True (DiT sees itself + VLM),
        #                                 'cross'   → False (DiT sees VLM only).
        self.concat_dit_kv = (vlm_fusion_mode == 'context')

        # Validate parameters
        valid_block_types = ['combined', 'separate', 'cross_only', 'full_attention']
        if dit_block_type not in valid_block_types:
            raise ValueError(f'dit_block_type must be one of {valid_block_types}, got {dit_block_type}')

        valid_fusion_modes = ['cross', 'context', 'joint']
        if vlm_fusion_mode not in valid_fusion_modes:
            raise ValueError(f'vlm_fusion_mode must be one of {valid_fusion_modes}, got {vlm_fusion_mode}')

        # Multi-layer cross attention mapping
        # Format: [[dit_start, dit_end, vlm_layer], ...]
        # Example: [[0, 23, 6], [24, 47, 13], [48, 71, 20], [72, 95, 27]]
        # If None, defaults to using last VLM layer for all DiT layers
        self.cross_layer_mapping = cross_layer_mapping or [[0, num_layers - 1, -1]]
        self._build_layer_to_vlm_map()

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim, compute_dtype=compute_dtype)
        self.interleave_self_attention = interleave_self_attention
        self.flip_self_cross = flip_self_cross
        self.cross_attention_dim_override = cross_attention_dim
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

        if dit_block_type == 'combined':
            # Legacy mode: each block has self + cross + ffn
            self.transformer_blocks = nn.ModuleList(
                [
                    CombinedTransformerBlock(
                        self.inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        positional_embeddings=positional_embeddings,
                        num_positional_embeddings=max_num_positional_embeddings,
                        final_dropout=final_dropout,
                        qk_norm=qk_norm,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif dit_block_type == 'separate':
            # New mode: separate self and cross blocks
            # Pattern: [self, cross, self, cross, ...]
            # num_layers should be even for this mode
            if num_layers % 2 != 0:
                raise ValueError(f'num_layers must be even for separate dit_block_type, got {num_layers}')

            self.transformer_blocks = nn.ModuleList()
            for i in range(num_layers):
                # Pattern: default [self, cross, self, cross, ...] (cross at odd idx).
                # With flip_self_cross=True: [cross, self, cross, self, ...] (cross at even idx, GR00T-N1.7).
                is_cross = (i % 2 == 0) if flip_self_cross else (i % 2 == 1)
                if not is_cross:
                    # Self-attention block
                    block = SeparateTransformerBlock(
                        self.inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        attention_mode='self',
                        dropout=dropout,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        upcast_attention=upcast_attention,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_type=norm_type,
                        norm_eps=norm_eps,
                        positional_embeddings=positional_embeddings,
                        num_positional_embeddings=max_num_positional_embeddings,
                        final_dropout=final_dropout,
                        ff_inner_dim=None,
                        ff_bias=True,
                        attention_out_bias=True,
                        has_ffn=self_has_ffn,
                        use_raw_kv=self.use_raw_kv,
                        concat_dit_kv=self.concat_dit_kv,
                        qk_norm=qk_norm,
                    )
                else:
                    # Cross-attention block
                    block = SeparateTransformerBlock(
                        self.inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        attention_mode='cross',
                        dropout=dropout,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        upcast_attention=upcast_attention,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_type=norm_type,
                        norm_eps=norm_eps,
                        final_dropout=final_dropout,
                        ff_inner_dim=None,
                        ff_bias=True,
                        attention_out_bias=True,
                        has_ffn=True,  # Cross blocks always have FFN
                        use_raw_kv=self.use_raw_kv,
                        concat_dit_kv=self.concat_dit_kv,
                        qk_norm=qk_norm,
                        cross_attention_dim=cross_attention_dim,
                    )
                self.transformer_blocks.append(block)
        elif dit_block_type == 'cross_only':
            # Single attention + FFN per block, no dedicated self-attention layer
            self.transformer_blocks = nn.ModuleList(
                [
                    CrossOnlyTransformerBlock(
                        self.inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        upcast_attention=upcast_attention,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_type=norm_type,
                        norm_eps=norm_eps,
                        final_dropout=final_dropout,
                        use_raw_kv=self.use_raw_kv,
                        concat_dit_kv=self.concat_dit_kv,
                        qk_norm=qk_norm,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif dit_block_type == 'full_attention':
            # Full attention: joint self-attention over concatenated [VLM, action] tokens.
            # VLM tokens are concatenated before entering blocks; blocks only do self-attention.
            self.vlm_token_type_embed = nn.Parameter(torch.zeros(1, 1, self.inner_dim))
            self.action_token_type_embed = nn.Parameter(torch.zeros(1, 1, self.inner_dim))
            nn.init.normal_(self.vlm_token_type_embed, std=0.02)
            nn.init.normal_(self.action_token_type_embed, std=0.02)

            self.transformer_blocks = nn.ModuleList(
                [
                    FullAttentionBlock(
                        self.inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        positional_embeddings=None,
                        num_positional_embeddings=None,
                        final_dropout=final_dropout,
                        qk_norm=qk_norm,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            raise ValueError(
                f'Unknown dit_block_type: {dit_block_type}. '
                f'Must be "combined", "separate", "cross_only", or "full_attention".'
            )

        # v2 token router: install a per-token router on every cross_only block.
        # The candidate order is the sorted-unique cross_layer_mapping VLM layers;
        # each block's incumbent column is its span owner (token_router_owners).
        self.token_router_enabled = token_router
        if token_router:
            assert dit_block_type == 'cross_only', 'token_router requires dit_block_type="cross_only"'
            self._router_source_order = sorted(
                {vlm_layer for _, _, vlm_layer in self.cross_layer_mapping}
            )
            col_of_layer = {layer: c for c, layer in enumerate(self._router_source_order)}
            # per-block incumbent column = its span owner's position in the source order
            owners = [0] * num_layers
            for start, end, vlm_layer in self.cross_layer_mapping:
                for i in range(start, end + 1):
                    owners[i] = col_of_layer[vlm_layer]
            for i, block in enumerate(self.transformer_blocks):
                block.install_token_router(
                    num_sources=len(self._router_source_order),
                    incumbent_col=owners[i],
                    top_k=token_router_top_k,
                    temperature=token_router_temperature,
                    init_bias=token_router_init_bias,
                    tap=token_router_tap,
                    temb_stopgrad=token_router_temb_stopgrad,
                )

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)
        print(
            'Total number of DiT parameters: ',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def _apply_hidden_context_fusion(self, current_encoder, previous_layer_output, encoder_attention_mask):
        """
        Apply context fusion for hidden state mode: concat previous layer output with VLM features.

        Returns:
            (current_encoder, current_attention_mask) with context concatenated
        """
        current_attention_mask = encoder_attention_mask
        if previous_layer_output is not None:
            current_encoder = torch.cat([previous_layer_output, current_encoder], dim=1)
            if encoder_attention_mask is not None:
                prev_mask = torch.ones(
                    previous_layer_output.shape[0], previous_layer_output.shape[1],
                    device=previous_layer_output.device, dtype=encoder_attention_mask.dtype,
                )
                current_attention_mask = torch.cat([prev_mask, encoder_attention_mask], dim=1)
        return current_encoder, current_attention_mask

    def _build_layer_to_vlm_map(self):
        """
        Build mapping from DiT layer index to VLM layer index.

        This allows different DiT layers to attend to different VLM layers.
        """
        self.dit_to_vlm_map = {}
        for dit_start, dit_end, vlm_layer in self.cross_layer_mapping:
            for dit_idx in range(dit_start, dit_end + 1):
                if dit_idx in self.dit_to_vlm_map:
                    raise ValueError(
                        f'DiT layer {dit_idx} is mapped to multiple VLM layers: '
                        f'{self.dit_to_vlm_map[dit_idx]} and {vlm_layer}'
                    )
                self.dit_to_vlm_map[dit_idx] = vlm_layer

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: Union[torch.Tensor, dict],  # Shape: (B, S, D) or dict of tensors
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through DiT.

        Args:
            hidden_states: DiT input tokens [B, T, D] where D = num_heads * head_dim.
            encoder_hidden_states: VLM features. Format depends on mode:
                - Hidden mode, single-layer: tensor [B, S_vlm, D_vlm]
                - Hidden mode, multi-layer:  dict {vlm_layer_idx: tensor [B, S_vlm, D_vlm]}
                - Raw KV mode, single-layer: (K, V) tuple of [B, H, S_vlm, head_dim]
                - Raw KV mode, multi-layer:  dict {vlm_layer_idx: (K [B, H, S_vlm, head_dim],
                                                                    V [B, H, S_vlm, head_dim])}
            timestep: Diffusion timestep [B]
            encoder_attention_mask: Padding mask [B, S_vlm] (1=attend, 0=mask)
            return_all_hidden_states: Whether to return intermediate states
            image_mask: [B, S_vlm] bool — True at positions that are image tokens. When provided
                together with backbone_attention_mask, cross blocks alternate attending to
                text-only vs image-only tokens (GR00T-N1.7 AlternateVL behavior). Ignored if None.
            backbone_attention_mask: [B, S_vlm] bool — True at non-padding VLM positions. Required
                companion to image_mask for AlternateVL routing.

        Returns:
            Output tensor [B, T, output_dim] or (output, all_hidden_states) tuple
        """
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()

        if self.dit_block_type == 'full_attention':
            return self._forward_full_attention(
                hidden_states, encoder_hidden_states, temb, encoder_attention_mask, return_all_hidden_states,
            )

        if self.token_router_enabled:
            return self._forward_token_router(
                hidden_states, encoder_hidden_states, temb, encoder_attention_mask, return_all_hidden_states,
            )

        return self._forward_cross_attention(
            hidden_states, encoder_hidden_states, temb, encoder_attention_mask, return_all_hidden_states,
            image_mask=image_mask, backbone_attention_mask=backbone_attention_mask,
        )

    def _forward_token_router(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Union[torch.Tensor, dict],
        temb: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        return_all_hidden_states: bool,
    ):
        """v2: every block sees ALL K candidate sources; each query token routes
        over them (context fusion + mixing handled inside the block)."""
        assert isinstance(encoder_hidden_states, dict), (
            'token_router requires the multi-layer condition dict {vlm_layer: [B, S, D]}'
        )
        # stack candidates in the fixed source order (same for every block)
        candidate_stack = torch.stack(
            [encoder_hidden_states[layer].contiguous() for layer in self._router_source_order]
        )  # [K, B, S, D]

        all_hidden_states = [hidden_states]
        previous_output = None
        for block in self.transformer_blocks:
            hidden_states = block.forward_token_routed(
                hidden_states,
                candidate_stack=candidate_stack,
                candidate_mask=encoder_attention_mask,
                previous_output=previous_output,
                temb=temb,
            )
            previous_output = hidden_states  # context fusion: prev block output prepended next
            all_hidden_states.append(hidden_states)

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)

    def _forward_full_attention(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Union[torch.Tensor, dict],
        temb: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        return_all_hidden_states: bool,
    ):
        """Forward pass for full_attention mode: joint self-attention over [VLM, action] tokens."""
        # Extract VLM hidden states (full_attention only supports single-layer hidden mode)
        if isinstance(encoder_hidden_states, dict):
            vlm_tokens = encoder_hidden_states[-1]
        else:
            vlm_tokens = encoder_hidden_states
        vlm_tokens = vlm_tokens.contiguous()

        action_seq_len = hidden_states.shape[1]
        vlm_seq_len = vlm_tokens.shape[1]

        # Add token type embeddings to distinguish VLM vs action tokens
        vlm_tokens = vlm_tokens + self.vlm_token_type_embed
        hidden_states = hidden_states + self.action_token_type_embed

        # Concatenate into single sequence: [B, S+T, D]
        combined = torch.cat([vlm_tokens, hidden_states], dim=1)

        # Build combined attention mask: [B, S+T] where True=attend, False=mask
        combined_mask = None
        if encoder_attention_mask is not None:
            if encoder_attention_mask.shape[-1] == vlm_seq_len:
                action_mask = torch.ones(
                    combined.shape[0], action_seq_len, device=combined.device, dtype=torch.bool,
                )
                combined_mask = torch.cat([encoder_attention_mask.bool(), action_mask], dim=1)

        all_hidden_states = [combined]

        # Joint self-attention: all tokens attend to all tokens
        for block in self.transformer_blocks:
            combined = block(combined, attention_mask=combined_mask, temb=temb)
            all_hidden_states.append(combined)

        # Extract action tokens only: [B, S+T, D] -> [B, T, D]
        hidden_states = combined[:, vlm_seq_len:]

        # Output: AdaNorm + projection
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out_2(hidden_states)

        if return_all_hidden_states:
            return output, all_hidden_states
        return output

    def _forward_cross_attention(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Union[torch.Tensor, dict],
        temb: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        return_all_hidden_states: bool,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass for combined/separate/cross_only modes (cross-attention based)."""
        # Pre-compute alternate-VL masks: when image_mask is provided, cross blocks alternate
        # between attending text-only and image-only VLM tokens (GR00T-N1.7 behavior).
        # Only honored in 'separate' mode; other block types fall back to encoder_attention_mask.
        use_alternate_vl = (
            image_mask is not None
            and backbone_attention_mask is not None
            and self.dit_block_type == 'separate'
        )
        if use_alternate_vl:
            assert image_mask is not None and backbone_attention_mask is not None
            text_mask_pre = (~image_mask.bool()) & backbone_attention_mask.bool()
            image_mask_pre = image_mask.bool() & backbone_attention_mask.bool()
        else:
            text_mask_pre = None
            image_mask_pre = None

        # Determine if using multi-layer cross attention
        is_multi_layer = isinstance(encoder_hidden_states, dict)

        # Make encoder_hidden_states contiguous
        if is_multi_layer:
            assert isinstance(encoder_hidden_states, dict)
            if self.use_raw_kv:
                # Raw KV: values are (key, value) tuples
                encoder_hidden_states = {
                    k: (v[0].contiguous(), v[1].contiguous()) if isinstance(v, tuple) else v.contiguous()
                    for k, v in encoder_hidden_states.items()
                }
            else:
                encoder_hidden_states = {k: v.contiguous() for k, v in encoder_hidden_states.items()}
        elif encoder_hidden_states is not None:
            assert isinstance(encoder_hidden_states, torch.Tensor)
            encoder_hidden_states = encoder_hidden_states.contiguous()

        # Validate encoder_attention_mask shape if single encoder
        if not is_multi_layer and encoder_attention_mask is not None and encoder_hidden_states is not None:
            mask_len = encoder_attention_mask.shape[-1]
            assert isinstance(encoder_hidden_states, torch.Tensor)
            encoder_len = encoder_hidden_states.shape[1]
            if mask_len != encoder_len:
                # Mask length doesn't match encoder length, disable mask to avoid shape errors
                encoder_attention_mask = None

        all_hidden_states = [hidden_states]

        # Initialize previous_layer_output for context fusion mode (vlm_fusion_mode='context')
        # This stores the output from the immediately previous DiT layer
        previous_layer_output = None

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            # Get appropriate encoder for this layer
            if is_multi_layer:
                assert isinstance(encoder_hidden_states, dict)
                vlm_layer = self.dit_to_vlm_map.get(idx, -1)
                current_encoder = encoder_hidden_states.get(vlm_layer, None)
            else:
                current_encoder = encoder_hidden_states

            # Prepare block kwargs (unified signature for all block types)
            vlm_k, vlm_v = None, None
            current_attention_mask = encoder_attention_mask

            if self.use_raw_kv:
                # Raw KV mode: extract (key, value) tuple; context fusion handled by concat_dit_kv
                # vlm_k, vlm_v: [B, H, S_vlm, head_dim] — passed directly to RawKVAttention
                if current_encoder is not None and isinstance(current_encoder, tuple):
                    vlm_k, vlm_v = current_encoder
                current_encoder = None  # blocks use vlm_key/value_cache instead
            else:
                # Hidden state mode: apply context fusion if needed
                use_context = self.vlm_fusion_mode == 'context' and current_encoder is not None
                # For separate blocks, context fusion only applies to cross blocks. The cross-block
                # parity depends on flip_self_cross: default (False) puts cross at odd idx;
                # flip_self_cross=True (GR00T-N1.7) puts cross at even idx.
                is_cross_block = False
                if self.dit_block_type == 'separate':
                    is_cross_block = (idx % 2 == 0) if self.flip_self_cross else (idx % 2 == 1)
                    use_context = use_context and is_cross_block
                # For combined blocks, interleave_self_attention skips encoder on odd layers
                if self.dit_block_type == 'combined' and idx % 2 == 1 and self.interleave_self_attention:
                    current_encoder = None
                    current_attention_mask = None

                if use_context:
                    # After fusion: current_encoder [B, S_prev + S_vlm, D], mask [B, S_prev + S_vlm]
                    current_encoder, current_attention_mask = self._apply_hidden_context_fusion(
                        current_encoder, previous_layer_output, encoder_attention_mask,
                    )

                # AlternateVL routing: cross blocks alternate text-only vs image-only attention.
                # Override current_attention_mask only after context fusion (so we don't clobber
                # the prepended-context portion of the mask).
                if use_alternate_vl and is_cross_block and not use_context:
                    if idx % (2 * self.attend_text_every_n_blocks) == 0:
                        current_attention_mask = text_mask_pre
                    else:
                        current_attention_mask = image_mask_pre

            hidden_states = block(
                hidden_states,
                encoder_hidden_states=current_encoder,
                encoder_attention_mask=current_attention_mask,
                attention_mask=None,
                temb=temb,
                vlm_key_cache=vlm_k,
                vlm_value_cache=vlm_v,
            )

            # Update previous_layer_output for context fusion
            if not self.use_raw_kv and self.vlm_fusion_mode == 'context':
                previous_layer_output = hidden_states

            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)
