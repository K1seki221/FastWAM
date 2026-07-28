from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from models.backbone.embedding_utils import replace_placeholder_embeddings
from models.backbone.registry import register_backbone
from transformers import Cache, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextModel,
)
from transformers.utils import auto_docstring, is_torchdynamo_compiling


class Qwen3VLTextModelWithLayerExtraction(Qwen3VLTextModel):
    """
    Extended Qwen3VLTextModel that supports extracting hidden states from intermediate layers.

    This class enables multi-layer cross-attention by collecting and normalizing hidden states
    from specified layers during the forward pass. The implementation follows transformers'
    standard patterns and ensures all parameters are correctly passed through the layer loop.

    Key Features:
        - Extract hidden states from any specified layers
        - Apply parameter-free RMSNorm to stabilize extracted features
        - Full compatibility with original Qwen3VLTextModel interface
        - Correct handling of all parameters including use_cache, DeepStack processing

    Args:
        config: Qwen3VLTextConfig instance

    Example:
        >>> text_model = Qwen3VLTextModelWithLayerExtraction(config)
        >>> outputs = text_model(
        ...     inputs_embeds=embeds,
        ...     extract_layers=[6, 13, 20, 27],  # Extract from these layers
        ... )
        >>> outputs.layer_features[6].shape  # [B, seq_len, hidden_size]
    """

    def __init__(self, config):
        super().__init__(config)

        # Create parameter-free RMSNorm for feature extraction
        # This avoids gradient conflicts when extracting from multiple layers
        # (e.g., layers 6, 13, 20, 27 all use the same norm without learnable weights)
        self.feature_extraction_norm = self._create_parameter_free_rmsnorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @staticmethod
    def _create_parameter_free_rmsnorm(hidden_size, eps=1e-6):
        """
        Create RMSNorm without learnable parameters for feature extraction.

        Using parameter-free normalization ensures that:
        1. No gradient conflicts when same norm is applied to different layers
        2. Consistent normalization behavior across all extracted layers
        3. No additional parameters to train

        Args:
            hidden_size: Hidden dimension size
            eps: Epsilon for numerical stability

        Returns:
            nn.Module that performs RMSNorm without learnable weight
        """

        class RMSNormNoWeight(nn.Module):
            def __init__(self, hidden_size, eps):
                super().__init__()
                self.eps = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
                return hidden_states.to(input_dtype)

        return RMSNormNoWeight(hidden_size, eps)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        extract_layers: Optional[list[int]] = None,
        output_attentions: Optional[bool] = None,
        **kwargs: Any,
    ):
        """
        Forward pass with optional layer-wise feature extraction.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cache for generation
            inputs_embeds: Input embeddings (alternative to input_ids)
            use_cache: Whether to use KV caching
            cache_position: Position in cache
            visual_pos_masks: Mask for visual token positions (for DeepStack)
            deepstack_visual_embeds: DeepStack visual features
            extract_layers: List of layer indices to extract features from
                Example: [6, 13, 20, 27]
            **kwargs: Additional arguments

        Returns:
            Qwen3VLModelOutputWithLastHiddenState:
                - last_hidden_state: Final layer output
                - past_key_values: KV cache
                - layer_features: dict[layer_idx, normalized_hidden_states] if extract_layers provided

        Note:
            This implementation follows transformers' Qwen3VLTextModel closely,
            ensuring all parameters are correctly passed to decoder layers.
        """
        # Import necessary functions from transformers
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask

        # Input validation
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')

        # Prepare cache
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        assert inputs_embeds is not None

        # Prepare position info
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        assert position_ids is not None

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        # Create attention mask
        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Initialize layer features collection
        layer_features: dict[int, torch.Tensor] | None = {} if extract_layers else None
        layer_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = {} if extract_layers else None
        captured_attn_weights: list[torch.Tensor] | None = [] if output_attentions else None

        # Process through decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                **kwargs,
            )
            if output_attentions:
                hidden_states = layer_outputs[0]
                assert captured_attn_weights is not None
                captured_attn_weights.append(layer_outputs[1])
            else:
                hidden_states = layer_outputs

            # DeepStack processing if needed (must be after decoder layer, before feature extraction)
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                assert visual_pos_masks is not None
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

            # Collect features from specified layers
            if extract_layers and layer_idx in extract_layers:
                assert layer_features is not None
                if layer_idx == len(self.layers) - 1:
                    # Last layer: use backbone's learnable RMSNorm (self.norm) so that
                    # norm.weight participates in the gradient graph via layer_features,
                    # preventing DDP unused-parameter errors in multi-layer VLM mode.
                    layer_features[layer_idx] = self.norm(hidden_states)
                else:
                    # Intermediate layers: use parameter-free normalization
                    layer_features[layer_idx] = self.feature_extraction_norm(hidden_states)

                # Extract KV cache from DynamicCache if use_cache is enabled
                if use_cache and past_key_values is not None:
                    assert layer_kv_cache is not None
                    # DynamicCache stores KV for each layer, access via indexing
                    # past_key_values[layer_idx] returns (key, value) tuple
                    layer_kv_cache[layer_idx] = past_key_values[layer_idx]

        # Final layer norm — reuse already-computed result if last layer was extracted
        last_layer_idx = len(self.layers) - 1
        if extract_layers and last_layer_idx in extract_layers:
            assert layer_features is not None
            hidden_states = layer_features[last_layer_idx]
        else:
            hidden_states = self.norm(hidden_states)

        return Qwen3VLModelOutputWithLastHiddenState(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            attentions=tuple(captured_attn_weights) if captured_attn_weights is not None else None,
            layer_features=layer_features,
            layer_kv_cache=layer_kv_cache,
        )


class Qwen3VLModelWrapper(Qwen3VLModel):
    """
    Wrapper for Qwen3VLModel that uses layer-extraction-capable text model.

    This wrapper replaces the standard language_model with Qwen3VLTextModelWithLayerExtraction,
    enabling multi-layer feature extraction while maintaining full compatibility with the
    original Qwen3VLModel interface.
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace language_model with layer-extraction-capable version
        self.language_model = Qwen3VLTextModelWithLayerExtraction._from_config(config.text_config)

        # ===== vit_condition runtime state (set via configure_vit_condition) =====
        self._vit_cond_tap: str | None = None
        self._vit_cond_sources: list[str] = []
        self._vit_lora_wrappers = None
        self._vit_capture_active = False
        self._vit_premerger_cache: dict[str, torch.Tensor] = {}
        # ===== dual_stream runtime state (set via configure_dual_stream) =====
        # When set, the wrapper emits BOTH condition streams: the base pass
        # (LoRA disabled everywhere) and a second, LoRA-enabled ViT+LLM pass.
        self._llm_lora_wrappers = None
        self._lora_vision_stash: tuple | None = None

    # ------------------------------------------------------------------
    # vit_condition: capture ViT features (DeepStack taps + final layer)
    # for head-side DiT conditioning. See configs ViTConditionConfig.
    # ------------------------------------------------------------------

    def configure_vit_condition(self, *, tap: str, sources: list[str], lora_wrappers=None) -> None:
        """Enable ViT feature capture. Called once from IronVLA init.

        Args:
            tap: 'post_merger' (reuse the merged outputs the VLM already
                computes — zero extra cost) or 'pre_merger' (raw 1024-dim ViT
                block outputs, captured via forward-pre-hooks on the mergers).
            sources: subset of ['ds0', 'ds1', 'ds2', 'final'] to emit —
                dsN = deepstack_visual_indexes[N] (ascending ViT depth).
            lora_wrappers: SwitchableLoRALinear list when ViT->DiT LoRA is on;
                triggers a second, LoRA-enabled ViT pass for the tap while the
                VLM path keeps base weights.
        """
        assert tap in ('post_merger', 'pre_merger')
        self._vit_cond_tap = tap
        self._vit_cond_sources = list(sources)
        self._vit_lora_wrappers = lora_wrappers

        if tap == 'pre_merger':
            num_deepstack = len(self.visual.deepstack_merger_list)

            def _make_hook(key):
                def _pre_hook(module, args):
                    if self._vit_capture_active:
                        self._vit_premerger_cache[key] = args[0]
                    return None

                return _pre_hook

            if 'final' in self._vit_cond_sources:
                self.visual.merger.register_forward_pre_hook(_make_hook('final'))
            for i in range(num_deepstack):
                if f'ds{i}' in self._vit_cond_sources:
                    self.visual.deepstack_merger_list[i].register_forward_pre_hook(_make_hook(f'ds{i}'))

            # condition_router pool extras: tap arbitrary ViT block outputs.
            # Same flat [total_patches, 1024] form and capture lifecycle as the
            # merger pre-hooks, so downstream rebatching is shared.
            def _make_block_hook(key):
                def _hook(module, args, output):
                    if self._vit_capture_active:
                        out = output[0] if isinstance(output, tuple) else output
                        self._vit_premerger_cache[key] = out

                return _hook

            for source in self._vit_cond_sources:
                if source.startswith('blk'):
                    block_idx = int(source[3:])
                    assert 0 <= block_idx < len(self.visual.blocks), (
                        f'condition_router extra_vit_blocks: block {block_idx} out of range '
                        f'(ViT has {len(self.visual.blocks)} blocks)'
                    )
                    self.visual.blocks[block_idx].register_forward_hook(_make_block_hook(source))
        else:
            assert not any(s.startswith('blk') for s in self._vit_cond_sources), (
                'blk* ViT sources require tap="pre_merger"'
            )

    def configure_dual_stream(self, *, llm_lora_wrappers) -> None:
        """Enable dual-stream emission (base + LoRA condition). Called from
        IronVLA init after configure_vit_condition; requires ViT LoRA wrappers
        (validated in PolicyConfig)."""
        assert self._vit_lora_wrappers, 'dual_stream requires vit_condition LoRA wrappers'
        self._llm_lora_wrappers = llm_lora_wrappers

    def _vit_capture(self):
        """Context manager arming the pre-merger hooks for one ViT pass."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            self._vit_premerger_cache.clear()
            self._vit_capture_active = True
            try:
                yield
            finally:
                self._vit_capture_active = False

        return _ctx()

    def _collect_vit_flat_features(
        self, pixel_values, image_grid_thw, image_embeds_flat, deepstack_image_embeds
    ) -> dict[str, torch.Tensor]:
        """Return the requested flat ViT features [total_tokens, D].

        No LoRA + post_merger: reuse the tensors from the VLM's own ViT pass.
        No LoRA + pre_merger:  hooks captured them during that same pass.
        LoRA: run a second, LoRA-enabled ViT pass and take features from it
        (the VLM pass above used base weights — LoRA is never merged).

        Gradient isolation of the LoRA pass: the ViT base weights are
        temporarily set requires_grad=False around the second forward, so
        gradients from the tapped (ViT->DiT) features reach ONLY the LoRA
        params — the base ViT is trained solely by the VLM path, even under
        stage=vla-full-train. requires_grad is recorded per-operation at
        forward time, so the first (VLM) pass is unaffected; activation
        gradients still flow through base ops to reach shallower LoRA
        modules. No-op when the base is already frozen.
        """
        from models.backbone.vit_lora import set_vit_lora_enabled

        tap = self._vit_cond_tap
        if self._vit_lora_wrappers:
            set_vit_lora_enabled(self._vit_lora_wrappers, True)
            base_params = [
                p for name, p in self.visual.named_parameters()
                if 'lora_' not in name and p.requires_grad
            ]
            for p in base_params:
                p.requires_grad_(False)
            try:
                with self._vit_capture():
                    vision_output = self.get_image_features(pixel_values, image_grid_thw)
            finally:
                for p in base_params:
                    p.requires_grad_(True)
                set_vit_lora_enabled(self._vit_lora_wrappers, False)
            if isinstance(vision_output, tuple):
                lora_embeds, lora_deepstack = vision_output
            else:
                lora_embeds = vision_output.pooler_output
                lora_deepstack = vision_output.deepstack_features
            # dual_stream: the LoRA pass's post-merger outputs feed the second
            # (LLM-LoRA) pass as its visual embeds, whatever the tap is.
            if self._llm_lora_wrappers is not None:
                self._lora_vision_stash = (
                    torch.cat(lora_embeds, dim=0), list(lora_deepstack)
                )
            if tap == 'post_merger':
                final_flat = torch.cat(lora_embeds, dim=0)
                deepstack_flat = list(lora_deepstack)
            else:
                final_flat = self._vit_premerger_cache.get('final')
                deepstack_flat = None  # looked up from cache below
        elif tap == 'post_merger':
            final_flat = image_embeds_flat
            deepstack_flat = list(deepstack_image_embeds)
        else:  # pre_merger, hooks fired during the VLM's ViT pass
            final_flat = self._vit_premerger_cache.get('final')
            deepstack_flat = None

        features: dict[str, torch.Tensor] = {}
        for source in self._vit_cond_sources:
            if source == 'final':
                assert final_flat is not None, 'vit_condition: final-layer capture missing'
                features['final'] = final_flat
            elif tap == 'pre_merger':
                cached = self._vit_premerger_cache.get(source)
                assert cached is not None, f'vit_condition: pre_merger capture missing for {source}'
                features[source] = cached
            else:
                idx = int(source[2:])
                assert deepstack_flat is not None, (
                    f'vit_condition: deepstack capture missing for {source}'
                )
                features[source] = deepstack_flat[idx]
        return features

    @staticmethod
    def _rebatch_flat_visual(flat: torch.Tensor, counts: torch.Tensor, pad_multiple: int = 1):
        """[total_tokens, D] + per-sample counts [B] -> ([B, T_max, D], [B, T_max] bool).

        Flat visual tokens are ordered row-major by sample (the same ordering
        ``masked_scatter`` relies on). T_max is rounded up to ``pad_multiple``
        so the merger adapter's 2x2 view-merge stays valid under padding.
        """
        counts_list = [int(c) for c in counts.tolist()]
        t_max = max(counts_list) if counts_list else 0
        if pad_multiple > 1 and t_max % pad_multiple != 0:
            t_max += pad_multiple - (t_max % pad_multiple)
        batch = len(counts_list)
        out = flat.new_zeros(batch, t_max, flat.shape[-1])
        mask = torch.zeros(batch, t_max, dtype=torch.bool, device=flat.device)
        offset = 0
        for i, count in enumerate(counts_list):
            out[i, :count] = flat[offset:offset + count]
            mask[i, :count] = True
            offset += count
        assert offset == flat.shape[0], (
            f'vit_condition rebatch mismatch: consumed {offset} of {flat.shape[0]} flat tokens'
        )
        return out, mask

    def _rebatch_source_features(self, flat_features, visual_pos_masks):
        """Rebatch a {source: flat [total, D]} dict to ({source: [B, T, D]}, mask)."""
        merge_area = self.visual.spatial_merge_size ** 2
        merged_counts = visual_pos_masks.sum(dim=1)
        if self._vit_cond_tap == 'pre_merger':
            counts = merged_counts * merge_area
            pad_multiple = merge_area
        else:
            counts = merged_counts
            pad_multiple = 1

        vit_features = {}
        vit_features_mask = None
        for key, flat in flat_features.items():
            vit_features[key], vit_features_mask = self._rebatch_flat_visual(flat, counts, pad_multiple)
        return vit_features, vit_features_mask

    def _build_vit_condition_outputs(
        self, pixel_values, image_grid_thw, image_embeds_flat, deepstack_image_embeds, visual_pos_masks
    ):
        """Assemble the (vit_features, vit_features_mask) output pair."""
        flat_features = self._collect_vit_flat_features(
            pixel_values, image_grid_thw, image_embeds_flat, deepstack_image_embeds
        )
        return self._rebatch_source_features(flat_features, visual_pos_masks)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if 'multi_task_tokens' in kwargs:
                proprio_placeholder_token_id = kwargs.get('proprio_placeholder_token_id')
                assert proprio_placeholder_token_id is not None, (
                    'proprio_placeholder_token_id is required when multi_task_tokens is provided. '
                    'Ensure add_proprio_placeholder=True in BackboneAutoProcessorTransform.'
                )
                assert input_ids is not None, 'input_ids is required for proprio token replacement'

                inputs_embeds = replace_placeholder_embeddings(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    replacement_embeddings=kwargs['multi_task_tokens'],
                    placeholder_token_id=proprio_placeholder_token_id,
                    placeholder_name='PROPRIO',
                    row_mask=kwargs.get('proprio_row_mask'),
                )
        assert inputs_embeds is not None

        image_mask = None
        video_mask = None

        # STM: number of frames (current + history) per sample; defaults to 1
        # so non-STM runs hit the original single-frame code path.
        stm_num_frames = kwargs.pop('stm_num_frames', 1)

        if pixel_values is not None:
            K = stm_num_frames
            temporal_vit = getattr(self, 'temporal_vit_encoder', None)
            # Route through STM whenever the encoder is attached. The encoder's
            # own forward picks the right sub-path (multi-frame vs cached) and
            # K=1 is a degenerate-but-valid call into _forward_multi_frame —
            # required by state-history-only configs (num_history_frames=0)
            # so the temporal params stay in the DDP graph under
            # find_unused_parameters=False.
            if temporal_vit is not None:
                assert image_grid_thw is not None
                pixel_values_typed = pixel_values.type(self.visual.dtype)
                vision_output = temporal_vit(
                    pixel_values_typed, grid_thw=image_grid_thw, num_frames=K,
                )
                if K > 1:
                    # Multi-frame: keep only current-frame grid_thw so downstream
                    # RoPE / split_sizes operate on the encoder's returned tokens.
                    num_images = image_grid_thw.shape[0]
                    current_indices = torch.arange(K - 1, num_images, K, device=image_grid_thw.device)
                    image_grid_thw = image_grid_thw[current_indices]
                split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
                image_embeds = torch.split(vision_output.pooler_output, split_sizes)
                deepstack_image_embeds = vision_output.deepstack_features
            else:
                # vit_condition pre_merger tap without LoRA reuses this very
                # pass — arm the merger pre-hooks around it. (With LoRA the
                # capture happens in a second, LoRA-enabled pass later; in
                # dual_stream mode this base pass is ALSO captured, giving the
                # pristine ViT stream.)
                capture_this_pass = self._vit_cond_tap == 'pre_merger' and (
                    not self._vit_lora_wrappers or self._llm_lora_wrappers is not None
                )
                if capture_this_pass:
                    with self._vit_capture():
                        vision_output = self.get_image_features(pixel_values, image_grid_thw)
                else:
                    vision_output = self.get_image_features(pixel_values, image_grid_thw)
                if isinstance(vision_output, tuple):
                    # Old transformers: returns (embeds, deepstack_features) tuple
                    image_embeds, deepstack_image_embeds = vision_output
                else:
                    # New transformers: returns object with pooler_output/deepstack_features
                    image_embeds = vision_output.pooler_output
                    deepstack_image_embeds = vision_output.deepstack_features
                if self._llm_lora_wrappers is not None:
                    # snapshot the base pass's pre-merger captures NOW — the
                    # LoRA pass inside _collect_vit_flat_features clears the cache
                    dual_base_premerger = dict(self._vit_premerger_cache)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            if self._llm_lora_wrappers is not None:
                # pass-2 (LLM-LoRA) rebuilds the visual scatter with the
                # ViT-LoRA embeds; keep the pre-scatter embeddings (text +
                # proprio, shared by both passes) and the scatter mask.
                dual_pre_scatter_embeds = inputs_embeds
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds            )
            if self._llm_lora_wrappers is not None:
                dual_scatter_mask = image_mask
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            vision_output = self.get_video_features(pixel_values_videos, video_grid_thw)
            if isinstance(vision_output, tuple):
                video_embeds, deepstack_video_embeds = vision_output
            else:
                video_embeds = vision_output.pooler_output
                deepstack_video_embeds = vision_output.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # ===== vit_condition: capture ViT features for the action head =====
        vit_features = None
        vit_features_mask = None
        vit_features_base = None
        if self._vit_cond_tap is not None and pixel_values is not None:
            assert pixel_values_videos is None, 'vit_condition does not support video inputs'
            assert getattr(self, 'temporal_vit_encoder', None) is None, (
                'vit_condition is incompatible with STM (validated in config; should not reach here)'
            )
            vit_features, vit_features_mask = self._build_vit_condition_outputs(
                pixel_values, image_grid_thw, image_embeds, deepstack_image_embeds, visual_pos_masks
            )
            if self._llm_lora_wrappers is not None:
                # dual_stream: pristine ViT stream from the base pass. With the
                # pre_merger tap, sources come from the snapshot taken before the
                # LoRA pass; with post_merger, from the base pass outputs directly.
                if self._vit_cond_tap == 'pre_merger':
                    base_flat = {
                        source: dual_base_premerger[source] for source in self._vit_cond_sources
                    }
                else:
                    base_flat = {}
                    for source in self._vit_cond_sources:
                        if source == 'final':
                            base_flat['final'] = image_embeds
                        else:
                            base_flat[source] = deepstack_image_embeds[int(source[2:])]
                vit_features_base, _ = self._rebatch_source_features(base_flat, visual_pos_masks)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask['full_attention']
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            self.rope_deltas: torch.Tensor | None
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                else:
                    delta = torch.zeros((batch_size,), device=inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        # ===== dual_stream pass 2: LoRA-enabled LLM on the ViT-LoRA embeds =====
        layer_features_lora = None
        if self._llm_lora_wrappers is not None and pixel_values is not None:
            from models.backbone.vit_lora import set_vit_lora_enabled

            assert self._lora_vision_stash is not None, (
                'dual_stream: LoRA vision outputs missing — _collect_vit_flat_features '
                'should have stashed them during the vit_condition capture'
            )
            lora_image_embeds, lora_deepstack = self._lora_vision_stash
            self._lora_vision_stash = None
            lora_image_embeds = lora_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds_lora = dual_pre_scatter_embeds.masked_scatter(
                dual_scatter_mask, lora_image_embeds
            )
            kwargs_lora = dict(kwargs)
            kwargs_lora['use_cache'] = False
            set_vit_lora_enabled(self._llm_lora_wrappers, True)
            try:
                outputs_lora = self.language_model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    inputs_embeds=inputs_embeds_lora,
                    cache_position=cache_position,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=lora_deepstack,
                    **kwargs_lora,
                )
            finally:
                set_vit_lora_enabled(self._llm_lora_wrappers, False)
            layer_features_lora = getattr(outputs_lora, 'layer_features', None)

        return Qwen3VLModelOutputWithLastHiddenState(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            attentions=getattr(outputs, 'attentions', None),
            rope_deltas=self.rope_deltas,
            labels=labels,
            metadata={'vision_mask': visual_pos_masks} if visual_pos_masks is not None else None,
            layer_features=getattr(outputs, 'layer_features', None),
            layer_kv_cache=getattr(outputs, 'layer_kv_cache', None),
            layer_features_lora=layer_features_lora,
            # visual_pos_masks (image|video) drives image-vs-text attention
            # routing in heads such as FlowMatchingActionGr00tN1d7. The 2D
            # [B, S] form is what _flash_attention_forward / AlternateVL expect.
            image_mask=visual_pos_masks,
            vit_features=vit_features,
            vit_features_mask=vit_features_mask,
            vit_features_base=vit_features_base,
        )


@register_backbone
class Qwen3_VL_Wrapper(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModelWrapper(config)
        # HF Qwen3VLVisionAttention.__init__ hardcodes `self.attention_dropout = 0.0`
        # (modeling_qwen3_vl.py:179) — the forward path reads this attribute, so we
        # override it post-init when config.vision_attention_dropout > 0. Must come
        # after `self.model = Qwen3VLModelWrapper(config)` because that line
        # re-creates self.model.visual; patching before it would target a discarded
        # module. SDPA already gates dropout on self.training, so eval-time is safe.
        vision_drop = getattr(config, 'vision_attention_dropout', 0.0)
        if vision_drop > 0:
            for block in self.visual.blocks:
                block.attn.attention_dropout = vision_drop
        self.set_attention_mask(use_causal_mask=config.use_causal_mask)
        self.freeze_backbones(stage=config.stage)
        if(config.enable_gradient_checkpointing):
            self.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def _get_language_model(self):
        """Compatible accessor: new transformers uses self.model.language_model, old uses self.language_model."""
        if hasattr(self, 'model') and hasattr(self.model, 'language_model'):
            return self.model.language_model
        return self.language_model

    def _get_visual(self):
        """Compatible accessor: new transformers uses self.model.visual, old uses self.visual."""
        if hasattr(self, 'model') and hasattr(self.model, 'visual'):
            return self.model.visual
        return self.visual

    def set_attention_mask(self, use_causal_mask: bool) -> None:
        for layer in self._get_language_model().layers:
            if hasattr(layer.self_attn, 'is_causal'):
                layer.self_attn.is_causal = use_causal_mask

    def freeze_backbones(self, stage: str) -> None:
        lang_model = self._get_language_model()
        visual = self._get_visual()
        if stage == 'vla-full-train':
            lang_model.requires_grad_(True)
            visual.requires_grad_(True)
        elif stage == 'finetune-frozen-llm':
            lang_model.requires_grad_(False)
            visual.requires_grad_(True)
        elif stage == 'freeze-backbone':
            lang_model.requires_grad_(False)
            visual.requires_grad_(False)
        else:
            raise ValueError(f'Invalid stage: {stage}')

    def configure_vit_condition(self, *, tap: str, sources: list[str], lora_wrappers=None) -> None:
        """Passthrough to the inner Qwen3VLModelWrapper (see its docstring)."""
        self.model.configure_vit_condition(tap=tap, sources=sources, lora_wrappers=lora_wrappers)

    def configure_dual_stream(self, *, llm_lora_wrappers) -> None:
        """Passthrough to the inner Qwen3VLModelWrapper (see its docstring)."""
        self.model.configure_dual_stream(llm_lora_wrappers=llm_lora_wrappers)

    @property
    def embed_dim(self) -> int:
        # this is the hidden size that forms the hidden dim of the LLM
        # the visual encoder maps its tokens to this hidden dim
        return self.config.text_config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            labels=labels,
            **kwargs,
        )
        assert not isinstance(outputs, tuple)
        if outputs.labels is not None:
            labels = outputs.labels
        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return Qwen3VLCausalLMOutputWithLastHiddenState(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            last_hidden_state=outputs.last_hidden_state,
            labels=labels,
            metadata=getattr(outputs, 'metadata', None),
            layer_features=getattr(outputs, 'layer_features', None),
            layer_kv_cache=getattr(outputs, 'layer_kv_cache', None),
            image_mask=getattr(outputs, 'image_mask', None),
            vit_features=getattr(outputs, 'vit_features', None),
            vit_features_mask=getattr(outputs, 'vit_features_mask', None),
            vit_features_base=getattr(outputs, 'vit_features_base', None),
            layer_features_lora=getattr(outputs, 'layer_features_lora', None),
        )


@dataclass
class Qwen3VLCausalLMOutputWithLastHiddenState(Qwen3VLCausalLMOutputWithPast):
    """
    Modified output that includes last_hidden_state and optional layer_features.

    Attributes:
        last_hidden_state: Final layer output [B, seq_len, hidden_size]
        labels: Labels for loss computation (proprioception support)
        layer_features: Optional dict of intermediate layer features {layer_idx: Tensor}
            Only populated when extract_layers is provided to forward()
        layer_kv_cache: Optional dict of KV cache from intermediate layers {layer_idx: (key, value)}
            Only populated when extract_layers is provided and use_cache=True
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    # need to output labels given the proprio token is incorporated inside backbone
    labels: Optional[torch.FloatTensor] = None
    metadata: Optional[dict[str, Any]] = None
    # layer-wise features for multi-layer cross attention
    layer_features: Optional[dict[int, torch.FloatTensor]] = None
    # layer-wise KV cache for Pi-KV architecture
    layer_kv_cache: Optional[dict[int, tuple[torch.FloatTensor, torch.FloatTensor]]] = None
    # [B, S_vlm] bool — True at visual-token positions (image or video) in the
    # text sequence; consumed by heads that route image-vs-text attention
    # (e.g. FlowMatchingActionGr00tN1d7 AlternateVL DiT). None when input has
    # no visual modalities.
    image_mask: Optional[torch.Tensor] = None
    # ViT features for head-side vit_condition, re-batched to [B, T_vis, D]:
    # {'ds0'/'ds1'/'ds2': deepstack taps (ascending ViT depth), 'final': last
    # ViT layer}. Only populated when configure_vit_condition() was called.
    vit_features: Optional[dict[str, torch.Tensor]] = None
    # [B, T_vis] bool — True at real (non-padding) ViT token positions.
    vit_features_mask: Optional[torch.Tensor] = None
    # dual_stream: pristine-base ViT features (same keys/shapes as vit_features,
    # which carries the LoRA-adapted stream when dual_stream is active).
    vit_features_base: Optional[dict[str, torch.Tensor]] = None
    # dual_stream: layer features from the LoRA-enabled second LLM pass.
    layer_features_lora: Optional[dict[int, torch.FloatTensor]] = None


@dataclass
class Qwen3VLModelOutputWithLastHiddenState(Qwen3VLModelOutputWithPast):
    """
    Modified output that includes last_hidden_state and optional layer_features.

    Attributes:
        last_hidden_state: Final layer output [B, seq_len, hidden_size]
        labels: Labels for loss computation (proprioception support)
        layer_features: Optional dict of intermediate layer features {layer_idx: Tensor}
            Only populated when extract_layers is provided to forward()
        layer_kv_cache: Optional dict of KV cache from intermediate layers {layer_idx: (key, value)}
            Only populated when extract_layers is provided and use_cache=True
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    # need to output labels given the proprio token is incorporated inside backbone
    labels: Optional[torch.FloatTensor] = None
    metadata: Optional[dict[str, Any]] = None
    # layer-wise features for multi-layer cross attention
    layer_features: Optional[dict[int, torch.FloatTensor]] = None
    # layer-wise KV cache for Pi-KV architecture
    layer_kv_cache: Optional[dict[int, tuple[torch.FloatTensor, torch.FloatTensor]]] = None
    # [B, S_vlm] bool — True at visual-token positions (image or video) in the
    # text sequence; consumed by heads that route image-vs-text attention
    # (e.g. FlowMatchingActionGr00tN1d7 AlternateVL DiT). None when input has
    # no visual modalities.
    image_mask: Optional[torch.Tensor] = None
    # ViT features for head-side vit_condition, re-batched to [B, T_vis, D]:
    # {'ds0'/'ds1'/'ds2': deepstack taps (ascending ViT depth), 'final': last
    # ViT layer}. Only populated when configure_vit_condition() was called.
    vit_features: Optional[dict[str, torch.Tensor]] = None
    # [B, T_vis] bool — True at real (non-padding) ViT token positions.
    vit_features_mask: Optional[torch.Tensor] = None
    # dual_stream: pristine-base ViT features (same keys/shapes as vit_features,
    # which carries the LoRA-adapted stream when dual_stream is active).
    vit_features_base: Optional[dict[str, torch.Tensor]] = None
    # dual_stream: layer features from the LoRA-enabled second LLM pass.
    layer_features_lora: Optional[dict[int, torch.FloatTensor]] = None
