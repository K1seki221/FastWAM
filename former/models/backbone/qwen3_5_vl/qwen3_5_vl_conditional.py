from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from models.backbone.embedding_utils import replace_placeholder_embeddings
from models.backbone.registry import register_backbone
from transformers import Cache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5DynamicCache,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5TextModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling


class Qwen3_5TextModelWithLayerExtraction(Qwen3_5TextModel):
    """
    Extended Qwen3_5TextModel that supports extracting hidden states from intermediate layers.

    Enables multi-layer cross-attention by collecting and normalizing hidden states
    from specified layers during the forward pass. Handles Qwen3.5's hybrid architecture
    (standard attention + linear attention layers).
    """

    def __init__(self, config):
        super().__init__(config)
        # Parameter-free RMSNorm for intermediate layer feature extraction
        self.feature_extraction_norm = self._create_parameter_free_rmsnorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @staticmethod
    def _create_parameter_free_rmsnorm(hidden_size, eps=1e-6):
        """Create RMSNorm without learnable parameters for feature extraction."""

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
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        extract_layers: list[int] | None = None,
        **kwargs: Any,
    ):
        """Forward pass with optional layer-wise feature extraction.

        Args:
            extract_layers: List of layer indices to extract features from.
                Example: [6, 13, 20, 27]
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # mrope: the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Initialize layer features collection
        layer_features: dict[int, torch.Tensor] | None = {} if extract_layers else None
        layer_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = {} if extract_layers else None

        # Process through decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # Qwen3.5 hybrid: linear_attn layers use linear_attn_mask, others use causal_mask
            layer_mask = linear_attn_mask if decoder_layer.layer_type == 'linear_attention' else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                # FlashAttention uses position_ids only to infer varlen sequence
                # boundaries, so pass the 2D text ids rather than 3D MRoPE ids.
                # MRoPE T/H/W info is already baked into position_embeddings via
                # rotary_emb above, so dropping the 3D shape here loses nothing
                # for rotary; passing the 3D tensor would make flash-attn-2's
                # _flash_attention_forward miscompute cu_seqlens and crash on
                # sm_89 (4090) deploy.
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
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

                # Extract KV cache if use_cache is enabled
                if use_cache and past_key_values is not None:
                    assert layer_kv_cache is not None
                    layer_kv_cache[layer_idx] = past_key_values[layer_idx]

        # Final layer norm — reuse already-computed result if last layer was extracted
        last_layer_idx = len(self.layers) - 1
        if extract_layers and last_layer_idx in extract_layers:
            assert layer_features is not None
            hidden_states = layer_features[last_layer_idx]
        else:
            hidden_states = self.norm(hidden_states)

        return Qwen3_5ModelOutputWithLastHiddenState(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            layer_features=layer_features,
            layer_kv_cache=layer_kv_cache,
        )


class Qwen3_5ModelWrapper(Qwen3_5Model):
    def __init__(self, config):
        super().__init__(config)
        # Replace language_model with layer-extraction-capable version
        self.language_model = Qwen3_5TextModelWithLayerExtraction._from_config(config.text_config)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3_5ModelOutputWithPast]:
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
                )

        # Track 3D [B, S, hidden] visual-token masks so we can surface a 2D
        # [B, S] union to consumers (e.g. FlowMatchingActionGr00tN1d7 AlternateVL).
        image_mask: Optional[torch.Tensor] = None
        video_mask: Optional[torch.Tensor] = None

        if pixel_values is not None:
            # Qwen3.5 returns BaseModelOutputWithPooling
            # pooler_output is a tuple/list of embeddings that need to be concatenated
            image_output = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = image_output.pooler_output

            # Concatenate the list of image embeddings
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            # Qwen3.5 returns BaseModelOutputWithPooling
            # pooler_output is a tuple/list of embeddings that need to be concatenated
            video_output = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = video_output.pooler_output

            # Concatenate the list of video embeddings
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Collapse the per-hidden-dim duplication and union image | video into a
        # single [B, S] visual-token mask for downstream heads. None when input
        # has no visual modality.
        if image_mask is not None and video_mask is not None:
            visual_pos_masks: Optional[torch.Tensor] = image_mask[..., 0] | video_mask[..., 0]
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
        else:
            visual_pos_masks = None

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
            **kwargs,
        )

        return Qwen3_5ModelOutputWithLastHiddenState(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            labels=labels,
            layer_features=getattr(outputs, 'layer_features', None),
            layer_kv_cache=getattr(outputs, 'layer_kv_cache', None),
            # Visual-token positions (image|video), [B, S]; consumed by heads
            # that route image-vs-text attention (e.g. AlternateVL DiT).
            image_mask=visual_pos_masks,
        )


@register_backbone
class Qwen3_5_VL_Wrapper(Qwen3_5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5ModelWrapper(config)
        self.set_attention_mask(use_causal_mask=config.use_causal_mask)
        self.freeze_backbones(stage=config.stage)
        if config.enable_gradient_checkpointing:
            self.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def set_attention_mask(self, use_causal_mask: bool) -> None:
        # Qwen3.5 uses GatedDeltaNet (linear_attn) instead of traditional self_attn
        # GatedDeltaNet inherently handles causality differently, so we skip this for Qwen3.5
        # If needed in future, check for linear_attn and its specific causal handling
        for layer in self.model.language_model.layers:
            # Check for traditional self_attn (not present in Qwen3.5)
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'is_causal'):
                layer.self_attn.is_causal = use_causal_mask
            # Qwen3.5 uses linear_attn, which handles causality differently
            # No action needed for GatedDeltaNet as it has its own causal mechanism

    def freeze_backbones(self, stage: str) -> None:
        if stage == 'vla-full-train':
            self.model.language_model.requires_grad_(True)
            self.model.visual.requires_grad_(True)
        elif stage == 'finetune-frozen-llm':
            self.model.language_model.requires_grad_(False)
            self.model.visual.requires_grad_(True)
        elif stage == 'freeze-backbone':
            self.model.language_model.requires_grad_(False)
            self.model.visual.requires_grad_(False)
        else:
            raise ValueError(f'Invalid stage: {stage}')

    @property
    def embed_dim(self) -> int:
        # this is the hidden size that forms the hidden dim of the LLM
        # the visual encoder maps its tokens to this hidden dim
        return self.config.text_config.hidden_size

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3_5CausalLMOutputWithPast]:
        """Forward pass for Qwen3.5 VLA.

        Args:
            labels: Labels for computing the masked language modeling loss.
            image_grid_thw: The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw: The temporal, height and width of feature shape of each video in LLM.
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
        if outputs.labels is not None:
            labels = outputs.labels
        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return Qwen3_5CausalLMOutputWithLastHiddenState(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            last_hidden_state=outputs.last_hidden_state,
            labels=labels,
            layer_features=getattr(outputs, 'layer_features', None),
            layer_kv_cache=getattr(outputs, 'layer_kv_cache', None),
            image_mask=getattr(outputs, 'image_mask', None),
        )


@dataclass
class Qwen3_5CausalLMOutputWithLastHiddenState(Qwen3_5CausalLMOutputWithPast):
    """Modified output that includes last_hidden_state and optional layer_features."""

    last_hidden_state: Optional[torch.FloatTensor] = None
    # need to output labels given the proprio token is incorporated inside backbone
    labels: Optional[torch.FloatTensor] = None
    # layer-wise features for multi-layer cross attention
    layer_features: Optional[dict[int, torch.FloatTensor]] = None
    # layer-wise KV cache for Pi-KV architecture
    layer_kv_cache: Optional[dict[int, tuple[torch.FloatTensor, torch.FloatTensor]]] = None
    # [B, S_vlm] bool — True at visual-token positions (image or video) in the
    # text sequence; consumed by heads that route image-vs-text attention
    # (e.g. FlowMatchingActionGr00tN1d7 AlternateVL DiT). None when input has
    # no visual modalities.
    image_mask: Optional[torch.Tensor] = None


@dataclass
class Qwen3_5ModelOutputWithLastHiddenState(Qwen3_5ModelOutputWithPast):
    """Modified output that includes last_hidden_state and optional layer_features."""

    last_hidden_state: Optional[torch.FloatTensor] = None
    # need to output labels given the proprio token is incorporated inside backbone
    labels: Optional[torch.FloatTensor] = None
    # layer-wise features for multi-layer cross attention
    layer_features: Optional[dict[int, torch.FloatTensor]] = None
    # layer-wise KV cache for Pi-KV architecture
    layer_kv_cache: Optional[dict[int, tuple[torch.FloatTensor, torch.FloatTensor]]] = None
    # [B, S_vlm] bool — True at visual-token positions (image or video) in the
    # text sequence; consumed by heads that route image-vs-text attention
    # (e.g. FlowMatchingActionGr00tN1d7 AlternateVL DiT). None when input has
    # no visual modalities.
    image_mask: Optional[torch.Tensor] = None
