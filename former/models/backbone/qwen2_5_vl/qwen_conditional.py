from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from models.backbone.embedding_utils import replace_placeholder_embeddings
from models.backbone.registry import register_backbone
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLModelOutputWithPast,
)
from transformers.utils import auto_docstring, is_torchdynamo_compiling
from transformers.utils.generic import can_return_tuple


class Qwen2_5_VLModel_Wrapper(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # special handling: only if we don't pre-input the input_ids, we will
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

        # Track 3D [B, S, hidden] visual-token masks so we can surface a 2D
        # [B, S] union to consumers (e.g. FlowMatchingActionGr00tN1d7 AlternateVL).
        image_mask: Optional[torch.Tensor] = None
        video_mask: Optional[torch.Tensor] = None

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
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
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                pos_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = pos_ids.view(1, 1, -1).expand(3, batch_size, -1)
                assert position_ids is not None
                if cache_position is not None:
                    assert self.rope_deltas is not None
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids + delta.to(position_ids.device)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPastWithLastHiddenState(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            labels=labels,
            # Visual-token positions (image|video), [B, S]; consumed by heads
            # that route image-vs-text attention (e.g. AlternateVL DiT).
            image_mask=visual_pos_masks,
        )
        return output if return_dict else output.to_tuple()


@register_backbone
class Qwen2_5_Conditional_VL_Wrapper(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLModel_Wrapper(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
        # HF Qwen2_5_VLVisionAttention.__init__ hardcodes `self.attention_dropout = 0.0`
        # (modeling_qwen2_5_vl.py:193) — override post-init when configured. Same
        # rationale as Qwen3_VL_Wrapper: must be after self.model assignment so we
        # patch the live visual blocks; SDPA gates dropout on self.training so eval
        # is unaffected.
        vision_drop = getattr(config, 'vision_attention_dropout', 0.0)
        if vision_drop > 0:
            for block in self.visual.blocks:
                block.attn.attention_dropout = vision_drop
        self.set_attention_mask(use_causal_mask=config.use_causal_mask)
        self.freeze_backbones(stage=config.stage)
        if(config.enable_gradient_checkpointing):
            self.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    # @property
    # def config(self):
    #     return self._config
    def set_attention_mask(self, use_causal_mask: bool) -> None:
        for layer in self.language_model.layers:
            if hasattr(layer.self_attn, 'is_causal'):
                layer.self_attn.is_causal = use_causal_mask

    def freeze_backbones(self, stage: str) -> None:
        if stage == 'vla-full-train':
            self.language_model.requires_grad_(True)
            self.visual.requires_grad_(True)
        elif stage == 'finetune-frozen-llm':
            self.language_model.requires_grad_(False)
            self.visual.requires_grad_(True)
        elif stage == 'freeze-backbone':
            self.language_model.requires_grad_(False)
            self.visual.requires_grad_(False)
        else:
            raise ValueError(f'Invalid stage: {stage}')

    # @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
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

        return Qwen2_5_VLCausalLMOutputWithLastHiddenState(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            last_hidden_state=outputs.last_hidden_state,
            labels=labels,
            image_mask=getattr(outputs, 'image_mask', None),
        )

    @property
    def embed_dim(self) -> int:
        return self.config.text_config.hidden_size


@dataclass
class Qwen2_5_VLCausalLMOutputWithLastHiddenState(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Modified output that includes last_hidden_state.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    # need to output labels given the proprio token is incorporated inside backbone
    labels: Optional[torch.FloatTensor] = None
    # [B, S_vlm] bool — True at visual-token positions (image or video) in the
    # text sequence; consumed by heads that route image-vs-text attention
    # (e.g. FlowMatchingActionGr00tN1d7 AlternateVL DiT). None when input has
    # no visual modalities.
    image_mask: Optional[torch.Tensor] = None


@dataclass
class Qwen2_5_VLModelOutputWithPastWithLastHiddenState(Qwen2_5_VLModelOutputWithPast):
    """
    Modified output that includes last_hidden_state.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    # need to output labels given the proprio token is incorporated inside backbone
    labels: Optional[torch.FloatTensor] = None
    # [B, S_vlm] bool — True at visual-token positions (image or video) in the
    # text sequence; consumed by heads that route image-vs-text attention
    # (e.g. FlowMatchingActionGr00tN1d7 AlternateVL DiT). None when input has
    # no visual modalities.
    image_mask: Optional[torch.Tensor] = None
