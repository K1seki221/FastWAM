from collections import OrderedDict
from typing import Any, Dict

import torch
from configs.config_schema import PolicyConfig
from models.backbone.registry import build_backbone
from models.common.moe import CategorySpecificLinear
from models.head.registry import build_head
from models.utils.stage_token_utils import extract_stage_token_from_token_ids
from models.utils.vlm_feature_utils import VLMInterface, collect_required_vlm_layers, resolve_layer_indices
from torch import nn

from dataset.util import registered_tokens


class IronVLMModule(nn.Module):
    def __init__(self, policy_config: PolicyConfig):
        super().__init__()
        self.policy_config = policy_config

        # ===== Backbone =====
        backbone_cfg = policy_config.backbone
        hf_backbone_config = backbone_cfg.hf_config
        assert hf_backbone_config is not None

        self.backbone = build_backbone(hf_backbone_config)
        self.backbone.freeze_backbones(hf_backbone_config.stage)
        self.backbone_hidden_dim = self.backbone.embed_dim
        self.backbone_config = hf_backbone_config

        # ===== Proprio =====
        backbone_cfg = policy_config.backbone
        self.input_proprio = backbone_cfg.input_proprio
        self.proprio_input_encoder = (
            CategorySpecificLinear(
                num_categories=policy_config.num_categories,
                input_dim=policy_config.state_dim,
                hidden_dim=self.backbone_hidden_dim,
            )
            if self.input_proprio
            else None
        )

        # ===== Runtime config injected by wrapper =====
        self.vlm_extract_layers = None
        self.vlm_use_cache = False

    def set_vlm_runtime_config(self, extract_layers=None, use_cache: bool = False):
        self.vlm_extract_layers = extract_layers
        self.vlm_use_cache = use_cache

    def _prepare_backbone_kwargs(self, **kwargs):
        kwargs = dict(kwargs)

        if self.input_proprio:
            kwargs['multi_task_tokens'] = self.proprio_input_encoder(
                kwargs['state'],
                kwargs['embodiment_id'],
            )
            if 'PROPRIO' not in registered_tokens:
                raise ValueError(
                    'PROPRIO token not registered but input_proprio=True. '
                    'Ensure update_processor_cache() is called with PROPRIO token before inference.'
                )
            kwargs['proprio_placeholder_token_id'] = registered_tokens['PROPRIO'].item()

            # Per-row gate for proprio placeholder replacement: only rows with
            # at least one valid state dim (state_mask truthy somewhere) carry
            # a PROPRIO placeholder in input_ids. VL samples from VLDataset
            # have state_mask=all-False (see get_action_state_padding) and
            # skip the placeholder; the embedding we still compute from their
            # zero state is discarded downstream. Fallback (state_mask absent)
            # treats every row as a robot row for legacy code paths.
            state_mask = kwargs.get('state_mask')
            if state_mask is not None:
                kwargs['proprio_row_mask'] = state_mask.flatten(start_dim=1).any(dim=-1)

        if self.vlm_extract_layers is not None:
            kwargs['extract_layers'] = self.vlm_extract_layers

        if self.vlm_use_cache:
            kwargs['use_cache'] = True

        return kwargs

    def forward(self, **kwargs) -> VLMInterface:
        backbone_kwargs = self._prepare_backbone_kwargs(**kwargs)
        output = self.backbone(**backbone_kwargs)

        vlm_interface = VLMInterface.from_backbone_output(output)

        if kwargs.get('return_stage_token', False):
            if not (hasattr(output, 'logits') and output.logits is not None):
                raise ValueError('return_stage_token=True requires backbone output with logits')

            pred_token_ids = torch.argmax(output.logits, dim=-1)  # (B, L)
            model_id = self.backbone_config.model_id
            # ``extract_stage_token_from_token_ids`` returns ``None`` for VL/VQA
            # batches whose ``input_ids`` carry no ``<|CLS|>``; leave
            # ``vlm_interface.stage_token`` unset (handled downstream by the
            # ``if st is not None`` check in ``IronVLA.forward``).
            stage_token = extract_stage_token_from_token_ids(
                token_ids=pred_token_ids,
                input_ids=kwargs['input_ids'],
                model_id=model_id,
            )
            if stage_token is not None:
                vlm_interface.stage_token = stage_token

        return vlm_interface

    def _get_output_dict(self, pred_tokens, **kwargs):
        out_dict = {
            'type': 'token',
            'pred': pred_tokens,
            'gt': kwargs.get('output_ids', None),
        }
        # ``head_cfg`` is a Pydantic model (e.g. ``ElementWiseLMHeadConfig``)
        # when configured, or ``None`` when no next_token_predict head is
        # set up — use ``getattr`` for attribute access, not the dict
        # ``.get`` method that Pydantic models don't expose.
        head_cfg = getattr(self.policy_config.heads, 'next_token_predict', None)
        task = getattr(head_cfg, 'task', 'text') if head_cfg is not None else 'text'
        return {task: out_dict}

    def generate(self, **kwargs):
        # See _get_output_dict above: head_cfg is a Pydantic model, use
        # getattr. ``ElementWiseLMHeadConfig`` doesn't declare a
        # ``generation_config`` field but ``extra='allow'`` lets one pass
        # through from yaml; getattr returns ``None`` if absent, then
        # ``or {}`` falls back to the empty-dict default for kwargs spread.
        head_cfg = getattr(self.policy_config.heads, 'next_token_predict', None)
        generation_config = (getattr(head_cfg, 'generation_config', None) or {}) if head_cfg is not None else {}

        generation_input = {
            k: v
            for k, v in kwargs.items()
            if k in ['input_ids', 'image_grid_thw', 'pixel_values', 'attention_mask']
        }

        pred_tokens = self.backbone.generate(**generation_input, **generation_config)
        pred_tokens = [pred_tokens.detach().cpu()[i] for i in range(pred_tokens.shape[0])]
        return self._get_output_dict(pred_tokens, **kwargs)


class IronVLA(nn.Module):
    def __init__(self, policy_config: PolicyConfig):
        super().__init__()
        self.policy_config = policy_config
        stm_config = getattr(policy_config, 'stm', None)
        self.stm_config = stm_config if (stm_config is not None and getattr(stm_config, 'enabled', False)) else None

        self.use_vlm_backbone = policy_config.use_vlm_backbone
        self.hidden_dim = policy_config.hidden_dim

        if not self.use_vlm_backbone:
            self.vlm = None
            self.backbone_hidden_dim = 10
        else:
            self.vlm = IronVLMModule(policy_config)
            self.backbone_hidden_dim = self.vlm.backbone_hidden_dim

        # STM (temporal attention in ViT) — attached when stm_config.enabled.
        # When disabled, the backbone wrapper's `temporal_vit_encoder` attribute
        # stays None and Qwen3VLModelWrapper falls through to the base path.
        self.temporal_vit_encoder = None
        if self.use_vlm_backbone and self.stm_config is not None:
            from models.backbone.temporal_vit import TemporalViTEncoder

            assert self.vlm is not None  # use_vlm_backbone=True implies vlm initialised
            self.temporal_vit_encoder = TemporalViTEncoder(
                self.vlm.backbone.model.visual, self.stm_config,
            )
            self.vlm.backbone.model.temporal_vit_encoder = self.temporal_vit_encoder
            print(f'STM enabled: {len(self.temporal_vit_encoder.temporal_attn_layers)} temporal attention layers')

        self.norm_stats = None

        # Get active heads from HeadsConfig - check which fields are not None
        self.active_heads = [
            name
            for name in ['action_predict', 'next_token_predict', 'value_predict', 'bbox_detr_predict']
            if getattr(policy_config.heads, name, None) is not None
        ]

        self._init_heads_from_config()

        # Determine VLM output configuration at initialization (not at forward time)
        # This resolves the responsibility leakage issue
        self._init_vlm_output_config()

        # ViT-feature conditioning for the action head (DeepStack + final layer)
        self._init_vit_condition()

        # Dual-stream condition: LLM LoRA + second (base+LoRA) pass emission
        self._init_dual_stream()

    def _init_heads_from_config(self):
        self.heads = nn.ModuleDict()
        self.head_configs = {}

        heads_config = self.policy_config.heads
        if heads_config is None:
            raise ValueError('Config must contain a "heads" section.')

        # Iterate over all active heads (explicit + dynamic)
        for head_name in self.active_heads:
            head_cfg = getattr(heads_config, head_name, None)
            if head_cfg is None:
                continue

            # Build runtime kwargs (values determined after backbone init, not from config)
            runtime_kwargs = {'backbone_hidden_dim': self.backbone_hidden_dim}

            # For next_token_predict, add vocab_size
            if head_name == 'next_token_predict' and self.use_vlm_backbone:
                # Try multiple paths to get vocab_size, supporting different model architectures
                vocab_size = (
                    getattr(self.vlm.backbone, 'vocab_size', None)
                    or getattr(self.vlm.backbone.config, 'vocab_size', None)
                    or getattr(getattr(self.vlm.backbone.config, 'text_config', None), 'vocab_size', 151936)
                )
                runtime_kwargs['out_features'] = vocab_size
                runtime_kwargs['model_id'] = self.policy_config.backbone.model_id

            if head_name == 'bbox_detr_predict' and self.use_vlm_backbone:
                runtime_kwargs['model_id'] = self.policy_config.backbone.model_id

            self.heads[head_name] = build_head(head_cfg.type, head_cfg, **runtime_kwargs)
            self.head_configs[head_name] = head_cfg

    def _init_vlm_output_config(self):
        """
        Determine VLM output configuration at initialization time.

        This method inspects head configs once during initialization to determine:
        - Which VLM layers need to be extracted
        - Whether KV cache is needed

        This avoids the responsibility leakage of checking head configs at forward time.
        """
        if not self.use_vlm_backbone:
            return

        # Collect required VLM layers for multi-layer cross attention
        raw_extract_layers = collect_required_vlm_layers(self.head_configs)

        if raw_extract_layers:
            num_vlm_layers = self.vlm.backbone.config.text_config.num_hidden_layers
            resolved_layers = resolve_layer_indices(raw_extract_layers, num_vlm_layers)
        else:
            resolved_layers = None

        # Check if any head uses KV format, which requires use_cache=True
        vlm_use_cache = any(
            getattr(head_cfg, 'vlm_feature_format', 'hidden') == 'kv'
            for head_cfg in self.head_configs.values()
        )

        self.vlm.set_vlm_runtime_config(
            extract_layers=resolved_layers,
            use_cache=vlm_use_cache,
        )

    def _init_vit_condition(self):
        """Configure backbone-side ViT feature capture for action-head vit_condition.

        Mirrors ``_init_vlm_output_config``: inspects the head config once at
        init and injects runtime configuration into the backbone wrapper.
        Config-level validation (backbone type, STM, kv-format, span count)
        already ran in the Pydantic schema; the asserts here are defensive.
        """
        action_cfg = self.head_configs.get('action_predict')
        vit_condition = getattr(action_cfg, 'vit_condition', None) if action_cfg else None
        if vit_condition is None or not vit_condition.enabled:
            return

        assert self.use_vlm_backbone and self.vlm is not None
        assert self.stm_config is None, 'vit_condition is incompatible with STM'
        wrapper = self.vlm.backbone  # Qwen3_VL_Wrapper (validated in PolicyConfig)

        sources = (
            ['ds0', 'ds1', 'ds2', 'final'] if vit_condition.mode == 'per_layer' else ['final']
        )
        router = getattr(action_cfg, 'condition_router', None)
        if router is not None and router.enabled and router.extra_vit_blocks:
            sources = sources + [f'blk{n}' for n in sorted(router.extra_vit_blocks)]

        lora_wrappers = None
        if vit_condition.lora.enabled:
            from models.backbone.vit_lora import install_vit_lora, mark_vit_lora_trainable

            # Installed after from_pretrained and after both freeze_backbones
            # calls (wrapper __init__ and IronVLMModule.__init__), so re-enable
            # grads on the LoRA params only; the base ViT keeps its stage policy.
            lora_wrappers = install_vit_lora(
                wrapper.model.visual,
                r=vit_condition.lora.r,
                alpha=vit_condition.lora.alpha,
                dropout=vit_condition.lora.dropout,
                targets=list(vit_condition.lora.targets),
            )
            mark_vit_lora_trainable(lora_wrappers)
            print(f'vit_condition LoRA: {len(lora_wrappers)} wrapped Linears '
                  f'(r={vit_condition.lora.r}, targets={list(vit_condition.lora.targets)})')

        wrapper.configure_vit_condition(
            tap=vit_condition.tap, sources=sources, lora_wrappers=lora_wrappers
        )
        print(f'vit_condition enabled: mode={vit_condition.mode}, order={vit_condition.order}, '
              f'tap={vit_condition.tap}, adapter={vit_condition.adapter}, sources={sources}')

    def _init_dual_stream(self):
        """Install LLM LoRA and enable dual-stream emission on the backbone.

        Runs after _init_vit_condition (needs the ViT LoRA wrappers already
        installed). Cross-field constraints (llm_lora <-> dual_stream <->
        stage=freeze-backbone) were validated in PolicyConfig.
        """
        action_cfg = self.head_configs.get('action_predict')
        dual = getattr(action_cfg, 'dual_stream', None) if action_cfg else None
        if dual is None or not dual.enabled:
            return
        assert self.use_vlm_backbone and self.vlm is not None
        llm_lora = self.policy_config.backbone.llm_lora
        assert llm_lora is not None and llm_lora.enabled  # PolicyConfig-validated
        wrapper = self.vlm.backbone

        from models.backbone.vit_lora import install_llm_lora, mark_vit_lora_trainable

        llm_wrappers = install_llm_lora(
            wrapper.model.language_model,
            r=llm_lora.r,
            alpha=llm_lora.alpha,
            dropout=llm_lora.dropout,
            targets=list(llm_lora.targets),
        )
        mark_vit_lora_trainable(llm_wrappers)
        wrapper.configure_dual_stream(llm_lora_wrappers=llm_wrappers)
        print(f'dual_stream enabled: LLM LoRA on {len(llm_wrappers)} Linears '
              f'(r={llm_lora.r}, targets={list(llm_lora.targets)}), '
              f'gates {dual.gate} base_init={dual.base_gate_init} lora_init={dual.lora_gate_init}')

    def reset_active_heads(self) -> None:
        heads_config = self.policy_config.heads
        active_heads = []
        if getattr(heads_config, 'action_predict', None) is not None:
            active_heads.append('action_predict')
        if getattr(heads_config, 'next_token_predict', None) is not None:
            active_heads.append('next_token_predict')
        if getattr(heads_config, 'value_predict', None) is not None:
            active_heads.append('value_predict')
        if getattr(heads_config, 'bbox_detr_predict', None) is not None:
            active_heads.append('bbox_detr_predict')
        self.active_heads = active_heads

    def set_active_heads(self, active_heads: list[str]) -> None:
        self.active_heads = active_heads

    def get_inference_strategy(self, inference_type: str):
        strategies = {
            'forward': self.forward,
            'generate': self.generate,
        }
        if inference_type not in strategies:
            raise ValueError(f'Invalid inference type: {inference_type}')
        return strategies[inference_type]

    def run_inference(self, inference_type: str = 'forward', **kwargs):
        strategy = self.get_inference_strategy(inference_type)
        return strategy(**kwargs)

    def forward(self, compute_loss: bool | None = None, **kwargs):
        """Run policy forward pass.

        Args:
            compute_loss: Selects the dispatch branch independently of
                ``self.training``. ``True`` runs the loss-producing branch
                (heads see ``is_training=True``, output goes through
                ``compute_loss`` to a ``loss_dict``); ``False`` runs the
                inference branch (heads see ``is_training=False``, returns
                per-task prediction dicts). ``None`` (default) falls back
                to ``self.training`` so the historical contract — train
                mode produces loss, eval mode produces predictions —
                still holds for callers that don't pass it explicitly
                (the train loop's ``forward_pass``, ``run_inference``).
                Validation paths that want loss without dropout active
                (``validation_loss_evaluation``) call ``policy.eval()``
                and pass ``compute_loss=True`` explicitly so dropout/BN
                are governed by ``self.training`` while the dispatch is
                governed by this flag.
            input_ids: tensor of integer encodings from tokenizer, language should be in the format of input_ids
            embodiment_id: batch of integers representing index to take from norm stats, default is 0 if no moe
        """
        if compute_loss is None:
            compute_loss = self.training
        if self.use_vlm_backbone:
            assert self.vlm is not None  # narrowed: use_vlm_backbone=True implies vlm initialised
            if self.temporal_vit_encoder is not None:
                # STM routing: number of frames (current + history) per sample.
                # Deployment path with the cache active receives K=1 and pulls
                # history from the cache instead of re-running the ViT.
                assert self.stm_config is not None  # set together with temporal_vit_encoder in __init__
                if self.temporal_vit_encoder.cache is not None:
                    kwargs['stm_num_frames'] = 1
                else:
                    kwargs['stm_num_frames'] = self.stm_config.num_history_frames + 1
            kwargs['vlm_output'] = self.vlm(**kwargs)
            # Lift the VLM's visual-token mask (image|video positions in the
            # text sequence, [B, S_vlm] bool) into the head kwargs so heads
            # that route attention by modality (e.g. FlowMatchingActionGr00tN1d7's
            # AlternateVL DiT) actually receive it. Only set when the backbone
            # surfaced a non-None mask — heads / backbones that don't use it
            # are unaffected (kwargs['image_mask'] simply stays absent).
            vlm_image_mask = getattr(kwargs['vlm_output'], 'image_mask', None)
            if vlm_image_mask is not None:
                kwargs['image_mask'] = vlm_image_mask

        outputs = {}
        for head_type, head in self.heads.items():
            if head_type not in self.active_heads:
                continue
            kwargs['is_training'] = compute_loss
            head_cfg = getattr(self.policy_config.heads, head_type)
            task = head_cfg.task if head_cfg else head_type
            vlm_out = kwargs.get('vlm_output')
            head_kw = {k: v for k, v in kwargs.items() if k != 'vlm_output'}

            outputs[task] = head(vlm_out, **head_kw)

        if compute_loss:
            return self.compute_loss(outputs)

        if (
            self.use_vlm_backbone
            and kwargs.get('return_stage_token', False)
            and kwargs.get('vlm_output') is not None
        ):
            st = getattr(kwargs['vlm_output'], 'stage_token', None)
            if st is not None:
                # Mirror the action task's dict storage so the eval pipeline
                # (results_to_dataframe + MetricKind.task_type filter) treats
                # stage_token uniformly. GT is added to this same dict by
                # IronInferencer._extract_gt_stage_token; both pred/gt get
                # decoded in place by extract_stage_tokens_strings.
                outputs['stage_token'] = {'pred': st}

        return outputs

    def compute_loss(self, outputs):
        loss_dict = {'loss': 0}
        for head_type, head in self.heads.items():
            if head_type not in self.active_heads:
                continue
            head_cfg = getattr(self.policy_config.heads, head_type)
            task = head_cfg.task if head_cfg else head_type
            head_output = outputs[task]
            head_loss_result = head.compute_loss(head_output)

            if isinstance(head_loss_result, dict):
                # Dict return: extract 'loss' for total, keep full dict for TB logging
                loss_dict[task] = head_loss_result
                loss_dict['loss'] += head_loss_result['loss'] * head.loss_weight
            else:
                # Scalar tensor return (backward compatible)
                loss_dict[task] = head_loss_result
                loss_dict['loss'] += head_loss_result * head.loss_weight

        return loss_dict

    def support_fsdp(self) -> bool:
        return False

    @staticmethod
    def _remap_legacy_flat_vla_to_vlm_submodule(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Map flat ckpt keys (backbone.* / proprio*) to nested self.vlm.*. Heads stay heads.* (not IRON_VLA_SPLIT)."""
        remapped: OrderedDict[str, Any] = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith('backbone.'):
                new_key = new_key.replace('backbone.', 'vlm.backbone.', 1)
            elif new_key.startswith('proprio_input_encoder.'):
                new_key = new_key.replace('proprio_input_encoder.', 'vlm.proprio_input_encoder.', 1)
            remapped[new_key] = value
        return remapped

    @staticmethod
    def _remap_vlm_submodule_to_legacy_flat_vla(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Map vlm.backbone.* / vlm.proprio_* back to flat backbone.* for legacy-compatible saves."""
        remapped: OrderedDict[str, Any] = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith('vlm.backbone.'):
                new_key = new_key.replace('vlm.backbone.', 'backbone.', 1)
            elif new_key.startswith('vlm.proprio_input_encoder.'):
                new_key = new_key.replace('vlm.proprio_input_encoder.', 'proprio_input_encoder.', 1)
            remapped[new_key] = value
        return remapped

    def serialize(self):
        """Export state dict in legacy flat layout (backbone.*, heads.*)."""
        return self._remap_vlm_submodule_to_legacy_flat_vla(self.state_dict())

    def deserialize(self, model_dict: Dict[str, Any], strict: bool = True) -> Any:
        """Load weights: legacy flat keys or already nested vlm.*; never remap heads.* to heads.heads.*."""
        if any(k.startswith('vlm.') for k in model_dict.keys()):
            to_load = model_dict
        else:
            to_load = self._remap_legacy_flat_vla_to_vlm_submodule(model_dict)
        return self.load_state_dict(to_load, strict=strict)

    def generate(self, **kwargs):
        if not self.use_vlm_backbone or self.vlm is None:
            raise RuntimeError('generate() requires use_vlm_backbone=True')
        return self.vlm.generate(**kwargs)

