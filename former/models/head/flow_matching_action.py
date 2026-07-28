import torch
import torch.nn as nn
from configs.config_schema import FlowMatchingHeadConfig

from models.common.moe import CategorySpecificMLP, MultiEmbodimentActionEncoder
from models.common.projector import PlainProjector
from models.head.base_head import BaseHead
from models.head.component.language_task import LanguageOnehotEncoder
from models.head.component.vision_encoder import VisionEncoder
from models.head.registry import register_head
from models.head.transformer.dit import DiT
from models.utils.vlm_feature_utils import (
    VLMInterface,
    prepare_encoder_states_for_head,
    prepare_kv_cache_for_head,
    prepare_raw_kv_for_head,
)


@register_head
class FlowMatchingAction(BaseHead):
    def __init__(self, head_config: FlowMatchingHeadConfig, *, backbone_hidden_dim: int):
        super().__init__(head_config)
        self.model_id = head_config.model_id
        self.backbone_hidden_dim = backbone_hidden_dim
        self.num_layers = head_config.num_layers
        self.num_heads = head_config.num_heads
        self.dropout = head_config.dropout
        self.activation = head_config.activation
        self.beta_dist = torch.distributions.Beta(head_config.noise_beta_alpha, head_config.noise_beta_beta)
        self.noise_s = head_config.noise_s
        self.interleave_self_attention = head_config.interleave_self_attention

        self.num_timestep_buckets = head_config.num_timestep_buckets
        self.num_inference_timesteps = head_config.num_inference_timesteps
        # step sizes for inference
        self.t_schedule = self.get_step_schedule(
            self.num_inference_timesteps, head_config.inference_step_schedule
        )

        self.vision_encoder_name = head_config.vision_encoder
        self.language_encoder_name = getattr(head_config, 'language_encoder', None)

        self.add_pos_embed = head_config.add_pos_embed
        if self.add_pos_embed:
            assert self.hidden_dim is not None, 'hidden_dim must be set when add_pos_embed is True'
            self.position_embedding = nn.Embedding(head_config.max_seq_len, self.hidden_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        # Optional state-side PE, gated behind a separate flag (default off).
        # Off  ⇒ no module is created, ``state_dict`` has zero footprint, and
        #         checkpoints trained before state-history existed load
        #         identically to the baseline (state-emb distribution unchanged).
        # On   ⇒ a learnable embedding mirroring the action-side
        #         ``position_embedding`` (small Gaussian init). 64 slots cover
        #         ``HistorySamplingConfig.num_states (<=31)`` + current + headroom.
        # Use for state-history experiments where the DiT needs to distinguish
        # state tokens by time.
        self.add_dit_state_pos_embed = getattr(head_config, 'add_dit_state_pos_embed', False)
        if self.add_dit_state_pos_embed:
            assert self.hidden_dim is not None, (
                'hidden_dim must be set when add_dit_state_pos_embed is True'
            )
            self.state_position_embedding = nn.Embedding(64, self.hidden_dim)
            nn.init.normal_(self.state_position_embedding.weight, mean=0.0, std=0.02)
        self.action_horizon = head_config.action_horizon

        # Architecture parameters
        self.dit_block_type = head_config.dit_block_type
        self.vlm_fusion_mode = head_config.vlm_fusion_mode
        self.vlm_feature_format = head_config.vlm_feature_format
        self.multi_layer_vlm = head_config.multi_layer_vlm
        self.cross_layer_mapping = head_config.cross_layer_mapping
        self.self_has_ffn = head_config.self_has_ffn
        self.qk_norm = getattr(head_config, 'qk_norm', None)
        self.vlm_gate = getattr(head_config, 'vlm_gate', 'none')
        self.vlm_gate_init = getattr(head_config, 'vlm_gate_init', 1.0)
        dual_stream = getattr(head_config, 'dual_stream', None)
        self.dual_stream = dual_stream if (dual_stream is not None and dual_stream.enabled) else None
        condition_router = getattr(head_config, 'condition_router', None)
        self.condition_router = (
            condition_router if (condition_router is not None and condition_router.enabled) else None
        )
        # v1 (static): head premixes with a 32xK matrix. v2 (token): routing lives
        # inside the DiT blocks; the head just emits the raw per-layer condition dict.
        self._router_token = self.condition_router is not None and self.condition_router.mode == 'token'
        self._router_static = self.condition_router is not None and self.condition_router.mode == 'static'

        self._init_model()
        self.get_loss_fn(head_config.loss_type)

    def _init_model(self):
        # Raw KV reuse: skip VLM projectors and use VLM KV cache directly in attention.
        # Supported by separate and cross_only; combined is kept as legacy baseline.
        self.use_raw_kv = (self.dit_block_type != 'combined' and self.vlm_feature_format == 'kv')

        # Initialize VLM projectors (not needed for raw KV mode)
        self.vlm_projectors = nn.ModuleDict()

        if not self.use_raw_kv:
            if self.multi_layer_vlm:
                unique_vlm_layers = set(vlm_layer for _, _, vlm_layer in self.cross_layer_mapping)
                if self.condition_router is not None:
                    # router pool extras get their own per-candidate projectors
                    unique_vlm_layers |= set(self.condition_router.extra_llm_layers)
                for vlm_layer in sorted(unique_vlm_layers):
                    self.vlm_projectors[str(vlm_layer)] = PlainProjector(
                        input_dim=self.backbone_hidden_dim, output_dim=self.hidden_dim
                    )
            else:
                self.vlm_projectors['-1'] = PlainProjector(
                    input_dim=self.backbone_hidden_dim, output_dim=self.hidden_dim
                )

        # vlm_gate: LayerScale-style gate on the projected VLM condition tokens,
        # mirroring vit_condition.fusion.gate ('scalar' | 'channel'; logged as
        # VLMCond/gamma_*). Unlike the ViT segment gate this defaults to init
        # 1.0 — the VLM tokens are the primary condition, so training starts
        # from the identity of the ungated model.
        if self.vlm_gate != 'none':
            assert not self.use_raw_kv, 'vlm_gate requires hidden-format VLM features'
            gate_shape = (1,) if self.vlm_gate == 'scalar' else (self.hidden_dim,)
            self.vlm_gates = nn.ParameterDict(
                {
                    key: nn.Parameter(torch.full(gate_shape, float(self.vlm_gate_init)))
                    for key in self.vlm_projectors.keys()
                }
            )

        self.state_encoder = CategorySpecificMLP(
            num_categories=self.num_categories,
            input_dim=self.state_dim,
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.hidden_dim,
            num_embodiments=self.num_categories,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=self.num_categories,
            input_dim=self.hidden_dim,
            output_dim=self.action_dim,
            hidden_dim=self.hidden_dim // 2,
        )

        # Build DiT kwargs
        dit_kwargs = {
            'num_layers': self.num_layers[0],
            'output_dim': self.hidden_dim,
            'num_attention_heads': self.num_heads,
            'attention_head_dim': self.hidden_dim // self.num_heads,
            'interleave_self_attention': self.interleave_self_attention,
            'activation_fn': self.activation,
            'dropout': self.dropout,
            # Static router (v1): one synthetic span per DiT block, keyed by block
            # index — the head emits {block: premixed_condition}. Token router (v2)
            # and vanilla multi-layer keep the ORIGINAL mapping.
            'cross_layer_mapping': (
                [[i, i, i] for i in range(self.num_layers[0])]
                if self._router_static
                else (self.cross_layer_mapping if self.multi_layer_vlm else None)
            ),
            'dit_block_type': self.dit_block_type,
            'vlm_fusion_mode': self.vlm_fusion_mode,
            'vlm_feature_format': self.vlm_feature_format,
            'qk_norm': self.qk_norm,
        }
        if self._router_token:
            dit_kwargs.update(
                token_router=True,
                token_router_top_k=self.condition_router.top_k,
                token_router_temperature=self.condition_router.temperature,
                token_router_init_bias=self.condition_router.init_bias,
                token_router_tap=self.condition_router.tap,
                token_router_temb_stopgrad=self.condition_router.temb_stopgrad,
            )

        # Add self_has_ffn for separate block type
        if self.dit_block_type == 'separate':
            dit_kwargs['self_has_ffn'] = self.self_has_ffn

        self.backbone = DiT(**dit_kwargs)
        if self.vision_encoder_name is not None:
            vision_history = getattr(self.config, 'vision_history', None)
            self.vision_encoder = VisionEncoder(
                self.vision_encoder_name,
                self.hidden_dim,
                history=(
                    vision_history
                    if (vision_history is not None and vision_history.enabled)
                    else None
                ),
            )
            train_ve_backbone = getattr(self.config, 'train_action_head_vision_encoder', True)
            if not train_ve_backbone and hasattr(self.vision_encoder, 'encoder'):
                self.vision_encoder.encoder.requires_grad_(False)

        # vit_condition: adapters projecting the VLM's own ViT features
        # (DeepStack taps + final layer) into the DiT hidden space.
        self.vit_condition = getattr(self.config, 'vit_condition', None)
        if self.vit_condition is not None and self.vit_condition.enabled:
            from models.head.component.vit_adapters import build_vit_adapter

            assert self.vit_condition.vit_hidden_size is not None, (
                'vit_condition dims unresolved — build the head through PolicyConfig '
                '(its validator copies vision dims from the backbone hf_config)'
            )
            vit_sources = (
                ['ds0', 'ds1', 'ds2', 'final'] if self.vit_condition.mode == 'per_layer' else ['final']
            )
            if self.condition_router is not None and self.condition_router.extra_vit_blocks:
                # router pool extras: block-output taps, each with its own adapter
                vit_sources = vit_sources + [
                    f'blk{n}' for n in sorted(self.condition_router.extra_vit_blocks)
                ]
            self.vit_adapters = nn.ModuleDict(
                {
                    source: build_vit_adapter(
                        self.vit_condition.adapter,
                        self.vit_condition.tap,
                        vit_hidden_size=self.vit_condition.vit_hidden_size,
                        vit_out_hidden_size=self.vit_condition.vit_out_hidden_size,
                        spatial_merge_size=self.vit_condition.spatial_merge_size,
                        output_dim=self.hidden_dim,
                    )
                    for source in vit_sources
                }
            )

            # Segment shaping (vit_condition.fusion): all default-off knobs.
            fusion = self.vit_condition.fusion
            if fusion.gate != 'none':
                gate_shape = (1,) if fusion.gate == 'scalar' else (self.hidden_dim,)
                # gate_init=0: the segment starts content-free (zero values;
                # near-zero attention logits after qk_norm). Not a bit-exact
                # no-op — the masked-in zero tokens still enter the softmax
                # denominator — but the appended CONTENT is zero until gamma
                # opens. Gamma trajectories are logged to TB (ViTCond/gamma_*).
                self.vit_fusion_gates = nn.ParameterDict(
                    {
                        source: nn.Parameter(
                            torch.full(gate_shape, float(fusion.gate_init))
                        )
                        for source in vit_sources
                    }
                )
            if fusion.source_embed:
                self.vit_source_embeds = nn.ParameterDict(
                    {
                        source: nn.Parameter(torch.empty(self.hidden_dim).normal_(mean=0.0, std=0.02))
                        for source in vit_sources
                    }
                )
            if fusion.pos_embed == 'learned':
                # Shared across sources (they share one token layout); per-source
                # identity is carried by source_embed. For fixed-format robot
                # data the index implicitly encodes (camera, spatial) position.
                self.vit_pos_embed = nn.Embedding(fusion.max_pos_tokens, self.hidden_dim)
                nn.init.normal_(self.vit_pos_embed.weight, mean=0.0, std=0.02)

            # dual_stream: second (LoRA-adapted) stream gets its own projectors,
            # adapters and per-stream gates; the existing vlm_projectors /
            # vit_adapters then serve the pristine base stream. Fusion:
            #   gamma_base * base + gamma_lora * lora, per layer/source.
            # gamma_lora starts small-positive (not 0): the summed branch carries
            # full-scale content at init (LoRA-B is zero-init, so lora==base),
            # giving the gate real gradient from step 0 (the fuse123 lesson).
            if self.dual_stream is not None:
                assert not self.use_raw_kv, 'dual_stream requires hidden-format VLM features'
                gate_shape = (1,) if self.dual_stream.gate == 'scalar' else (self.hidden_dim,)

                def _stream_gates(keys, init):
                    return nn.ParameterDict(
                        {key: nn.Parameter(torch.full(gate_shape, float(init))) for key in keys}
                    )

                self.vlm_projectors_lora = nn.ModuleDict(
                    {
                        key: PlainProjector(
                            input_dim=self.backbone_hidden_dim, output_dim=self.hidden_dim
                        )
                        for key in self.vlm_projectors.keys()
                    }
                )
                self.vlm_gates_base = _stream_gates(
                    self.vlm_projectors.keys(), self.dual_stream.base_gate_init
                )
                self.vlm_gates_lora = _stream_gates(
                    self.vlm_projectors.keys(), self.dual_stream.lora_gate_init
                )
                self.vit_adapters_lora = nn.ModuleDict(
                    {
                        source: build_vit_adapter(
                            self.vit_condition.adapter,
                            self.vit_condition.tap,
                            vit_hidden_size=self.vit_condition.vit_hidden_size,
                            vit_out_hidden_size=self.vit_condition.vit_out_hidden_size,
                            spatial_merge_size=self.vit_condition.spatial_merge_size,
                            output_dim=self.hidden_dim,
                        )
                        for source in vit_sources
                    }
                )
                self.vit_gates_base = _stream_gates(vit_sources, self.dual_stream.base_gate_init)
                self.vit_gates_lora = _stream_gates(vit_sources, self.dual_stream.lora_gate_init)
        # condition_router (Tier A, static): fixed sources, learned block-level
        # routing. Row i of each logit matrix mixes the K sources for DiT block i;
        # identity init biases toward the original span mapping (ViT rows honor
        # order=reversed). route_*=False freezes that half as a one-hot buffer.
        # Token mode (v2) has no head-side router — routing lives in the DiT.
        if self._router_static:
            n_blocks = self.num_layers[0]
            owners: list[int | None] = [None] * n_blocks
            for span_idx, (start, end, _layer) in enumerate(self.cross_layer_mapping):
                for i in range(start, end + 1):
                    assert 0 <= i < n_blocks and owners[i] is None, (
                        f'cross_layer_mapping must cover each DiT block exactly once (block {i})'
                    )
                    owners[i] = span_idx
            assert all(o is not None for o in owners), 'cross_layer_mapping must cover every DiT block'
            self._router_owners = owners
            self._router_llm_pool = [entry[2] for entry in self.cross_layer_mapping] + sorted(
                self.condition_router.extra_llm_layers
            )
            n_sources = len(self._router_llm_pool)
            self._router_llm_labels = [str(layer) for layer in self._router_llm_pool]

            def _router_matrix(col_of_block, as_logits):
                mat = torch.zeros(n_blocks, n_sources)
                bias = float(self.condition_router.init_bias) if as_logits else 1.0
                identity = (self.condition_router.init == 'identity') or not as_logits
                if identity:
                    for i in range(n_blocks):
                        mat[i, col_of_block(i)] = bias
                return mat

            if self.condition_router.route_llm:
                self.router_logits_llm = nn.Parameter(_router_matrix(lambda i: owners[i], True))
            else:
                self.router_logits_llm = None
                self.register_buffer('router_onehot_llm', _router_matrix(lambda i: owners[i], False))

            self.router_logits_vit = None
            if self.vit_condition is not None and self.vit_condition.enabled:
                incumbents = ['ds0', 'ds1', 'ds2', 'final']
                ordered = incumbents[::-1] if self.vit_condition.order == 'reversed' else incumbents
                pool = incumbents + [
                    f'blk{n}' for n in sorted(self.condition_router.extra_vit_blocks)
                ]
                self._router_vit_pool = pool
                self._router_vit_labels = pool

                def _vit_col(i):
                    return pool.index(ordered[owners[i]])

                def _vit_matrix(as_logits):
                    mat = torch.zeros(n_blocks, len(pool))
                    bias = float(self.condition_router.init_bias) if as_logits else 1.0
                    if (self.condition_router.init == 'identity') or not as_logits:
                        for i in range(n_blocks):
                            mat[i, _vit_col(i)] = bias
                    return mat

                if self.condition_router.route_vit:
                    self.router_logits_vit = nn.Parameter(_vit_matrix(True))
                else:
                    self.register_buffer('router_onehot_vit', _vit_matrix(False))

        if self.language_encoder_name is not None:
            # note(luome): Now Only support onehot encoder
            # TODO: Support more language encoders
            self.language_encoder = LanguageOnehotEncoder(self.hidden_dim)

    def get_loss_fn(self, loss_type: str):
        self.loss_weight = self.config.loss_weight  # weight to balance the multi-task loss
        self.loss_fn: nn.Module
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f'Loss type {loss_type} not supported')

    def compute_loss(self, output_dict) -> torch.Tensor:
        mask = output_dict.get('mask', None)
        if mask is None:
            mask = torch.ones_like(output_dict['gt'], dtype=torch.bool)

        loss = self.loss_fn(output_dict['pred'], output_dict['gt'])
        loss = loss * mask

        # Per-sample advantage weighting (precomputed offline)
        sample_weight = output_dict.get('sample_weight', None)
        if sample_weight is not None:
            loss = loss * sample_weight[:, None, None]  # (B,) → (B, T, D)

        if self.temporal_weighting:
            temporal_weights = torch.linspace(1, 0.5, output_dict['pred'].shape[1]).to(output_dict['pred'].device)
            loss = loss * temporal_weights[None, :, None].to(output_dict['pred'].device)
        loss = loss.sum()
        mask_sum = mask.sum()

        # Avoid division by zero while keeping computation graph connected for DDP.
        # Using torch.tensor(0.0) would detach from the graph, causing
        # "collective mismatch" in DDP when find_unused_parameters=False.
        final_loss = loss / mask_sum.clamp(min=1)
        return final_loss

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def add_position_embedding(self, x):
        pos_ids = torch.arange(x.shape[1], device=x.device).long()
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        x = x + pos_embs
        return x

    def add_state_position_embedding(self, x):
        if not self.add_dit_state_pos_embed:
            return x
        pos_ids = torch.arange(x.shape[1], device=x.device).long()
        pos_embs = self.state_position_embedding(pos_ids).unsqueeze(0)
        return x + pos_embs

    def get_input_embeddings(self, state, action, embodiment_id, timestep, state_emb=None):
        state_emb = self.state_encoder(state, embodiment_id) if state_emb is None else state_emb
        action_emb = self.action_encoder(action, timestep, embodiment_id)
        state_emb = self.add_state_position_embedding(state_emb)
        if self.add_pos_embed:
            action_emb = self.add_position_embedding(action_emb)
        return torch.cat((state_emb, action_emb), dim=1)

    def get_vlm_tokens(self, vlm_output: VLMInterface):
        """
        Extract VLM features from specified layers.

        Returns:
            - Raw KV mode: dict[vlm_layer_idx, (key_tensor, value_tensor)]
              Each tensor shape: [B, num_kv_heads, seq_len, head_dim]
            - Hidden/pragmatic KV mode: dict[vlm_layer_idx, Tensor [B, seq_len, hidden_dim]]
            - Single-layer mode: {-1: ...}
            - Multi-layer mode: {6: ..., 13: ..., -1: ...}
        """
        layer_mapping = (self.cross_layer_mapping or []) if self.multi_layer_vlm else []
        if self.condition_router is not None and self.condition_router.extra_llm_layers:
            # synthetic entries so prepare_encoder_states_for_head projects the
            # router pool extras too (it only reads entry[2])
            layer_mapping = list(layer_mapping) + [
                [0, 0, layer] for layer in sorted(self.condition_router.extra_llm_layers)
            ]
        assert layer_mapping is not None
        if self.use_raw_kv:
            # True Pi-KV: return raw KV cache directly (no projection)
            return prepare_raw_kv_for_head(
                vlm_output,
                layer_mapping,
            )
        elif self.vlm_feature_format == 'kv':
            # Pragmatic KV: convert KV cache to hidden states then project
            return prepare_kv_cache_for_head(
                vlm_output,
                layer_mapping,
                self.vlm_projectors,
            )
        else:
            # Hidden format: use VLM hidden states
            tokens = prepare_encoder_states_for_head(
                vlm_output,
                layer_mapping,
                self.vlm_projectors,
            )
            if self.dual_stream is not None:
                import dataclasses

                assert vlm_output.layer_features_lora is not None, (
                    'dual_stream enabled but the backbone emitted no layer_features_lora — '
                    'was configure_dual_stream() called (IronVLA._init_dual_stream)?'
                )
                lora_view = dataclasses.replace(
                    vlm_output, layer_features=vlm_output.layer_features_lora
                )
                lora_tokens = prepare_encoder_states_for_head(
                    lora_view, layer_mapping, self.vlm_projectors_lora
                )
                tokens = {
                    k: v * self.vlm_gates_base[str(k)] + lora_tokens[k] * self.vlm_gates_lora[str(k)]
                    for k, v in tokens.items()
                }
            elif self.vlm_gate != 'none':
                tokens = {k: v * self.vlm_gates[str(k)] for k, v in tokens.items()}
            return tokens

    def get_vision_tokens(self, pixel_values, frame_offsets=None):
        vision_tokens = self.vision_encoder(pixel_values, frame_offsets=frame_offsets)
        return vision_tokens

    def get_language_tokens(self, language, device):
        language_tokens = self.language_encoder(language, device=device)
        return language_tokens

    @staticmethod
    def _append_ones_to_encoder_mask(vlm_mask: torch.Tensor, extra_lengths: list[int]) -> torch.Tensor:
        """Append valid positions (ones) for extra encoder segments (e.g. vision, language) after the VLM span."""
        out = vlm_mask
        b = vlm_mask.shape[0]
        device = vlm_mask.device
        for L in extra_lengths:
            if L <= 0:
                continue
            tail = torch.ones(b, L, device=device, dtype=vlm_mask.dtype)
            out = torch.cat([out, tail], dim=1)
        return out

    def _shape_vit_segment(self, source, tokens, mask):
        """Apply the vit_condition.fusion knobs to one adapted segment.

        Order: (+ pos_embed) -> (+ source_embed) -> (* gate). Segment dropout
        (training only) is applied on the shared mask by the caller-facing
        return, so a dropped sample cannot attend to the segment at all.
        """
        fusion = self.vit_condition.fusion
        if fusion.pos_embed == 'learned':
            seq_len = tokens.shape[1]
            assert seq_len <= self.vit_pos_embed.num_embeddings, (
                f'ViT segment length {seq_len} exceeds fusion.max_pos_tokens '
                f'{self.vit_pos_embed.num_embeddings}'
            )
            tokens = tokens + self.vit_pos_embed.weight[:seq_len].unsqueeze(0)
        if fusion.source_embed:
            tokens = tokens + self.vit_source_embeds[source]
        if fusion.gate != 'none':
            tokens = tokens * self.vit_fusion_gates[source]
        return tokens, mask

    def _apply_vit_segment_dropout(self, vit_mask):
        """Per-sample segment dropout on the validity mask (training only)."""
        fusion = self.vit_condition.fusion
        if not self.training or fusion.segment_dropout <= 0 or vit_mask is None:
            return vit_mask
        keep = (
            torch.rand(vit_mask.shape[0], device=vit_mask.device) >= fusion.segment_dropout
        )
        return vit_mask & keep[:, None]

    def _append_vit_condition_tokens(self, vlm_tokens_dict, attention_mask, vlm_output):
        """Concatenate adapted VLM-ViT features onto the per-layer encoder dict.

        * ``final_all``:  adapter('final') tokens appended to EVERY entry
          (same shape as the external-vision_encoder path above).
        * ``per_layer``:  sources [ds0, ds1, ds2, final] (ascending ViT depth;
          ``order='reversed'`` flips them) are zipped onto the
          ``cross_layer_mapping`` entries in their listed span order, so each
          DiT span cross-attends [its VLM-layer tokens | its ViT-source tokens].

        All sources share one token count, so the single DiT-wide encoder
        attention mask stays valid; the appended segment uses the REAL
        per-sample validity mask from the backbone (padding-aware), not ones.
        """
        assert vlm_output is not None and vlm_output.vit_features is not None, (
            'vit_condition enabled but the backbone emitted no vit_features — '
            'was configure_vit_condition() called (IronVLA._init_vit_condition)?'
        )
        vit_features = vlm_output.vit_features
        vis_mask_in = vlm_output.vit_features_mask

        vit_features_base = None
        if self.dual_stream is not None:
            # dual_stream: vit_features carries the LoRA-adapted stream and
            # vit_features_base the pristine one; fuse per source with the
            # per-stream gates (vit_adapters = base, vit_adapters_lora = lora).
            vit_features_base = vlm_output.vit_features_base
            assert vit_features_base is not None, (
                'dual_stream enabled but the backbone emitted no vit_features_base'
            )

        def _adapted_segment(source):
            if self.dual_stream is not None:
                base_tokens, seg_mask = self.vit_adapters[source](
                    vit_features_base[source], vis_mask_in
                )
                lora_tokens, _ = self.vit_adapters_lora[source](vit_features[source], vis_mask_in)
                fused = (
                    base_tokens * self.vit_gates_base[source]
                    + lora_tokens * self.vit_gates_lora[source]
                )
                return fused, seg_mask
            return self.vit_adapters[source](vit_features[source], vis_mask_in)

        # The shared encoder mask must exist to carry the appended segment's
        # padding. If the caller passed none, synthesize ones for the VLM part.
        if attention_mask is None:
            reference = next(iter(vlm_tokens_dict.values()))
            attention_mask = torch.ones(
                reference.shape[0], reference.shape[1], device=reference.device, dtype=torch.long
            )

        if self.vit_condition.mode == 'final_all':
            vit_tokens, vit_mask = _adapted_segment('final')
            vit_tokens, vit_mask = self._shape_vit_segment('final', vit_tokens, vit_mask)
            for key in vlm_tokens_dict.keys():
                vlm_tokens_dict[key] = torch.cat([vlm_tokens_dict[key], vit_tokens], dim=1)
        else:  # per_layer
            sources = ['ds0', 'ds1', 'ds2', 'final']
            if self.vit_condition.order == 'reversed':
                sources = sources[::-1]
            mapping_keys = [entry[2] for entry in self.cross_layer_mapping]
            assert len(set(mapping_keys)) == len(mapping_keys), (
                f'per_layer vit_condition requires distinct vlm_layer keys in '
                f'cross_layer_mapping, got {mapping_keys}'
            )
            vit_mask = None
            for key, source in zip(mapping_keys, sources):
                vit_tokens, vit_mask = _adapted_segment(source)
                vit_tokens, vit_mask = self._shape_vit_segment(source, vit_tokens, vit_mask)
                vlm_tokens_dict[key] = torch.cat([vlm_tokens_dict[key], vit_tokens], dim=1)

        # Per-sample segment dropout (training only): a dropped sample's mask
        # zeroes the whole ViT segment across every span consistently.
        vit_mask = self._apply_vit_segment_dropout(vit_mask)

        if vit_mask is not None:
            attention_mask = torch.cat([attention_mask, vit_mask.to(attention_mask.dtype)], dim=1)
        else:
            s_vit = next(iter(vlm_tokens_dict.values())).shape[1] - attention_mask.shape[1]
            attention_mask = self._append_ones_to_encoder_mask(attention_mask, [s_vit])
        return vlm_tokens_dict, attention_mask

    def _router_weights(self, which: str) -> torch.Tensor:
        """Routing weights [n_blocks, n_sources]: softmax(logits/T) or one-hot buffer.

        With ``top_k > 0`` (static mode) each row keeps its top_k sources and
        renormalizes — dense compute, sparse mix. Dropped logits still receive
        gradient through the softmax denominator, so candidates can re-enter.
        """
        from typing import cast

        assert self.condition_router is not None
        logits = self.router_logits_llm if which == 'llm' else self.router_logits_vit
        if logits is None:
            buffer = self.router_onehot_llm if which == 'llm' else self.router_onehot_vit
            return cast(torch.Tensor, buffer)
        w = torch.softmax(logits / self.condition_router.temperature, dim=-1)
        if self.condition_router.top_k > 0:
            from models.head.transformer.blocks import _top_k_renorm

            w = _top_k_renorm(w, self.condition_router.top_k)
        return w

    def _route_condition_tokens(self, vlm_tokens_dict, attention_mask, vlm_output):
        """Tier-A router: per-block weighted mix of the fixed condition sources.

        LLM candidates share the S_vlm token grid and ViT segments share the
        T_vis grid + one validity mask, so per-block weighted sums are exact
        drop-ins for the fixed span zip. Emits {block_index: condition} matching
        the synthetic one-block-per-span DiT mapping built in _init_model.
        """
        stack_llm = torch.stack([vlm_tokens_dict[k] for k in self._router_llm_pool])  # [K, B, S, D]
        w_llm = self._router_weights('llm').to(stack_llm.dtype)
        routed_llm = torch.einsum('ik,kbsd->ibsd', w_llm, stack_llm)

        if attention_mask is None:
            reference = stack_llm[0]
            attention_mask = torch.ones(
                reference.shape[0], reference.shape[1], device=reference.device, dtype=torch.long
            )

        vit_routed = None
        vit_mask = None
        if self.vit_condition is not None and self.vit_condition.enabled:
            vit_features = vlm_output.vit_features
            assert vit_features is not None, (
                'condition_router with vit_condition enabled but the backbone emitted no vit_features'
            )
            vis_mask_in = vlm_output.vit_features_mask
            segments = []
            for source in self._router_vit_pool:
                seg, vit_mask = self.vit_adapters[source](vit_features[source], vis_mask_in)
                seg, vit_mask = self._shape_vit_segment(source, seg, vit_mask)
                segments.append(seg)
            stack_vit = torch.stack(segments)  # [J, B, T, D]
            w_vit = self._router_weights('vit').to(stack_vit.dtype)
            vit_routed = torch.einsum('ij,jbtd->ibtd', w_vit, stack_vit)
            vit_mask = self._apply_vit_segment_dropout(vit_mask)

        out: dict[int, torch.Tensor] = {}
        for i in range(self.num_layers[0]):
            if vit_routed is not None:
                out[i] = torch.cat([routed_llm[i], vit_routed[i]], dim=1)
            else:
                out[i] = routed_llm[i]

        if vit_routed is not None:
            if vit_mask is not None:
                attention_mask = torch.cat([attention_mask, vit_mask.to(attention_mask.dtype)], dim=1)
            else:
                attention_mask = self._append_ones_to_encoder_mask(attention_mask, [vit_routed.shape[2]])
        return out, attention_mask

    def get_tokens(self, vlm_output: VLMInterface | None, **kwargs):
        """
        Prepare encoder tokens for DiT.

        Returns:
            - Raw KV mode: (dict[vlm_layer, (key, value)], attention_mask)
            - Multi-layer mode: (dict[vlm_layer, Tensor], attention_mask)
            - Single-layer mode: (Tensor [B, seq_len, dim], attention_mask)

        Note:
            Pi-KV raw ``(K, V)`` format returns before optional vision concat; appending vision there would
            require K/V in VLM head shape—use hidden/pragmatic-KV multilayer path for head-side vision tokens.
        """
        attention_mask = None
        vlm_tokens_dict = None

        if vlm_output is not None:
            vlm_tokens_dict = self.get_vlm_tokens(vlm_output)
            if 'attention_mask' in kwargs:
                attention_mask = kwargs['attention_mask']

        if self.use_raw_kv and vlm_tokens_dict is not None:
            return vlm_tokens_dict, attention_mask

        if self.multi_layer_vlm and vlm_tokens_dict is not None:
            if hasattr(self, 'vision_encoder') and kwargs.get('images') is not None:
                vision_tokens = self.get_vision_tokens(
                    kwargs['images'], frame_offsets=kwargs.get('video_delta_indices')
                )
                s_vis = vision_tokens.shape[1]
                for k in vlm_tokens_dict.keys():
                    vlm_tokens_dict[k] = torch.cat([vlm_tokens_dict[k], vision_tokens], dim=1)
                if attention_mask is not None:
                    attention_mask = self._append_ones_to_encoder_mask(attention_mask, [s_vis])
            if self._router_static:
                return self._route_condition_tokens(vlm_tokens_dict, attention_mask, vlm_output)
            # Token router (v2): return the raw per-layer dict; the DiT blocks
            # route + mix per query token internally.
            if self.vit_condition is not None and self.vit_condition.enabled:
                vlm_tokens_dict, attention_mask = self._append_vit_condition_tokens(
                    vlm_tokens_dict, attention_mask, vlm_output
                )
            return vlm_tokens_dict, attention_mask

        # Single-layer mode: VLM + optional vision + optional language, same order as before
        tokens: list[torch.Tensor] = []
        extra_seg_lens: list[int] = []
        if vlm_tokens_dict is not None:
            tokens.append(vlm_tokens_dict[-1])
        if hasattr(self, 'vision_encoder') and kwargs.get('images') is not None:
            vis_t = self.get_vision_tokens(
                kwargs['images'], frame_offsets=kwargs.get('video_delta_indices')
            )
            tokens.append(vis_t)
            extra_seg_lens.append(vis_t.shape[1])
        if hasattr(self, 'language_encoder'):
            lang_device = kwargs['images'].device if kwargs.get('images') is not None else tokens[0].device
            lang_t = self.get_language_tokens(kwargs['raw_language.tasks'], device=lang_device).permute(0, 2, 1)
            tokens.append(lang_t)
            extra_seg_lens.append(lang_t.shape[1])
        if not tokens:
            return None, attention_mask
        encoder_tensor = torch.cat(tokens, dim=1) if len(tokens) > 1 else tokens[0]
        if attention_mask is not None and extra_seg_lens:
            attention_mask = self._append_ones_to_encoder_mask(attention_mask, extra_seg_lens)
        return encoder_tensor, attention_mask

    def forward(self, vlm_output: VLMInterface | None, **kwargs):
        if kwargs['is_training']:
            encoder_tokens, encoder_attention_mask = self.get_tokens(vlm_output, **kwargs)
            actions = kwargs['action']
            noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)

            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype).reshape(-1, 1, 1)
            t_discretized = (t * self.num_timestep_buckets).view(t.shape[0]).long()

            noisy_trajectory = (1 - t) * noise + t * actions
            velocity = actions - noise

            sa_embs = self.get_input_embeddings(
                kwargs['state'], noisy_trajectory, kwargs['embodiment_id'], t_discretized
            )
            hs = self.backbone(
                hidden_states=sa_embs,
                encoder_hidden_states=encoder_tokens,
                encoder_attention_mask=encoder_attention_mask,
                timestep=t_discretized,
            )
            hs = hs[:, -actions.shape[1] :]

            pred_velocity = self.action_decoder(hs, kwargs['embodiment_id'])

            # TODO: align forward output with policy requirements
            # TODO: align loss calculation
            out_dict = {
                'type': 'trajectory',
                'gt': velocity,
                'pred': pred_velocity,
                'mask': kwargs['action_mask'],
            }
            if 'sample_weight' in kwargs:
                out_dict['sample_weight'] = kwargs['sample_weight']
            return out_dict

        else:
            encoder_tokens, encoder_attention_mask = self.get_tokens(vlm_output, **kwargs)
            pred_action = self.get_action(
                kwargs['state'], encoder_tokens, kwargs['embodiment_id'], encoder_attention_mask
            )
            return {
                'type': 'trajectory',
                'gt': kwargs['action'],
                'pred': pred_action,
                'mask': kwargs['action_mask'],
            }

    def get_step_schedule(self, num_steps, type='average'):
        if type == 'average':
            # average: weights = [1/N, 1/N, ..., 1/N], sum = 1
            return [1 / num_steps for _ in range(num_steps)]
        elif type == 'linear_decay':
            if num_steps == 1:
                return [1.0]
            # Linear decay: weights = [N, N-1, ..., 1], sum = N*(N+1)/2
            total = num_steps * (num_steps + 1) / 2
            return [(num_steps - t) / total for t in range(num_steps)]
        else:
            raise ValueError(f'Step schedule type {type} not supported')

    @torch.no_grad()
    def get_action(self, state, vlm_emb, embodiment_id, encoder_attention_mask=None):
        if isinstance(vlm_emb, dict):
            # Dict format: values can be Tensor or (key, value) tuple (raw KV)
            first_val = next(iter(vlm_emb.values()))
            if isinstance(first_val, tuple):
                # Raw KV: extract from key tensor [B, num_heads, seq, head_dim]
                batch_size = first_val[0].shape[0]
                device = first_val[0].device
            else:
                batch_size = first_val.shape[0]
                device = first_val.device
        else:
            # Single-layer mode with concatenated tokens: direct access
            batch_size = vlm_emb.shape[0]
            device = vlm_emb.device
        state_emb = self.state_encoder(state, embodiment_id)

        actions = torch.randn((batch_size, self.action_horizon, self.action_dim), device=device, dtype=state.dtype)

        for t in range(self.num_inference_timesteps):
            t_cont = sum(self.t_schedule[:t])
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            sa_embs = self.get_input_embeddings(None, actions, embodiment_id, timesteps_tensor, state_emb)

            model_output = self.backbone(
                hidden_states=sa_embs,
                encoder_hidden_states=vlm_emb,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )

            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + self.t_schedule[t] * pred_velocity

        return actions
