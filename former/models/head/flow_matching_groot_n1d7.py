# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Flow-matching action head matching the NVIDIA Isaac-GR00T-N1.7 architecture.

The submodule layout (``vlln`` → ``vl_self_attention`` → ``model`` (DiT) →
``action_decoder``, plus ``state_encoder`` / ``action_encoder`` / ``position_embedding``)
mirrors ``Gr00tN1d7ActionHead`` upstream. Submodule attribute names are kept
identical so the safetensors checkpoint can be loaded with only a thin name
remapper (handled by the GR00T → iron_vla converter).
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.common.moe import CategorySpecificMLP, MultiEmbodimentActionEncoder
from models.head.base_head import BaseHead
from models.head.groot_n1d7_components import SelfAttentionStack
from models.head.registry import register_head
from models.head.transformer.dit import DiT
from models.utils.vlm_feature_utils import VLMInterface


@register_head
class FlowMatchingActionGr00tN1d7(BaseHead):
    """GR00T-N1.7 flow-matching action head with AlternateVLDiT routing.

    Forward expects (matching iron_vla head contract):
      - ``vlm_output``: ``VLMInterface`` with ``last_hidden_state`` [B, S_vlm, backbone_hidden_dim]
      - ``kwargs['state']``: [B, max_state_dim] or [B, state_history_length, max_state_dim]
      - ``kwargs['action']``: [B, action_horizon, max_action_dim] (training)
      - ``kwargs['action_mask']``: [B, action_horizon, max_action_dim]
      - ``kwargs['embodiment_id']``: [B] long
      - ``kwargs['attention_mask']``: [B, S_vlm] (VLM padding mask, optional)
      - ``kwargs['image_mask']``: [B, S_vlm] bool, True at vision tokens (optional;
            required for AlternateVL routing — falls back to text-only attention if absent)
      - ``kwargs['is_training']``: bool

    Output (training): ``{'type': 'trajectory', 'gt': velocity, 'pred': pred_velocity, 'mask': action_mask}``.
    Output (inference): ``{'type': 'trajectory', 'gt': action, 'pred': pred_action, 'mask': action_mask}``.
    """

    def __init__(self, head_config, *, backbone_hidden_dim: int):
        super().__init__(head_config)
        cfg = head_config

        self.backbone_hidden_dim = backbone_hidden_dim  # e.g. 2048 for Cosmos-Reason2-2B
        # iron_vla's IronVLA.forward multiplies by head.loss_weight to combine head losses;
        # cache it on self as the other heads do.
        self.loss_weight = float(cfg.loss_weight)

        # Architectural shape params (must match N1.7-3B exactly to load weights strictly).
        self.action_horizon = int(cfg.action_horizon)
        self.max_action_dim = int(cfg.max_action_dim)
        self.max_state_dim = int(cfg.max_state_dim)
        self.state_history_length = int(getattr(cfg, 'state_history_length', 1))
        self.input_embedding_dim = int(cfg.input_embedding_dim)  # = inner_dim of DiT (1536)
        self.hidden_size = int(cfg.hidden_size)                  # = output_dim of DiT (1024)
        self.max_num_embodiments = int(cfg.max_num_embodiments)
        self.max_seq_len = int(cfg.max_seq_len)                  # 1024
        self.add_pos_embed = bool(cfg.add_pos_embed)
        self.use_vlln = bool(cfg.use_vlln)
        self.attend_text_every_n_blocks = int(getattr(cfg, 'attend_text_every_n_blocks', 2))

        # Diffusion model (DiT) sub-config.
        dm_cfg = cfg.diffusion_model_cfg
        self.num_dit_layers = int(dm_cfg['num_layers'])
        self.dit_num_heads = int(dm_cfg['num_attention_heads'])
        self.dit_head_dim = int(dm_cfg['attention_head_dim'])
        self.dit_dropout = float(dm_cfg.get('dropout', 0.2))
        self.dit_final_dropout = bool(dm_cfg.get('final_dropout', True))
        self.dit_norm_type = str(dm_cfg.get('norm_type', 'ada_norm'))
        self.dit_output_dim = int(dm_cfg.get('output_dim', self.hidden_size))
        self.dit_interleave = bool(dm_cfg.get('interleave_self_attention', True))

        # Sanity: inner_dim from heads × head_dim must equal input_embedding_dim.
        inner_dim = self.dit_num_heads * self.dit_head_dim
        if inner_dim != self.input_embedding_dim:
            raise ValueError(
                f'DiT inner_dim ({self.dit_num_heads}×{self.dit_head_dim}={inner_dim}) '
                f'must equal input_embedding_dim ({self.input_embedding_dim})'
            )

        # Flow matching schedule.
        self.num_inference_timesteps = int(cfg.num_inference_timesteps)
        self.num_timestep_buckets = int(cfg.num_timestep_buckets)
        self.beta_dist = torch.distributions.Beta(
            float(cfg.noise_beta_alpha), float(cfg.noise_beta_beta)
        )
        self.noise_s = float(cfg.noise_s)

        # State dropout (training-time augmentation).
        self.state_dropout_prob = float(getattr(cfg, 'state_dropout_prob', 0.0))

        # ---- Sub-modules (names match the N1.7 ckpt exactly) ----

        # vlln: LayerNorm over backbone features (2048-dim) before vl_self_attention.
        self.vlln = nn.LayerNorm(self.backbone_hidden_dim) if self.use_vlln else nn.Identity()

        # vl_self_attention: 4-layer self-attn block over VLM features.
        vl_sa_cfg = getattr(cfg, 'vl_self_attention_cfg', None)
        if vl_sa_cfg and int(vl_sa_cfg.get('num_layers', 0)) > 0:
            self.vl_self_attention: nn.Module = SelfAttentionStack(
                num_layers=int(vl_sa_cfg['num_layers']),
                num_attention_heads=int(vl_sa_cfg['num_attention_heads']),
                attention_head_dim=int(vl_sa_cfg['attention_head_dim']),
                dropout=float(vl_sa_cfg.get('dropout', 0.0)),
                final_dropout=bool(vl_sa_cfg.get('final_dropout', True)),
                attention_bias=True,
                activation_fn='gelu-approximate',
            )
        else:
            self.vl_self_attention = nn.Identity()

        # state_encoder: per-embodiment MLP, max_state_dim*history → hidden_size → input_embedding_dim.
        self.state_encoder = CategorySpecificMLP(
            num_categories=self.max_num_embodiments,
            input_dim=self.max_state_dim * self.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )

        # action_encoder: per-embodiment MLP with sin/cos timestep mixing.
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.max_action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=self.max_num_embodiments,
        )

        # action_decoder: per-embodiment MLP, hidden_size → hidden_size → max_action_dim.
        self.action_decoder = CategorySpecificMLP(
            num_categories=self.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.max_action_dim,
        )

        # position_embedding: nn.Embedding(max_seq_len, input_embedding_dim).
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(self.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # model (DiT, AlternateVL-style): 32 layers, [cross, self] alternating, K/V from VLM(2048).
        self.model = DiT(
            num_attention_heads=self.dit_num_heads,
            attention_head_dim=self.dit_head_dim,
            output_dim=self.dit_output_dim,
            num_layers=self.num_dit_layers,
            dropout=self.dit_dropout,
            attention_bias=True,
            norm_type=self.dit_norm_type,
            norm_elementwise_affine=False,
            final_dropout=self.dit_final_dropout,
            positional_embeddings=None,
            activation_fn='gelu-approximate',
            dit_block_type='separate',
            vlm_fusion_mode='cross',
            vlm_feature_format='hidden',
            self_has_ffn=True,
            flip_self_cross=True,
            cross_attention_dim=self.backbone_hidden_dim,
            attend_text_every_n_blocks=self.attend_text_every_n_blocks,
        )

    # --- Helpers ----------------------------------------------------------------

    def _sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return (1 - sample) * self.noise_s

    def _build_dit_inputs(
        self,
        state: torch.Tensor,                # [B, state_history_length, max_state_dim]
        actions_or_noise: torch.Tensor,     # [B, action_horizon, max_action_dim]
        embodiment_id: torch.Tensor,        # [B]
        timesteps_discrete: torch.Tensor,   # [B] long
    ) -> torch.Tensor:
        """Encode (state, noisy-actions, t) to the DiT input tokens [B, 1+action_horizon, input_embedding_dim]."""
        b = state.shape[0]
        state_flat = state.view(b, 1, -1)  # [B, 1, max_state_dim*history]
        state_features = self.state_encoder(state_flat, embodiment_id)  # [B, 1, input_embedding_dim]

        # State dropout (training-only augmentation).
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (torch.rand(b, device=state_features.device) < self.state_dropout_prob)
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        action_features = self.action_encoder(actions_or_noise, timesteps_discrete, embodiment_id)
        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        return torch.cat((state_features, action_features), dim=1)  # [B, 1+AH, input_embedding_dim]

    def _process_backbone(
        self, backbone_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply vlln + vl_self_attention; return processed VLM features [B, S, backbone_hidden_dim]."""
        x = self.vlln(backbone_features)
        return self.vl_self_attention(x)

    def _normalize_state_shape(self, state: torch.Tensor) -> torch.Tensor:
        """Accept state as [B, D] or [B, T, D]; return [B, state_history_length, max_state_dim]."""
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.shape[1] != self.state_history_length:
            raise ValueError(
                f'state.shape[1] ({state.shape[1]}) != state_history_length ({self.state_history_length})'
            )
        return state

    # --- Forward ---------------------------------------------------------------

    def forward(self, vlm_output: Optional[VLMInterface], **kwargs) -> Dict[str, Any]:
        if vlm_output is None or vlm_output.last_hidden_state is None:
            raise ValueError('FlowMatchingActionGr00tN1d7 requires vlm_output with last_hidden_state')

        backbone_features = vlm_output.last_hidden_state  # [B, S, 2048]
        backbone_attention_mask = kwargs.get('attention_mask')
        image_mask = kwargs.get('image_mask')

        embodiment_id = kwargs['embodiment_id'].long()
        state = self._normalize_state_shape(kwargs['state'])

        vl_embeds = self._process_backbone(backbone_features)

        is_training = bool(kwargs.get('is_training', self.training))

        if is_training:
            actions = kwargs['action']
            action_mask = kwargs['action_mask']
            noise = torch.randn_like(actions)
            t = self._sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
            t = t[:, None, None]  # [B, 1, 1]
            noisy_trajectory = (1 - t) * noise + t * actions
            velocity = actions - noise
            t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

            sa_embs = self._build_dit_inputs(state, noisy_trajectory, embodiment_id, t_discretized)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=backbone_attention_mask,
                timestep=t_discretized,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )  # [B, 1+AH, dit_output_dim]

            pred = self.action_decoder(model_output, embodiment_id)  # [B, 1+AH, max_action_dim]
            pred_velocity = pred[:, -actions.shape[1]:]              # [B, AH, max_action_dim]

            out = {
                'type': 'trajectory',
                'gt': velocity,
                'pred': pred_velocity,
                'mask': action_mask,
            }
            if 'sample_weight' in kwargs:
                out['sample_weight'] = kwargs['sample_weight']
            return out

        # Inference: Euler integration over num_inference_timesteps.
        b = state.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype
        actions = torch.randn(
            (b, self.action_horizon, self.max_action_dim), device=device, dtype=dtype,
        )
        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_discretized = torch.full(
                (b,), int(t_cont * self.num_timestep_buckets), dtype=torch.long, device=device,
            )
            sa_embs = self._build_dit_inputs(state, actions, embodiment_id, t_discretized)
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=backbone_attention_mask,
                timestep=t_discretized,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity

        return {
            'type': 'trajectory',
            'gt': kwargs.get('action', actions),
            'pred': actions,
            'mask': kwargs.get('action_mask', torch.ones_like(actions)),
        }

    def compute_loss(self, output_dict: Dict[str, Any]) -> torch.Tensor:
        """Masked velocity loss, type chosen by ``head_config.loss_type``.

        - ``l1`` / ``mae``: ``|pred - gt| * mask`` averaged over masked positions.
        - ``l2`` / ``mse``: ``(pred - gt)**2 * mask`` averaged over masked positions.

        ``output_dict`` must contain ``pred``, ``gt``, ``mask`` from ``forward``.
        """
        pred = output_dict['pred']
        gt = output_dict['gt']
        mask = output_dict['mask']
        diff = pred - gt
        loss_type = str(self.config.loss_type).lower()
        if loss_type == 'l1':
            per_elem = diff.abs() * mask
        elif loss_type in ('l2', 'mse'):
            per_elem = diff.pow(2) * mask
        else:
            raise ValueError(f'Unsupported loss_type: {loss_type!r} (expected l1, l2, or mse)')
        return per_elem.sum() / (mask.sum() + 1e-6)
