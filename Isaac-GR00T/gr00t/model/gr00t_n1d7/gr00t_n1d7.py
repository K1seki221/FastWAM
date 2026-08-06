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

import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


logger = logging.getLogger(__name__)


class ConditionRouter(nn.Module):
    """Learned per-cross-attn-block mixture over candidate VLM layers (v1, static).

    Replaces the hardcoded single tap (final selected layer -> all cross blocks)
    with cond_b = sum_k softmax(logits)[b, k] * norm_k(h_k). Identity-init puts
    `init_bias` logits on the incumbent (deepest) candidate so training starts
    close to the stock wiring.
    """

    def __init__(self, num_cross_blocks: int, num_candidates: int, dim: int, init_bias: float, init_mode: str = "last", frozen: bool = False, candidate_proj: bool = False, mix_renorm: bool = False, gate_mode: str = "softmax", gate_init_hi: float = 0.9, gate_init_lo: float = 0.1):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_candidates)])
        # Optional per-candidate linear adapters (identity-init): each VLM depth
        # gets its own alignment into the conditioning space before mixing; the
        # blocks' shared to_k/to_v then project the mixture (factorized
        # per-candidate K/V projection, block-shared). Without these, moving the
        # mix across the linear to_k/to_v is a mathematical no-op.
        self.projs = (
            nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_candidates)])
            if candidate_proj
            else None
        )
        if self.projs is not None:
            for lin in self.projs:
                nn.init.eye_(lin.weight)
                nn.init.zeros_(lin.bias)
        self.init_bias = init_bias
        self.init_mode = init_mode
        # Rescale the mixture by 1/sqrt(sum w^2): a convex mix of unit-RMS,
        # decorrelated candidates has RMS ~ sqrt(sum w^2), so without this the
        # conditioning magnitude depends on routing entropy (one-hot 1.0,
        # uniform K=4 ~0.5, K=28 ~0.19). Deterministic, exactly 1 at one-hot
        # => preserves exact stock identity at hard init.
        self.mix_renorm = mix_renorm
        assert gate_mode in ("softmax", "sigmoid"), gate_mode
        self.gate_mode = gate_mode
        if gate_mode == "sigmoid":
            # Independent accumulation gates: favored candidate opens at
            # gate_init_hi, the rest at gate_init_lo. Both off the saturation
            # rails (grad factor g(1-g) stays alive; zero logits would mean
            # 0.5 gates, NOT a neutral start).
            def _logit(p):
                return float(torch.logit(torch.tensor(p)))
            logits = torch.full((num_cross_blocks, num_candidates), _logit(gate_init_lo))
            if init_mode == "span":
                for b in range(num_cross_blocks):
                    logits[b, b * num_candidates // num_cross_blocks] = _logit(gate_init_hi)
            else:
                logits[:, -1] = _logit(gate_init_hi)
        else:
            logits = torch.zeros(num_cross_blocks, num_candidates)
            if init_mode == "span":
                # iron_vla depth-aligned identity: block-span i -> candidate i
                for b in range(num_cross_blocks):
                    logits[b, b * num_candidates // num_cross_blocks] = init_bias
            else:
                logits[:, -1] = init_bias  # incumbent = deepest candidate (stock tap)
        self.logits = nn.Parameter(logits)
        if frozen:
            self.logits.requires_grad_(False)

    def forward(self, features_all: torch.Tensor) -> torch.Tensor:
        """features_all: [B, K, S, D] -> routed conditions [B, N_cross, S, D]."""
        xs = [norm(features_all[:, k]) for k, norm in enumerate(self.norms)]
        if self.projs is not None:
            xs = [proj(x) for proj, x in zip(self.projs, xs)]
        normed = torch.stack(xs, dim=1)  # [B, K, S, D]
        if self.gate_mode == "sigmoid":
            weights = torch.sigmoid(self.logits).to(normed.dtype)  # [N, K], independent
        else:
            weights = self.logits.softmax(dim=-1).to(normed.dtype)  # [N, K]
        mixed = torch.einsum("nk,bksd->bnsd", weights, normed)
        if self.mix_renorm:
            scale = weights.pow(2).sum(dim=-1).clamp_min(1e-9).rsqrt()  # [N]
            mixed = mixed * scale.view(1, -1, 1, 1)
        # Actual conditioning loudness (post-rescale if enabled): lets curves
        # separate "which layers" from "how loud" (mix of unit-RMS candidates
        # shrinks toward sqrt(sum w^2) as entropy rises).
        with torch.no_grad():
            self._last_mix_rms = mixed.detach().float().pow(2).mean(dim=-1).sqrt().mean()
        return mixed

    @torch.no_grad()
    def mixture_stats(self) -> dict[str, torch.Tensor]:
        if self.gate_mode == "sigmoid":
            w = torch.sigmoid(self.logits)  # [N, K] gates
            # Bernoulli entropy per gate, summed over candidates: -> 0 as
            # gates saturate (dead-gate / commitment signal).
            entropy = -(
                w * (w + 1e-9).log() + (1 - w) * (1 - w + 1e-9).log()
            ).sum(dim=-1)
            extra = {
                # learned "conditioning budget" per block (softmax pins it to 1)
                "router_gate_sum": w.sum(dim=-1).mean().detach(),
                "router_gate_min": w.min().detach(),
            }
        else:
            w = self.logits.softmax(dim=-1)  # [N, K]
            entropy = -(w * (w + 1e-9).log()).sum(dim=-1)  # [N]
            extra = {}
        return {
            "router_weights": w.detach(),
            "router_entropy": entropy.mean().detach(),
            # sqrt(sum w^2): the decorrelated-candidates prediction for the
            # mixture RMS (1 at one-hot, 1/sqrt(K) at uniform).
            "router_sqrt_sum_w2": w.pow(2).sum(dim=-1).sqrt().mean().detach(),
            "router_mix_rms": getattr(self, "_last_mix_rms", None),
            **extra,
        }


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        if getattr(config, "use_condition_router", False):
            assert config.use_alternate_vl_dit, "condition router requires AlternateVLDiT"
            assert config.use_vlln, "condition router expects use_vlln=True (norm init source)"
            candidates = config.router_candidate_layers
            if candidates is None:
                candidates = list(range(config.select_layer + 1))
            num_cross_blocks = (config.diffusion_model_cfg["num_layers"] + 1) // 2
            self.condition_router = ConditionRouter(
                num_cross_blocks=num_cross_blocks,
                num_candidates=len(candidates),
                dim=config.backbone_embedding_dim,
                init_bias=config.router_init_bias,
                init_mode=getattr(config, "router_init_mode", "last"),
                frozen=getattr(config, "router_frozen", False),
                candidate_proj=getattr(config, "router_candidate_proj", False),
                mix_renorm=getattr(config, "router_mix_renorm", False),
                gate_mode=getattr(config, "router_gate_mode", "softmax"),
                gate_init_hi=getattr(config, "router_gate_init_hi", 0.9),
                gate_init_lo=getattr(config, "router_gate_init_lo", 0.1),
            )
            if (
                getattr(config, "router_gate_mode", "softmax") == "sigmoid"
                and float(getattr(config, "router_entropy_coef", 0.0) or 0.0) > 0.0
            ):
                raise ValueError(
                    "router_entropy_coef assumes softmax-distribution entropy; "
                    "incompatible with router_gate_mode='sigmoid'"
                )

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly. The action head can
        # be instantiated under a meta / no_init_weights default-device context
        # (e.g. nested from_pretrained). A Beta built from bare Python floats
        # would then place its concentration tensors on the meta device (or in
        # the active default dtype, e.g. bf16). With validate_args enabled that
        # already fails here in __init__ (Beta's internal .item() check cannot
        # run on meta); even with validation off, sample_time would later raise
        # or return garbage. Explicit device/dtype here makes the sampler depend
        # only on the config, not on the construction-time device/dtype context,
        # so the noise schedule is identical across SDPA/FA2/FA4 and meta vs.
        # real-device loads. config is the canonical source for these values.
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        # Fixed-mapping arm: the blanket requires_grad=True above would undo
        # the ctor freeze, so re-apply it here (norms stay trainable).
        if getattr(self, "condition_router", None) is not None and getattr(
            self.config, "router_frozen", False
        ):
            self.condition_router.logits.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        if hasattr(self, "condition_router") and "backbone_features_all" in backbone_output:
            # Routed path: per-cross-block conditions [B, N_cross, S, D]; the
            # per-candidate norms replace vlln (incumbent norm is vlln-initialized).
            backbone_output["backbone_features"] = self.condition_router(
                backbone_output["backbone_features_all"]
            )
            return backbone_output
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def init_condition_router_from_vlln(self):
        """Identity-init helper: copy trained vlln affine into the incumbent
        (deepest) candidate norm so run start matches the stock wiring.

        Also (re-)applies the logits bias: the ctor sets it, but HF
        from_pretrained re-initializes missing-key params afterwards, wiping
        the ctor state — so the full identity init must happen here, at the
        point setup.py detects the router keys missing from the checkpoint."""
        if hasattr(self, "condition_router"):
            with torch.no_grad():
                self.condition_router.logits.zero_()
                if getattr(self.condition_router, "init_mode", "last") == "span":
                    B, K = self.condition_router.logits.shape
                    for b in range(B):
                        self.condition_router.logits[b, b * K // B] = self.condition_router.init_bias
                else:
                    self.condition_router.logits[:, -1] = self.condition_router.init_bias
                for norm in self.condition_router.norms:
                    norm.weight.fill_(1.0)
                    norm.bias.zero_()
                if self.condition_router.projs is not None:
                    for lin in self.condition_router.projs:
                        lin.weight.copy_(torch.eye(lin.weight.shape[0]))
                        lin.bias.zero_()
                if isinstance(self.vlln, nn.LayerNorm):
                    self.condition_router.norms[-1].weight.copy_(self.vlln.weight)
                    self.condition_router.norms[-1].bias.copy_(self.vlln.bias)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        out = {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
        if hasattr(self, "condition_router"):
            out.update(self.condition_router.mixture_stats())
        return out

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    # Substring (not org/repo prefix) so local snapshot paths like
    # .../models--nvidia--Cosmos-Reason2-2B/snapshots/<hash> also match.
    if "Cosmos-Reason2" in config.model_name or "Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        backbone_kwargs = {}
        if getattr(config, "use_condition_router", False):
            candidates = config.router_candidate_layers
            if candidates is None:
                candidates = list(range(config.select_layer + 1))
            backbone_kwargs["extra_hidden_layer_indices"] = candidates
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
            **backbone_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
