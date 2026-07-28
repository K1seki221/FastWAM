import torch
from configs.config_schema import RTCHeadConfig

from models.head.flow_matching_action import FlowMatchingAction
from models.head.registry import register_head
from models.utils.vlm_feature_utils import VLMInterface


@register_head
class RealTimeChunkingAction(FlowMatchingAction):
    def __init__(self, head_config: RTCHeadConfig, *, backbone_hidden_dim: int):
        super().__init__(head_config, backbone_hidden_dim=backbone_hidden_dim)

        # RTC-specific attributes
        self.rtc_max_delay = head_config.rtc_max_delay
        # RTC delay sampling config (training only).
        # By default we keep the original behavior: uniform sampling in [0, rtc_max_delay].
        # You may switch to a "gaussian_mixture" (discrete Gaussian + uniform epsilon) to bias the delay distribution.
        self.rtc_delay_sampling = head_config.rtc_delay_sampling
        self.rtc_delay_mu = head_config.rtc_delay_mu
        self.rtc_delay_sigma = head_config.rtc_delay_sigma
        self.rtc_delay_epsilon = head_config.rtc_delay_epsilon
        self.rtc_delay_mu_choices = getattr(head_config, 'rtc_delay_mu_choices', None)
        self.rtc_delay_mu_probs = getattr(head_config, 'rtc_delay_mu_probs', None)
        # Default RTC delay and shift step for inference (can be overridden via set_rtc_params).
        self.delay = 0
        self.shift_step = 1

    # during inference, set the RTC delay and shift step
    def set_rtc_params(self, delay, shift_step):
        self.delay = delay
        self.shift_step = shift_step

    def sample_rtc_delay(self, batch_size: int, max_delay: int, device: torch.device) -> torch.Tensor:
        """
        Sample per-sample RTC delay (integer steps) for training.

        Args:
            batch_size: Batch size.
            max_delay: Maximum delay to sample (inclusive).
            device: Target device.

        Returns:
            torch.LongTensor of shape (B,) with values in [0, max_delay].
        """
        max_delay = int(max_delay)
        if max_delay <= 0:
            return torch.zeros((batch_size,), device=device, dtype=torch.long)

        sampling = str(self.rtc_delay_sampling or 'uniform').lower()
        if sampling == 'uniform':
            return torch.randint(0, max_delay + 1, (batch_size,), device=device, dtype=torch.long)

        # Parameters
        eps = float(self.rtc_delay_epsilon or 0.0)
        eps = max(0.0, min(1.0, eps))
        sigma = float(self.rtc_delay_sigma or 1.0)
        # Guard against invalid sigma (treat as near-delta around mu).
        sigma = max(sigma, 1e-6)

        support = torch.arange(max_delay + 1, device=device, dtype=torch.float32)  # (D,)

        # Sample mu per sample if choices are provided; otherwise use a fixed mu.
        mu_choices = self.rtc_delay_mu_choices
        if isinstance(mu_choices, (list, tuple)) and len(mu_choices) > 0:
            mu_choices_t = torch.tensor(mu_choices, device=device, dtype=torch.float32)
            probs = self.rtc_delay_mu_probs
            if isinstance(probs, (list, tuple)) and len(probs) == len(mu_choices):
                probs_t = torch.tensor(probs, device=device, dtype=torch.float32).clamp(min=0)
                probs_sum = probs_t.sum()
                probs_t = probs_t / probs_sum if probs_sum > 0 else torch.full_like(probs_t, 1.0 / len(probs_t))
            else:
                probs_t = torch.full((len(mu_choices_t),), 1.0 / len(mu_choices_t), device=device)
            mu_idx = torch.multinomial(probs_t, num_samples=batch_size, replacement=True)  # (B,)
            mu = mu_choices_t[mu_idx]  # (B,)
        else:
            mu_val = self.rtc_delay_mu
            mu_val = float(mu_val) if mu_val is not None else float(min(3, max_delay))
            mu = torch.full((batch_size,), mu_val, device=device, dtype=torch.float32)  # (B,)

        mu = mu.clamp(min=0.0, max=float(max_delay))

        # Discrete Gaussian over [0..max_delay], per sample.
        gauss = torch.exp(-((support[None, :] - mu[:, None]) ** 2) / (2 * sigma * sigma))  # (B, D)
        gauss = gauss / gauss.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if eps > 0:
            uniform = torch.full_like(gauss, 1.0 / float(max_delay + 1))
            probs = (1.0 - eps) * gauss + eps * uniform
        else:
            probs = gauss

        # Sample one delay per row.
        return torch.multinomial(probs, num_samples=1).squeeze(1).to(dtype=torch.long)

    def compute_loss(self, output_dict) -> torch.Tensor:
        pred = output_dict['pred']
        gt = output_dict['gt']

        # Base mask (e.g. dataset-provided action_mask)
        mask = output_dict.get('mask', None)
        if mask is None:
            mask = torch.ones_like(gt, dtype=torch.bool, device=pred.device)
        else:
            mask = mask.to(device=pred.device, dtype=torch.bool)
            # Allow mask to be (B, T) or (B, T, D)
            if mask.dim() == 2:
                mask = mask[..., None]

        # RTC: exclude action prefix from the loss (postfix only)
        prefix_mask = output_dict.get('prefix_mask', None)
        if prefix_mask is not None:
            prefix_mask = prefix_mask.to(device=pred.device, dtype=torch.bool)
            # prefix_mask can be (T,) or (B, T)
            if prefix_mask.dim() == 1:
                prefix_mask = prefix_mask[None, :].expand(pred.shape[0], -1)
            postfix_mask = (~prefix_mask)[..., None]  # (B, T, 1)
            mask = mask & postfix_mask

        loss = self.loss_fn(pred, gt)
        loss = loss * mask

        if self.temporal_weighting:
            temporal_weights = torch.linspace(1, 0.5, pred.shape[1]).to(device=pred.device, dtype=pred.dtype)
            loss = loss * temporal_weights[None, :, None].to(pred.device)

        loss = loss.sum()
        mask_sum = mask.sum()

        # Avoid division by zero
        final_loss = loss / mask_sum if mask_sum > 0 else torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        return final_loss

    def forward(self, vlm_output: VLMInterface | None, **kwargs):
        if kwargs['is_training']:
            encoder_tokens, encoder_attention_mask = self.get_tokens(vlm_output, **kwargs)
            actions = kwargs['action']
            noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)

            b, ah, _ = actions.shape

            # Sample per-sample delay and build per-token prefix mask (B, T)
            max_delay = min(int(self.rtc_max_delay), int(ah))
            delay = self.sample_rtc_delay(b, max_delay=max_delay, device=actions.device)  # (B,)
            prefix_mask = torch.arange(ah, device=actions.device)[None, :] < delay[:, None]  # (B, T)

            # Continuous time per batch item: (B,)
            t_scalar = self.sample_time(b, device=actions.device, dtype=actions.dtype).view(-1)
            # Discrete timestep for DiT time embedding: MUST be 1D (B,)
            t_discretized = (t_scalar * self.num_timestep_buckets).long()

            # Token-wise time for mixing (B, T): prefix -> 1.0, postfix -> t_scalar
            t_token = t_scalar[:, None].expand(-1, ah)
            t_token = torch.where(prefix_mask, torch.ones_like(t_token), t_token)
            t_mix = t_token.unsqueeze(-1)  # (B, T, 1)

            noisy_trajectory = (1 - t_mix) * noise + t_mix * actions
            velocity = actions - noise

            # Token-wise timestep for action encoder time embedding (B, T)
            t_token_discretized = t_discretized[:, None].expand(-1, ah)
            prefix_bucket = torch.full_like(t_token_discretized, fill_value=self.num_timestep_buckets)
            t_token_discretized = torch.where(prefix_mask, prefix_bucket, t_token_discretized)

            sa_embs = self.get_input_embeddings(
                kwargs['state'], noisy_trajectory, kwargs['embodiment_id'], t_token_discretized
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
                'prefix_mask': prefix_mask,
            }
            return out_dict

        else:
            encoder_tokens, encoder_attention_mask = self.get_tokens(vlm_output, **kwargs)
            pred_action = self.get_action(
                kwargs['state'],
                encoder_tokens,
                kwargs['embodiment_id'],
                encoder_attention_mask=encoder_attention_mask,
                prev_action=kwargs.get('prev_action', None),
            )
            out_dict = {
                'type': 'trajectory',
                'gt': kwargs['action'],
                'pred': pred_action,
                'mask': kwargs['action_mask'],
            }

            if kwargs.get('return_stage_token', False):
                if vlm_output is None:
                    raise ValueError('return_stage_token requires vlm_output')
                out_dict['stage_token'] = self.get_stage_token(vlm_output, kwargs['input_ids'])

            return out_dict

    @torch.no_grad()
    def get_action(self, state, vlm_emb, embodiment_id, encoder_attention_mask=None, prev_action=None):
        batch_size = vlm_emb.shape[0]
        device = vlm_emb.device
        state_emb = self.state_encoder(state, embodiment_id)
        delay = int(self.delay.item()) if isinstance(self.delay, torch.Tensor) else int(self.delay)
        assert 0 <= delay <= self.action_horizon, (delay, self.action_horizon)

        # If we don't have a previous chunk, fall back to vanilla flow-matching sampling (exactly).
        # This keeps the first frame identical to `FlowMatchingAction.get_action()`.
        if delay == 0 or prev_action is None:
            return super().get_action(
                state,
                vlm_emb,
                embodiment_id,
                encoder_attention_mask=encoder_attention_mask,
            )

        prefix_mask = (torch.arange(self.action_horizon, device=device) < delay)  # (T,)

        actions = torch.randn((batch_size, self.action_horizon, self.action_dim), device=device, dtype=state.dtype)

        prev_action = prev_action.to(device=device, dtype=actions.dtype)
        if prev_action.dim() == 2:
            prev_action = prev_action.unsqueeze(0).expand(batch_size, -1, -1)
        assert prev_action.shape[0] == batch_size, (prev_action.shape, batch_size)
        assert prev_action.shape[1] == self.action_horizon, (prev_action.shape, self.action_horizon)

        # Align previous chunk to the current timestep: shift LEFT by shift_step and zero-pad the tail.
        prev_action = torch.roll(prev_action, -self.shift_step, dims=1)
        prev_action[:, -self.shift_step:, :] = 0.0

        # Keep the prefix in "clean action" space (t=1) throughout sampling to match training.
        actions = torch.where(prefix_mask[None, :, None], prev_action, actions)

        for t in range(self.num_inference_timesteps):
            t_cont = sum(self.t_schedule[:t])
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            # Token-wise timestep for action encoder time embedding (B, T):
            # prefix -> 1.0 (bucket = num_timestep_buckets), postfix -> current timestep bucket
            t_token_discretized = timesteps_tensor[:, None].expand(-1, self.action_horizon)
            prefix_bucket = torch.full_like(t_token_discretized, fill_value=self.num_timestep_buckets)
            t_token_discretized = torch.where(prefix_mask[None, :], prefix_bucket, t_token_discretized)

            sa_embs = self.get_input_embeddings(None, actions, embodiment_id, t_token_discretized, state_emb)

            model_output = self.backbone(
                hidden_states=sa_embs,
                encoder_hidden_states=vlm_emb,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )

            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            postfix_mask = (~prefix_mask)[None, :, None].to(dtype=pred_velocity.dtype)
            actions = actions + self.t_schedule[t] * pred_velocity * postfix_mask
            # Re-lock prefix after every update to avoid drift.
            actions = torch.where(prefix_mask[None, :, None], prev_action, actions)

        return actions

    def get_stage_token(self, vlm_output, input_ids):
        logits = vlm_output.logits
        pred_tokens = [tok[input_ids.shape[1] - 1] for tok in torch.argmax(logits, dim=-1)]
        return pred_tokens
