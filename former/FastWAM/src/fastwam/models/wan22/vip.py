from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class VIPSpatialPool(nn.Module):
    """Parameter-efficient convolutional pooling for spatial DiT tokens."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 256):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError(
                "VIP pool dimensions must be positive, "
                f"got input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}."
            )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=self.hidden_dim,
                bias=False,
            ),
            nn.GroupNorm(1, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.output_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must be [B,S,D], got shape {tuple(tokens.shape)}.")
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"VIP pool input dim mismatch: expected {self.input_dim}, got {tokens.shape[-1]}."
            )
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        if grid_h <= 0 or grid_w <= 0 or tokens.shape[1] != grid_h * grid_w:
            raise ValueError(
                "VIP spatial grid does not match token count: "
                f"tokens={tokens.shape[1]}, grid={grid_h}x{grid_w}."
            )
        feature_map = tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], grid_h, grid_w
        )
        conv_dtype = next(self.net.parameters()).dtype
        return self.net(feature_map.to(dtype=conv_dtype))


def sample_vip_frame_indices(
    *,
    batch_size: int,
    num_frames: int,
    image_is_pad: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-clip ``(o_0, o_g, o_t, o_{t+1})`` frame indices."""

    if batch_size <= 0 or num_frames <= 1:
        raise ValueError(
            f"VIP sampling requires batch_size > 0 and num_frames > 1, got {batch_size}, {num_frames}."
        )
    if image_is_pad is None:
        valid_count = torch.full((batch_size,), num_frames, dtype=torch.long, device=device)
        first_frame_valid = torch.ones((batch_size,), dtype=torch.bool, device=device)
    else:
        if image_is_pad.shape != (batch_size, num_frames):
            raise ValueError(
                "`image_is_pad` shape mismatch for VIP sampling: "
                f"expected {(batch_size, num_frames)}, got {tuple(image_is_pad.shape)}."
            )
        image_is_pad = image_is_pad.to(device=device, dtype=torch.bool)
        valid_count = (~image_is_pad).sum(dim=1)
        first_frame_valid = ~image_is_pad[:, 0]

    valid_sample = first_frame_valid & (valid_count >= 2)
    max_goal_index = (valid_count - 1).clamp(min=1, max=num_frames - 1)
    goal_index = 1 + torch.floor(
        torch.rand((batch_size,), device=device) * max_goal_index.to(dtype=torch.float32)
    ).to(dtype=torch.long)
    transition_index = torch.floor(
        torch.rand((batch_size,), device=device) * goal_index.to(dtype=torch.float32)
    ).to(dtype=torch.long)
    next_index = transition_index + 1
    initial_index = torch.zeros_like(goal_index)
    indices = torch.stack([initial_index, goal_index, transition_index, next_index], dim=1)
    return indices, valid_sample


def negative_l2_value(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    if state.shape != goal.shape:
        raise ValueError(f"VIP state/goal shape mismatch: {tuple(state.shape)} vs {tuple(goal.shape)}.")
    return -torch.linalg.vector_norm(state - goal, ord=2, dim=-1)


def sample_same_instruction_negative_indices(
    instruction_ids: torch.Tensor,
    *,
    num_negatives: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pair each eligible goal with a different sample sharing its instruction."""

    if not isinstance(instruction_ids, torch.Tensor):
        raise TypeError(
            f"`instruction_ids` must be a tensor, got {type(instruction_ids).__name__}."
        )
    if instruction_ids.ndim != 1:
        raise ValueError(
            f"`instruction_ids` must be 1D [B], got shape {tuple(instruction_ids.shape)}."
        )
    if num_negatives < 0:
        raise ValueError(f"`num_negatives` must be non-negative, got {num_negatives}.")

    eligible = torch.zeros_like(instruction_ids, dtype=torch.bool)
    target_indices = []
    source_indices = []
    if num_negatives == 0:
        empty = torch.empty((0,), dtype=torch.long, device=instruction_ids.device)
        return empty, empty, eligible

    for instruction_id in torch.unique(instruction_ids):
        group_indices = torch.nonzero(
            instruction_ids == instruction_id,
            as_tuple=False,
        ).flatten()
        if group_indices.numel() < 2:
            continue
        eligible[group_indices] = True
        for _ in range(int(num_negatives)):
            # A random cycle is a derangement: source and target always use
            # the same instruction, while no sample is paired with itself.
            shuffled = group_indices[
                torch.randperm(group_indices.numel(), device=instruction_ids.device)
            ]
            target_indices.append(shuffled)
            source_indices.append(torch.roll(shuffled, shifts=1, dims=0))

    if not target_indices:
        empty = torch.empty((0,), dtype=torch.long, device=instruction_ids.device)
        return empty, empty, eligible
    return torch.cat(target_indices), torch.cat(source_indices), eligible


def compute_vip_loss(
    features: torch.Tensor,
    valid_sample: torch.Tensor,
    *,
    gamma: float = 0.98,
    num_negatives: int = 0,
    instruction_ids: Optional[torch.Tensor] = None,
    same_instruction_negatives: bool = False,
    l1_weight: float = 0.0,
    l2_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the VIP objective with optional instruction-matched extra negatives."""

    if features.ndim != 3 or features.shape[1] != 4:
        raise ValueError(f"`features` must be [B,4,D], got shape {tuple(features.shape)}.")
    if valid_sample.shape != (features.shape[0],):
        raise ValueError(
            f"`valid_sample` must be [B], got {tuple(valid_sample.shape)} for B={features.shape[0]}."
        )
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"VIP gamma must be in (0,1), got {gamma}.")
    if num_negatives < 0 or l1_weight < 0.0 or l2_weight < 0.0:
        raise ValueError(
            "VIP num_negatives and regularization weights must be non-negative, "
            f"got {num_negatives}, {l1_weight}, {l2_weight}."
        )
    if same_instruction_negatives:
        if instruction_ids is None:
            raise ValueError(
                "`instruction_ids` is required when `same_instruction_negatives=True`."
            )
        if not isinstance(instruction_ids, torch.Tensor):
            raise TypeError(
                f"`instruction_ids` must be a tensor, got {type(instruction_ids).__name__}."
            )
        if instruction_ids.shape != (features.shape[0],):
            raise ValueError(
                "`instruction_ids` must be [B] and align with `features`, "
                f"got {tuple(instruction_ids.shape)} for B={features.shape[0]}."
            )

    valid_sample = valid_sample.to(device=features.device, dtype=torch.bool)
    valid_features = features[valid_sample].float()
    if valid_features.shape[0] == 0:
        zero = features.float().sum() * 0.0
        return zero, {
            "value": zero.detach(),
            "negative": zero.detach(),
            "negative_valid_fraction": zero.detach(),
            "l1": zero.detach(),
            "l2": zero.detach(),
            "valid_fraction": valid_sample.float().mean().detach(),
        }

    initial, goal, state, next_state = valid_features.unbind(dim=1)
    value_initial = negative_l2_value(initial, goal)
    value_state = negative_l2_value(state, goal)
    value_next = negative_l2_value(next_state, goal)
    reward = -torch.ones_like(value_state)
    bellman_residual = reward + float(gamma) * value_next - value_state
    positive_lme = torch.logsumexp(-bellman_residual, dim=0) - math.log(bellman_residual.numel())
    value_loss = (1.0 - float(gamma)) * (-value_initial.mean()) + positive_lme

    negative_loss = value_loss.new_zeros(())
    negative_valid_fraction = value_loss.new_zeros(())
    if num_negatives > 0 and valid_features.shape[0] > 1:
        if same_instruction_negatives:
            valid_instruction_ids = instruction_ids.to(
                device=features.device,
                dtype=torch.long,
            )[valid_sample]
            target_indices, source_indices, negative_valid = (
                sample_same_instruction_negative_indices(
                    valid_instruction_ids,
                    num_negatives=int(num_negatives),
                )
            )
            negative_valid_fraction = negative_valid.float().mean()
            if target_indices.numel() > 0:
                value_state_negative = negative_l2_value(
                    state[source_indices],
                    goal[target_indices],
                )
                value_next_negative = negative_l2_value(
                    next_state[source_indices],
                    goal[target_indices],
                )
                negative_residual = (
                    -torch.ones_like(value_state_negative)
                    + float(gamma) * value_next_negative
                    - value_state_negative
                )
                negative_loss = torch.logsumexp(-negative_residual, dim=0) - math.log(
                    negative_residual.numel()
                )
        else:
            negative_valid_fraction = value_loss.new_ones(())
            negative_residuals = []
            for _ in range(int(num_negatives)):
                permutation = torch.randperm(valid_features.shape[0], device=features.device)
                value_state_negative = negative_l2_value(state[permutation], goal)
                value_next_negative = negative_l2_value(next_state[permutation], goal)
                reward_negative = -torch.ones_like(value_state_negative)
                negative_residuals.append(
                    reward_negative + float(gamma) * value_next_negative - value_state_negative
                )
            negative_residual = torch.cat(negative_residuals, dim=0)
            negative_loss = torch.logsumexp(-negative_residual, dim=0) - math.log(
                negative_residual.numel()
            )

    l1_norm = torch.linalg.vector_norm(valid_features, ord=1, dim=-1).mean()
    l2_norm = torch.linalg.vector_norm(valid_features, ord=2, dim=-1).mean()
    total = value_loss + negative_loss + float(l1_weight) * l1_norm + float(l2_weight) * l2_norm
    return total, {
        "value": value_loss.detach(),
        "negative": negative_loss.detach(),
        "negative_valid_fraction": negative_valid_fraction.detach(),
        "l1": l1_norm.detach(),
        "l2": l2_norm.detach(),
        "valid_fraction": valid_sample.float().mean().detach(),
    }
