from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class VIPDPTHead(nn.Module):
    """Lightweight multi-layer VideoDiT feature aggregation for VIP."""

    DEFAULT_LAYER_NUMBERS = (6, 12, 18, 24, 30)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 256,
        layer_numbers: Sequence[int] = DEFAULT_LAYER_NUMBERS,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError(
                "VIP DPT dimensions must be positive, "
                f"got input_dim={input_dim}, hidden_dim={hidden_dim}, "
                f"output_dim={output_dim}."
            )
        normalized_layers = tuple(int(layer_number) for layer_number in layer_numbers)
        if not normalized_layers:
            raise ValueError("VIP DPT requires at least one VideoDiT layer.")
        if len(set(normalized_layers)) != len(normalized_layers):
            raise ValueError(
                f"VIP DPT layer numbers must be unique, got {normalized_layers}."
            )
        if any(layer_number <= 0 for layer_number in normalized_layers):
            raise ValueError(
                "VIP DPT layer numbers are one-based and must be positive, "
                f"got {normalized_layers}."
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.layer_numbers = normalized_layers

        self.layer_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.input_dim, eps=1.0e-6),
                    nn.Linear(self.input_dim, self.hidden_dim, bias=False),
                )
                for _ in self.layer_numbers
            ]
        )
        fused_dim = len(self.layer_numbers) * self.hidden_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_dim, self.hidden_dim, kernel_size=1, bias=False),
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

    def forward(
        self,
        layer_tokens: Mapping[int, torch.Tensor],
        grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        missing_layers = [
            layer_number
            for layer_number in self.layer_numbers
            if layer_number not in layer_tokens
        ]
        if missing_layers:
            raise ValueError(
                f"VIP DPT is missing VideoDiT layer outputs: {missing_layers}."
            )

        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"VIP DPT grid must be positive, got {grid_h}x{grid_w}.")

        feature_maps = []
        expected_batch = None
        expected_tokens = grid_h * grid_w
        projection_dtype = next(self.layer_projections.parameters()).dtype
        for layer_number, projection in zip(
            self.layer_numbers,
            self.layer_projections,
        ):
            tokens = layer_tokens[layer_number]
            if tokens.ndim != 3:
                raise ValueError(
                    "VIP DPT layer tokens must be [B,S,D], "
                    f"got layer {layer_number}: {tuple(tokens.shape)}."
                )
            if tokens.shape[1] != expected_tokens or tokens.shape[2] != self.input_dim:
                raise ValueError(
                    f"VIP DPT layer {layer_number} shape mismatch: "
                    f"expected [B,{expected_tokens},{self.input_dim}], "
                    f"got {tuple(tokens.shape)}."
                )
            if expected_batch is None:
                expected_batch = int(tokens.shape[0])
            elif tokens.shape[0] != expected_batch:
                raise ValueError(
                    "VIP DPT layer batch sizes must match, "
                    f"expected {expected_batch}, got {tokens.shape[0]} "
                    f"at layer {layer_number}."
                )

            projected = projection(tokens.to(dtype=projection_dtype))
            feature_maps.append(
                projected.transpose(1, 2).reshape(
                    tokens.shape[0],
                    self.hidden_dim,
                    grid_h,
                    grid_w,
                )
            )

        return self.fusion(torch.cat(feature_maps, dim=1))
