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
"""GR00T-N1.7 head support modules unique to this head.

The per-embodiment building blocks (CategorySpecificLinear / CategorySpecificMLP /
MultiEmbodimentActionEncoder / SinusoidalPositionalEncoding) already exist in
``models.common.moe`` — the head re-uses those directly. Only the 4-layer
``vl_self_attention`` stack is unique to GR00T-N1.7 and lives here.
"""
from typing import Optional

import torch
from torch import nn

from models.head.transformer.blocks import SeparateTransformerBlock


class SelfAttentionStack(nn.Module):
    """Stack of self-attention transformer blocks used as GR00T-N1.7's ``vl_self_attention``.

    Wraps iron_vla's ``SeparateTransformerBlock`` in self-attention mode with vanilla
    LayerNorm (norm_type='layer_norm', norm_elementwise_affine=True). Sub-attribute name
    ``transformer_blocks`` matches N1.7's nested ``vl_self_attention.transformer_blocks.<i>.*``.

    Block-level sub-names differ from upstream BasicTransformerBlock (iron_vla uses
    ``norm_attn``/``norm_ffn``/``attn`` while upstream uses ``norm1``/``norm3``/``attn1``);
    the GR00T → iron_vla converter handles those renames.
    """

    def __init__(
        self,
        num_layers: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        activation_fn: str = 'gelu-approximate',
        attention_bias: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.transformer_blocks = nn.ModuleList(
            [
                SeparateTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    attention_mode='self',
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_type='layer_norm',
                    norm_elementwise_affine=True,
                    norm_eps=norm_eps,
                    final_dropout=final_dropout,
                    ff_inner_dim=ff_inner_dim,
                    ff_bias=True,
                    attention_out_bias=True,
                    has_ffn=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states``: [B, S, inner_dim]. Returns [B, S, inner_dim]."""
        hidden_states = hidden_states.contiguous()
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states
