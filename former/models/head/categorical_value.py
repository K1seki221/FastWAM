from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config_schema import CategoricalValueHeadConfig

from models.common.moe import CategorySpecificMLP
from models.common.projector import PlainProjector
from models.head.base_head import BaseHead
from models.head.registry import register_head
from models.utils.vlm_feature_utils import VLMInterface


class LightCrossAttentionPooling(nn.Module):
    """Perceiver-style lightweight cross-attention aggregator.

    Uses a small number of learnable query tokens to cross-attend to
    a variable-length conditioning sequence, producing a fixed-size
    representation.

    Architecture per layer:
        1. Cross-attention: queries attend to encoder_hidden_states (VLM tokens)
        2. Self-attention: queries attend to themselves
        3. Feed-forward: MLP with GELU activation
    All with pre-norm residual connections.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        # Cross-attention: queries -> encoder tokens
                        'cross_norm_q': nn.LayerNorm(hidden_dim),
                        'cross_norm_kv': nn.LayerNorm(hidden_dim),
                        'cross_attn': nn.MultiheadAttention(
                            embed_dim=hidden_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        # Self-attention among queries
                        'self_norm': nn.LayerNorm(hidden_dim),
                        'self_attn': nn.MultiheadAttention(
                            embed_dim=hidden_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        # Feed-forward
                        'ff_norm': nn.LayerNorm(hidden_dim),
                        'ff': nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * ff_mult),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim * ff_mult, hidden_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate variable-length sequence into fixed-size representation.

        Args:
            encoder_hidden_states: (B, S, D) VLM token embeddings.
            encoder_attention_mask: (B, S) where 1=attend, 0=pad.

        Returns:
            (B, num_queries, D) aggregated representation.
        """
        B = encoder_hidden_states.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)  # (B, Q, D)

        # nn.MultiheadAttention key_padding_mask: True = ignore
        key_padding_mask = None
        if encoder_attention_mask is not None:
            key_padding_mask = ~encoder_attention_mask.bool()

        for layer in self.layers:
            # Cross-attention (pre-norm)
            q_normed = layer['cross_norm_q'](queries)
            kv_normed = layer['cross_norm_kv'](encoder_hidden_states)
            cross_out, _ = layer['cross_attn'](
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,
            )
            queries = queries + cross_out

            # Self-attention (pre-norm)
            q_normed = layer['self_norm'](queries)
            self_out, _ = layer['self_attn'](
                query=q_normed,
                key=q_normed,
                value=q_normed,
            )
            queries = queries + self_out

            # Feed-forward (pre-norm)
            queries = queries + layer['ff'](layer['ff_norm'](queries))

        return self.final_norm(queries)  # (B, Q, D)


@register_head
class CategoricalValueHead(BaseHead):
    """Distributional RL value head (C51-style).

    Takes the VLM backbone's last_hidden_state, aggregates the variable-length
    sequence via light cross-attention pooling into a fixed representation,
    conditions on robot state, and outputs a categorical distribution over
    value atoms.

    During training: outputs logits over atoms; loss is cross-entropy between
        predicted distribution and the projected Bellman target.
    During inference: returns expected value (weighted sum of atoms by softmax
        probabilities) and the full distribution.
    """

    def __init__(self, head_config: CategoricalValueHeadConfig, *, backbone_hidden_dim: int):
        super().__init__(head_config)
        self.backbone_hidden_dim = backbone_hidden_dim

        # C51 categorical distribution parameters
        self.num_atoms = head_config.num_atoms
        self.v_min = head_config.v_min
        self.v_max = head_config.v_max

        support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.register_buffer('support', support)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Light cross-attention pooling config
        self.num_queries = getattr(head_config, 'num_queries', 4)
        self.num_pool_layers = getattr(head_config, 'num_pool_layers', 2)
        self.num_heads = getattr(head_config, 'num_heads', 8)
        self.dropout = getattr(head_config, 'dropout', 0.1)

        self._init_model()
        self.get_loss_fn(getattr(head_config, 'loss_type', 'categorical_cross_entropy'))

    def _init_model(self):
        # Project backbone hidden dim to head hidden dim
        self.reduce_dim_proj = PlainProjector(input_dim=self.backbone_hidden_dim, output_dim=self.hidden_dim)

        # Light cross-attention pooling
        self.pooling = LightCrossAttentionPooling(
            hidden_dim=self.hidden_dim,
            num_queries=self.num_queries,
            num_heads=self.num_heads,
            num_layers=self.num_pool_layers,
            dropout=self.dropout,
        )

        # State conditioning (always enabled for V(s))
        self.state_encoder = CategorySpecificMLP(
            num_categories=self.num_categories,
            input_dim=self.state_dim,
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
        )

        # Aggregation: flatten pooled queries + state token -> project
        pool_output_dim = self.hidden_dim * self.num_queries + self.hidden_dim
        self.aggregate_mlp = nn.Sequential(
            nn.Linear(pool_output_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # Output projection to atom logits
        self.value_head = nn.Linear(self.hidden_dim, self.num_atoms)

    def get_loss_fn(self, loss_type: str):
        self.loss_weight = self.config.loss_weight
        if loss_type not in ('categorical_cross_entropy',):
            raise ValueError(f'Loss type {loss_type} not supported for CategoricalValueHead')

    def get_vlm_tokens(self, vlm_output):
        """Project VLM hidden states from backbone dim to head dim."""
        vlm_tokens = vlm_output.last_hidden_state  # (B, S, backbone_hidden_dim)
        vlm_tokens = self.reduce_dim_proj(vlm_tokens)  # (B, S, hidden_dim)
        return vlm_tokens

    def forward(self, vlm_output: VLMInterface | None, **kwargs) -> Dict[str, Any]:
        if vlm_output is None:
            raise ValueError('CategoricalValueHead requires vlm_output')
        # 1. Get VLM tokens (full sequence)
        vlm_tokens = self.get_vlm_tokens(vlm_output)

        # 2. Get attention mask if available
        attention_mask = kwargs.get('attention_mask', None)

        # 3. Cross-attention pooling: (B, S, D) -> (B, Q, D)
        pooled = self.pooling(vlm_tokens, attention_mask)

        # 4. Flatten pooled queries: (B, Q*D)
        B = pooled.shape[0]
        pooled_flat = pooled.reshape(B, -1)

        # 5. State conditioning
        state = kwargs['state']
        embodiment_id = kwargs['embodiment_id']
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (B, 1, state_dim)
        state_emb = self.state_encoder(state, embodiment_id)  # (B, 1, hidden_dim)
        state_emb = state_emb.squeeze(1)  # (B, hidden_dim)
        pooled_flat = torch.cat([pooled_flat, state_emb], dim=-1)

        # 6. Aggregate and predict atom logits
        features = self.aggregate_mlp(pooled_flat)  # (B, hidden_dim)
        atom_logits = self.value_head(features)  # (B, num_atoms)

        # 7. Compute probabilities and expected value
        probs = F.softmax(atom_logits, dim=-1)
        expected_value = (probs * self.support).sum(dim=-1)  # (B,)

        if kwargs.get('is_training', False):
            return {
                'type': 'value',
                'atom_logits': atom_logits,
                'probs': probs,
                'expected_value': expected_value,
                'value_target': kwargs['value_target'],
            }
        else:
            return {
                'type': 'value',
                'expected_value': expected_value,
                'probs': probs,
                'atom_logits': atom_logits,
            }

    def _project_target_distribution(self, target_values: torch.Tensor) -> torch.Tensor:
        """Project scalar target values onto the categorical support (C51).

        Each scalar target is distributed across the two nearest atoms
        proportionally to its distance from each.

        Args:
            target_values: (B,) scalar target values.

        Returns:
            target_dist: (B, num_atoms) target probability distribution.
        """
        target_values = target_values.clamp(self.v_min, self.v_max)

        # Position of each target on the support: b in [0, num_atoms-1]
        b = (target_values - self.v_min) / self.delta_z

        lower = b.floor().long().clamp(0, self.num_atoms - 1)
        upper = b.ceil().long().clamp(0, self.num_atoms - 1)

        target_dist = torch.zeros(
            target_values.shape[0],
            self.num_atoms,
            device=target_values.device,
            dtype=target_values.dtype,
        )

        upper_weight = b - lower.float()
        lower_weight = 1.0 - upper_weight

        target_dist.scatter_add_(1, lower.unsqueeze(1), lower_weight.unsqueeze(1))
        target_dist.scatter_add_(1, upper.unsqueeze(1), upper_weight.unsqueeze(1))

        return target_dist

    def compute_loss(self, output_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Categorical cross-entropy between predicted and projected target distributions.

        Returns a dict with:
          - loss: scalar cross-entropy loss (for backprop)
          - mean_pred_value: mean predicted expected value (for monitoring)
          - mean_gt_value: mean ground-truth value target (for monitoring)
        """
        atom_logits = output_dict['atom_logits'].float()  # (B, num_atoms), upcast for stability
        target_values = output_dict['value_target']  # (B,)

        # Project scalar targets onto categorical support
        target_dist = self._project_target_distribution(target_values)

        # Cross-entropy: -sum(target_dist * log_softmax(logits))
        log_probs = F.log_softmax(atom_logits, dim=-1)
        loss = -(target_dist * log_probs).sum(dim=-1)  # (B,)

        return {
            'loss': loss.mean(),
            'mean_pred_value': output_dict['expected_value'].detach().mean(),
            'mean_gt_value': target_values.detach().mean(),
        }
