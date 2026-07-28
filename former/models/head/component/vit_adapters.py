"""Trainable adapters projecting VLM-ViT features into the DiT hidden space.

Both adapters take ``(tokens [B, T, D_in], mask [B, T] | None)`` and return
``(tokens [B, T_out, output_dim], mask [B, T_out] | None)`` so the head can
extend the DiT encoder attention mask with real (padding-aware) validity.
"""

import torch
import torch.nn as nn


class MLPViTAdapter(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear. Token count preserved.

    The leading LayerNorm absorbs the scale difference between raw
    (un-normalized) pre-merger ViT outputs and the DiT projector inputs.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear_fc1 = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.linear_fc2 = nn.Linear(output_dim, output_dim)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.linear_fc2(self.act(self.linear_fc1(self.norm(tokens)))), mask


class MergerViTAdapter(nn.Module):
    """Clone of the Qwen3-VL ViT->LLM ``Qwen3VLVisionPatchMerger`` architecture.

    ``merge_factor`` mirrors ``spatial_merge_size**2``:

    * pre_merger tap  -> ``merge_factor=4``: performs the same 2x2 spatial
      view-merge as the VLM's merger (valid because pre-merger tokens arrive
      in merge-block order: 4 consecutive tokens = one 2x2 spatial block),
      reducing T -> T/4. The mask is merged with ``all`` over each block so a
      merged token is valid only if all 4 constituents are real (the wrapper
      pads per-sample token counts to a multiple of 4 for this tap).
    * post_merger tap -> ``merge_factor=1``: plain LN -> fc1 -> GELU -> fc2
      at unchanged token count (matches the merger MLP shape, no re-merge).
    """

    def __init__(self, input_dim: int, output_dim: int, merge_factor: int = 1):
        super().__init__()
        assert merge_factor >= 1
        self.merge_factor = merge_factor
        merged_dim = input_dim * merge_factor
        # use_postshuffle_norm=True style: LayerNorm on the merged dim.
        self.norm = nn.LayerNorm(merged_dim)
        self.linear_fc1 = nn.Linear(merged_dim, merged_dim)
        self.act = nn.GELU()
        self.linear_fc2 = nn.Linear(merged_dim, output_dim)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.merge_factor > 1:
            batch, seq_len, dim = tokens.shape
            assert seq_len % self.merge_factor == 0, (
                f'pre_merger token count {seq_len} not divisible by merge_factor '
                f'{self.merge_factor}; the wrapper must pad to a multiple of it'
            )
            tokens = tokens.reshape(batch, seq_len // self.merge_factor, dim * self.merge_factor)
            if mask is not None:
                mask = mask.reshape(batch, seq_len // self.merge_factor, self.merge_factor).all(dim=-1)
        return self.linear_fc2(self.act(self.linear_fc1(self.norm(tokens)))), mask


def build_vit_adapter(adapter_type: str, tap: str, *, vit_hidden_size: int,
                      vit_out_hidden_size: int, spatial_merge_size: int, output_dim: int) -> nn.Module:
    """Size and build one adapter from the resolved ViTConditionConfig dims."""
    if tap == 'pre_merger':
        input_dim = vit_hidden_size
        merge_factor = spatial_merge_size ** 2
    else:  # post_merger
        input_dim = vit_out_hidden_size
        merge_factor = 1

    if adapter_type == 'mlp':
        return MLPViTAdapter(input_dim=input_dim, output_dim=output_dim)
    if adapter_type == 'merger':
        return MergerViTAdapter(input_dim=input_dim, output_dim=output_dim, merge_factor=merge_factor)
    raise ValueError(f'Unknown vit_condition adapter: {adapter_type}')
