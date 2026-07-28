"""
Utility functions for VLM multi-layer feature extraction.

This module provides helper functions to simplify multi-layer cross-attention logic
across Policy, Backbone, and Head components.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class KVCache(Protocol):
    """Protocol for KV cache objects that support len() and indexing."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass
class VLMInterface:
    """Structured bundle passed from VLM backbone to heads (replaces ad-hoc model outputs)."""

    last_hidden_state: torch.Tensor
    logits: torch.Tensor | None = None
    stage_token: torch.Tensor | None = None      # (B,)
    hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None
    past_key_values: Any | None = None
    layer_features: dict[int, torch.Tensor] | None = None
    layer_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    labels: Any | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    metadata: dict[str, Any] | None = None
    # [B, S_vlm] bool — True at positions occupied by image placeholder tokens
    # (set by VLM backbones that route visual embeddings into the text sequence
    # via ``masked_scatter``). Heads that drive image-vs-text attention routing
    # (e.g. FlowMatchingActionGr00tN1d7's AlternateVL DiT) consume this. Stays
    # None when the input has no visual tokens or when the backbone does not
    # surface it; consumers must handle the None branch.
    image_mask: torch.Tensor | None = None
    # ViT features for head-side vit_condition, re-batched to [B, T_vis, D]:
    # keys 'ds0'/'ds1'/'ds2' (DeepStack taps, ascending ViT depth) and
    # 'final' (last ViT layer). None unless the backbone was configured via
    # configure_vit_condition().
    vit_features: dict[str, torch.Tensor] | None = None
    # [B, T_vis] bool — True at real (non-padding) ViT token positions.
    vit_features_mask: torch.Tensor | None = None
    # dual_stream: pristine-base ViT features (same keys/shapes as vit_features,
    # which carries the LoRA-adapted stream when dual_stream is active).
    vit_features_base: dict[str, torch.Tensor] | None = None
    # dual_stream: layer features from the LoRA-enabled second LLM pass.
    layer_features_lora: dict[int, torch.Tensor] | None = None

    @classmethod
    def from_backbone_output(cls, output: Any) -> 'VLMInterface':
        """Map raw backbone forward output; add or rename fields only here."""
        return cls(
            last_hidden_state=output.last_hidden_state,
            logits=getattr(output, 'logits', None),
            stage_token=getattr(output, 'stage_token', None),
            hidden_states=getattr(output, 'hidden_states', None),
            past_key_values=getattr(output, 'past_key_values', None),
            layer_features=getattr(output, 'layer_features', None),
            layer_kv_cache=getattr(output, 'layer_kv_cache', None),
            labels=getattr(output, 'labels', None),
            attentions=getattr(output, 'attentions', None),
            metadata=getattr(output, 'metadata', None),
            image_mask=getattr(output, 'image_mask', None),
            vit_features=getattr(output, 'vit_features', None),
            vit_features_mask=getattr(output, 'vit_features_mask', None),
            vit_features_base=getattr(output, 'vit_features_base', None),
            layer_features_lora=getattr(output, 'layer_features_lora', None),
        )


def rms_norm(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply parameter-free RMSNorm to hidden states.

    This matches the normalization applied to extracted hidden states in
    Qwen3VLTextModelWithLayerExtraction.feature_extraction_norm.

    Args:
        hidden_states: Input tensor [..., hidden_dim]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)


def collect_required_vlm_layers(head_configs: dict) -> list[int]:
    """
    Collect all VLM layer indices required by heads from their configurations.

    This function scans all head configs and collects unique VLM layer indices
    that need to be extracted from the backbone. Supports `-1` to represent
    the last VLM layer (resolved later at runtime).

    Args:
        head_configs: Dictionary of head configurations
            Example: {'action_predict': {'multi_layer_vlm': True, 'cross_layer_mapping': [...]}}

    Returns:
        List of unique VLM layer indices (sorted, may include -1)
        Empty list if no heads require multi-layer extraction

    Example:
        >>> head_configs = {
        ...     'action_predict': {
        ...         'multi_layer_vlm': True,
        ...         'cross_layer_mapping': [[0, 23, 6], [24, 47, -1]]
        ...     }
        ... }
        >>> collect_required_vlm_layers(head_configs)
        [6, -1]
    """
    required_layers = set()

    for head_name, head_config in head_configs.items():
        # Check if this head uses multi-layer VLM features
        multi_layer_vlm = getattr(head_config, 'multi_layer_vlm', False)

        if not multi_layer_vlm:
            continue

        # Extract VLM layer indices from cross_layer_mapping
        cross_layer_mapping = getattr(head_config, 'cross_layer_mapping', [])
        for dit_start, dit_end, vlm_layer in cross_layer_mapping:
            required_layers.add(vlm_layer)

        # condition_router pool extras also need extraction
        router = getattr(head_config, 'condition_router', None)
        if router is not None and getattr(router, 'enabled', False):
            for layer in getattr(router, 'extra_llm_layers', []):
                required_layers.add(layer)

    return sorted(required_layers)


def resolve_layer_indices(layer_indices: list[int], num_vlm_layers: int) -> list[int]:
    """
    Resolve layer indices by replacing -1 with the actual last layer index.

    Args:
        layer_indices: List of layer indices (may contain -1)
        num_vlm_layers: Total number of VLM layers

    Returns:
        List of resolved layer indices (no -1)

    Example:
        >>> resolve_layer_indices([6, 13, -1], 28)
        [6, 13, 27]
    """
    last_layer_idx = num_vlm_layers - 1
    return [last_layer_idx if idx == -1 else idx for idx in layer_indices]


def prepare_encoder_states_for_head(
    vlm_output: VLMInterface,
    cross_layer_mapping: Sequence[Sequence[int]],
    projectors: nn.ModuleDict,
) -> dict[int, torch.Tensor]:
    """
    Prepare encoder states for head consumption from VLM hidden states.

    This function extracts and projects VLM features from specified layers.
    Always returns a dict for consistency, with single-layer mode as a special case.

    Args:
        vlm_output: Output from VLM backbone forward pass
            - Must have `last_hidden_state` attribute
            - If multi-layer mode: must have `layer_features` attribute
        cross_layer_mapping: List of (dit_start, dit_end, vlm_layer) tuples
            - Empty list: single-layer mode (use last_hidden_state only)
            - Non-empty: multi-layer mode (extract specific VLM layers)
        projectors: nn.ModuleDict mapping VLM layer indices to projectors
            - Keys are str(vlm_layer_idx), e.g., '6', '13', '-1'
            - Single-layer mode: should have key '-1' for last layer

    Returns:
        dict[vlm_layer_idx, Tensor [B, seq_len, head_dim]]
        - Single-layer mode: {-1: projected_tensor}
        - Multi-layer mode: {6: tensor1, 13: tensor2, -1: tensor3}

    Raises:
        ValueError: If multi-layer mode but vlm_output lacks layer_features
        ValueError: If required VLM layer is not found in layer_features
        ValueError: If required projector is not found in ModuleDict

    Example (single-layer):
        >>> projectors = nn.ModuleDict({'-1': nn.Linear(2048, 1024)})
        >>> encoder_states = prepare_encoder_states_for_head(
        ...     vlm_output, [], projectors
        ... )
        >>> encoder_states[-1].shape  # [B, seq_len, 1024]

    Example (multi-layer):
        >>> projectors = nn.ModuleDict({
        ...     '6': nn.Linear(2048, 1024),
        ...     '-1': nn.Linear(2048, 1024)
        ... })
        >>> encoder_states = prepare_encoder_states_for_head(
        ...     vlm_output, [[0, 47, 6], [48, 95, -1]], projectors
        ... )
        >>> encoder_states[6].shape   # [B, seq_len, 1024]
        >>> encoder_states[-1].shape  # [B, seq_len, 1024]
    """
    # Single-layer mode: empty cross_layer_mapping
    if not cross_layer_mapping:
        # Use last_hidden_state and projector with key '-1'
        vlm_tokens = vlm_output.last_hidden_state
        if '-1' not in projectors:
            raise ValueError(
                f'Single-layer mode requires projector with key "-1". '
                f'Available projectors: {list(projectors.keys())}'
            )
        return {-1: projectors['-1'](vlm_tokens)}

    # Multi-layer mode: extract and project features from specified layers
    if not hasattr(vlm_output, 'layer_features') or vlm_output.layer_features is None:
        raise ValueError(
            'multi_layer_vlm=True requires vlm_output to have layer_features, '
            'but layer_features is None. Ensure backbone forward is called with extract_layers.'
        )

    # Collect unique VLM layers from mapping
    unique_vlm_layers = set(vlm_layer for _, _, vlm_layer in cross_layer_mapping)

    # Dynamically resolve -1 to actual last layer index
    available_layers = list(vlm_output.layer_features.keys())
    if not available_layers:
        raise ValueError('vlm_output.layer_features is empty')
    last_layer_idx = max(available_layers)

    # Extract and project features for each unique VLM layer
    encoder_states = {}
    for vlm_layer in unique_vlm_layers:
        # Resolve -1 to actual last layer
        actual_vlm_layer = last_layer_idx if vlm_layer == -1 else vlm_layer

        # Check if layer exists in extracted features
        if actual_vlm_layer not in vlm_output.layer_features:
            raise ValueError(
                f'VLM layer {actual_vlm_layer} not found in layer_features. '
                f'Available layers: {sorted(vlm_output.layer_features.keys())}'
            )

        # Get raw features
        raw_features = vlm_output.layer_features[actual_vlm_layer]

        # Find the correct projector (try both -1 and actual_vlm_layer as keys)
        projector_key = str(vlm_layer)  # Keep original key (may be -1)
        if projector_key not in projectors:
            projector_key = str(actual_vlm_layer)
            if projector_key not in projectors:
                raise ValueError(
                    f'Projector for VLM layer {vlm_layer} (actual: {actual_vlm_layer}) not found. '
                    f'Available projectors: {list(projectors.keys())}'
                )

        # Project features
        projector = projectors[projector_key]
        projected = projector(raw_features)

        # Store with original key (may be -1)
        encoder_states[vlm_layer] = projected

    return encoder_states


def prepare_raw_kv_for_head(
    vlm_output: VLMInterface,
    cross_layer_mapping: Sequence[Sequence[int]],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """
    Prepare raw KV cache from VLM layers for true Pi-KV architecture.

    Unlike prepare_kv_cache_for_head which converts KV to hidden states and re-projects,
    this function returns raw KV tensors to be directly concatenated with DiT's own KV
    in the attention layer. This avoids unnecessary conversions when VLM KV heads and
    DiT heads have matching dimensions.

    Args:
        vlm_output: Output from VLM backbone forward pass
            - Must have `past_key_values` for single-layer mode
            - Must have `layer_kv_cache` for multi-layer mode
        cross_layer_mapping: List of (dit_start, dit_end, vlm_layer) tuples
            - Empty list: single-layer mode (use last layer KV cache)
            - Non-empty: multi-layer mode (extract specific VLM layers)

    Returns:
        dict[vlm_layer_idx, (key_cache, value_cache)]
        - key_cache: [B, num_kv_heads, seq_len, head_dim]
        - value_cache: [B, num_kv_heads, seq_len, head_dim]
        - Single-layer mode: {-1: (key, value)}
        - Multi-layer mode: {6: (k1, v1), 13: (k2, v2), ...}
    """
    # Single-layer mode
    if not cross_layer_mapping:
        if not hasattr(vlm_output, 'past_key_values') or vlm_output.past_key_values is None:
            raise ValueError('Pi-KV mode requires past_key_values in VLM output (use_cache=True)')

        cache = vlm_output.past_key_values
        last_layer_idx = len(cache) - 1
        key_cache, value_cache = cache[last_layer_idx]
        return {-1: (key_cache.contiguous(), value_cache.contiguous())}

    # Multi-layer mode
    if not hasattr(vlm_output, 'layer_kv_cache') or vlm_output.layer_kv_cache is None:
        raise ValueError(
            'multi_layer_vlm=True with raw KV requires vlm_output to have layer_kv_cache.'
        )

    unique_vlm_layers = set(vlm_layer for _, _, vlm_layer in cross_layer_mapping)
    available_layers = list(vlm_output.layer_kv_cache.keys())
    if not available_layers:
        raise ValueError('vlm_output.layer_kv_cache is empty')
    last_layer_idx = max(available_layers)

    raw_kv = {}
    for vlm_layer in unique_vlm_layers:
        actual_vlm_layer = last_layer_idx if vlm_layer == -1 else vlm_layer

        if actual_vlm_layer not in vlm_output.layer_kv_cache:
            raise ValueError(
                f'VLM layer {actual_vlm_layer} KV cache not found. '
                f'Available layers: {sorted(vlm_output.layer_kv_cache.keys())}'
            )

        key_cache, value_cache = vlm_output.layer_kv_cache[actual_vlm_layer]
        raw_kv[vlm_layer] = (key_cache.contiguous(), value_cache.contiguous())

    return raw_kv


def prepare_kv_cache_for_head(
    vlm_output: VLMInterface,
    cross_layer_mapping: Sequence[Sequence[int]],
    projectors: nn.ModuleDict,
) -> dict[int, torch.Tensor]:
    """
    Prepare KV cache from VLM layers for Pi-KV architecture.

    Note: This is a pragmatic implementation that converts KV cache back to hidden state format.
    The KV cache is reshaped from [B, num_heads, seq, head_dim] to [B, seq, dim] and then
    projected to head dimension. While not as efficient as using raw KV directly, this approach:
    - Works with existing attention infrastructure
    - Still provides multi-layer benefits
    - Can be optimized later with custom attention

    Always returns a dict for consistency, with single-layer mode as a special case.

    Args:
        vlm_output: Output from VLM backbone forward pass
            - Must have `past_key_values` for single-layer mode
            - Must have `layer_kv_cache` for multi-layer mode
        cross_layer_mapping: List of (dit_start, dit_end, vlm_layer) tuples
            - Empty list: single-layer mode (use last layer KV cache)
            - Non-empty: multi-layer mode (extract specific VLM layers)
        projectors: nn.ModuleDict mapping VLM layer indices to projectors
            - Keys are str(vlm_layer_idx), e.g., '6', '13', '-1'
            - Single-layer mode: should have key '-1' for last layer

    Returns:
        dict[vlm_layer_idx, Tensor [B, seq_len, head_dim]]
        - Single-layer mode: {-1: projected_tensor}
        - Multi-layer mode: {6: tensor1, 13: tensor2, -1: tensor3}

    Raises:
        ValueError: If multi-layer mode but vlm_output lacks layer_kv_cache
        ValueError: If required VLM layer KV cache is not found
        ValueError: If required projector is not found
    """

    def kv_cache_to_hidden_states(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        target_hidden_size: int
    ) -> torch.Tensor:
        """
        Convert KV cache back to hidden state format.

        Strategy:
        1. Use Value cache as primary representation (contains output content)
        2. Apply RMSNorm for stability (matching hidden state extraction)
        3. For GQA: repeat key-value heads to match attention heads

        Args:
            key_cache: [B, num_key_value_heads, seq, head_dim]
            value_cache: [B, num_key_value_heads, seq, head_dim]
            target_hidden_size: Expected hidden size (e.g., 2048 for Qwen3-VL-3B)

        Returns:
            hidden_states: [B, seq, target_hidden_size]
        """
        # Use Value cache as primary representation
        # (Value contains the content to be output, Key is for similarity matching)
        B, num_kv_heads, seq_len, head_dim = value_cache.shape

        # Reshape to [B, seq, num_kv_heads * head_dim]
        hidden_states = value_cache.transpose(1, 2)  # [B, seq, num_kv_heads, head_dim]
        kv_dim = num_kv_heads * head_dim

        # For GQA: repeat key-value heads to match attention heads
        if kv_dim != target_hidden_size:
            repeat_factor = target_hidden_size // kv_dim
            if repeat_factor * kv_dim == target_hidden_size:
                # Perfect division: repeat each head
                hidden_states = hidden_states.repeat(1, 1, 1, repeat_factor)
                hidden_states = hidden_states.reshape(B, seq_len, target_hidden_size)
            else:
                # Imperfect division: flatten first then repeat
                hidden_states = hidden_states.reshape(B, seq_len, kv_dim)
                hidden_states = hidden_states.repeat(1, 1, repeat_factor)
                # Truncate or pad to exact target size
                if hidden_states.shape[-1] > target_hidden_size:
                    hidden_states = hidden_states[..., :target_hidden_size]
                elif hidden_states.shape[-1] < target_hidden_size:
                    padding = target_hidden_size - hidden_states.shape[-1]
                    hidden_states = torch.nn.functional.pad(
                        hidden_states, (0, padding), mode='constant', value=0
                    )
        else:
            hidden_states = hidden_states.reshape(B, seq_len, kv_dim)

        # Apply RMSNorm to match hidden state extraction
        # This is critical for stable training!
        hidden_states = rms_norm(hidden_states, eps=1e-6)

        return hidden_states

    # Get target hidden size from VLM output
    target_hidden_size = vlm_output.last_hidden_state.shape[-1]

    # Single-layer mode: empty cross_layer_mapping
    if not cross_layer_mapping:
        # Use KV from last layer in past_key_values
        if not hasattr(vlm_output, 'past_key_values') or vlm_output.past_key_values is None:
            raise ValueError('Pi-KV mode requires past_key_values in VLM output (use_cache=True)')

        # past_key_values is a Cache object, get the last layer's KV
        cache = vlm_output.past_key_values

        # Access DynamicCache via indexing
        try:
            # DynamicCache supports indexing: cache[layer_idx] returns (key, value)
            last_layer_idx = len(cache) - 1
            key_cache, value_cache = cache[last_layer_idx]

            # Convert to hidden states with correct target size and project
            hidden_states = kv_cache_to_hidden_states(key_cache, value_cache, target_hidden_size)

            if '-1' not in projectors:
                raise ValueError(
                    f'Single-layer mode requires projector with key "-1". '
                    f'Available projectors: {list(projectors.keys())}'
                )

            return {-1: projectors['-1'](hidden_states)}
        except Exception as e:
            raise ValueError(f'Failed to extract KV cache from {type(cache)}: {str(e)}')

    # Multi-layer mode: extract KV from specified layers
    if not hasattr(vlm_output, 'layer_kv_cache') or vlm_output.layer_kv_cache is None:
        raise ValueError(
            'multi_layer_vlm=True with KV format requires vlm_output to have layer_kv_cache, '
            'but layer_kv_cache is None. Ensure backbone forward is called with extract_layers and use_cache=True.'
        )

    # Collect unique VLM layers from mapping
    unique_vlm_layers = set(vlm_layer for _, _, vlm_layer in cross_layer_mapping)

    # Dynamically resolve -1 to actual last layer index
    available_layers = list(vlm_output.layer_kv_cache.keys())
    if not available_layers:
        raise ValueError('vlm_output.layer_kv_cache is empty')
    last_layer_idx = max(available_layers)

    # Extract and convert KV cache for each unique VLM layer
    encoder_states = {}
    for vlm_layer in unique_vlm_layers:
        # Resolve -1 to actual last layer
        actual_vlm_layer = last_layer_idx if vlm_layer == -1 else vlm_layer

        # Check if layer exists in extracted KV cache
        if actual_vlm_layer not in vlm_output.layer_kv_cache:
            raise ValueError(
                f'VLM layer {actual_vlm_layer} KV cache not found. '
                f'Available layers: {sorted(vlm_output.layer_kv_cache.keys())}'
            )

        # Get KV cache (tuple of key, value tensors)
        key_cache, value_cache = vlm_output.layer_kv_cache[actual_vlm_layer]

        # Convert to hidden states with correct target size
        hidden_states = kv_cache_to_hidden_states(key_cache, value_cache, target_hidden_size)

        # Find the correct projector
        projector_key = str(vlm_layer)
        if projector_key not in projectors:
            projector_key = str(actual_vlm_layer)
            if projector_key not in projectors:
                raise ValueError(
                    f'Projector for VLM layer {vlm_layer} (actual: {actual_vlm_layer}) not found. '
                    f'Available projectors: {list(projectors.keys())}'
                )

        # Project features
        projector = projectors[projector_key]
        projected = projector(hidden_states)

        # Store with original key (may be -1)
        encoder_states[vlm_layer] = projected

    return encoder_states
