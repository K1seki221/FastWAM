"""Configuration validation for architecture combinations.

This module provides strict validation for architecture configurations,
ensuring only valid and meaningful combinations of parameters are used.

Example:
    from models.utils.config_validation import validate_architecture_config

    validate_architecture_config(
        dit_block_type='cross_only',
        vlm_fusion_mode='context',
        vlm_feature_format='hidden',
        multi_layer_vlm=True,
    )

    # Full attention:
    validate_architecture_config(
        dit_block_type='full_attention',
        vlm_fusion_mode='joint',
        vlm_feature_format='hidden',
        multi_layer_vlm=False,
    )
"""

import os
from typing import Literal


class InvalidArchitectureError(ValueError):
    """Raised when an invalid architecture configuration is detected."""

    pass


# Valid architecture combinations
# Format: (dit_block_type, vlm_fusion_mode, multi_layer_vlm) -> description
VALID_ARCHITECTURES = {
    # Baseline: Combined block with standard cross attention
    ('combined', 'cross', False): 'Baseline: 48-layer combined DiT with single-layer VLM cross attention',
    ('combined', 'cross', True): 'Multi-layer: 96-layer combined DiT with multi-layer VLM cross attention',
    # Separate: Separate self and cross blocks
    ('separate', 'cross', False): 'Separate: 96-layer separate DiT with single-layer VLM cross attention',
    ('separate', 'cross', True): 'Separate: 96-layer separate DiT with multi-layer VLM cross attention',
    # Pi Architecture: Cross-only block with context fusion
    ('cross_only', 'context', False): 'Pi-Single: 32-layer Pi architecture with single-layer VLM context',
    ('cross_only', 'context', True): 'Pi-Multi: 32-layer Pi architecture with multi-layer VLM context',
    # Full Attention: Joint self-attention over concatenated [VLM, action] tokens
    ('full_attention', 'joint', False): 'FullAttn: Joint self-attention DiT with single-layer VLM',
}

# Invalid combinations with explanations
INVALID_ARCHITECTURES = {
    ('cross_only', 'cross'): (
        'cross_only + cross is not recommended. '
        'cross_only blocks are designed for Pi architecture (context fusion), '
        'not standard cross attention. Use combined or separate blocks for cross attention.'
    ),
    ('full_attention', 'cross'): (
        'full_attention + cross is invalid. '
        'full_attention uses joint self-attention (vlm_fusion_mode=joint), not cross attention.'
    ),
    ('full_attention', 'context'): (
        'full_attention + context is invalid. '
        'full_attention uses joint self-attention (vlm_fusion_mode=joint), not context fusion.'
    ),
}


def validate_architecture_config(
    dit_block_type: Literal['combined', 'separate', 'cross_only', 'full_attention'],
    vlm_fusion_mode: Literal['cross', 'context', 'joint'],
    vlm_feature_format: Literal['hidden', 'kv'],
    multi_layer_vlm: bool,
    strict: bool = True,
) -> None:
    """Validate that the architecture configuration is valid and meaningful.

    Args:
        dit_block_type: Type of DiT block
            - 'combined': Self + Cross + FFN in one block
            - 'separate': Separate Self and Cross blocks
            - 'cross_only': Only one attention (Pi architecture)
            - 'full_attention': Joint self-attention over [VLM, action] tokens
        vlm_fusion_mode: How VLM information is fused
            - 'cross': Standard cross attention (Q=DiT, KV=VLM)
            - 'context': Pi architecture (Q=DiT, KV=concat[DiT_history, VLM])
            - 'joint': Full attention (Q/K/V = concat[VLM, DiT], bidirectional)
        vlm_feature_format: Format of VLM features
            - 'hidden': Use VLM hidden states
            - 'kv': Use VLM KV cache (converted to hidden states)
        multi_layer_vlm: Whether to use features from multiple VLM layers
        strict: If True (default), raise InvalidArchitectureError for invalid configs.
                If False, only warn. Can be overridden by env var ALLOW_INVALID_ARCHITECTURE=1.

    Raises:
        InvalidArchitectureError: If configuration is invalid and strict=True

    Example:
        >>> # Valid Pi architecture
        >>> validate_architecture_config(
        ...     dit_block_type='cross_only',
        ...     vlm_fusion_mode='context',
        ...     vlm_feature_format='hidden',
        ...     multi_layer_vlm=True
        ... )

        >>> # Invalid: cross_only with cross mode
        >>> validate_architecture_config(
        ...     dit_block_type='cross_only',
        ...     vlm_fusion_mode='cross',
        ...     vlm_feature_format='hidden',
        ...     multi_layer_vlm=False
        ... )  # Raises InvalidArchitectureError
    """
    # Check if env var allows invalid architectures
    allow_invalid = os.environ.get('ALLOW_INVALID_ARCHITECTURE', '0') == '1'

    # Check for explicitly invalid combinations
    config_key = (dit_block_type, vlm_fusion_mode)
    if config_key in INVALID_ARCHITECTURES:
        error_msg = INVALID_ARCHITECTURES[config_key]

        if strict and not allow_invalid:
            raise InvalidArchitectureError(
                f'Invalid architecture configuration: {error_msg}\n'
                f'Got: dit_block_type={dit_block_type!r}, vlm_fusion_mode={vlm_fusion_mode!r}\n'
                f'To override this check, set environment variable ALLOW_INVALID_ARCHITECTURE=1'
            )
        else:
            import warnings

            warnings.warn(f'Architecture configuration warning: {error_msg}', UserWarning)
            return

    # Check if combination is in valid list (optional - for documentation purposes)
    full_key = (dit_block_type, vlm_fusion_mode, multi_layer_vlm)
    if full_key not in VALID_ARCHITECTURES:
        import warnings

        warnings.warn(
            f'Architecture combination not in predefined valid list: '
            f'dit_block_type={dit_block_type!r}, vlm_fusion_mode={vlm_fusion_mode!r}, '
            f'multi_layer_vlm={multi_layer_vlm!r}. '
            f'This may be a new experimental configuration.',
            UserWarning,
        )

    # Validate that vlm_feature_format is consistent with requirements
    # (Currently both 'hidden' and 'kv' are valid for all modes, so no validation needed)
    # This is a placeholder for future constraints if needed

    return None


def get_architecture_description(
    dit_block_type: str,
    vlm_fusion_mode: str,
    multi_layer_vlm: bool,
) -> str:
    """Get a human-readable description of the architecture.

    Args:
        dit_block_type: Type of DiT block
        vlm_fusion_mode: How VLM information is fused
        multi_layer_vlm: Whether to use features from multiple VLM layers

    Returns:
        Human-readable description of the architecture, or 'Unknown architecture' if not found
    """
    key = (dit_block_type, vlm_fusion_mode, multi_layer_vlm)
    return VALID_ARCHITECTURES.get(key, 'Unknown architecture')
