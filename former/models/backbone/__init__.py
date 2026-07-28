from .embedding_utils import PlaceholderReplacementError, replace_placeholder_embeddings
from .qwen2_5_vl.qwen_conditional import Qwen2_5_Conditional_VL_Wrapper

# Import all backbone wrappers to trigger registration
from .qwen3_vl.qwen3_vl_conditional import Qwen3_VL_Wrapper
from .registry import BACKBONE_REGISTRY, build_backbone, register_backbone

# Qwen3.5 requires transformers with qwen3_5 module; skip if unavailable
try:
    from .qwen3_5_vl.qwen3_5_vl_conditional import Qwen3_5_VL_Wrapper  # noqa: F401
except ImportError:
    pass

__all__ = [
    'BACKBONE_REGISTRY',
    'build_backbone',
    'register_backbone',
    'replace_placeholder_embeddings',
    'PlaceholderReplacementError',
    'Qwen3_VL_Wrapper',
    'Qwen2_5_Conditional_VL_Wrapper',
]
