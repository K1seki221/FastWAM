HEAD_REGISTRY = {}


def register_head(cls):
    HEAD_REGISTRY[cls.__name__] = cls
    return cls


def build_head(head_type, config, **runtime_kwargs):
    """Build a head from registry.

    Args:
        head_type: The registered head class name
        config: The head config from YAML (should not be modified)
        **runtime_kwargs: Runtime-determined values (e.g., backbone_hidden_dim)
    """
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f'Unknown head type: {head_type}')
    return HEAD_REGISTRY[head_type](config, **runtime_kwargs)


# --- Trigger registration by importing modules that define heads ---
from models.head.flow_matching_action import FlowMatchingAction  # noqa: E402, F401, I001
from models.head.flow_matching_groot_n1d7 import FlowMatchingActionGr00tN1d7  # noqa: E402, F401
from models.head.auto_regressive_dummy import AutoRegressiveDummy  # noqa: E402, F401
from models.head.real_time_chunking_action import RealTimeChunkingAction  # noqa: E402, F401
from models.head.element_wise_lm_head import ElementWiseLMHead  # noqa: E402, F401
from models.head.categorical_value import CategoricalValueHead  # noqa: E402, F401
from models.head.bbox_detr_head import BboxDetrHead  # noqa: E402, F401
