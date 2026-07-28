import torch

BACKBONE_REGISTRY = {}


def register_backbone(cls):
    BACKBONE_REGISTRY[cls.__name__] = cls
    return cls


def build_backbone(backbone_config):
    if backbone_config.type not in BACKBONE_REGISTRY:
        raise ValueError(f'Unknown backbone type: {backbone_config.type}')
    model_id = backbone_config.model_id

    return BACKBONE_REGISTRY[backbone_config.type].from_pretrained(
        model_id,
        config=backbone_config,
        dtype=torch.bfloat16,
        attn_implementation=backbone_config.attn_implementation,
    )


# --- Trigger registration by importing modules that define heads ---
