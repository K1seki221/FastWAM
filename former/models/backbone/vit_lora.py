"""Switchable LoRA for the VLM ViT, used only on the ViT->DiT feature pass.

Deliberately hand-rolled (no peft): the training image is not guaranteed to
ship peft, state_dict keys stay predictable for the serialize()/deserialize()
legacy remap, and the never-merged, second-pass-only semantics are explicit.

Usage (see IronVLA._init_vit_condition):
    wrappers = install_vit_lora(visual, r=16, alpha=32, dropout=0.0,
                                targets=['attn_qkv', 'attn_proj'])
    set_vit_lora_enabled(wrappers, True)   # before the ViT->DiT pass
    set_vit_lora_enabled(wrappers, False)  # before the VLM pass (default)

With LoRA disabled the wrapper is bit-identical to the base Linear.
"""

import math

import torch
import torch.nn as nn

# target name -> attribute path inside one Qwen3VLVisionBlock
_TARGET_PATHS = {
    'attn_qkv': ('attn', 'qkv'),
    'attn_proj': ('attn', 'proj'),
    'mlp_fc1': ('mlp', 'linear_fc1'),
    'mlp_fc2': ('mlp', 'linear_fc2'),
}

# target name -> attribute path inside one Qwen3VLTextDecoderLayer
_LLM_TARGET_PATHS = {
    'attn_q': ('self_attn', 'q_proj'),
    'attn_k': ('self_attn', 'k_proj'),
    'attn_v': ('self_attn', 'v_proj'),
    'attn_o': ('self_attn', 'o_proj'),
    'mlp_gate': ('mlp', 'gate_proj'),
    'mlp_up': ('mlp', 'up_proj'),
    'mlp_down': ('mlp', 'down_proj'),
}


class SwitchableLoRALinear(nn.Module):
    """Wraps a base ``nn.Linear`` with a rank-``r`` residual path gated by ``enabled``.

    ``lora_B`` is zero-initialized, so an enabled-but-untrained wrapper is also
    an identity delta. ``enabled`` is a plain attribute (not a buffer): it is
    process-local runtime state toggled around the two ViT passes, never saved.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scaling = alpha / r
        self.enabled = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Match the (frozen) base weights' dtype/device so the LoRA path does
        # not upcast the bf16 ViT activations.
        self.lora_A.to(base.weight.device, base.weight.dtype)
        self.lora_B.to(base.weight.device, base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return out


def install_vit_lora(visual: nn.Module, *, r: int, alpha: int, dropout: float,
                     targets: list[str]) -> list[SwitchableLoRALinear]:
    """Replace target Linears in every ViT block with SwitchableLoRALinear.

    Must run AFTER pretrained weights are loaded and AFTER freeze_backbones —
    the caller re-enables requires_grad on the returned wrappers' lora_A/B.
    Idempotence guard: raises if a target is already wrapped.
    """
    wrappers: list[SwitchableLoRALinear] = []
    blocks = visual.blocks
    assert isinstance(blocks, nn.ModuleList)
    for block in blocks:
        for target in targets:
            parent_name, attr_name = _TARGET_PATHS[target]
            parent = getattr(block, parent_name)
            base = getattr(parent, attr_name)
            if isinstance(base, SwitchableLoRALinear):
                raise RuntimeError(f'ViT LoRA already installed on {parent_name}.{attr_name}')
            assert isinstance(base, nn.Linear), f'{parent_name}.{attr_name} is not nn.Linear'
            setattr(parent, attr_name, SwitchableLoRALinear(base, r=r, alpha=alpha, dropout=dropout))
            wrappers.append(getattr(parent, attr_name))
    return wrappers


def install_llm_lora(language_model: nn.Module, *, r: int, alpha: int, dropout: float,
                     targets: list[str]) -> list[SwitchableLoRALinear]:
    """Replace target Linears in every LLM decoder layer with SwitchableLoRALinear.

    Same contract as install_vit_lora: run AFTER pretrained load and AFTER
    freeze_backbones; the caller re-enables requires_grad on lora_A/B only.
    The embedding table and lm_head are never wrapped (tied weights stay frozen).
    """
    wrappers: list[SwitchableLoRALinear] = []
    layers = language_model.layers
    assert isinstance(layers, nn.ModuleList)
    for layer in layers:
        for target in targets:
            parent_name, attr_name = _LLM_TARGET_PATHS[target]
            parent = getattr(layer, parent_name)
            base = getattr(parent, attr_name)
            if isinstance(base, SwitchableLoRALinear):
                raise RuntimeError(f'LLM LoRA already installed on {parent_name}.{attr_name}')
            assert isinstance(base, nn.Linear), f'{parent_name}.{attr_name} is not nn.Linear'
            setattr(parent, attr_name, SwitchableLoRALinear(base, r=r, alpha=alpha, dropout=dropout))
            wrappers.append(getattr(parent, attr_name))
    return wrappers


def set_vit_lora_enabled(wrappers: list[SwitchableLoRALinear], enabled: bool) -> None:
    for wrapper in wrappers:
        wrapper.enabled = enabled


def mark_vit_lora_trainable(wrappers: list[SwitchableLoRALinear]) -> None:
    """Re-enable grads on LoRA params (base stays under stage freeze policy)."""
    for wrapper in wrappers:
        wrapper.lora_A.weight.requires_grad_(True)
        wrapper.lora_B.weight.requires_grad_(True)
