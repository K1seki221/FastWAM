import torch
from configs.config_schema import ElementWiseLMHeadConfig
from torch import nn
from torch.nn import CrossEntropyLoss

from dataset.util import get_processor
from models.head.base_head import BaseHead
from models.head.registry import register_head
from models.utils.vlm_feature_utils import VLMInterface


@register_head
class ElementWiseLMHead(BaseHead):
    """Element-wise language modeling head for unified CLS + AR prediction.

    Unlike AutoRegressiveDummy which shifts logits/labels inside the loss,
    this head computes element-wise cross-entropy: logits[i] vs labels[i].

    The shift for autoregressive positions is expected to be baked into the
    labels during data construction:
        - CLS positions:  label[i] = target token (e.g. BoA/BoR)
        - AR positions:   label[i] = input_ids[i+1]  (pre-shifted)
        - Ignored:        label[i] = -100
    """

    def __init__(
        self, head_config: ElementWiseLMHeadConfig, *, out_features: int, model_id: str | None = None, **_kwargs
    ):
        nn.Module.__init__(self)
        self.config: ElementWiseLMHeadConfig = head_config
        self.model_id = model_id
        self.vocab_size = out_features
        self.get_loss_fn(head_config.loss_type)

    def preprocess_tokens(self, inputs):
        tokens = []
        return tokens

    def get_loss_fn(self, loss_type: str):
        self.loss_weight = self.config.loss_weight
        self.weight_tensor = self.build_weight_tensor(self.vocab_size)
        if loss_type == 'cross_entropy':
            self.loss_fn = CrossEntropyLoss(
                weight=self.weight_tensor, reduction='mean'
            )
            # CrossEntropyLoss registers ``weight`` as a *persistent* buffer
            # (via _WeightedLoss.__init__), so it ends up in state_dict.
            # weight_tensor is deterministic from class_weight config and is
            # rebuilt at module construction, so it shouldn't be serialized.
            # Re-register the same buffer as non-persistent. Without this,
            # any ckpt saved before this head was added to vlmact_r01 (e.g.
            # action-only pretrains from before 2026-02-09) cannot be strict-
            # loaded because the key is missing.
            self.loss_fn.register_buffer(
                'weight', self.weight_tensor, persistent=False
            )
        else:
            raise ValueError(f'Loss type {loss_type} not supported')

    def get_token_id(self, token: str):
        assert self.model_id is not None, 'model_id must be set to get token IDs'
        tokenizer = getattr(get_processor(self.model_id), 'tokenizer')
        return tokenizer(token, return_tensors='pt')['input_ids'].reshape(-1)

    def build_weight_tensor(
        self, vocab_size: int, device: torch.device | None = None
    ):
        """Build weight tensor of size [vocab_size] with class weights."""
        if device is None:
            device = torch.device('cpu')
        class_weight_dict = self.config.class_weight or {}
        weight = torch.ones(vocab_size, device=device, dtype=torch.float32)
        for token, class_weight in class_weight_dict.items():
            token_id = self.get_token_id(f'<|{token}|>')
            weight[token_id] = float(class_weight)
        return weight

    def _ensure_weight_size(self, vocab_size: int, device: torch.device):
        """Lazily resize the CE weight tensor when vocab grows (e.g. after resize_token_embeddings)."""
        if self.weight_tensor is not None and self.weight_tensor.shape[0] != vocab_size:
            new_weight = torch.ones(vocab_size, device=device, dtype=torch.float32)
            old_size = self.weight_tensor.shape[0]
            new_weight[:old_size] = self.weight_tensor.to(device)
            self.weight_tensor = new_weight
            self.loss_fn = CrossEntropyLoss(weight=new_weight, reduction='mean')

    def compute_loss(self, output_dict):
        logits = output_dict['pred']
        labels = output_dict['gt']
        loss = None
        if labels is not None:
            vocab_size = logits.size(-1)
            device = logits.device
            self._ensure_weight_size(vocab_size, device)

            # Element-wise: logits[i] predicts labels[i] directly (no shift)
            flat_logits = logits.contiguous().view(-1, vocab_size)
            flat_labels = labels.contiguous().view(-1)
            flat_labels = flat_labels.to(device)
            valid_mask = flat_labels != -100
            if valid_mask.any():
                valid_logits = flat_logits[valid_mask]
                valid_labels = flat_labels[valid_mask]
                loss = self.loss_fn(valid_logits, valid_labels)
            else:
                loss = flat_logits.sum() * 0.0
        return loss

    def forward(self, vlm_output: VLMInterface | None, **kwargs):
        if vlm_output is None:
            raise ValueError('ElementWiseLMHead requires vlm_output')
        logits = vlm_output.logits
        if logits is None:
            raise ValueError('ElementWiseLMHead expects vlm_output.logits from the backbone forward.')
        if kwargs['is_training']:
            labels = getattr(vlm_output, 'labels', None)
            out_dict = {
                'type': 'token',
                'gt': labels,
                'pred': logits,
            }
        else:
            # Element-wise: argmax gives direct predictions (no shift)
            pred_sequence = logits.argmax(dim=-1)

            out_dict = {
                'type': 'token',
                'pred': pred_sequence,
            }

        return out_dict
