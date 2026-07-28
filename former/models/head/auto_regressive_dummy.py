import torch
from configs.config_schema import AutoRegressiveHeadConfig
from torch.nn import CrossEntropyLoss

from dataset.util import get_processor
from models.head.base_head import BaseHead
from models.head.registry import register_head
from models.utils.vlm_feature_utils import VLMInterface


@register_head
class AutoRegressiveDummy(BaseHead):
    def __init__(
        self, head_config: AutoRegressiveHeadConfig, *, out_features: int, model_id: str | None = None, **_kwargs
    ):
        super().__init__(head_config)
        self.config: AutoRegressiveHeadConfig = head_config
        self.model_id = model_id
        self.vocab_size = out_features
        self.get_loss_fn(head_config.loss_type)

    def preprocess_tokens(self, inputs):
        tokens = []
        return tokens

    def get_loss_fn(self, loss_type: str):
        self.loss_weight = self.config.loss_weight  # weight to balance the multi-task loss
        self.weight_tensor = self.build_weight_tensor(self.vocab_size)  # cross entropy class weights
        if loss_type == 'cross_entropy':
            self.loss_fn = CrossEntropyLoss(weight=self.weight_tensor, reduction='mean')
        else:
            raise ValueError(f'Loss type {loss_type} not supported')

    def get_token_id(self, token: str):
        assert self.model_id is not None, 'model_id must be set to get token IDs'
        tokenizer = getattr(get_processor(self.model_id), 'tokenizer')
        return tokenizer(token, return_tensors='pt')['input_ids'].reshape(-1)

    def build_weight_tensor(self, vocab_size: int, device: torch.device | None = None):
        """Build weight tensor of size [vocab_size] with class weights applied"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weight_dict = self.config.class_weight or {}
        weight = torch.ones(vocab_size, device=device, dtype=torch.float32)
        for token, class_weight in class_weight_dict.items():
            token_id = self.get_token_id(f'<|{token}|>')
            weight[token_id] = float(class_weight)
        return weight

    def compute_loss(self, output_dict):
        logits = output_dict['pred']
        labels = output_dict['gt']
        loss = None
        if labels is not None:
            vocab_size = logits.size(-1)
            device = logits.device

            # Flatten the tokens (logits and labels are already aligned)
            flat_logits = logits.contiguous().view(-1, vocab_size)
            flat_labels = labels.contiguous().view(-1)
            # Enable model parallelism
            flat_labels = flat_labels.to(device)
            valid_mask = flat_labels != -100
            if valid_mask.any():
                # Apply valid mask to get only valid labels
                valid_logits = flat_logits[valid_mask]
                valid_labels = flat_labels[valid_mask]
                loss = self.loss_fn(valid_logits, valid_labels)
            else:
                loss = torch.tensor(0.0, device=device)
        return loss

    def forward(self, vlm_output: VLMInterface | None, **kwargs):
        if vlm_output is None:
            raise ValueError('AutoRegressiveDummy requires vlm_output')
        logits = vlm_output.logits
        if logits is None:
            raise ValueError('AutoRegressiveDummy expects vlm_output.logits from the backbone forward.')
        if kwargs['is_training']:
            labels = getattr(vlm_output, 'labels', None)
            out_dict = {'type': 'token', 'gt': labels, 'pred': logits}
        else:
            pred_sequence = logits.argmax(dim=-1)

            out_dict = {
                'type': 'token',
                'pred': pred_sequence,
            }

            # if 'action' in kwargs and 'action_mask' in kwargs:
            #     # add raw action gt and mask to the output dict
            #     out_dict['action_util'] = {
            #         'gt_action': kwargs['action'],
            #         'mask': kwargs['action_mask'],
            #     }

        return out_dict
