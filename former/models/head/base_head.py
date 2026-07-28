from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn
from configs.config_schema import BaseHeadConfig

from models.utils.vlm_feature_utils import VLMInterface


class BaseHead(nn.Module, ABC):
    def __init__(self, head_config: BaseHeadConfig):
        super().__init__()
        self.config = head_config

        self.state_dim = head_config.state_dim
        self.action_dim = head_config.action_dim
        self.num_categories = head_config.num_categories
        self.temporal_weighting = head_config.temporal_weighting
        self.hidden_dim = head_config.hidden_dim

    @abstractmethod
    def forward(self, vlm_output: VLMInterface | None, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def compute_loss(self, output_dict: Dict[str, Any]) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute the loss for the given output dictionary.
        Example input:
            output_dict: {
                'pred': torch.Tensor,
                'gt': torch.Tensor,
                'mask': torch.Tensor,
            }
        Output:
            loss: torch.Tensor
        """
        pass
