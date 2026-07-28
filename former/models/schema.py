"""Pydantic schema definitions for model outputs.

These schemas define typed outputs for model heads and inference results.
Supports both batched tensors (B, ...) from heads and lists from post-processing.

``extra='forbid'`` on every model means adding a new field to any output
requires updating the schema first — drift is caught loudly rather than
silently passed through.

Usage:
    from models.schema import ActionOutput, TextOutput, StageTokenOutput

    # Head returns batched tensor
    return ActionOutput(pred=pred_action, gt=gt_action, mask=mask)

    # Post-processing returns list
    return ActionOutput(pred=[action_1, action_2], ...)
"""

from typing import List, Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field


class ActionOutput(BaseModel):
    """Output from action heads (FlowMatchingAction, RealTimeChunkingAction).

    Supports both batched tensors (B, H, D) and lists of (H, D) tensors.

    Attributes:
        type: Always 'trajectory'.
        pred: Predicted actions - batched (B, H, D) or list of (H, D).
        gt: Ground truth actions.
        mask: Validity mask.
        prefix_mask: Optional RTC prefix mask (B, H) - training only.

    Stage tokens were previously nested here (``stage_token``,
    ``stage_tokens``); they now live at top-level ``infer_result['stage_token']``
    as :class:`StageTokenOutput`. See :class:`InferenceResult`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    type: Literal['trajectory'] = 'trajectory'
    pred: Union[torch.Tensor, List[torch.Tensor]] = Field(..., description='Predicted actions')
    gt: Optional[Union[torch.Tensor, List[torch.Tensor]]] = Field(default=None)
    mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = Field(default=None)
    prefix_mask: Optional[torch.Tensor] = Field(default=None)


class TextOutput(BaseModel):
    """Output from text heads (ElementWiseLMHead).

    Supports both batched tensors (B, S) and lists of (S,) tensors.

    Attributes:
        type: Always 'token'.
        pred: Predicted logits/tokens - batched (B, S, V) or list of (S,).
        gt: Ground truth labels.
        decoded_pred: Optional decoded prediction strings.
        decoded_gt: Optional decoded ground truth strings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    type: Literal['token'] = 'token'
    pred: Union[torch.Tensor, List[torch.Tensor]] = Field(..., description='Predicted tokens')
    gt: Optional[Union[torch.Tensor, List[torch.Tensor]]] = Field(default=None)
    decoded_pred: Optional[List[str]] = Field(default=None)
    decoded_gt: Optional[List[str]] = Field(default=None)


class StageTokenOutput(BaseModel):
    """Output for the stage_token task — uniform shape with the action task.

    Lifecycle:
    - Policy emits ``{'pred': tensor}`` (just the predicted ids).
    - :meth:`IronInferencer._extract_gt_stage_token` adds ``'gt'`` (tensor of
      ground-truth ids at the CLS position) on the train-time eval path.
    - :meth:`IronInferencer.extract_stage_tokens_strings` decodes both pred/gt
      tensors into ``list[str]`` in place and stamps ``type='stage_token'``.

    So ``pred`` (and ``gt`` when present) is a tensor pre-decode and a
    ``list[str]`` post-decode — the union annotation covers both states.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    type: Literal['stage_token'] = 'stage_token'
    pred: Union[torch.Tensor, List[str]] = Field(..., description='Predicted stage-token ids or decoded strings')
    gt: Optional[Union[torch.Tensor, List[str]]] = Field(default=None)
