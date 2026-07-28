"""DETR-style bbox auxiliary head using BQ_X1 anchor hidden states.

Reads ``vlm_output.last_hidden_state``, gathers features at BQ_X1 positions
grouped by **slot chains** (consecutive ``BQ_X1`` indices spaced by 4 for
``X1,Y1,X2,Y2`` repeats), one chain per class in fixed prompt order.
Supervision uses per-class Hungarian matching with Focal + L1 + GIoU losses.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.special_tokens import SpecialTokens
from dataset.util import get_processor
from models.head.base_head import BaseHead
from models.head.component.box_ops import generalized_box_iou
from models.head.component.matcher import PerClassHungarianMatcher
from models.head.registry import register_head
from models.utils.vlm_feature_utils import VLMInterface


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss for binary classification per class channel.

    Args:
        logits: (N, C) raw logits.
        targets: (N, C) one-hot or soft targets.
        alpha: Weighting factor for the rare class.
        gamma: Focusing parameter.

    Returns:
        Scalar mean loss.
    """
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * ce
    return loss.mean()


@register_head
class BboxDetrHead(BaseHead):
    """Lightweight DETR-style bbox auxiliary prediction head.

    Uses BQ_X1 hidden states as object queries, projects them through
    small MLPs to predict xyxy boxes and has_object / no_object logits.
    """

    NUM_CLS = 2  # [has_object, no_object]

    def __init__(self, head_config, *, backbone_hidden_dim: int, model_id: Optional[str] = None, **_kwargs):
        nn.Module.__init__(self)
        self.config = head_config
        self.model_id = model_id
        self.loss_weight = head_config.loss_weight

        head_dim = getattr(head_config, 'head_hidden_dim', 256)
        self.w_cls = getattr(head_config, 'loss_weight_cls', 1.0)
        self.w_bbox = getattr(head_config, 'loss_weight_bbox', 5.0)
        self.w_giou = getattr(head_config, 'loss_weight_giou', 2.0)
        self.focal_alpha = getattr(head_config, 'focal_alpha', 0.25)
        self.focal_gamma = getattr(head_config, 'focal_gamma', 2.0)

        self.projection = nn.Sequential(
            nn.Linear(backbone_hidden_dim, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
        )

        self.box_head = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 4),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, self.NUM_CLS),
        )

        self.matcher = PerClassHungarianMatcher(
            cost_class=self.w_cls,
            cost_bbox=self.w_bbox,
            cost_giou=self.w_giou,
        )

        self._class_names = SpecialTokens.bbox_query_class_names()
        self._bq_x1_id: int | None = None

    def _ensure_token_ids(self) -> int:
        """Lazily cache BQ_X1 token id."""
        if self._bq_x1_id is not None:
            return self._bq_x1_id
        if self.model_id is None:
            raise ValueError('BboxDetrHead requires model_id to resolve bbox query tokens')
        tokenizer = getattr(get_processor(self.model_id), 'tokenizer')
        self._bq_x1_id = int(tokenizer.convert_tokens_to_ids(SpecialTokens.BQ_X1))
        return self._bq_x1_id

    @staticmethod
    def _split_bq_x1_chains(ids: list[int], bq_x1_id: int, num_classes: int) -> list[list[int]]:
        """Split flat ``input_ids`` into per-class BQ_X1 index lists.

        Each bbox slot is ``BQ_X1,BQ_Y1,BQ_X2,BQ_Y2`` (4 ids).  ``' '.join(parts)``
        inserts a space between consecutive slots, so consecutive **row** ``BQ_X1``
        positions differ by **5** (sometimes **4** if the space merges in BPE).
        """
        positions = [i for i, t in enumerate(ids) if t == bq_x1_id]
        if not positions:
            return [[] for _ in range(num_classes)]

        chains: list[list[int]] = []
        cur = [positions[0]]
        for p in positions[1:]:
            gap = p - cur[-1]
            if gap in (4, 5):
                cur.append(p)
            else:
                chains.append(cur)
                cur = [p]
        chains.append(cur)

        out: list[list[int]] = []
        for i in range(num_classes):
            out.append(chains[i] if i < len(chains) else [])
        return out

    def _gather_bq_x1_by_class(
        self, input_ids: torch.Tensor, hidden: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """For each sample gather BQ_X1 hidden states grouped by class.

        Returns dict[class_name] -> (B, max_Q, D) with zero-padding where
        a sample has fewer queries.  Also returns a mask dict.
        """
        bq_x1_id = self._ensure_token_ids()
        B = input_ids.shape[0]
        D = hidden.shape[-1]
        n_cls = len(self._class_names)

        class_features: dict[str, list[list[int]]] = {cn: [] for cn in self._class_names}

        for b in range(B):
            ids = input_ids[b].tolist()
            chains = self._split_bq_x1_chains(ids, bq_x1_id, n_cls)
            for cls_name, positions in zip(self._class_names, chains):
                class_features[cls_name].append(positions)

        result: dict[str, torch.Tensor] = {}
        result_mask: dict[str, torch.Tensor] = {}
        for cls_name in self._class_names:
            all_pos = class_features[cls_name]
            max_q = max((len(p) for p in all_pos), default=0)
            if max_q == 0:
                result[cls_name] = hidden.new_zeros(B, 0, D)
                result_mask[cls_name] = hidden.new_zeros(B, 0, dtype=torch.bool)
                continue
            feat = hidden.new_zeros(B, max_q, D)
            mask = hidden.new_zeros(B, max_q, dtype=torch.bool)
            for b, positions in enumerate(all_pos):
                for q, pos in enumerate(positions):
                    feat[b, q] = hidden[b, pos]
                    mask[b, q] = True
            result[cls_name] = feat
            result_mask[cls_name] = mask

        return result, result_mask

    def forward(self, vlm_output: VLMInterface | None, **kwargs) -> Dict[str, Any]:
        if vlm_output is None:
            raise ValueError('BboxDetrHead requires vlm_output from the VLM backbone')
        hidden = vlm_output.last_hidden_state  # (B, S, D)
        input_ids = kwargs['input_ids']
        is_training = kwargs.get('is_training', False)

        class_feats, class_masks = self._gather_bq_x1_by_class(input_ids, hidden)

        all_pred_boxes: dict[str, torch.Tensor] = {}
        all_pred_logits: dict[str, torch.Tensor] = {}

        for cls_name in self._class_names:
            feats = class_feats[cls_name]  # (B, Q, D)
            if feats.shape[1] == 0:
                B = hidden.shape[0]
                all_pred_boxes[cls_name] = hidden.new_zeros(B, 0, 4)
                all_pred_logits[cls_name] = hidden.new_zeros(B, 0, self.NUM_CLS)
                continue
            proj = self.projection(feats)         # (B, Q, head_dim)
            boxes = self.box_head(proj).sigmoid()  # (B, Q, 4) xyxy in [0,1]
            logits = self.cls_head(proj)           # (B, Q, 2)
            all_pred_boxes[cls_name] = boxes
            all_pred_logits[cls_name] = logits

        out: Dict[str, Any] = {
            'type': 'bbox_detr',
            'pred_boxes': all_pred_boxes,
            'pred_logits': all_pred_logits,
            'class_masks': class_masks,
        }

        if is_training:
            bbox_detr_targets = kwargs.get('bbox_detr_targets', None)  # torch.Size([32, 32, 5])
            bbox_detr_num_gt = kwargs.get('bbox_detr_num_gt', None)   # torch.Size([32])
            out['gt_targets'] = bbox_detr_targets
            out['gt_num'] = bbox_detr_num_gt

        return out

    def _dummy_loss(self) -> torch.Tensor:
        """Return a zero loss that still touches every parameter for DDP gradient sync."""
        params = list(self.parameters())
        if not params:
            return torch.tensor(0.0)
        loss = params[0].sum() * 0
        for param in params[1:]:
            loss = loss + param.sum() * 0
        return loss

    def compute_loss(self, output_dict: Dict[str, Any]) -> torch.Tensor:
        pred_boxes = output_dict['pred_boxes']
        pred_logits = output_dict['pred_logits']
        class_masks = output_dict['class_masks']
        gt_targets = output_dict.get('gt_targets')
        gt_num = output_dict.get('gt_num')

        if gt_targets is None:
            return self._dummy_loss()

        B = gt_targets.shape[0]
        device = gt_targets.device
        total_focal = torch.tensor(0.0, device=device)
        total_l1 = torch.tensor(0.0, device=device)
        total_giou = torch.tensor(0.0, device=device)
        num_matched = 0

        for b in range(B):
            n_gt = int(gt_num[b].item()) if gt_num is not None else 0
            sample_gt = gt_targets[b, :n_gt]  # (n_gt, 5): [x1,y1,x2,y2, class_id]

            for cls_idx, cls_name in enumerate(self._class_names):
                p_boxes = pred_boxes[cls_name]   # (B, Q, 4)
                p_logits = pred_logits[cls_name] # (B, Q, 2)
                mask = class_masks[cls_name]     # (B, Q)

                if p_boxes.shape[1] == 0:
                    continue

                q_boxes = p_boxes[b]   # (Q, 4)
                q_logits = p_logits[b] # (Q, 2)
                q_mask = mask[b]       # (Q,)

                gt_mask = sample_gt[:, 4].long() == cls_idx
                cls_gt_boxes = sample_gt[gt_mask, :4]  # (G, 4)

                matched_pred, matched_gt = self.matcher.match(q_logits, q_boxes, cls_gt_boxes)
                n_match = matched_pred.shape[0]

                # --- Classification loss (all valid queries) ---
                valid_idx = q_mask.nonzero(as_tuple=True)[0]
                if valid_idx.numel() > 0:
                    cls_targets = torch.zeros_like(q_logits[valid_idx])
                    cls_targets[:, 1] = 1.0  # default: no_object
                    if n_match > 0:
                        local_matched_indices = []
                        for mi in matched_pred:
                            loc = (valid_idx == mi).nonzero(as_tuple=True)[0]
                            if loc.numel() > 0:
                                local_matched_indices.append(loc[0])
                        if local_matched_indices:
                            local_matched = torch.stack(local_matched_indices)
                            cls_targets[local_matched, 0] = 1.0
                            cls_targets[local_matched, 1] = 0.0
                    total_focal = total_focal + sigmoid_focal_loss(
                        q_logits[valid_idx], cls_targets,
                        alpha=self.focal_alpha, gamma=self.focal_gamma,
                    )

                # --- Box regression losses (matched queries only) ---
                if n_match > 0:
                    m_pred = q_boxes[matched_pred]
                    m_gt = cls_gt_boxes[matched_gt]
                    total_l1 = total_l1 + F.l1_loss(m_pred, m_gt)
                    giou = generalized_box_iou(m_pred, m_gt)
                    total_giou = total_giou + (1.0 - giou.diag()).mean()
                    num_matched += n_match

        n_classes = len(self._class_names)
        focal_norm = self.w_cls * total_focal / max(B * n_classes, 1)
        l1_norm = self.w_bbox * total_l1 / max(num_matched, 1)
        giou_norm = self.w_giou * total_giou / max(num_matched, 1)
        loss = focal_norm + l1_norm + giou_norm
        loss = loss + self._dummy_loss()
        print(
            f'[bbox_detr_loss] '
            f'focal={focal_norm.item():.6f}, '
            f'l1={l1_norm.item():.6f}, '
            f'giou={giou_norm.item():.6f}, '
            f'total={loss.item():.6f}, '
            f'matched={num_matched}'
        )
        return loss
