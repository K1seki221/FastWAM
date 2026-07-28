"""Per-class Hungarian matcher for DETR-style bbox prediction."""

import torch
from models.head.component.box_ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment


class PerClassHungarianMatcher:
    """Match predicted boxes to GT boxes independently within each class.

    Cost = cost_class * w_class + L1_cost * w_bbox + giou_cost * w_giou
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def match(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Hungarian matching for one class within one sample.

        Args:
            pred_logits: (Q, 2) logits [has_object, no_object] for Q queries.
            pred_boxes: (Q, 4) predicted boxes in xyxy [0, 1].
            gt_boxes: (G, 4) GT boxes in xyxy [0, 1]. G may be 0.

        Returns:
            matched_pred_idx: (M,) indices into pred, M <= min(Q, G).
            matched_gt_idx:   (M,) indices into GT.
        """
        num_gt = gt_boxes.shape[0]
        device = pred_logits.device

        if num_gt == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty

        prob = pred_logits.softmax(dim=-1)
        cost_cls = -prob[:, 0]  # higher prob of "has_object" → lower cost

        cost_bbox = torch.cdist(pred_boxes.float(), gt_boxes.float(), p=1)  # (Q, G)

        cost_giou = -generalized_box_iou(pred_boxes.float(), gt_boxes.float())  # (Q, G)

        C = (
            self.cost_class * cost_cls[:, None]
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        pred_idx, gt_idx = linear_sum_assignment(C.detach().cpu().numpy())
        return (
            torch.as_tensor(pred_idx, dtype=torch.long, device=device),
            torch.as_tensor(gt_idx, dtype=torch.long, device=device),
        )
