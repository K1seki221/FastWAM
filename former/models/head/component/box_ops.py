"""Box operation utilities for DETR-style bbox prediction."""

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute area of boxes in xyxy format.

    Args:
        boxes: (N, 4) tensor in [x1, y1, x2, y2] format, values in [0, 1].
    """
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Generalized IoU (GIoU) between two sets of xyxy boxes.

    Args:
        boxes1: (N, 4) xyxy boxes, values in [0, 1].
        boxes2: (M, 4) xyxy boxes, values in [0, 1].

    Returns:
        (N, M) pairwise GIoU matrix, values in [-1, 1].
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / union.clamp(min=1e-6)

    enclosing_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enclosing_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enclosing_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enclosing_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])

    enclosing_area = (enclosing_x2 - enclosing_x1).clamp(min=0) * (enclosing_y2 - enclosing_y1).clamp(min=0)

    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-6)
