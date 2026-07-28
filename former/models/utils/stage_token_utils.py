import torch

from dataset.util import get_token_insertion_idx


def extract_stage_token_from_token_ids(
    token_ids: torch.Tensor,
    input_ids: torch.Tensor,
    model_id: str,
) -> torch.Tensor | None:
    """Greedily predicted id at each row's ``<|CLS|>`` column, shape (B,).

    Returns ``None`` for batches without exactly one ``<|CLS|>`` per row
    (e.g. VL/VQA eval samples whose ``input_ids`` carry no CLS) — mirrors
    the symmetric guard in ``IronInferencer._extract_gt_stage_token``.

    Call site: :meth:`IronVLMModule.forward` (single ``argmax(logits)`` per pass).
    """
    if token_ids.ndim != 2:
        raise ValueError(f'token_ids must be (B, L), got {tuple(token_ids.shape)}')
    if input_ids.ndim != 2:
        raise ValueError(f'input_ids must be (B, L), got {tuple(input_ids.shape)}')
    if token_ids.shape[0] != input_ids.shape[0]:
        raise ValueError('token_ids and input_ids batch size mismatch')

    cls_insertion_idx = get_token_insertion_idx(input_ids, '<|CLS|>', model_id)
    batch_size = token_ids.size(0)
    if cls_insertion_idx.shape[0] != batch_size:
        return None
    batch_idx = torch.arange(batch_size, device=token_ids.device)
    return token_ids[batch_idx, cls_insertion_idx]
