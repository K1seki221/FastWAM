"""
Utility functions for embedding operations in VLM backbones.
Shared across Qwen2.5 VL and Qwen3 VL wrappers.
"""

import torch


class PlaceholderReplacementError(ValueError):
    """Exception raised when placeholder replacement validation fails."""

    pass


def replace_placeholder_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.LongTensor,
    replacement_embeddings: torch.Tensor,
    placeholder_token_id: int,
    placeholder_name: str = 'PLACEHOLDER',
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Replace placeholder token embeddings with actual embeddings using masked_scatter.

    This function finds all positions where input_ids == placeholder_token_id,
    and replaces the corresponding embeddings in inputs_embeds with replacement_embeddings.

    Args:
        inputs_embeds: Tensor of shape (batch_size, seq_len, hidden_dim)
            The input embeddings to modify.
        input_ids: Tensor of shape (batch_size, seq_len)
            The input token IDs used to locate placeholder positions.
        replacement_embeddings: Tensor of shape (batch_size, num_tokens, hidden_dim)
            The embeddings to insert at placeholder positions.
        placeholder_token_id: int
            The token ID that marks placeholder positions.
        placeholder_name: str
            Human-readable name for error messages (e.g., "PROPRIO", "CLS").
        row_mask: Optional Tensor of shape (batch_size,), bool
            When provided, only rows where ``row_mask=True`` are expected to
            contain ``num_tokens`` placeholder positions and have their
            embeddings replaced; rows where ``row_mask=False`` must contain
            zero placeholders (the replacement embedding for those rows is
            computed by the caller but discarded). Use case: cotrain
            batches mixing robot samples (with PROPRIO) and VL samples
            (without PROPRIO) — pass a mask derived from ``state_mask``.
            When ``None`` (default), every row must carry ``num_tokens``
            placeholders (legacy single-modality behavior).

    Returns:
        Modified inputs_embeds with placeholder embeddings replaced.

    Raises:
        PlaceholderReplacementError: If validation fails (count mismatch, no placeholders found).

    Example:
        >>> inputs_embeds = torch.randn(2, 10, 768)  # (batch=2, seq=10, hidden=768)
        >>> input_ids = torch.tensor([[1, 2, 100, 100, 3], [1, 100, 100, 2, 3]])  # 100 is placeholder
        >>> replacement = torch.randn(2, 2, 768)  # 2 replacements per sample
        >>> result = replace_placeholder_embeddings(
        ...     inputs_embeds, input_ids, replacement, placeholder_token_id=100, placeholder_name="PROPRIO"
        ... )
    """
    # Move replacement embeddings to same device and dtype
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    replacement_embeddings = replacement_embeddings.to(device=device, dtype=dtype)

    # Create mask for placeholder positions
    placeholder_mask = input_ids == placeholder_token_id
    num_placeholders_per_sample = placeholder_mask.sum(dim=-1)
    num_replacements = replacement_embeddings.shape[1]

    # Legacy path: no row filter → every row must carry num_replacements placeholders.
    if row_mask is None:
        if not (num_placeholders_per_sample == num_replacements).all():
            if (num_placeholders_per_sample == 0).any():
                raise PlaceholderReplacementError(
                    f'No {placeholder_name} tokens found in input_ids but replacement embeddings provided. '
                    f'Set add_{placeholder_name.lower()}_placeholder=True in BackboneAutoProcessorTransform.'
                )
            raise PlaceholderReplacementError(
                f'{placeholder_name} token count mismatch: input_ids has {num_placeholders_per_sample.tolist()} '
                f'{placeholder_name} tokens per sample, but replacement has {num_replacements}. '
                f'Ensure num_{placeholder_name.lower()}_tokens in BackboneAutoProcessorTransform matches model config.'
            )

        placeholder_mask_3d = placeholder_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(placeholder_mask_3d, replacement_embeddings)

    # Mixed-batch path: row_mask separates rows that should vs shouldn't carry the placeholder.
    # Validate counts per side.
    row_mask = row_mask.to(device=device, dtype=torch.bool)
    expected_counts = num_placeholders_per_sample[row_mask]
    unexpected_counts = num_placeholders_per_sample[~row_mask]
    if (expected_counts != num_replacements).any():
        raise PlaceholderReplacementError(
            f'{placeholder_name} token count mismatch on rows where row_mask=True: '
            f'counts={expected_counts.tolist()}, expected={num_replacements}.'
        )
    if (unexpected_counts != 0).any():
        raise PlaceholderReplacementError(
            f'{placeholder_name} placeholder(s) found on rows where row_mask=False '
            f'(counts={unexpected_counts.tolist()}). Those rows must carry zero placeholders.'
        )

    # No-op write path when every row is masked out — but keep
    # ``replacement_embeddings`` in the autograd graph. Returning
    # ``inputs_embeds`` directly detaches the upstream encoder that
    # produced ``replacement_embeddings`` (e.g. ``proprio_input_encoder``):
    # under DDP ``find_unused_parameters=False`` that encoder's params then
    # never receive a gradient on this batch, while sibling ranks with at
    # least one ``row_mask=True`` row do reduce them — desynced bucket
    # firing surfaces as ``Detected mismatch between collectives on ranks``
    # at the next ``all_reduce``. A 0-weighted sum is a numerical no-op
    # (``x + 0 == x``) but preserves the connection so the encoder's
    # params are seen as "used" by every rank.
    if not row_mask.any():
        return inputs_embeds + 0.0 * replacement_embeddings.sum()

    # ``masked_scatter`` reads source elements in flat order, so it's not row-aware:
    # if row 0 has 0 placeholders and row 1 has 1, the first D values of the source
    # (intended for row 0) would be written into row 1's placeholder slot. To avoid
    # that scramble, filter to rows with row_mask=True, scatter, and write back.
    sub_embeds = inputs_embeds[row_mask]
    sub_input_ids = input_ids[row_mask]
    sub_repl = replacement_embeddings[row_mask]
    sub_mask_3d = (sub_input_ids == placeholder_token_id).unsqueeze(-1).expand_as(sub_embeds)
    sub_embeds = sub_embeds.masked_scatter(sub_mask_3d, sub_repl)

    result = inputs_embeds.clone()
    result[row_mask] = sub_embeds
    return result
