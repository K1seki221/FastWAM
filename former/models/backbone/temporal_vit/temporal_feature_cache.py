"""FIFO per-layer cache of pre-temporal-attention ViT features for sequential
single-sample inference.

``slot_offsets`` picks which cached frames to retrieve (inference-rate units,
freq-scaled when deploy/training rates differ). ``time_offsets`` are the
canonical training-rate labels reported back to the encoder, so the model
always sees the integer values it was trained on. Equal when rates match.

V1: single sample only (B=1); reset() at episode boundaries.
"""

from collections import deque

import torch


class TemporalFeatureCache:
    """FIFO cache for ViT intermediate layer features, keyed by temporal-layer index."""

    def __init__(
        self,
        max_frames: int,
        num_temporal_layers: int = 3,
        slot_offsets: list[int] | None = None,
        time_offsets: list[int] | None = None,
    ):
        if slot_offsets is not None and time_offsets is not None:
            if len(slot_offsets) != len(time_offsets):
                raise ValueError(
                    f'slot_offsets and time_offsets must have the same length, '
                    f'got {len(slot_offsets)} vs {len(time_offsets)}',
                )
        self.max_frames = max_frames
        self.num_temporal_layers = num_temporal_layers
        self.slot_offsets = slot_offsets
        # Default time labels to slot offsets — they match when rates do.
        self.time_offsets = time_offsets if time_offsets is not None else slot_offsets
        self.cache: list[deque[torch.Tensor]] = [
            deque(maxlen=max_frames) for _ in range(num_temporal_layers)
        ]

    def push(self, layer_idx: int, features: torch.Tensor) -> None:
        """Cache pre-temporal-attention features (N, D). Detached to free the graph."""
        self.cache[layer_idx].append(features.detach())

    def get_history(self, layer_idx: int) -> torch.Tensor | None:
        """Return (K_hist, N, D) stacked oldest → newest, or None if no valid history."""
        cache = self.cache[layer_idx]
        if len(cache) == 0:
            return None

        if self.slot_offsets is None:
            return torch.stack(list(cache), dim=0)

        selected = []
        for slot in self.slot_offsets:
            idx = len(cache) + slot
            if 0 <= idx < len(cache):
                selected.append(cache[idx])
        if not selected:
            return None
        return torch.stack(selected, dim=0)

    def get_history_time_offsets(self, layer_idx: int) -> list[int]:
        """Time labels for slots currently available in the cache."""
        cache_len = len(self.cache[layer_idx])
        if self.slot_offsets is None:
            return list(range(-cache_len, 0))
        time_labels = self.time_offsets if self.time_offsets is not None else self.slot_offsets
        return [
            time_labels[i]
            for i, slot in enumerate(self.slot_offsets)
            if 0 <= cache_len + slot < cache_len
        ]

    def reset(self) -> None:
        """Drop all cached features — call at episode boundaries."""
        for c in self.cache:
            c.clear()

    @property
    def num_cached_frames(self) -> int:
        return len(self.cache[0]) if self.cache else 0
