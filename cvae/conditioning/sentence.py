"""Sentence/PCA condition-table lookup helpers."""
from __future__ import annotations

import numpy as np


def lookup_condition(
    phase_id: int,
    grip_id: int,
    hand_id: int,
    condition_table: np.ndarray,
    key_order: np.ndarray,
) -> np.ndarray:
    """Return the condition vector for a (phase, grip, hand) triple."""
    if not (condition_table.ndim == 2 and condition_table.shape[0] == 12):
        raise ValueError(f"condition_table must be (12, condition_dim), got {condition_table.shape}")
    if key_order.shape != (12, 3):
        raise ValueError(f"key_order must be (12, 3), got {key_order.shape}")
    matches = np.where(
        (key_order == np.array([phase_id, grip_id, hand_id])).all(axis=1)
    )[0]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 match for ({phase_id},{grip_id},{hand_id}), got {len(matches)}"
        )
    return condition_table[matches[0]].astype(np.float32)
