"""One-hot instruction encoder: (grip_id, hand_id, angle_id) -> float32 (8,)

Layout of the 8-dimensional vector:
  positions 0-1  : grip  one-hot  (0=power,     1=precision)
  positions 2-3  : hand  one-hot  (0=left,       1=right)
  positions 4-7  : angle one-hot  (0=0deg, 1=45deg, 2=90deg, 3=135deg)

All vectors are produced programmatically from integer IDs — never hand-crafted.
"""
from __future__ import annotations

import numpy as np

INSTR_DIM = 8  # 2 (grip) + 2 (hand) + 4 (angle)


def encode_onehot(grip: int, hand: int, angle: int) -> np.ndarray:
    """Return shape (8,) float32 one-hot for a single trial label."""
    v = np.zeros(INSTR_DIM, dtype=np.float32)
    v[grip] = 1.0
    v[2 + hand] = 1.0
    v[4 + angle] = 1.0
    return v


def encode_batch(
    grips: np.ndarray,
    hands: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    """Return shape (N, 8) float32 for N trials."""
    n = len(grips)
    out = np.zeros((n, INSTR_DIM), dtype=np.float32)
    out[np.arange(n), grips] = 1.0
    out[np.arange(n), 2 + hands] = 1.0
    out[np.arange(n), 4 + angles] = 1.0
    return out


def zero_instruction(n: int) -> np.ndarray:
    """Return shape (N, 8) all-zeros — used for zeroing ablation."""
    return np.zeros((n, INSTR_DIM), dtype=np.float32)


def wrong_instruction(grip: int, hand: int, angle: int) -> np.ndarray:
    """Return a maximally wrong instruction: flip grip and hand, rotate angle 180 deg."""
    return encode_onehot(1 - grip, 1 - hand, (angle + 2) % 4)
