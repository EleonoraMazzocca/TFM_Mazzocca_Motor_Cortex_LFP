"""One-hot condition encoding for (phase, grip, hand)."""
from __future__ import annotations

import numpy as np

from transformer_encoder.joint_embedding_data import GRIP_TO_ID, HAND_TO_ID, PHASE_NAMES

N_PHASES = len(PHASE_NAMES)
CONDITION_DIM = 7  # 3 phase + 2 grip + 2 hand
PHASE_ONEHOT = np.eye(N_PHASES, dtype=np.float32)
GRIP_ONEHOT = np.eye(2, dtype=np.float32)
HAND_ONEHOT = np.eye(2, dtype=np.float32)
ID_TO_GRIP = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND = {v: k for k, v in HAND_TO_ID.items()}
ID_TO_PHASE = {i: n for i, n in enumerate(PHASE_NAMES)}


def make_condition_vector(phase_idx: int, grip_id: int, hand_id: int) -> np.ndarray:
    """Concatenate one-hot (phase, grip, hand) into a (7,) float32 vector."""
    return np.concatenate([
        PHASE_ONEHOT[phase_idx],
        GRIP_ONEHOT[grip_id],
        HAND_ONEHOT[hand_id],
    ]).astype(np.float32, copy=False)
