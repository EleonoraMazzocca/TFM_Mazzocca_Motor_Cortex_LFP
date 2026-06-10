"""Backward-compatibility shim — all constants now live in joint_embedding_data.

This file re-exports the shared constants so that existing imports like
``from transformer_encoder.data import PHASE_NAMES`` continue to work.
The canonical definitions are in ``transformer_encoder.joint_embedding_data``.
"""
from transformer_encoder.joint_embedding_data import (  # noqa: F401
    N_PHASES,
    PHASE_NAMES,
    AREA_NAMES,
    AREA_SLICES,
    N_AREAS,
    MAX_AREA_CHANNELS,
    AREA_FEATURE_DIM,
    GRIP_TO_ID,
    HAND_TO_ID,
    ANGLE_TO_ID,
    ID_TO_PHASE,
    ID_TO_GRIP,
    ID_TO_HAND,
    ID_TO_ANGLE,
)
