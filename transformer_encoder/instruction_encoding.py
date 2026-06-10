"""Instruction encoding: class_name string → float32 vector.

Supports four encoding schemes:
  onehot  (8-dim)  — structured binary, one bit per attribute value
  bow     (9-dim)  — bag-of-words over a 9-word vocabulary
  minilm  (384-dim) — sentence embeddings from all-MiniLM-L6-v2
  none    (0-dim)  — no instruction; placeholder zero tensor

All public functions take class_name strings ("precision_right_135"),
never raw integer IDs. This keeps encoding logic independent of data.py
label conventions.
"""
from __future__ import annotations

import numpy as np

# Attribute vocabularies (order defines bit positions for onehot/bow)
GRIP  = ["power", "precision"]        # onehot positions 0-1
HAND  = ["left", "right"]             # onehot positions 2-3
ANGLE = ["0", "45", "90", "135"]      # onehot positions 4-7

ONEHOT_DIM = 8    # 2 + 2 + 4
BOW_DIM    = 9    # power precision left right 0 45 90 135 degrees
MINILM_DIM = 384

# One canonical sentence per (grip, hand, angle) combination.
MINILM_TEMPLATE: dict[tuple[str, str, str], str] = {
    ("power",     "left",  "0"):   "perform a power grip with the left hand at 0 degrees",
    ("power",     "left",  "45"):  "perform a power grip with the left hand at 45 degrees",
    ("power",     "left",  "90"):  "perform a power grip with the left hand at 90 degrees",
    ("power",     "left",  "135"): "perform a power grip with the left hand at 135 degrees",
    ("power",     "right", "0"):   "perform a power grip with the right hand at 0 degrees",
    ("power",     "right", "45"):  "perform a power grip with the right hand at 45 degrees",
    ("power",     "right", "90"):  "perform a power grip with the right hand at 90 degrees",
    ("power",     "right", "135"): "perform a power grip with the right hand at 135 degrees",
    ("precision", "left",  "0"):   "reach toward the target with the left hand at 0 degrees",
    ("precision", "left",  "45"):  "reach toward the target with the left hand at 45 degrees",
    ("precision", "left",  "90"):  "reach toward the target with the left hand at 90 degrees",
    ("precision", "left",  "135"): "reach toward the target with the left hand at 135 degrees",
    ("precision", "right", "0"):   "reach toward the target with the right hand at 0 degrees",
    ("precision", "right", "45"):  "reach toward the target with the right hand at 45 degrees",
    ("precision", "right", "90"):  "reach toward the target with the right hand at 90 degrees",
    ("precision", "right", "135"): "reach toward the target with the right hand at 135 degrees",
}

# Lazy singleton for the MiniLM model (loaded once on first call)
_MINILM_MODEL = None


def parse_class_name(class_name: str) -> tuple[str, str, str]:
    """Parse "precision_right_135" → ("precision", "right", "135").

    Raises ValueError on unrecognized input.
    """
    parts = class_name.split("_")
    if len(parts) != 3:
        raise ValueError(f"Expected 'grip_hand_angle' format, got: {class_name!r}")
    grip, hand, angle = parts
    if grip not in GRIP:
        raise ValueError(f"Unknown grip {grip!r} in {class_name!r}")
    if hand not in HAND:
        raise ValueError(f"Unknown hand {hand!r} in {class_name!r}")
    if angle not in ANGLE:
        raise ValueError(f"Unknown angle {angle!r} in {class_name!r}")
    return grip, hand, angle


def encode_onehot(class_name: str) -> np.ndarray:
    """Return shape (8,) float32 one-hot vector.

    Acceptance tests:
        encode_onehot("power_left_0").tolist()       == [1,0,1,0,1,0,0,0]
        encode_onehot("precision_right_135").tolist() == [0,1,0,1,0,0,0,1]
    """
    grip, hand, angle = parse_class_name(class_name)
    v = np.zeros(ONEHOT_DIM, dtype=np.float32)
    v[GRIP.index(grip)] = 1.0
    v[2 + HAND.index(hand)] = 1.0
    v[4 + ANGLE.index(angle)] = 1.0
    return v


def encode_bow(class_name: str) -> np.ndarray:
    """Return shape (9,) float32 bag-of-words vector.

    Vocabulary order: [power, precision, left, right, 0, 45, 90, 135, degrees].
    "degrees" is always set (position 8 always 1).

    Acceptance tests:
        v = encode_bow("precision_right_135")
        v[1] == 1  # precision
        v[3] == 1  # right
        v[7] == 1  # 135
        v[8] == 1  # degrees
        v.sum() == 4
    """
    VOCAB = ["power", "precision", "left", "right", "0", "45", "90", "135", "degrees"]
    grip, hand, angle = parse_class_name(class_name)
    v = np.zeros(BOW_DIM, dtype=np.float32)
    v[VOCAB.index(grip)] = 1.0
    v[VOCAB.index(hand)] = 1.0
    v[VOCAB.index(angle)] = 1.0
    v[8] = 1.0  # "degrees" always present
    return v


def encode_minilm(
    sentences: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Encode sentences with MiniLM. Returns shape (N, 384) float32.

    The model is lazy-loaded on first call and cached as a module singleton.
    """
    global _MINILM_MODEL
    if _MINILM_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MINILM_MODEL = SentenceTransformer(model_name)
    embs = _MINILM_MODEL.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embs.astype(np.float32)


def get_instruction_dim(encoding: str) -> int:
    """Return embedding dimension for the given encoding.

    encoding ∈ {"onehot", "bow", "minilm", "none"}
    """
    dims = {"onehot": ONEHOT_DIM, "bow": BOW_DIM, "minilm": MINILM_DIM, "none": 0}
    if encoding not in dims:
        raise ValueError(f"Unknown encoding {encoding!r}; choices: {list(dims)}")
    return dims[encoding]


def build_instruction_matrix(
    class_names: list[str],
    encoding: str,
    sentences: list[str] | None = None,
) -> np.ndarray:
    """Build instruction matrix for a dataset split.

    Returns shape (N, instruction_dim) float32.
    For "minilm", sentences must be provided (one natural-language sentence per sample).
    For "onehot"/"bow", class_names are sufficient.
    For "none", returns shape (N, 0).
    """
    n = len(class_names)
    if encoding == "none":
        return np.zeros((n, 0), dtype=np.float32)
    if encoding == "onehot":
        return np.stack([encode_onehot(cn) for cn in class_names], axis=0)
    if encoding == "bow":
        return np.stack([encode_bow(cn) for cn in class_names], axis=0)
    if encoding == "minilm":
        if sentences is None:
            raise ValueError("sentences must be provided for MiniLM encoding")
        return encode_minilm(sentences)
    raise ValueError(f"Unknown encoding {encoding!r}")
