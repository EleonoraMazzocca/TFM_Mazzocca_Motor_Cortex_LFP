"""Data utilities for joint phase/grip/hand embedding experiments.

The joint transformer uses the same token layout for MU and broadband6:
4 areas x 96 channel slots = 384 tokens.  MU has one feature per token;
broadband6 has six band-amplitude features per token.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
# ---------------------------------------------------------------------------
# Shared constants (canonical definitions — data.py re-exports these)
# ---------------------------------------------------------------------------
N_PHASES = 3
PHASE_NAMES = ["prereach", "reach", "grasp"]
AREA_NAMES = ["PMvR", "M1", "PMdR", "PMdL"]
AREA_SLICES = (
    ("PMvR", 0, 96),
    ("M1", 96, 128),
    ("PMdR", 128, 224),
    ("PMdL", 224, 256),
)
N_AREAS = len(AREA_SLICES)
MAX_AREA_CHANNELS = 96   # largest area (PMvR, PMdR); smaller areas are zero-padded
AREA_FEATURE_DIM = MAX_AREA_CHANNELS
GRIP_TO_ID = {"power": 0, "precision": 1}
HAND_TO_ID = {"left": 0, "right": 1}
ANGLE_TO_ID = {"0": 0, "45": 1, "90": 2, "135": 3}
ID_TO_PHASE = {i: name for i, name in enumerate(PHASE_NAMES)}
ID_TO_GRIP = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND = {v: k for k, v in HAND_TO_ID.items()}
ID_TO_ANGLE = {v: k for k, v in ANGLE_TO_ID.items()}

# ---------------------------------------------------------------------------
# Joint embedding constants
# ---------------------------------------------------------------------------
INPUT_MODES = ("mu", "broadband6")
FS_BROADBAND = 2034.5
BAND_NAMES_6 = ["beta", "low_gamma", "high_gamma", "low_ripple", "high_ripple", "MU"]
BAND_DEFINITIONS_6 = [
    ("beta", 15, 30),
    ("low_gamma", 30, 70),
    ("high_gamma", 70, 100),
    ("low_ripple", 100, 150),
    ("high_ripple", 150, 200),
    ("MU", 200, 500),
]

N_TOKENS = N_AREAS * MAX_AREA_CHANNELS
AREA_SIZES = [end - start for _, start, end in AREA_SLICES]
N_REAL_CHANNELS = int(sum(AREA_SIZES))

_MU_STEM = re.compile(
    r"^(power|precision)_unimanual_(left|right)_(0|45|90|135)_degrees_mua_200_500$"
)
_BROADBAND_STEM = re.compile(
    r"^(power|precision)_unimanual_(left|right)_(0|45|90|135)_degrees$"
)


def channel_valid_mask() -> np.ndarray:
    mask = np.zeros((N_AREAS, MAX_AREA_CHANNELS), dtype=bool)
    for ai, size in enumerate(AREA_SIZES):
        mask[ai, :size] = True
    return mask


CHANNEL_VALID = channel_valid_mask()
PADDING_MASK = ~CHANNEL_VALID.reshape(-1)
AREA_IDX = np.repeat(np.arange(N_AREAS), MAX_AREA_CHANNELS).astype(np.int64)


def _filters() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        signal.butter(4, [low / (FS_BROADBAND / 2), high / (FS_BROADBAND / 2)], btype="band")
        for _, low, high in BAND_DEFINITIONS_6
    ]


_FILTERS_6 = _filters()


def load_joint_trials(data_dir: Path | str, input_mode: str) -> dict:
    """Load class-file indices. Returned trials are not phase-expanded yet."""
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")
    data_dir = Path(data_dir)
    pattern = "*_mua_200_500.npy" if input_mode == "mu" else "*_degrees.npy"
    regex = _MU_STEM if input_mode == "mu" else _BROADBAND_STEM

    file_paths, file_idx_l, trial_idx_l = [], [], []
    y_grip_l, y_hand_l, y_angle_l = [], [], []

    for path in sorted(data_dir.glob(pattern)):
        if "bimanual" in path.name:
            continue
        if input_mode == "broadband6" and "_mua_" in path.name:
            continue
        match = regex.match(path.stem)
        if match is None:
            continue
        grip_name, hand_name, angle_name = match.groups()
        grip = GRIP_TO_ID[grip_name]
        hand = HAND_TO_ID[hand_name]
        angle = ANGLE_TO_ID[angle_name]
        arr = np.load(str(path), mmap_mode="r")
        n_trials = int(arr.shape[0])
        file_idx = len(file_paths)
        file_paths.append(os.fspath(path))
        file_idx_l.append(np.full(n_trials, file_idx, dtype=np.int32))
        trial_idx_l.append(np.arange(n_trials, dtype=np.int32))
        y_grip_l.append(np.full(n_trials, grip, dtype=np.int64))
        y_hand_l.append(np.full(n_trials, hand, dtype=np.int64))
        y_angle_l.append(np.full(n_trials, angle, dtype=np.int64))

    if not file_paths:
        raise ValueError(f"No {pattern} files found in {data_dir}")

    return {
        "file_paths": np.asarray(file_paths),
        "file_idx": np.concatenate(file_idx_l),
        "trial_idx": np.concatenate(trial_idx_l),
        "y_grip": np.concatenate(y_grip_l),
        "y_hand": np.concatenate(y_hand_l),
        "y_angle": np.concatenate(y_angle_l),
        "input_mode": input_mode,
        "data_dir": os.fspath(data_dir),
    }


def phase_expand(data: dict) -> dict:
    n = len(data["y_grip"])
    phase = np.tile(np.arange(len(PHASE_NAMES), dtype=np.int64), n)
    repeat = np.repeat(np.arange(n, dtype=np.int64), len(PHASE_NAMES))
    return {
        "file_paths": data["file_paths"],
        "file_idx": data["file_idx"][repeat],
        "trial_idx": data["trial_idx"][repeat],
        "y_phase": phase,
        "y_grip": data["y_grip"][repeat],
        "y_hand": data["y_hand"][repeat],
        "y_angle": data["y_angle"][repeat],
        "y_combo": phase * 4 + data["y_grip"][repeat] * 2 + data["y_hand"][repeat],
        "input_mode": data["input_mode"],
        "data_dir": data["data_dir"],
    }


def _cache_path(cache_dir: Path, data: dict, input_mode: str) -> Path:
    sample_hasher = hashlib.md5()
    for key in ("file_idx", "trial_idx", "y_phase"):
        sample_hasher.update(np.asarray(data[key]).tobytes())
    manifest = {
        "input_mode": input_mode,
        "data_dir": data["data_dir"],
        "n_samples": int(len(data["y_phase"])),
        "sample_index_digest": sample_hasher.hexdigest(),
        "files": [
            {"name": Path(p).name, "size": Path(p).stat().st_size}
            for p in data["file_paths"]
        ],
        "bands": BAND_DEFINITIONS_6 if input_mode == "broadband6" else ["MU_200_500"],
        "layout": "4areas_x_96channels",
    }
    digest = hashlib.md5(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    return cache_dir / f"joint_{input_mode}_{digest}.npy"


def extract_mu_feature(sample: np.ndarray, phase_idx: int) -> np.ndarray:
    phase_data = sample[phase_idx].astype(np.float32, copy=False)
    amp = np.mean(np.abs(phase_data), axis=1)
    out = np.zeros((N_AREAS, MAX_AREA_CHANNELS, 1), dtype=np.float32)
    for ai, (_, start, end) in enumerate(AREA_SLICES):
        out[ai, : end - start, 0] = amp[start:end]
    return out


def extract_broadband6_feature(sample: np.ndarray, phase_idx: int) -> np.ndarray:
    phase_data = sample[phase_idx].astype(np.float32, copy=False)
    out = np.zeros((N_AREAS, MAX_AREA_CHANNELS, len(BAND_DEFINITIONS_6)), dtype=np.float32)
    band_amp = np.empty((phase_data.shape[0], len(BAND_DEFINITIONS_6)), dtype=np.float32)
    for bi, (b, a) in enumerate(_FILTERS_6):
        filtered = signal.filtfilt(b, a, phase_data, axis=1)
        band_amp[:, bi] = np.mean(np.abs(filtered), axis=1)
    for ai, (_, start, end) in enumerate(AREA_SLICES):
        out[ai, : end - start, :] = band_amp[start:end]
    return out


def extract_and_cache_features(flat_data: dict, cache_dir: Path | str) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, flat_data, flat_data["input_mode"])
    if path.exists():
        print(f"Loading cached joint features from {path}")
        return np.load(path, mmap_mode="r")

    n = len(flat_data["y_grip"])
    n_bands = 1 if flat_data["input_mode"] == "mu" else len(BAND_DEFINITIONS_6)
    features = np.zeros((n, N_AREAS, MAX_AREA_CHANNELS, n_bands), dtype=np.float32)
    file_cache: dict[str, np.ndarray] = {}
    extractor = extract_mu_feature if flat_data["input_mode"] == "mu" else extract_broadband6_feature

    print(f"Extracting {flat_data['input_mode']} joint features ({n} phase samples) ...")
    for i in range(n):
        fp = os.fspath(flat_data["file_paths"][flat_data["file_idx"][i]])
        if fp not in file_cache:
            file_cache[fp] = np.load(fp, mmap_mode="r")
        sample = file_cache[fp][int(flat_data["trial_idx"][i])]
        features[i] = extractor(sample, int(flat_data["y_phase"][i]))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", end="\r", flush=True)
    print(f"  {n}/{n} done. Saving to {path}")
    np.save(path, features)
    return np.load(path, mmap_mode="r")


def split_indices(flat_data: dict, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(flat_data["y_grip"]))
    strat = (
        flat_data["y_phase"].astype(np.int64) * 32
        + flat_data["y_grip"].astype(np.int64) * 16
        + flat_data["y_hand"].astype(np.int64) * 8
        + flat_data["y_angle"].astype(np.int64)
    )
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, shuffle=True, stratify=strat
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, shuffle=True, stratify=strat[temp_idx]
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def compute_norm_stats(features: np.ndarray, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(features[train_idx], dtype=np.float32)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return {"mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32)}


class JointEmbeddingDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        flat_data: dict,
        indices: np.ndarray,
        norm_stats: dict[str, np.ndarray] | None = None,
    ):
        self.features = features
        self.indices = np.asarray(indices)
        self.y_phase = torch.tensor(flat_data["y_phase"][self.indices], dtype=torch.long)
        self.y_grip = torch.tensor(flat_data["y_grip"][self.indices], dtype=torch.long)
        self.y_hand = torch.tensor(flat_data["y_hand"][self.indices], dtype=torch.long)
        self.y_angle = torch.tensor(flat_data["y_angle"][self.indices], dtype=torch.long)
        self.mu = None if norm_stats is None else torch.tensor(norm_stats["mu"], dtype=torch.float32)
        self.sigma = None if norm_stats is None else torch.tensor(norm_stats["sigma"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.features[int(self.indices[idx])], dtype=torch.float32)
        if self.mu is not None:
            zero_mask = x == 0.0
            x = (x - self.mu) / self.sigma
            x[zero_mask] = 0.0
        return x, self.y_phase[idx], self.y_grip[idx], self.y_hand[idx], self.y_angle[idx]


class PermutedJointDataset(Dataset):
    def __init__(self, base: JointEmbeddingDataset, rng: np.random.Generator):
        self.base = base
        self.y_phase = torch.from_numpy(rng.permutation(base.y_phase.numpy()).copy())
        self.y_grip = torch.from_numpy(rng.permutation(base.y_grip.numpy()).copy())
        self.y_hand = torch.from_numpy(rng.permutation(base.y_hand.numpy()).copy())

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, _, _, _, y_angle = self.base[idx]
        return x, self.y_phase[idx], self.y_grip[idx], self.y_hand[idx], y_angle


def subset_flat(flat_data: dict, indices: np.ndarray) -> dict:
    out = {
        "file_paths": flat_data["file_paths"],
        "input_mode": flat_data["input_mode"],
        "data_dir": flat_data["data_dir"],
    }
    for key in ("file_idx", "trial_idx", "y_phase", "y_grip", "y_hand", "y_angle", "y_combo"):
        out[key] = flat_data[key][indices]
    return out
