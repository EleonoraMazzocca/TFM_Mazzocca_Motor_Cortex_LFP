"""Multi-band spectral feature extraction for LFP multiband specialist."""
from __future__ import annotations

import hashlib
import json as _json
import os
import re
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset

from data import AREA_SLICES, MAX_AREA_CHANNELS, N_AREAS
from data import GRIP_TO_ID, HAND_TO_ID, ANGLE_TO_ID

# Broadband data: 500 timepoints at 2034.5 Hz ≈ 245.7ms per phase
FS = 2034.5

N_BANDS = 6
BAND_NAMES = ["beta", "low_gamma", "high_gamma", "low_ripple", "high_ripple", "MU"]
BAND_DEFINITIONS = [
    ("beta",         15,  30),
    ("low_gamma",    30,  70),
    ("high_gamma",   70, 100),
    ("low_ripple",  100, 150),
    ("high_ripple", 150, 200),
    ("MU",          200, 500),
]

N_TOKENS = N_AREAS * MAX_AREA_CHANNELS              # 4 × 96 = 384
AREA_SIZES = [end - start for _, start, end in AREA_SLICES]  # [96, 32, 96, 32]


def _build_padding_mask() -> np.ndarray:
    mask = np.zeros(N_TOKENS, dtype=bool)
    for a, area_size in enumerate(AREA_SIZES):
        if area_size < MAX_AREA_CHANNELS:
            pad_start = a * MAX_AREA_CHANNELS + area_size
            mask[pad_start:(a + 1) * MAX_AREA_CHANNELS] = True
    return mask


# (384,) — True = padded token (M1 and PMdL zero-padded slots)
PADDING_MASK = _build_padding_mask()

# (384,) — area index 0-3 for each token
AREA_IDX = np.repeat(np.arange(N_AREAS), MAX_AREA_CHANNELS).astype(np.int64)

# (N_AREAS, MAX_AREA_CHANNELS) — True for real (non-padded) channels
CHANNEL_VALID = np.zeros((N_AREAS, MAX_AREA_CHANNELS), dtype=bool)
for _a, _sz in enumerate(AREA_SIZES):
    CHANNEL_VALID[_a, :_sz] = True
N_VALID_CHANNELS: int = int(CHANNEL_VALID.sum())   # 256

# Pre-compute filter coefficients once at import time
_FILTERS: list[tuple[np.ndarray, np.ndarray]] = [
    signal.butter(4, [low / (FS / 2), high / (FS / 2)], btype="band")
    for _, low, high in BAND_DEFINITIONS
]


def _cache_key(data_dir: str, phase_idx: int, fs: float, bands: list,
               file_paths=None) -> str:
    """Stable hash-based filename encoding data source, phase, FS, band definitions, and file sizes."""
    manifest: dict = {"data_dir": str(data_dir), "phase_idx": phase_idx, "fs": fs, "bands": bands}
    if file_paths is not None:
        manifest["files"] = sorted([
            {"name": Path(p).name, "size": Path(p).stat().st_size}
            for p in file_paths
        ])
    h = hashlib.md5(_json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    return f"multiband_ph{phase_idx}_{h}.npy"

# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

_BB_STEM = re.compile(
    r"^(power|precision)_unimanual_(left|right)_(0|45|90|135)_degrees$"
)


def load_broadband_dataset(
    data_dir: Path | str,
    heldout_grip: int = 1,
    heldout_hand: int = 1,
    heldout_angle: int = 3,
) -> dict:
    """Scan broadband directory and return same dict structure as data.load_dataset."""
    data_dir = Path(data_dir)
    file_paths: list[Path] = []
    grips, hands, angles, is_heldout_flags = [], [], [], []

    for path in sorted(data_dir.glob("*_degrees.npy")):
        if "bimanual" in path.name:
            continue
        m = _BB_STEM.match(path.stem)
        if m is None:
            continue
        gn, hn, an = m.groups()
        g, h, a = GRIP_TO_ID[gn], HAND_TO_ID[hn], ANGLE_TO_ID[an]
        file_paths.append(path)
        grips.append(g)
        hands.append(h)
        angles.append(a)
        is_heldout_flags.append(g == heldout_grip and h == heldout_hand and a == heldout_angle)

    if not file_paths:
        raise ValueError(f"No *_degrees.npy files found in {data_dir}")

    file_paths_arr = np.array([os.fspath(p) for p in file_paths])
    trial_idx_list, file_idx_list = [], []
    y_grip_list, y_hand_list, y_angle_list, heldout_list = [], [], [], []

    print(f"Loading broadband dataset from {data_dir} ...")
    for fi, (path, g, h, a, is_ho) in enumerate(
        zip(file_paths, grips, hands, angles, is_heldout_flags)
    ):
        arr = np.load(path, mmap_mode="r")
        n_trials = arr.shape[0]
        print(f"  {path.name}: {n_trials} trials")
        trial_idx_list.append(np.arange(n_trials, dtype=np.int32))
        file_idx_list.append(np.full(n_trials, fi, dtype=np.int16))
        y_grip_list.append(np.full(n_trials, g, dtype=np.int64))
        y_hand_list.append(np.full(n_trials, h, dtype=np.int64))
        y_angle_list.append(np.full(n_trials, a, dtype=np.int64))
        heldout_list.append(np.full(n_trials, is_ho, dtype=bool))

    return {
        "file_paths": file_paths_arr,
        "n_channels": np.array(256, dtype=np.int64),
        "file_idx": np.concatenate(file_idx_list),
        "trial_idx": np.concatenate(trial_idx_list),
        "y_grip": np.concatenate(y_grip_list),
        "y_hand": np.concatenate(y_hand_list),
        "y_angle": np.concatenate(y_angle_list),
        "is_heldout": np.concatenate(heldout_list),
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_multiband_features(
    sample: np.ndarray,  # (n_phases, n_channels, n_timepoints) int16
    phase_idx: int,
) -> np.ndarray:
    """Return (N_AREAS, MAX_AREA_CHANNELS, N_BANDS) float32 spectral amplitudes."""
    phase_data = sample[phase_idx].astype(np.float32)  # (256, 500)
    n_ch = phase_data.shape[0]

    multiband = np.empty((n_ch, N_BANDS), dtype=np.float32)
    for bi, (b, a) in enumerate(_FILTERS):
        filtered = signal.filtfilt(b, a, phase_data, axis=1)
        multiband[:, bi] = np.mean(np.abs(filtered), axis=1)

    result = np.zeros((N_AREAS, MAX_AREA_CHANNELS, N_BANDS), dtype=np.float32)
    for ai, (_, start, end) in enumerate(AREA_SLICES):
        result[ai, : end - start] = multiband[start:end]

    return result


def extract_and_cache_features(
    data: dict,
    phase_idx: int,
    cache_dir: Path,
    data_dir: str | Path,
    verbose: bool = True,
) -> np.ndarray:
    """Extract all-trial features, save to cache, return memory-mapped array.

    Cache filename encodes data_dir, phase_idx, FS, band definitions, and file sizes so
    stale caches from different configs or replaced files are never reused.

    Returns (n_trials, N_AREAS, MAX_AREA_CHANNELS, N_BANDS) float32.
    """
    cache_path = Path(cache_dir) / _cache_key(
        str(data_dir), phase_idx, FS,
        [(n, lo, hi) for n, lo, hi in BAND_DEFINITIONS],
        data["file_paths"],
    )
    if cache_path.exists():
        if verbose:
            print(f"Loading cached features from {cache_path}")
        return np.load(cache_path, mmap_mode="r")

    n = len(data["y_grip"])
    features = np.zeros((n, N_AREAS, MAX_AREA_CHANNELS, N_BANDS), dtype=np.float32)
    file_cache: dict[str, np.ndarray] = {}

    if verbose:
        print(f"Extracting multiband features for phase {phase_idx} ({n} trials) ...")
    for i in range(n):
        fp = str(data["file_paths"][data["file_idx"][i]])
        if fp not in file_cache:
            file_cache[fp] = np.load(fp, mmap_mode="r")
        features[i] = extract_multiband_features(
            file_cache[fp][int(data["trial_idx"][i])], phase_idx
        )
        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", end="\r", flush=True)

    if verbose:
        print(f"  {n}/{n} done. Saving to {cache_path} ...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), features)
    return np.load(cache_path, mmap_mode="r")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_multiband_norm_stats(
    features: np.ndarray,   # (n_total, N_AREAS, MAX_AREA_CHANNELS, N_BANDS)
    train_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-(area, channel, band) z-score stats computed from training trials only."""
    train_feats = features[train_idx].astype(np.float64)
    mu = train_feats.mean(axis=0).astype(np.float32)
    sigma = train_feats.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma).astype(np.float32)
    return {"mu": mu, "sigma": sigma}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class LFPMultibandDataset(Dataset):
    """Returns (x, y_grip, y_hand, y_angle), x: (N_AREAS, MAX_AREA_CHANNELS, N_BANDS)."""

    def __init__(
        self,
        features: np.ndarray,   # (n_total, N_AREAS, MAX_AREA_CHANNELS, N_BANDS)
        indices: np.ndarray,    # which rows of features to use
        data: dict,
        norm_stats: dict[str, np.ndarray] | None = None,
    ):
        self.features = features
        self.indices = indices
        self.y_grip  = torch.tensor(data["y_grip"][indices],  dtype=torch.long)
        self.y_hand  = torch.tensor(data["y_hand"][indices],  dtype=torch.long)
        self.y_angle = torch.tensor(data["y_angle"][indices], dtype=torch.long)
        if norm_stats is not None:
            self.mu    = torch.tensor(norm_stats["mu"],    dtype=torch.float32)
            self.sigma = torch.tensor(norm_stats["sigma"], dtype=torch.float32)
        else:
            self.mu = self.sigma = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.features[self.indices[i]].copy())
        if self.mu is not None:
            zero_mask = (x == 0.0)
            x = (x - self.mu) / self.sigma
            x[zero_mask] = 0.0
        return x, self.y_grip[i], self.y_hand[i], self.y_angle[i]


class PermutedMultibandDataset(Dataset):
    """Wraps LFPMultibandDataset with independently shuffled labels per head."""

    def __init__(self, base: LFPMultibandDataset, rng: np.random.Generator):
        self._base = base
        self.y_grip  = torch.from_numpy(rng.permutation(base.y_grip.numpy()).copy())
        self.y_hand  = torch.from_numpy(rng.permutation(base.y_hand.numpy()).copy())
        self.y_angle = torch.from_numpy(rng.permutation(base.y_angle.numpy()).copy())

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _, _, _ = self._base[i]
        return x, self.y_grip[i], self.y_hand[i], self.y_angle[i]
