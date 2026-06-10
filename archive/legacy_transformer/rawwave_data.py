"""Raw waveform feature extraction for LFP rawwave specialist."""
from __future__ import annotations

import hashlib
import json as _json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data import AREA_SLICES, N_AREAS, GRIP_TO_ID, HAND_TO_ID, ANGLE_TO_ID

N_REAL_CHANNELS = 256
N_TIMEPOINTS    = 500
FS              = 2034.5

AREA_IDX_256 = np.array([
    ai for ai, (_, start, end) in enumerate(AREA_SLICES)
    for _ in range(end - start)
], dtype=np.int64)
# [0]*96 + [1]*32 + [2]*96 + [3]*32 = 256 values

AREA_SIZES_REAL = [end - start for _, start, end in AREA_SLICES]
# [96, 32, 96, 32]

_BB_STEM = re.compile(
    r"^(power|precision)_unimanual_(left|right)_(0|45|90|135)_degrees$"
)


def _rawwave_cache_key(data_dir: str | Path, phase_idx: int, file_paths) -> str:
    """Hash encoding data source, phase, and file manifest including sizes."""
    manifest = {
        "data_dir": str(data_dir),
        "phase_idx": phase_idx,
        "files": [
            {"name": p.name, "size": p.stat().st_size}
            for p in sorted((Path(p) for p in file_paths), key=lambda p: p.name)
        ],
    }
    h = hashlib.md5(_json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    return f"rawwave_ph{phase_idx}_{h}.npy"


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_rawwave_dataset(
    data_dir: Path | str,
    heldout_grip: int = 1,
    heldout_hand: int = 1,
    heldout_angle: int = 3,
) -> dict:
    """Scan broadband directory and return same dict structure as load_broadband_dataset."""
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

    print(f"Loading rawwave dataset from {data_dir} ...")
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
        "file_idx":   np.concatenate(file_idx_list),
        "trial_idx":  np.concatenate(trial_idx_list),
        "y_grip":     np.concatenate(y_grip_list),
        "y_hand":     np.concatenate(y_hand_list),
        "y_angle":    np.concatenate(y_angle_list),
        "is_heldout": np.concatenate(heldout_list),
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_rawwave_features(
    sample: np.ndarray,  # (n_phases, n_channels, n_timepoints) int16
    phase_idx: int,
) -> np.ndarray:
    """Return (N_REAL_CHANNELS, N_TIMEPOINTS) float32 raw waveform."""
    phase_data = sample[phase_idx].astype(np.float32)
    assert phase_data.shape == (N_REAL_CHANNELS, N_TIMEPOINTS)
    return phase_data


def extract_and_cache_rawwave(
    data: dict,
    phase_idx: int,
    cache_dir: Path,
    data_dir: str | Path,
    verbose: bool = True,
) -> np.ndarray:
    """Extract all-trial raw waveforms, save to cache, return memory-mapped array.

    Returns (n_trials, N_REAL_CHANNELS, N_TIMEPOINTS) float32.
    """
    cache_path = Path(cache_dir) / _rawwave_cache_key(
        data_dir, phase_idx, data["file_paths"]
    )
    if cache_path.exists():
        if verbose:
            print(f"Loading cached rawwave features from {cache_path}")
        return np.load(cache_path, mmap_mode="r")

    n = len(data["y_grip"])
    features = np.zeros((n, N_REAL_CHANNELS, N_TIMEPOINTS), dtype=np.float32)
    file_cache: dict[str, np.ndarray] = {}

    if verbose:
        print(f"Extracting raw waveform features for phase {phase_idx} ({n} trials) ...")
    for i in range(n):
        fp = str(data["file_paths"][data["file_idx"][i]])
        if fp not in file_cache:
            file_cache[fp] = np.load(fp, mmap_mode="r")
        features[i] = extract_rawwave_features(
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

def compute_rawwave_norm_stats(
    features: np.ndarray,   # (n_total, N_REAL_CHANNELS, N_TIMEPOINTS)
    train_idx: np.ndarray,
    norm_mode: str = "per_channel",
) -> dict:
    """Normalization stats from training trials only.

    per_channel (default): one mean/std per channel across all trials and timepoints.
        Preserves waveform shape — only removes inter-channel amplitude differences.
        Filter FFT interpretation remains valid up to a constant scale factor per channel.
    per_timepoint: one mean/std per (channel, timepoint). Removes mean waveform shape.
        Use only as a follow-up control if per_channel result is ambiguous (0.32-0.40).
    """
    train_feats = features[train_idx].astype(np.float64)  # (n_train, 256, 500)
    if norm_mode == "per_channel":
        mu    = train_feats.mean(axis=(0, 2))               # (256,)
        sigma = train_feats.std(axis=(0, 2))                # (256,)
    elif norm_mode == "per_timepoint":
        mu    = train_feats.mean(axis=0)                    # (256, 500)
        sigma = train_feats.std(axis=0)                     # (256, 500)
    else:
        raise ValueError(
            f"Unknown norm_mode {norm_mode!r}. Choose 'per_channel' or 'per_timepoint'."
        )
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return {
        "mu":        mu.astype(np.float32),
        "sigma":     sigma.astype(np.float32),
        "norm_mode": norm_mode,
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class LFPRawwaveDataset(Dataset):
    """Returns (x, y_grip, y_hand, y_angle), x: (N_REAL_CHANNELS, N_TIMEPOINTS)."""

    def __init__(
        self,
        features: np.ndarray,   # (n_total, N_REAL_CHANNELS, N_TIMEPOINTS)
        indices: np.ndarray,
        data: dict,
        norm_stats: dict | None = None,
    ):
        self.features  = features
        self.indices   = indices
        self.y_grip    = torch.tensor(data["y_grip"][indices],  dtype=torch.long)
        self.y_hand    = torch.tensor(data["y_hand"][indices],  dtype=torch.long)
        self.y_angle   = torch.tensor(data["y_angle"][indices], dtype=torch.long)
        self.norm_mode = norm_stats.get("norm_mode", "per_channel") if norm_stats else None
        if norm_stats is not None:
            self.mu    = torch.tensor(norm_stats["mu"],    dtype=torch.float32)
            self.sigma = torch.tensor(norm_stats["sigma"], dtype=torch.float32)
        else:
            self.mu = self.sigma = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.features[self.indices[i]].copy())  # (256, 500)
        if self.mu is not None:
            if self.norm_mode == "per_channel":
                x = (x - self.mu[:, None]) / self.sigma[:, None]
            else:  # per_timepoint
                x = (x - self.mu) / self.sigma
        return x, self.y_grip[i], self.y_hand[i], self.y_angle[i]


class PermutedRawwaveDataset(Dataset):
    """Wraps LFPRawwaveDataset with independently shuffled labels per head."""

    def __init__(self, base: LFPRawwaveDataset, rng: np.random.Generator):
        self._base   = base
        self.y_grip  = torch.from_numpy(rng.permutation(base.y_grip.numpy()).copy())
        self.y_hand  = torch.from_numpy(rng.permutation(base.y_hand.numpy()).copy())
        self.y_angle = torch.from_numpy(rng.permutation(base.y_angle.numpy()).copy())

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _, _, _ = self._base[i]
        return x, self.y_grip[i], self.y_hand[i], self.y_angle[i]
