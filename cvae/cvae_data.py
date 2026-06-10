"""Data loading and preparation utilities for the cVAE experiment.

Two feature modes:
  spectral  — per-channel MU amplitude (256,) per trial per phase
  raw       — flat waveform (256 × 500 = 128 000,) per trial per phase

The held-out condition is a (phase, grip, hand) triplet across ALL angle variants.
This differs from the transformer's held-out which fixes grip+hand+angle (not phase).
"""
from __future__ import annotations

import re
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Path setup — import constants from the transformer module
# ---------------------------------------------------------------------------
from transformer_encoder.data import AREA_SLICES, N_AREAS, GRIP_TO_ID, HAND_TO_ID, ANGLE_TO_ID, PHASE_NAMES
from transformer_encoder.specialist_data import extract_phase_area_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_REAL_CHANNELS = 256           # 96 PMvR + 32 M1 + 96 PMdR + 32 PMdL
N_TIMEPOINTS    = 500           # samples per phase epoch
CONDITION_DIM   = 7             # 3 (phase) + 2 (grip) + 2 (hand)
N_PHASES        = len(PHASE_NAMES)  # 3

# Fixed one-hot encoders — never modified
PHASE_ONEHOT = np.eye(N_PHASES, dtype=np.float32)  # (3, 3)
GRIP_ONEHOT  = np.eye(2,        dtype=np.float32)  # (2, 2)
HAND_ONEHOT  = np.eye(2,        dtype=np.float32)  # (2, 2)

ID_TO_GRIP  = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND  = {v: k for k, v in HAND_TO_ID.items()}
ID_TO_PHASE = {i: n for i, n in enumerate(PHASE_NAMES)}

# Regex matching MU class file stems
_CLASS_STEM = re.compile(
    r"^(power|precision)_unimanual_(left|right)_(0|45|90|135)_degrees_mua_200_500$"
)


# ---------------------------------------------------------------------------
# Condition vector
# ---------------------------------------------------------------------------

def make_condition_vector(phase_idx: int, grip_id: int, hand_id: int) -> np.ndarray:
    """Concatenate one-hot (phase, grip, hand) → (7,) float32 condition vector."""
    return np.concatenate([
        PHASE_ONEHOT[phase_idx],
        GRIP_ONEHOT[grip_id],
        HAND_ONEHOT[hand_id],
    ])


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_cvae_features(
    sample: np.ndarray,  # (n_phases, n_channels, n_timepoints)
    phase_idx: int,
    mode: str = "spectral",
) -> np.ndarray:
    """Extract per-trial per-phase features for the cVAE.

    spectral → (256,)   per-channel MU amplitude, using specialist_data.py
    raw      → (128000,) full waveform flattened: 256 channels × 500 timepoints
    """
    phase_data = sample[phase_idx].astype(np.float32, copy=False)  # (n_ch, n_tp)

    if mode == "spectral":
        area_features = extract_phase_area_features(
            sample, phase_idx, n_bins=1, use_per_channel=True
        )
        real_channels = [
            area_features[area_idx, :end - start]
            for area_idx, (_, start, end) in enumerate(AREA_SLICES)
        ]
        return np.concatenate(real_channels).astype(np.float32, copy=False)  # (256,)
    if mode == "raw":
        return phase_data[:N_REAL_CHANNELS].flatten()  # (128000,)
    raise ValueError(f"Unknown mode '{mode}'. Use 'spectral' or 'raw'.")


# ---------------------------------------------------------------------------
# Dataset loading (index-only, lazy file access)
# ---------------------------------------------------------------------------

def load_cvae_dataset(
    data_dir: str | Path,
    heldout_phase_idx: int = 2,   # default: grasp
    heldout_grip_id:   int = 1,   # default: precision
    heldout_hand_id:   int = 1,   # default: right
    broadband_data_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Scan data_dir for all 16 MU class files and build a flat index.

    For each file (grip, hand, angle) × 3 phases → 48 conditions.
    is_heldout is True when phase==heldout_phase AND grip==heldout_grip
    AND hand==heldout_hand (all angle variants held out together).

    Returns a dict with keys:
      file_paths  object array of unique file path strings (index by file_idx)
      file_idx    (N,) int16
      trial_idx   (N,) int32 — trial row within file
      phase_idx   (N,) int8  — 0=prereach 1=reach 2=grasp
      y_grip      (N,) int64
      y_hand      (N,) int64
      y_angle     (N,) int64
      condition   (N, 7) float32 — one-hot (phase, grip, hand)
      is_heldout  (N,) bool
    """
    data_dir = Path(data_dir)
    broadband_data_dir = Path(broadband_data_dir) if broadband_data_dir else None
    print(f"Loading cVAE dataset from {data_dir} ...")
    if broadband_data_dir is not None:
        print(f"  Raw waveform source: {broadband_data_dir}")

    file_registry: list[str] = []
    raw_file_registry: list[str] = []
    lists: dict[str, list] = {k: [] for k in
        ("file_idx", "trial_idx", "phase_idx", "y_grip", "y_hand", "y_angle",
         "condition", "is_heldout")}

    for path in sorted(data_dir.glob("*_mua_200_500.npy")):
        if "bimanual" in path.name:
            continue
        m = _CLASS_STEM.match(path.stem)
        if m is None:
            continue

        grip_name, hand_name, angle_name = m.groups()
        grip_id  = GRIP_TO_ID[grip_name]
        hand_id  = HAND_TO_ID[hand_name]
        angle_id = ANGLE_TO_ID[angle_name]

        arr = np.load(str(path), mmap_mode="r")  # (n_trials, n_phases, n_ch, n_tp)
        n_trials = arr.shape[0]
        f_idx = len(file_registry)
        file_registry.append(str(path))
        if broadband_data_dir is not None:
            raw_name = path.name.replace("_mua_200_500.npy", ".npy")
            raw_path = broadband_data_dir / raw_name
            if not raw_path.exists():
                raise FileNotFoundError(
                    f"Missing raw waveform file for {path.name}: expected {raw_path}"
                )
            raw_arr = np.load(str(raw_path), mmap_mode="r")
            if raw_arr.shape[0] != n_trials:
                raise ValueError(
                    f"Trial count mismatch for {path.name}: MU has {n_trials}, raw has {raw_arr.shape[0]}"
                )
            raw_file_registry.append(str(raw_path))
        print(f"  {path.name}: {n_trials} trials × {N_PHASES} phases")

        for phase in range(N_PHASES):
            heldout = (
                phase    == heldout_phase_idx and
                grip_id  == heldout_grip_id   and
                hand_id  == heldout_hand_id
            )
            cond = make_condition_vector(phase, grip_id, hand_id)

            lists["file_idx"].append(np.full(n_trials, f_idx,      dtype=np.int16))
            lists["trial_idx"].append(np.arange(n_trials,           dtype=np.int32))
            lists["phase_idx"].append(np.full(n_trials, phase,      dtype=np.int8))
            lists["y_grip"].append(np.full(n_trials, grip_id,       dtype=np.int64))
            lists["y_hand"].append(np.full(n_trials, hand_id,       dtype=np.int64))
            lists["y_angle"].append(np.full(n_trials, angle_id,     dtype=np.int64))
            lists["condition"].append(np.tile(cond, (n_trials, 1)))
            lists["is_heldout"].append(np.full(n_trials, heldout,   dtype=bool))

    if not file_registry:
        raise ValueError(f"No *_mua_200_500.npy files found in {data_dir}")

    dataset = {"file_paths": np.array(file_registry, dtype=object)}
    if raw_file_registry:
        dataset["raw_file_paths"] = np.array(raw_file_registry, dtype=object)
    for key in ("file_idx", "trial_idx", "phase_idx", "y_grip", "y_hand",
                "y_angle", "is_heldout"):
        dataset[key] = np.concatenate(lists[key])
    dataset["condition"] = np.concatenate(lists["condition"], axis=0)

    n_total   = len(dataset["y_grip"])
    n_heldout = dataset["is_heldout"].sum()
    print(f"  Total: {n_total} samples  ({n_heldout} held-out)")
    return dataset


def _subset(dataset: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Return a view of the dataset restricted to the given index array."""
    result = {"file_paths": dataset["file_paths"]}
    if "raw_file_paths" in dataset:
        result["raw_file_paths"] = dataset["raw_file_paths"]
    for key in ("file_idx", "trial_idx", "phase_idx", "y_grip", "y_hand",
                "y_angle", "condition", "is_heldout"):
        result[key] = dataset[key][idx]
    return result


def split_cvae_dataset(
    dataset: dict[str, np.ndarray],
    train_frac: float = 0.85,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """Split into train (train_frac), val (1-train_frac), and heldout test sets.

    Heldout samples always go entirely to test.
    Remaining samples are stratified by (phase, grip, hand, angle) combo.
    Returns (train_dataset, val_dataset, heldout_test_dataset).
    """
    all_idx      = np.arange(len(dataset["y_grip"]))
    heldout_mask = dataset["is_heldout"]
    heldout_idx  = all_idx[heldout_mask]
    remaining    = all_idx[~heldout_mask]

    if len(heldout_idx) == 0:
        raise ValueError("No held-out trials found — check (phase, grip, hand) parameters.")

    # Stratify by (phase, grip, hand, angle) — unique combo key in [0, 47]
    strat = (
        dataset["phase_idx"][remaining].astype(np.int32) * 8
        + dataset["y_grip"][remaining].astype(np.int32) * 4
        + dataset["y_hand"][remaining].astype(np.int32) * 2
        + dataset["y_angle"][remaining].astype(np.int32)
    )

    n_remaining = len(remaining)
    val_frac = 1.0 - train_frac
    train_rel, val_rel = train_test_split(
        np.arange(n_remaining),
        test_size=val_frac,
        stratify=strat,
        random_state=seed,
        shuffle=True,
    )

    train_idx = remaining[train_rel]
    val_idx   = remaining[val_rel]

    print(
        f"Split | train={len(train_idx)}  val={len(val_idx)}  "
        f"heldout_test={len(heldout_idx)}"
    )
    return (
        _subset(dataset, train_idx),
        _subset(dataset, val_idx),
        _subset(dataset, heldout_idx),
    )


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------

def compute_cvae_norm_stats(
    train_dataset: dict[str, np.ndarray],
    mode: str = "spectral",
) -> dict[str, np.ndarray]:
    """Per-channel mean and std from training trials only.

    Spectral mode computes stats over per-channel MU amplitude features.
    Raw mode computes one mean/std per channel over all training trials and
    timepoints, preserving waveform shape.

    Returns {"mu": (256,), "sigma": (256,)} float32.
    """
    n_train    = len(train_dataset["y_grip"])
    sum_x      = np.zeros(N_REAL_CHANNELS, dtype=np.float64)
    sum_x2     = np.zeros(N_REAL_CHANNELS, dtype=np.float64)
    file_cache: dict[str, np.ndarray] = {}

    for i in range(n_train):
        if mode == "raw" and "raw_file_paths" in train_dataset:
            fp = str(train_dataset["raw_file_paths"][train_dataset["file_idx"][i]])
        else:
            fp = str(train_dataset["file_paths"][train_dataset["file_idx"][i]])
        if fp not in file_cache:
            file_cache[fp] = np.load(fp, mmap_mode="r")
        sample = file_cache[fp][int(train_dataset["trial_idx"][i])]  # (n_ph, n_ch, n_tp)
        ph     = int(train_dataset["phase_idx"][i])
        if mode == "spectral":
            feat = extract_cvae_features(sample, ph, mode="spectral").astype(np.float64)
            sum_x  += feat
            sum_x2 += feat ** 2
        elif mode == "raw":
            raw = sample[ph, :N_REAL_CHANNELS, :].astype(np.float64)
            sum_x  += raw.sum(axis=1)
            sum_x2 += (raw ** 2).sum(axis=1)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'spectral' or 'raw'.")

    n = max(n_train * (N_TIMEPOINTS if mode == "raw" else 1), 1)
    mu    = sum_x / n
    sigma = np.sqrt(np.maximum(sum_x2 / n - mu ** 2, 0.0))
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return {"mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32), "mode": mode}


# ---------------------------------------------------------------------------
# PyTorch datasets
# ---------------------------------------------------------------------------

class LFPCVAEDataset(Dataset):
    """Lazy-loading dataset for the cVAE.

    Returns (x, c, y_grip, y_hand, y_angle, y_phase) where:
      x     : feature vector (input_dim,) — z-scored using norm_stats if provided
      c     : condition vector (7,) one-hot (phase, grip, hand)
      labels: long tensors
    """

    def __init__(
        self,
        dataset: dict[str, np.ndarray],
        norm_stats: dict[str, np.ndarray] | None = None,
        mode: str = "spectral",
    ):
        self.file_paths = dataset["file_paths"].tolist()
        self.raw_file_paths = dataset.get("raw_file_paths", dataset["file_paths"]).tolist()
        self.file_idx   = dataset["file_idx"]
        self.trial_idx  = dataset["trial_idx"]
        self.phase_idx  = dataset["phase_idx"]
        self.y_grip     = torch.tensor(dataset["y_grip"],  dtype=torch.long)
        self.y_hand     = torch.tensor(dataset["y_hand"],  dtype=torch.long)
        self.y_angle    = torch.tensor(dataset["y_angle"], dtype=torch.long)
        self.y_phase    = torch.tensor(
            dataset["phase_idx"].astype(np.int64), dtype=torch.long
        )
        self.conditions = torch.tensor(dataset["condition"], dtype=torch.float32)
        self.mode       = mode
        self._file_cache: dict[str, np.ndarray] = {}

        if norm_stats is not None:
            self.mu    = torch.tensor(norm_stats["mu"],    dtype=torch.float32)
            self.sigma = torch.tensor(norm_stats["sigma"], dtype=torch.float32)
        else:
            self.mu = self.sigma = None

    def __len__(self) -> int:
        return len(self.y_grip)

    def _load(self, fp: str) -> np.ndarray:
        if fp not in self._file_cache:
            self._file_cache[fp] = np.load(fp, mmap_mode="r")
        return self._file_cache[fp]

    def __getitem__(self, idx: int):
        fp     = (self.raw_file_paths if self.mode == "raw" else self.file_paths)[self.file_idx[idx]]
        data   = self._load(fp)
        sample = data[int(self.trial_idx[idx])]    # (n_phases, n_ch, n_tp)
        ph     = int(self.phase_idx[idx])
        feat   = extract_cvae_features(sample, ph, self.mode)
        x      = torch.from_numpy(feat)

        if self.mu is not None:
            # Always z-score per real channel; for raw mode broadcast over time
            if self.mode == "spectral":
                zero_mask = x == 0.0
                x = (x - self.mu) / self.sigma
                x[zero_mask] = 0.0  # preserve bad-channel zeros
            else:
                # x shape: (256*500,) → reshape → normalize per channel → flatten
                raw2d     = x.view(N_REAL_CHANNELS, N_TIMEPOINTS)
                zero_mask = raw2d == 0.0
                raw2d     = (raw2d - self.mu.unsqueeze(1)) / self.sigma.unsqueeze(1)
                raw2d[zero_mask] = 0.0
                x = raw2d.flatten()

        return x, self.conditions[idx], self.y_grip[idx], self.y_hand[idx], \
               self.y_angle[idx], self.y_phase[idx]


class PermutedCVAEDataset(Dataset):
    """Wraps LFPCVAEDataset with independently shuffled y_grip, y_hand, y_angle.

    Used as a sanity check — labels are broken while feature distribution is intact.
    """

    def __init__(self, base: LFPCVAEDataset, rng: np.random.Generator):
        self._base   = base
        self.y_grip  = torch.from_numpy(rng.permutation(base.y_grip.numpy()).copy())
        self.y_hand  = torch.from_numpy(rng.permutation(base.y_hand.numpy()).copy())
        self.y_angle = torch.from_numpy(rng.permutation(base.y_angle.numpy()).copy())

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        x, c, _, _, _, y_phase = self._base[idx]
        return x, c, self.y_grip[idx], self.y_hand[idx], self.y_angle[idx], y_phase


# ---------------------------------------------------------------------------
# Specialist input conversion (for classifier probe)
# ---------------------------------------------------------------------------

def spectral_to_specialist_input(
    x_norm: torch.Tensor,
    mu_cvae: torch.Tensor,
    sigma_cvae: torch.Tensor,
    mu_spec: torch.Tensor,
    sigma_spec: torch.Tensor,
) -> torch.Tensor:
    """Convert cVAE-normalized (B, 256) features to specialist-normalized (B, 4, 96).

    Steps:
      1. Un-normalize from cVAE z-score → raw amplitude per channel
      2. Group channels by brain area with zero-padding to MAX_AREA_CHANNELS=96
      3. Apply specialist z-score normalization (preserving zero-padded channels)
    """
    from transformer_encoder.data import MAX_AREA_CHANNELS

    device  = x_norm.device
    batch   = x_norm.shape[0]

    # Un-normalize: raw amplitude (256,) per sample
    x_raw = x_norm * sigma_cvae.to(device) + mu_cvae.to(device)  # (B, 256)

    # Re-group into (B, n_areas, MAX_AREA_CHANNELS) with zero-padding
    areas = []
    for _, start, end in AREA_SLICES:
        area = x_raw[:, start:end]  # (B, area_size)
        if area.shape[1] < MAX_AREA_CHANNELS:
            pad  = torch.zeros(batch, MAX_AREA_CHANNELS - area.shape[1], device=device)
            area = torch.cat([area, pad], dim=1)
        areas.append(area)
    x_area = torch.stack(areas, dim=1)  # (B, n_areas, MAX_AREA_CHANNELS)

    # Apply specialist normalization
    # mu_spec / sigma_spec shape: (n_areas, MAX_AREA_CHANNELS)
    mu_s  = mu_spec.to(device).unsqueeze(0)    # (1, n_areas, 96)
    sig_s = sigma_spec.to(device).unsqueeze(0) # (1, n_areas, 96)
    zero_mask = x_area == 0.0
    x_area = (x_area - mu_s) / sig_s
    x_area[zero_mask] = 0.0  # keep bad-channel zeros
    return x_area  # (B, n_areas, MAX_AREA_CHANNELS)
