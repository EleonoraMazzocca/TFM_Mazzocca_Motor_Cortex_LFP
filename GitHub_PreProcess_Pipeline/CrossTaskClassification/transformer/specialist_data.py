from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from data import AREA_SLICES, MAX_AREA_CHANNELS, N_AREAS

N_TIMEPOINTS = 500  # timepoints per phase epoch


def extract_phase_area_features(
    sample: np.ndarray, phase_idx: int, n_bins: int | str = 1,
    use_per_channel: bool = False,
) -> np.ndarray:
    """
    (n_phases, n_channels, n_timepoints) -> (n_areas, actual_bins)  [area-avg mode]
                                         -> (n_areas, MAX_AREA_CHANNELS)  [per-channel mode]

    Area-avg mode (use_per_channel=False, default):
      Divides the selected phase into n_bins non-overlapping temporal windows, computes
      mean absolute amplitude per window per channel, then averages across channels
      within each brain area.
      n_bins=1: shape (n_areas, 1); n_bins=5/10/20: (n_areas, n_bins);
      n_bins="raw": (n_areas, N_TIMEPOINTS).

    Per-channel mode (use_per_channel=True):
      Collapses time (mean |x| over all timepoints), keeps all channels per area,
      zero-pads to MAX_AREA_CHANNELS → shape (n_areas, MAX_AREA_CHANNELS). n_bins ignored.
      NOTE: this is NOT the same as n_bins=1. n_bins=1 also averages channels within each
      area, producing one scalar per area. Per-channel preserves individual channel amplitudes.

    Bad channels zeroed during preprocessing remain 0 throughout.
    """
    phase_data = sample[phase_idx].astype(np.float32, copy=False)  # (n_channels, n_tp)

    if use_per_channel:
        amp = np.mean(np.abs(phase_data), axis=1)  # (n_channels,) — collapse time
        areas = []
        for _, start, end in AREA_SLICES:
            area_ch = amp[start:end]  # (area_size,)
            if len(area_ch) < MAX_AREA_CHANNELS:
                area_ch = np.concatenate(
                    [area_ch, np.zeros(MAX_AREA_CHANNELS - len(area_ch), dtype=np.float32)]
                )
            areas.append(area_ch)
        result = np.stack(areas, axis=0)  # (n_areas, MAX_AREA_CHANNELS)
        assert result.shape == (N_AREAS, MAX_AREA_CHANNELS)
        return result

    n_tp = phase_data.shape[1]
    actual_bins: int = n_tp if n_bins == "raw" else int(n_bins)
    bin_size: int = n_tp // actual_bins

    # (n_channels, actual_bins): mean |x| per window per channel
    amp = np.stack(
        [np.mean(np.abs(phase_data[:, i * bin_size:(i + 1) * bin_size]), axis=1)
         for i in range(actual_bins)],
        axis=1,
    )

    areas = []
    for _, start, end in AREA_SLICES:
        areas.append(amp[start:end].mean(axis=0))  # (actual_bins,)
    return np.stack(areas, axis=0)  # (n_areas, actual_bins)


def compute_specialist_norm_stats(
    train_data: dict[str, np.ndarray],
    phase_idx: int,
    n_bins: int | str = 1,
    use_per_channel: bool = False,
) -> dict[str, np.ndarray]:
    """Compute per-area z-score normalization stats from training data.

    Returns dict with keys "mu" and "sigma".
    Shape: (n_areas, MAX_AREA_CHANNELS) when use_per_channel=True,
           (n_areas, actual_bins) otherwise.
    Entries where sigma < 1e-8 get sigma=1.
    """
    actual_bins: int = (MAX_AREA_CHANNELS if use_per_channel
                        else (N_TIMEPOINTS if n_bins == "raw" else int(n_bins)))
    n_train = len(train_data["y_grip"])
    sum_x = np.zeros((N_AREAS, actual_bins), dtype=np.float64)
    sum_x2 = np.zeros((N_AREAS, actual_bins), dtype=np.float64)
    file_cache: dict[str, np.ndarray] = {}

    for idx in range(n_train):
        fp = str(train_data["file_paths"][train_data["file_idx"][idx]])
        if fp not in file_cache:
            file_cache[fp] = np.load(fp, mmap_mode="r")
        sample = file_cache[fp][int(train_data["trial_idx"][idx])]  # (n_phases, n_ch, n_tp)
        feat = extract_phase_area_features(
            sample, phase_idx, n_bins, use_per_channel=use_per_channel
        ).astype(np.float64)
        sum_x += feat
        sum_x2 += feat ** 2

    mu = sum_x / max(n_train, 1)
    variance = (sum_x2 / max(n_train, 1)) - mu ** 2
    sigma = np.sqrt(np.maximum(variance, 0.0))
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return {"mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32)}


class LFPSpecialistDataset(Dataset):
    """Dataset for a single movement phase.

    Returns 4-tuples (x, y_grip, y_hand, y_angle).
    x shape: (n_areas, n_bins) — z-scored if norm_stats provided.
    """

    def __init__(
        self,
        data: dict[str, np.ndarray],
        phase_idx: int,
        norm_stats: dict[str, np.ndarray] | None = None,
        n_bins: int | str = 1,
        use_per_channel: bool = False,
    ):
        self.file_paths = [os.fspath(p) for p in data["file_paths"].tolist()]
        self.file_idx = data["file_idx"]
        self.trial_idx = data["trial_idx"]
        self.y_grip = torch.tensor(data["y_grip"], dtype=torch.long)
        self.y_hand = torch.tensor(data["y_hand"], dtype=torch.long)
        self.y_angle = torch.tensor(data["y_angle"], dtype=torch.long)
        self.phase_idx = phase_idx
        self.n_bins = n_bins
        self.use_per_channel = use_per_channel
        self.mu = None if norm_stats is None else torch.tensor(norm_stats["mu"], dtype=torch.float32)
        self.sigma = None if norm_stats is None else torch.tensor(norm_stats["sigma"], dtype=torch.float32)
        self._file_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.y_grip)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fp = self.file_paths[self.file_idx[idx]]
        if fp not in self._file_cache:
            self._file_cache[fp] = np.load(fp, mmap_mode="r")
        sample = self._file_cache[fp][int(self.trial_idx[idx])]  # (n_phases, n_ch, n_tp)
        x = torch.from_numpy(
            extract_phase_area_features(sample, self.phase_idx, self.n_bins, self.use_per_channel)
        )
        if self.mu is not None:
            zero_mask = x == 0.0
            x = (x - self.mu) / self.sigma
            x[zero_mask] = 0.0  # keep fully-zeroed areas at zero after z-scoring
        return x, self.y_grip[idx], self.y_hand[idx], self.y_angle[idx]


class PermutedLabelDataset(Dataset):
    """Wraps LFPSpecialistDataset with independently permuted labels per head.

    Used for permutation-based significance testing. Features are loaded from
    the same underlying files (reuses the base dataset's mmap'd file cache), but
    the three label tensors are independently shuffled so label–feature
    associations are destroyed for all heads simultaneously.

    Each head is permuted with a fresh call to rng.permutation, so grip, hand,
    and angle shuffles are independent of each other.
    """

    def __init__(self, base: LFPSpecialistDataset, rng: np.random.Generator):
        self._base = base
        self.y_grip = torch.from_numpy(rng.permutation(base.y_grip.numpy()).copy())
        self.y_hand = torch.from_numpy(rng.permutation(base.y_hand.numpy()).copy())
        self.y_angle = torch.from_numpy(rng.permutation(base.y_angle.numpy()).copy())

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _, _, _ = self._base[idx]
        return x, self.y_grip[idx], self.y_hand[idx], self.y_angle[idx]
