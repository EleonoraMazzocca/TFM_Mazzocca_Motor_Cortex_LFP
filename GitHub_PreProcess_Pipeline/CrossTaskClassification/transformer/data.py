from __future__ import annotations

import re
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from instruction_embedding import encode_batch, INSTR_DIM
from instruction_encoding import (
    build_instruction_matrix,
    get_instruction_dim,
    parse_class_name,
    MINILM_TEMPLATE,
)

PARENT_DIR = Path(__file__).resolve().parents[1]
if os.fspath(PARENT_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(PARENT_DIR))

from data_paths import SEPARATED_CLASSES_DIR


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


@dataclass(frozen=True)
class ClassFile:
    path: Path
    grip: int
    hand: int
    angle: int
    combo_label: str
    is_heldout: bool


def parse_filename(path: Path) -> tuple[str, str, str] | None:
    match = re.fullmatch(
        r"(power|precision)_unimanual_(left|right)_(0|45|90|135)_degrees_mua_200_500",
        path.stem,
    )
    if match is None:
        return None
    return match.groups()


def find_class_files(
    classes_dir: Path = SEPARATED_CLASSES_DIR,
    heldout_grip: int = 1,
    heldout_hand: int = 1,
    heldout_angle: int = 3,
) -> list[ClassFile]:
    class_files: list[ClassFile] = []
    for path in sorted(classes_dir.glob("*_mua_200_500.npy")):
        if "bimanual" in path.name:
            continue
        parsed = parse_filename(path)
        if parsed is None:
            continue

        grip_name, hand_name, angle_name = parsed
        grip = GRIP_TO_ID[grip_name]
        hand = HAND_TO_ID[hand_name]
        angle = ANGLE_TO_ID[angle_name]
        is_heldout = (
            grip == heldout_grip
            and hand == heldout_hand
            and angle == heldout_angle
        )
        class_files.append(
            ClassFile(
                path=path,
                grip=grip,
                hand=hand,
                angle=angle,
                combo_label=f"{grip_name}_{hand_name}_{angle_name}",
                is_heldout=is_heldout,
            )
        )

    if not class_files:
        raise ValueError(f"No *_mua_200_500.npy files found in {classes_dir}")

    return class_files


def _cache_name(heldout_grip: int, heldout_hand: int, heldout_angle: int) -> str:
    return (
        f"mu_transformer_g{heldout_grip}_h{heldout_hand}_a{heldout_angle}.npz"
    )


def load_dataset(
    cache_dir: str | None = None,
    classes_dir: Path = SEPARATED_CLASSES_DIR,
    heldout_grip: int = 1,
    heldout_hand: int = 1,
    heldout_angle: int = 3,
) -> dict[str, np.ndarray]:
    if cache_dir:
        cache_path = Path(cache_dir) / _cache_name(
            heldout_grip=heldout_grip,
            heldout_hand=heldout_hand,
            heldout_angle=heldout_angle,
        )
        if cache_path.exists():
            print(f"Loading cached dataset from {cache_path}")
            raw = np.load(cache_path, allow_pickle=False)
            return {key: raw[key] for key in raw.files}

    class_files = find_class_files(
        classes_dir=classes_dir,
        heldout_grip=heldout_grip,
        heldout_hand=heldout_hand,
        heldout_angle=heldout_angle,
    )

    file_paths = np.array([os.fspath(class_file.path) for class_file in class_files])
    trial_idx_list = []
    file_idx_list = []
    y_grip_list = []
    y_hand_list = []
    y_angle_list = []
    is_heldout_list = []
    n_channels: int | None = None

    print("Loading MU features from precomputed class files...")
    for file_idx, class_file in enumerate(class_files):
        features = np.load(class_file.path, mmap_mode="r")
        n_trials = features.shape[0]
        print(f"  {class_file.path.name}: {n_trials} trials")
        if n_channels is None:
            n_channels = int(features.shape[2])

        trial_idx_list.append(np.arange(n_trials, dtype=np.int32))
        file_idx_list.append(np.full(n_trials, file_idx, dtype=np.int16))
        y_grip_list.append(np.full(n_trials, class_file.grip, dtype=np.int64))
        y_hand_list.append(np.full(n_trials, class_file.hand, dtype=np.int64))
        y_angle_list.append(np.full(n_trials, class_file.angle, dtype=np.int64))
        is_heldout_list.append(np.full(n_trials, class_file.is_heldout, dtype=bool))

    if n_channels is None:
        raise ValueError(f"No MU features found in {classes_dir}")

    data = {
        "file_paths": file_paths,
        "file_idx": np.concatenate(file_idx_list, axis=0),
        "trial_idx": np.concatenate(trial_idx_list, axis=0),
        "y_grip": np.concatenate(y_grip_list, axis=0),
        "y_hand": np.concatenate(y_hand_list, axis=0),
        "y_angle": np.concatenate(y_angle_list, axis=0),
        "is_heldout": np.concatenate(is_heldout_list, axis=0),
        "n_channels": np.array(n_channels, dtype=np.int64),
    }

    if cache_dir:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **data)
        print(f"Cached dataset to {cache_path}")

    return data


def _subset(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    subset: dict[str, np.ndarray] = {
        "file_paths": data["file_paths"],
        "n_channels": data["n_channels"],
    }
    for key in ("file_idx", "trial_idx", "y_grip", "y_hand", "y_angle", "is_heldout"):
        subset[key] = data[key][idx]
    return subset


def _load_sample_from_data(
    data: dict[str, np.ndarray],
    idx: int,
    file_cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    file_path = data["file_paths"][data["file_idx"][idx]]
    if file_cache is not None and file_path in file_cache:
        features = file_cache[file_path]
    else:
        features = np.load(file_path, mmap_mode="r")
        if file_cache is not None:
            file_cache[file_path] = features
    sample = features[data["trial_idx"][idx], :, :, :]
    return extract_area_features(sample)


def extract_area_features(sample: np.ndarray) -> np.ndarray:
    """
    (n_phases, n_channels, 500) -> (n_phases, n_areas, MAX_AREA_CHANNELS)

    Computes mean absolute amplitude over time (spectral amplitude) per channel,
    then groups channels by brain area and zero-pads to MAX_AREA_CHANNELS.
    Bad channels (zeroed during preprocessing) remain 0.
    """
    amp = np.mean(np.abs(sample.astype(np.float32, copy=False)), axis=-1)  # (n_phases, n_channels)
    areas = []
    for _, start, end in AREA_SLICES:
        area = amp[:, start:end]  # (n_phases, area_size)
        if area.shape[1] < MAX_AREA_CHANNELS:
            pad = np.zeros((area.shape[0], MAX_AREA_CHANNELS - area.shape[1]), dtype=np.float32)
            area = np.concatenate([area, pad], axis=1)
        areas.append(area)
    return np.stack(areas, axis=1)  # (n_phases, n_areas, MAX_AREA_CHANNELS)


def _normalise(train_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_train = len(train_data["y_grip"])
    sum_x = np.zeros((N_PHASES, N_AREAS, AREA_FEATURE_DIM), dtype=np.float64)
    sum_x2 = np.zeros((N_PHASES, N_AREAS, AREA_FEATURE_DIM), dtype=np.float64)
    file_cache: dict[str, np.ndarray] = {}

    for idx in range(n_train):
        sample = _load_sample_from_data(train_data, idx, file_cache=file_cache).astype(np.float64)
        sum_x += sample
        sum_x2 += sample ** 2

    mu = sum_x / max(n_train, 1)
    variance = (sum_x2 / max(n_train, 1)) - mu ** 2
    sigma = np.sqrt(np.maximum(variance, 0.0))
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    return {"mu": mu, "sigma": sigma}


def make_compositional_split(
    data: dict[str, np.ndarray],
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    all_idx = np.arange(len(data["y_grip"]))
    heldout_idx = all_idx[data["is_heldout"]]
    remaining_idx = all_idx[~data["is_heldout"]]

    if len(heldout_idx) == 0:
        raise ValueError("No held-out trials found. Check held-out grip/hand/angle.")

    remaining_combo = np.array(
        [
            f"{data['y_grip'][i]}_{data['y_hand'][i]}_{data['y_angle'][i]}"
            for i in remaining_idx
        ]
    )

    heldout_val_idx, heldout_test_idx = train_test_split(
        heldout_idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
    )
    remaining_train_idx, remaining_temp_idx = train_test_split(
        remaining_idx,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
        stratify=remaining_combo,
    )
    remaining_temp_combo = np.array(
        [
            f"{data['y_grip'][i]}_{data['y_hand'][i]}_{data['y_angle'][i]}"
            for i in remaining_temp_idx
        ]
    )
    remaining_val_idx, remaining_test_idx = train_test_split(
        remaining_temp_idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=remaining_temp_combo,
    )

    train_data = _subset(data, remaining_train_idx)
    val_data = _subset(data, np.concatenate([remaining_val_idx, heldout_val_idx]))
    seen_test_data = _subset(data, remaining_test_idx)
    heldout_test_data = _subset(data, heldout_test_idx)
    norm_stats = _normalise(train_data)

    print(
        "Split sizes | "
        f"train={len(train_data['y_grip'])} "
        f"val={len(val_data['y_grip'])} "
        f"seen_test={len(seen_test_data['y_grip'])} "
        f"heldout_test={len(heldout_test_data['y_grip'])}"
    )

    return train_data, val_data, seen_test_data, heldout_test_data, norm_stats


class LFPDataset(Dataset):
    def __init__(
        self,
        data: dict[str, np.ndarray],
        norm_stats: dict[str, np.ndarray] | None = None,
        random_instruction: bool = False,
    ):
        self.file_paths = [os.fspath(path) for path in data["file_paths"].tolist()]
        self.file_idx = data["file_idx"]
        self.trial_idx = data["trial_idx"]
        self.y_grip = torch.tensor(data["y_grip"], dtype=torch.long)
        self.y_hand = torch.tensor(data["y_hand"], dtype=torch.long)
        self.y_angle = torch.tensor(data["y_angle"], dtype=torch.long)
        self.instr = torch.tensor(
            encode_batch(data["y_grip"], data["y_hand"], data["y_angle"]),
            dtype=torch.float32,
        )  # shape: (N, 8)
        self.random_instruction = random_instruction
        # norm_stats["mu"] / ["sigma"] shape: (3, 4, 501)
        self.mu = None if norm_stats is None else torch.tensor(norm_stats["mu"], dtype=torch.float32)
        self.sigma = None if norm_stats is None else torch.tensor(norm_stats["sigma"], dtype=torch.float32)
        self._file_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.y_grip)

    def _get_features(self, file_path: str) -> np.ndarray:
        features = self._file_cache.get(file_path)
        if features is None:
            features = np.load(file_path, mmap_mode="r")
            self._file_cache[file_path] = features
        return features

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[self.file_idx[idx]]
        features = self._get_features(file_path)
        x = torch.from_numpy(extract_area_features(features[self.trial_idx[idx], :, :, :]))
        if self.mu is not None and self.sigma is not None:
            zero_mask = x == 0.0
            x = (x - self.mu) / self.sigma
            x[zero_mask] = 0.0
        instr = torch.randn(INSTR_DIM) if self.random_instruction else self.instr[idx]
        return x, self.y_grip[idx], self.y_hand[idx], self.y_angle[idx], instr


class BalancedInstructionDataset(Dataset):
    """Dataset with per-class balanced instruction masking, reshuffled every epoch.

    During training (is_test=False) each epoch a balanced fraction (mask_prob) of
    samples per class combination has its instruction zeroed, forcing the model to
    use LFP when the instruction is absent.  At test time (is_test=True) the
    instruction is always the zero vector regardless of mask_prob.

    Returns 5-tuples compatible with the existing train/evaluate loop:
        (lfp, y_grip, y_hand, y_angle, instruction)

    Args:
        data:       split dict from make_compositional_split / _subset
        norm_stats: normalisation stats from make_compositional_split
        encoding:   one of "onehot", "bow", "minilm", "none"
        mask_prob:  fraction of samples per class to mask each epoch (0.5 or 0.7)
        is_test:    if True, always zero the instruction — reshuffle_masks is a no-op
        sentences:  required only when encoding=="minilm" and you want to supply
                    custom sentences; if None the MINILM_TEMPLATE lookup is used
    """

    def __init__(
        self,
        data: dict[str, np.ndarray],
        norm_stats: dict[str, np.ndarray] | None,
        encoding: str,
        mask_prob: float = 0.5,
        is_test: bool = False,
        sentences: list[str] | None = None,
    ):
        self.file_paths = [os.fspath(path) for path in data["file_paths"].tolist()]
        self.file_idx = data["file_idx"]
        self.trial_idx = data["trial_idx"]
        self.y_grip = torch.tensor(data["y_grip"], dtype=torch.long)
        self.y_hand = torch.tensor(data["y_hand"], dtype=torch.long)
        self.y_angle = torch.tensor(data["y_angle"], dtype=torch.long)
        self.mu = None if norm_stats is None else torch.tensor(norm_stats["mu"], dtype=torch.float32)
        self.sigma = None if norm_stats is None else torch.tensor(norm_stats["sigma"], dtype=torch.float32)
        self._file_cache: dict[str, np.ndarray] = {}

        self.encoding = encoding
        self.mask_prob = mask_prob
        self.is_test = is_test
        self.instruction_dim = get_instruction_dim(encoding)

        n = len(data["y_grip"])

        # Derive string class names for each trial
        class_names = [
            f"{ID_TO_GRIP[int(data['y_grip'][i])]}_{ID_TO_HAND[int(data['y_hand'][i])]}_{ID_TO_ANGLE[int(data['y_angle'][i])]}"
            for i in range(n)
        ]

        # Build sentences for MiniLM if not provided
        if encoding == "minilm" and sentences is None:
            sentences = [MINILM_TEMPLATE[parse_class_name(cn)] for cn in class_names]

        # Pre-compute instruction matrix once — never rebuilt, masking is applied on-the-fly
        self._instructions = build_instruction_matrix(class_names, encoding, sentences)  # (N, dim)

        # Integer combo key for efficient per-class mask bookkeeping:
        # grip ∈ {0,1}, hand ∈ {0,1}, angle ∈ {0..3}  → unique key in [0, 15]
        self._combo_keys = (
            data["y_grip"].astype(np.int32) * 8
            + data["y_hand"].astype(np.int32) * 4
            + data["y_angle"].astype(np.int32)
        )

        # Initialise mask (True = zero the instruction)
        self._mask = np.zeros(n, dtype=bool)
        if is_test:
            self._mask[:] = True
        else:
            self.reshuffle_masks()

    def reshuffle_masks(self) -> None:
        """Recompute balanced per-class mask assignments.

        Must be called at the start of each training epoch (train.py does this).
        No-op when is_test=True.
        """
        if self.is_test:
            return
        n = len(self.y_grip)
        self._mask = np.zeros(n, dtype=bool)
        rng = np.random.default_rng()  # fresh seed every epoch — intentional
        for combo in np.unique(self._combo_keys):
            idx = np.where(self._combo_keys == combo)[0]
            n_mask = int(np.floor(len(idx) * self.mask_prob))
            if n_mask > 0:
                chosen = rng.choice(idx, size=n_mask, replace=False)
                self._mask[chosen] = True

    def __len__(self) -> int:
        return len(self.y_grip)

    def _get_features(self, file_path: str) -> np.ndarray:
        features = self._file_cache.get(file_path)
        if features is None:
            features = np.load(file_path, mmap_mode="r")
            self._file_cache[file_path] = features
        return features

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[self.file_idx[idx]]
        features = self._get_features(file_path)
        x = torch.from_numpy(extract_area_features(features[self.trial_idx[idx], :, :, :]))
        if self.mu is not None and self.sigma is not None:
            zero_mask = x == 0.0
            x = (x - self.mu) / self.sigma
            x[zero_mask] = 0.0

        if self._mask[idx] or self.instruction_dim == 0:
            instr = torch.zeros(max(self.instruction_dim, 1), dtype=torch.float32)
            if self.instruction_dim == 0:
                instr = torch.zeros(0, dtype=torch.float32)
        else:
            instr = torch.tensor(self._instructions[idx], dtype=torch.float32)

        return x, self.y_grip[idx], self.y_hand[idx], self.y_angle[idx], instr
