"""
Classifier 2 - Compositional linear baseline for LFP data.

Strategy:
- Load one file at a time
- Extract per-channel mean absolute amplitude immediately (spectral amplitude)
- Discard raw data
- Features are (n_trials, 3_phases, n_channels) floats → tiny
- Split, standardize, train logistic regression per phase per head
- Evaluate on seen combinations and held-out Precision+Right+135
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    from data_paths import SEPARATED_CLASSES_DIR
except ImportError:
    from .data_paths import SEPARATED_CLASSES_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE_NAMES = ["prereach", "reach", "grasp"]
ANGLE_TO_ID = {"0": 0, "45": 1, "90": 2, "135": 3}
GRIP_TO_ID = {"power": 0, "precision": 1}
HAND_TO_ID = {"left": 0, "right": 1}
ID_TO_ANGLE = {v: k for k, v in ANGLE_TO_ID.items()}
ID_TO_GRIP = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND = {v: k for k, v in HAND_TO_ID.items()}
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClassFile:
    path: Path
    grip: int
    hand: int
    angle: int
    combo_label: str
    is_heldout: bool


@dataclass
class Dataset:
    """Holds extracted features and labels for all trials."""
    # features: (n_trials, n_phases, n_channels) float32
    features: np.ndarray
    grip: np.ndarray    # (n_trials,) int
    hand: np.ndarray    # (n_trials,) int
    angle: np.ndarray   # (n_trials,) int
    combo_labels: List[str] = field(default_factory=list)
    is_heldout: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compositional linear classifier for LFP [grip, hand, angle]."
    )
    parser.add_argument(
        "--classes-dir",
        type=Path,
        default=SEPARATED_CLASSES_DIR,
        help="Path to Separated_Data/classes/ folder containing *_mua_200_500.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "logs" / "classifier2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0],
        help="Regularization values to try for logistic regression.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# File discovery and parsing
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> Optional[Tuple[str, str, str]]:
    """
    Parse grip, hand, angle from filename like:
    precision_unimanual_right_135_degrees_mua_200_500.npy
    """
    match = re.fullmatch(
        r"(power|precision)_unimanual_(left|right)_(0|45|90|135)_degrees_mua_200_500",
        path.stem,
    )
    if match is None:
        return None
    return match.groups()


def find_class_files(classes_dir: Path) -> List[ClassFile]:
    files = []
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
        combo_label = f"{grip_name}_{hand_name}_{angle_name}"
        # Held-out combination: Precision + Right + 135
        is_heldout = (grip == 1 and hand == 1 and angle == 3)
        files.append(ClassFile(
            path=path,
            grip=grip,
            hand=hand,
            angle=angle,
            combo_label=combo_label,
            is_heldout=is_heldout,
        ))
    if not files:
        raise ValueError(f"No *_mua_200_500.npy unimanual files found in {classes_dir}")
    print(f"Found {len(files)} class files")
    for f in files:
        tag = " [HELD-OUT]" if f.is_heldout else ""
        print(f"  {f.path.name}{tag}")
    return files


# ---------------------------------------------------------------------------
# Feature extraction - this is the memory-critical part
# We load one file at a time and extract features immediately
# ---------------------------------------------------------------------------

def extract_features_from_file(path: Path) -> np.ndarray:
    """
    Load one file and extract per-channel spectral amplitude.

    Input:  (n_trials, 3, 256, 500) int16
    Output: (n_trials, 3, 256) float32

    Spectral amplitude = mean absolute value across time per channel.
    This is exactly what DePass et al. used.
    We cast to float32 before computing to avoid int16 overflow.
    """
    print(f"  Loading {path.name} ...", end=" ", flush=True)
    raw = np.load(path, mmap_mode="r")          # (N, 3, 256, 500) int16, not in RAM yet
    raw_float = raw.astype(np.float32)           # now in RAM: N*3*256*500*4 bytes
    features = np.mean(np.abs(raw_float), axis=3)  # (N, 3, 256) float32
    del raw_float                                # free immediately
    print(f"shape={features.shape}, dtype={features.dtype}")
    return features                              # small: N*3*256*4 bytes ≈ <1MB per file


def build_dataset(class_files: List[ClassFile]) -> Dataset:
    """
    Load all files one at a time, extract features, concatenate.
    Peak RAM usage = one raw file at a time (~240MB) + growing features array (~few MB).
    """
    all_features = []
    all_grip = []
    all_hand = []
    all_angle = []
    all_combo = []
    all_heldout = []

    print("\nExtracting features file by file:")
    for cf in class_files:
        features = extract_features_from_file(cf.path)  # (N, 3, 256)
        n = features.shape[0]
        all_features.append(features)
        all_grip.append(np.full(n, cf.grip, dtype=np.int64))
        all_hand.append(np.full(n, cf.hand, dtype=np.int64))
        all_angle.append(np.full(n, cf.angle, dtype=np.int64))
        all_combo.extend([cf.combo_label] * n)
        all_heldout.append(np.full(n, cf.is_heldout, dtype=bool))

    return Dataset(
        features=np.concatenate(all_features, axis=0),   # (total_trials, 3, 256)
        grip=np.concatenate(all_grip),
        hand=np.concatenate(all_hand),
        angle=np.concatenate(all_angle),
        combo_labels=all_combo,
        is_heldout=np.concatenate(all_heldout),
    )


# ---------------------------------------------------------------------------
# Valid channel detection
# ---------------------------------------------------------------------------

def find_valid_channels(train_features: np.ndarray) -> np.ndarray:
    """
    A channel is valid if it has nonzero signal in at least one training trial.
    train_features: (n_train, 3, 256)
    Returns boolean mask of shape (256,)
    """
    # mean across trials and phases: if all zero → bad channel
    channel_mean = train_features.mean(axis=(0, 1))  # (256,)
    valid = channel_mean != 0.0
    print(f"  Valid channels: {valid.sum()} / {len(valid)}")
    bad_channel_idx = np.where(channel_mean == 0.0)[0]
    print(f"Bad channel index: {bad_channel_idx}")
    return valid


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def make_split(
    dataset: Dataset,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Returns index arrays for each split.

    - heldout (Precision+Right+135): split 50/50 into heldout_val / heldout_inference
    - remaining: split 85/7.5/7.5 stratified by combo_label
    """
    all_idx = np.arange(len(dataset.grip))
    heldout_idx = all_idx[dataset.is_heldout]
    remaining_idx = all_idx[~dataset.is_heldout]

    if len(heldout_idx) == 0:
        raise ValueError("No held-out Precision+Right+135 trials found.")

    print(f"\nSplit: {len(heldout_idx)} held-out trials, {len(remaining_idx)} remaining")

    heldout_val_idx, heldout_inf_idx = train_test_split(
        heldout_idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
    )

    remaining_labels = [dataset.combo_labels[i] for i in remaining_idx]
    remaining_train_idx, remaining_temp_idx = train_test_split(
        remaining_idx,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
        stratify=remaining_labels,
    )
    remaining_temp_labels = [dataset.combo_labels[i] for i in remaining_temp_idx]
    remaining_val_idx, remaining_inf_idx = train_test_split(
        remaining_temp_idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=remaining_temp_labels,
    )

    return {
        "train": remaining_train_idx,
        "val": np.concatenate([remaining_val_idx, heldout_val_idx]),
        "remaining_val": remaining_val_idx,
        "heldout_val": heldout_val_idx,
        "remaining_inference": remaining_inf_idx,
        "heldout_inference": heldout_inf_idx,
        "inference": np.concatenate([remaining_inf_idx, heldout_inf_idx]),
    }


# ---------------------------------------------------------------------------
# Balance report
# ---------------------------------------------------------------------------

def balance_report(
    dataset: Dataset,
    splits: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    split_names = ["train", "val", "remaining_inference", "heldout_inference"]
    all_combos = sorted(set(dataset.combo_labels))

    lines = ["Balance Report", ""]
    for combo in all_combos:
        lines.append(combo)
        percents = []
        for split_name in split_names:
            idx = splits[split_name]
            total = len(idx)
            count = sum(1 for i in idx if dataset.combo_labels[i] == combo)
            pct = 100.0 * count / total if total > 0 else 0.0
            percents.append(pct)
            lines.append(f"  {split_name:25s}: n={count:4d}  {pct:.1f}%")
        deviation = max(percents) - min(percents)
        if combo == "precision_right_135":
            lines.append(f"  → held-out combination, train absence expected")
        elif deviation > 5.0:
            lines.append(f"  → FLAG: deviation={deviation:.1f}% exceeds 5%")
        lines.append("")

    text = "\n".join(lines)
    print("\n" + text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Standardization and feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    features: np.ndarray,   # (n_trials, 3, 256)
    phase_index: int,
    valid_channels: np.ndarray,  # (256,) bool
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract one phase, keep valid channels, standardize.
    Returns (X, mean, std) where X is (n_trials, n_valid_channels).
    Mean and std are computed from data if not provided (use for train only).
    """
    x = features[:, phase_index, :][:, valid_channels]  # (n_trials, n_valid)

    if mean is None:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)

    # For samples where the channel was zeroed (bad channel in that session),
    # the feature will be 0.0 before standardization.
    # We detect these and keep them at 0.0 after standardization too.
    zero_mask = (x == 0.0)
    x = (x - mean) / std
    x[zero_mask] = 0.0

    return x.astype(np.float32), mean, std


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def label_names(head: str) -> List[str]:
    if head == "grip":
        return ["power", "precision"]
    if head == "hand":
        return ["left", "right"]
    if head == "angle":
        return ["0°", "45°", "90°", "135°"]
    raise ValueError(head)


def choose_best_c(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    c_values: Sequence[float],
    max_iter: int,
) -> Tuple[float, LogisticRegression]:
    best_c, best_model, best_f1 = None, None, -np.inf
    for c in c_values:
        model = LogisticRegression(
            C=c, penalty="l2", solver="lbfgs",
            max_iter=max_iter,
        )
        model.fit(train_x, train_y)
        f1 = f1_score(val_y, model.predict(val_x), average="macro", zero_division=0)
        print(f"    C={c:.3f}  val_macro_f1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_c, best_model = f1, c, model
    print(f"    → best C={best_c}")
    return best_c, best_model


def evaluate(
    model: LogisticRegression,
    x: np.ndarray,
    y: np.ndarray,
    names: List[str],
) -> Dict:
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y, pred, labels=np.arange(len(names))
        ).tolist(),
        "y_true": y.tolist(),
        "y_pred": pred.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    phase_dir: Path,
    results: Dict[str, Dict],  # head -> split -> results
) -> None:
    heads = ["grip", "hand", "angle"]
    split_keys = ["remaining_inference", "heldout_inference"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Confusion matrices — {phase_dir.name}")

    for row, split_key in enumerate(split_keys):
        for col, head in enumerate(heads):
            ax = axes[row, col]
            matrix = np.array(results[head][split_key]["confusion_matrix"])
            acc = results[head][split_key]["accuracy"]
            ax.imshow(matrix, cmap="Blues")
            names = label_names(head)
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            split_label = "seen" if split_key == "remaining_inference" else "HELD-OUT"
            ax.set_title(f"{head} | {split_label}\nacc={acc:.2f}")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, str(int(matrix[i, j])),
                            ha="center", va="center", fontsize=7)

    fig.tight_layout()
    out = phase_dir / "confusion_matrices.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_accuracy_summary(output_dir: Path, all_results: Dict) -> None:
    """
    Bar chart: accuracy per head per phase for seen vs held-out.
    """
    heads = ["grip", "hand", "angle"]
    phases = PHASE_NAMES
    splits = ["remaining_inference", "heldout_inference"]
    split_labels = ["seen combinations", "held-out (compositionality)"]
    colors = ["steelblue", "tomato"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("Accuracy: seen vs held-out combinations")

    for col, head in enumerate(heads):
        ax = axes[col]
        x = np.arange(len(phases))
        width = 0.35
        for s_idx, (split_key, split_label) in enumerate(zip(splits, split_labels)):
            accs = [
                all_results[phase][head][split_key]["accuracy"]
                for phase in phases
            ]
            bars = ax.bar(
                x + s_idx * width - width / 2,
                accs, width,
                label=split_label,
                color=colors[s_idx],
                alpha=0.85,
            )
            for bar, acc in zip(bars, accs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{acc:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )
        ax.set_title(head)
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("accuracy" if col == 0 else "")
        ax.axhline(0.5 if head != "angle" else 0.25, color="gray",
                   linestyle="--", linewidth=0.8, label="chance")
        if col == 0:
            ax.legend(fontsize=7)

    fig.tight_layout()
    out = output_dir / "accuracy_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover files
    class_files = find_class_files(args.classes_dir)

    # 2. Extract features file by file — peak RAM = one raw file at a time
    dataset = build_dataset(class_files)
    print(f"\nFull dataset: {dataset.features.shape} features, "
          f"{dataset.grip.shape[0]} trials total")
    print(f"  dtype={dataset.features.dtype}, "
          f"size={dataset.features.nbytes / 1e6:.1f} MB")

    # 3. Split
    splits = make_split(dataset, seed=args.seed)
    print("\nSplit sizes:")
    for name, idx in splits.items():
        print(f"  {name:25s}: {len(idx)}")

    # 4. Balance report
    balance_report(
        dataset=dataset,
        splits=splits,
        output_path=args.output_dir / "split_balance_report.txt",
    )

    # 5. Find valid channels from training data only
    train_features = dataset.features[splits["train"]]
    valid_channels = find_valid_channels(train_features)

    # 6. Train and evaluate per phase
    all_results: Dict[str, Dict] = {}

    for phase_index, phase_name in enumerate(PHASE_NAMES):
        print(f"\n{'='*60}")
        print(f"Phase: {phase_name}")
        print(f"{'='*60}")

        phase_dir = args.output_dir / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Prepare features for this phase
        train_x, mean, std = prepare_features(
            dataset.features[splits["train"]], phase_index, valid_channels
        )
        val_x, _, _ = prepare_features(
            dataset.features[splits["val"]], phase_index, valid_channels, mean, std
        )
        rem_inf_x, _, _ = prepare_features(
            dataset.features[splits["remaining_inference"]], phase_index,
            valid_channels, mean, std
        )
        held_inf_x, _, _ = prepare_features(
            dataset.features[splits["heldout_inference"]], phase_index,
            valid_channels, mean, std
        )

        print(f"  Feature shape: train={train_x.shape}, "
              f"val={val_x.shape}, "
              f"rem_inf={rem_inf_x.shape}, "
              f"held_inf={held_inf_x.shape}")

        head_targets = {
            "grip": (
                dataset.grip[splits["train"]],
                dataset.grip[splits["val"]],
                dataset.grip[splits["remaining_inference"]],
                dataset.grip[splits["heldout_inference"]],
            ),
            "hand": (
                dataset.hand[splits["train"]],
                dataset.hand[splits["val"]],
                dataset.hand[splits["remaining_inference"]],
                dataset.hand[splits["heldout_inference"]],
            ),
            "angle": (
                dataset.angle[splits["train"]],
                dataset.angle[splits["val"]],
                dataset.angle[splits["remaining_inference"]],
                dataset.angle[splits["heldout_inference"]],
            ),
        }

        phase_results: Dict[str, Dict] = {}
        all_results[phase_name] = {}

        for head, (tr_y, va_y, ri_y, hi_y) in head_targets.items():
            print(f"\n  Head: {head}")
            names = label_names(head)

            best_c, best_model = choose_best_c(
                train_x=train_x,
                train_y=tr_y,
                val_x=val_x,
                val_y=va_y,
                c_values=args.c_values,
                max_iter=args.max_iter,
            )

            rem_results = evaluate(best_model, rem_inf_x, ri_y, names)
            held_results = evaluate(best_model, held_inf_x, hi_y, names)

            print(f"  Seen combinations:    acc={rem_results['accuracy']:.3f}  "
                  f"f1={rem_results['macro_f1']:.3f}")
            print(f"  Held-out (composit.): acc={held_results['accuracy']:.3f}  "
                  f"f1={held_results['macro_f1']:.3f}")

            payload = {
                "phase": phase_name,
                "head": head,
                "selected_c": best_c,
                "n_valid_channels": int(valid_channels.sum()),
                "n_features": int(train_x.shape[1]),
                "remaining_inference": {
                    "accuracy": rem_results["accuracy"],
                    "macro_f1": rem_results["macro_f1"],
                    "confusion_matrix": rem_results["confusion_matrix"],
                },
                "heldout_inference": {
                    "accuracy": held_results["accuracy"],
                    "macro_f1": held_results["macro_f1"],
                    "confusion_matrix": held_results["confusion_matrix"],
                },
            }
            save_json(phase_dir / f"{head}_results.json", payload)

            phase_results[head] = {
                "remaining_inference": rem_results,
                "heldout_inference": held_results,
            }
            all_results[phase_name][head] = phase_results[head]

        plot_confusion_matrices(phase_dir, phase_results)

    # 7. Summary plot across all phases
    plot_accuracy_summary(args.output_dir, all_results)

    # 8. Save full results
    save_json(
        args.output_dir / "all_results.json",
        {
            phase: {
                head: {
                    split: {
                        k: v for k, v in res.items()
                        if k not in ("y_true", "y_pred")
                    }
                    for split, res in heads.items()
                }
                for head, heads in phases.items()
            }
            for phase, phases in all_results.items()
        },
    )

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
