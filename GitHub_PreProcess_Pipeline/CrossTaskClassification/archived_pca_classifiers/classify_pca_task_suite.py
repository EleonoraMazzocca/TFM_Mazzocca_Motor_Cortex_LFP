"""
Linear baseline for the original classification task suite using PCA features.

This script mirrors the structure of `data_classification.py`:
- 18 phase-specific tasks for each of PREREACH / REACH / GRASP
- 3 additional phase-classification tasks at the end

The default representation is a centered PCA trajectory:
1) fit PCA on training samples only
2) project each time point of each sample into PCA space
3) optionally center each trajectory at its first time point
4) flatten over time or mean-pool over time for a linear classifier
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Callable, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from data_paths import CLASS_FILES


ClassPredicate = Callable[[str], bool]
LabelFunction = Callable[[str], int | None]

PHASE_NAMES = ["PREREACH", "REACH", "GRASP"]
METRIC_NAMES = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def ensure_dir(path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)


def sample_class_array(class_name: str, max_trials_per_class: int, seed: int) -> np.ndarray:
    arr = np.load(CLASS_FILES[class_name], mmap_mode="r")
    if max_trials_per_class > 0 and arr.shape[0] > max_trials_per_class:
        rng = np.random.default_rng(seed + sum(ord(ch) for ch in class_name))
        idx = np.sort(rng.choice(arr.shape[0], size=max_trials_per_class, replace=False))
        out = np.array(arr[idx], copy=True).astype(np.float32, copy=False)
    else:
        out = np.array(arr, copy=True).astype(np.float32, copy=False)
    return out


def load_task_phase_dataset(
    phase: int,
    include_fn: ClassPredicate,
    label_fn: LabelFunction,
    max_trials_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_parts = []
    y_parts = []
    for class_name in CLASS_FILES:
        if not include_fn(class_name):
            continue
        label = label_fn(class_name)
        if label is None:
            continue
        arr = sample_class_array(class_name, max_trials_per_class=max_trials_per_class, seed=seed)
        X_parts.append(arr[:, phase])
        y_parts.append(np.full(arr.shape[0], label, dtype=np.int64))
    if not X_parts:
        raise ValueError("Task selection produced no samples.")
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def load_phase_classification_dataset(
    pp_filter: str,
    max_trials_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_parts = []
    y_parts = []
    for class_name in CLASS_FILES:
        if pp_filter != "ALL" and pp_filter not in class_name:
            continue
        arr = sample_class_array(class_name, max_trials_per_class=max_trials_per_class, seed=seed)
        for phase_idx in range(3):
            X_parts.append(arr[:, phase_idx])
            y_parts.append(np.full(arr.shape[0], phase_idx, dtype=np.int64))
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def build_split_features(X_train: np.ndarray, X_test: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    return build_split_features_with_options(
        X_train,
        X_test,
        n_components=n_components,
        center_mode="first",
        temporal_pooling="flatten",
    )


def build_split_features_with_options(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    center_mode: str,
    temporal_pooling: str,
) -> Tuple[np.ndarray, np.ndarray]:
    train_seq = np.transpose(X_train, (0, 2, 1))  # (n_samples, time, channels)
    test_seq = np.transpose(X_test, (0, 2, 1))

    x_fit = train_seq.reshape(-1, train_seq.shape[-1])
    pca = PCA(n_components=n_components, random_state=42).fit(x_fit)

    z_train = pca.transform(train_seq.reshape(-1, train_seq.shape[-1]))
    z_test = pca.transform(test_seq.reshape(-1, test_seq.shape[-1]))

    z_train = z_train.reshape(train_seq.shape[0], train_seq.shape[1], n_components)
    z_test = z_test.reshape(test_seq.shape[0], test_seq.shape[1], n_components)

    if center_mode == "first":
        z_train = z_train - z_train[:, :1, :]
        z_test = z_test - z_test[:, :1, :]
    elif center_mode != "none":
        raise ValueError(f"Unknown center_mode: {center_mode}")

    if temporal_pooling == "flatten":
        return z_train.reshape(z_train.shape[0], -1), z_test.reshape(z_test.shape[0], -1)
    if temporal_pooling == "mean":
        return np.mean(z_train, axis=1), np.mean(z_test, axis=1)
    raise ValueError(f"Unknown temporal_pooling: {temporal_pooling}")


def evaluate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    n_splits: int,
    test_size: float,
    seed: int,
    max_iter: int,
    center_mode: str,
    temporal_pooling: str,
) -> np.ndarray:
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, solver="lbfgs")),
        ]
    )
    scores = []
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = build_split_features_with_options(
            X[train_idx],
            X[test_idx],
            n_components=n_components,
            center_mode=center_mode,
            temporal_pooling=temporal_pooling,
        )
        y_train = y[train_idx]
        y_test = y[test_idx]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            pred,
            average="macro",
            zero_division=0,
        )
        scores.append([acc, precision, recall, f1])
    return np.mean(np.array(scores), axis=0)


def make_phase_task_defs() -> List[Tuple[str, ClassPredicate, LabelFunction]]:
    return [
        (
            "get_task_power_precision",
            lambda l: True,
            lambda l: 0 if "PRECISION" in l else 1 if "POWER" in l else None,
        ),
        (
            "get_task_power_precision_hand(hand=L)",
            lambda l: "_L_" in l,
            lambda l: 0 if "PRECISION" in l else 1 if "POWER" in l else None,
        ),
        (
            "get_task_power_precision_hand(hand=R)",
            lambda l: "_R_" in l,
            lambda l: 0 if "PRECISION" in l else 1 if "POWER" in l else None,
        ),
        (
            "get_task_power_precision_nobi",
            lambda l: "BIMANUAL" not in l,
            lambda l: 0 if "PRECISION" in l else 1 if "POWER" in l else None,
        ),
        (
            "get_task_angles_bimanual",
            lambda l: "BIMANUAL" in l,
            lambda l: 3 if "135_45" in l else 2 if "45_135" in l else 1 if "135" in l else 0 if "45" in l else None,
        ),
        (
            "get_task_left_right",
            lambda l: "UNIMANUAL" in l,
            lambda l: 0 if "_L_" in l else 1 if "_R_" in l else None,
        ),
        (
            "get_task_left_right_precision",
            lambda l: "PRECISION_UNIMANUAL" in l,
            lambda l: 0 if "_L_" in l else 1 if "_R_" in l else None,
        ),
        (
            "get_task_left_right_power",
            lambda l: "POWER_UNIMANUAL" in l,
            lambda l: 0 if "_L_" in l else 1 if "_R_" in l else None,
        ),
        (
            "get_task_angles_hand(hand=L, pp=ALL)",
            lambda l: "_L_" in l,
            lambda l: 0 if "_0" in l else 1 if "_45" in l else 2 if "_90" in l else 3 if "_135" in l else None,
        ),
        (
            "get_task_angles_hand(hand=L, pp=PRECISION)",
            lambda l: "_L_" in l and "PRECISION" in l,
            lambda l: 0 if "_0" in l else 1 if "_45" in l else 2 if "_90" in l else 3 if "_135" in l else None,
        ),
        (
            "get_task_angles_hand(hand=L, pp=POWER)",
            lambda l: "_L_" in l and "POWER" in l,
            lambda l: 0 if "_0" in l else 1 if "_45" in l else 2 if "_90" in l else 3 if "_135" in l else None,
        ),
        (
            "get_task_angles_hand(hand=R, pp=ALL)",
            lambda l: "_R_" in l,
            lambda l: 0 if "_0" in l else 1 if "_45" in l else 2 if "_90" in l else 3 if "_135" in l else None,
        ),
        (
            "get_task_angles_hand(hand=R, pp=PRECISION)",
            lambda l: "_R_" in l and "PRECISION" in l,
            lambda l: 0 if "_0" in l else 1 if "_45" in l else 2 if "_90" in l else 3 if "_135" in l else None,
        ),
        (
            "get_task_angles_hand(hand=R, pp=POWER)",
            lambda l: "_R_" in l and "POWER" in l,
            lambda l: 0 if "_0" in l else 1 if "_45" in l else 2 if "_90" in l else 3 if "_135" in l else None,
        ),
        (
            "get_task_angles_any_hand(pp=ALL)",
            lambda l: "UNIMANUAL" in l,
            lambda l: 0 if "_0" in l else 1 if "_45" in l else 2 if "_90" in l else 3 if "_135" in l else None,
        ),
        (
            "get_task_angles_any_hand(pp=PRECISION)",
            lambda l: "UNIMANUAL" in l and "PRECISION" in l,
            lambda l: 0 if "_0" in l else 1 if "_45" in l else 2 if "_90" in l else 3 if "_135" in l else None,
        ),
        (
            "get_task_angles_any_hand(pp=POWER)",
            lambda l: "UNIMANUAL" in l and "POWER" in l,
            lambda l: 0 if "_0" in l else 1 if "_45" in l else 2 if "_90" in l else 3 if "_135" in l else None,
        ),
        (
            "get_task_unimanual_bimanual",
            lambda l: "PRECISION" in l,
            lambda l: 0 if "UNIMANUAL" in l else 1 if "BIMANUAL" in l else None,
        ),
    ]


def resolve_selected_task_indices(task_indices: List[int] | None) -> List[int]:
    if not task_indices:
        return list(range(18))
    resolved = []
    for idx in task_indices:
        if idx < 1 or idx > 18:
            raise ValueError("Phase task indices must be between 1 and 18.")
        resolved.append(idx - 1)
    return resolved


def run_task_suite(args: argparse.Namespace) -> Tuple[List[List[np.ndarray]], List[List[str]]]:
    phase_task_defs = make_phase_task_defs()
    selected_task_indices = resolve_selected_task_indices(args.task_indices)
    scores: List[List[np.ndarray]] = [[] for _ in range(3)]
    task_name_blocks: List[List[str]] = [[] for _ in range(3)]

    for phase_idx in range(3):
        print("=" * 100)
        print("PHASE", PHASE_NAMES[phase_idx])
        print("=" * 100)
        for task_idx in selected_task_indices:
            task_name, include_fn, label_fn = phase_task_defs[task_idx]
            X, y = load_task_phase_dataset(
                phase=phase_idx,
                include_fn=include_fn,
                label_fn=label_fn,
                max_trials_per_class=args.max_trials_per_class,
                seed=args.seed,
            )
            res = evaluate_dataset(
                X=X,
                y=y,
                n_components=args.n_components,
                n_splits=args.n_splits,
                test_size=args.test_size,
                seed=args.seed,
                max_iter=args.max_iter,
                center_mode=args.center_mode,
                temporal_pooling=args.temporal_pooling,
            )
            scores[phase_idx].append(res)
            task_name_blocks[phase_idx].append(task_name)
            print(f"{task_name}\n\tFinal score {res}")

    if not args.skip_phase_classification_tasks:
        phase_extra_defs = [
            ("get_task_phases(pp=ALL)", "ALL"),
            ("get_task_phases(pp=PRECISION)", "PRECISION"),
            ("get_task_phases(pp=POWER)", "POWER"),
        ]
        for target_idx, (task_name, pp_filter) in enumerate(phase_extra_defs):
            X, y = load_phase_classification_dataset(
                pp_filter=pp_filter,
                max_trials_per_class=args.max_trials_per_class,
                seed=args.seed,
            )
            res = evaluate_dataset(
                X=X,
                y=y,
                n_components=args.n_components,
                n_splits=args.n_splits,
                test_size=args.test_size,
                seed=args.seed,
                max_iter=args.max_iter,
                center_mode=args.center_mode,
                temporal_pooling=args.temporal_pooling,
            )
            scores[target_idx].append(res)
            task_name_blocks[target_idx].append(task_name)
            print(f"{task_name}\n\tFinal score {res}")

    return scores, task_name_blocks


def format_summary_text(scores: List[List[np.ndarray]], task_name_blocks: List[List[str]], args: argparse.Namespace) -> str:
    lines = []
    lines.append("PCA trajectory task-suite classification summary")
    lines.append("")
    lines.append(f"n_components: {args.n_components}")
    lines.append(f"max_trials_per_class: {args.max_trials_per_class}")
    lines.append(f"center_mode: {args.center_mode}")
    lines.append(f"temporal_pooling: {args.temporal_pooling}")
    lines.append(f"n_splits: {args.n_splits}")
    lines.append(f"test_size: {args.test_size}")
    lines.append(f"seed: {args.seed}")
    if args.task_indices:
        lines.append(f"task_indices: {' '.join(str(x) for x in args.task_indices)}")
    lines.append(f"skip_phase_classification_tasks: {args.skip_phase_classification_tasks}")
    lines.append("")
    for phase_idx, phase_scores in enumerate(scores):
        lines.append(f"[{PHASE_NAMES[phase_idx]}]")
        for task_idx, metric_values in enumerate(phase_scores):
            metric_text = ", ".join(
                f"{name}={float(value):.6f}"
                for name, value in zip(METRIC_NAMES, metric_values)
            )
            lines.append(f"task_{task_idx + 1}: {task_name_blocks[phase_idx][task_idx]} | {metric_text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the original task suite with PCA-trajectory linear decoding.")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--max-trials-per-class", type=int, default=40)
    parser.add_argument("--center-mode", choices=["first", "none"], default="first")
    parser.add_argument("--temporal-pooling", choices=["flatten", "mean"], default="flatten")
    parser.add_argument(
        "--task-indices",
        nargs="+",
        type=int,
        help="Optional subset of the 18 phase-task indices to run, e.g. 1 2 3 4.",
    )
    parser.add_argument(
        "--skip-phase-classification-tasks",
        action="store_true",
        help="Skip the three extra phase-classification tasks appended at the end.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores, task_name_blocks = run_task_suite(args)

    out_dir = "GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/pca_task_suite_classification"
    ensure_dir(os.path.join(out_dir, "dummy.txt"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_text = format_summary_text(scores, task_name_blocks, args)
    txt_path = os.path.join(out_dir, f"pca_task_suite_{timestamp}.txt")
    latest_path = os.path.join(out_dir, "pca_task_suite_latest.txt")

    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write(summary_text)
    with open(latest_path, "w", encoding="utf-8") as fp:
        fp.write(summary_text)

    print(f"[DONE] TXT={txt_path}")
    print(f"[DONE] LATEST={latest_path}")


if __name__ == "__main__":
    main()
