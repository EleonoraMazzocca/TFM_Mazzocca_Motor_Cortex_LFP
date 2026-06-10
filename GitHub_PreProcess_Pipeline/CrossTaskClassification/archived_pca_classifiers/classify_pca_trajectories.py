"""
Linear baseline for classifying PCA trajectories.

This companion script decodes one comparison at a time using:
1) raw trial tensors from the separated class files
2) PCA fit on training data only
3) centered PCA trajectories per trial
4) flattened trajectory features
5) a linear classifier (logistic regression)

The comparison definitions mirror the PCA plotting scripts so visual inspection
and classification can be compared directly.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

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


@dataclass
class Comparison:
    name: str
    group_a_label: str
    group_a_classes: List[str]
    group_b_label: str
    group_b_classes: List[str]


def ensure_dir(path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)


def sanitize(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("-", "_")
    )


def build_comparisons() -> List[Comparison]:
    power_all = [k for k in CLASS_FILES if k.startswith("POWER_")]
    precision_all = [k for k in CLASS_FILES if k.startswith("PRECISION_")]

    power_uni_l = [k for k in CLASS_FILES if k.startswith("POWER_UNIMANUAL_L_")]
    power_uni_r = [k for k in CLASS_FILES if k.startswith("POWER_UNIMANUAL_R_")]
    precision_uni_l = [k for k in CLASS_FILES if k.startswith("PRECISION_UNIMANUAL_L_")]
    precision_uni_r = [k for k in CLASS_FILES if k.startswith("PRECISION_UNIMANUAL_R_")]

    power_uni_l_0 = ["POWER_UNIMANUAL_L_0"]
    power_uni_l_others = ["POWER_UNIMANUAL_L_45", "POWER_UNIMANUAL_L_90", "POWER_UNIMANUAL_L_135"]
    precision_uni_l_0 = ["PRECISION_UNIMANUAL_L_0"]
    precision_uni_l_others = ["PRECISION_UNIMANUAL_L_45", "PRECISION_UNIMANUAL_L_90", "PRECISION_UNIMANUAL_L_135"]

    precision_bi_45 = ["PRECISION_BIMANUAL_45"]
    precision_bi_others = ["PRECISION_BIMANUAL_135", "PRECISION_BIMANUAL_45_135", "PRECISION_BIMANUAL_135_45"]
    precision_bimanual = [k for k in CLASS_FILES if k.startswith("PRECISION_BIMANUAL_")]
    precision_unimanual = [k for k in CLASS_FILES if k.startswith("PRECISION_UNIMANUAL_")]

    comps: List[Comparison] = [
        Comparison("1_power_all_vs_precision_all", "Power all", power_all, "Precision all", precision_all),
        Comparison("2_power_uni_left_vs_precision_uni_left", "Power uni left", power_uni_l, "Precision uni left", precision_uni_l),
        Comparison("3_power_uni_right_vs_precision_uni_right", "Power uni right", power_uni_r, "Precision uni right", precision_uni_r),
        Comparison("4_power_uni_left_vs_power_uni_right", "Power uni left", power_uni_l, "Power uni right", power_uni_r),
        Comparison("5_precision_uni_left_vs_precision_uni_right", "Precision uni left", precision_uni_l, "Precision uni right", precision_uni_r),
        Comparison("6_power_uni_left_0_vs_left_45_90_135", "Power left 0", power_uni_l_0, "Power left 45/90/135", power_uni_l_others),
        Comparison("8_precision_uni_left_0_vs_left_45_90_135", "Precision left 0", precision_uni_l_0, "Precision left 45/90/135", precision_uni_l_others),
        Comparison("10_precision_bimanual_45_45_vs_other_bimanual_angles", "Precision bi 45-45", precision_bi_45, "Precision bi other angles", precision_bi_others),
        Comparison("11_precision_bimanual_vs_precision_unimanual", "Precision bimanual", precision_bimanual, "Precision unimanual", precision_unimanual),
    ]

    for deg in ["0", "45", "90", "135"]:
        comps.append(
            Comparison(
                f"7_power_left_vs_right_{deg}",
                f"Power left {deg}",
                [f"POWER_UNIMANUAL_L_{deg}"],
                f"Power right {deg}",
                [f"POWER_UNIMANUAL_R_{deg}"],
            )
        )

    for deg in ["0", "45", "90", "135"]:
        comps.append(
            Comparison(
                f"9_precision_left_vs_right_{deg}",
                f"Precision left {deg}",
                [f"PRECISION_UNIMANUAL_L_{deg}"],
                f"Precision right {deg}",
                [f"PRECISION_UNIMANUAL_R_{deg}"],
            )
        )

    return comps


def load_trials(classes: List[str], max_trials_per_class: int = 0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for c in classes:
        arr = np.load(CLASS_FILES[c], mmap_mode="r")
        if max_trials_per_class > 0 and arr.shape[0] > max_trials_per_class:
            idx = np.sort(rng.choice(arr.shape[0], size=max_trials_per_class, replace=False))
            part = np.array(arr[idx], copy=True).astype(np.float32, copy=False)
        else:
            part = np.array(arr, copy=True).astype(np.float32, copy=False)
        chunks.append(part)
    return np.concatenate(chunks, axis=0)


def make_binary_dataset(comp: Comparison, max_trials_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    a_trials = load_trials(comp.group_a_classes, max_trials_per_class=max_trials_per_class, seed=seed)
    b_trials = load_trials(comp.group_b_classes, max_trials_per_class=max_trials_per_class, seed=seed)
    X = np.concatenate([a_trials, b_trials], axis=0)
    y = np.concatenate(
        [
            np.zeros(a_trials.shape[0], dtype=np.int64),
            np.ones(b_trials.shape[0], dtype=np.int64),
        ],
        axis=0,
    )
    return X, y


def phase_indices(phase_mode: str) -> List[int]:
    if phase_mode == "all":
        return [0, 1, 2]
    return [int(phase_mode)]


def build_split_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    phase_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_blocks = []
    test_blocks = []
    for ph in phase_indices(phase_mode):
        train_phase = np.transpose(X_train[:, ph], (0, 2, 1))  # (n_trials, time, channels)
        test_phase = np.transpose(X_test[:, ph], (0, 2, 1))

        x_fit = train_phase.reshape(-1, train_phase.shape[-1])
        pca = PCA(n_components=n_components, random_state=42).fit(x_fit)

        z_train = pca.transform(train_phase.reshape(-1, train_phase.shape[-1]))
        z_test = pca.transform(test_phase.reshape(-1, test_phase.shape[-1]))

        z_train = z_train.reshape(train_phase.shape[0], train_phase.shape[1], n_components)
        z_test = z_test.reshape(test_phase.shape[0], test_phase.shape[1], n_components)

        z_train = z_train - z_train[:, :1, :]
        z_test = z_test - z_test[:, :1, :]

        train_blocks.append(z_train.reshape(z_train.shape[0], -1))
        test_blocks.append(z_test.reshape(z_test.shape[0], -1))

    return np.concatenate(train_blocks, axis=1), np.concatenate(test_blocks, axis=1)


def evaluate_comparison(
    comp: Comparison,
    n_components: int,
    phase_mode: str,
    n_splits: int,
    test_size: float,
    max_trials_per_class: int,
    seed: int,
    max_iter: int,
) -> Dict[str, float]:
    X, y = make_binary_dataset(comp, max_trials_per_class=max_trials_per_class, seed=seed)

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    metrics = []

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, solver="lbfgs")),
        ]
    )

    for train_idx, test_idx in splitter.split(X, y):
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train_feat, X_test_feat = build_split_features(
            X_train_raw,
            X_test_raw,
            n_components=n_components,
            phase_mode=phase_mode,
        )

        clf.fit(X_train_feat, y_train)
        pred = clf.predict(X_test_feat)

        acc = accuracy_score(y_test, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            pred,
            average="macro",
            zero_division=0,
        )
        metrics.append([acc, precision, recall, f1])

    metric_arr = np.mean(np.array(metrics), axis=0)
    return {
        "accuracy": float(metric_arr[0]),
        "precision_macro": float(metric_arr[1]),
        "recall_macro": float(metric_arr[2]),
        "f1_macro": float(metric_arr[3]),
        "n_samples": int(X.shape[0]),
        "n_group_a": int(np.sum(y == 0)),
        "n_group_b": int(np.sum(y == 1)),
        "n_components": int(n_components),
        "phase_mode": phase_mode,
    }


def results_to_text(comp: Comparison, results: List[Dict[str, float]], args: argparse.Namespace) -> str:
    lines = []
    lines.append("PCA trajectory linear classification summary")
    lines.append("")
    lines.append(f"comparison: {comp.name}")
    lines.append(f"group_a: {comp.group_a_label}")
    lines.append(f"group_b: {comp.group_b_label}")
    lines.append(f"phase_mode: {args.phase_mode}")
    lines.append(f"n_splits: {args.n_splits}")
    lines.append(f"test_size: {args.test_size}")
    lines.append(f"max_trials_per_class: {args.max_trials_per_class}")
    lines.append(f"seed: {args.seed}")
    lines.append("")
    for res in results:
        lines.append(
            "n_components={n_components}: accuracy={accuracy:.6f}, "
            "precision_macro={precision_macro:.6f}, "
            "recall_macro={recall_macro:.6f}, "
            "f1_macro={f1_macro:.6f}, "
            "n_samples={n_samples} (A={n_group_a}, B={n_group_b})".format(**res)
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify PCA trajectories with a linear model.")
    parser.add_argument(
        "--comparison",
        default="1_power_all_vs_precision_all",
        help="Comparison name to decode.",
    )
    parser.add_argument(
        "--phase-mode",
        choices=["0", "1", "2", "all"],
        default="all",
        help="Use one phase only or concatenate all phases.",
    )
    parser.add_argument(
        "--n-components-list",
        nargs="+",
        type=int,
        default=[2, 3, 5, 10],
        help="One or more PCA dimensionalities to evaluate.",
    )
    parser.add_argument("--max-trials-per-class", type=int, default=80)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparisons = build_comparisons()
    comps_by_name = {c.name: c for c in comparisons}
    if args.comparison not in comps_by_name:
        valid = "\n".join(sorted(comps_by_name.keys()))
        raise ValueError(f"Unknown comparison '{args.comparison}'. Valid names:\n{valid}")

    comp = comps_by_name[args.comparison]
    out_dir = "GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/pca_trajectory_classification"
    ensure_dir(os.path.join(out_dir, "dummy.txt"))

    results = []
    for n_components in args.n_components_list:
        res = evaluate_comparison(
            comp=comp,
            n_components=n_components,
            phase_mode=args.phase_mode,
            n_splits=args.n_splits,
            test_size=args.test_size,
            max_trials_per_class=args.max_trials_per_class,
            seed=args.seed,
            max_iter=args.max_iter,
        )
        results.append(res)
        print(
            f"[RESULT] {comp.name} | phase_mode={args.phase_mode} | n_components={n_components} | "
            f"acc={res['accuracy']:.4f} | f1={res['f1_macro']:.4f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = sanitize(comp.name)
    txt_path = os.path.join(out_dir, f"{base_name}_{timestamp}.txt")
    latest_path = os.path.join(out_dir, f"{base_name}_latest.txt")
    json_path = os.path.join(out_dir, f"{base_name}_{timestamp}.json")

    text = results_to_text(comp, results, args)
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write(text)
    with open(latest_path, "w", encoding="utf-8") as fp:
        fp.write(text)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "comparison": comp.name,
                "group_a_label": comp.group_a_label,
                "group_b_label": comp.group_b_label,
                "args": vars(args),
                "results": results,
            },
            fp,
            indent=2,
        )

    print(f"[DONE] TXT={txt_path}")
    print(f"[DONE] LATEST={latest_path}")
    print(f"[DONE] JSON={json_path}")


if __name__ == "__main__":
    main()
