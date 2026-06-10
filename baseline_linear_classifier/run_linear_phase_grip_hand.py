"""Linear baseline for the joint phase/grip/hand transformer task.

This script intentionally mirrors the joint-transformer data path:

  class files -> phase expansion -> same MU/broadband6 feature extraction ->
  same held-out phase/grip/hand split -> three logistic-regression heads

It is meant as the classical baseline for `transformer_encoder.run_joint_embedding`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from transformer_encoder.data import GRIP_TO_ID, HAND_TO_ID, PHASE_NAMES
from transformer_encoder.joint_embedding_data import (
    CHANNEL_VALID,
    INPUT_MODES,
    extract_and_cache_features,
    load_joint_trials,
    phase_expand,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ID_TO_GRIP = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND = {v: k for k, v in HAND_TO_ID.items()}
HEAD_NAMES = {
    "phase": PHASE_NAMES,
    "grip": [ID_TO_GRIP[i] for i in range(len(ID_TO_GRIP))],
    "hand": [ID_TO_HAND[i] for i in range(len(ID_TO_HAND))],
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Logistic-regression baseline for phase, grip, and hand."
    )
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--input_mode", choices=INPUT_MODES, default="mu")
    p.add_argument("--cache_dir", type=str, default="/tmp/lfp_linear_baseline_cache")
    p.add_argument("--out_dir", type=str, default=str(SCRIPT_DIR / "results" / "linear_phase_grip_hand"))
    split = p.add_mutually_exclusive_group()
    split.add_argument("--heldout", action="store_true", default=True)
    split.add_argument("--no_heldout", action="store_false", dest="heldout")
    p.add_argument("--heldout_phase", choices=PHASE_NAMES, default="grasp")
    p.add_argument("--heldout_grip", choices=["power", "precision"], default="precision")
    p.add_argument("--heldout_hand", choices=["left", "right"], default="right")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_iter", "--max-iter", type=int, default=2000)
    p.add_argument("--c_values", "--c-values", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0])
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args(argv)


def _train_test_split_maybe_stratified(
    idx: np.ndarray,
    test_size: float,
    random_state: int,
    stratify: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if stratify is not None:
        _, counts = np.unique(stratify, return_counts=True)
        if len(counts) == 0 or counts.min() < 2:
            stratify = None
    return train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )


def heldout_split_indices(
    flat: dict,
    heldout_phase: int,
    heldout_grip: int,
    heldout_hand: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(flat["y_phase"]))
    heldout_mask = (
        (flat["y_phase"] == heldout_phase)
        & (flat["y_grip"] == heldout_grip)
        & (flat["y_hand"] == heldout_hand)
    )
    heldout_idx = all_idx[heldout_mask]
    remaining_idx = all_idx[~heldout_mask]
    if len(heldout_idx) < 2:
        raise ValueError("Held-out phase/grip/hand combination has fewer than two samples.")

    strat_remaining = (
        flat["y_phase"][remaining_idx].astype(np.int64) * 16
        + flat["y_grip"][remaining_idx].astype(np.int64) * 8
        + flat["y_hand"][remaining_idx].astype(np.int64) * 4
        + flat["y_angle"][remaining_idx].astype(np.int64)
    )
    train_idx, temp_idx = _train_test_split_maybe_stratified(
        remaining_idx,
        test_size=0.2,
        random_state=seed,
        stratify=strat_remaining,
    )
    strat_temp = (
        flat["y_phase"][temp_idx].astype(np.int64) * 16
        + flat["y_grip"][temp_idx].astype(np.int64) * 8
        + flat["y_hand"][temp_idx].astype(np.int64) * 4
        + flat["y_angle"][temp_idx].astype(np.int64)
    )
    seen_val_idx, seen_test_idx = _train_test_split_maybe_stratified(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=strat_temp,
    )
    held_val_idx, held_test_idx = train_test_split(
        heldout_idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
    )
    val_idx = np.concatenate([seen_val_idx, held_val_idx])
    return np.sort(train_idx), np.sort(val_idx), np.sort(seen_test_idx), np.sort(held_test_idx)


def normal_split_indices(flat: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(flat["y_grip"]))
    strat = (
        flat["y_phase"].astype(np.int64) * 32
        + flat["y_grip"].astype(np.int64) * 16
        + flat["y_hand"].astype(np.int64) * 8
        + flat["y_angle"].astype(np.int64)
    )
    train_idx, temp_idx = _train_test_split_maybe_stratified(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=strat,
    )
    strat_temp = (
        flat["y_phase"][temp_idx].astype(np.int64) * 32
        + flat["y_grip"][temp_idx].astype(np.int64) * 16
        + flat["y_hand"][temp_idx].astype(np.int64) * 8
        + flat["y_angle"][temp_idx].astype(np.int64)
    )
    val_idx, test_idx = _train_test_split_maybe_stratified(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=strat_temp,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def flatten_features(features: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Flatten channel-token features and standardize from training samples only."""
    valid = CHANNEL_VALID.reshape(-1)
    x = np.asarray(features, dtype=np.float32).reshape(len(features), -1, features.shape[-1])
    x = x[:, valid, :].reshape(len(features), -1)

    train_x = x[train_idx]
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    zero_mask = x == 0.0
    x = (x - mu) / sigma
    x[zero_mask] = 0.0
    return x.astype(np.float32), {"mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32)}


def choose_best_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_values: Sequence[float],
    max_iter: int,
) -> tuple[float, LogisticRegression, float]:
    best_c = float(c_values[0])
    best_model: LogisticRegression | None = None
    best_f1 = -np.inf
    for c in c_values:
        model = LogisticRegression(
            C=float(c),
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            n_jobs=None,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        score = f1_score(y_val, pred, average="macro", zero_division=0)
        print(f"    C={float(c):g} val_macro_f1={score:.4f}")
        if score > best_f1:
            best_c = float(c)
            best_model = model
            best_f1 = float(score)
    if best_model is None:
        raise RuntimeError("No logistic-regression model was trained.")
    print(f"    selected C={best_c:g}")
    return best_c, best_model, best_f1


def evaluate(model: LogisticRegression, x: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(len(labels))).tolist(),
    }


def save_confusion_plot(path: Path, matrix: list[list[int]], labels: list[str], title: str) -> None:
    mat = np.asarray(matrix, dtype=np.int64)
    row_sums = mat.sum(axis=1, keepdims=True).astype(np.float64)
    norm = np.divide(mat, row_sums, out=np.zeros_like(mat, dtype=np.float64), where=row_sums != 0)

    fig_size = max(4.0, 1.1 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input_mode} class files from {args.data_dir}")
    trials = load_joint_trials(Path(args.data_dir), args.input_mode)
    flat = phase_expand(trials)
    features = extract_and_cache_features(flat, Path(args.cache_dir))

    if args.heldout:
        train_idx, val_idx, seen_test_idx, heldout_test_idx = heldout_split_indices(
            flat,
            PHASE_NAMES.index(args.heldout_phase),
            GRIP_TO_ID[args.heldout_grip],
            HAND_TO_ID[args.heldout_hand],
            args.seed,
        )
    else:
        train_idx, val_idx, seen_test_idx = normal_split_indices(flat, args.seed)
        heldout_test_idx = None

    print(
        "Split sizes: "
        f"train={len(train_idx)} val={len(val_idx)} seen_test={len(seen_test_idx)} "
        f"heldout_test={0 if heldout_test_idx is None else len(heldout_test_idx)}"
    )

    x, stats = flatten_features(features, train_idx)
    print(f"Feature matrix: {x.shape} ({args.input_mode}, valid channel tokens only)")

    targets = {
        "phase": flat["y_phase"],
        "grip": flat["y_grip"],
        "hand": flat["y_hand"],
    }
    split_indices = {
        "val": val_idx,
        "seen_test": seen_test_idx,
    }
    if heldout_test_idx is not None:
        split_indices["heldout_test"] = heldout_test_idx

    all_results: dict[str, object] = {
        "config": {
            "data_dir": str(args.data_dir),
            "input_mode": args.input_mode,
            "heldout": bool(args.heldout),
            "heldout_phase": args.heldout_phase,
            "heldout_grip": args.heldout_grip,
            "heldout_hand": args.heldout_hand,
            "seed": int(args.seed),
            "c_values": [float(v) for v in args.c_values],
            "max_iter": int(args.max_iter),
        },
        "splits": {name: int(len(idx)) for name, idx in split_indices.items()} | {"train": int(len(train_idx))},
        "heads": {},
    }

    for head, y in targets.items():
        print(f"\nHead: {head}")
        labels = HEAD_NAMES[head]
        selected_c, model, val_f1 = choose_best_model(
            x[train_idx],
            y[train_idx],
            x[val_idx],
            y[val_idx],
            args.c_values,
            args.max_iter,
        )

        head_results: dict[str, object] = {
            "selected_c": selected_c,
            "val_macro_f1_for_c_selection": val_f1,
            "labels": labels,
        }
        for split_name, idx in split_indices.items():
            result = evaluate(model, x[idx], y[idx], labels)
            head_results[split_name] = result
            print(
                f"  {split_name:12s} "
                f"acc={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}"
            )
            if not args.no_plot:
                save_confusion_plot(
                    out_dir / f"confusion_{head}_{split_name}.png",
                    result["confusion_matrix"],
                    labels,
                    f"{head} | {split_name}",
                )

        save_json(out_dir / f"{head}_results.json", head_results)
        all_results["heads"][head] = head_results

    np.savez_compressed(
        out_dir / "normalization_stats.npz",
        mu=stats["mu"],
        sigma=stats["sigma"],
    )
    save_json(out_dir / "summary.json", all_results)
    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
