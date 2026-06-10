from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data import ID_TO_ANGLE, ID_TO_GRIP, ID_TO_HAND, BalancedInstructionDataset, load_dataset, make_compositional_split
from evaluate import HEADS, evaluate_model, plot_confusion_matrix, plot_training_history, print_results
from instruction_encoding import get_instruction_dim
from model import LFPInstructionTransformer, LFPTransformerClassifier
from train import DEFAULT_CONFIG, train_model


# ---------------------------------------------------------------------------
# Experiment table
# ---------------------------------------------------------------------------
# Run | --encoding | --mask_prob | --out_dir                  | Key question
# ----+------------+-------------+----------------------------+-----------------------------
#   0 | none       | —           | results/run0_baseline       | Baseline, no instruction
#  1a | onehot     | 0.5         | results/run1a_onehot_50     | Structured label at 50/50
#  1b | onehot     | 0.7         | results/run1b_onehot_70     | More masking → better LFP use?
#  2a | bow        | 0.5         | results/run2a_bow_50        | Word co-occurrence > onehot?
#  2b | bow        | 0.7         | results/run2b_bow_70        |
#  3a | minilm     | 0.5         | results/run3a_minilm_50     | Semantic structure helps?
#  3b | minilm     | 0.7         | results/run3b_minilm_70     |
# ---------------------------------------------------------------------------
# Primary comparison: held-out angle accuracy across all runs.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a bimodal transformer (LFP + instruction) with balanced per-class "
            "instruction masking.  At test time the instruction is always the zero vector, "
            "so the model must decode from LFP alone."
        )
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/run0_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout_grip", type=int, default=1)
    parser.add_argument("--heldout_hand", type=int, default=1)
    parser.add_argument("--heldout_angle", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--feedforward_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument(
        "--encoding",
        choices=["none", "onehot", "bow", "minilm"],
        default="none",
        help="Instruction encoding.  'none' trains LFPTransformerClassifier (baseline).",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.5,
        help="Fraction of samples per class to mask (zero the instruction) each epoch. "
             "Typical values: 0.5 or 0.7.",
    )
    parser.add_argument(
        "--instruction_proj_dim",
        type=int,
        default=32,
        help="Dimension of the instruction projection before concat with LFP representation.",
    )
    return parser.parse_args()


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed everything so runs are reproducible across encodings and mask rates.
    # --seed controls the data split AND weight init / DataLoader shuffling.
    import random
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    heldout_label = (
        f"{ID_TO_GRIP[args.heldout_grip]}_"
        f"{ID_TO_HAND[args.heldout_hand]}_"
        f"{ID_TO_ANGLE[args.heldout_angle]}"
    )
    print(f"Held-out compositional combination: {heldout_label}")
    print(f"Encoding: {args.encoding}  mask_prob: {args.mask_prob}")

    data = load_dataset(
        cache_dir=args.cache_dir,
        heldout_grip=args.heldout_grip,
        heldout_hand=args.heldout_hand,
        heldout_angle=args.heldout_angle,
    )
    print(
        "Dataset loaded: "
        f"trials={len(data['y_grip'])} "
        f"channels={int(data['n_channels'])}"
    )

    train_data, val_data, seen_test_data, heldout_test_data, norm_stats = make_compositional_split(
        data,
        seed=args.seed,
    )

    # Training: balanced masking, reshuffled at the start of every epoch by train.py
    train_ds = BalancedInstructionDataset(
        train_data, norm_stats,
        encoding=args.encoding,
        mask_prob=args.mask_prob,
        is_test=False,
    )
    # Validation and test: instruction is always the zero vector
    val_ds = BalancedInstructionDataset(
        val_data, norm_stats,
        encoding=args.encoding,
        mask_prob=1.0,
        is_test=True,
    )
    seen_test_ds = BalancedInstructionDataset(
        seen_test_data, norm_stats,
        encoding=args.encoding,
        mask_prob=1.0,
        is_test=True,
    )
    heldout_test_ds = BalancedInstructionDataset(
        heldout_test_data, norm_stats,
        encoding=args.encoding,
        mask_prob=1.0,
        is_test=True,
    )

    use_instruction = args.encoding != "none"
    instruction_dim = get_instruction_dim(args.encoding)

    if use_instruction:
        model = LFPInstructionTransformer(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            feedforward_dim=args.feedforward_dim,
            dropout=args.dropout,
            instruction_dim=instruction_dim,
            instruction_proj_dim=args.instruction_proj_dim,
        )
    else:
        model = LFPTransformerClassifier(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            feedforward_dim=args.feedforward_dim,
            dropout=args.dropout,
        )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    config = {
        **DEFAULT_CONFIG,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
    }
    checkpoint_path = out_dir / "checkpoint.pt"
    model, history = train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        save_path=str(checkpoint_path),
        use_instruction=use_instruction,
    )

    seen_results = evaluate_model(
        model, seen_test_ds,
        batch_size=args.batch_size,
        use_instruction=use_instruction,
    )
    heldout_results = evaluate_model(
        model, heldout_test_ds,
        batch_size=args.batch_size,
        use_instruction=use_instruction,
    )

    print_results(seen_results, title="Seen combinations")
    print_results(heldout_results, title="Held-out combinations")

    for head, names in HEADS.items():
        np.save(out_dir / f"seen_{head}_confusion_matrix.npy", seen_results[head]["confusion_matrix"])
        np.save(out_dir / f"heldout_{head}_confusion_matrix.npy", heldout_results[head]["confusion_matrix"])
        _save_text(out_dir / f"seen_{head}_report.txt", seen_results[head]["report"])
        _save_text(out_dir / f"heldout_{head}_report.txt", heldout_results[head]["report"])

    np.savez_compressed(out_dir / "normalization_stats.npz", **norm_stats)

    summary = {
        "heldout_label": heldout_label,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "seen_test_size": len(seen_test_ds),
        "heldout_test_size": len(heldout_test_ds),
        "seen_accuracy": {h: seen_results[h]["accuracy"] for h in HEADS},
        "heldout_accuracy": {h: heldout_results[h]["accuracy"] for h in HEADS},
        "encoding": args.encoding,
        "mask_prob": args.mask_prob,
        "instruction_dim": instruction_dim,
        "config": config,
        "model": {
            "type": "LFPInstructionTransformer" if use_instruction else "LFPTransformerClassifier",
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "feedforward_dim": args.feedforward_dim,
            "dropout": args.dropout,
            "instruction_dim": instruction_dim,
            "instruction_proj_dim": args.instruction_proj_dim if use_instruction else None,
            "n_params": n_params,
        },
    }
    _save_text(out_dir / "summary.json", json.dumps(summary, indent=2))

    if not args.no_plot:
        for head, names in HEADS.items():
            plot_confusion_matrix(
                seen_results[head]["confusion_matrix"],
                target_names=names,
                title=f"{head} — seen combinations",
                save_path=str(out_dir / f"seen_{head}_confusion_matrix.png"),
            )
            plot_confusion_matrix(
                heldout_results[head]["confusion_matrix"],
                target_names=names,
                title=f"{head} — held-out combinations",
                save_path=str(out_dir / f"heldout_{head}_confusion_matrix.png"),
            )
        plot_training_history(
            history,
            save_path=str(out_dir / "training_curves.png"),
        )


if __name__ == "__main__":
    main()
