"""Train a cVAE on joint-transformer embeddings.

This is the second stage of the joint embedding pipeline:

  segmented data -> trained JointFactorTransformer -> pooled embeddings -> cVAE

The cVAE holds out one full (phase, grip, hand) combination and generates the
missing joint-transformer embedding for that condition.  The input mode must
match the transformer checkpoint: "mu" for 1-band MU channel tokens or
"broadband6" for 6-band broadband-derived channel tokens.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
from transformer_encoder.joint_embedding_data import INPUT_MODES  # noqa: E402
from transformer_encoder.joint_embedding_data import PHASE_NAMES  # noqa: E402
import cvae.embedding_cvae_pipeline as embedding_cvae_pipeline  # noqa: E402


def expected_cvae_run_name(args: argparse.Namespace) -> str:
    return f"cvae_{args.heldout_phase}_{args.heldout_grip}_{args.heldout_hand}"


def validate_output_dir(args: argparse.Namespace, out_dir: Path) -> None:
    expected = expected_cvae_run_name(args)
    actual = out_dir.name
    if actual != expected and not actual.startswith(f"{expected}_"):
        raise SystemExit(
            "Refusing to write cVAE outputs to mismatched --out_dir.\n"
            f"  heldout combo: {args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}\n"
            f"  expected folder name: {expected} or {expected}_...\n"
            f"  got folder name: {actual}\n"
            "Use a matching --out_dir so checkpoint metadata and result files cannot diverge."
        )


def validate_joint_checkpoint(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.joint_checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"--joint_checkpoint does not exist: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})

    mismatches = []
    expected = {
        "input_mode": args.input_mode,
        "heldout_phase": args.heldout_phase,
        "heldout_grip": args.heldout_grip,
        "heldout_hand": args.heldout_hand,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            mismatches.append(f"{key}: checkpoint={config.get(key)!r}, requested={value!r}")
    if config.get("heldout") is not True:
        mismatches.append(f"heldout: checkpoint={config.get('heldout')!r}, requested=True")

    if mismatches:
        joined = "\n  ".join(mismatches)
        raise SystemExit(
            "Refusing to run cVAE with a joint checkpoint that does not match the requested held-out combo.\n"
            f"  checkpoint: {checkpoint_path}\n"
            f"  {joined}"
        )


def load_saved_data_dir(out_dir: str) -> str:
    """Recover the original data directory from a saved checkpoint."""
    checkpoint_path = Path(out_dir) / "checkpoint.pt"
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved_data_dir = ckpt.get("args", {}).get("data_dir")
    if not saved_data_dir:
        raise SystemExit("--diag_only requires --data_dir because the checkpoint does not save it")
    return saved_data_dir


def append_optional_flags(args: argparse.Namespace, forwarded: list[str]) -> list[str]:
    """Keep the wrapper small by handling simple on/off forwarding in one place."""
    if args.sentence_condition_path:
        forwarded += ["--sentence_condition_path", args.sentence_condition_path]
    if args.sentence_key_order_path:
        forwarded += ["--sentence_key_order_path", args.sentence_key_order_path]
    if args.baseline_dirs:
        forwarded += ["--baseline_dirs", *args.baseline_dirs]
    if args.aug_dirs:
        forwarded += ["--aug_dirs", *args.aug_dirs]

    flag_map = {
        "no_plot": "--no_plot",
        "dry_run": "--dry_run",
        "denoising_aug": "--denoising_aug",
        "cond_dropout": "--cond_dropout",
        "mmd_loss": "--mmd_loss",
        "no_early_stopping": "--no_early_stopping",
    }
    for attr, flag in flag_map.items():
        if getattr(args, attr, False):
            forwarded.append(flag)
    return forwarded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--joint_checkpoint", type=str, default=None,
                   help="Required unless --compare_only is set.")
    p.add_argument("--input_mode", choices=INPUT_MODES, default=None)
    p.add_argument("--joint_cache_dir", type=str, default="/tmp/lfp_joint_embedding_cache")
    p.add_argument("--heldout_phase", choices=PHASE_NAMES, default="grasp")
    p.add_argument("--heldout_grip", choices=["power", "precision"], default="precision")
    p.add_argument("--heldout_hand", choices=["left", "right"], default="right")
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--beta_max", type=float, default=1.0)
    p.add_argument("--beta_anneal_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Defaults to joint_embedding/results/{input_mode}/cvae_{phase}_{grip}_{hand}.",
    )
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    # Denoising augmentation and experiment control.
    p.add_argument("--denoising_aug", action="store_true",
                   help="Enable denoising augmentation during cVAE training.")
    p.add_argument("--noise_scale", type=float, default=0.1)
    p.add_argument("--amplitude_scale_range", type=float, nargs=2, default=[0.85, 1.15],
                   metavar=("LOW", "HIGH"))
    p.add_argument("--aug_n_dropout_dims", type=int, default=2)
    p.add_argument("--cond_dropout", action="store_true",
                   help="Enable partial condition dropout on decoder during training.")
    p.add_argument("--p_cond_single", type=float, default=0.15,
                   help="Per-factor single-dropout probability for phase, grip, and hand.")
    p.add_argument("--p_cond_double", type=float, default=0.04,
                   help="Per-pair double-dropout probability.")
    p.add_argument("--p_cond_all", type=float, default=0.03,
                   help="Full condition dropout probability.")
    p.add_argument("--free_bits", type=float, default=0.0,
                   help="Per-dimension minimum KL (nats). Uses a clamped KL for the loss "
                        "while reporting raw KL in logs. 0.0 = disabled (default). Try 0.5 or 1.0.")
    p.add_argument("--mmd_loss", action="store_true",
                   help="Replace ELBO KL with aggregate MMD loss (MMD-VAE). "
                        "Cannot be combined with --free_bits.")
    p.add_argument("--lambda_mmd", type=float, default=10.0,
                   help="MMD loss weight. Sweep 1, 3, 10, 30.")
    p.add_argument("--condition_type", choices=["onehot", "sentence"], default="onehot",
                   help="Condition vector type. 'sentence' uses 5-dim PCA embedding (Option D).")
    p.add_argument("--sentence_condition_path", type=str, default=None,
                   help="Path to condition_vectors_D_pca5.npy (12 × condition_dim).")
    p.add_argument("--sentence_key_order_path", type=str, default=None,
                   help="Path to condition_keys_D_pca5.npy (12 × 3).")
    p.add_argument("--no_early_stopping", action="store_true")
    p.add_argument("--split_seed", type=int, default=42,
                   help="Seed for train/val split. Fixed across seeds; only --seed varies.")
    p.add_argument("--diag_only", action="store_true",
                   help="Skip training. Load checkpoint and recompute diagnostics only.")
    p.add_argument("--compare_only", action="store_true",
                   help="Skip training. Generate comparison plots from completed run dirs.")
    p.add_argument("--baseline_dirs", type=str, nargs="+", default=None)
    p.add_argument("--aug_dirs", type=str, nargs="+", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    # These modes reuse the implementation module directly and only need
    # minimal wrapper validation.
    if args.compare_only:
        if not args.out_dir or not args.baseline_dirs or not args.aug_dirs:
            raise SystemExit("--compare_only requires --out_dir, --baseline_dirs, and --aug_dirs")
        forwarded = [
            "--compare_only",
            "--out_dir", args.out_dir,
            "--baseline_dirs", *args.baseline_dirs,
            "--aug_dirs", *args.aug_dirs,
        ]
        return embedding_cvae_pipeline.main(forwarded)

    if args.diag_only:
        if not args.out_dir:
            raise SystemExit("--diag_only requires --out_dir")
        forwarded = [
            "--diag_only",
            "--out_dir", args.out_dir,
            "--device", args.device,
        ]
        if args.data_dir:
            forwarded += ["--data_dir", args.data_dir]
        else:
            forwarded += ["--data_dir", load_saved_data_dir(args.out_dir)]
        return embedding_cvae_pipeline.main(forwarded)

    # Normal training requires --data_dir, --joint_checkpoint, --input_mode.
    if not args.data_dir:
        raise SystemExit("--data_dir is required unless --compare_only is set")
    if not args.joint_checkpoint:
        raise SystemExit("--joint_checkpoint is required unless --compare_only is set")
    if not args.input_mode:
        raise SystemExit("--input_mode is required unless --compare_only is set")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            _HERE
            / "results"
            / args.input_mode
            / f"cvae_{args.heldout_phase}_{args.heldout_grip}_{args.heldout_hand}"
        )

    out_dir = Path(out_dir)

    validate_output_dir(args, out_dir)
    validate_joint_checkpoint(args)

    forwarded = [
        "--data_dir", args.data_dir,
        "--joint_checkpoint", args.joint_checkpoint,
        "--joint_input_mode", args.input_mode,
        "--joint_cache_dir", args.joint_cache_dir,
        "--heldout_phase", args.heldout_phase,
        "--heldout_grip", args.heldout_grip,
        "--heldout_hand", args.heldout_hand,
        "--latent_dim", str(args.latent_dim),
        "--dropout", str(args.dropout),
        "--beta_max", str(args.beta_max),
        "--beta_anneal_epochs", str(args.beta_anneal_epochs),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--split_seed", str(args.split_seed),
        "--out_dir", str(out_dir),
        "--device", args.device,
        "--hidden_dims", *[str(v) for v in args.hidden_dims],
        "--noise_scale", str(args.noise_scale),
        "--amplitude_scale_range", str(args.amplitude_scale_range[0]), str(args.amplitude_scale_range[1]),
        "--aug_n_dropout_dims", str(args.aug_n_dropout_dims),
        "--p_cond_single", str(args.p_cond_single),
        "--p_cond_double", str(args.p_cond_double),
        "--p_cond_all", str(args.p_cond_all),
        "--free_bits", str(args.free_bits),
        "--lambda_mmd", str(args.lambda_mmd),
        "--condition_type", args.condition_type,
    ]
    forwarded = append_optional_flags(args, forwarded)
    return embedding_cvae_pipeline.main(forwarded)


if __name__ == "__main__":
    main()
