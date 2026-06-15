"""Pipeline implementation for embedding-space cVAE experiments.

This is a representation-space generative control.  With --joint_checkpoint,
it trains a cVAE to generate a joint transformer's pooled embedding
conditioned on the same factors the joint transformer was trained to predict:
(phase, grip, hand).

It does not generate LFP.  It tests whether compositional structure is easier
to model in the learned task-relevant representation space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


from transformer_encoder.joint_embedding_data import PHASE_NAMES, GRIP_TO_ID, HAND_TO_ID
from cvae.conditioning.onehot import make_condition_vector
from cvae.conditioning.sentence import lookup_condition
from cvae.cvae_model import LFPCVAE
from cvae.embedding_cvae_eval import (
    compute_collapse_diagnostics,
    evaluate_generation,
    reconstruct_seen,
)
from cvae.embedding_cvae_plots import (
    generate_comparison_plots,
    plot_generation_diagnostics,
)
from cvae.training import train_cvae
from transformer_encoder.joint_embedding_data import (  # noqa: E402
    BAND_NAMES_6,
    JointEmbeddingDataset,
    extract_and_cache_features,
    load_joint_trials,
    phase_expand,
    subset_flat,
)
from transformer_encoder.joint_embedding_model import JointFactorTransformer  # noqa: E402


ID_TO_GRIP = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND = {v: k for k, v in HAND_TO_ID.items()}


# ---------------------------------------------------------------------------
# Dataset Wrapper
# ---------------------------------------------------------------------------


class EmbeddingCVAEDataset(Dataset):
    def __init__(self, payload: dict, norm_stats: dict | None = None):
        x = payload["embeddings"].astype(np.float32)
        if norm_stats is not None:
            # Normalize each embedding coordinate independently using training-set
            # statistics: mu and sigma are vectors of shape (embedding_dim,), not scalars.
            x = (x - norm_stats["mu"]) / norm_stats["sigma"]
        self.x = torch.tensor(x, dtype=torch.float32)
        self.c = torch.tensor(payload["condition"], dtype=torch.float32)
        self.y_grip = torch.tensor(payload["y_grip"], dtype=torch.long)
        self.y_hand = torch.tensor(payload["y_hand"], dtype=torch.long)
        self.y_angle = torch.tensor(payload["y_angle"], dtype=torch.long)
        self.y_phase = torch.tensor(payload["y_phase"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            self.x[idx],
            self.c[idx],
            self.y_grip[idx],
            self.y_hand[idx],
            self.y_angle[idx],
            self.y_phase[idx],
        )


# ---------------------------------------------------------------------------
# CLI and Argument Validation
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 1b: cVAE on transformer pooled embeddings."
    )
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--checkpoint_reach", type=str, default=None)
    p.add_argument("--checkpoint_prereach", type=str, default=None)
    p.add_argument("--checkpoint_grasp", type=str, default=None)
    p.add_argument(
        "--joint_checkpoint",
        type=str,
        default=None,
        help="Optional JointFactorTransformer checkpoint. If set, use this instead of phase specialist checkpoints.",
    )
    p.add_argument(
        "--joint_input_mode",
        choices=["mu", "broadband6"],
        default="mu",
        help="Input mode used by the joint checkpoint.",
    )
    p.add_argument("--joint_cache_dir", type=str, default="/tmp/lfp_joint_embedding_cache")
    p.add_argument("--heldout_phase", choices=PHASE_NAMES, default="grasp")
    p.add_argument("--heldout_grip", choices=["power", "precision"], default="precision")
    p.add_argument("--heldout_hand", choices=["left", "right"], default="right")
    p.add_argument("--latent_dim", type=int, default=16)
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
    p.add_argument("--out_dir", type=str, default=str(_HERE / "results" / "cvae_embeddings"))
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    # Specialist architecture fallback values.
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--feedforward_dim", type=int, default=128)
    # Denoising augmentation.
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
                   help="Condition vector type. 'onehot' = 7-dim one-hot (default). "
                        "'sentence' = 5-dim PCA sentence embedding (Option D).")
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
    args = p.parse_args(argv)
    # Validate aug arguments.
    if args.amplitude_scale_range[0] > args.amplitude_scale_range[1]:
        p.error("--amplitude_scale_range LOW must be <= HIGH")
    if args.noise_scale < 0.0:
        p.error("--noise_scale must be non-negative")
    if args.aug_n_dropout_dims < 0:
        p.error("--aug_n_dropout_dims must be non-negative")
    for name in ("p_cond_single", "p_cond_double", "p_cond_all"):
        if getattr(args, name) < 0.0:
            p.error(f"--{name} must be non-negative")
    total_drop = 3*args.p_cond_single + 3*args.p_cond_double + args.p_cond_all
    if total_drop >= 1.0:
        p.error(
            f"--p_cond_single, --p_cond_double, --p_cond_all sum to {total_drop:.3f}. "
            f"Total must be < 1.0 to allow some training samples with full condition."
        )
    if args.free_bits < 0.0:
        p.error("--free_bits must be non-negative")
    if args.lambda_mmd <= 0:
        p.error("--lambda_mmd must be positive.")
    if args.mmd_loss and args.free_bits > 0:
        p.error("--mmd_loss and --free_bits cannot be used together.")
    if args.condition_type == "sentence":
        if not args.sentence_condition_path or not args.sentence_key_order_path:
            p.error("--condition_type sentence requires both "
                    "--sentence_condition_path and --sentence_key_order_path")
        if getattr(args, "cond_dropout", False):
            p.error("--cond_dropout is incompatible with --condition_type sentence. "
                    "apply_condition_dropout assumes one-hot layout (slices 0:3, 3:5, 5:7).")
    return args


# ---------------------------------------------------------------------------
# Payload Construction
# ---------------------------------------------------------------------------


def _subset(payload: dict, idx: np.ndarray) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(payload["y_grip"]) else v)
            for k, v in payload.items()}


def _torch_load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_embedding_payload(args: argparse.Namespace, device: torch.device) -> tuple:
    if args.joint_checkpoint:
        return build_joint_embedding_payload(args, device)
    raise SystemExit(
        "--joint_checkpoint is required. Run transformer_encoder.run_joint_embedding "
        "first, then pass its checkpoint here."
    )


def build_joint_embedding_payload(args: argparse.Namespace, device: torch.device) -> tuple:
    checkpoint_path = Path(args.joint_checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"--joint_checkpoint does not exist: {checkpoint_path}")

    ckpt = _torch_load_checkpoint(checkpoint_path, device)
    config = ckpt.get("config", {})
    n_bands = 1 if args.joint_input_mode == "mu" else len(BAND_NAMES_6)
    model = JointFactorTransformer(
        n_bands=n_bands,
        d_model=int(config.get("d_model", args.d_model)),
        n_heads=int(config.get("n_heads", args.n_heads)),
        n_layers=int(config.get("n_layers", args.n_layers)),
        feedforward_dim=int(config.get("feedforward_dim", args.feedforward_dim)),
        dropout=float(config.get("dropout", args.dropout)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    raw = load_joint_trials(Path(args.data_dir), args.joint_input_mode)
    flat = phase_expand(raw)
    if args.dry_run and len(flat["y_grip"]) > 500:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(flat["y_grip"]), size=500, replace=False)
        keep.sort()
        flat = subset_flat(flat, keep)
    features = extract_and_cache_features(flat, args.joint_cache_dir)
    idx = np.arange(len(flat["y_grip"]))
    ds = JointEmbeddingDataset(features, flat, idx, ckpt.get("norm_stats"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    chunks = []
    with torch.no_grad():
        for x, *_ in loader:
            chunks.append(model.extract_embedding(x.to(device)).cpu().numpy())
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    _ct = getattr(args, "condition_table",    None)
    _ck = getattr(args, "condition_key_order", None)
    conditions = np.stack([
        lookup_condition(int(ph), int(g), int(h), _ct, _ck)
        if _ct is not None else make_condition_vector(int(ph), int(g), int(h))
        for ph, g, h in zip(flat["y_phase"], flat["y_grip"], flat["y_hand"])
    ]).astype(np.float32)
    rep = {
        "embeddings": embeddings,
        "condition": conditions,
        "y_phase": flat["y_phase"],
        "y_grip": flat["y_grip"],
        "y_hand": flat["y_hand"],
        "y_angle": flat["y_angle"],
        "source": "joint_factor_transformer",
        "joint_checkpoint": str(checkpoint_path),
    }
    rep["is_heldout"] = (
        (rep["y_phase"] == PHASE_NAMES.index(args.heldout_phase)) &
        (rep["y_grip"] == GRIP_TO_ID[args.heldout_grip]) &
        (rep["y_hand"] == HAND_TO_ID[args.heldout_hand])
    )
    return rep, model


# ---------------------------------------------------------------------------
# Split and Normalization
# ---------------------------------------------------------------------------


def _stratify_combo_labels(
    y_phase: np.ndarray,
    y_grip: np.ndarray,
    y_hand: np.ndarray,
    y_angle: np.ndarray,
) -> np.ndarray:
    """Pack phase/grip/hand/angle into one stratification label per sample.

    This mirrors the transformer's encoding convention. Only uniqueness matters
    for stratification; the numeric values are not interpreted ordinally.
    """
    return (
        y_phase.astype(np.int32) * 32
        + y_grip.astype(np.int32) * 16
        + y_hand.astype(np.int32) * 8
        + y_angle.astype(np.int32)
    )


def split_payload(payload: dict, seed: int) -> tuple[dict, dict, dict, dict]:
    all_idx = np.arange(len(payload["y_grip"]))
    heldout_idx = all_idx[payload["is_heldout"]]
    remaining = all_idx[~payload["is_heldout"]]
    if len(heldout_idx) == 0:
        raise ValueError("No held-out embedding samples found.")

    # Match the transformer protocol: 80% seen train, 10% seen val, 10% seen test,
    # with the held-out phase/grip/hand combination kept entirely separate.
    strat = _stratify_combo_labels(
        payload["y_phase"][remaining],
        payload["y_grip"][remaining],
        payload["y_hand"][remaining],
        payload["y_angle"][remaining],
    )
    tr_rel, tmp_rel = train_test_split(
        np.arange(len(remaining)),
        test_size=0.2,
        stratify=strat,
        random_state=seed,
        shuffle=True,
    )
    strat_tmp = _stratify_combo_labels(
        payload["y_phase"][remaining[tmp_rel]],
        payload["y_grip"][remaining[tmp_rel]],
        payload["y_hand"][remaining[tmp_rel]],
        payload["y_angle"][remaining[tmp_rel]],
    )
    va_rel, seen_test_rel = train_test_split(
        tmp_rel,
        test_size=0.5,
        stratify=strat_tmp,
        random_state=seed,
        shuffle=True,
    )
    return (
        _subset(payload, remaining[tr_rel]),
        _subset(payload, remaining[va_rel]),
        _subset(payload, remaining[seen_test_rel]),
        _subset(payload, heldout_idx),
    )


def norm_stats(train_payload: dict) -> dict:
    x = train_payload["embeddings"].astype(np.float32)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return {"mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32)}


# ---------------------------------------------------------------------------
# Main Run Modes
# ---------------------------------------------------------------------------


def main(argv=None) -> dict:
    args = parse_args(argv)

    # Compare-only mode loads completed runs and produces aggregate figures.
    if getattr(args, "compare_only", False):
        if not args.baseline_dirs or not args.aug_dirs:
            raise SystemExit("--compare_only requires both --baseline_dirs and --aug_dirs")
        if args.out_dir == str(_HERE / "results" / "cvae_embeddings"):
            raise SystemExit("--compare_only requires an explicit --out_dir")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        generate_comparison_plots(args.baseline_dirs, args.aug_dirs, out_dir)
        return {}

    if args.data_dir is None:
        raise SystemExit("--data_dir is required unless --compare_only is set")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the condition representation once so every downstream helper sees
    # the same one-hot or sentence-based condition vectors.
    if getattr(args, "condition_type", "onehot") == "sentence":
        args.condition_table     = np.load(args.sentence_condition_path)   # (12, k)
        args.condition_key_order = np.load(args.sentence_key_order_path)   # (12, 3)
        args.condition_dim       = int(args.condition_table.shape[1])
    else:
        args.condition_table     = None
        args.condition_key_order = None
        args.condition_dim       = 7

    print("\n" + "=" * 72)
    print("  Step 1b — cVAE on transformer embeddings")
    print(f"  Held-out: {args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}")
    print(f"  Device: {device} | Output: {out_dir}")
    print(f"  Condition: {args.condition_type} | dim={args.condition_dim}")
    if args.condition_type == "sentence":
        print(f"    table: {args.sentence_condition_path}")
    if getattr(args, "denoising_aug", False):
        print(f"  Augmentation: noise={args.noise_scale}  scale={args.amplitude_scale_range}"
              f"  dropout={args.aug_n_dropout_dims}")
    if getattr(args, "cond_dropout", False):
        print(
            f"  Condition dropout: ON | "
            f"p_single={args.p_cond_single} "
            f"p_double={args.p_cond_double} "
            f"p_all={args.p_cond_all} | "
            f"p_full_c={1 - 3*args.p_cond_single - 3*args.p_cond_double - args.p_cond_all:.2f}"
        )
    else:
        print("  Condition dropout: OFF")
    fb = getattr(args, "free_bits", 0.0)
    if fb > 0.0:
        print(f"  Free bits: {fb} nats/dim  (min KL per dim enforced)")
    else:
        print("  Free bits: OFF")
    if getattr(args, "mmd_loss", False):
        print(f"  Loss: MMD-VAE | lambda_mmd={args.lambda_mmd} | "
              f"val logs ELBO-KL for comparability")
    else:
        print(f"  Loss: ELBO | beta_max={args.beta_max} "
              f"anneal={args.beta_anneal_epochs} epochs")
    print("=" * 72)

    # Diagnostic-only mode rebuilds the payload/model context from disk and
    # recomputes diagnostics without running cVAE training again.
    if getattr(args, "diag_only", False):
        ckpt_path = out_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"--diag_only: no checkpoint.pt found in {out_dir}")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        saved_args = ckpt.get("args", {})
        # Override CLI with saved config so model matches checkpoint exactly.
        args.split_seed = saved_args.get("split_seed", saved_args.get("seed", 42))
        for key in ("latent_dim", "hidden_dims", "dropout",
                    "heldout_phase", "heldout_grip", "heldout_hand",
                    "joint_checkpoint", "joint_input_mode"):
            if key in saved_args:
                setattr(args, key, saved_args[key])
        # Load condition setup from checkpoint.
        args.condition_type          = saved_args.get("condition_type", "onehot")
        args.sentence_condition_path = saved_args.get("sentence_condition_path", None)
        args.sentence_key_order_path = saved_args.get("sentence_key_order_path", None)
        if args.condition_type == "sentence":
            args.condition_table     = np.load(args.sentence_condition_path)
            args.condition_key_order = np.load(args.sentence_key_order_path)
            args.condition_dim       = int(args.condition_table.shape[1])
        else:
            args.condition_table     = None
            args.condition_key_order = None
            args.condition_dim       = 7
        print(f"  [diag_only] Loaded args from checkpoint: "
              f"latent_dim={args.latent_dim} hidden={args.hidden_dims} "
              f"condition={args.condition_type}(dim={args.condition_dim})")
        payload, _ = build_embedding_payload(args, device)
        payload["sample_index"] = np.arange(len(payload["y_grip"]), dtype=np.int64)
        train_p, val_p, seen_test_p, held_p = split_payload(payload, args.split_seed)
        stats_np = np.load(out_dir / "normalization_stats.npz")
        stats = {"mu": stats_np["mu"], "sigma": stats_np["sigma"]}
        val_ds  = EmbeddingCVAEDataset(val_p,  stats)
        held_ds = EmbeddingCVAEDataset(held_p, stats)
        input_dim = int(val_p["embeddings"].shape[1])
        model = LFPCVAE(
            input_dim=input_dim,
            condition_dim=args.condition_dim,
            latent_dim=args.latent_dim,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        compute_collapse_diagnostics(
            model, val_ds, held_ds, args, device, out_dir,
            val_indices=val_p["sample_index"],
        )
        history = ckpt.get("history", {})
        if history.get("train_kl") and history.get("val_kl"):
            _is_mmd = bool(ckpt.get("use_mmd_loss", False))
            np.savez_compressed(
                out_dir / "kl_history.npz",
                train_kl_mean_per_dim=np.asarray(history["train_kl"]),
                val_kl_mean_per_dim=np.asarray(history["val_kl"]),
                train_kl_sum_dims=(
                    np.asarray(history["train_kl"])
                    if _is_mmd else
                    np.asarray(history["train_kl"]) * args.latent_dim
                ),
                val_kl_sum_dims=np.asarray(history["val_kl"]) * args.latent_dim,
                latent_dim=np.array(args.latent_dim),
                loss_mode=np.bytes_("mmd" if _is_mmd else "elbo"),
                third_metric_name=np.bytes_("mmd" if _is_mmd else "kl_per_dim"),
            )
            print(f"  [diag_only] Saved kl_history.npz from checkpoint history")

        if "best_val_recon" not in ckpt or "best_val_kl_sum" not in ckpt:
            val_loss = np.asarray(history.get("val_loss", []), dtype=np.float64)
            val_recon = np.asarray(history.get("val_recon", []), dtype=np.float64)
            val_kl = np.asarray(history.get("val_kl", []), dtype=np.float64)
            if len(val_loss) and len(val_recon) == len(val_loss) and len(val_kl) == len(val_loss):
                _is_mmd = bool(ckpt.get("use_mmd_loss", ckpt.get("args", {}).get("mmd_loss", False)))
                selection_metric = "val_recon" if _is_mmd else "val_loss"
                selection_values = val_recon if _is_mmd else val_loss
                best_epoch = int(np.argmin(selection_values))
                ckpt.update({
                    "best_epoch": best_epoch,
                    "best_selection_metric": selection_metric,
                    "best_selection_score": float(selection_values[best_epoch]),
                    "best_val_loss": float(val_loss[best_epoch]),
                    "best_val_recon": float(val_recon[best_epoch]),
                    "best_val_kl_mean": float(val_kl[best_epoch]),
                    "best_val_kl_sum": float(val_kl[best_epoch] * args.latent_dim),
                })
                torch.save(ckpt, ckpt_path)
                print(f"  [diag_only] Backfilled best-checkpoint metrics from history")
            else:
                print("  WARNING: checkpoint history is insufficient to backfill best-validation metrics")
        return {}

    # Normal training path: payload -> split -> normalize -> train -> diagnose.
    payload, transformer_model = build_embedding_payload(args, device)
    # Stable global sample indices, carried through split_payload via _subset().
    payload["sample_index"] = np.arange(len(payload["y_grip"]), dtype=np.int64)
    train_p, val_p, seen_test_p, held_p = split_payload(payload, args.split_seed)
    stats = norm_stats(train_p)
    np.savez_compressed(out_dir / "normalization_stats.npz", **stats)

    train_ds = EmbeddingCVAEDataset(train_p, stats)
    val_ds = EmbeddingCVAEDataset(val_p, stats)
    held_ds = EmbeddingCVAEDataset(held_p, stats)

    input_dim = int(train_p["embeddings"].shape[1])
    if getattr(args, "aug_n_dropout_dims", 0) > input_dim:
        raise SystemExit(
            f"--aug_n_dropout_dims ({args.aug_n_dropout_dims}) must be <= input_dim ({input_dim})"
        )
    model = LFPCVAE(
        input_dim=input_dim,
        condition_dim=args.condition_dim,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    print(f"Model: input_dim={input_dim} condition_dim={args.condition_dim} "
          f"latent_dim={args.latent_dim} hidden={args.hidden_dims}")

    _scale_range = tuple(args.amplitude_scale_range)
    history = train_cvae(
        model, train_ds, val_ds, args, device,
        save_path=str(out_dir / "checkpoint.pt"),
        use_aug=getattr(args, "denoising_aug", False),
        noise_scale=getattr(args, "noise_scale", 0.1),
        amplitude_scale_range=_scale_range,
        n_dropout_dims=getattr(args, "aug_n_dropout_dims", 2),
        use_cond_dropout=getattr(args, "cond_dropout", False),
        p_cond_single=getattr(args, "p_cond_single", 0.15),
        p_cond_double=getattr(args, "p_cond_double", 0.04),
        p_cond_all=getattr(args, "p_cond_all", 0.03),
        free_bits=getattr(args, "free_bits", 0.0),
        use_mmd_loss=getattr(args, "mmd_loss", False),
        lambda_mmd=getattr(args, "lambda_mmd", 10.0),
    )

    # Save KL/MMD history. In MMD mode, train column contains MMD values (not
    # per-dim KL), so train_kl_sum_dims stores raw MMD rather than MMD*latent_dim.
    # Val column is always ELBO-KL regardless of training mode.
    is_mmd = getattr(args, "mmd_loss", False)
    np.savez_compressed(
        out_dir / "kl_history.npz",
        train_kl_mean_per_dim=np.array(history["train_kl"]),
        val_kl_mean_per_dim=np.array(history["val_kl"]),
        train_kl_sum_dims=(
            np.array(history["train_kl"])
            if is_mmd else
            np.array(history["train_kl"]) * args.latent_dim
        ),
        val_kl_sum_dims=np.array(history["val_kl"]) * args.latent_dim,
        latent_dim=np.array(args.latent_dim),
        loss_mode=np.bytes_("mmd" if is_mmd else "elbo"),
        third_metric_name=np.bytes_("mmd" if is_mmd else "kl_per_dim"),
    )

    compute_collapse_diagnostics(
        model, val_ds, held_ds, args, device, out_dir,
        val_indices=val_p["sample_index"],
    )

    recon = reconstruct_seen(model, val_ds, device, args.batch_size)

    heldout_phase_idx = PHASE_NAMES.index(args.heldout_phase)
    heldout_grip_id = GRIP_TO_ID[args.heldout_grip]
    heldout_hand_id = HAND_TO_ID[args.heldout_hand]
    _ct = getattr(args, "condition_table",    None)
    _ck = getattr(args, "condition_key_order", None)
    condition = (
        lookup_condition(heldout_phase_idx, heldout_grip_id, heldout_hand_id, _ct, _ck)
        if _ct is not None else make_condition_vector(heldout_phase_idx, heldout_grip_id, heldout_hand_id)
    )
    gen, gen_plot_data = evaluate_generation(
        model, transformer_model, train_p, held_p, stats, condition, args, device,
    )
    if not args.no_plot:
        plot_generation_diagnostics(gen_plot_data, out_dir, args)

    print("\nEmbedding cVAE summary:")
    print(f"  Reconstruction MSE: {recon['mse_mean']:.4f}")
    print(f"  Reconstruction r:   {recon['pearsonr_mean']:.4f}")
    print(f"  MMD ratio:          {gen['mmd_ratio']:.3f}")
    print(f"  Centroid distance:  {gen['centroid_distance']:.3f}")
    print(f"  Relative dist mean: {gen['relative_centroid_distance_mean']:.3f}")
    print(f"  Relative dist min:  {gen['relative_centroid_distance_min']:.3f}")
    print(f"  Nearest centroid:   {gen['nearest_centroid_target_rate']:.3f}")
    closer = gen.get("centroids_closer_than_generated", [])
    if closer:
        print("  Centroids closer to target than generated centroid:")
        for row in closer[:5]:
            print(f"    {row['combo']}: dist={row['distance_to_target']:.3f}")
    else:
        print("  No competing centroid is closer to target than the generated centroid.")
    ha = gen.get("head_accuracy")
    if ha:
        print("\n  Transformer head accuracy — generated vs real held-out:")
        print(f"  {'factor':<6}  {'generated':>10}  {'real heldout':>12}")
        for factor in ("phase", "grip", "hand"):
            print(f"  {factor:<6}  {ha['generated'][factor]:>10.3f}  {ha['real_heldout'][factor]:>12.3f}")

    _args_to_save = {k: v for k, v in vars(args).items()
                     if k not in ("condition_table", "condition_key_order")}
    summary = {
        "heldout_label": f"{args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}",
        "input_mode": "transformer_embedding",
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "hidden_dims": args.hidden_dims,
        "reconstruction": recon,
        "generation": gen,
        "history": {
            "final_val_recon": float(history["val_recon"][-1]) if history["val_recon"] else None,
            "final_val_kl": float(history["val_kl"][-1]) if history["val_kl"] else None,
            "n_epochs": len(history["train_loss"]),
        },
        "args": _args_to_save,
    }
    # Load the checkpoint train_cvae() saved (which has best_epoch, best_val_kl_* etc.)
    # and update it with the generation summary rather than overwriting from scratch.
    try:
        _ckpt = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    except TypeError:
        _ckpt = torch.load(out_dir / "checkpoint.pt", map_location="cpu")
    _ckpt.update({
        "model_state":        model.state_dict(),
        "history":            history,
        "args":               _args_to_save,
        "summary":            summary,
        "checkpoint_stage":   "post_generation_outputs",
    })
    torch.save(_ckpt, out_dir / "checkpoint.pt")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved outputs to {out_dir}")
    return summary


if __name__ == "__main__":
    main()
