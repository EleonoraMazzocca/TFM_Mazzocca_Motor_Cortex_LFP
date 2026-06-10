"""Step 1b — Conditional VAE on transformer embeddings.

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
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

_HERE = Path(__file__).resolve().parent
_TRANSFORMER = _HERE.parent / "transformer"
_JOINT = _HERE.parent / "joint_embedding"
if str(_TRANSFORMER) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_JOINT) not in sys.path:
    sys.path.insert(0, str(_JOINT))

from data import PHASE_NAMES, GRIP_TO_ID, HAND_TO_ID
from cvae_data import make_condition_vector
from cvae_model import LFPCVAE
from run_cvae import train_cvae, compute_mmd
from separability_check import load_all_trials, collect_all_embeddings
from joint_embedding_data import (  # noqa: E402
    BAND_NAMES_6,
    JointEmbeddingDataset,
    extract_and_cache_features,
    load_joint_trials,
    phase_expand,
    subset_flat,
)
from joint_embedding_model import JointFactorTransformer  # noqa: E402


ID_TO_GRIP = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND = {v: k for k, v in HAND_TO_ID.items()}


def lookup_condition(
    phase_id: int, grip_id: int, hand_id: int,
    condition_table: np.ndarray,
    key_order: np.ndarray,
) -> np.ndarray:
    """Return the (condition_dim,) float32 vector for a (phase, grip, hand) triple.

    Robust to any ordering of the 12-row table — uses key_order to locate the row.
    """
    if not (condition_table.ndim == 2 and condition_table.shape[0] == 12):
        raise ValueError(f"condition_table must be (12, condition_dim), got {condition_table.shape}")
    if key_order.shape != (12, 3):
        raise ValueError(f"key_order must be (12, 3), got {key_order.shape}")
    matches = np.where(
        (key_order == np.array([phase_id, grip_id, hand_id])).all(axis=1)
    )[0]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 match for ({phase_id},{grip_id},{hand_id}), got {len(matches)}"
        )
    return condition_table[matches[0]].astype(np.float32)


class EmbeddingCVAEDataset(Dataset):
    def __init__(self, payload: dict, norm_stats: dict | None = None):
        x = payload["embeddings"].astype(np.float32)
        if norm_stats is not None:
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

    checkpoints = {
        1: args.checkpoint_reach,
    }
    if args.checkpoint_prereach:
        checkpoints[0] = args.checkpoint_prereach
    if args.checkpoint_grasp:
        checkpoints[2] = args.checkpoint_grasp
    checkpoints = {ph: p for ph, p in checkpoints.items() if p and Path(p).exists()}
    if 1 not in checkpoints:
        raise SystemExit("--checkpoint_reach must exist unless --joint_checkpoint is used")

    data = load_all_trials(Path(args.data_dir))
    if args.dry_run and len(data["y_grip"]) > 500:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(data["y_grip"]), size=500, replace=False)
        keep.sort()
        for key in ("file_idx", "trial_idx", "y_grip", "y_hand", "y_angle"):
            data[key] = data[key][keep]

    rep = collect_all_embeddings(checkpoints, data, args, device)
    _ct = getattr(args, "condition_table",    None)
    _ck = getattr(args, "condition_key_order", None)
    conditions = np.stack([
        lookup_condition(int(ph), int(g), int(h), _ct, _ck)
        if _ct is not None else make_condition_vector(int(ph), int(g), int(h))
        for ph, g, h in zip(rep["y_phase"], rep["y_grip"], rep["y_hand"])
    ]).astype(np.float32)
    rep["condition"] = conditions
    rep["is_heldout"] = (
        (rep["y_phase"] == PHASE_NAMES.index(args.heldout_phase)) &
        (rep["y_grip"] == GRIP_TO_ID[args.heldout_grip]) &
        (rep["y_hand"] == HAND_TO_ID[args.heldout_hand])
    )
    return rep, None


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


def split_payload(payload: dict, seed: int) -> tuple[dict, dict, dict]:
    all_idx = np.arange(len(payload["y_grip"]))
    heldout_idx = all_idx[payload["is_heldout"]]
    remaining = all_idx[~payload["is_heldout"]]
    if len(heldout_idx) == 0:
        raise ValueError("No held-out embedding samples found.")
    strat = (
        payload["y_phase"][remaining].astype(np.int32) * 16
        + payload["y_grip"][remaining].astype(np.int32) * 8
        + payload["y_hand"][remaining].astype(np.int32) * 4
        + payload["y_angle"][remaining].astype(np.int32)
    )
    tr_rel, va_rel = train_test_split(
        np.arange(len(remaining)),
        test_size=0.15,
        stratify=strat,
        random_state=seed,
        shuffle=True,
    )
    return _subset(payload, remaining[tr_rel]), _subset(payload, remaining[va_rel]), _subset(payload, heldout_idx)


def norm_stats(train_payload: dict) -> dict:
    x = train_payload["embeddings"].astype(np.float32)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return {"mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32)}


def reconstruct_seen(model, val_ds, device, batch_size=128) -> dict:
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    xs, xrs, gs, hs, ps = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, c, yg, yh, ya, yp in loader:
            xr, _, _ = model(x.to(device), c.to(device))
            xs.append(x.numpy())
            xrs.append(xr.cpu().numpy())
            gs.append(yg.numpy())
            hs.append(yh.numpy())
            ps.append(yp.numpy())
    x = np.concatenate(xs)
    xr = np.concatenate(xrs)
    g = np.concatenate(gs)
    h = np.concatenate(hs)
    p = np.concatenate(ps)

    by_combo = {}
    for ph in range(len(PHASE_NAMES)):
        for gi in (0, 1):
            for hi in (0, 1):
                mask = (p == ph) & (g == gi) & (h == hi)
                if not mask.any():
                    continue
                label = f"{PHASE_NAMES[ph]} + {ID_TO_GRIP[gi]} + {ID_TO_HAND[hi]}"
                mse = float(np.mean((x[mask] - xr[mask]) ** 2))
                r = float(np.corrcoef(x[mask].reshape(-1), xr[mask].reshape(-1))[0, 1])
                by_combo[label] = {"mse": mse, "pearsonr": r}
    return {
        "by_combo": by_combo,
        "mse_mean": float(np.mean([v["mse"] for v in by_combo.values()])),
        "pearsonr_mean": float(np.mean([v["pearsonr"] for v in by_combo.values()])),
    }


def evaluate_generation(model, transformer_model, train_payload, heldout_payload, stats, condition, out_dir, args, device) -> dict:
    x_real = (heldout_payload["embeddings"].astype(np.float32) - stats["mu"]) / stats["sigma"]
    c = torch.tensor(condition, dtype=torch.float32)
    x_gen = model.generate(c, n_samples=500, device=device).cpu().numpy()

    head_accuracy = None
    _preds_gen = None
    _preds_real = None
    _head_targets = None
    _head_class_names = {"phase": PHASE_NAMES, "grip": ["power", "precision"], "hand": ["left", "right"]}
    if transformer_model is not None:
        transformer_model.eval()
        mu_np = stats["mu"].astype(np.float32)
        sigma_np = stats["sigma"].astype(np.float32)
        x_gen_denorm = torch.tensor(x_gen * sigma_np + mu_np, dtype=torch.float32).to(device)
        x_real_denorm = torch.tensor(heldout_payload["embeddings"].astype(np.float32), dtype=torch.float32).to(device)
        target_phase_idx = PHASE_NAMES.index(args.heldout_phase)
        target_grip_id = GRIP_TO_ID[args.heldout_grip]
        target_hand_id = HAND_TO_ID[args.heldout_hand]
        _head_targets = {"phase": target_phase_idx, "grip": target_grip_id, "hand": target_hand_id}
        with torch.no_grad():
            _preds_gen = {
                "phase": transformer_model.head_phase(x_gen_denorm).argmax(1).cpu().numpy(),
                "grip":  transformer_model.head_grip(x_gen_denorm).argmax(1).cpu().numpy(),
                "hand":  transformer_model.head_hand(x_gen_denorm).argmax(1).cpu().numpy(),
            }
            _preds_real = {
                "phase": transformer_model.head_phase(x_real_denorm).argmax(1).cpu().numpy(),
                "grip":  transformer_model.head_grip(x_real_denorm).argmax(1).cpu().numpy(),
                "hand":  transformer_model.head_hand(x_real_denorm).argmax(1).cpu().numpy(),
            }
        head_accuracy = {
            "generated":    {f: float((_preds_gen[f]  == _head_targets[f]).mean()) for f in ("phase", "grip", "hand")},
            "real_heldout": {f: float((_preds_real[f] == _head_targets[f]).mean()) for f in ("phase", "grip", "hand")},
        }

    p_vals = []
    for j in range(x_real.shape[1]):
        _, p = ttest_ind(x_real[:, j], x_gen[:, j], equal_var=False)
        p_vals.append(float(p))
    frac_ns = float(np.mean(np.asarray(p_vals) > 0.05))
    mmd_gen = compute_mmd(x_gen, x_real)
    half = len(x_real) // 2
    mmd_base = compute_mmd(x_real[:half], x_real[half:]) if half > 1 else float("nan")
    mmd_ratio = float(mmd_gen / max(mmd_base, 1e-10)) if not np.isnan(mmd_base) else float("nan")

    x_train = (train_payload["embeddings"].astype(np.float32) - stats["mu"]) / stats["sigma"]
    target_phase = PHASE_NAMES.index(args.heldout_phase)
    target_grip = GRIP_TO_ID[args.heldout_grip]
    target_hand = HAND_TO_ID[args.heldout_hand]
    target_combo = target_grip * 6 + target_hand * 3 + target_phase

    train_combo = (
        train_payload["y_grip"].astype(np.int64) * 6
        + train_payload["y_hand"].astype(np.int64) * 3
        + train_payload["y_phase"].astype(np.int64)
    )

    # Nearest-centroid placement in embedding space.
    target_centroid = x_real.mean(axis=0)
    gen_centroid = x_gen.mean(axis=0)
    centroid_dist = float(np.linalg.norm(gen_centroid - target_centroid))
    centroids = {}
    for combo in sorted(np.unique(train_combo)):
        centroids[int(combo)] = x_train[train_combo == combo].mean(axis=0)
    centroids[int(target_combo)] = target_centroid
    centroid_keys = np.array(sorted(centroids))
    centroid_mat = np.stack([centroids[int(k)] for k in centroid_keys], axis=0)
    other_mask = centroid_keys != target_combo
    other_keys = centroid_keys[other_mask]
    target_to_other = np.linalg.norm(centroid_mat[other_mask] - target_centroid[None, :], axis=1)
    target_to_other_mean = float(np.mean(target_to_other)) if len(target_to_other) else float("nan")
    target_to_other_min = float(np.min(target_to_other)) if len(target_to_other) else float("nan")
    relative_centroid_distance_mean = float(centroid_dist / max(target_to_other_mean, 1e-10))
    relative_centroid_distance_min = float(centroid_dist / max(target_to_other_min, 1e-10))
    centroid_distance_table = []
    for combo, dist in sorted(zip(other_keys, target_to_other), key=lambda x: float(x[1])):
        centroid_distance_table.append({
            "combo_id": int(combo),
            "combo": f"{PHASE_NAMES[int(combo) % 3]}+{ID_TO_GRIP[int(combo) // 6]}+{ID_TO_HAND[(int(combo) // 3) % 2]}",
            "distance_to_target": float(dist),
            "closer_than_generated": bool(float(dist) < centroid_dist),
            "distance_minus_generated": float(dist - centroid_dist),
        })
    closer_than_generated = [row for row in centroid_distance_table if row["closer_than_generated"]]
    dists = np.linalg.norm(x_gen[:, None, :] - centroid_mat[None, :, :], axis=2)
    nearest = centroid_keys[np.argmin(dists, axis=1)]
    nearest_target_rate = float((nearest == target_combo).mean())
    mean_nearest_margin = float(
        np.mean(
            np.partition(dists, kth=1, axis=1)[:, 1]
            - np.min(dists, axis=1)
        )
    )

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            def _combo_name(combo: int) -> str:
                ph = combo % 3
                h = (combo // 3) % 2
                g = combo // 6
                return f"{PHASE_NAMES[ph]}+{ID_TO_GRIP[g]}+{ID_TO_HAND[h]}"

            x_train_plot = x_train
            pca = PCA(n_components=3, random_state=args.seed)
            pca.fit(x_train_plot)
            tr_pc = pca.transform(x_train_plot)
            real_pc = pca.transform(x_real)
            gen_pc = pca.transform(x_gen)
            centroid_pc = pca.transform(centroid_mat)
            gen_centroid_pc = pca.transform(gen_centroid[None, :])
            target_centroid_pc = pca.transform(target_centroid[None, :])

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, xi, yi, ylabel in [
                (axes[0], 0, 1, "PC2"),
                (axes[1], 0, 2, "PC3"),
            ]:
                ax.scatter(tr_pc[:, xi], tr_pc[:, yi], c="grey", alpha=0.12, s=8, label="seen train")
                ax.scatter(real_pc[:, xi], real_pc[:, yi], c="#4C72B0", alpha=0.75, s=20, label="real held-out")
                ax.scatter(gen_pc[:, xi], gen_pc[:, yi], c="#DD8452", alpha=0.75, s=20, label="generated")
                ax.set_xlabel("PC1")
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)
            fig.suptitle("Embedding cVAE generation")
            plt.tight_layout()
            plt.savefig(out_dir / "embedding_generation_pca.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, xi, yi, ylabel in [
                (axes[0], 0, 1, "PC2"),
                (axes[1], 0, 2, "PC3"),
            ]:
                ax.scatter(tr_pc[:, xi], tr_pc[:, yi], c="grey", alpha=0.06, s=6, label="seen train")
                for row, combo in enumerate(centroid_keys):
                    color = "#4C72B0" if combo == target_combo else "#222222"
                    marker = "*" if combo == target_combo else "o"
                    size = 170 if combo == target_combo else 55
                    ax.scatter(
                        centroid_pc[row, xi], centroid_pc[row, yi],
                        c=color, marker=marker, s=size, edgecolors="white", linewidths=0.7,
                        label="target centroid" if combo == target_combo and xi == 0 and yi == 1 else None,
                    )
                    if combo != target_combo and xi == 0 and yi == 1:
                        ax.text(
                            centroid_pc[row, xi], centroid_pc[row, yi],
                            str(int(combo)), fontsize=6, alpha=0.8,
                        )
                ax.scatter(
                    gen_centroid_pc[0, xi], gen_centroid_pc[0, yi],
                    c="#DD8452", marker="X", s=150, edgecolors="white", linewidths=0.7,
                    label="generated centroid",
                )
                ax.plot(
                    [target_centroid_pc[0, xi], gen_centroid_pc[0, xi]],
                    [target_centroid_pc[0, yi], gen_centroid_pc[0, yi]],
                    color="#DD8452", linewidth=1.5, alpha=0.8,
                )
                ax.set_xlabel("PC1")
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)
            fig.suptitle("Embedding centroids in PCA space")
            plt.tight_layout()
            plt.savefig(out_dir / "embedding_centroids_pca.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(["target centroid"], [nearest_target_rate], color=["#937860"])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Generated samples nearest to held-out centroid")
            ax.set_title("Embedding cVAE centroid assignment")
            ax.text(
                bars[0].get_x() + bars[0].get_width() / 2,
                nearest_target_rate + 0.03,
                f"{nearest_target_rate:.2f}",
                ha="center",
                fontsize=10,
            )
            plt.tight_layout()
            plt.savefig(out_dir / "embedding_centroid_assignment.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ["gen→target", "target→nearest other", "target→mean other"]
            vals = [centroid_dist, target_to_other_min, target_to_other_mean]
            colors = ["#DD8452", "#7F7F7F", "#4C72B0"]
            bars = ax.bar(labels, vals, color=colors)
            ax.set_ylabel("Euclidean distance in normalized embedding space")
            ax.set_title("Centroid distance context")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.03, f"{val:.2f}", ha="center", fontsize=9)
            ax.text(
                0.02, 0.95,
                f"relative to mean other = {relative_centroid_distance_mean:.2f}\n"
                f"relative to nearest other = {relative_centroid_distance_min:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
            plt.tight_layout()
            plt.savefig(out_dir / "embedding_centroid_distance_context.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 9))
            target_label = f"{args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}"
            rows = centroid_distance_table + [{
                "combo": f"generated {target_label}",
                "distance_to_target": centroid_dist,
                "closer_than_generated": False,
                "is_generated": True,
            }]
            rows = sorted(rows, key=lambda row: row["distance_to_target"])
            labels = [row["combo"] for row in rows]
            vals = [row["distance_to_target"] for row in rows]
            colors = [
                "#DD8452" if row.get("is_generated") else
                "#C44E52" if row["closer_than_generated"] else
                "#7F7F7F"
                for row in rows
            ]
            ax.bar(range(len(vals)), vals, color=colors)
            ax.axhline(centroid_dist, color="#DD8452", linewidth=2, label="generated→target")
            ax.set_ylim(0, max(vals) * 1.08)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Distance to real held-out target centroid")
            ax.set_title("Distance to held-out target centroid")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / "embedding_target_centroid_distances.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            if head_accuracy is not None:
                factors = ["phase", "grip", "hand"]

                # accuracy bar chart + summary table
                fig, axes = plt.subplots(1, 2, figsize=(13, 4))
                ax = axes[0]
                x_pos = np.arange(len(factors))
                width = 0.32
                bars_gen  = ax.bar(x_pos - width / 2, [head_accuracy["generated"][f]    for f in factors], width, label="generated",    color="#DD8452")
                bars_real = ax.bar(x_pos + width / 2, [head_accuracy["real_heldout"][f] for f in factors], width, label="real held-out", color="#4C72B0")
                for bar in list(bars_gen) + list(bars_real):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{bar.get_height():.2f}", ha="center", fontsize=9)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(factors)
                ax.set_ylim(0, 1.15)
                ax.set_ylabel("Accuracy")
                ax.set_title("Head accuracy: generated vs real held-out")
                ax.legend()
                ax = axes[1]
                ax.axis("off")
                table_data = []
                for f in factors:
                    n_cls = len(_head_class_names[f])
                    gen_top  = int(np.bincount(_preds_gen[f],  minlength=n_cls).argmax())
                    real_top = int(np.bincount(_preds_real[f], minlength=n_cls).argmax())
                    table_data.append([
                        f,
                        _head_class_names[f][_head_targets[f]],
                        f"{head_accuracy['generated'][f]:.2f}",
                        f"{head_accuracy['real_heldout'][f]:.2f}",
                        _head_class_names[f][gen_top],
                        _head_class_names[f][real_top],
                    ])
                tbl = ax.table(
                    cellText=table_data,
                    colLabels=["factor", "true class", "gen acc", "real acc", "gen top pred", "real top pred"],
                    loc="center", cellLoc="center",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(10)
                tbl.scale(1, 1.8)
                ax.set_title("Prediction summary", pad=20)
                fig.suptitle(f"Transformer head accuracy — {args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}")
                plt.tight_layout()
                plt.savefig(out_dir / "embedding_head_accuracy.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {out_dir / 'embedding_head_accuracy.png'}")

                # predicted class distribution per head
                fig, axes = plt.subplots(1, 3, figsize=(13, 4))
                for ax, factor in zip(axes, factors):
                    cls_names = _head_class_names[factor]
                    n_cls = len(cls_names)
                    x_cls = np.arange(n_cls)
                    w = 0.35
                    dist_gen  = np.array([(_preds_gen[factor]  == c).mean() for c in range(n_cls)])
                    dist_real = np.array([(_preds_real[factor] == c).mean() for c in range(n_cls)])
                    ax.bar(x_cls - w / 2, dist_gen,  w, label="generated",    color="#DD8452", alpha=0.85)
                    ax.bar(x_cls + w / 2, dist_real, w, label="real held-out", color="#4C72B0", alpha=0.85)
                    ax.axvline(_head_targets[factor], color="black", linewidth=1.5, linestyle="--",
                               label=f"true: {cls_names[_head_targets[factor]]}")
                    ax.set_xticks(x_cls)
                    ax.set_xticklabels(cls_names)
                    ax.set_ylim(0, 1.1)
                    ax.set_ylabel("Fraction of samples")
                    ax.set_title(factor)
                    ax.legend(fontsize=8)
                fig.suptitle(
                    f"Predicted class distribution — generated vs real held-out\n"
                    f"({args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand})"
                )
                plt.tight_layout()
                plt.savefig(out_dir / "embedding_head_class_distribution.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {out_dir / 'embedding_head_class_distribution.png'}")

            # ── 11+1 panel: generated vs each seen condition ──────────────────
            _GRIP_NAMES = ["power", "precision"]
            _HAND_NAMES = ["left", "right"]
            # column order: power+L, power+R, precision+L, precision+R
            _gh_cols = [(0, 0), (0, 1), (1, 0), (1, 1)]

            fig, axes = plt.subplots(3, 4, figsize=(18, 12))
            for ph in range(len(PHASE_NAMES)):
                for col, (gr, ha) in enumerate(_gh_cols):
                    ax = axes[ph, col]
                    is_heldout = (ph == target_phase and gr == target_grip and ha == target_hand)
                    cond_name = (
                        f"{PHASE_NAMES[ph]}+{_GRIP_NAMES[gr]}+{_HAND_NAMES[ha]}"
                    )
                    if is_heldout:
                        ax.scatter(real_pc[:, 0], real_pc[:, 1],
                                   c="#4C72B0", alpha=0.55, s=12, linewidths=0, label="real held-out")
                        ax.scatter(gen_pc[:, 0], gen_pc[:, 1],
                                   c="#DD8452", alpha=0.55, s=12, linewidths=0, label="generated")
                        ax.set_facecolor("#FFF0F0")
                        ax.set_title(f"{cond_name}\n[TARGET — held-out]", fontsize=7, color="#C44E52")
                        ax.legend(fontsize=6, markerscale=2, loc="best")
                    else:
                        mask = (
                            (train_payload["y_phase"] == ph)
                            & (train_payload["y_grip"] == gr)
                            & (train_payload["y_hand"] == ha)
                        )
                        if mask.any():
                            cond_pc = pca.transform(x_train[mask])
                            ax.scatter(cond_pc[:, 0], cond_pc[:, 1],
                                       c="#4C72B0", alpha=0.4, s=10, linewidths=0, label="seen condition")
                        ax.scatter(gen_pc[:, 0], gen_pc[:, 1],
                                   c="#DD8452", alpha=0.25, s=8, linewidths=0, label="generated")
                        ax.set_title(cond_name, fontsize=7)
                    ax.set_xticks([])
                    ax.set_yticks([])

            fig.suptitle(
                f"Generated ({args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand})"
                f" overlaid on each seen condition\n"
                f"Orange = generated  |  Blue = seen condition  |  Red cell = true target",
                fontsize=10,
            )
            plt.tight_layout()
            plt.savefig(out_dir / "embedding_generated_vs_conditions.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_dir / 'embedding_generated_vs_conditions.png'}")

        except Exception as e:
            print(f"  WARNING: embedding PCA plot failed: {e}")

    return {
        "frac_dims_p_gt_005": frac_ns,
        "mmd_generated": float(mmd_gen),
        "mmd_baseline": float(mmd_base),
        "mmd_ratio": mmd_ratio,
        "centroid_distance": centroid_dist,
        "target_to_other_centroid_mean": target_to_other_mean,
        "target_to_other_centroid_min": target_to_other_min,
        "relative_centroid_distance_mean": relative_centroid_distance_mean,
        "relative_centroid_distance_min": relative_centroid_distance_min,
        "nearest_centroid_target_rate": nearest_target_rate,
        "nearest_centroid_mean_margin": mean_nearest_margin,
        "centroid_distance_table": centroid_distance_table,
        "centroids_closer_than_generated": closer_than_generated,
        "head_accuracy": head_accuracy,
    }


def compute_collapse_diagnostics(
    model,
    val_ds,
    heldout_ds,
    args,
    device,
    out_dir: Path,
    val_indices: np.ndarray,
    generation_seed: int = 0,
) -> dict:
    """Encode val set and compute posterior-collapse diagnostics.

    Augmentation is always disabled here. Saves collapse_diagnostics.npz
    to out_dir and returns a summary dict.
    """
    from cvae_data import make_condition_vector

    model.eval()

    # -- Encode all val samples -----------------------------------------------
    loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    mu_list, lv_list, grip_l, hand_l, phase_l = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x_b, c_b = batch[0].to(device), batch[1].to(device)
            mu_b, lv_b = model.encode(x_b, c_b)
            mu_list.append(mu_b.cpu().numpy())
            lv_list.append(lv_b.cpu().numpy())
            grip_l.append(batch[2].numpy())
            hand_l.append(batch[3].numpy())
            phase_l.append(batch[5].numpy())

    mu_all  = np.concatenate(mu_list,  axis=0)   # (N, latent_dim)
    lv_all  = np.concatenate(lv_list,  axis=0)
    grip_v  = np.concatenate(grip_l)
    hand_v  = np.concatenate(hand_l)
    phase_v = np.concatenate(phase_l)

    val_combo_labels = (grip_v * 6 + hand_v * 3 + phase_v).astype(np.int64)

    # -- Diagnostics 1-3 -------------------------------------------------------
    std_mu       = mu_all.std(axis=0, ddof=0)
    mean_sigma   = np.exp(0.5 * lv_all).mean(axis=0)
    second_moment = (mu_all ** 2 + np.exp(lv_all)).mean(axis=0)

    print(f"\n  Collapse diagnostics (latent_dim={mu_all.shape[1]}):")
    print(f"    std(mu) — mean={std_mu.mean():.4f}  min={std_mu.min():.4f}  max={std_mu.max():.4f}")
    print(f"    mean(sigma) — mean={mean_sigma.mean():.4f}")
    print(f"    second_moment — mean={second_moment.mean():.4f}  (target ~1.0)")

    # -- Diagnostic 7: Seen-class MMD ------------------------------------------
    seen_mmd_records = {}
    seen_bandwidths  = {}
    unique_combos = np.unique(val_combo_labels)
    for combo in unique_combos:
        mask = val_combo_labels == combo
        x_real_combo = np.stack([
            val_ds[int(i)][0].numpy() for i in np.where(mask)[0]
        ])
        if len(x_real_combo) < 2:
            continue
        # Bandwidth from real samples only, self-distances excluded.
        from sklearn.metrics.pairwise import euclidean_distances as _edist
        dists_combo = _edist(x_real_combo, x_real_combo)
        np.fill_diagonal(dists_combo, np.nan)
        bw = float(np.nanmedian(dists_combo))
        bw = max(bw, 1e-6)

        # Reconstruct condition from combo id: combo = grip*6 + hand*3 + phase
        phase_id = int(combo % 3)
        hand_id  = int((combo // 3) % 2)
        grip_id  = int(combo // 6)
        _ct = getattr(args, "condition_table",    None)
        _ck = getattr(args, "condition_key_order", None)
        cond_vec = (
            lookup_condition(phase_id, grip_id, hand_id, _ct, _ck)
            if _ct is not None else make_condition_vector(phase_id, grip_id, hand_id)
        )
        c_tensor = torch.tensor(cond_vec, dtype=torch.float32)
        gen_rng     = torch.Generator(device=device).manual_seed(generation_seed)
        x_gen_combo = model.generate(c_tensor, n_samples=500, device=device,
                                     generator=gen_rng).cpu().numpy()

        from run_cvae import compute_mmd
        mmd_val = compute_mmd(x_gen_combo, x_real_combo, bandwidth=bw)
        ph_name = PHASE_NAMES[phase_id]
        grip_name = ID_TO_GRIP[grip_id]
        hand_name = ID_TO_HAND[hand_id]
        label = f"{ph_name}+{grip_name}+{hand_name}"
        seen_mmd_records[label] = {
            "mmd": mmd_val, "bandwidth": bw,
            "n_real": int(x_real_combo.shape[0]), "n_generated": 500,
        }
        seen_bandwidths[label] = bw
        print(f"    seen MMD [{label}]: {mmd_val:.5f}  (bw={bw:.4f})")

    # -- Diagnostic 8: Held-out MMD --------------------------------------------
    held_loader = DataLoader(heldout_ds, batch_size=256, shuffle=False)
    x_held_l = []
    with torch.no_grad():
        for batch in held_loader:
            x_held_l.append(batch[0].numpy())
    x_held = np.concatenate(x_held_l, axis=0)

    from sklearn.metrics.pairwise import euclidean_distances as _edist2
    dists_held = _edist2(x_held, x_held)
    np.fill_diagonal(dists_held, np.nan)
    bw_held = float(np.nanmedian(dists_held))
    bw_held = max(bw_held, 1e-6)

    heldout_phase_idx = PHASE_NAMES.index(args.heldout_phase)
    heldout_grip_id   = GRIP_TO_ID[args.heldout_grip]
    heldout_hand_id   = HAND_TO_ID[args.heldout_hand]
    _ct = getattr(args, "condition_table",    None)
    _ck = getattr(args, "condition_key_order", None)
    cond_held = (
        lookup_condition(heldout_phase_idx, heldout_grip_id, heldout_hand_id, _ct, _ck)
        if _ct is not None else make_condition_vector(heldout_phase_idx, heldout_grip_id, heldout_hand_id)
    )
    c_held_t  = torch.tensor(cond_held, dtype=torch.float32)
    gen_rng_held = torch.Generator(device=device).manual_seed(generation_seed)
    x_gen_held   = model.generate(c_held_t, n_samples=500, device=device,
                                  generator=gen_rng_held).cpu().numpy()

    from run_cvae import compute_mmd as _cmmd
    mmd_held = _cmmd(x_gen_held, x_held, bandwidth=bw_held)
    print(f"    held-out MMD: {mmd_held:.5f}  (bw={bw_held:.4f}  n_real={len(x_held)})")

    # -- Save ------------------------------------------------------------------
    np.savez_compressed(
        out_dir / "collapse_diagnostics.npz",
        std_mu=std_mu,
        mean_sigma=mean_sigma,
        second_moment=second_moment,
        mu_all=mu_all,
        val_combo_labels=val_combo_labels,
        val_indices=val_indices,
        mmd_seen=np.bytes_(json.dumps(seen_mmd_records)),
        mmd_seen_bandwidths=np.bytes_(json.dumps(seen_bandwidths)),
        mmd_heldout=np.array(mmd_held),
        mmd_heldout_bandwidth=np.array(bw_held),
        generation_seed=np.array(generation_seed),
        latent_dim=np.array(args.latent_dim),
        split_seed=np.array(args.split_seed),
        heldout_combo=np.array([args.heldout_phase, args.heldout_grip, args.heldout_hand]),
    )
    print(f"  Saved collapse_diagnostics.npz to {out_dir}")

    return {
        "std_mu_mean": float(std_mu.mean()),
        "mean_sigma_mean": float(mean_sigma.mean()),
        "second_moment_mean": float(second_moment.mean()),
        "mmd_heldout": float(mmd_held),
    }


def _load_mmd_seen(data: dict) -> dict:
    """Deserialise mmd_seen from collapse_diagnostics.npz."""
    return json.loads(data["mmd_seen"].item())


def generate_comparison_plots(
    baseline_dirs: list[str],
    aug_dirs: list[str],
    out_dir: Path,
) -> None:
    """Load diagnostics from completed run dirs and generate 5 comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping comparison plots")
        return

    def _load_dir(d: str) -> tuple[dict, dict, dict, Path]:
        dp = Path(d)
        diag = dict(np.load(dp / "collapse_diagnostics.npz", allow_pickle=False))
        kl   = dict(np.load(dp / "kl_history.npz",           allow_pickle=False))
        try:
            ckpt = torch.load(dp / "checkpoint.pt", map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(dp / "checkpoint.pt", map_location="cpu")
        return diag, kl, ckpt, dp

    def _aug_group_label(ckpt: dict) -> str:
        a = ckpt.get("args", {})
        denoising = bool(a.get("denoising_aug", False))
        cond_drop = bool(a.get("cond_dropout", False))
        scale = a.get("amplitude_scale_range", [0.85, 1.15])
        parts = []
        if denoising:
            parts.append(
                f"denoising: n_drop={a.get('aug_n_dropout_dims', 2)}, "
                f"noise={a.get('noise_scale', 0.1)}, "
                f"scale=({scale[0]},{scale[1]})"
            )
        if cond_drop:
            p_single = a.get("p_cond_single", 0.15)
            p_double = a.get("p_cond_double", 0.04)
            p_all = a.get("p_cond_all", 0.03)
            p_full = 1 - 3*p_single - 3*p_double - p_all
            parts.append(
                f"cond_drop: p1={p_single}, p2={p_double}, "
                f"p_all={p_all}, p_full={p_full:.2f}"
            )
        fb = float(a.get("free_bits", 0.0))
        if fb > 0.0:
            parts.append(f"free_bits={fb}")
        beta = float(a.get("beta_max", 1.0))
        if beta != 1.0:
            parts.append(f"beta={beta}")
        if a.get("mmd_loss", False):
            parts.append(f"mmd=True, lam={a.get('lambda_mmd', 10.0)}")
        return " + ".join(parts) if parts else "no augmentation"

    all_baseline = [_load_dir(d) for d in baseline_dirs]
    all_aug      = [_load_dir(d) for d in aug_dirs]
    all_runs     = all_baseline + all_aug

    # Group aug runs by augmentation configuration (separate curve per config).
    aug_groups: dict[str, list] = {}
    for item in all_aug:
        key = _aug_group_label(item[2])
        aug_groups.setdefault(key, []).append(item)
    aug_group_keys = sorted(aug_groups.keys())

    # Ordered list: baseline first, then each aug config group.
    all_groups = [("baseline", all_baseline)] + [(gkey, aug_groups[gkey]) for gkey in aug_group_keys]
    n_groups = len(all_groups)

    # -- Compatibility checks --------------------------------------------------
    ref_diag       = all_runs[0][0]
    ref_split_seed  = int(ref_diag["split_seed"])
    ref_latent_dim  = int(ref_diag["latent_dim"])
    ref_gen_seed    = int(ref_diag["generation_seed"])
    ref_heldout     = ref_diag["heldout_combo"].tolist()
    ref_val_indices = ref_diag["val_indices"]
    ref_n_epochs    = len(all_runs[0][1]["val_kl_mean_per_dim"])
    ref_bw          = _load_mmd_seen(ref_diag)
    ref_ckpt_args   = all_runs[0][2].get("args", {})

    errors = []
    for diag, kl, ckpt, run_dir in all_runs[1:]:
        label = str(run_dir)
        a = ckpt.get("args", {})
        if int(diag["split_seed"])          != ref_split_seed:
            errors.append(f"{label}: split_seed mismatch")
        if int(diag["latent_dim"])          != ref_latent_dim:
            errors.append(f"{label}: latent_dim mismatch")
        if int(diag["generation_seed"])     != ref_gen_seed:
            errors.append(f"{label}: generation_seed mismatch")
        if diag["heldout_combo"].tolist()   != ref_heldout:
            errors.append(f"{label}: heldout_combo mismatch")
        if not np.array_equal(diag["val_indices"], ref_val_indices):
            errors.append(f"{label}: val_indices mismatch — validation sets differ")
        if len(kl["val_kl_mean_per_dim"])   != ref_n_epochs:
            errors.append(f"{label}: epoch count mismatch ({len(kl['val_kl_mean_per_dim'])} vs {ref_n_epochs})")
        run_bw = _load_mmd_seen(diag)
        if set(run_bw) != set(ref_bw):
            errors.append(f"{label}: seen-class MMD combination keys differ from reference")
        else:
            for combo in ref_bw:
                if not np.isclose(run_bw[combo]["bandwidth"], ref_bw[combo]["bandwidth"],
                                  atol=1e-6, rtol=1e-6):
                    errors.append(f"{label}: MMD bandwidth mismatch for {combo}")
        if a.get("joint_checkpoint") != ref_ckpt_args.get("joint_checkpoint"):
            errors.append(f"{label}: joint_checkpoint mismatch")
        if a.get("hidden_dims") != ref_ckpt_args.get("hidden_dims"):
            errors.append(f"{label}: hidden_dims mismatch")
        if not np.isclose(
            float(diag["mmd_heldout_bandwidth"]),
            float(ref_diag["mmd_heldout_bandwidth"]),
            atol=1e-6,
            rtol=1e-6,
        ):
            errors.append(f"{label}: held-out MMD bandwidth mismatch")
    if errors:
        print("  Compatibility check FAILED — cannot generate comparison plots:")
        for e in errors:
            print(f"    {e}")
        return

    epochs = np.arange(1, ref_n_epochs + 1)

    cmap_tab10 = plt.get_cmap("tab10")
    cmap_tab20 = plt.get_cmap("tab20")
    group_colors: dict[str, object] = {"baseline": "#4C72B0"}
    for i, gkey in enumerate(aug_group_keys):
        group_colors[gkey] = cmap_tab10(i % 10)

    w = min(0.7 / max(n_groups, 1), 0.35)
    offsets = np.linspace(-(n_groups - 1) * w / 2, (n_groups - 1) * w / 2, n_groups)

    # ── Plot 1: Collapse diagnostics panel ──────────────────────────────────
    diag_keys = [
        ("std_mu",        "std(mu) per latent dim\n[PRIMARY collapse detector]"),
        ("mean_sigma",    "mean(sigma) per latent dim"),
        ("second_moment", "E[mu²+exp(log_var)] per latent dim\n[secondary — not reliable alone]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, title) in zip(axes, diag_keys):
        x_dim = np.arange(ref_latent_dim)
        for glabel, group_runs in all_groups:
            curves = [np.sort(diag[key])[::-1] for diag, _, _, _ in group_runs]
            arr    = np.stack(curves)
            mean_  = arr.mean(axis=0)
            std_   = arr.std(axis=0)
            ax.plot(x_dim, mean_, color=group_colors[glabel], label=glabel, linewidth=1.5)
            ax.fill_between(x_dim, mean_ - std_, mean_ + std_, color=group_colors[glabel], alpha=0.25)
        ax.set_xlabel("Latent dimension (sorted per seed)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
    plt.suptitle("Collapse diagnostics: baseline vs augmented", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "collapse_diagnostics_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved collapse_diagnostics_panel.png")

    # ── Plot 2: KL trajectory ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, kl_key, title in [
        (axes[0], "val_kl_mean_per_dim", "Val KL mean per dim"),
        (axes[1], "val_kl_sum_dims",     "Val KL summed across dims"),
    ]:
        for glabel, group_runs in all_groups:
            curves = [kl[kl_key] for _, kl, _, _ in group_runs]
            arr    = np.stack(curves)
            mean_  = arr.mean(axis=0)
            std_   = arr.std(axis=0)
            ax.plot(epochs, mean_, color=group_colors[glabel], label=glabel, linewidth=1.5)
            ax.fill_between(epochs, mean_ - std_, mean_ + std_, color=group_colors[glabel], alpha=0.25)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.suptitle("KL trajectory over epochs", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "kl_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved kl_trajectory.png")

    # ── Plot 3: MMD two-panel bar chart ─────────────────────────────────────
    ref_combos = sorted(_load_mmd_seen(ref_diag).keys())
    x_pos = np.arange(len(ref_combos))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(ref_combos) * 1.2), 10))
    ax = axes[0]
    for gi, (glabel, group_runs) in enumerate(all_groups):
        gm = np.array([
            np.mean([_load_mmd_seen(d)[c]["mmd"] for d, _, _, _ in group_runs if c in _load_mmd_seen(d)])
            for c in ref_combos
        ])
        gs_ = np.array([
            np.std([_load_mmd_seen(d)[c]["mmd"]  for d, _, _, _ in group_runs if c in _load_mmd_seen(d)])
            for c in ref_combos
        ])
        ax.bar(x_pos + offsets[gi], gm, w, yerr=gs_,
               label=glabel, color=group_colors[glabel], capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ref_combos, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MMD")
    ax.set_title("Seen-class MMD (mean ± std across seeds)")
    ax.legend()

    ax = axes[1]
    for gi, (glabel, group_runs) in enumerate(all_groups):
        held_vals = [float(d["mmd_heldout"]) for d, _, _, _ in group_runs]
        hm = float(np.mean(held_vals))
        hs = float(np.std(held_vals))
        ax.bar([offsets[gi]], [hm], w, yerr=[[hs]],
               label=glabel, color=group_colors[glabel], capsize=6)
    held_label = "+".join(str(v) for v in ref_diag["heldout_combo"].tolist())
    ax.set_xticks([0])
    ax.set_xticklabels([held_label])
    ax.set_ylabel("MMD")
    ax.set_title("Held-out class MMD (mean ± std across seeds)")
    ax.legend()
    plt.suptitle("MMD comparison: seen vs held-out (never merged)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "mmd_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved mmd_comparison.png")

    # ── Plot 4: Procrustes-aligned 2D PCA ───────────────────────────────────
    ref_mu       = all_baseline[0][0]["mu_all"]
    ref_combos_v = all_baseline[0][0]["val_combo_labels"]
    pca_ref = PCA(n_components=2, random_state=0)
    pca_ref.fit(ref_mu)
    unique_c = np.unique(ref_combos_v)

    def _plot_aligned(ax, runs_data, title):
        # Per-seed Procrustes-aligned centroids, averaged across seeds.
        all_centroids: dict[int, list] = {int(c): [] for c in unique_c}
        for ri, (diag, _, _, _) in enumerate(runs_data):
            run_mu = diag["mu_all"]
            R, _ = orthogonal_procrustes(run_mu, ref_mu)
            aligned = run_mu @ R
            pc = pca_ref.transform(aligned)
            combo_v = diag["val_combo_labels"]
            for ci, combo in enumerate(unique_c):
                mask = combo_v == combo
                ax.scatter(pc[mask, 0], pc[mask, 1],
                           c=[cmap_tab20(ci % 20)], alpha=0.3, s=6,
                           label=str(combo) if ri == 0 else None)
                if mask.any():
                    all_centroids[int(combo)].append(
                        (float(pc[mask, 0].mean()), float(pc[mask, 1].mean()))
                    )
        for ci, combo in enumerate(unique_c):
            pts = all_centroids[int(combo)]
            if pts:
                cx = float(np.mean([p[0] for p in pts]))
                cy = float(np.mean([p[1] for p in pts]))
                ax.scatter(cx, cy, c=[cmap_tab20(ci % 20)], s=80,
                           marker="*", edgecolors="black", linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=6, ncol=2, title="combo", title_fontsize=6)

    fig, axes = plt.subplots(1, n_groups, figsize=(8 * n_groups, 6), squeeze=False)
    for ax, (glabel, group_runs) in zip(axes[0], all_groups):
        _plot_aligned(ax, group_runs, glabel)
    plt.suptitle("Latent space 2D PCA with Procrustes alignment", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "latent_pca_procrustes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved latent_pca_procrustes.png")

    # ── Plot 5: Reconstruction vs KL tradeoff scatter ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for diag, kl, ckpt, run_dir in all_baseline:
        best_recon = float(ckpt.get("best_val_recon", float("nan")))
        best_kl    = float(ckpt.get("best_val_kl_sum", float("nan")))
        ax.scatter(best_recon, best_kl, c=[group_colors["baseline"]], s=80,
                   marker="o", edgecolors="black", linewidths=0.6,
                   label="baseline (n_drop=0)")
    for gkey in aug_group_keys:
        for diag, kl, ckpt, run_dir in aug_groups[gkey]:
            best_recon = float(ckpt.get("best_val_recon", float("nan")))
            best_kl    = float(ckpt.get("best_val_kl_sum", float("nan")))
            ax.scatter(best_recon, best_kl, c=[group_colors[gkey]], s=80,
                       marker="^", edgecolors="black", linewidths=0.6,
                       label=gkey)
    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.set_xlabel("Best val reconstruction loss")
    ax.set_ylabel("Best val KL (summed across dims)")
    ax.set_title("Reconstruction vs KL tradeoff\n(o=baseline, ^=augmented; colour=aug config)")
    plt.tight_layout()
    plt.savefig(out_dir / "recon_vs_kl_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved recon_vs_kl_scatter.png")

    print(f"\nAll comparison plots saved to {out_dir}")


def main(argv=None) -> dict:
    args = parse_args(argv)

    # ── --compare_only: no model or data needed, just load saved files ───────
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

    # ── Load sentence condition table once; attach to args for helper functions ─
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

    # ── --diag_only: rebuild model from saved checkpoint, skip training ──────
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
        train_p, val_p, held_p = split_payload(payload, args.split_seed)
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

    # ── Normal training path ─────────────────────────────────────────────────
    payload, transformer_model = build_embedding_payload(args, device)
    # Stable global sample indices, carried through split_payload via _subset().
    payload["sample_index"] = np.arange(len(payload["y_grip"]), dtype=np.int64)
    train_p, val_p, held_p = split_payload(payload, args.split_seed)
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
    gen = evaluate_generation(model, transformer_model, train_p, held_p, stats, condition, out_dir, args, device)

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
