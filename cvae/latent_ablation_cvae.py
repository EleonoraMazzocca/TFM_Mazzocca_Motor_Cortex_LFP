"""Latent ablation diagnostic for cVAE checkpoints.

Compares generation quality under three z-injection modes to directly test
whether the decoder uses z or ignores it:

  prior    — z ~ N(0, I)          standard generation
  zero     — z = 0                z entirely ablated
  shuffled — z encoded from val samples, shuffled across samples
              (tests whether sample-specific structure in z matters)

No retraining. Read-only with respect to checkpoints.

Usage:
    python latent_ablation_cvae.py \\
        --run_dirs results/broadband6/cvae_grasp_precision_right_seed42 \\
                   results/broadband6/cvae_grasp_precision_right_mmd_l0.3_s42 \\
                   results/broadband6/cvae_grasp_precision_right_mmd_l10_s42 \\
        --data_dir /path/to/data \\
        --input_mode broadband6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup — mirrors run_embedding_cvae.py
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
from transformer_encoder.data import PHASE_NAMES, GRIP_TO_ID, HAND_TO_ID
from cvae.cvae_data import make_condition_vector
from cvae.cvae_model import LFPCVAE
from run_cvae import compute_mmd
from run_cvae_embeddings import (
    build_embedding_payload,
    split_payload,
    EmbeddingCVAEDataset,
    lookup_condition,
)

ID_TO_GRIP = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND = {v: k for k, v in HAND_TO_ID.items()}

_MODES = ("prior", "zero", "shuffled")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ckpt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_args(saved_args: dict, data_dir: str | None, input_mode: str | None) -> argparse.Namespace:
    """Reconstruct args namespace from saved checkpoint args with optional CLI overrides."""
    ns = argparse.Namespace(**saved_args)
    if data_dir:
        ns.data_dir = data_dir
    if input_mode and hasattr(ns, "joint_input_mode"):
        ns.joint_input_mode = input_mode
    # Ensure fields that build_embedding_payload requires exist
    for attr, default in [
        ("dry_run", False), ("batch_size", 128), ("seed", 42),
        ("d_model", 64), ("n_heads", 4), ("n_layers", 2),
        ("feedforward_dim", 128), ("dropout", 0.2),
        ("joint_cache_dir", "/tmp/lfp_joint_embedding_cache"),
        ("checkpoint_reach", None), ("checkpoint_prereach", None),
        ("checkpoint_grasp", None),
    ]:
        if not hasattr(ns, attr):
            setattr(ns, attr, default)
    return ns


def _compute_metrics(
    x_gen:       np.ndarray,
    x_real:      np.ndarray,
    x_train:     np.ndarray,
    train_combo: np.ndarray,
    target_combo: int,
    stats:       dict,
    transformer_model,
    device:      torch.device,
    target_idxs: dict,
) -> dict:
    """Compute generation metrics for pre-generated normalized embeddings.

    All arrays are z-scored (normalized) embeddings.
    Metric definitions match evaluate_generation() in run_cvae_embeddings.py.
    """
    # --- MMD ratio ---
    mmd_gen  = compute_mmd(x_gen,  x_real)
    half     = len(x_real) // 2
    mmd_base = compute_mmd(x_real[:half], x_real[half:]) if half > 1 else float("nan")
    mmd_ratio = float(mmd_gen / max(mmd_base, 1e-10)) if not np.isnan(mmd_base) else float("nan")

    # --- Centroid placement ---
    target_centroid = x_real.mean(axis=0)
    gen_centroid    = x_gen.mean(axis=0)
    centroid_dist   = float(np.linalg.norm(gen_centroid - target_centroid))

    centroids: dict[int, np.ndarray] = {}
    for combo in np.unique(train_combo):
        centroids[int(combo)] = x_train[train_combo == combo].mean(axis=0)
    centroids[target_combo] = target_centroid
    centroid_keys = np.array(sorted(centroids))
    centroid_mat  = np.stack([centroids[int(k)] for k in centroid_keys])
    other_mask    = centroid_keys != target_combo
    target_to_other = np.linalg.norm(
        centroid_mat[other_mask] - target_centroid[None, :], axis=1
    )
    target_to_other_mean = float(np.mean(target_to_other)) if len(target_to_other) else float("nan")
    relative_centroid    = float(centroid_dist / max(target_to_other_mean, 1e-10))

    dists = np.linalg.norm(x_gen[:, None, :] - centroid_mat[None, :, :], axis=2)
    nearest_target_rate = float((centroid_keys[np.argmin(dists, axis=1)] == target_combo).mean())

    # --- Transformer head accuracy ---
    head_acc = None
    if transformer_model is not None:
        mu_np    = stats["mu"].astype(np.float32)
        sigma_np = stats["sigma"].astype(np.float32)
        x_denorm = torch.tensor(x_gen * sigma_np + mu_np, dtype=torch.float32).to(device)
        transformer_model.eval()
        with torch.no_grad():
            preds = {
                "phase": transformer_model.head_phase(x_denorm).argmax(1).cpu().numpy(),
                "grip":  transformer_model.head_grip(x_denorm).argmax(1).cpu().numpy(),
                "hand":  transformer_model.head_hand(x_denorm).argmax(1).cpu().numpy(),
            }
        head_acc = {f: float((preds[f] == target_idxs[f]).mean()) for f in ("phase", "grip", "hand")}

    return {
        "mmd_ratio":           mmd_ratio,
        "centroid_distance":   centroid_dist,
        "relative_centroid":   relative_centroid,
        "nearest_target_rate": nearest_target_rate,
        "head_accuracy":       head_acc,
    }


# ---------------------------------------------------------------------------
# Per-run ablation
# ---------------------------------------------------------------------------

def run_ablation(
    run_dir:    Path,
    data_dir:   str | None,
    input_mode: str | None,
    n_samples:  int,
    gen_seed:   int,
    device:     torch.device,
) -> dict:
    run_dir = Path(run_dir)
    print(f"\n{'='*60}")
    print(f"  {run_dir.name}")
    print(f"{'='*60}")

    # --- Load checkpoint ---
    ckpt = _load_ckpt(run_dir / "checkpoint.pt")
    saved_args = ckpt.get("args", {})
    args = _build_args(saved_args, data_dir, input_mode)

    # --- Condition setup from checkpoint (must happen before build_embedding_payload) ---
    condition_type = saved_args.get("condition_type", "onehot")
    if condition_type == "sentence":
        _condition_table     = np.load(saved_args["sentence_condition_path"])
        _condition_key_order = np.load(saved_args["sentence_key_order_path"])
        _condition_dim       = int(_condition_table.shape[1])
    else:
        _condition_table     = None
        _condition_key_order = None
        _condition_dim       = 7
    # Attach to args so build_embedding_payload uses the correct condition vectors.
    args.condition_table     = _condition_table
    args.condition_key_order = _condition_key_order
    args.condition_dim       = _condition_dim
    args.condition_type      = condition_type

    # --- Rebuild payload and transformer ---
    print("  Loading embedding payload...")
    payload, transformer_model = build_embedding_payload(args, device)
    payload["sample_index"] = np.arange(len(payload["y_grip"]), dtype=np.int64)

    split_seed = int(saved_args.get("split_seed", saved_args.get("seed", 42)))
    train_p, val_p, held_p = split_payload(payload, split_seed)

    # --- Normalization stats ---
    stats_raw = np.load(run_dir / "normalization_stats.npz")
    stats = {
        "mu":    stats_raw["mu"].astype(np.float32),
        "sigma": stats_raw["sigma"].astype(np.float32),
    }
    val_ds = EmbeddingCVAEDataset(val_p, stats)

    # --- Rebuild cVAE model ---
    input_dim = int(train_p["embeddings"].shape[1])
    model = LFPCVAE(
        input_dim     = input_dim,
        condition_dim = _condition_dim,
        latent_dim    = int(saved_args.get("latent_dim", 32)),
        hidden_dims   = list(saved_args.get("hidden_dims", [128, 64, 32])),
        dropout       = float(saved_args.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- Held-out condition and reference data ---
    heldout_phase = PHASE_NAMES.index(saved_args["heldout_phase"])
    heldout_grip  = GRIP_TO_ID[saved_args["heldout_grip"]]
    heldout_hand  = HAND_TO_ID[saved_args["heldout_hand"]]
    condition = (
        lookup_condition(heldout_phase, heldout_grip, heldout_hand,
                         _condition_table, _condition_key_order)
        if _condition_table is not None
        else make_condition_vector(heldout_phase, heldout_grip, heldout_hand)
    )
    target_combo  = heldout_grip * 6 + heldout_hand * 3 + heldout_phase
    target_idxs   = {"phase": heldout_phase, "grip": heldout_grip, "hand": heldout_hand}

    c_batch = torch.tensor(condition, dtype=torch.float32).unsqueeze(0).expand(n_samples, -1).to(device)

    x_real  = (held_p["embeddings"].astype(np.float32)  - stats["mu"]) / stats["sigma"]
    x_train = (train_p["embeddings"].astype(np.float32) - stats["mu"]) / stats["sigma"]
    train_combo_arr = (
        train_p["y_grip"].astype(np.int64) * 6
        + train_p["y_hand"].astype(np.int64) * 3
        + train_p["y_phase"].astype(np.int64)
    )

    # CPU generator — avoids CUDA device mismatch, ensures reproducibility
    rng = torch.Generator()
    rng.manual_seed(gen_seed)

    results: dict[str, dict] = {}

    with torch.no_grad():
        # ---- Mode 1: prior z ~ N(0, I) ----
        z = torch.randn(n_samples, model.latent_dim, generator=rng).to(device)
        x_gen = model.decode(z, c_batch).cpu().numpy()
        results["prior"] = _compute_metrics(
            x_gen, x_real, x_train, train_combo_arr, target_combo,
            stats, transformer_model, device, target_idxs,
        )

        # ---- Mode 2: z = 0 ----
        z = torch.zeros(n_samples, model.latent_dim, device=device)
        x_gen = model.decode(z, c_batch).cpu().numpy()
        results["zero"] = _compute_metrics(
            x_gen, x_real, x_train, train_combo_arr, target_combo,
            stats, transformer_model, device, target_idxs,
        )

        # ---- Mode 3: shuffled encoded z from validation set ----
        zs = []
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
        for x_b, c_b, *_ in val_loader:
            mu_b, lv_b = model.encode(x_b.to(device), c_b.to(device))
            # reparameterize on CPU to keep generator device-agnostic
            mu_cpu  = mu_b.cpu()
            std_cpu = torch.exp(0.5 * lv_b).cpu()
            eps     = torch.randn_like(std_cpu, generator=rng)
            zs.append(mu_cpu + std_cpu * eps)
        z_pool = torch.cat(zs, dim=0)   # (N_val, latent_dim) on CPU
        idx    = torch.randperm(len(z_pool), generator=rng)[:n_samples]
        z      = z_pool[idx].to(device)
        x_gen  = model.decode(z, c_batch).cpu().numpy()
        results["shuffled"] = _compute_metrics(
            x_gen, x_real, x_train, train_combo_arr, target_combo,
            stats, transformer_model, device, target_idxs,
        )

    # --- Save per-run results ---
    (run_dir / "latent_ablation.json").write_text(
        json.dumps({"run": run_dir.name, "n_samples": n_samples,
                    "gen_seed": gen_seed, "results": results},
                   indent=2, default=float),
        encoding="utf-8",
    )
    print(f"  Saved latent_ablation.json")
    return results


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _print_table(all_results: dict[str, dict[str, dict]]) -> None:
    ha_key = "head_accuracy"
    hdr = (f"{'run':<32} {'mode':<10} {'mmd_ratio':>10} {'centroid':>9} "
           f"{'rel_c':>7} {'near%':>7} {'phase':>7} {'grip':>7} {'hand':>7}")
    print("\n" + "="*len(hdr))
    print(hdr)
    print("-"*len(hdr))
    for run_name, modes in all_results.items():
        short = run_name[-30:] if len(run_name) > 30 else run_name
        for mode in _MODES:
            m = modes[mode]
            ha = m.get(ha_key) or {}
            print(
                f"{short:<32} {mode:<10}"
                f" {m['mmd_ratio']:>10.3f}"
                f" {m['centroid_distance']:>9.3f}"
                f" {m['relative_centroid']:>7.3f}"
                f" {m['nearest_target_rate']:>7.3f}"
                f" {ha.get('phase', float('nan')):>7.3f}"
                f" {ha.get('grip',  float('nan')):>7.3f}"
                f" {ha.get('hand',  float('nan')):>7.3f}"
            )
        print()
    print("="*len(hdr))


def _print_interpretation(all_results: dict[str, dict[str, dict]]) -> None:
    THRESHOLD = 0.05   # centroid ratio difference below which change is noise
    print("\nInterpretation summary (centroid distance, threshold = 0.05):")
    for run_name, modes in all_results.items():
        c_prior    = modes["prior"]["centroid_distance"]
        c_zero     = modes["zero"]["centroid_distance"]
        c_shuffled = modes["shuffled"]["centroid_distance"]
        best_mode  = min(modes, key=lambda m: modes[m]["centroid_distance"])

        diff_pz = abs(c_prior - c_zero)
        diff_ps = abs(c_prior - c_shuffled)

        if diff_pz < THRESHOLD and diff_ps < THRESHOLD:
            verdict = "decoder ignores z — all modes equivalent"
        elif c_zero < c_prior - THRESHOLD:
            verdict = "prior z HURTS generation — sampled noise degrades output"
        elif diff_pz >= THRESHOLD or diff_ps >= THRESHOLD:
            verdict = f"z matters — best mode: {best_mode}"
        else:
            verdict = "inconclusive"

        print(f"  {run_name[-45:]:<45}  {verdict}")
        print(f"    prior={c_prior:.3f}  zero={c_zero:.3f}  shuffled={c_shuffled:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Latent ablation: compare prior / zero / shuffled z generation."
    )
    p.add_argument("--run_dirs", type=str, nargs="+", required=True,
                   help="cVAE result directories to analyse.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Override data_dir from checkpoint (use when running on a different machine).")
    p.add_argument("--input_mode", choices=["mu", "broadband6"], default=None,
                   help="Override joint_input_mode from checkpoint.")
    p.add_argument("--n_samples", type=int, default=500,
                   help="Number of samples to generate per mode (default 500).")
    p.add_argument("--generation_seed", type=int, default=0,
                   help="Fixed seed for all random generation (default 0).")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    args = p.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}  |  n_samples: {args.n_samples}  |  gen_seed: {args.generation_seed}")

    all_results: dict[str, dict] = {}
    for run_dir in args.run_dirs:
        name = Path(run_dir).name
        all_results[name] = run_ablation(
            run_dir    = Path(run_dir),
            data_dir   = args.data_dir,
            input_mode = args.input_mode,
            n_samples  = args.n_samples,
            gen_seed   = args.generation_seed,
            device     = device,
        )

    _print_table(all_results)
    _print_interpretation(all_results)

    # Combined output next to the first run_dir's parent
    out_path = Path(args.run_dirs[0]).parent / "latent_ablation_combined.json"
    out_path.write_text(
        json.dumps(all_results, indent=2, default=float), encoding="utf-8"
    )
    print(f"\nCombined results saved to {out_path}")


if __name__ == "__main__":
    main()
