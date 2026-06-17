"""Standalone evaluation of cVAE generation quality.

Run after run_joint_embedding.py and run_embedding_cvae.py.
Loads saved artifacts (no retraining) and produces metrics and plots covering:
  1. Embedding space quality   (PCA variance, latent Gaussianity)
  2. Embedding structure       (scatter plots, centroid matrix, compositionality)
  3. Geometric quality         (centroid distance ratio)
  4. Semantic quality          (classifier probe, confusion matrices, class distributions)
  5. Distributional quality    (sliced Wasserstein, MMD, per-dim KS, marginal KDEs)

All geometry/distribution comparisons operate in cVAE-normalized embedding space.
Only the classifier probe uses denormalized embeddings (transformer heads expect original scale).

Usage:
    python evaluate_generation.py \\
        --joint_checkpoint  results/broadband6/transformer_heldout_grasp_precision_right/checkpoint.pt \\
        --cvae_checkpoint   results/broadband6/cvae_grasp_precision_right/checkpoint.pt \\
        --seen_embeddings   results/broadband6/transformer_heldout_grasp_precision_right/seen_embeddings.npz \\
        --heldout_embeddings results/broadband6/transformer_heldout_grasp_precision_right/heldout_embeddings.npz \\
        --cvae_norm_stats   results/broadband6/cvae_grasp_precision_right/normalization_stats.npz \\
        --heldout_phase grasp --heldout_grip precision --heldout_hand right \\
        --n_generate 500 --out_dir results/evaluation/grasp_precision_right
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance, kstest
from sklearn.decomposition import PCA

_HERE = Path(__file__).resolve().parent
from transformer_encoder.joint_embedding_data import PHASE_NAMES, GRIP_TO_ID, HAND_TO_ID  # noqa: E402
from cvae.conditioning.onehot import CONDITION_DIM, make_condition_vector  # noqa: E402
from cvae.conditioning.sentence import lookup_condition  # noqa: E402
from cvae.cvae_model import LFPCVAE  # noqa: E402
from transformer_encoder.joint_embedding_model import JointFactorTransformer  # noqa: E402
from transformer_encoder.joint_embedding_data import BAND_NAMES_6  # noqa: E402
from cvae.metrics import compute_mmd  # noqa: E402

ID_TO_GRIP = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND = {v: k for k, v in HAND_TO_ID.items()}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 100,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    directions = rng.standard_normal((n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    total = sum(wasserstein_distance(x @ d, y @ d) for d in directions)
    return float(total / n_projections)


def compute_centroids(
    embeddings: np.ndarray,
    y_phase: np.ndarray,
    y_grip: np.ndarray,
    y_hand: np.ndarray,
) -> dict:
    """Return dict mapping (phase_idx, grip_idx, hand_idx) → mean embedding."""
    centroids = {}
    for ph in range(len(PHASE_NAMES)):
        for gi in (0, 1):
            for hi in (0, 1):
                mask = (y_phase == ph) & (y_grip == gi) & (y_hand == hi)
                if mask.any():
                    centroids[(ph, gi, hi)] = embeddings[mask].mean(axis=0)
    return centroids


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_transformer(args, device):
    ckpt = _torch_load(args.joint_checkpoint, device)
    config = ckpt.get("config", {})
    n_bands = 1 if config.get("input_mode", "mu") == "mu" else len(BAND_NAMES_6)
    model = JointFactorTransformer(
        n_bands=n_bands,
        d_model=int(config.get("d_model", args.d_model)),
        n_heads=int(config.get("n_heads", args.n_heads)),
        n_layers=int(config.get("n_layers", args.n_layers)),
        feedforward_dim=int(config.get("feedforward_dim", args.feedforward_dim)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_cvae(args, device) -> tuple:
    """Load cVAE checkpoint.

    Returns (model, latent_dim, pca_data, saved_args).
    saved_args is the dict stored in the checkpoint under "args"; callers use it
    to reconstruct the condition type and sentence table paths.
    pca_data is None unless the checkpoint was trained with --pca_components > 0,
    in which case it is a dict with keys 'components' (n_comp × emb_dim) and 'mean' (emb_dim,).
    """
    ckpt = _torch_load(args.cvae_checkpoint, device)
    saved = ckpt.get("args", {})
    latent_dim  = saved.get("latent_dim",  16)
    hidden_dims = saved.get("hidden_dims", [128, 64, 32])
    dropout     = saved.get("dropout",     0.2)
    pca_n       = saved.get("pca_components", 0)
    pca_data    = ckpt.get("pca_data", None)

    if pca_n > 0 and pca_data is None:
        print(f"  WARNING: checkpoint missing pca_data (saved before final write). "
              f"Reconstructing PCA({pca_n}) from seen embeddings.")

    # Read condition_dim from checkpoint; fall back to 7 (one-hot) for old runs.
    condition_dim = saved.get("condition_dim", CONDITION_DIM)

    input_dim = pca_n if pca_n > 0 else 64
    model = LFPCVAE(
        input_dim=input_dim,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, latent_dim, pca_data, saved


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint_checkpoint",   type=str, required=True)
    p.add_argument("--cvae_checkpoint",    type=str, required=True)
    p.add_argument("--seen_embeddings",    type=str, required=True)
    p.add_argument("--heldout_embeddings", type=str, required=True)
    p.add_argument("--cvae_norm_stats",    type=str, required=True,
                   help="normalization_stats.npz from the cVAE output dir (NOT the transformer one)")
    p.add_argument("--heldout_phase", choices=PHASE_NAMES,             default="grasp")
    p.add_argument("--heldout_grip",  choices=["power", "precision"],  default="precision")
    p.add_argument("--heldout_hand",  choices=["left", "right"],       default="right")
    p.add_argument("--n_generate",    type=int,  default=500)
    p.add_argument("--n_projections", type=int,  default=100,
                   help="Random projections for sliced Wasserstein.")
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--out_dir",       type=str,  default="results/evaluation/")
    p.add_argument("--no_umap",       action="store_true")
    p.add_argument("--no_plot",       action="store_true")
    p.add_argument("--device",        choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--d_model",       type=int,  default=64)
    p.add_argument("--n_heads",       type=int,  default=4)
    p.add_argument("--n_layers",      type=int,  default=2)
    p.add_argument("--feedforward_dim", type=int, default=128)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    args = parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ────────────────────────────────────────────────────────────
    seen  = {k: v for k, v in np.load(args.seen_embeddings).items()}
    held  = {k: v for k, v in np.load(args.heldout_embeddings).items()}
    stats = {k: v for k, v in np.load(args.cvae_norm_stats).items()}

    mu_norm    = stats["mu"].astype(np.float32)
    sigma_norm = stats["sigma"].astype(np.float32)
    emb_dim    = int(seen["embeddings"].shape[1])

    if mu_norm.shape != (emb_dim,):
        raise SystemExit(
            f"cvae_norm_stats mu shape {mu_norm.shape} != ({emb_dim},). "
            "Pass the cVAE normalization_stats.npz, not the transformer one."
        )
    if sigma_norm.shape != (emb_dim,):
        raise SystemExit(f"cvae_norm_stats sigma shape {sigma_norm.shape} != ({emb_dim},).")
    if not (sigma_norm > 0).all():
        raise SystemExit("cvae_norm_stats sigma contains zero or negative values — normalization would divide by zero.")

    # cVAE-normalized embeddings (zeros kept zero, matching training)
    seen_emb  = seen["embeddings"].astype(np.float32)
    held_emb  = held["embeddings"].astype(np.float32)
    seen_norm = np.where(seen_emb == 0.0, 0.0, (seen_emb - mu_norm) / sigma_norm)
    held_norm = np.where(held_emb == 0.0, 0.0, (held_emb - mu_norm) / sigma_norm)

    heldout_phase_idx = PHASE_NAMES.index(args.heldout_phase)
    heldout_grip_id   = GRIP_TO_ID[args.heldout_grip]
    heldout_hand_id   = HAND_TO_ID[args.heldout_hand]

    # ── Load models ──────────────────────────────────────────────────────────
    transformer_model                        = load_transformer(args, device)
    cvae_model, latent_dim, pca_data, saved_cvae_args = load_cvae(args, device)

    # Build condition helpers that match what the cVAE was trained with.
    condition_type = saved_cvae_args.get("condition_type", "onehot")
    if condition_type == "sentence":
        _cond_table    = np.load(saved_cvae_args["sentence_condition_path"])
        _cond_key_order = np.load(saved_cvae_args["sentence_key_order_path"])
        def _make_cond(ph, gr, ha):
            return lookup_condition(ph, gr, ha, _cond_table, _cond_key_order)
    else:
        _cond_table = _cond_key_order = None
        def _make_cond(ph, gr, ha):
            return make_condition_vector(ph, gr, ha)

    print(f"\n{'='*80}")
    print(f"  GENERATION EVALUATION")
    print(f"  Held-out: {args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}"
          f"  |  n_generate={args.n_generate}")
    print(f"  Condition type: {condition_type}")
    print(f"  Device:   {device}  |  Output: {out_dir}")
    print(f"{'='*80}")

    # ── Generate samples ─────────────────────────────────────────────────────
    cond     = _make_cond(heldout_phase_idx, heldout_grip_id, heldout_hand_id)
    c_tensor = torch.tensor(cond, dtype=torch.float32)
    gen_rng  = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        x_gen_norm = cvae_model.generate(
            c_tensor, n_samples=args.n_generate, device=device, generator=gen_rng
        ).cpu().numpy()

    if pca_data is not None:
        components = pca_data["components"]
        pca_mean   = pca_data["mean"]
        x_gen_norm = x_gen_norm @ components + pca_mean

    x_gen_denorm  = x_gen_norm * sigma_norm + mu_norm
    x_held_denorm = held_emb

    # ── Centroids (normalized space) ─────────────────────────────────────────
    seen_centroids = compute_centroids(
        seen_norm, seen["y_phase"], seen["y_grip"], seen["y_hand"]
    )
    held_centroid = held_norm.mean(axis=0)
    gen_centroid  = x_gen_norm.mean(axis=0)

    seen_keys  = list(seen_centroids.keys())
    seen_cents = np.stack([seen_centroids[k] for k in seen_keys])
    dists_held_to_seen = np.linalg.norm(seen_cents - held_centroid[None, :], axis=1)
    nearest_seen_dist  = float(dists_held_to_seen.min())
    nearest_seen_key   = seen_keys[int(np.argmin(dists_held_to_seen))]

    # ── Section 1A: PCA variance ─────────────────────────────────────────────
    print("\n── Section 1A: PCA variance ──────────────────────────────────────────────────")
    pca_full = PCA(n_components=min(10, emb_dim), random_state=args.seed)
    pca_full.fit(seen_norm)
    ev = pca_full.explained_variance_ratio_
    pca_pc1pc2    = float(ev[0] + ev[1])
    pca_pc1pc2pc3 = float(ev[0] + ev[1] + ev[2]) if len(ev) > 2 else pca_pc1pc2
    print(f"\nPCA explained variance (seen test embeddings):")
    for i in range(min(5, len(ev))):
        print(f"  PC{i+1}: {ev[i]:.3f}   cumulative: {ev[:i+1].sum():.3f}")
    print(f"  PC1+PC2:      {pca_pc1pc2:.3f}")
    print(f"  PC1+PC2+PC3:  {pca_pc1pc2pc3:.3f}")

    # ── Section 1B: Latent Gaussianity ────────────────────────────────────────
    print("\n── Section 1B: Latent Gaussianity ────────────────────────────────────────────")
    cvae_model.eval()
    mu_list = []
    lv_list = []
    bs = 256
    for i in range(0, len(seen_norm), bs):
        x_b  = torch.tensor(seen_norm[i:i+bs], dtype=torch.float32).to(device)
        ph_b = seen["y_phase"][i:i+bs]
        gr_b = seen["y_grip"][i:i+bs]
        ha_b = seen["y_hand"][i:i+bs]
        c_b  = torch.tensor(
            np.stack([_make_cond(int(p), int(g), int(h))
                      for p, g, h in zip(ph_b, gr_b, ha_b)]),
            dtype=torch.float32,
        ).to(device)
        with torch.no_grad():
            mu_b, lv_b = cvae_model.encode(x_b, c_b)
        mu_list.append(mu_b.cpu().numpy())
        lv_list.append(lv_b.cpu().numpy())
    mu_all = np.concatenate(mu_list, axis=0)
    lv_all = np.concatenate(lv_list, axis=0)
    sigma_all = np.exp(0.5 * lv_all)

    latent_mean = float(mu_all.mean())
    latent_std  = float(mu_all.std())
    posterior_std_mean = float(sigma_all.mean())
    ks_pct_latent = float(
        np.mean([kstest(mu_all[:, j], "norm").pvalue > 0.05 for j in range(mu_all.shape[1])])
    )
    print(f"\nLatent space Gaussianity (seen test embeddings — indicative only):")
    print(f"  Mean across dims:       {latent_mean:.4f}  (target 0.0)")
    print(f"  Std  across dims:       {latent_std:.4f}   (target 1.0)")
    print(f"  Posterior std mean:     {posterior_std_mean:.4f}  (target ~1.0)")
    print(f"  % dims p>0.05 (KS):    {ks_pct_latent*100:.1f}%  (higher = better regularized)")

    # ── Section 2C: Compositionality test ─────────────────────────────────────
    print("\n── Section 2C: Compositionality test ─────────────────────────────────────────")
    ph  = heldout_phase_idx
    gi  = heldout_grip_id
    hi  = heldout_hand_id
    gi_ = 1 - gi   # alternate grip
    hi_ = 1 - hi   # alternate hand

    # Grip/hand analogy: c(ph,gi,hi') + c(ph,gi',hi) - c(ph,gi',hi')
    analogy_pred = None
    if all(k in seen_centroids for k in [(ph,gi,hi_), (ph,gi_,hi), (ph,gi_,hi_)]):
        analogy_pred = (
            seen_centroids[(ph, gi, hi_)]
            + seen_centroids[(ph, gi_, hi)]
            - seen_centroids[(ph, gi_, hi_)]
        )

    analogical_error_gen  = float(np.linalg.norm(gen_centroid  - analogy_pred)) if analogy_pred is not None else float("nan")
    analogical_error_real = float(np.linalg.norm(held_centroid - analogy_pred)) if analogy_pred is not None else float("nan")
    compositionality_score = (
        float(analogical_error_gen / analogical_error_real)
        if analogy_pred is not None and analogical_error_real > 0
        else float("nan")
    )

    # Phase transfer: c(ph-1,gi,hi) + c(ph,gi',hi) - c(ph-1,gi',hi)
    ph_ = (ph - 1) % len(PHASE_NAMES)
    phase_pred = None
    if all(k in seen_centroids for k in [(ph_,gi,hi), (ph,gi_,hi), (ph_,gi_,hi)]):
        phase_pred = (
            seen_centroids[(ph_, gi,  hi)]
            + seen_centroids[(ph,  gi_, hi)]
            - seen_centroids[(ph_, gi_, hi)]
        )

    phase_error_gen  = float(np.linalg.norm(gen_centroid  - phase_pred)) if phase_pred is not None else float("nan")
    phase_error_real = float(np.linalg.norm(held_centroid - phase_pred)) if phase_pred is not None else float("nan")

    # ── Section 3: Geometric quality ─────────────────────────────────────────
    print("\n── Section 3: Geometric quality ──────────────────────────────────────────────")
    centroid_dist_gen_target = float(np.linalg.norm(gen_centroid - held_centroid))
    centroid_dist_ratio      = (
        float(centroid_dist_gen_target / nearest_seen_dist)
        if nearest_seen_dist > 0 else float("nan")
    )

    # ── Section 4: Classifier probe ───────────────────────────────────────────
    print("\n── Section 4: Classifier probe ───────────────────────────────────────────────")
    transformer_model.eval()
    x_gen_t  = torch.tensor(x_gen_denorm,  dtype=torch.float32).to(device)
    x_held_t = torch.tensor(x_held_denorm, dtype=torch.float32).to(device)
    targets  = {"phase": heldout_phase_idx, "grip": heldout_grip_id, "hand": heldout_hand_id}
    with torch.no_grad():
        preds_gen = {
            "phase": transformer_model.head_phase(x_gen_t).argmax(1).cpu().numpy(),
            "grip":  transformer_model.head_grip(x_gen_t).argmax(1).cpu().numpy(),
            "hand":  transformer_model.head_hand(x_gen_t).argmax(1).cpu().numpy(),
        }
        preds_real = {
            "phase": transformer_model.head_phase(x_held_t).argmax(1).cpu().numpy(),
            "grip":  transformer_model.head_grip(x_held_t).argmax(1).cpu().numpy(),
            "hand":  transformer_model.head_hand(x_held_t).argmax(1).cpu().numpy(),
        }
    acc_gen  = {f: float((preds_gen[f]  == targets[f]).mean()) for f in ("phase", "grip", "hand")}
    acc_real = {f: float((preds_real[f] == targets[f]).mean()) for f in ("phase", "grip", "hand")}
    phase_dist_gen = [float((preds_gen["phase"] == ph_i).mean()) for ph_i in range(len(PHASE_NAMES))]

    # ── Section 5: Distributional quality ─────────────────────────────────────
    print("\n── Section 5: Distributional quality ─────────────────────────────────────────")
    swd_gen  = sliced_wasserstein(x_gen_norm, held_norm, args.n_projections, args.seed)
    half     = len(held_norm) // 2
    swd_base = sliced_wasserstein(held_norm[:half], held_norm[half:], args.n_projections, args.seed) if half > 1 else float("nan")
    swd_ratio = float(swd_gen / swd_base) if not np.isnan(swd_base) and swd_base > 0 else float("nan")

    nearest_mask = (
        (seen["y_phase"] == nearest_seen_key[0])
        & (seen["y_grip"]  == nearest_seen_key[1])
        & (seen["y_hand"]  == nearest_seen_key[2])
    )
    nearest_norm = seen_norm[nearest_mask]
    swd_control  = sliced_wasserstein(x_gen_norm, nearest_norm, args.n_projections, args.seed) if len(nearest_norm) > 1 else float("nan")

    mmd_gen  = float(compute_mmd(x_gen_norm, held_norm))
    mmd_base_val = float(compute_mmd(held_norm[:half], held_norm[half:])) if half > 1 else float("nan")
    mmd_ratio    = float(mmd_gen / mmd_base_val) if not np.isnan(mmd_base_val) and mmd_base_val > 0 else float("nan")

    ks_results     = [kstest(x_gen_norm[:, j], held_norm[:, j]) for j in range(x_gen_norm.shape[1])]
    ks_pvalues     = np.array([r.pvalue for r in ks_results])
    per_dim_ks_raw = float((ks_pvalues > 0.05).mean())
    per_dim_ks_fdr = None
    try:
        from statsmodels.stats.multitest import multipletests
        _, pv_corr, _, _ = multipletests(ks_pvalues, method="fdr_bh")
        per_dim_ks_fdr = float((pv_corr > 0.05).mean())
    except ImportError:
        print("  (install statsmodels for FDR correction)")

    gen_gaussian_pct = float(
        np.mean([kstest(x_gen_norm[:, j], "norm").pvalue > 0.05 for j in range(x_gen_norm.shape[1])])
    )

    print(f"\nDistributional quality (cVAE-normalized space):")
    print(f"  Sliced Wasserstein gen→target:  {swd_gen:.4f}")
    print(f"  Sliced Wasserstein baseline:    {swd_base:.4f}")
    print(f"  SW ratio (1.0=excellent):       {swd_ratio:.4f}")
    print(f"  SW control (vs nearest cluster):{swd_control:.4f}")
    print(f"  MMD ratio (kept for compat.):   {mmd_ratio:.4f}")
    print(f"  Per-dim KS not different (raw): {per_dim_ks_raw*100:.1f}%")
    if per_dim_ks_fdr is not None:
        print(f"  Per-dim KS not different (FDR): {per_dim_ks_fdr*100:.1f}%")
    print(f"  Generated distribution Gaussian: {gen_gaussian_pct*100:.1f}% dims")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        _make_plots(
            args, out_dir, pca_full, ev,
            seen_norm, seen, held_norm, held,
            x_gen_norm, mu_all, sigma_all, ks_pvalues,
            seen_centroids, held_centroid, gen_centroid, seen_cents,
            dists_held_to_seen, seen_keys, centroid_dist_gen_target,
            analogy_pred, phase_pred,
            preds_gen, preds_real, targets,
            heldout_phase_idx, heldout_grip_id, heldout_hand_id,
        )

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "heldout":     f"{args.heldout_phase}_{args.heldout_grip}_{args.heldout_hand}",
        "n_generated": args.n_generate,
        "embedding_quality": {
            "pca_variance_pc1_pc2":        pca_pc1pc2,
            "pca_variance_pc1_pc2_pc3":    pca_pc1pc2pc3,
            "latent_mean_across_dims":     latent_mean,
            "latent_std_across_dims":      latent_std,
            "posterior_std_mean":          posterior_std_mean,
            "latent_gaussian_pct_dims_ks": ks_pct_latent,
            "note": "latent gaussianity computed on seen test set, not training set",
        },
        "geometric": {
            "centroid_distance_gen_target":             centroid_dist_gen_target,
            "centroid_distance_target_nearest":         nearest_seen_dist,
            "centroid_distance_ratio":                  centroid_dist_ratio,
            "analogical_prediction_error_generated":    analogical_error_gen,
            "analogical_prediction_error_real_heldout": analogical_error_real,
            "compositionality_score":                   compositionality_score,
            "phase_transfer_prediction_error_generated":    phase_error_gen,
            "phase_transfer_prediction_error_real_heldout": phase_error_real,
        },
        "classifier_probe": {
            "generated":            acc_gen,
            "real_heldout_ceiling": acc_real,
            "ratio_to_ceiling": {
                f: float(acc_gen[f] / acc_real[f]) if acc_real[f] > 0 else float("nan")
                for f in ("grip", "hand")
            },
            "phase_distribution_generated": phase_dist_gen,
        },
        "distributional": {
            "sliced_wasserstein_gen_target": swd_gen,
            "sliced_wasserstein_baseline":   swd_base,
            "sliced_wasserstein_ratio":      swd_ratio,
            "sliced_wasserstein_control":    swd_control,
            "mmd_gen_target":  mmd_gen,
            "mmd_baseline":    mmd_base_val,
            "mmd_ratio":       mmd_ratio,
            "per_dim_ks_pct_raw":            per_dim_ks_raw,
            "per_dim_ks_pct_fdr_corrected":  per_dim_ks_fdr,
            "generated_gaussian_pct_dims":   gen_gaussian_pct,
        },
    }
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )

    print(f"\n{'='*80}")
    print(f"  GENERATION EVALUATION SUMMARY")
    print(f"Held-out: {args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}")
    print(f"\nEmbedding space:")
    print(f"  PCA PC1+PC2 variance:          {pca_pc1pc2:.3f}")
    print(f"  Latent Gaussianity (test set): {ks_pct_latent*100:.1f}% dims non-rejected (KS)")
    print(f"  Posterior std mean:            {posterior_std_mean:.3f}")
    print(f"\nGeometric quality:")
    print(f"  Centroid distance ratio:       {centroid_dist_ratio:.3f}  (< 1.0 = correct region)")
    if not np.isnan(compositionality_score):
        print(f"  Compositionality score:        {compositionality_score:.3f}  (< 1.0 = confirmed)")
    print(f"\nClassifier probe:")
    for f in ("phase", "grip", "hand"):
        print(f"  {f:<5}: gen={acc_gen[f]:.3f}  ceiling={acc_real[f]:.3f}")
    print(f"  Phase dist: {[f'{v:.2f}' for v in phase_dist_gen]}")
    print(f"\nDistributional quality:")
    print(f"  Sliced Wasserstein ratio:      {swd_ratio:.3f}  (~1.0 = excellent, >3.0 = poor)")
    print(f"  MMD ratio (compat.):           {mmd_ratio:.3f}")
    ks_str = f"{per_dim_ks_raw*100:.1f}%"
    if per_dim_ks_fdr is not None:
        ks_str += f"  ({per_dim_ks_fdr*100:.1f}% FDR)"
    print(f"  Per-dim KS not different:      {ks_str}")
    print(f"\nAll outputs saved to {out_dir}")

    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(
    args, out_dir, pca_full, ev,
    seen_norm, seen, held_norm, held,
    x_gen_norm, mu_all, sigma_all, ks_pvalues,
    seen_centroids, held_centroid, gen_centroid, seen_cents,
    dists_held_to_seen, seen_keys, centroid_dist_gen_target,
    analogy_pred, phase_pred,
    preds_gen, preds_real, targets,
    heldout_phase_idx, heldout_grip_id, heldout_hand_id,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    cmap10  = plt.get_cmap("tab10")
    cmap20  = plt.get_cmap("tab20")
    C_BLUE  = "#4C72B0"
    C_ORA   = "#DD8452"
    C_RED   = "#C44E52"

    # ── PCA explained variance ───────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        x_pc = np.arange(1, len(ev) + 1)
        ax.bar(x_pc, ev * 100, color=C_BLUE, alpha=0.7, label="per-component")
        ax.plot(x_pc, np.cumsum(ev) * 100, color=C_ORA, marker="o", label="cumulative")
        ax.axhline(50, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Variance explained (%)")
        ax.set_title("PCA explained variance (seen test embeddings)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "pca_explained_variance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved pca_explained_variance.png")
    except Exception as e:
        print(f"  WARNING: pca_explained_variance plot failed: {e}")

    # ── Latent Gaussianity ───────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(mu_all.mean(axis=0), bins=20, color=C_BLUE, edgecolor="white")
        axes[0].set_xlabel("Per-dim posterior mean")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Aggregate posterior means (target: 0)")
        axes[1].hist(sigma_all.mean(axis=0),
                     bins=20, color=C_ORA, edgecolor="white")
        axes[1].set_xlabel("Per-dim posterior std")
        axes[1].set_title("Aggregate posterior stds (target: 1)")
        plt.tight_layout()
        plt.savefig(out_dir / "latent_gaussianity.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 3))
        dims = np.arange(mu_all.shape[1])
        ks_pvs = np.array([kstest(mu_all[:, j], "norm").pvalue for j in dims])
        colors = [C_BLUE if p > 0.05 else C_RED for p in ks_pvs]
        ax.bar(dims, ks_pvs, color=colors)
        ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8, label="p=0.05")
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel("KS p-value vs N(0,1)")
        ax.set_title("Latent Gaussianity: KS p-values per dim (blue = not rejected)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "latent_dim_ks_pvalues.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved latent_gaussianity.png, latent_dim_ks_pvalues.png")
    except Exception as e:
        print(f"  WARNING: latent gaussianity plots failed: {e}")

    # ── Embedding scatter plots (PCA) ────────────────────────────────────────
    pca2 = PCA(n_components=2, random_state=0)
    all_for_pca = np.concatenate([seen_norm, held_norm, x_gen_norm], axis=0)
    pca2.fit(all_for_pca)
    seen_pc  = pca2.transform(seen_norm)
    held_pc  = pca2.transform(held_norm)
    gen_pc   = pca2.transform(x_gen_norm)

    umap_ok = False
    if not args.no_umap:
        try:
            import umap
            reducer   = umap.UMAP(n_components=2, random_state=args.seed)
            seen_um   = reducer.fit_transform(seen_norm)
            held_um   = reducer.transform(held_norm)
            gen_um    = reducer.transform(x_gen_norm)
            umap_ok   = True
            print("umap-learn found — UMAP scatter plots will be generated.")
        except Exception:
            print("WARNING: umap-learn not installed. PCA-only scatter plots.")

    scatter_configs = [
        ("Embedding by phase",        "phase",   seen["y_phase"], "embedding_by_phase"),
        ("Embedding by grip×hand",     "griphand",
         seen["y_grip"] * 2 + seen["y_hand"],       "embedding_by_griphand"),
        ("Embedding by full combination", "combo",
         seen["y_phase"] * 4 + seen["y_grip"] * 2 + seen["y_hand"], "embedding_by_full_combo"),
        ("Embedding by grip",          "grip",    seen["y_grip"],  "embedding_by_grip"),
        ("Embedding by hand",          "hand",    seen["y_hand"],  "embedding_by_hand"),
    ]

    for title, _, labels, fname in scatter_configs:
        try:
            for suffix, coords_s, coords_h, coords_g, xl, yl in [
                ("_pca", seen_pc, held_pc, gen_pc, "PC1", "PC2"),
            ] + ([
                ("_umap", seen_um, held_um, gen_um, "UMAP1", "UMAP2"),
            ] if umap_ok else []):
                unique_l = np.unique(labels)
                fig, ax = plt.subplots(figsize=(6, 5))
                for i, lv in enumerate(unique_l):
                    m = labels == lv
                    ax.scatter(coords_s[m, 0], coords_s[m, 1],
                               c=[cmap20(i % 20)], alpha=0.3, s=8, label=str(lv))
                ax.scatter(coords_h[:, 0], coords_h[:, 1],
                           c="black", alpha=0.5, s=12, marker="^", label="held-out (real)")
                ax.scatter(coords_g[:, 0], coords_g[:, 1],
                           c=C_ORA, alpha=0.4, s=8, marker="x")
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.set_title(title + (" (UMAP)" if suffix == "_umap" else ""))
                ax.legend(fontsize=6, ncol=2)
                plt.tight_layout()
                plt.savefig(out_dir / f"{fname}{suffix}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            print(f"  WARNING: scatter plot {fname} failed: {e}")

    # ── Centroid distance matrix ─────────────────────────────────────────────
    try:
        all_keys  = sorted(seen_centroids.keys()) + [(
            heldout_phase_idx, heldout_grip_id, heldout_hand_id
        )]
        all_cents = np.stack([
            seen_centroids.get(k, held_centroid) for k in all_keys
        ])
        dist_mat  = np.linalg.norm(
            all_cents[:, None, :] - all_cents[None, :, :], axis=2
        )
        lbls = [
            f"{PHASE_NAMES[k[0]]}+{ID_TO_GRIP[k[1]]}+{ID_TO_HAND[k[2]]}"
            for k in all_keys
        ]
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(dist_mat, cmap="viridis")
        ax.set_xticks(range(len(lbls)))
        ax.set_yticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(lbls, fontsize=7)
        plt.colorbar(im, ax=ax, label="L2 distance (normalized)")
        ax.set_title("Centroid distance matrix (normalized space)")
        plt.tight_layout()
        plt.savefig(out_dir / "centroid_distance_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  WARNING: centroid_distance_matrix plot failed: {e}")

    # ── Compositional prediction ─────────────────────────────────────────────
    try:
        for suffix, coords_s, coords_h, coords_g, xl, yl in [
            ("_pca", seen_pc, held_pc, gen_pc, "PC1", "PC2"),
        ] + ([("_umap", seen_um, held_um, gen_um, "UMAP1", "UMAP2")] if umap_ok else []):
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(coords_s[:, 0], coords_s[:, 1],
                       c="grey", alpha=0.15, s=6, label="seen")
            ax.scatter(coords_h[:, 0], coords_h[:, 1],
                       c=C_BLUE, alpha=0.5, s=12, label="real held-out")
            ax.scatter(coords_g[:, 0], coords_g[:, 1],
                       c=C_ORA, alpha=0.4, s=8, label="generated")
            if analogy_pred is not None:
                ap_pc = pca2.transform(analogy_pred[None, :])
                ax.scatter(ap_pc[0, 0], ap_pc[0, 1],
                           c=C_RED, s=120, marker="*", zorder=5, label="grip/hand analogy")
            if phase_pred is not None:
                pp_pc = pca2.transform(phase_pred[None, :])
                ax.scatter(pp_pc[0, 0], pp_pc[0, 1],
                           c="purple", s=120, marker="P", zorder=5, label="phase transfer")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"Compositionality — {args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"compositional_prediction{suffix}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"  WARNING: compositional_prediction plot failed: {e}")

    # ── Target centroid distances bar chart ──────────────────────────────────
    try:
        dists_gen_to_seen = np.linalg.norm(seen_cents - gen_centroid[None, :], axis=1)
        rows = sorted(
            [(seen_keys[i], float(dists_held_to_seen[i]), float(dists_gen_to_seen[i]))
             for i in range(len(seen_keys))],
            key=lambda r: r[1],
        )
        row_labels = [
            f"{PHASE_NAMES[k[0]]}+{ID_TO_GRIP[k[1]]}+{ID_TO_HAND[k[2]]}"
            for k, _, _ in rows
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = np.arange(len(rows))
        w = 0.35
        ax.bar(x_pos - w/2, [r[1] for r in rows], w, label="real held-out → centroid", color=C_BLUE)
        ax.bar(x_pos + w/2, [r[2] for r in rows], w, label="gen → centroid",           color=C_ORA)
        ax.axhline(centroid_dist_gen_target, color=C_RED, linestyle="--",
                   linewidth=1.5, label=f"gen→target = {centroid_dist_gen_target:.2f}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(row_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("L2 distance (normalized)")
        ax.set_title("Distance to target held-out centroid")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "embedding_target_centroid_distances.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  WARNING: target centroid distances plot failed: {e}")

    # ── Classifier probe ──────────────────────────────────────────────────────
    try:
        factors      = ["phase", "grip", "hand"]
        cls_names    = {
            "phase": PHASE_NAMES,
            "grip":  ["power", "precision"],
            "hand":  ["left",  "right"],
        }
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, factor in zip(axes, factors):
            n_cls   = len(cls_names[factor])
            x_cls   = np.arange(n_cls)
            w       = 0.35
            dist_g  = [(preds_gen[factor]  == c).mean() for c in range(n_cls)]
            dist_r  = [(preds_real[factor] == c).mean() for c in range(n_cls)]
            ax.bar(x_cls - w/2, dist_g, w, label="generated",    color=C_ORA, alpha=0.85)
            ax.bar(x_cls + w/2, dist_r, w, label="real held-out", color=C_BLUE, alpha=0.85)
            ax.axvline(targets[factor], color="black", linewidth=1.5, linestyle="--",
                       label=f"true: {cls_names[factor][targets[factor]]}")
            ax.set_xticks(x_cls)
            ax.set_xticklabels(cls_names[factor])
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Fraction of samples")
            ax.set_title(f"{factor} predicted class distribution")
            ax.legend(fontsize=8)
        plt.suptitle(
            f"Classifier probe: {args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(out_dir / "classifier_probe.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        for factor in factors:
            n_cls   = len(cls_names[factor])
            fig, ax = plt.subplots(figsize=(6, 4))
            dist_g  = [(preds_gen[factor] == c).mean() for c in range(n_cls)]
            ax.bar(range(n_cls), dist_g, color=C_ORA)
            ax.axvline(targets[factor], color="black", linestyle="--", linewidth=1.5)
            ax.set_xticks(range(n_cls))
            ax.set_xticklabels(cls_names[factor])
            ax.set_ylabel("Fraction")
            ax.set_title(f"{factor} predicted class distribution (generated, n={len(preds_gen[factor])})")
            plt.tight_layout()
            plt.savefig(out_dir / f"{factor}_class_distribution_generated.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

            n = len(preds_gen[factor])
            conf = np.zeros((n_cls, n_cls), dtype=int)
            for pred in preds_gen[factor]:
                conf[targets[factor], pred] += 1
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(conf, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{factor} confusion matrix (generated, n={n})")
            ax.set_xticks(range(n_cls))
            ax.set_xticklabels(cls_names[factor], fontsize=8)
            ax.set_yticks(range(n_cls))
            ax.set_yticklabels(cls_names[factor], fontsize=8)
            for ii in range(n_cls):
                for jj in range(n_cls):
                    ax.text(jj, ii, str(conf[ii, jj]), ha="center", va="center", fontsize=10)
            plt.tight_layout()
            plt.savefig(out_dir / f"{factor}_confusion_generated.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"  Saved classifier_probe.png, *_class_distribution_generated.png, *_confusion_generated.png")
    except Exception as e:
        print(f"  WARNING: classifier probe plots failed: {e}")

    # ── Per-dim KS + marginal distributions ──────────────────────────────────
    try:
        dims    = np.arange(x_gen_norm.shape[1])
        colors  = [C_BLUE if p > 0.05 else C_RED for p in ks_pvalues]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(dims, ks_pvalues, color=colors)
        ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8, label="p=0.05")
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel("KS p-value")
        ax.set_title("Per-dim KS: generated vs real held-out (blue = not different)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "per_dim_ks_pvalues.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        n_show = min(8, x_gen_norm.shape[1])
        fig, axes = plt.subplots(2, n_show // 2, figsize=(14, 5))
        axes = axes.flatten()
        for j in range(n_show):
            ax = axes[j]
            ax.hist(held_norm[:, j],  bins=20, density=True, alpha=0.5,
                    color=C_BLUE, label="real held-out (norm.)")
            ax.hist(x_gen_norm[:, j], bins=20, density=True, alpha=0.5,
                    color=C_ORA,  label="generated (norm.)")
            ax.set_title(f"dim {j}", fontsize=8)
            if j == 0:
                ax.legend(fontsize=6)
        plt.suptitle("Marginal distributions (cVAE-normalized space)", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / "marginal_distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved per_dim_ks_pvalues.png, marginal_distributions.png")
    except Exception as e:
        print(f"  WARNING: distributional plots failed: {e}")


if __name__ == "__main__":
    main()
