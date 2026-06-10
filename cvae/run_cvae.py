"""Step 1 — Conditional VAE on spectral amplitude.

Trains a cVAE conditioned on (phase, grip, hand) to generate per-channel MU
spectral amplitude features (256-dim). Holds out grasp+precision+right across
all angle variants and tests compositional generation.

Usage:
    python run_cvae.py --data_dir /path/to/mua_files
    python run_cvae.py --data_dir /path/to/mua_files --dry_run  # 5 epochs
    python run_cvae.py --data_dir /path/to/mua_files \\
        --classifier_checkpoint results/specialist_grasp_per_channel/checkpoint.pt
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset, TensorDataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
from transformer_encoder.data import AREA_SLICES, MAX_AREA_CHANNELS, N_AREAS, PHASE_NAMES, GRIP_TO_ID, HAND_TO_ID
from transformer_encoder.specialist_data import compute_specialist_norm_stats, LFPSpecialistDataset
from transformer_encoder.specialist_model import LFPSpecialistTransformer

# Local imports
from cvae.cvae_data import (
    load_cvae_dataset, split_cvae_dataset, compute_cvae_norm_stats,
    LFPCVAEDataset, make_condition_vector, spectral_to_specialist_input,
    N_REAL_CHANNELS, N_TIMEPOINTS, CONDITION_DIM,
    ID_TO_GRIP, ID_TO_HAND, ID_TO_PHASE,
)
from cvae.cvae_model import LFPCVAE, cvae_loss


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 1: spectral amplitude cVAE for LFP compositional generation."
    )
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--broadband_data_dir", type=str, default=None,
                   help="Raw waveform directory for --input_mode raw. If omitted, raw mode uses --data_dir.")
    p.add_argument("--heldout_phase", choices=PHASE_NAMES, default="grasp")
    p.add_argument("--heldout_grip",  choices=["power", "precision"], default="precision")
    p.add_argument("--heldout_hand",  choices=["left", "right"],      default="right")
    p.add_argument("--input_mode",  choices=["spectral", "raw"], default="spectral",
                   help="Feature mode: 'spectral' (default) or 'raw' waveform.")
    p.add_argument("--latent_dim",  type=int, default=32)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--beta_max",    type=float, default=1.0)
    p.add_argument("--beta_anneal_epochs", type=int, default=10)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--patience",    type=int, default=15)
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out_dir",     type=str, default=None)
    p.add_argument("--classifier_checkpoint", type=str, default=None,
                   help="Specialist checkpoint for classifier probe (optional).")
    p.add_argument("--device",      choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--no_plot",     action="store_true")
    p.add_argument("--dry_run",     action="store_true", help="5 epochs only.")
    # Model arch for loading specialist checkpoint
    p.add_argument("--spec_d_model",        type=int, default=64)
    p.add_argument("--spec_n_heads",        type=int, default=4)
    p.add_argument("--spec_n_layers",       type=int, default=2)
    p.add_argument("--spec_feedforward_dim",type=int, default=128)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

# Condition vector dim ranges — must match cvae_data.py layout exactly.
_COND_PHASE_DIMS = slice(0, 3)   # dims 0-2
_COND_GRIP_DIMS  = slice(3, 5)   # dims 3-4
_COND_HAND_DIMS  = slice(5, 7)   # dims 5-6


def apply_condition_dropout(
    c: torch.Tensor,
    p_single: float = 0.15,
    p_double: float = 0.04,
    p_all:    float = 0.03,
) -> torch.Tensor:
    """Per-sample partial condition masking for decoder input.

    Encoder always receives the original unmasked c. This function produces
    only the decoder copy c_dec.
    """
    c_dec = c.clone()
    batch_size = c.shape[0]

    p_none = 1.0 - 3*p_single - 3*p_double - p_all
    if p_none < 0:
        raise ValueError(
            f"Condition dropout probabilities sum to {1-p_none:.3f} which exceeds 1.0. "
            f"Reduce p_single, p_double, or p_all."
        )
    probs = torch.tensor([
        p_none,               # 0: keep all
        p_single,             # 1: drop phase
        p_single,             # 2: drop grip
        p_single,             # 3: drop hand
        p_double,             # 4: drop phase+grip
        p_double,             # 5: drop phase+hand
        p_double,             # 6: drop grip+hand
        p_all,                # 7: drop all
    ], dtype=torch.float32, device=c.device)

    cases = torch.multinomial(probs, num_samples=batch_size, replacement=True)

    for i in range(batch_size):
        case = cases[i].item()
        if case == 0:
            pass
        elif case == 1:
            c_dec[i, _COND_PHASE_DIMS] = 0.0
        elif case == 2:
            c_dec[i, _COND_GRIP_DIMS]  = 0.0
        elif case == 3:
            c_dec[i, _COND_HAND_DIMS]  = 0.0
        elif case == 4:
            c_dec[i, _COND_PHASE_DIMS] = 0.0
            c_dec[i, _COND_GRIP_DIMS]  = 0.0
        elif case == 5:
            c_dec[i, _COND_PHASE_DIMS] = 0.0
            c_dec[i, _COND_HAND_DIMS]  = 0.0
        elif case == 6:
            c_dec[i, _COND_GRIP_DIMS]  = 0.0
            c_dec[i, _COND_HAND_DIMS]  = 0.0
        elif case == 7:
            c_dec[i, :]                = 0.0

    return c_dec


def augment_embedding(
    x_clean: torch.Tensor,
    noise_scale: float = 0.1,
    amplitude_scale_range: tuple[float, float] = (0.85, 1.15),
    n_dropout_dims: int = 2,
) -> torch.Tensor:
    """Denoising augmentation for dense embedding vectors.

    Operates per sample independently. True no-op when noise_scale=0.0,
    amplitude_scale_range=(1.0, 1.0), and n_dropout_dims=0.
    """
    x_aug = x_clean.clone()
    batch_size, embed_dim = x_aug.shape
    for i in range(batch_size):
        if noise_scale != 0.0:
            std_i = x_aug[i].std(unbiased=False)
            x_aug[i] = x_aug[i] + torch.randn_like(x_aug[i]) * noise_scale * std_i
        if amplitude_scale_range != (1.0, 1.0):
            lo, hi = amplitude_scale_range
            s = lo + (hi - lo) * torch.rand(1, device=x_aug.device).item()
            x_aug[i] = x_aug[i] * s
        if n_dropout_dims > 0:
            drop_idx = torch.randperm(embed_dim, device=x_aug.device)[:n_dropout_dims]
            x_aug[i][drop_idx] = 0.0
    return x_aug


# Fixed multi-scale RBF bandwidths for MMD. Covers the expected pairwise distance
# range in 64-D latent space (mean inter-sample dist ≈ √(2·64) ≈ 11.3).
_MMD_BANDWIDTHS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]


def mmd_loss_torch(
    z_enc:   torch.Tensor,
    z_prior: torch.Tensor,
) -> torch.Tensor:
    """Differentiable multi-kernel MMD between z_enc and z_prior.

    Sums RBF kernels over fixed bandwidths — more stable than a single
    median-heuristic bandwidth when z_enc is collapsed early in training.
    z_enc   — (B, latent_dim) reparameterized encoder samples
    z_prior — (B, latent_dim) samples from N(0,I), same shape
    Returns scalar, differentiable w.r.t. z_enc.
    """
    mmd_total = z_enc.new_zeros(1)
    for bw in _MMD_BANDWIDTHS:
        gamma = 1.0 / (2.0 * bw ** 2)
        XX = torch.exp(-gamma * torch.cdist(z_enc,   z_enc  ).pow(2)).mean()
        YY = torch.exp(-gamma * torch.cdist(z_prior, z_prior).pow(2)).mean()
        XY = torch.exp(-gamma * torch.cdist(z_enc,   z_prior).pow(2)).mean()
        mmd_total = mmd_total + (XX + YY - 2.0 * XY)
    return mmd_total


def run_epoch(
    model: LFPCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    beta: float,
    device: torch.device,
    use_aug: bool = False,
    noise_scale: float = 0.1,
    amplitude_scale_range: tuple[float, float] = (0.85, 1.15),
    n_dropout_dims: int = 2,
    use_cond_dropout: bool = False,
    p_cond_single: float = 0.15,
    p_cond_double: float = 0.04,
    p_cond_all:    float = 0.03,
    free_bits:     float = 0.0,
    use_mmd_loss:  bool  = False,
    lambda_mmd:    float = 10.0,
) -> tuple[float, float, float]:
    """Run one train or eval epoch. Returns (total_loss, recon_loss, third_metric).

    third_metric is KL in ELBO mode and MMD in MMD mode (training only).
    Val always uses the ELBO path — val third_metric is always ELBO-KL,
    which is comparable across all runs regardless of training loss mode.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = total_recon = total_kl = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch in loader:
            x, c = batch[0].to(device), batch[1].to(device)

            x_clean = x
            if training and use_aug:
                x_input = augment_embedding(x_clean, noise_scale, amplitude_scale_range, n_dropout_dims)
            else:
                x_input = x_clean

            if training and use_cond_dropout:
                c_dec = apply_condition_dropout(c, p_cond_single, p_cond_double, p_cond_all)
            else:
                c_dec = c

            if training and use_mmd_loss:
                # MMD-VAE path: explicit reparameterize so z is accessible.
                # Always stochastic — model.reparameterize() returns mu at eval time,
                # but this branch only fires during training.
                mu, log_var = model.encode(x_input, c)
                std     = torch.exp(0.5 * log_var)
                z       = mu + std * torch.randn_like(std)
                x_recon = model.decode(z, c_dec)
                z_prior = torch.randn_like(z)
                recon   = nn.functional.mse_loss(x_recon, x_clean, reduction="mean")
                mmd     = mmd_loss_torch(z, z_prior)
                loss    = recon + lambda_mmd * mmd
                kl      = mmd  # stored in kl slot; logged as MMD in MMD mode
            else:
                # ELBO path — also used for all val passes in MMD mode.
                # Val always logs ELBO-KL for comparability across runs.
                x_recon, mu, log_var = model(x_input, c, c_dec=c_dec)
                loss, recon, kl = cvae_loss(x_recon, x_clean, mu, log_var, beta, free_bits=free_bits)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n = x.shape[0]
            total_loss  += loss.item()  * n
            total_recon += recon.item() * n
            total_kl    += kl.item()    * n

    n_samples = len(loader.dataset)
    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples


def train_cvae(
    model: LFPCVAE,
    train_ds: LFPCVAEDataset,
    val_ds:   LFPCVAEDataset,
    args:     argparse.Namespace,
    device:   torch.device,
    save_path: str | None = None,
    use_aug: bool = False,
    noise_scale: float = 0.1,
    amplitude_scale_range: tuple[float, float] = (0.85, 1.15),
    n_dropout_dims: int = 2,
    use_cond_dropout: bool = False,
    p_cond_single: float = 0.15,
    p_cond_double: float = 0.04,
    p_cond_all:    float = 0.03,
    free_bits:     float = 0.0,
    use_mmd_loss:  bool  = False,
    lambda_mmd:    float = 10.0,
) -> dict:
    """Train cVAE with KL annealing and optional early stopping.

    Augmentation and loss kwargs are forwarded to run_epoch() for training only.
    Val always uses the ELBO path (use_mmd_loss=False) for comparable logging.
    Existing callers that pass no new kwargs get the no-op defaults and are
    completely unaffected.

    Returns history dict with per-epoch losses.
    """
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = {
        "train_loss": [], "train_recon": [], "train_kl": [],
        "val_loss":   [], "val_recon":   [], "val_kl":   [],
    }
    best_val_loss  = float("inf")
    best_score     = float("inf")
    best_state     = None
    best_epoch     = 0
    best_val_recon = float("inf")
    best_val_kl    = float("inf")
    selection_metric = "val_recon" if use_mmd_loss else "val_loss"
    patience_ctr   = 0
    no_early_stop  = getattr(args, "no_early_stopping", False)
    n_epochs       = 5 if args.dry_run else args.epochs
    latent_dim     = getattr(args, "latent_dim", model.latent_dim)

    for epoch in range(n_epochs):
        # KL annealing: beta ramps from 0 to beta_max over beta_anneal_epochs
        beta = min(1.0, (epoch + 1) / max(args.beta_anneal_epochs, 1)) * args.beta_max

        tl, tr, tk = run_epoch(
            model, train_loader, optimizer, beta, device,
            use_aug=use_aug, noise_scale=noise_scale,
            amplitude_scale_range=amplitude_scale_range,
            n_dropout_dims=n_dropout_dims,
            use_cond_dropout=use_cond_dropout,
            p_cond_single=p_cond_single,
            p_cond_double=p_cond_double,
            p_cond_all=p_cond_all,
            free_bits=free_bits,
            use_mmd_loss=use_mmd_loss,
            lambda_mmd=lambda_mmd,
        )
        # Val always uses ELBO path — logs comparable KL regardless of training mode.
        vl, vr, vk = run_epoch(model, val_loader, None, beta, device, free_bits=free_bits)

        history["train_loss"].append(tl)
        history["train_recon"].append(tr)
        history["train_kl"].append(tk)
        history["val_loss"].append(vl)
        history["val_recon"].append(vr)
        history["val_kl"].append(vk)

        third_label = "mmd" if use_mmd_loss else "kl"
        print(
            f"Epoch {epoch+1:03d}/{n_epochs} | "
            f"recon={tr:.4f} {third_label}={tk:.4f} | "
            f"val_recon={vr:.4f} val_kl={vk:.4f}"
        )

        selection_score = vr if use_mmd_loss else vl
        if selection_score < best_score:
            best_score     = selection_score
            best_val_loss  = vl
            best_val_recon = vr
            best_val_kl    = vk
            best_epoch     = epoch
            best_state     = copy.deepcopy(model.state_dict())
            patience_ctr   = 0
        else:
            patience_ctr += 1
            if not no_early_stop and patience_ctr >= args.patience:
                print(
                    f"  Early stopping at epoch {epoch+1}  "
                    f"(best {selection_metric}={best_score:.4f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path:
        torch.save(
            {
                "model_state":       best_state or model.state_dict(),
                "history":           history,
                "args":              vars(args),
                "best_epoch":        best_epoch,
                "best_selection_metric": selection_metric,
                "best_selection_score":  best_score,
                "best_val_loss":     best_val_loss,
                "best_val_recon":    best_val_recon,
                "best_val_kl_mean":  best_val_kl,
                "best_val_kl_sum":   best_val_kl * latent_dim,
                "use_mmd_loss":      use_mmd_loss,
                "lambda_mmd":        lambda_mmd,
            },
            save_path,
        )
        print(f"  Saved checkpoint: {save_path}")

    return history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def reconstruct_seen(
    model: LFPCVAE,
    val_ds: LFPCVAEDataset,
    norm_stats: dict,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    """Phase 1: per-combination reconstruction quality on seen validation set.

    Returns dict mapping combo_label → {mse, pearsonr}.
    """
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    all_x, all_xr, all_grip, all_hand, all_phase = [], [], [], [], []

    with torch.no_grad():
        for x, c, yg, yh, ya, yp in loader:
            x, c    = x.to(device), c.to(device)
            xr, _, _ = model(x, c)
            all_x.append(x.cpu().numpy())
            all_xr.append(xr.cpu().numpy())
            all_grip.append(yg.numpy())
            all_hand.append(yh.numpy())
            all_phase.append(yp.numpy())

    x_all   = np.concatenate(all_x)
    xr_all  = np.concatenate(all_xr)
    g_all   = np.concatenate(all_grip)
    h_all   = np.concatenate(all_hand)
    p_all   = np.concatenate(all_phase)

    results: dict = {}
    for g in (0, 1):
        for h in (0, 1):
            for ph in range(3):
                mask  = (g_all == g) & (h_all == h) & (p_all == ph)
                if mask.sum() == 0:
                    continue
                label = f"{PHASE_NAMES[ph]} + {ID_TO_GRIP[g]} + {ID_TO_HAND[h]}"
                xi    = x_all[mask]
                xri   = xr_all[mask]
                mse   = float(np.mean((xi - xri) ** 2))
                # Per-channel Pearson r averaged across channels.
                rs = []
                if xi.shape[1] == N_REAL_CHANNELS * N_TIMEPOINTS:
                    xi_corr = xi.reshape(len(xi), N_REAL_CHANNELS, N_TIMEPOINTS)
                    xr_corr = xri.reshape(len(xri), N_REAL_CHANNELS, N_TIMEPOINTS)
                    for ch in range(N_REAL_CHANNELS):
                        a = xi_corr[:, ch, :].reshape(-1)
                        b = xr_corr[:, ch, :].reshape(-1)
                        if a.std() > 1e-8:
                            rs.append(float(np.corrcoef(a, b)[0, 1]))
                else:
                    for ch in range(xi.shape[1]):
                        if xi[:, ch].std() > 1e-8:
                            r = float(np.corrcoef(xi[:, ch], xri[:, ch])[0, 1])
                            rs.append(r)
                results[label] = {"mse": mse, "pearsonr": float(np.mean(rs)) if rs else float("nan")}

    # Print table
    data_var = float(np.var(x_all))
    print("\nReconstruction quality (seen validation set):")
    print(f"  {'Combination':<35} {'MSE':>8} {'r':>8}  {'OK?':>5}")
    print("  " + "-" * 60)
    for label, vals in sorted(results.items()):
        mse, r = vals["mse"], vals["pearsonr"]
        ok = "✓" if mse < data_var and r > 0.60 else "✗"
        print(f"  {label:<35} {mse:>8.4f} {r:>8.3f}  {ok:>5}")
    print(f"  Data variance: {data_var:.4f}  |  Target: MSE < data_var, r > 0.60")

    results["data_variance"] = data_var
    return results


def compute_mmd(X: np.ndarray, Y: np.ndarray, bandwidth: float | None = 1.0) -> float:
    """Maximum Mean Discrepancy with RBF kernel between two sample arrays.

    Pass bandwidth=None to use the median heuristic computed from X only
    (self-distances excluded). Existing callers that pass no bandwidth get
    the fixed 1.0 default and are unaffected.
    """
    from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
    if bandwidth is None:
        dists = euclidean_distances(X, X)
        np.fill_diagonal(dists, np.nan)
        bandwidth = float(np.nanmedian(dists))
        bandwidth = max(bandwidth, 1e-6)
    gamma = 1.0 / (2.0 * bandwidth ** 2)
    XX    = rbf_kernel(X, X, gamma=gamma).mean()
    YY    = rbf_kernel(Y, Y, gamma=gamma).mean()
    XY    = rbf_kernel(X, Y, gamma=gamma).mean()
    return float(XX + YY - 2.0 * XY)


def evaluate_generation(
    model: LFPCVAE,
    heldout_ds: LFPCVAEDataset,
    norm_stats: dict,
    condition: np.ndarray,     # (7,) one-hot condition for held-out combo
    device: torch.device,
    out_dir: Path,
    no_plot: bool,
    train_ds: LFPCVAEDataset,  # for PCA fitting
    n_generated: int = 500,
    batch_size: int = 128,
) -> dict:
    """Phase 2: distributional validation of generated held-out samples."""
    # Collect real held-out features
    loader    = DataLoader(heldout_ds, batch_size=batch_size, shuffle=False)
    x_real_l  = []
    with torch.no_grad():
        for x, *_ in loader:
            x_real_l.append(x.numpy())
    x_real = np.concatenate(x_real_l, axis=0)  # (N_real, 256)

    # Generate held-out samples
    c_tensor = torch.tensor(condition, dtype=torch.float32)
    x_gen    = model.generate(c_tensor, n_samples=n_generated, device=device).cpu().numpy()

    results: dict = {}

    # -- 1. Per-channel t-test --
    if x_real.shape[1] == N_REAL_CHANNELS * N_TIMEPOINTS:
        x_real_test = x_real.reshape(len(x_real), N_REAL_CHANNELS, N_TIMEPOINTS).mean(axis=2)
        x_gen_test  = x_gen.reshape(len(x_gen), N_REAL_CHANNELS, N_TIMEPOINTS).mean(axis=2)
    else:
        x_real_test = x_real
        x_gen_test  = x_gen
    n_ch     = x_real_test.shape[1]
    p_vals   = []
    for ch in range(n_ch):
        if x_real_test[:, ch].std() < 1e-8:
            p_vals.append(1.0)   # bad channel — trivially "same"
            continue
        _, p = ttest_ind(x_real_test[:, ch], x_gen_test[:, ch], equal_var=False)
        p_vals.append(float(p))
    frac_ns  = float(np.mean(np.array(p_vals) > 0.05))
    print(f"\n  Per-channel t-test: {frac_ns:.2%} channels p > 0.05  (target > 70%)")
    results["frac_channels_p_gt_005"] = frac_ns

    # -- 2. MMD --
    mmd_gen      = compute_mmd(x_gen, x_real)
    # Baseline MMD: two halves of real held-out
    half         = len(x_real) // 2
    mmd_baseline = compute_mmd(x_real[:half], x_real[half:]) if half > 1 else float("nan")
    ratio        = mmd_gen / max(mmd_baseline, 1e-10) if not np.isnan(mmd_baseline) else float("nan")
    print(f"  MMD generated/baseline ratio: {ratio:.3f}  (near 1.0 = excellent)")
    results["mmd_generated"] = mmd_gen
    results["mmd_baseline"]  = mmd_baseline
    results["mmd_ratio"]     = ratio

    # -- 3. PCA visualization --
    if not no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Fit PCA on all training data
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            x_train_l, g_train_l, h_train_l, p_train_l = [], [], [], []
            with torch.no_grad():
                for x, c, yg, yh, ya, yp in train_loader:
                    x_train_l.append(x.numpy())
                    g_train_l.append(yg.numpy())
                    h_train_l.append(yh.numpy())
                    p_train_l.append(yp.numpy())
            x_train = np.concatenate(x_train_l)
            g_train = np.concatenate(g_train_l)
            h_train = np.concatenate(h_train_l)
            p_train = np.concatenate(p_train_l)

            pca = PCA(n_components=3, random_state=42)
            pca.fit(x_train)

            tr_pc   = pca.transform(x_train)
            real_pc = pca.transform(x_real)
            gen_pc  = pca.transform(x_gen)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, xi, yi, xlabel, ylabel in [
                (axes[0], 0, 1, "PC1", "PC2"),
                (axes[1], 0, 2, "PC1", "PC3"),
            ]:
                ax.scatter(tr_pc[:, xi],   tr_pc[:, yi],   c="grey",   alpha=0.1, s=5, label="train")
                ax.scatter(real_pc[:, xi], real_pc[:, yi], c="#4C72B0",alpha=0.6, s=15, label="real held-out")
                ax.scatter(gen_pc[:, xi],  gen_pc[:, yi],  c="#DD8452",alpha=0.6, s=15, label="generated")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)
            fig.suptitle("PCA: real training vs real held-out vs generated", fontsize=10)
            plt.tight_layout()
            plt.savefig(out_dir / "pca_generation.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved pca_generation.png")

            # Held-out phase composition geometry: seen phase combos + real/generated held-out.
            heldout_phase = int(np.argmax(condition[:3]))
            heldout_phase_name = PHASE_NAMES[heldout_phase]
            if x_train.shape[1] == x_real.shape[1]:
                fig, ax = plt.subplots(figsize=(7, 5))
                phase_pc = pca.transform(x_train[p_train == heldout_phase])
                phase_g = g_train[p_train == heldout_phase]
                phase_h = h_train[p_train == heldout_phase]
                combo_styles = {
                    (0, 0): ("power+left", "#7F7F7F"),
                    (0, 1): ("power+right", "#9467BD"),
                    (1, 0): ("precision+left", "#2CA02C"),
                    (1, 1): ("precision+right train", "#8C564B"),
                }
                for (g, h), (label, color) in combo_styles.items():
                    mask = (phase_g == g) & (phase_h == h)
                    if mask.any():
                        ax.scatter(phase_pc[mask, 0], phase_pc[mask, 1],
                                   c=color, s=12, alpha=0.35, label=label)
                ax.scatter(real_pc[:, 0], real_pc[:, 1], c="#4C72B0", s=22, alpha=0.75,
                           label="real held-out")
                ax.scatter(gen_pc[:, 0], gen_pc[:, 1], c="#DD8452", s=22, alpha=0.75,
                           label="generated held-out")
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
                ax.set_title(f"{heldout_phase_name.capitalize()}-phase compositional geometry")
                ax.legend(fontsize=7)
                plt.tight_layout()
                out_name = f"pca_{heldout_phase_name}_composition.png"
                plt.savefig(out_dir / out_name, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved {out_name}")

        except Exception as e:
            print(f"  WARNING: PCA plot failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Classifier probe
# ---------------------------------------------------------------------------

def _load_specialist_for_probe(
    checkpoint_path: str,
    norm_stats_path: str | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[LFPSpecialistTransformer, dict | None]:
    """Load specialist checkpoint and its normalization stats."""
    # Try reading architecture from summary.json
    arch = {}
    summary_p = Path(checkpoint_path).parent / "summary.json"
    if summary_p.exists():
        try:
            arch = json.loads(summary_p.read_text()).get("model", {})
        except Exception:
            pass

    model = LFPSpecialistTransformer(
        use_per_channel  = True,
        input_dim        = MAX_AREA_CHANNELS,
        d_model          = arch.get("d_model",         args.spec_d_model),
        n_heads          = arch.get("n_heads",         args.spec_n_heads),
        n_layers         = arch.get("n_layers",        args.spec_n_layers),
        feedforward_dim  = arch.get("feedforward_dim", args.spec_feedforward_dim),
        dropout          = 0.0,
        n_bins           = 1,
        n_angle_classes  = 4,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.to(device).eval()

    # Load normalization stats from the checkpoint directory
    ns_path = Path(checkpoint_path).parent / "normalization_stats.npz"
    spec_norm = None
    if ns_path.exists():
        raw  = np.load(str(ns_path))
        spec_norm = {"mu": raw["mu"], "sigma": raw["sigma"]}
    else:
        print("  WARNING: normalization_stats.npz not found next to checkpoint. "
              "Probe may be inaccurate.")

    return model, spec_norm


def run_classifier_probe(
    model_cvae: LFPCVAE,
    norm_stats_cvae: dict,
    condition: np.ndarray,
    classifier_checkpoint: str,
    args: argparse.Namespace,
    device: torch.device,
    n_generated: int = 500,
    batch_size: int = 128,
) -> dict:
    """Validation 2: run generated samples through a specialist classifier.

    Reports only conditioned factors. Angle is deliberately omitted because the
    cVAE condition does not include angle and generated samples have no true
    angle label.
    """
    print("\n  Classifier probe on generated held-out (grasp+precision+right):")

    if model_cvae.input_dim != N_REAL_CHANNELS:
        print("    Skipping specialist probe for raw waveform output; probe expects 256 spectral features.")
        return {"skipped": True, "reason": "raw_waveform_output"}

    spec_model, spec_norm = _load_specialist_for_probe(
        classifier_checkpoint, None, args, device
    )
    if spec_norm is None:
        print("  WARNING: No specialist norm stats — skipping probe.")
        return {}

    # Generate samples
    c_tensor = torch.tensor(condition, dtype=torch.float32)
    x_gen    = model_cvae.generate(c_tensor, n_samples=n_generated, device=device)

    # Convert cVAE-normalized (B, 256) → specialist-normalized (B, n_areas, MAX_AREA_CH)
    mu_cvae    = torch.tensor(norm_stats_cvae["mu"],    dtype=torch.float32)
    sig_cvae   = torch.tensor(norm_stats_cvae["sigma"], dtype=torch.float32)
    mu_spec    = torch.tensor(spec_norm["mu"],    dtype=torch.float32)
    sig_spec   = torch.tensor(spec_norm["sigma"], dtype=torch.float32)

    x_spec = spectral_to_specialist_input(
        x_gen.cpu(), mu_cvae, sig_cvae, mu_spec, sig_spec
    ).to(device)   # (B, n_areas, MAX_AREA_CHANNELS)

    # Run through specialist
    results: dict = {}
    all_grip, all_hand = [], []
    spec_model.eval()
    with torch.no_grad():
        for i in range(0, len(x_spec), batch_size):
            batch  = x_spec[i:i+batch_size]
            lg, lh, _ = spec_model(batch)
            all_grip.append(lg.argmax(1).cpu().numpy())
            all_hand.append(lh.argmax(1).cpu().numpy())

    grip_pred  = np.concatenate(all_grip)
    hand_pred  = np.concatenate(all_hand)

    target_grip = int(np.argmax(condition[3:5]))
    target_hand = int(np.argmax(condition[5:7]))
    # Expected targets: grip/hand from condition.
    grip_acc  = float((grip_pred  == target_grip).mean())
    hand_acc  = float((hand_pred  == target_hand).mean())

    print(f"    grip accuracy:  {grip_acc:.4f}  (expected ~1.0)")
    print(f"    hand accuracy:  {hand_acc:.4f}  (expected ~0.85)")

    results = {
        "grip_accuracy":  grip_acc,
        "hand_accuracy":  hand_acc,
        "n_generated":    n_generated,
    }
    return results


# ---------------------------------------------------------------------------
# Augmentation validation (Validation 3)
# ---------------------------------------------------------------------------

def run_augmentation_validation(
    generated_x: np.ndarray,        # (n_gen, 256) z-scored features
    norm_stats_cvae: dict,
    train_ds: LFPCVAEDataset,
    heldout_ds: LFPCVAEDataset,
    condition: np.ndarray,
    classifier_checkpoint: str,
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    """Validation 3: retrain specialist with augmented held-out data.

    Generates grasp+precision+right samples, adds them to training data,
    retrains specialist, compares hand accuracy on real held-out.

    Returns {"baseline_hand_acc", "augmented_hand_acc", "delta"}.
    """
    from run_specialists import train_specialist, evaluate_specialist, HEAD_NAMES

    print("\n  Augmentation validation ...")
    if generated_x.shape[1] != N_REAL_CHANNELS:
        print("  Skipping augmentation for raw waveform output; specialist expects 256 spectral features.")
        return {"skipped": True, "reason": "raw_waveform_output"}

    spec_model_base, spec_norm = _load_specialist_for_probe(
        classifier_checkpoint, None, args, device
    )
    if spec_norm is None:
        print("  WARNING: No specialist norm stats — skipping augmentation.")
        return {}

    phase_idx = PHASE_NAMES.index("grasp")

    # Evaluate baseline (no augmentation) on real held-out
    mu_cvae  = torch.tensor(norm_stats_cvae["mu"],    dtype=torch.float32)
    sig_cvae = torch.tensor(norm_stats_cvae["sigma"], dtype=torch.float32)
    mu_spec  = torch.tensor(spec_norm["mu"],          dtype=torch.float32)
    sig_spec = torch.tensor(spec_norm["sigma"],       dtype=torch.float32)

    # Collect real held-out features in specialist format for evaluation only.
    loader   = DataLoader(heldout_ds, batch_size=batch_size, shuffle=False)
    real_x_l, real_g_l, real_h_l, real_a_l = [], [], [], []
    with torch.no_grad():
        for x, c, yg, yh, ya, yp in loader:
            xs = spectral_to_specialist_input(x, mu_cvae, sig_cvae, mu_spec, sig_spec)
            real_x_l.append(xs)
            real_g_l.append(yg); real_h_l.append(yh); real_a_l.append(ya)

    real_x = torch.cat(real_x_l)   # (N_real, n_areas, 96)
    real_g = torch.cat(real_g_l)
    real_h = torch.cat(real_h_l)
    real_a = torch.cat(real_a_l)

    # Baseline evaluation
    spec_ds_real = TensorDataset(real_x, real_g, real_h, real_a)
    spec_model_base.eval()
    baseline_preds = []
    with torch.no_grad():
        for x, *_ in DataLoader(spec_ds_real, batch_size=batch_size):
            _, lh, _ = spec_model_base(x.to(device))
            baseline_preds.append(lh.argmax(1).cpu())
    baseline_hand_acc = float((torch.cat(baseline_preds) == real_h).float().mean())

    # Collect real training features. This avoids leaking held-out trials into the
    # augmented training set.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    train_x_l, train_g_l, train_h_l, train_a_l = [], [], [], []
    with torch.no_grad():
        for x, c, yg, yh, ya, yp in train_loader:
            xs = spectral_to_specialist_input(x, mu_cvae, sig_cvae, mu_spec, sig_spec)
            train_x_l.append(xs)
            train_g_l.append(yg); train_h_l.append(yh); train_a_l.append(ya)
    train_x = torch.cat(train_x_l)
    train_g = torch.cat(train_g_l)
    train_h = torch.cat(train_h_l)
    train_a = torch.cat(train_a_l)

    # Convert generated features to specialist format
    gen_t   = torch.tensor(generated_x, dtype=torch.float32)
    gen_xs  = spectral_to_specialist_input(gen_t, mu_cvae, sig_cvae, mu_spec, sig_spec)
    # Labels for generated: grip/hand from held-out condition, angle unknown (use 0)
    gen_n   = len(gen_xs)
    gen_g   = torch.full((gen_n,), int(np.argmax(condition[3:5])), dtype=torch.long)
    gen_h   = torch.full((gen_n,), int(np.argmax(condition[5:7])), dtype=torch.long)
    gen_a   = torch.zeros(gen_n, dtype=torch.long)   # angle placeholder

    aug_x = torch.cat([train_x, gen_xs])
    aug_g = torch.cat([train_g, gen_g])
    aug_h = torch.cat([train_h, gen_h])
    aug_a = torch.cat([train_a, gen_a])

    aug_ds = TensorDataset(aug_x, aug_g, aug_h, aug_a)

    # Re-train a fresh specialist on augmented data
    train_config = {"batch_size": batch_size, "lr": 3e-4, "epochs": 20, "patience": 5}
    arch = {}
    summary_p = Path(classifier_checkpoint).parent / "summary.json"
    if summary_p.exists():
        try:
            arch = json.loads(summary_p.read_text()).get("model", {})
        except Exception:
            pass

    fresh_model = LFPSpecialistTransformer(
        use_per_channel=True, input_dim=MAX_AREA_CHANNELS,
        d_model=arch.get("d_model", args.spec_d_model),
        n_heads=arch.get("n_heads", args.spec_n_heads),
        n_layers=arch.get("n_layers", args.spec_n_layers),
        feedforward_dim=arch.get("feedforward_dim", args.spec_feedforward_dim),
        dropout=0.3, n_bins=1, n_angle_classes=4,
    )

    # Use 80% for train, 20% for val (simple split)
    n_aug   = len(aug_ds)
    val_n   = max(1, n_aug // 5)
    tr_idx  = list(range(n_aug - val_n))
    va_idx  = list(range(n_aug - val_n, n_aug))

    fresh_model, _, _ = train_specialist(
        fresh_model,
        Subset(aug_ds, tr_idx),
        Subset(aug_ds, va_idx),
        train_config, save_path=None, device=device, verbose=False,
    )

    # Evaluate augmented model on real held-out
    aug_preds = []
    fresh_model.eval()
    with torch.no_grad():
        for x, *_ in DataLoader(spec_ds_real, batch_size=batch_size):
            _, lh, _ = fresh_model(x.to(device))
            aug_preds.append(lh.argmax(1).cpu())
    aug_hand_acc = float((torch.cat(aug_preds) == real_h).float().mean())

    delta = aug_hand_acc - baseline_hand_acc
    print(f"    Baseline hand accuracy: {baseline_hand_acc:.4f}")
    print(f"    Augmented hand accuracy: {aug_hand_acc:.4f}  (delta={delta:+.4f})")

    return {
        "baseline_hand_acc":  baseline_hand_acc,
        "augmented_hand_acc": aug_hand_acc,
        "delta":              delta,
    }


# ---------------------------------------------------------------------------
# Latent space analysis
# ---------------------------------------------------------------------------

def analyze_latent_space(
    model: LFPCVAE,
    train_ds: LFPCVAEDataset,
    heldout_ds: LFPCVAEDataset,
    out_dir: Path,
    no_plot: bool,
    device: torch.device,
    batch_size: int = 128,
) -> None:
    """Encode all training + held-out trials, visualize latent space (PCA + UMAP)."""
    model.eval()

    def _encode_ds(ds):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        mus_l, g_l, h_l, p_l = [], [], [], []
        with torch.no_grad():
            for x, c, yg, yh, ya, yph in loader:
                mu, _ = model.encode(x.to(device), c.to(device))
                mus_l.append(mu.cpu().numpy())
                g_l.append(yg.numpy()); h_l.append(yh.numpy()); p_l.append(yph.numpy())
        return (np.concatenate(mus_l), np.concatenate(g_l),
                np.concatenate(h_l),   np.concatenate(p_l))

    z_train, g_tr, h_tr, p_tr = _encode_ds(train_ds)
    z_held,  g_ho, h_ho, p_ho = _encode_ds(heldout_ds)

    z_all   = np.concatenate([z_train, z_held])
    g_all   = np.concatenate([g_tr, g_ho])
    h_all   = np.concatenate([h_tr, h_ho])
    p_all   = np.concatenate([p_tr, p_ho])
    src_all = np.array(["train"] * len(z_train) + ["heldout"] * len(z_held))
    combo_all = g_all * 6 + h_all * 3 + p_all

    if no_plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # PCA
        pca    = PCA(n_components=2, random_state=42)
        z_pca  = pca.fit_transform(z_all)
        z_tr_pca = z_pca[:len(z_train)]
        z_ho_pca = z_pca[len(z_train):]

        combo_names = {
            g * 6 + h * 3 + p: f"{PHASE_NAMES[p]}+{ID_TO_GRIP[g]}+{ID_TO_HAND[h]}"
            for g in (0, 1) for h in (0, 1) for p in range(3)
        }
        cmap = plt.get_cmap("tab20")

        fig, ax = plt.subplots(figsize=(8, 6))
        for pos, combo in enumerate(sorted(np.unique(combo_all))):
            mask = combo_all == combo
            held_mask = mask & (src_all == "heldout")
            train_mask = mask & (src_all == "train")
            color = cmap(pos % 20)
            if train_mask.any():
                ax.scatter(z_pca[train_mask, 0], z_pca[train_mask, 1],
                           c=[color], alpha=0.35, s=8, label=combo_names.get(combo, str(combo)))
            if held_mask.any():
                ax.scatter(z_pca[held_mask, 0], z_pca[held_mask, 1],
                           c=[color], alpha=0.9, s=28, marker="x")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("Latent space PCA by phase+grip+hand")
        ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / "latent_space_pca.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # UMAP / fallback
        try:
            import umap
            reducer    = umap.UMAP(n_components=2, random_state=42)
            z_2d       = reducer.fit_transform(z_all)
            method_lbl = "UMAP"
        except ImportError:
            print("  umap-learn not installed — using PCA for latent space plot.")
            z_2d       = z_pca
            method_lbl = "PCA(fallback)"

        fig, ax  = plt.subplots(figsize=(8, 6))
        for pos, combo in enumerate(sorted(np.unique(combo_all))):
            mask = combo_all == combo
            held_mask = mask & (src_all == "heldout")
            train_mask = mask & (src_all == "train")
            color = cmap(pos % 20)
            if train_mask.any():
                ax.scatter(z_2d[train_mask, 0], z_2d[train_mask, 1],
                           c=[color], alpha=0.35, s=8, label=combo_names.get(combo, str(combo)))
            if held_mask.any():
                ax.scatter(z_2d[held_mask, 0], z_2d[held_mask, 1],
                           c=[color], alpha=0.9, s=28, marker="x")
        ax.set_xlabel("D1"); ax.set_ylabel("D2")
        ax.set_title(f"Latent space {method_lbl} by phase+grip+hand")
        ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / "latent_space_umap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved latent space plots to {out_dir}")
    except Exception as e:
        print(f"  WARNING: latent space plot failed: {e}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary_table(results: dict, mode: str) -> None:
    input_label = f"Spectral({N_REAL_CHANNELS})" if mode == "spectral" else "Raw(128k)"
    print(f"\n{'='*72}")
    print(f"  CVAE SUMMARY — {input_label}")
    print(f"{'='*72}")
    print(f"  Held-out: {results.get('heldout_label', 'N/A')}")
    print()
    print(f"  Reconstruction MSE (seen):  {results.get('recon_mse_seen', 'N/A'):.4f}")
    print(f"  Reconstruction r   (seen):  {results.get('recon_r_seen', 'N/A'):.4f}")
    print()
    gen = results.get("generation", {})
    print(f"  Fraction channels p>0.05:   {gen.get('frac_channels_p_gt_005', float('nan')):.3f}")
    print(f"  MMD ratio (gen/baseline):   {gen.get('mmd_ratio', float('nan')):.3f}")
    probe = results.get("probe", {})
    if probe:
        print(f"\n  Classifier probe (generated held-out):")
        print(f"    grip accuracy:  {probe.get('grip_accuracy', float('nan')):.4f}")
        print(f"    hand accuracy:  {probe.get('hand_accuracy', float('nan')):.4f}")
    aug = results.get("augmentation", {})
    if aug:
        print(f"\n  Augmentation delta hand:  {aug.get('delta', float('nan')):+.4f}")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    args = parse_args(argv)

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Output directory
    mode = args.input_mode
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = _HERE / "results" / f"cvae_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    heldout_phase_idx = PHASE_NAMES.index(args.heldout_phase)
    heldout_grip_id   = GRIP_TO_ID[args.heldout_grip]
    heldout_hand_id   = HAND_TO_ID[args.heldout_hand]
    heldout_label     = f"{args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}"

    print(f"\n{'='*60}")
    print(f"  cVAE — input mode: {mode.upper()}")
    print(f"  Held-out: {heldout_label} (all angles)")
    print(f"  Device: {device}  |  Output: {out_dir}")
    print(f"{'='*60}\n")

    # ---- Data ----
    dataset = load_cvae_dataset(
        args.data_dir,
        heldout_phase_idx=heldout_phase_idx,
        heldout_grip_id=heldout_grip_id,
        heldout_hand_id=heldout_hand_id,
        broadband_data_dir=args.broadband_data_dir if mode == "raw" else None,
    )
    train_ds_raw, val_ds_raw, heldout_ds_raw = split_cvae_dataset(
        dataset, train_frac=0.85, seed=args.seed
    )

    print("\nComputing normalization stats from training set ...")
    norm_stats = compute_cvae_norm_stats(train_ds_raw, mode=mode)

    train_ds   = LFPCVAEDataset(train_ds_raw,   norm_stats, mode=mode)
    val_ds     = LFPCVAEDataset(val_ds_raw,     norm_stats, mode=mode)
    heldout_ds = LFPCVAEDataset(heldout_ds_raw, norm_stats, mode=mode)

    # Determine input_dim from mode
    if mode == "spectral":
        input_dim  = N_REAL_CHANNELS            # 256
    else:
        input_dim  = N_REAL_CHANNELS * N_TIMEPOINTS  # 128000
        # For raw mode override hidden_dims if still at spectral default
        if args.hidden_dims == [256, 128, 64]:
            args.hidden_dims = [2048, 1024, 512, 256]
            print(f"  [raw mode] Auto-setting hidden_dims={args.hidden_dims}")

    # ---- Model ----
    model = LFPCVAE(
        input_dim     = input_dim,
        condition_dim = CONDITION_DIM,
        latent_dim    = args.latent_dim,
        hidden_dims   = args.hidden_dims,
        dropout       = args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: input_dim={input_dim}  latent_dim={args.latent_dim}  "
          f"hidden={args.hidden_dims}  params={n_params:,}")

    np.savez_compressed(out_dir / "normalization_stats.npz", **norm_stats)

    # ---- Train ----
    print("\nTraining ...")
    history = train_cvae(
        model, train_ds, val_ds, args, device,
        save_path=str(out_dir / "checkpoint.pt"),
    )

    # Save training curves
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            epochs = range(1, len(history["train_loss"]) + 1)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, key, title in [
                (axes[0], "recon", "Reconstruction loss"),
                (axes[1], "kl",   "KL loss"),
            ]:
                ax.plot(epochs, history[f"train_{key}"], label="train")
                ax.plot(epochs, history[f"val_{key}"],   label="val")
                ax.set_xlabel("Epoch"); ax.set_title(title); ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"  WARNING: training curve plot failed: {e}")

    # ---- Phase 1: Reconstruction quality ----
    print("\nPhase 1 — Reconstruction quality (seen combinations, val set):")
    model.eval()
    recon_results = reconstruct_seen(model, val_ds, norm_stats, device, args.batch_size)
    # Aggregate summary metrics
    mses = [v["mse"]      for v in recon_results.values() if isinstance(v, dict)]
    rs   = [v["pearsonr"] for v in recon_results.values() if isinstance(v, dict)]
    recon_mse_seen = float(np.mean(mses)) if mses else float("nan")
    recon_r_seen   = float(np.nanmean(rs)) if rs else float("nan")

    # ---- Phase 2: Generation ----
    print("\nPhase 2 — Generation for held-out combination:")
    cond = make_condition_vector(heldout_phase_idx, heldout_grip_id, heldout_hand_id)

    gen_results = evaluate_generation(
        model, heldout_ds, norm_stats, cond,
        device, out_dir, args.no_plot, train_ds,
        n_generated=500, batch_size=args.batch_size,
    )

    # ---- Classifier probe (optional) ----
    probe_results: dict = {}
    if mode == "spectral" and args.classifier_checkpoint and Path(args.classifier_checkpoint).exists():
        print("\nValidation 2 — Classifier probe:")
        probe_results = run_classifier_probe(
            model, norm_stats, cond, args.classifier_checkpoint,
            args, device, n_generated=500, batch_size=args.batch_size,
        )
    elif mode == "spectral" and args.classifier_checkpoint:
        print(f"  WARNING: classifier checkpoint not found: {args.classifier_checkpoint}")
    elif mode == "raw" and args.classifier_checkpoint:
        print("  Skipping classifier probe in raw mode; specialist probe expects spectral 256 features.")

    # ---- Augmentation validation (optional) ----
    aug_results: dict = {}
    if mode == "spectral" and args.classifier_checkpoint and Path(args.classifier_checkpoint).exists():
        print("\nValidation 3 — Augmentation experiment:")
        try:
            # Generate 500 samples for augmentation
            c_tensor = torch.tensor(cond, dtype=torch.float32)
            x_gen_np = model.generate(c_tensor, n_samples=500, device=device).cpu().numpy()
            aug_results = run_augmentation_validation(
                x_gen_np, norm_stats, train_ds, heldout_ds, cond,
                args.classifier_checkpoint, args, device, args.batch_size,
            )
        except Exception as e:
            print(f"  WARNING: augmentation experiment failed: {e}")

    # ---- Latent space analysis ----
    print("\nLatent space analysis ...")
    analyze_latent_space(
        model, train_ds, heldout_ds, out_dir, args.no_plot, device, args.batch_size
    )

    # ---- Save summary ----
    full_results = {
        "heldout_label":   heldout_label,
        "input_mode":      mode,
        "input_dim":       input_dim,
        "latent_dim":      args.latent_dim,
        "hidden_dims":     args.hidden_dims,
        "recon_mse_seen":  recon_mse_seen,
        "recon_r_seen":    recon_r_seen,
        "generation":      gen_results,
        "probe":           probe_results,
        "augmentation":    aug_results,
        "reconstruction_by_combo": {
            k: v for k, v in recon_results.items() if isinstance(v, dict)
        },
        "history": {
            "final_val_recon": float(history["val_recon"][-1]) if history["val_recon"] else None,
            "final_val_kl":    float(history["val_kl"][-1])    if history["val_kl"]    else None,
            "n_epochs":        len(history["train_loss"]),
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(full_results, indent=2, default=float), encoding="utf-8"
    )
    _print_summary_table(full_results, mode)
    print(f"\nAll outputs saved to {out_dir}")
    return full_results


if __name__ == "__main__":
    main()
