"""Step 3 — Data scaling analysis.

Quantifies how much training data the raw cVAE needs to match spectral cVAE.
Trains multiple models at increasing data fractions and tracks:
  - MMD ratio (generated vs real held-out)
  - Hand probe accuracy
  - Per-channel Pearson correlation

Fits log-linear extrapolation to estimate the crossing point.

Usage:
    python run_cvae_scaling.py --data_dir /path/to/mua_files
    python run_cvae_scaling.py --data_dir /path/to/mua_files \\
        --input_modes spectral raw --n_repeats 3 --fractions 0.1 0.2 0.4 0.6 0.8 1.0
    python run_cvae_scaling.py --data_dir /path/to/mua_files --dry_run
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset

from transformer_encoder.data import PHASE_NAMES, GRIP_TO_ID, HAND_TO_ID
from cvae.cvae_data import (
    load_cvae_dataset, split_cvae_dataset, compute_cvae_norm_stats,
    LFPCVAEDataset, make_condition_vector,
    N_REAL_CHANNELS, N_TIMEPOINTS, CONDITION_DIM,
)
from cvae.cvae_model import LFPCVAE, cvae_loss
from cvae.run_cvae import (
    run_epoch, compute_mmd, run_classifier_probe,
    _load_specialist_for_probe, spectral_to_specialist_input,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 3: data scaling analysis for spectral vs raw cVAE."
    )
    p.add_argument("--data_dir",    type=str, required=True)
    p.add_argument("--broadband_data_dir", type=str, default=None,
                   help="Raw waveform directory; required when input_modes includes raw.")
    p.add_argument("--heldout_phase", choices=PHASE_NAMES, default="grasp")
    p.add_argument("--heldout_grip",  choices=["power", "precision"], default="precision")
    p.add_argument("--heldout_hand",  choices=["left", "right"],      default="right")
    p.add_argument("--fractions",   type=float, nargs="+",
                   default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--input_modes", type=str, nargs="+", default=["spectral", "raw"],
                   choices=["spectral", "raw"])
    p.add_argument("--n_repeats",   type=int, default=3,
                   help="Number of random seeds per (fraction, mode).")
    # Training hyperparams (same defaults as run_cvae.py)
    p.add_argument("--latent_dim",  type=int, default=32)
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
    p.add_argument("--classifier_checkpoint", type=str, default=None)
    p.add_argument("--device",      choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--no_plot",     action="store_true")
    p.add_argument("--dry_run",     action="store_true",
                   help="2 fractions × 2 repeats × 3 epochs for speed check.")
    p.add_argument("--spec_d_model",         type=int, default=64)
    p.add_argument("--spec_n_heads",         type=int, default=4)
    p.add_argument("--spec_n_layers",        type=int, default=2)
    p.add_argument("--spec_feedforward_dim", type=int, default=128)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Single-run training + evaluation
# ---------------------------------------------------------------------------

def _hidden_dims_for_mode(mode: str) -> list[int]:
    return [2048, 1024, 512, 256] if mode == "raw" else [256, 128, 64]


def train_one_run(
    train_idx_subset: np.ndarray,
    train_ds_full:    LFPCVAEDataset,
    val_ds:           LFPCVAEDataset,
    heldout_ds:       LFPCVAEDataset,
    norm_stats:       dict,
    condition:        np.ndarray,
    args:             argparse.Namespace,
    mode:             str,
    seed:             int,
    device:           torch.device,
) -> dict:
    """Train one cVAE and evaluate MMD ratio, hand probe, per-channel correlation.

    Returns dict with keys: mmd_ratio, hand_accuracy, pearsonr_mean, n_train.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim   = N_REAL_CHANNELS if mode == "spectral" else N_REAL_CHANNELS * N_TIMEPOINTS
    hidden_dims = _hidden_dims_for_mode(mode)
    batch_size  = max(16, args.batch_size // (4 if mode == "raw" else 1))

    model = LFPCVAE(
        input_dim=input_dim, condition_dim=CONDITION_DIM,
        latent_dim=args.latent_dim, hidden_dims=hidden_dims, dropout=args.dropout,
    ).to(device)

    train_subset = Subset(train_ds_full, train_idx_subset.tolist())
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,      batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val  = float("inf")
    best_state= None
    patience  = 0
    n_epochs  = args.epochs

    for epoch in range(n_epochs):
        beta = min(1.0, (epoch + 1) / max(args.beta_anneal_epochs, 1)) * args.beta_max
        run_epoch(model, train_loader, optimizer, beta, device)
        vl, _, _ = run_epoch(model, val_loader, None, beta, device)
        if vl < best_val:
            best_val  = vl
            best_state= copy.deepcopy(model.state_dict())
            patience  = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ---- Collect real held-out features ----
    loader   = DataLoader(heldout_ds, batch_size=batch_size, shuffle=False)
    x_real_l = []
    with torch.no_grad():
        for x, *_ in loader:
            x_real_l.append(x.numpy())
    x_real = np.concatenate(x_real_l)  # (N_real, input_dim)

    # ---- Generate held-out samples ----
    c_tensor = torch.tensor(condition, dtype=torch.float32)
    x_gen    = model.generate(c_tensor, n_samples=500, device=device).cpu().numpy()

    # ---- MMD ratio ----
    mmd_gen  = compute_mmd(x_gen, x_real)
    half     = len(x_real) // 2
    mmd_base = compute_mmd(x_real[:half], x_real[half:]) if half > 1 else 1e-10
    mmd_ratio= float(mmd_gen / max(mmd_base, 1e-10))

    # ---- Per-channel Pearson r (reconstruction on val set) ----
    x_val_l, xr_val_l = [], []
    model.eval()
    with torch.no_grad():
        for x, c, *_ in DataLoader(val_ds, batch_size=batch_size):
            xr, _, _ = model(x.to(device), c.to(device))
            x_val_l.append(x.numpy())
            xr_val_l.append(xr.cpu().numpy())
    x_val  = np.concatenate(x_val_l)
    xr_val = np.concatenate(xr_val_l)
    rs = []
    if x_val.shape[1] == N_REAL_CHANNELS * N_TIMEPOINTS:
        xv = x_val.reshape(len(x_val), N_REAL_CHANNELS, N_TIMEPOINTS)
        xrv = xr_val.reshape(len(xr_val), N_REAL_CHANNELS, N_TIMEPOINTS)
        for ch in range(N_REAL_CHANNELS):
            a = xv[:, ch, :].reshape(-1)
            b = xrv[:, ch, :].reshape(-1)
            if a.std() > 1e-8:
                r, _ = pearsonr(a, b)
                rs.append(r)
    else:
        for ch in range(min(x_val.shape[1], N_REAL_CHANNELS)):
            if x_val[:, ch].std() > 1e-8:
                r, _ = pearsonr(x_val[:, ch], xr_val[:, ch])
                rs.append(r)
    mean_r = float(np.nanmean(rs)) if rs else float("nan")

    # ---- Hand probe (optional) ----
    hand_acc = float("nan")
    if args.classifier_checkpoint and Path(args.classifier_checkpoint).exists():
        try:
            probe = run_classifier_probe(
                model, norm_stats, condition, args.classifier_checkpoint,
                args, device, n_generated=500, batch_size=batch_size,
            )
            hand_acc = probe.get("hand_accuracy", float("nan"))
        except Exception:
            pass

    return {
        "mmd_ratio":     mmd_ratio,
        "hand_accuracy": hand_acc,
        "pearsonr_mean": mean_r,
        "n_train":       len(train_idx_subset),
    }


# ---------------------------------------------------------------------------
# Log-linear extrapolation
# ---------------------------------------------------------------------------

def fit_log_linear(n_trials: list[int], metric: list[float]) -> tuple[float, float, float]:
    """Fit metric ~ a + b * log(n_trials), return (a, b, r²)."""
    valid = [(n, m) for n, m in zip(n_trials, metric) if not np.isnan(m) and n > 0]
    if len(valid) < 2:
        return (float("nan"), float("nan"), float("nan"))
    ns, ms = zip(*valid)
    log_ns = np.log(ns)
    b, a   = np.polyfit(log_ns, ms, 1)
    ms_hat = a + b * np.array(log_ns)
    ss_res = np.sum((np.array(ms) - ms_hat) ** 2)
    ss_tot = np.sum((np.array(ms) - np.mean(ms)) ** 2)
    r2     = 1.0 - ss_res / max(ss_tot, 1e-10)
    return (float(a), float(b), float(r2))


def extrapolate_crossing(
    ns_spec: list[int], ms_spec: list[float],
    ns_raw:  list[int], ms_raw:  list[float],
    target_metric: float = None,
) -> dict:
    """Estimate when raw cVAE reaches the spectral cVAE's metric value.

    If target_metric is None, uses mean of spectral curve as target.
    Returns estimated n_trials for raw to match spectral.
    """
    a_s, b_s, _ = fit_log_linear(ns_spec, ms_spec)
    a_r, b_r, _ = fit_log_linear(ns_raw,  ms_raw)

    # Target: spectral plateau (value at max n)
    if target_metric is None and not np.isnan(a_s):
        target_metric = a_s + b_s * np.log(max(ns_spec))

    if np.isnan(target_metric) or np.isnan(b_r) or abs(b_r) < 1e-10:
        return {"target": target_metric, "estimated_n_raw": float("nan")}

    log_n_cross = (target_metric - a_r) / b_r
    n_cross = float(np.exp(log_n_cross))
    return {
        "target":           float(target_metric),
        "estimated_n_raw":  n_cross,
        "a_spectral":       a_s, "b_spectral": b_s,
        "a_raw":            a_r, "b_raw":      b_r,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_scaling_curves(
    results_by_mode: dict,
    metric_key: str,
    ylabel: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {"spectral": "#4C72B0", "raw": "#DD8452"}
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, data_points in results_by_mode.items():
        ns   = sorted(data_points.keys())
        means= [np.nanmean(data_points[n][metric_key]) for n in ns]
        stds = [np.nanstd(data_points[n][metric_key])  for n in ns]
        ax.plot(ns, means, "o-", label=mode, color=colors.get(mode, "grey"))
        ax.fill_between(
            ns,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2, color=colors.get(mode, "grey"),
        )

    ax.set_xlabel("Training trials")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Scaling analysis — {ylabel}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    args = parse_args(argv)

    if args.dry_run:
        args.fractions = [0.1, 0.5]
        args.n_repeats = 2
        args.epochs    = 3
        args.patience  = 2
        print("[dry-run] fractions=[0.1, 0.5]  n_repeats=2  epochs=3")

    if "raw" in args.input_modes and not args.broadband_data_dir:
        raise SystemExit("--broadband_data_dir is required when --input_modes includes raw")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir or (_HERE / "results" / "cvae_scaling"))
    out_dir.mkdir(parents=True, exist_ok=True)

    heldout_phase_idx = PHASE_NAMES.index(args.heldout_phase)
    heldout_grip_id   = GRIP_TO_ID[args.heldout_grip]
    heldout_hand_id   = HAND_TO_ID[args.heldout_hand]
    condition = make_condition_vector(heldout_phase_idx, heldout_grip_id, heldout_hand_id)

    print(f"\nCVAE scaling analysis | device={device} | out={out_dir}")
    print(f"  Modes: {args.input_modes}  |  fractions: {args.fractions}")
    print(f"  n_repeats: {args.n_repeats}  |  held-out: {args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}")

    # ---- Load dataset once ----
    dataset = load_cvae_dataset(
        args.data_dir,
        heldout_phase_idx=heldout_phase_idx,
        heldout_grip_id=heldout_grip_id,
        heldout_hand_id=heldout_hand_id,
        broadband_data_dir=args.broadband_data_dir if "raw" in args.input_modes else None,
    )
    train_ds_raw, val_ds_raw, heldout_ds_raw = split_cvae_dataset(
        dataset, train_frac=0.85, seed=args.seed
    )
    n_train_full = len(train_ds_raw["y_grip"])

    print(f"  Full training set: {n_train_full} samples")

    # ---- Run scaling experiments ----
    # results_by_mode[mode][n_trials] = list of metric dicts (one per seed)
    results_by_mode: dict[str, dict[int, list]] = {m: {} for m in args.input_modes}

    total_runs = len(args.input_modes) * len(args.fractions) * args.n_repeats
    run_idx    = 0

    for mode in args.input_modes:
        norm_stats = compute_cvae_norm_stats(train_ds_raw, mode=mode)
        train_ds   = LFPCVAEDataset(train_ds_raw,   norm_stats, mode=mode)
        val_ds     = LFPCVAEDataset(val_ds_raw,     norm_stats, mode=mode)
        heldout_ds = LFPCVAEDataset(heldout_ds_raw, norm_stats, mode=mode)

        for frac in args.fractions:
            n_subset = max(1, int(n_train_full * frac))
            results_by_mode[mode].setdefault(n_subset, [])

            for repeat in range(args.n_repeats):
                seed = args.seed + repeat * 100
                run_idx += 1

                # Subsample training indices
                rng   = np.random.default_rng(seed)
                idx   = rng.choice(n_train_full, size=n_subset, replace=False)
                idx.sort()

                print(
                    f"\n[{run_idx}/{total_runs}] mode={mode:<8}  "
                    f"frac={frac:.1f}  n={n_subset}  seed={seed}"
                )

                metrics = train_one_run(
                    train_idx_subset=idx,
                    train_ds_full=train_ds,
                    val_ds=val_ds,
                    heldout_ds=heldout_ds,
                    norm_stats=norm_stats,
                    condition=condition,
                    args=args,
                    mode=mode,
                    seed=seed,
                    device=device,
                )
                results_by_mode[mode][n_subset].append(metrics)
                print(
                    f"  → mmd_ratio={metrics['mmd_ratio']:.3f}  "
                    f"hand_acc={metrics['hand_accuracy']:.3f}  "
                    f"r={metrics['pearsonr_mean']:.3f}"
                )

    # ---- Extrapolation ----
    extrap: dict = {}
    if "spectral" in results_by_mode and "raw" in results_by_mode:
        for metric_key, maximize in [
            ("mmd_ratio",     False),  # lower is better; raw should reach spectral's low value
            ("hand_accuracy", True),
            ("pearsonr_mean", True),
        ]:
            ns_spec = sorted(results_by_mode["spectral"].keys())
            ns_raw  = sorted(results_by_mode["raw"].keys())
            ms_spec = [np.nanmean([r[metric_key] for r in results_by_mode["spectral"][n]])
                       for n in ns_spec]
            ms_raw  = [np.nanmean([r[metric_key] for r in results_by_mode["raw"][n]])
                       for n in ns_raw]
            # For mmd_ratio: spectral target = min (best raw should reach spectral's best)
            if not maximize:
                target = min(ms_spec) if ms_spec else None
            else:
                target = max(ms_spec) if ms_spec else None
            extrap[metric_key] = extrapolate_crossing(ns_spec, ms_spec, ns_raw, ms_raw, target)

    # ---- Print summary ----
    print(f"\n{'='*70}")
    print("  DATA SCALING EXTRAPOLATION")
    print(f"{'='*70}")
    for metric_key in ("mmd_ratio", "hand_accuracy", "pearsonr_mean"):
        e = extrap.get(metric_key, {})
        n_raw = e.get("estimated_n_raw", float("nan"))
        if not np.isnan(n_raw):
            ns_spec = sorted(results_by_mode.get("spectral", {}).keys())
            n_spec  = max(ns_spec) if ns_spec else float("nan")
            factor  = n_raw / max(n_spec, 1)
            print(f"\n  {metric_key}:")
            print(f"    Spectral saturates at: ~{n_spec} trials")
            print(f"    Raw achieves equivalent at: ~{n_raw:.0f} trials")
            print(f"    Raw requires ~{factor:.1f}× more data than spectral")
    print(f"{'='*70}")

    # ---- Plot scaling curves ----
    if not args.no_plot:
        for metric_key, ylabel in [
            ("mmd_ratio",     "MMD ratio (gen/baseline)"),
            ("hand_accuracy", "Hand probe accuracy"),
            ("pearsonr_mean", "Mean per-channel Pearson r"),
        ]:
            # Aggregate per (mode, n_trials)
            rb = {}
            for mode in args.input_modes:
                rb[mode] = {n: [r[metric_key] for r in reps]
                            for n, reps in results_by_mode[mode].items()}
            fname = {"mmd_ratio": "scaling_mmd", "hand_accuracy": "scaling_hand_acc",
                     "pearsonr_mean": "scaling_correlation"}[metric_key]
            _plot_scaling_curves(rb, metric_key, ylabel, out_dir / f"{fname}.png")

    # ---- Save results ----
    # Convert int keys to strings for JSON serialization
    serializable = {}
    for mode, n_dict in results_by_mode.items():
        serializable[mode] = {
            str(n): reps for n, reps in n_dict.items()
        }

    summary = {
        "results":      serializable,
        "extrapolation":extrap,
        "args": {
            "fractions":    args.fractions,
            "input_modes":  args.input_modes,
            "n_repeats":    args.n_repeats,
            "heldout":      f"{args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    print(f"\nResults saved to {out_dir}")
    return summary


if __name__ == "__main__":
    main()
