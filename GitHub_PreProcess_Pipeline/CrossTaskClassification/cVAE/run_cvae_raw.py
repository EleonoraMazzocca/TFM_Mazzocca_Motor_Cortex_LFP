"""Step 2 — Conditional VAE on raw time series.

Replaces spectral amplitude input with full 256×500 raw waveform.
Same held-out protocol and validation metrics as Step 1.
Expected to overfit with small data — this is by design and becomes
the subject of the scaling analysis in Step 3.

Usage:
    python run_cvae_raw.py --data_dir /path/to/mua_files
    python run_cvae_raw.py --data_dir /path/to/mua_files --dry_run
    python run_cvae_raw.py --data_dir /path/to/mua_files \\
        --classifier_checkpoint results/specialist_grasp_per_channel/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_TRANSFORMER = _HERE.parent / "transformer"
if str(_TRANSFORMER) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data import PHASE_NAMES
from cvae_data import N_REAL_CHANNELS, N_TIMEPOINTS

# Re-use all logic from run_cvae.py — only the defaults differ
from run_cvae import main as _run_cvae_main, parse_args as _parse_cvae_args


# ---------------------------------------------------------------------------
# Argument parsing (raw-mode defaults)
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Same flags as run_cvae.py, but default to raw mode and raw-appropriate architecture."""
    p = argparse.ArgumentParser(
        description="Step 2: raw waveform cVAE for LFP compositional generation."
    )
    p.add_argument("--data_dir",   type=str, required=True,
                   help="MU spectral data directory, used for labels and spectral comparison.")
    p.add_argument("--broadband_data_dir", type=str, required=True,
                   help="Broadband/raw waveform directory containing *_degrees.npy files.")
    p.add_argument("--heldout_phase", choices=PHASE_NAMES, default="grasp")
    p.add_argument("--heldout_grip",  choices=["power", "precision"], default="precision")
    p.add_argument("--heldout_hand",  choices=["left", "right"],      default="right")
    # Raw mode is fixed for this script; no --input_mode flag needed.
    p.add_argument("--latent_dim",  type=int, default=32)
    # Larger hidden dims for raw (128k) input
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[2048, 1024, 512, 256])
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--beta_max",    type=float, default=1.0)
    p.add_argument("--beta_anneal_epochs", type=int, default=10)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--patience",    type=int, default=15)
    p.add_argument("--batch_size",  type=int, default=32,
                   help="Smaller default batch for raw mode (128k dim).")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out_dir",     type=str, default=None)
    p.add_argument("--classifier_checkpoint", type=str, default=None)
    p.add_argument("--device",      choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--no_plot",     action="store_true")
    p.add_argument("--dry_run",     action="store_true")
    p.add_argument("--spec_d_model",        type=int, default=64)
    p.add_argument("--spec_n_heads",        type=int, default=4)
    p.add_argument("--spec_n_layers",       type=int, default=2)
    p.add_argument("--spec_feedforward_dim",type=int, default=128)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(
    spectral_results: dict | None,
    raw_results: dict,
) -> None:
    """Print side-by-side generation quality comparison (spectral vs raw)."""
    print(f"\n{'='*70}")
    print("  GENERATION QUALITY COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<35} {'Spectral(256)':>15} {'Raw(128k)':>12}")
    print("  " + "-" * 65)

    def _val(d: dict | None, *keys, fmt=".4f") -> str:
        if d is None:
            return "N/A"
        v = d
        for k in keys:
            if not isinstance(v, dict):
                return "N/A"
            v = v.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return format(v, fmt)

    rows = [
        ("Recon MSE (seen)",        ["recon_mse_seen"],         ".4f"),
        ("Recon r (seen)",          ["recon_r_seen"],           ".4f"),
        ("Frac channels p>0.05",    ["generation", "frac_channels_p_gt_005"], ".3f"),
        ("MMD ratio (gen/baseline)",["generation", "mmd_ratio"], ".3f"),
        ("Grip probe accuracy",     ["probe", "grip_accuracy"], ".4f"),
        ("Hand probe accuracy",     ["probe", "hand_accuracy"], ".4f"),
        ("Augmentation delta hand", ["augmentation", "delta"],  "+.4f"),
    ]

    for label, keys, fmt in rows:
        sv = _val(spectral_results, *keys, fmt=fmt)
        rv = _val(raw_results,      *keys, fmt=fmt)
        print(f"  {label:<35} {sv:>15} {rv:>12}")

    print(f"{'='*70}")
    print("\n  Decision guide:")
    print("    MMD ratio < 0.5:    excellent compositional generation")
    print("    MMD ratio 0.5-1.0:  moderate — some distributional mismatch")
    print("    MMD ratio > 1.0:    poor — cVAE failed to compose")
    print("    hand probe > 0.70:  hand lateralization composed successfully")

    # Quick verdict
    def _verdict(results: dict | None, label: str) -> None:
        if results is None:
            return
        ratio    = results.get("generation", {}).get("mmd_ratio", float("nan"))
        hand_acc = results.get("probe", {}).get("hand_accuracy", float("nan"))
        if np.isnan(ratio):
            mmd_v = "UNKNOWN"
        elif ratio < 0.5:
            mmd_v = "EXCELLENT"
        elif ratio < 1.0:
            mmd_v = "MODERATE"
        else:
            mmd_v = "POOR"
        if np.isnan(hand_acc):
            hand_v = "UNKNOWN"
        elif hand_acc > 0.70:
            hand_v = "CAPTURED"
        elif hand_acc > 0.50:
            hand_v = "PARTIAL"
        else:
            hand_v = "FAILED"
        print(f"\n  {label}: [{mmd_v}], hand [{hand_v}]")

    _verdict(spectral_results, "Spectral")
    _verdict(raw_results,      "Raw")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    args = parse_args(argv)

    # Set default out_dir for raw mode
    if args.out_dir is None:
        args.out_dir = str(_HERE / "results" / "cvae_raw")

    # Inject input_mode=raw into the args namespace so run_cvae.main() uses it
    args.input_mode = "raw"

    print(f"\n{'='*60}")
    print(f"  cVAE — Step 2: RAW WAVEFORM MODE")
    print(f"  Input dim: {N_REAL_CHANNELS * N_TIMEPOINTS:,} ({N_REAL_CHANNELS} channels × {N_TIMEPOINTS} timepoints)")
    print(f"{'='*60}\n")

    # ---- Run raw cVAE ----
    raw_results = _run_raw(args)

    # ---- Try to load spectral results for comparison ----
    spectral_out = Path(args.out_dir).parent / "cvae_spectral" / "summary.json"
    spectral_results = None
    if spectral_out.exists():
        try:
            spectral_results = json.loads(spectral_out.read_text())
            print(f"  Loaded spectral results from {spectral_out}")
        except Exception:
            pass
    else:
        # Try the default spectral output directory
        spectral_out2 = _HERE / "results" / "cvae_spectral" / "summary.json"
        if spectral_out2.exists():
            try:
                spectral_results = json.loads(spectral_out2.read_text())
            except Exception:
                pass

    print_comparison_table(spectral_results, raw_results)
    return raw_results


def _run_raw(args: argparse.Namespace) -> dict:
    """Run the full cVAE pipeline in raw mode using the provided args namespace."""
    import run_cvae
    # Directly call the pipeline components using the args object
    return run_cvae.main(argv=_args_to_argv(args))


def _args_to_argv(args: argparse.Namespace) -> list[str]:
    """Convert a Namespace back to a CLI argv list for run_cvae.main()."""
    argv = [
        "--data_dir",    args.data_dir,
        "--broadband_data_dir", args.broadband_data_dir,
        "--heldout_phase", args.heldout_phase,
        "--heldout_grip",  args.heldout_grip,
        "--heldout_hand",  args.heldout_hand,
        "--input_mode",    "raw",
        "--latent_dim",    str(args.latent_dim),
        "--dropout",       str(args.dropout),
        "--beta_max",      str(args.beta_max),
        "--beta_anneal_epochs", str(args.beta_anneal_epochs),
        "--lr",            str(args.lr),
        "--weight_decay",  str(args.weight_decay),
        "--epochs",        str(args.epochs),
        "--patience",      str(args.patience),
        "--batch_size",    str(args.batch_size),
        "--seed",          str(args.seed),
        "--out_dir",       str(args.out_dir),
        "--device",        args.device,
        "--spec_d_model",         str(args.spec_d_model),
        "--spec_n_heads",         str(args.spec_n_heads),
        "--spec_n_layers",        str(args.spec_n_layers),
        "--spec_feedforward_dim", str(args.spec_feedforward_dim),
    ]
    # Variadic --hidden_dims
    argv += ["--hidden_dims"] + [str(h) for h in args.hidden_dims]
    if args.classifier_checkpoint:
        argv += ["--classifier_checkpoint", args.classifier_checkpoint]
    if args.no_plot:
        argv.append("--no_plot")
    if args.dry_run:
        argv.append("--dry_run")
    return argv


if __name__ == "__main__":
    main()
