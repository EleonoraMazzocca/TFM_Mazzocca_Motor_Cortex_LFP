"""Orchestrator for all three cVAE generative experiment steps.

Runs Steps 1, 2, 3 as subprocesses (so memory is fully released between steps)
and prints a unified comparison table at the end.

Usage:
    # Verify pipeline (fast)
    python run_all_cvae.py --data_dir /path/to/mua_files --dry_run --no_plot

    # Step 1 only (spectral cVAE)
    python run_all_cvae.py --data_dir /path/to/mua_files --run_steps 1

    # Steps 1 and 2 with classifier probe
    python run_all_cvae.py \\
        --data_dir /path/to/mua_files \\
        --classifier_checkpoint results/specialist_grasp_per_channel/checkpoint.pt \\
        --run_steps 1 2

    # Full pipeline
    python run_all_cvae.py \\
        --data_dir /path/to/mua_files \\
        --classifier_checkpoint results/specialist_grasp_per_channel/checkpoint.pt \\
        --run_steps 1 2 3 --epochs 100 --patience 15 --n_repeats 3 --device cuda
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent

import numpy as np

from transformer_encoder.data import PHASE_NAMES


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orchestrate the cVAE generative experiment (Steps 1, 2, 3)."
    )
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--broadband_data_dir", type=str, default=None,
                   help="Raw waveform directory; required for Step 2 or Step 3 raw mode.")
    p.add_argument("--run_steps",  type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3],
                   help="Which steps to run (default: all three).")
    p.add_argument("--heldout_phase", choices=PHASE_NAMES, default="grasp")
    p.add_argument("--heldout_grip",  choices=["power", "precision"], default="precision")
    p.add_argument("--heldout_hand",  choices=["left", "right"],      default="right")
    # Training hyperparams (forwarded to individual scripts)
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
    p.add_argument("--out_base",    type=str, default=str(_HERE / "results"),
                   help="Base output directory; each step writes to a subdirectory.")
    p.add_argument("--classifier_checkpoint", type=str, default=None)
    p.add_argument("--device",      choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--no_plot",     action="store_true")
    p.add_argument("--dry_run",     action="store_true")
    # Step 3 specific
    p.add_argument("--fractions",   type=float, nargs="+",
                   default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--input_modes", type=str, nargs="+", default=["spectral", "raw"],
                   choices=["spectral", "raw"])
    p.add_argument("--n_repeats",   type=int, default=3)
    # Specialist arch (for probe)
    p.add_argument("--spec_d_model",         type=int, default=64)
    p.add_argument("--spec_n_heads",         type=int, default=4)
    p.add_argument("--spec_n_layers",        type=int, default=2)
    p.add_argument("--spec_feedforward_dim", type=int, default=128)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _common_flags(args: argparse.Namespace) -> list[str]:
    """Flags shared by all three step scripts."""
    flags = [
        "--data_dir",    args.data_dir,
        "--heldout_phase", args.heldout_phase,
        "--heldout_grip",  args.heldout_grip,
        "--heldout_hand",  args.heldout_hand,
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
        "--device",        args.device,
        "--spec_d_model",         str(args.spec_d_model),
        "--spec_n_heads",         str(args.spec_n_heads),
        "--spec_n_layers",        str(args.spec_n_layers),
        "--spec_feedforward_dim", str(args.spec_feedforward_dim),
    ]
    if args.broadband_data_dir:
        flags += ["--broadband_data_dir", args.broadband_data_dir]
    if args.classifier_checkpoint:
        flags += ["--classifier_checkpoint", args.classifier_checkpoint]
    if args.no_plot:
        flags.append("--no_plot")
    if args.dry_run:
        flags.append("--dry_run")
    return flags


def _run_step(script: Path, extra_flags: list[str], step_label: str) -> None:
    cmd = [sys.executable, str(script)] + extra_flags
    print(f"\n{'='*70}")
    print(f"  Launching: {step_label}")
    print(f"  Command: {' '.join(cmd[:6])} ...")
    print(f"{'='*70}")
    subprocess.run(cmd, check=True)


def run_step1(args: argparse.Namespace, out_base: Path) -> Path:
    """Run spectral cVAE (Step 1). Returns output directory."""
    out_dir = out_base / "cvae_spectral"
    script  = _HERE / "run_cvae.py"
    extra   = _common_flags(args) + [
        "--input_mode", "spectral",
        "--out_dir",    str(out_dir),
    ]
    _run_step(script, extra, "Step 1 — Spectral cVAE")
    return out_dir


def run_step2(args: argparse.Namespace, out_base: Path) -> Path:
    """Run raw waveform cVAE (Step 2). Returns output directory."""
    out_dir   = out_base / "cvae_raw"
    script    = _HERE / "run_cvae_raw.py"
    # raw mode uses smaller batch size by default
    batch_raw = max(16, args.batch_size // 4)
    extra = [
        "--data_dir",    args.data_dir,
        "--broadband_data_dir", args.broadband_data_dir,
        "--heldout_phase", args.heldout_phase,
        "--heldout_grip",  args.heldout_grip,
        "--heldout_hand",  args.heldout_hand,
        "--latent_dim",    str(args.latent_dim),
        "--dropout",       str(args.dropout),
        "--beta_max",      str(args.beta_max),
        "--beta_anneal_epochs", str(args.beta_anneal_epochs),
        "--lr",            str(args.lr),
        "--weight_decay",  str(args.weight_decay),
        "--epochs",        str(args.epochs),
        "--patience",      str(args.patience),
        "--batch_size",    str(batch_raw),
        "--seed",          str(args.seed),
        "--out_dir",       str(out_dir),
        "--device",        args.device,
        "--spec_d_model",         str(args.spec_d_model),
        "--spec_n_heads",         str(args.spec_n_heads),
        "--spec_n_layers",        str(args.spec_n_layers),
        "--spec_feedforward_dim", str(args.spec_feedforward_dim),
    ]
    if args.classifier_checkpoint:
        extra += ["--classifier_checkpoint", args.classifier_checkpoint]
    if args.no_plot:
        extra.append("--no_plot")
    if args.dry_run:
        extra.append("--dry_run")
    _run_step(script, extra, "Step 2 — Raw waveform cVAE")
    return out_dir


def run_step3(args: argparse.Namespace, out_base: Path) -> Path:
    """Run data scaling analysis (Step 3). Returns output directory."""
    out_dir = out_base / "cvae_scaling"
    script  = _HERE / "run_cvae_scaling.py"
    extra   = [
        "--data_dir",    args.data_dir,
        "--heldout_phase", args.heldout_phase,
        "--heldout_grip",  args.heldout_grip,
        "--heldout_hand",  args.heldout_hand,
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
        "--n_repeats",     str(args.n_repeats),
        "--out_dir",       str(out_dir),
        "--device",        args.device,
        "--spec_d_model",         str(args.spec_d_model),
        "--spec_n_heads",         str(args.spec_n_heads),
        "--spec_n_layers",        str(args.spec_n_layers),
        "--spec_feedforward_dim", str(args.spec_feedforward_dim),
    ]
    if args.broadband_data_dir:
        extra += ["--broadband_data_dir", args.broadband_data_dir]
    extra += ["--fractions"]   + [str(f) for f in args.fractions]
    extra += ["--input_modes"] + args.input_modes
    if args.classifier_checkpoint:
        extra += ["--classifier_checkpoint", args.classifier_checkpoint]
    if args.no_plot:
        extra.append("--no_plot")
    if args.dry_run:
        extra.append("--dry_run")
    _run_step(script, extra, "Step 3 — Data scaling analysis")
    return out_dir


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _load_summary(path: Path) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def print_unified_summary(
    step1_dir: Path | None,
    step2_dir: Path | None,
    step3_dir: Path | None,
    args: argparse.Namespace,
) -> None:
    """Print the final unified comparison table as specified."""
    s1 = _load_summary((step1_dir / "summary.json") if step1_dir else Path("/dev/null"))
    s2 = _load_summary((step2_dir / "summary.json") if step2_dir else Path("/dev/null"))
    s3 = _load_summary((step3_dir / "summary.json") if step3_dir else Path("/dev/null"))

    def _v(d, *keys, fmt=".4f"):
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

    W1, W2 = 18, 14
    print(f"\n{'='*80}")
    print("  CVAE SUMMARY")
    print(f"{'='*80}")
    print(f"  Held-out: {args.heldout_phase} + {args.heldout_grip} + {args.heldout_hand} (all angles)")
    print("  Scientific question: can the model compose temporal structure (phase)")
    print("  with movement parameters (grip, hand) seen separately?\n")
    print(f"  {'Metric':<36} {'Spectral(256)':>{W1}} {'Raw(128k)':>{W2}}")
    print("  " + "-" * (36 + W1 + W2 + 2))

    rows = [
        ("Reconstruction MSE (seen)",        ["recon_mse_seen"], ".4f"),
        ("Reconstruction r (seen)",          ["recon_r_seen"],   ".4f"),
        ("",                                 [],                 ""),
        ("Distributional:",                  [],                 ""),
        ("  Fraction channels p>0.05",       ["generation", "frac_channels_p_gt_005"], ".3f"),
        ("  MMD ratio (gen/baseline)",        ["generation", "mmd_ratio"],              ".3f"),
        ("",                                  [],                 ""),
        ("Classifier probe (held-out):",      [],                 ""),
        ("  grip accuracy",                  ["probe", "grip_accuracy"], ".4f"),
        ("  hand accuracy",                  ["probe", "hand_accuracy"], ".4f"),
        ("Augmentation delta hand",          ["augmentation", "delta"], "+.4f"),
    ]

    for label, keys, fmt in rows:
        if not keys:
            if label:
                print(f"\n  {label}")
            else:
                print()
            continue
        sv = _v(s1, *keys, fmt=fmt)
        rv = _v(s2, *keys, fmt=fmt)
        print(f"  {label:<36} {sv:>{W1}} {rv:>{W2}}")

    # Step 3 scaling
    if s3:
        e_mmd  = s3.get("extrapolation", {}).get("mmd_ratio", {})
        n_raw  = e_mmd.get("estimated_n_raw", float("nan"))

        print(f"\n  {'─'*70}")
        print("  Data scaling (Step 3):")
        ns_spec = [int(k) for k in s3.get("results", {}).get("spectral", {}).keys()]
        n_spec_max = max(ns_spec) if ns_spec else float("nan")
        factor = n_raw / max(n_spec_max, 1) if not np.isnan(n_raw) and not np.isnan(n_spec_max) else float("nan")
        print(f"    Spectral saturates at:    ~{n_spec_max} trials")
        if not np.isnan(n_raw):
            print(f"    Raw needs ~{n_raw:.0f} trials to match spectral")
            print(f"    Raw requires ~{factor:.1f}× more data than spectral")

    print(f"\n  {'─'*70}")
    print("  Decision guide:")
    print("    MMD ratio < 0.5:   excellent compositional generation")
    print("    MMD ratio 0.5-1.0: moderate — some distributional mismatch")
    print("    MMD ratio > 1.0:   poor — cVAE failed to compose")
    print()
    print("    hand probe > 0.70: hand lateralization composed successfully")
    print("    hand probe < 0.50: hand factor was not captured")

    def _verdict_str(s: dict | None) -> str:
        if s is None:
            return "(not run)"
        ratio = s.get("generation", {}).get("mmd_ratio", float("nan"))
        hand  = s.get("probe", {}).get("hand_accuracy", float("nan"))
        if np.isnan(ratio):
            mmd_v = "UNKNOWN"
        elif ratio < 0.5:
            mmd_v = "EXCELLENT"
        elif ratio < 1.0:
            mmd_v = "MODERATE"
        else:
            mmd_v = "POOR"
        if np.isnan(hand):
            hand_v = "UNKNOWN"
        elif hand > 0.70:
            hand_v = "CAPTURED"
        elif hand > 0.50:
            hand_v = "PARTIAL"
        else:
            hand_v = "FAILED"
        return f"[{mmd_v}], hand [{hand_v}]"

    print("\n  Current verdict:")
    print(f"    Spectral: {_verdict_str(s1)}")
    print(f"    Raw:      {_verdict_str(s2)}")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Thesis summary figure
# ---------------------------------------------------------------------------

def _make_thesis_figure(
    step1_dir: Path | None,
    step2_dir: Path | None,
    step3_dir: Path | None,
    out_path: Path,
) -> None:
    """Panel figure: PCA | MMD bars | Probe bars | Scaling curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping thesis figure.")
        return

    s1 = _load_summary((step1_dir / "summary.json") if step1_dir else None)
    s2 = _load_summary((step2_dir / "summary.json") if step2_dir else None)
    s3 = _load_summary((step3_dir / "summary.json") if step3_dir else None)

    n_panels = sum([s1 is not None, True, True, s3 is not None])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel A: PCA (load from image if available)
    ax = axes[0]
    pca_img_path = (step1_dir / "pca_generation.png") if step1_dir else None
    if pca_img_path and pca_img_path.exists():
        img = plt.imread(str(pca_img_path))
        ax.imshow(img); ax.axis("off")
        ax.set_title("A  PCA: real vs generated", fontsize=10)
    else:
        ax.text(0.5, 0.5, "PCA\n(run Step 1)", ha="center", va="center")
        ax.set_title("A  PCA", fontsize=10)

    # Panel B: MMD ratio bar chart
    ax = axes[1]
    modes  = []
    ratios = []
    if s1:
        modes.append("Spectral"); ratios.append(s1.get("generation", {}).get("mmd_ratio", np.nan))
    if s2:
        modes.append("Raw");      ratios.append(s2.get("generation", {}).get("mmd_ratio", np.nan))
    colors = ["#4C72B0", "#DD8452"]
    bars   = ax.bar(modes, ratios, color=colors[:len(modes)], alpha=0.85)
    ax.axhline(1.0, linestyle="--", color="grey", linewidth=1, label="baseline MMD ratio=1")
    ax.set_ylabel("MMD ratio (generated / baseline)")
    ax.set_title("B  MMD ratio", fontsize=10)
    ax.legend(fontsize=7)
    for bar, v in zip(bars, ratios):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=8)

    # Panel C: Classifier probe bars
    ax    = axes[2]
    heads = ["grip", "hand"]
    xs    = np.arange(len(heads))
    width = 0.35
    chance= [0.5, 0.5]
    for offset, (s, label, color) in enumerate(
        [(s1, "Spectral", "#4C72B0"), (s2, "Raw", "#DD8452")]
    ):
        if s is None:
            continue
        probe = s.get("probe", {})
        vals  = [probe.get(f"{h}_accuracy", np.nan) for h in heads]
        ax.bar(xs + (offset - 0.5) * width, vals, width, label=label, color=color, alpha=0.85)
    for xi, (h, ch) in enumerate(zip(heads, chance)):
        ax.axhline(ch, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(heads)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("C  Classifier probe", fontsize=10)
    ax.legend(fontsize=7)

    # Panel D: Scaling curves (MMD ratio)
    ax = axes[3]
    if s3:
        results = s3.get("results", {})
        for mode, color in [("spectral", "#4C72B0"), ("raw", "#DD8452")]:
            if mode not in results:
                continue
            ns   = sorted(int(k) for k in results[mode].keys())
            means= [np.nanmean([r["mmd_ratio"] for r in results[mode][str(n)]]) for n in ns]
            stds = [np.nanstd( [r["mmd_ratio"] for r in results[mode][str(n)]]) for n in ns]
            ax.plot(ns, means, "o-", label=mode, color=color)
            ax.fill_between(ns,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.2, color=color)
        ax.set_xlabel("Training trials")
        ax.set_ylabel("MMD ratio")
        ax.set_title("D  Data scaling", fontsize=10)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Run Step 3\nfor scaling curves", ha="center", va="center")
        ax.set_title("D  Data scaling", fontsize=10)

    plt.suptitle("cVAE compositional generation — thesis summary", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved thesis figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    if (2 in args.run_steps or (3 in args.run_steps and "raw" in args.input_modes)) and not args.broadband_data_dir:
        raise SystemExit("--broadband_data_dir is required for Step 2 and for Step 3 raw mode")
    out_base= Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    heldout = f"{args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}"
    print(f"\n{'='*70}")
    print("  CVAE EXPERIMENT ORCHESTRATOR")
    print(f"  Steps to run: {args.run_steps}")
    print(f"  Held-out: {heldout}")
    print(f"  Output base: {out_base}")
    print(f"{'='*70}")

    step1_dir = step2_dir = step3_dir = None

    if 1 in args.run_steps:
        step1_dir = run_step1(args, out_base)

    if 2 in args.run_steps:
        step2_dir = run_step2(args, out_base)

    if 3 in args.run_steps:
        step3_dir = run_step3(args, out_base)

    # ---- Unified summary table ----
    print_unified_summary(step1_dir, step2_dir, step3_dir, args)

    # ---- Thesis summary figure ----
    if not args.no_plot:
        _make_thesis_figure(
            step1_dir, step2_dir, step3_dir,
            out_base / "cvae_thesis_summary.png",
        )

    print("\nAll cVAE steps complete.")


if __name__ == "__main__":
    main()
