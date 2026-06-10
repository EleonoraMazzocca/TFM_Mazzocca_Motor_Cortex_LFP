"""Run all three phase specialists sequentially and print the thesis summary table.

Each specialist runs in its own subprocess so memory is fully released between runs.

Usage:
    python run_all_specialists.py
    python run_all_specialists.py --n_permutations 100   # fast dev run
    python run_all_specialists.py --phases reach grasp   # subset
    python run_all_specialists.py --n_bins 10            # multi-bin features
    python run_all_specialists.py --heldout power_left_0 # different held-out combo
    python run_all_specialists.py --angles binary        # 0° vs 135° only
    python run_all_specialists.py --sweep_bins 1 5 10 20 # ablation sweep table
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SPECIALIST_SCRIPT = _HERE / "run_specialists.py"

PHASE_NAMES = ["prereach", "reach", "grasp"]
AREA_NAMES = ["PMvR", "M1", "PMdR", "PMdL"]
HEAD_NAMES = ["grip", "hand", "angle"]


# ---------------------------------------------------------------------------
# Argument parsing helpers (shared with run_specialists.py)
# ---------------------------------------------------------------------------

_VALID_N_BINS = {1, 5, 10, 20}


def _n_bins_type(s: str) -> int | str:
    if s == "raw":
        return "raw"
    v = int(s)
    if v not in _VALID_N_BINS:
        raise argparse.ArgumentTypeError(
            f"--n_bins must be one of {sorted(_VALID_N_BINS)} or 'raw', got {v}"
        )
    return v


def _heldout_str(args: argparse.Namespace) -> str:
    """Build the --heldout string for subprocess forwarding."""
    if getattr(args, "heldout", None):
        return args.heldout
    grip_name = {0: "power", 1: "precision"}[args.heldout_grip]
    hand_name = {0: "left", 1: "right"}[args.heldout_hand]
    angle_name = {0: "0", 1: "45", 2: "90", 3: "135"}[args.heldout_angle]
    return f"{grip_name}_{hand_name}_{angle_name}"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all phase specialists and produce the thesis summary table."
    )
    parser.add_argument("--phases", nargs="+", choices=PHASE_NAMES, default=PHASE_NAMES,
                        help="Which phases to run (default: all three).")
    # Held-out combination — string form (preferred) or separate ints for backward compat
    parser.add_argument("--heldout", type=str, default=None,
                        help="Held-out combination, e.g. 'precision_right_135'. "
                             "Overrides --heldout_grip/hand/angle.")
    parser.add_argument("--heldout_grip", type=int, default=1)
    parser.add_argument("--heldout_hand", type=int, default=1)
    parser.add_argument("--heldout_angle", type=int, default=3)
    # Temporal resolution
    parser.add_argument("--n_bins", type=_n_bins_type, default=1,
                        help="Temporal bins per area token for normal (non-sweep) mode. "
                             "1=single avg (default), 5/10/20=multi-bin, raw=500.")
    parser.add_argument("--angles", choices=["all", "binary"], default="all",
                        help="all: 4 angles (default); binary: 0° and 135° only.")
    # Sweep mode
    parser.add_argument("--sweep_bins", nargs="+", type=_n_bins_type, default=None,
                        help="Sweep over multiple n_bins values, e.g. --sweep_bins 1 5 10 20. "
                             "Runs all phases × all n_bins and prints a 2-D accuracy table. "
                             "Incompatible with --n_bins (sweep takes precedence).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_permutations", type=int, default=1000)
    parser.add_argument("--out_base", type=str, default=str(_HERE / "results"),
                        help="Base output directory. Each phase goes into out_base/specialist_{phase}/")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--no_heldout", action="store_true",
                        help="Standard stratified 80/10/10 split; no combination held out. "
                             "Skips permutation test.")
    parser.add_argument("--per_channel", action="store_true",
                        help="Use per-channel features. Mutually exclusive with --sweep_bins.")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Forward --dry_run to each specialist (2 epochs, 3 permutations).")
    # Model hyperparams — forwarded unchanged to run_specialists.py
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--feedforward_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Run one specialist as a subprocess
# ---------------------------------------------------------------------------

def _build_cmd(
    phase: str,
    args: argparse.Namespace,
    n_bins: int | str,
    out_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable, str(_SPECIALIST_SCRIPT),
        "--phase", phase,
        "--n_bins", str(n_bins),
    ]
    if not getattr(args, "no_heldout", False):
        cmd += ["--heldout", _heldout_str(args)]
    cmd += [
        "--angles", args.angles,
        "--seed", str(args.seed),
        "--n_permutations", str(args.n_permutations),
        "--out_dir", str(out_dir),
        "--d_model", str(args.d_model),
        "--n_heads", str(args.n_heads),
        "--n_layers", str(args.n_layers),
        "--feedforward_dim", str(args.feedforward_dim),
        "--dropout", str(args.dropout),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
    ]
    if args.cache_dir:
        cmd += ["--cache_dir", args.cache_dir]
    if getattr(args, "no_heldout", False):
        cmd.append("--no_heldout")
    if getattr(args, "per_channel", False):
        cmd.append("--per_channel")
    if args.no_plot:
        cmd.append("--no_plot")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def run_phase_specialist(phase: str, args: argparse.Namespace) -> Path:
    """Launch run_specialists.py for one phase (normal mode) and return its output dir."""
    if getattr(args, "no_heldout", False):
        out_dir = Path(args.out_base) / f"specialist_{phase}_no_heldout"
        label = "no_heldout"
    elif getattr(args, "per_channel", False):
        out_dir = Path(args.out_base) / f"specialist_{phase}_per_channel"
        label = "per_channel"
    else:
        out_dir = Path(args.out_base) / f"specialist_{phase}"
        label = f"n_bins={args.n_bins}"
    cmd = _build_cmd(phase, args, args.n_bins, out_dir)

    print(f"\n{'='*70}")
    print(f"  Launching {phase.upper()} specialist  ({label})")
    print(f"{'='*70}")
    subprocess.run(cmd, check=True)
    return out_dir


def run_phase_specialist_sweep(
    phase: str, n_bins: int | str, args: argparse.Namespace
) -> Path:
    """Launch run_specialists.py for one phase + one n_bins value in sweep mode."""
    out_dir = Path(args.out_base) / f"specialist_{phase}_nbins{n_bins}"
    cmd = _build_cmd(phase, args, n_bins, out_dir)

    print(f"\n{'='*70}")
    print(f"  Launching {phase.upper()} specialist  (n_bins={n_bins})  [sweep]")
    print(f"{'='*70}")
    subprocess.run(cmd, check=True)
    return out_dir


def load_summary(out_dir: Path) -> dict:
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Summary tables — normal mode
# ---------------------------------------------------------------------------

def format_accuracy_table(summaries: dict[str, dict], angles: str = "all") -> str:
    """Main thesis table: per-head accuracy + angle p-value by phase."""
    n_angle_classes = 2 if angles == "binary" else 4
    chance = 1.0 / n_angle_classes

    C, V = 12, 11  # column widths
    header = (
        f"{'Phase':<{C}} | "
        f"{'Grip(seen)':>{V}} | {'Grip(held)':>{V}} | "
        f"{'Hand(seen)':>{V}} | {'Hand(held)':>{V}} | "
        f"{'Angle(seen)':>{V}} | {'Angle(held)':>{V}} | "
        f"{'Angle p-val':>{V}}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for phase in PHASE_NAMES:
        if phase not in summaries:
            lines.append(f"{phase:<{C}} | {'(not run)':>{V}}")
            continue
        s = summaries[phase]
        seen = s["seen_accuracy"]
        held = s["heldout_accuracy"]
        p = s["p_values"]["angle"]
        sig = " *" if p < 0.05 else "  "
        lines.append(
            f"{phase:<{C}} | "
            f"{seen['grip']:>{V}.4f} | {held['grip']:>{V}.4f} | "
            f"{seen['hand']:>{V}.4f} | {held['hand']:>{V}.4f} | "
            f"{seen['angle']:>{V}.4f} | {held['angle']:>{V}.4f} | "
            f"{p:>{V}.4f}{sig}"
        )

    lines += [
        sep,
        "* p < 0.05 (permutation test, heldout labels shuffled independently per head)",
        f"angles={angles}  |  angle chance level: {chance:.2f} ({n_angle_classes} classes)",
    ]
    return "\n".join(lines)


def format_attention_table(summaries: dict[str, dict]) -> str:
    """Area importance by phase (normalized attention received, last layer, heldout set)."""
    C, V = 12, 10
    header = (
        f"{'Phase':<{C}} | "
        + " | ".join(f"{name:>{V}}" for name in AREA_NAMES)
    )
    sep = "-" * len(header)
    lines = [
        "\nBrain area importance — normalized attention received (held-out test, last attn layer):",
        sep, header, sep,
    ]
    for phase in PHASE_NAMES:
        if phase not in summaries:
            continue
        imp = summaries[phase]["area_importance_heldout"]
        lines.append(
            f"{phase:<{C}} | "
            + " | ".join(f"{imp[name]:>{V}.4f}" for name in AREA_NAMES)
        )
    lines.append(sep)
    lines.append(
        "Interpretation: higher = other areas attend more to this area "
        "(information source); columns sum to ~1 per row."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary table — sweep mode
# ---------------------------------------------------------------------------

def format_sweep_table(
    sweep_summaries: dict[str, dict],
    sweep_bins: list[int | str],
    angles: str = "all",
) -> str:
    """2-D table: rows = phases, columns = n_bins values, cells = angle held-out accuracy."""
    n_angle_classes = 2 if angles == "binary" else 4
    chance = 1.0 / n_angle_classes

    col_w = 11
    header = f"{'':14}" + "".join(f"n_bins={b}".rjust(col_w) for b in sweep_bins)
    sep = "-" * len(header)
    lines = [
        f"\nAngle held-out accuracy by phase and temporal resolution "
        f"(angles={angles}, chance={chance:.2f}):",
        sep, header, sep,
    ]
    for phase in PHASE_NAMES:
        row = f"{phase + ':':<14}"
        for b in sweep_bins:
            summary = sweep_summaries.get(phase, {}).get(b)
            if summary and summary.get("heldout_accuracy"):
                acc = summary["heldout_accuracy"].get("angle", float("nan"))
                row += f"{acc:.4f}".rjust(col_w)
            else:
                row += "N/A".rjust(col_w)
        lines.append(row)
    lines.append(sep)
    lines.append(f"Chance level: {chance:.4f} ({n_angle_classes} classes)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined attention plot (all phases, side by side) — normal mode
# ---------------------------------------------------------------------------

def plot_combined_attention(summaries: dict[str, dict], save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping combined attention plot.")
        return

    phases = [p for p in PHASE_NAMES if p in summaries]
    if not phases:
        return

    x = np.arange(len(AREA_NAMES))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, phase in enumerate(phases):
        imp = summaries[phase]["area_importance_heldout"]
        values = [imp[name] for name in AREA_NAMES]
        bars = ax.bar(x + i * width, values, width, label=phase, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xlabel("Brain area")
    ax.set_ylabel("Normalized attention received")
    ax.set_title("Brain area importance by movement phase\n(held-out test, last attention layer)")
    ax.set_xticks(x + width * (len(phases) - 1) / 2)
    ax.set_xticklabels(AREA_NAMES)
    ax.legend(title="Phase")
    ax.set_ylim(0, 0.65)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined attention plot to {save_path}")


def plot_combined_angle_accuracy(
    summaries: dict[str, dict],
    save_path: Path,
    n_angle_classes: int = 4,
) -> None:
    """Bar chart of angle accuracy (seen + heldout) across phases."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    phases = [p for p in PHASE_NAMES if p in summaries]
    if not phases:
        return

    seen_accs = [summaries[p]["seen_accuracy"]["angle"] for p in phases]
    held_accs = [summaries[p]["heldout_accuracy"]["angle"] for p in phases]
    p_vals = [summaries[p]["p_values"]["angle"] for p in phases]

    x = np.arange(len(phases))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, seen_accs, width, label="Seen test", color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + width / 2, held_accs, width, label="Held-out test", color="#DD8452", alpha=0.85)

    for bar, p in zip(b2, p_vals):
        sig = "*" if p < 0.05 else "n.s."
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"p={p:.3f}\n{sig}",
            ha="center", va="bottom", fontsize=8,
        )

    chance = 1.0 / n_angle_classes
    ax.axhline(chance, linestyle="--", color="gray", linewidth=1,
               label=f"Chance ({n_angle_classes} classes)")
    ax.set_xlabel("Movement phase")
    ax.set_ylabel("Angle accuracy")
    ax.set_title("Angle classification accuracy by phase specialist")
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved angle accuracy comparison to {save_path}")


# ---------------------------------------------------------------------------
# Per-channel comparison table
# ---------------------------------------------------------------------------

def _print_per_channel_comparison(
    pc_summaries: dict[str, dict],
    out_base: Path,
    angles: str,
) -> None:
    """Print and save per-channel vs. n_bins=1 angle accuracy comparison (seen + held-out)."""
    n_angle_classes = 2 if angles == "binary" else 4
    chance = 1.0 / n_angle_classes
    C, V = 12, 13
    header = (
        f"{'Phase':<{C}} | "
        f"{'seen nb1':>{V}} | {'seen pc':>{V}} | "
        f"{'held nb1':>{V}} | {'held pc':>{V}} | "
        f"{'delta(held)':>{V}}"
    )
    sep = "-" * len(header)
    lines = [
        f"\nAngle accuracy: per-channel vs. n_bins=1  (angles={angles}, chance={chance:.2f})",
        sep, header, sep,
    ]
    for phase in PHASE_NAMES:
        nbins1_dir = out_base / f"specialist_{phase}_nbins1"
        nb1_seen: float | None = None
        nb1_held: float | None = None
        if (nbins1_dir / "summary.json").exists():
            try:
                nb1 = json.loads((nbins1_dir / "summary.json").read_text(encoding="utf-8"))
                nb1_seen = nb1["seen_accuracy"]["angle"]
                nb1_held = nb1["heldout_accuracy"]["angle"]
            except Exception:
                pass
        pc_s = pc_summaries.get(phase, {})
        pc_seen: float | None = pc_s.get("seen_accuracy", {}).get("angle")
        pc_held: float | None = pc_s.get("heldout_accuracy", {}).get("angle")

        def _f(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "N/A"

        delta = f"{pc_held - nb1_held:+.4f}" if (pc_held is not None and nb1_held is not None) else "N/A"
        lines.append(
            f"{phase:<{C}} | "
            f"{_f(nb1_seen):>{V}} | {_f(pc_seen):>{V}} | "
            f"{_f(nb1_held):>{V}} | {_f(pc_held):>{V}} | "
            f"{delta:>{V}}"
        )
    lines += [sep, f"Chance level: {chance:.4f} ({n_angle_classes} classes)"]
    table = "\n".join(lines)
    print("\n\n" + "=" * 70)
    print("  PER-CHANNEL vs. N_BINS=1 COMPARISON")
    print("=" * 70)
    print(table)
    save_path = out_base / "specialist_per_channel_vs_nbins1.txt"
    save_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved comparison to {save_path}")


# ---------------------------------------------------------------------------
# Summary tables — no_heldout mode
# ---------------------------------------------------------------------------

def format_no_heldout_table(summaries: dict[str, dict], angles: str = "all") -> str:
    """Standard 80-20 split accuracy table: one test column per head."""
    n_angle_classes = 2 if angles == "binary" else 4
    C, V = 12, 12
    header = (
        f"{'Phase':<{C}} | "
        f"{'Grip(test)':>{V}} | {'Hand(test)':>{V}} | {'Angle(test)':>{V}}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for phase in PHASE_NAMES:
        if phase not in summaries:
            lines.append(f"{phase:<{C}} | {'(not run)':>{V}}")
            continue
        acc = summaries[phase]["test_accuracy"]
        lines.append(
            f"{phase:<{C}} | "
            f"{acc['grip']:>{V}.4f} | {acc['hand']:>{V}.4f} | {acc['angle']:>{V}.4f}"
        )
    lines += [
        sep,
        f"Chance level: 0.5 (grip), 0.5 (hand), {1/n_angle_classes:.2f} (angle, {n_angle_classes} classes)",
    ]
    return "\n".join(lines)


def _print_no_heldout_comparison(
    nh_summaries: dict[str, dict],
    out_base: Path,
) -> None:
    """Compare no_heldout test accuracy vs per-channel held-out seen accuracy."""
    C, V = 12, 18
    header = (
        f"{'Phase':<{C}} | "
        f"{'no_heldout(test)':>{V}} | {'heldout_run(seen)':>{V}} | {'delta':>{V}}"
    )
    sep = "-" * len(header)
    lines = [
        "\nAngle accuracy: no-heldout (standard split) vs per-channel held-out (seen combinations only):",
        sep, header, sep,
    ]
    for phase in PHASE_NAMES:
        nh_acc: float | None = nh_summaries.get(phase, {}).get("test_accuracy", {}).get("angle")
        pc_acc: float | None = None
        pc_path = out_base / f"specialist_{phase}_per_channel" / "summary.json"
        if pc_path.exists():
            try:
                pc_acc = json.loads(pc_path.read_text(encoding="utf-8"))["seen_accuracy"]["angle"]
            except Exception:
                pass

        def _f(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "N/A"

        delta = f"{nh_acc - pc_acc:+.4f}" if (nh_acc is not None and pc_acc is not None) else "N/A"
        lines.append(
            f"{phase:<{C}} | {_f(nh_acc):>{V}} | {_f(pc_acc):>{V}} | {delta:>{V}}"
        )
    lines += [
        sep,
        "If delta is large: angle signal exists but fails to generalize compositionally.",
        "If delta is small: angle signal is weak regardless of generalization requirement.",
    ]
    table = "\n".join(lines)
    print("\n\n" + "=" * 70)
    print("  NO-HELDOUT vs PER-CHANNEL HELD-OUT COMPARISON")
    print("=" * 70)
    print(table)
    save_path = out_base / "specialist_no_heldout_vs_per_channel.txt"
    save_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved comparison to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.no_heldout and args.heldout is not None:
        print("WARNING: --heldout ignored when --no_heldout is set")
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    n_angle_classes = 2 if args.angles == "binary" else 4

    if args.per_channel and args.sweep_bins is not None:
        sys.exit("Error: --per_channel and --sweep_bins are mutually exclusive.")

    # ------------------------------------------------------------------
    # Sweep mode: iterate over n_bins × phases
    # ------------------------------------------------------------------
    if args.sweep_bins is not None:
        sweep_summaries: dict[str, dict[int | str, dict]] = {}

        for n_bins in args.sweep_bins:
            for phase in args.phases:
                out_dir = run_phase_specialist_sweep(phase, n_bins, args)
                summary = load_summary(out_dir)
                sweep_summaries.setdefault(phase, {})[n_bins] = summary

        sweep_table = format_sweep_table(sweep_summaries, args.sweep_bins, args.angles)

        print("\n\n" + "=" * 80)
        print("  SWEEP SUMMARY — Angle held-out accuracy by phase × temporal resolution")
        print("=" * 80)
        print(sweep_table)

        sweep_table_path = out_base / "specialist_sweep_table.txt"
        sweep_table_path.write_text(sweep_table + "\n", encoding="utf-8")
        print(f"\nSaved sweep table to {sweep_table_path}")

        sweep_json_path = out_base / "specialist_sweep_summaries.json"
        # JSON keys must be strings; convert n_bins values
        serialisable = {
            phase: {str(b): s for b, s in phase_dict.items()}
            for phase, phase_dict in sweep_summaries.items()
        }
        sweep_json_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        print(f"Saved sweep summaries to {sweep_json_path}")

        print("\nDone.")
        return

    # ------------------------------------------------------------------
    # Normal mode: run each selected phase once
    # ------------------------------------------------------------------
    summaries: dict[str, dict] = {}

    for phase in args.phases:
        out_dir = run_phase_specialist(phase, args)
        summaries[phase] = load_summary(out_dir)

    # ------------------------------------------------------------------
    # Print and save the thesis summary table
    # ------------------------------------------------------------------
    combined_json_path = out_base / "specialist_all_summaries.json"
    combined_json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Saved combined summaries to {combined_json_path}")

    if args.no_heldout:
        nh_table = format_no_heldout_table(summaries, angles=args.angles)
        print("\n\n" + "=" * 80)
        print("  PHASE SPECIALIST SUMMARY — Standard 80-20 split, all 16 combinations in train and test")
        print("=" * 80)
        print(nh_table)
        table_path = out_base / "specialist_no_heldout_summary_table.txt"
        table_path.write_text(nh_table + "\n", encoding="utf-8")
        print(f"\nSaved summary table to {table_path}")
        _print_no_heldout_comparison(summaries, out_base)
    else:
        acc_table = format_accuracy_table(summaries, angles=args.angles)
        attn_table = format_attention_table(summaries)

        print("\n\n" + "=" * 80)
        print("  PHASE SPECIALIST SUMMARY")
        print("=" * 80)
        print(acc_table)
        print(attn_table)

        full_text = acc_table + "\n" + attn_table + "\n"
        table_path = out_base / "specialist_summary_table.txt"
        table_path.write_text(full_text, encoding="utf-8")
        print(f"\nSaved summary table to {table_path}")

        if args.per_channel:
            _print_per_channel_comparison(summaries, out_base, args.angles)

        if not args.no_plot:
            plot_combined_attention(summaries, out_base / "specialist_combined_attention.png")
            plot_combined_angle_accuracy(
                summaries,
                out_base / "specialist_angle_accuracy.png",
                n_angle_classes=n_angle_classes,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
