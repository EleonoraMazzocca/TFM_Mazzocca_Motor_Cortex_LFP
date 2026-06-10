"""Run all three multiband phase specialists sequentially and print summary tables.

Usage:
    python run_all_multiband.py --broadband_data_dir /path/to/broadband
    python run_all_multiband.py --broadband_data_dir /path/to/bb --phases reach
    python run_all_multiband.py --broadband_data_dir /path/to/bb --no_heldout --phases reach
    python run_all_multiband.py --broadband_data_dir /path/to/bb --dry_run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "run_multiband.py"

PHASE_NAMES = ["prereach", "reach", "grasp"]
AREA_NAMES  = ["PMvR", "M1", "PMdR", "PMdL"]
HEAD_NAMES  = ["grip", "hand", "angle"]
BAND_ABBREV = ["beta", "low_g", "high_g", "l_rip", "h_rip", "MU"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all multiband phase specialists and produce summary tables."
    )
    p.add_argument("--broadband_data_dir", type=str, required=True)
    p.add_argument("--phases", nargs="+", choices=PHASE_NAMES, default=PHASE_NAMES)
    split_group = p.add_mutually_exclusive_group()
    split_group.add_argument("--heldout", type=str, default="precision_right_135",
                             help="Held-out combination, e.g. 'precision_right_135'.")
    split_group.add_argument("--no_heldout", action="store_true",
                             help="Standard stratified 80/10/10 split; no combination held out.")
    p.add_argument("--n_permutations", type=int, default=100)
    p.add_argument("--epochs",   type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--dropout",  type=float, default=0.4)
    p.add_argument("--d_model",  type=int, default=64)
    p.add_argument("--n_heads",  type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--feedforward_dim", type=int, default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--cache_dir",  type=str,   default="/tmp/lfp_multiband_cache")
    p.add_argument("--out_base",   type=str,   default=str(_HERE / "results"))
    p.add_argument("--device",     type=str,   default=None)
    p.add_argument("--no_plot",  action="store_true")
    p.add_argument("--dry_run",  action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------

def _build_cmd(phase: str, args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable, str(_SCRIPT),
        "--phase", phase,
        "--broadband_data_dir", args.broadband_data_dir,
        "--n_permutations", str(args.n_permutations),
        "--epochs",   str(args.epochs),
        "--patience", str(args.patience),
        "--dropout",  str(args.dropout),
        "--d_model",  str(args.d_model),
        "--n_heads",  str(args.n_heads),
        "--n_layers", str(args.n_layers),
        "--feedforward_dim", str(args.feedforward_dim),
        "--lr",         str(args.lr),
        "--batch_size", str(args.batch_size),
        "--seed",       str(args.seed),
        "--cache_dir",  args.cache_dir,
        "--out_dir",    str(out_dir),
    ]
    if not args.no_heldout:
        cmd += ["--heldout", args.heldout]
    if args.no_heldout:
        cmd.append("--no_heldout")
    if args.device:
        cmd += ["--device", args.device]
    if args.no_plot:
        cmd.append("--no_plot")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def run_phase(phase: str, args: argparse.Namespace) -> Path:
    if args.no_heldout:
        out_dir = Path(args.out_base) / f"multiband_{phase}_no_heldout"
        label = "no_heldout"
    else:
        out_dir = Path(args.out_base) / f"multiband_{phase}"
        label = args.heldout
    cmd = _build_cmd(phase, args, out_dir)
    print(f"\n{'='*70}")
    print(f"  Launching {phase.upper()} multiband specialist  ({label})")
    print(f"{'='*70}")
    subprocess.run(cmd, check=True)
    return out_dir


def load_summary(out_dir: Path) -> dict:
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def format_accuracy_table(summaries: dict[str, dict], no_heldout: bool = False) -> str:
    n_angle_classes = 4
    chance = 1.0 / n_angle_classes
    C, V = 9, 10

    if no_heldout:
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
        lines += [sep, f"Chance level: 0.5 (grip), 0.5 (hand), {chance:.2f} (angle, 4 classes)"]
    else:
        P = 6
        header = (
            f"{'Phase':<{C}} |"
            f"{'Grip(seen)':>{V}}|{'Grip(held)':>{V}}|{'Grip p':>{P}}|"
            f"{'Hand(seen)':>{V}}|{'Hand(held)':>{V}}|{'Hand p':>{P}}|"
            f"{'Angle(seen)':>{V+1}}|{'Angle(held)':>{V+1}}|{'Angle p':>{P}}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for phase in PHASE_NAMES:
            if phase not in summaries:
                lines.append(f"{phase:<{C}} | {'(not run)'}")
                continue
            s = summaries[phase]
            seen = s["seen_accuracy"]
            held = s["heldout_accuracy"]
            pv   = s.get("p_values") or {}
            def _p(h: str) -> str:
                val = pv.get(h)
                return f"{val:.3f}" if val is not None else "N/A"
            lines.append(
                f"{phase:<{C}} |"
                f"{seen['grip']:>{V}.4f}|{held['grip']:>{V}.4f}|{_p('grip'):>{P}}|"
                f"{seen['hand']:>{V}.4f}|{held['hand']:>{V}.4f}|{_p('hand'):>{P}}|"
                f"{seen['angle']:>{V+1}.4f}|{held['angle']:>{V+1}.4f}|{_p('angle'):>{P}}"
            )
        lines += [
            sep,
            "* p < 0.05 (permutation test)",
            f"angle chance level: {chance:.2f} (4 classes)",
        ]

    return "\n".join(lines)


def format_band_importance_table(summaries: dict[str, dict]) -> str:
    W = 7
    col_header = "".join(f"{a:>{W}}" for a in BAND_ABBREV)
    lines = [
        f"\nBand importance — ANGLE head (GradCAM, seen test):",
        f"  {'Phase':<10s}{col_header}",
        "  " + "-" * (10 + W * len(BAND_ABBREV)),
    ]
    for phase in PHASE_NAMES:
        if phase not in summaries:
            continue
        bi = summaries[phase].get("band_importance_gradcam")
        if bi is None:
            continue
        angle_imp = bi.get("angle", {}).get("importance")
        if angle_imp is None:
            continue
        row = f"  {phase:<10s}" + "".join(f"{v:>{W}.3f}" for v in angle_imp)
        lines.append(row)
    return "\n".join(lines)


def _print_comparison(summaries: dict[str, dict], out_base: Path) -> None:
    """Compare multiband vs per-channel MU specialist."""
    C, V = 9, 10
    header = (
        f"{'Phase':<{C}} | "
        f"{'mb(seen)':>{V}} | {'pc(seen)':>{V}} | "
        f"{'mb(held)':>{V}} | {'pc(held)':>{V}}"
    )
    sep = "-" * len(header)
    lines = [
        "\nAngle accuracy: multiband vs MU per-channel:",
        sep, header, sep,
    ]
    for phase in PHASE_NAMES:
        mb_s = mb_h = pc_s = pc_h = None
        if phase in summaries:
            mb_s = summaries[phase].get("seen_accuracy",    {}).get("angle")
            mb_h = summaries[phase].get("heldout_accuracy", {}).get("angle")

        pc_path = out_base / f"specialist_{phase}_per_channel" / "summary.json"
        if pc_path.exists():
            try:
                pc = json.loads(pc_path.read_text(encoding="utf-8"))
                pc_s = pc.get("seen_accuracy",    {}).get("angle")
                pc_h = pc.get("heldout_accuracy", {}).get("angle")
            except Exception:
                pass

        def _f(v) -> str:
            return f"{v:.4f}" if v is not None else "N/A"

        lines.append(
            f"{phase:<{C}} | "
            f"{_f(mb_s):>{V}} | {_f(pc_s):>{V}} | "
            f"{_f(mb_h):>{V}} | {_f(pc_h):>{V}}"
        )
    lines.append(sep)
    table = "\n".join(lines)
    print("\n\n" + "=" * 70)
    print("  MULTIBAND vs PER-CHANNEL MU COMPARISON")
    print("=" * 70)
    print(table)
    save_path = out_base / "multiband_vs_per_channel.txt"
    save_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved comparison to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.no_heldout and args.heldout != "precision_right_135":
        print("WARNING: --heldout ignored when --no_heldout is set")

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for phase in args.phases:
        out_dir = run_phase(phase, args)
        summaries[phase] = load_summary(out_dir)

    acc_table  = format_accuracy_table(summaries, no_heldout=args.no_heldout)
    band_table = format_band_importance_table(summaries)

    print("\n\n" + "=" * 80)
    print("  MULTIBAND SPECIALIST SUMMARY")
    print("=" * 80)
    print(acc_table)
    print(band_table)

    full_text = acc_table + "\n" + band_table + "\n"
    label = "no_heldout" if args.no_heldout else "heldout"
    table_path = out_base / f"multiband_summary_table_{label}.txt"
    table_path.write_text(full_text, encoding="utf-8")
    print(f"\nSaved summary table to {table_path}")

    combined_json = out_base / f"multiband_all_summaries_{label}.json"
    combined_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Saved combined summaries to {combined_json}")

    if not args.no_heldout:
        _print_comparison(summaries, out_base)

    print("\nDone.")


if __name__ == "__main__":
    main()
