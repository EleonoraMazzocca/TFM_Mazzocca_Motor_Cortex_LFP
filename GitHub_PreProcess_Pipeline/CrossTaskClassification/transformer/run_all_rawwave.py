"""Run rawwave phase specialists sequentially and print summary tables.

Usage:
    # Dry run — verify pipeline end to end
    python run_all_rawwave.py --broadband_data_dir /path/to/bb --phases reach --dry_run --no_heldout --no_plot

    # Step 1 — standard split, reach only
    python run_all_rawwave.py --broadband_data_dir /path/to/bb --phases reach --no_heldout

    # Step 2 — heldout, all phases (only if Step 1 angle > 0.40)
    python run_all_rawwave.py --broadband_data_dir /path/to/bb
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE   = Path(__file__).resolve().parent
_SCRIPT = _HERE / "run_rawwave.py"

PHASE_NAMES = ["prereach", "reach", "grasp"]
AREA_NAMES  = ["PMvR", "M1", "PMdR", "PMdL"]
HEAD_NAMES  = ["grip", "hand", "angle"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all rawwave phase specialists and produce summary tables."
    )
    p.add_argument("--broadband_data_dir", type=str, required=True)
    p.add_argument("--phases", nargs="+", choices=PHASE_NAMES, default=PHASE_NAMES)
    split_group = p.add_mutually_exclusive_group()
    split_group.add_argument("--heldout", type=str, default="precision_right_135",
                             help="Held-out combination, e.g. 'precision_right_135'.")
    split_group.add_argument("--no_heldout", action="store_true",
                             help="Standard stratified 80/10/10 split; no combination held out.")
    p.add_argument("--norm_mode",       choices=["per_channel", "per_timepoint"],
                   default="per_channel")
    p.add_argument("--n_permutations",   type=int,   default=100)
    p.add_argument("--n_perm_noheldout", type=int,   default=20)
    p.add_argument("--epochs",           type=int,   default=40)
    p.add_argument("--patience",         type=int,   default=8)
    p.add_argument("--dropout",          type=float, default=0.5)
    p.add_argument("--d_model",          type=int,   default=32)
    p.add_argument("--n_heads",          type=int,   default=4)
    p.add_argument("--n_layers",         type=int,   default=2)
    p.add_argument("--feedforward_dim",  type=int,   default=64)
    p.add_argument("--weight_decay",     type=float, default=1e-3)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--cache_dir",        type=str,   default="/tmp/lfp_rawwave_cache")
    p.add_argument("--out_base",         type=str,   default=str(_HERE / "results"))
    p.add_argument("--device",           type=str,   default=None)
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
        "--norm_mode",          args.norm_mode,
        "--n_permutations",     str(args.n_permutations),
        "--n_perm_noheldout",   str(args.n_perm_noheldout),
        "--epochs",             str(args.epochs),
        "--patience",           str(args.patience),
        "--dropout",            str(args.dropout),
        "--d_model",            str(args.d_model),
        "--n_heads",            str(args.n_heads),
        "--n_layers",           str(args.n_layers),
        "--feedforward_dim",    str(args.feedforward_dim),
        "--weight_decay",       str(args.weight_decay),
        "--lr",                 str(args.lr),
        "--batch_size",         str(args.batch_size),
        "--seed",               str(args.seed),
        "--cache_dir",          args.cache_dir,
        "--out_dir",            str(out_dir),
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
        out_dir = Path(args.out_base) / f"rawwave_{phase}_no_heldout"
        label   = "no_heldout"
    else:
        out_dir = Path(args.out_base) / f"rawwave_{phase}"
        label   = args.heldout
    cmd = _build_cmd(phase, args, out_dir)
    print(f"\n{'='*70}")
    print(f"  Launching {phase.upper()} rawwave specialist  ({label})")
    print(f"{'='*70}")
    subprocess.run(cmd, check=True)
    return out_dir


def load_summary(out_dir: Path) -> dict:
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def format_accuracy_table(summaries: dict[str, dict], no_heldout: bool = False) -> str:
    chance = 0.25  # 4 angle classes
    C, V   = 9, 10

    if no_heldout:
        header = (
            f"{'Phase':<{C}} | "
            f"{'Grip(test)':>{V}} | {'Hand(test)':>{V}} | {'Angle(test)':>{V}}"
        )
        sep   = "-" * len(header)
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
        P      = 6
        header = (
            f"{'Phase':<{C}} |"
            f"{'Grip(seen)':>{V}}|{'Grip(held)':>{V}}|{'Grip p':>{P}}|"
            f"{'Hand(seen)':>{V}}|{'Hand(held)':>{V}}|{'Hand p':>{P}}|"
            f"{'Angle(seen)':>{V+1}}|{'Angle(held)':>{V+1}}|{'Angle p':>{P}}"
        )
        sep   = "-" * len(header)
        lines = [sep, header, sep]
        for phase in PHASE_NAMES:
            if phase not in summaries:
                lines.append(f"{phase:<{C}} | {'(not run)'}")
                continue
            s    = summaries[phase]
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


def format_saliency_area_table(summaries: dict[str, dict]) -> str:
    W          = 7
    col_header = "".join(f"{a:>{W}}" for a in AREA_NAMES)
    lines      = [
        f"\nArea importance (input gradient saliency, angle head, seen test):",
        f"  {'Phase':<10s}{col_header}",
        "  " + "-" * (10 + W * len(AREA_NAMES)),
    ]
    for phase in PHASE_NAMES:
        if phase not in summaries:
            continue
        si = summaries[phase].get("channel_importance_gradient_saliency")
        if si is None:
            continue
        angle_imp = si.get("angle", {}).get("importance")
        if angle_imp is None:
            continue
        row = f"  {phase:<10s}" + "".join(f"{v:>{W}.3f}" for v in angle_imp)
        lines.append(row)
    return "\n".join(lines)


def _print_comparison_3way(summaries: dict[str, dict], out_base: Path) -> None:
    """Compare rawwave vs multiband vs per-channel MU specialist (angle accuracy)."""
    C, V   = 9, 10
    header = (
        f"{'Phase':<{C}} | "
        f"{'raw(seen)':>{V}} | {'mb(seen)':>{V}} | {'pc(seen)':>{V}} | "
        f"{'raw(held)':>{V}} | {'mb(held)':>{V}} | {'pc(held)':>{V}}"
    )
    sep   = "-" * len(header)
    lines = [
        "\nAngle accuracy: rawwave vs multiband vs MU per-channel:",
        sep, header, sep,
    ]
    for phase in PHASE_NAMES:
        rw_s = rw_h = mb_s = mb_h = pc_s = pc_h = None

        if phase in summaries:
            s    = summaries[phase]
            # heldout mode uses seen_accuracy; no_heldout mode uses test_accuracy as proxy
            rw_s = (s.get("seen_accuracy") or s.get("test_accuracy") or {}).get("angle")
            rw_h = s.get("heldout_accuracy", {}).get("angle")

        mb_path = out_base / f"multiband_{phase}" / "summary.json"
        if mb_path.exists():
            try:
                mb   = json.loads(mb_path.read_text(encoding="utf-8"))
                mb_s = mb.get("seen_accuracy",    {}).get("angle")
                mb_h = mb.get("heldout_accuracy", {}).get("angle")
            except Exception:
                pass

        pc_path = out_base / f"specialist_{phase}_per_channel" / "summary.json"
        if pc_path.exists():
            try:
                pc   = json.loads(pc_path.read_text(encoding="utf-8"))
                pc_s = pc.get("seen_accuracy",    {}).get("angle")
                pc_h = pc.get("heldout_accuracy", {}).get("angle")
            except Exception:
                pass

        def _f(v) -> str:
            return f"{v:.4f}" if v is not None else "N/A"

        lines.append(
            f"{phase:<{C}} | "
            f"{_f(rw_s):>{V}} | {_f(mb_s):>{V}} | {_f(pc_s):>{V}} | "
            f"{_f(rw_h):>{V}} | {_f(mb_h):>{V}} | {_f(pc_h):>{V}}"
        )
    lines.append(sep)
    table = "\n".join(lines)
    print("\n\n" + "=" * 70)
    print("  RAWWAVE vs MULTIBAND vs PER-CHANNEL COMPARISON")
    print("=" * 70)
    print(table)
    save_path = out_base / "rawwave_vs_multiband_vs_per_channel.txt"
    save_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved comparison to {save_path}")


def _print_decision_guide(summaries: dict[str, dict]) -> None:
    print("\nDecision guide (no-heldout angle accuracy):")
    print("  < 0.32:        signal absent — skip heldout, move to cVAE")
    print("  0.32 - 0.40:   ambiguous — rerun with --norm_mode per_timepoint")
    print("                 if still < 0.40, skip heldout")
    print("  > 0.40:        signal present — run heldout all phases")
    if summaries:
        print()
        for phase in PHASE_NAMES:
            if phase not in summaries:
                continue
            acc = summaries[phase].get("test_accuracy", {}).get("angle")
            if acc is None:
                continue
            if acc < 0.32:
                verdict = "SIGNAL ABSENT"
            elif acc < 0.40:
                verdict = "AMBIGUOUS"
            else:
                verdict = "SIGNAL PRESENT"
            print(f"  {phase:<10s}: angle={acc:.4f}  → {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for phase in args.phases:
        out_dir = run_phase(phase, args)
        summaries[phase] = load_summary(out_dir)

    acc_table     = format_accuracy_table(summaries, no_heldout=args.no_heldout)
    saliency_table = format_saliency_area_table(summaries)

    print("\n\n" + "=" * 80)
    print("  RAWWAVE SPECIALIST SUMMARY")
    print("=" * 80)
    print(acc_table)
    print(saliency_table)

    if args.no_heldout:
        _print_decision_guide(summaries)

    full_text  = acc_table + "\n" + saliency_table + "\n"
    label      = "no_heldout" if args.no_heldout else "heldout"
    table_path = out_base / f"rawwave_summary_table_{label}.txt"
    table_path.write_text(full_text, encoding="utf-8")
    print(f"\nSaved summary table to {table_path}")

    combined_json = out_base / f"rawwave_all_summaries_{label}.json"
    combined_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Saved combined summaries to {combined_json}")

    if not args.no_heldout:
        _print_comparison_3way(summaries, out_base)

    print("\nDone.")


if __name__ == "__main__":
    main()
