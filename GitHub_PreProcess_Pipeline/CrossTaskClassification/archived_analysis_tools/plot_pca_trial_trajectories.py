"""
Comparison suite: PCA trajectories with individual trials visible.

This script is a companion to `plot_pca_mean_phase_grid.py`.
It keeps the original mean-trajectory workflow untouched and adds a
separate view where all centered trial trajectories can be inspected.

For each comparison, this script:
1) Pools trials from group A classes and group B classes.
2) For each phase (0, 1, 2), fits PCA jointly on A+B data (shared basis).
3) Projects each trial into PCA space and centers it at its first time point.
4) Plots all trial trajectories with low alpha, plus an optional mean overlay.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from data_paths import CLASS_FILES
except ImportError:
    from GitHub_PreProcess_Pipeline.CrossTaskClassification.data_paths import CLASS_FILES


@dataclass
class Comparison:
    name: str
    group_a_label: str
    group_a_classes: List[str]
    group_b_label: str
    group_b_classes: List[str]


def ensure_dir(path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)


def load_trials(classes: List[str], max_trials_per_class: int = 0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for c in classes:
        arr = np.load(CLASS_FILES[c], mmap_mode="r")
        if max_trials_per_class > 0 and arr.shape[0] > max_trials_per_class:
            idx = np.sort(rng.choice(arr.shape[0], size=max_trials_per_class, replace=False))
            part = np.array(arr[idx], copy=True).astype(np.float32, copy=False)
        else:
            part = np.array(arr, copy=True).astype(np.float32, copy=False)
        chunks.append(part)
    return np.concatenate(chunks, axis=0)


def smooth_traj(z: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return z
    out = z.copy()
    out[:, 0] = gaussian_filter1d(out[:, 0], sigma=sigma, mode="nearest")
    out[:, 1] = gaussian_filter1d(out[:, 1], sigma=sigma, mode="nearest")
    return out


def centered_trial_trajectories(trials: np.ndarray, phase: int, pca: PCA, smooth_sigma: float) -> np.ndarray:
    z_list = []
    for i in range(trials.shape[0]):
        z = pca.transform(trials[i, phase].T)
        z = z - z[0]
        z_list.append(smooth_traj(z, smooth_sigma))
    return np.stack(z_list, axis=0)


def plot_group_trials(
    ax,
    z_trials: np.ndarray,
    color: str,
    label: str,
    alpha: float,
    linewidth: float,
    show_mean: bool,
) -> None:
    for z in z_trials:
        ax.plot(z[:, 0], z[:, 1], color=color, alpha=alpha, linewidth=linewidth)

    if show_mean:
        z_mean = np.mean(z_trials, axis=0)
        ax.plot(z_mean[:, 0], z_mean[:, 1], color=color, linewidth=2.8, label=label)
        ax.scatter(z_mean[0, 0], z_mean[0, 1], marker="*", s=64, c="black")
        ax.scatter(z_mean[-1, 0], z_mean[-1, 1], marker="X", s=54, c="black")


def sanitize(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("-", "_")
    )


def run_comparison_plot(
    comp: Comparison,
    out_dir: str,
    n_components: int = 3,
    smooth_sigma: float = 2.0,
    max_trials_per_class: int = 0,
    seed: int = 42,
    trial_alpha: float = 0.12,
    trial_linewidth: float = 0.9,
    show_mean: bool = True,
) -> str:
    a_trials = load_trials(comp.group_a_classes, max_trials_per_class=max_trials_per_class, seed=seed)
    b_trials = load_trials(comp.group_b_classes, max_trials_per_class=max_trials_per_class, seed=seed)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    group_colors = {"A": "#1f77b4", "B": "#ff7f0e"}

    for ph in [0, 1, 2]:
        ax = axes[ph]

        x_ab = np.concatenate(
            [
                np.transpose(a_trials[:, ph], (0, 2, 1)).reshape(-1, a_trials.shape[2]),
                np.transpose(b_trials[:, ph], (0, 2, 1)).reshape(-1, b_trials.shape[2]),
            ],
            axis=0,
        )
        pca = PCA(n_components=n_components, random_state=42).fit(x_ab)

        a_z = centered_trial_trajectories(a_trials, ph, pca, smooth_sigma=smooth_sigma)
        b_z = centered_trial_trajectories(b_trials, ph, pca, smooth_sigma=smooth_sigma)

        plot_group_trials(
            ax,
            a_z,
            color=group_colors["A"],
            label=comp.group_a_label,
            alpha=trial_alpha,
            linewidth=trial_linewidth,
            show_mean=show_mean,
        )
        plot_group_trials(
            ax,
            b_z,
            color=group_colors["B"],
            label=comp.group_b_label,
            alpha=trial_alpha,
            linewidth=trial_linewidth,
            show_mean=show_mean,
        )

        ax.set_title(f"Phase {ph}")
        ax.set_xlabel("PC1 (centered)")
        ax.set_ylabel("PC2 (centered)")
        ax.grid(alpha=0.2)
        try:
            ax.set_aspect("equal", adjustable="box")
        except Exception:
            pass

    if show_mean:
        handles = [
            plt.Line2D([0], [0], color=group_colors["A"], lw=2.8, label=comp.group_a_label),
            plt.Line2D([0], [0], color=group_colors["B"], lw=2.8, label=comp.group_b_label),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2)
        layout_rect = [0, 0.07, 1, 0.93]
    else:
        layout_rect = [0, 0.02, 1, 0.93]

    fig.suptitle(
        f"{comp.name} | Individual centered trajectories | "
        f"Gaussian sigma={smooth_sigma} | A n={a_trials.shape[0]} vs B n={b_trials.shape[0]}",
        y=0.98,
    )
    plt.tight_layout(rect=layout_rect)

    out_path = os.path.join(out_dir, f"{sanitize(comp.name)}.png")
    ensure_dir(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def build_comparisons() -> List[Comparison]:
    power_all = [k for k in CLASS_FILES if k.startswith("POWER_")]
    precision_all = [k for k in CLASS_FILES if k.startswith("PRECISION_")]

    power_uni_l = [k for k in CLASS_FILES if k.startswith("POWER_UNIMANUAL_L_")]
    power_uni_r = [k for k in CLASS_FILES if k.startswith("POWER_UNIMANUAL_R_")]
    precision_uni_l = [k for k in CLASS_FILES if k.startswith("PRECISION_UNIMANUAL_L_")]
    precision_uni_r = [k for k in CLASS_FILES if k.startswith("PRECISION_UNIMANUAL_R_")]

    power_uni_l_0 = ["POWER_UNIMANUAL_L_0"]
    power_uni_l_others = ["POWER_UNIMANUAL_L_45", "POWER_UNIMANUAL_L_90", "POWER_UNIMANUAL_L_135"]
    precision_uni_l_0 = ["PRECISION_UNIMANUAL_L_0"]
    precision_uni_l_others = ["PRECISION_UNIMANUAL_L_45", "PRECISION_UNIMANUAL_L_90", "PRECISION_UNIMANUAL_L_135"]

    precision_bi_45 = ["PRECISION_BIMANUAL_45"]
    precision_bi_others = ["PRECISION_BIMANUAL_135", "PRECISION_BIMANUAL_45_135", "PRECISION_BIMANUAL_135_45"]
    precision_bimanual = [k for k in CLASS_FILES if k.startswith("PRECISION_BIMANUAL_")]
    precision_unimanual = [k for k in CLASS_FILES if k.startswith("PRECISION_UNIMANUAL_")]

    comps: List[Comparison] = [
        Comparison("1_power_all_vs_precision_all", "Power all", power_all, "Precision all", precision_all),
        Comparison("2_power_uni_left_vs_precision_uni_left", "Power uni left", power_uni_l, "Precision uni left", precision_uni_l),
        Comparison("3_power_uni_right_vs_precision_uni_right", "Power uni right", power_uni_r, "Precision uni right", precision_uni_r),
        Comparison("4_power_uni_left_vs_power_uni_right", "Power uni left", power_uni_l, "Power uni right", power_uni_r),
        Comparison("5_precision_uni_left_vs_precision_uni_right", "Precision uni left", precision_uni_l, "Precision uni right", precision_uni_r),
        Comparison("6_power_uni_left_0_vs_left_45_90_135", "Power left 0", power_uni_l_0, "Power left 45/90/135", power_uni_l_others),
        Comparison("8_precision_uni_left_0_vs_left_45_90_135", "Precision left 0", precision_uni_l_0, "Precision left 45/90/135", precision_uni_l_others),
        Comparison("10_precision_bimanual_45_45_vs_other_bimanual_angles", "Precision bi 45-45", precision_bi_45, "Precision bi other angles", precision_bi_others),
        Comparison("11_precision_bimanual_vs_precision_unimanual", "Precision bimanual", precision_bimanual, "Precision unimanual", precision_unimanual),
    ]

    for deg in ["0", "45", "90", "135"]:
        comps.append(
            Comparison(
                f"7_power_left_vs_right_{deg}",
                f"Power left {deg}",
                [f"POWER_UNIMANUAL_L_{deg}"],
                f"Power right {deg}",
                [f"POWER_UNIMANUAL_R_{deg}"],
            )
        )

    for deg in ["0", "45", "90", "135"]:
        comps.append(
            Comparison(
                f"9_precision_left_vs_right_{deg}",
                f"Precision left {deg}",
                [f"PRECISION_UNIMANUAL_L_{deg}"],
                f"Precision right {deg}",
                [f"PRECISION_UNIMANUAL_R_{deg}"],
            )
        )

    return comps


def parse_args():
    parser = argparse.ArgumentParser(description="Run PCA trajectory comparisons with individual trial overlays.")
    parser.add_argument(
        "--mode",
        choices=["one", "all"],
        default="one",
        help="one: run a single comparison, all: run the full suite.",
    )
    parser.add_argument(
        "--comparison",
        default="1_power_all_vs_precision_all",
        help="Comparison name to run when --mode one.",
    )
    parser.add_argument("--max-trials-per-class", type=int, default=60, help="Cap samples per class to reduce clutter/load.")
    parser.add_argument("--smooth-sigma", type=float, default=1.5)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial-alpha", type=float, default=0.12, help="Opacity for individual trial trajectories.")
    parser.add_argument("--trial-linewidth", type=float, default=0.9, help="Line width for individual trials.")
    parser.add_argument(
        "--hide-mean",
        action="store_true",
        help="Hide the bold mean trajectory overlay and show only individual trials.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = "GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/pca_trial_trajectories"
    ensure_dir(os.path.join(out_dir, "dummy.txt"))

    comparisons = build_comparisons()
    comps_by_name = {c.name: c for c in comparisons}

    if args.mode == "one":
        if args.comparison not in comps_by_name:
            valid = "\n".join(sorted(comps_by_name.keys()))
            raise ValueError(f"Unknown comparison '{args.comparison}'. Valid names:\n{valid}")
        comp = comps_by_name[args.comparison]
        out_path = run_comparison_plot(
            comp,
            out_dir=out_dir,
            n_components=args.n_components,
            smooth_sigma=args.smooth_sigma,
            max_trials_per_class=args.max_trials_per_class,
            seed=args.seed,
            trial_alpha=args.trial_alpha,
            trial_linewidth=args.trial_linewidth,
            show_mean=not args.hide_mean,
        )
        print(f"[DONE] {comp.name} -> {out_path}")
    else:
        print(f"[INFO] Running {len(comparisons)} comparisons...")
        for i, comp in enumerate(comparisons, start=1):
            out_path = run_comparison_plot(
                comp,
                out_dir=out_dir,
                n_components=args.n_components,
                smooth_sigma=args.smooth_sigma,
                max_trials_per_class=args.max_trials_per_class,
                seed=args.seed,
                trial_alpha=args.trial_alpha,
                trial_linewidth=args.trial_linewidth,
                show_mean=not args.hide_mean,
            )
            print(f"[{i:02d}/{len(comparisons)}] {comp.name} -> {out_path}")


if __name__ == "__main__":
    main()
