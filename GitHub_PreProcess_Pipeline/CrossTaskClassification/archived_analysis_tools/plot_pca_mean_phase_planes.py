"""
Comparison suite: mean PCA trajectories across multiple 2D planes.

This companion script leaves `plot_pca_mean_phase_grid.py` untouched and adds
an alternate visualization where each phase is shown in three projections:
- PC1 vs PC2
- PC1 vs PC3
- PC2 vs PC3

The trajectory construction is the same as the existing mean-trajectory script:
1) load trials for group A and group B
2) fit PCA jointly on A+B for each phase
3) project each trial, center at the first time point, then average over trials
4) plot the mean trajectories for both groups
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
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


def add_gradient_line(ax, z2d: np.ndarray, cmap_name: str, lw: float = 2.2) -> None:
    if z2d.shape[0] < 2:
        return
    points = z2d.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap_name), norm=Normalize(0, len(segments)))
    lc.set_array(np.arange(len(segments)))
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    ax.scatter(z2d[0, 0], z2d[0, 1], marker="*", s=58, c="black")
    ax.scatter(z2d[-1, 0], z2d[-1, 1], marker="X", s=48, c="black")


def mean_centered_trajectory(trials: np.ndarray, phase: int, pca: PCA) -> np.ndarray:
    z_list = []
    for i in range(trials.shape[0]):
        z = pca.transform(trials[i, phase].T)
        z_list.append(z - z[0])
    return np.mean(np.stack(z_list, axis=0), axis=0)


def smooth_traj(z: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return z
    out = z.copy()
    for pc_idx in range(min(out.shape[1], 3)):
        out[:, pc_idx] = gaussian_filter1d(out[:, pc_idx], sigma=sigma, mode="nearest")
    return out


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


def run_comparison_plot(
    comp: Comparison,
    out_dir: str,
    n_components: int = 3,
    smooth_sigma: float = 2.0,
    max_trials_per_class: int = 0,
    seed: int = 42,
) -> str:
    if n_components < 3:
        raise ValueError("This plot needs at least 3 PCA components.")

    a_trials = load_trials(comp.group_a_classes, max_trials_per_class=max_trials_per_class, seed=seed)
    b_trials = load_trials(comp.group_b_classes, max_trials_per_class=max_trials_per_class, seed=seed)

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    phase_cmaps = {"A": "Blues", "B": "Oranges"}
    phase_legend_colors = {"A": "#1f77b4", "B": "#ff7f0e"}
    plane_defs: List[Tuple[Tuple[int, int], str, str]] = [
        ((0, 1), "PC1 (centered)", "PC2 (centered)"),
        ((0, 2), "PC1 (centered)", "PC3 (centered)"),
        ((1, 2), "PC2 (centered)", "PC3 (centered)"),
    ]

    for ph in [0, 1, 2]:
        x_ab = np.concatenate(
            [
                np.transpose(a_trials[:, ph], (0, 2, 1)).reshape(-1, a_trials.shape[2]),
                np.transpose(b_trials[:, ph], (0, 2, 1)).reshape(-1, b_trials.shape[2]),
            ],
            axis=0,
        )
        pca = PCA(n_components=n_components, random_state=42).fit(x_ab)

        a_mean = smooth_traj(mean_centered_trajectory(a_trials, ph, pca), smooth_sigma)
        b_mean = smooth_traj(mean_centered_trajectory(b_trials, ph, pca), smooth_sigma)

        for col_idx, (pc_pair, xlabel, ylabel) in enumerate(plane_defs):
            ax = axes[ph, col_idx]
            add_gradient_line(ax, a_mean[:, list(pc_pair)], phase_cmaps["A"], lw=2.2)
            add_gradient_line(ax, b_mean[:, list(pc_pair)], phase_cmaps["B"], lw=2.2)
            ax.autoscale()
            if ph == 0:
                ax.set_title(f"Plane {pc_pair[0] + 1}-{pc_pair[1] + 1}")
            if col_idx == 0:
                ax.text(
                    -0.24,
                    0.5,
                    f"Phase {ph}",
                    rotation=90,
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            try:
                ax.set_aspect("equal", adjustable="box")
            except Exception:
                pass

    handles = [
        plt.Line2D([0], [0], color=phase_legend_colors["A"], lw=2.5, label=comp.group_a_label),
        plt.Line2D([0], [0], color=phase_legend_colors["B"], lw=2.5, label=comp.group_b_label),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.suptitle(
        f"{comp.name} | Mean centered trajectories across PCA planes | "
        f"Gaussian sigma={smooth_sigma} | A n={a_trials.shape[0]} vs B n={b_trials.shape[0]}",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    out_path = os.path.join(out_dir, f"{sanitize(comp.name)}.png")
    ensure_dir(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run PCA mean trajectory comparisons across multiple 2D planes.")
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
    parser.add_argument("--max-trials-per-class", type=int, default=80, help="Cap samples per class to reduce load.")
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--n-components", type=int, default=3, help="Must be at least 3.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = "GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/pca_mean_planes"
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
            )
            print(f"[{i:02d}/{len(comparisons)}] {comp.name} -> {out_path}")


if __name__ == "__main__":
    main()
