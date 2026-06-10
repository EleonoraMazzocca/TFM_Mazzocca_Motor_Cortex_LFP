"""
Lightweight session-level PCA health check.

Goal:
- Compare one suspicious session against several healthy sessions.
- Support multiple preprocessing variants (for example, with and without
  bad-channel rejection) without loading the full dataset at once.
- Keep memory usage modest by:
  - working from per-session structured files
  - capping trials per session/class
  - fitting PCA one class + one phase at a time

Expected inputs:
- Cleaned_Data/structured/data_<SESSION><TAG>.npy
- Cleaned_Data/structured/info_<SESSION><TAG>.pkl

Each data file is expected to have shape:
- (n_trials, 3, channels, time)
"""

import argparse
import os
import pickle
from collections import Counter
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from data_paths import CLEANED_STRUCTURED_DIR
except ImportError:
    from GitHub_PreProcess_Pipeline.CrossTaskClassification.data_paths import CLEANED_STRUCTURED_DIR


PHASE_NAMES = {0: "PREREACH", 1: "REACH", 2: "GRASP"}
DEFAULT_HEALTHY = ["20180607Y", "20180608Y", "20180614Y", "20180618Y", "20180619Y"]
DEFAULT_SUSPICIOUS = "20180615Y"
DEFAULT_PIPELINE_TAGS = ["", "__no_bad_channels"]


def ensure_dir(path):
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)


def parse_info_to_class_names(info):
    n_trials = len(info["Precision/Power"])
    class_names = []
    for i in range(n_trials):
        precision_power = info["Precision/Power"][i]
        uni_bi = info["Unimanual/Bimanual"][i]
        left_angle = str(info["LeftAngle"][i])
        right_angle = str(info["RightAngle"][i])

        if precision_power == 0 and uni_bi == 1:
            if left_angle == "45" and right_angle == "45":
                class_names.append("PRECISION_BIMANUAL_45")
            elif left_angle == "135" and right_angle == "135":
                class_names.append("PRECISION_BIMANUAL_135")
            elif left_angle == "45" and right_angle == "135":
                class_names.append("PRECISION_BIMANUAL_45_135")
            elif left_angle == "135" and right_angle == "45":
                class_names.append("PRECISION_BIMANUAL_135_45")
            else:
                class_names.append("UNKNOWN")
            continue

        prefix = "PRECISION" if precision_power == 0 else "POWER"
        if left_angle == "-1":
            class_names.append(f"{prefix}_UNIMANUAL_R_{right_angle}")
        elif right_angle == "-1":
            class_names.append(f"{prefix}_UNIMANUAL_L_{left_angle}")
        else:
            class_names.append("UNKNOWN")
    return np.array(class_names, dtype=object)


def load_session_structured(session, tag):
    data_path = CLEANED_STRUCTURED_DIR / f"data_{session}{tag}.npy"
    info_path = CLEANED_STRUCTURED_DIR / f"info_{session}{tag}.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info file: {info_path}")

    data = np.load(data_path, mmap_mode="r")
    with open(info_path, "rb") as fp:
        info = pickle.load(fp)
    class_names = parse_info_to_class_names(info)
    if data.shape[0] != class_names.shape[0]:
        raise ValueError(
            f"Trial count mismatch for {session}{tag}: data has {data.shape[0]} rows, "
            f"class metadata has {class_names.shape[0]} rows."
        )
    return data, class_names


def sample_class_trials(data, class_names, class_name, max_trials, rng):
    idx = np.where(class_names == class_name)[0]
    if idx.size == 0:
        return None
    if max_trials > 0 and idx.size > max_trials:
        idx = np.sort(rng.choice(idx, size=max_trials, replace=False))
    return np.array(data[idx], copy=True).astype(np.float32, copy=False)


def mean_centered_trajectory(trials, phase, pca):
    z_list = []
    for i in range(trials.shape[0]):
        z = pca.transform(trials[i, phase].T)
        z_list.append(z - z[0])
    return np.mean(np.stack(z_list, axis=0), axis=0)


def smooth_traj(z, sigma):
    if sigma <= 0:
        return z
    out = z.copy()
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    for col in range(min(2, out.shape[1])):
        out[:, col] = np.convolve(out[:, col], kernel, mode="same")
    return out


def add_gradient_line(ax, z2d, cmap_name, lw=2.0, alpha=1.0):
    if z2d.shape[0] < 2:
        return
    points = z2d[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap_name), norm=Normalize(0, len(segments)))
    lc.set_array(np.arange(len(segments)))
    lc.set_linewidth(lw)
    lc.set_alpha(alpha)
    ax.add_collection(lc)
    ax.scatter(z2d[0, 0], z2d[0, 1], marker="*", s=40, c="black")
    ax.scatter(z2d[-1, 0], z2d[-1, 1], marker="X", s=34, c="black")


def collect_class_counts(sessions, tag):
    counts = {}
    for session in sessions:
        data, class_names = load_session_structured(session, tag)
        counts[session] = Counter(class_names.tolist())
        del data
    return counts


def pick_common_classes(counts_by_session, min_trials_per_session, max_classes):
    common = None
    for counts in counts_by_session.values():
        eligible = {name for name, n in counts.items() if name != "UNKNOWN" and n >= min_trials_per_session}
        common = eligible if common is None else common & eligible
    ranked = sorted(
        common,
        key=lambda name: sum(counts_by_session[session][name] for session in counts_by_session),
        reverse=True,
    ) if common else []
    return ranked[:max_classes]


def build_phase_pca(healthy_trials_by_session, phase, n_components):
    pooled = []
    for trials in healthy_trials_by_session.values():
        pooled.append(np.transpose(trials[:, phase], (0, 2, 1)).reshape(-1, trials.shape[2]))
    x = np.concatenate(pooled, axis=0)
    return PCA(n_components=n_components, random_state=42).fit(x)


def run_pipeline_variant(tag, sessions_healthy, suspicious_session, class_names, out_path, args):
    rng = np.random.default_rng(args.seed)

    n_rows = len(class_names)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, max(4.5, 4.1 * n_rows)), squeeze=False)
    counts_lines = []

    healthy_colors = ["Blues", "Greens", "Purples", "Greys", "cividis", "viridis"]

    for row, class_name in enumerate(class_names):
        per_session_trials = {}
        for session in sessions_healthy + [suspicious_session]:
            data, labels = load_session_structured(session, tag)
            trials = sample_class_trials(data, labels, class_name, args.max_trials_per_session, rng)
            del data
            if trials is None or trials.shape[0] == 0:
                raise ValueError(f"Class {class_name} is missing from session {session}{tag}")
            per_session_trials[session] = trials
            counts_lines.append(f"{tag or '[default]'}, {class_name}, {session}, {trials.shape[0]}")

        healthy_trials = {session: per_session_trials[session] for session in sessions_healthy}

        for phase in (0, 1, 2):
            ax = axes[row, phase]
            pca = build_phase_pca(healthy_trials, phase, args.n_components)

            for i, session in enumerate(sessions_healthy):
                z = mean_centered_trajectory(per_session_trials[session], phase, pca)
                z = smooth_traj(z, args.smooth_sigma)
                add_gradient_line(ax, z[:, :2], healthy_colors[i % len(healthy_colors)], lw=1.8, alpha=0.8)

            z_bad = mean_centered_trajectory(per_session_trials[suspicious_session], phase, pca)
            z_bad = smooth_traj(z_bad, args.smooth_sigma)
            add_gradient_line(ax, z_bad[:, :2], "Reds", lw=2.8, alpha=1.0)

            ax.autoscale()
            if row == 0:
                ax.set_title(PHASE_NAMES[phase])
            if phase == 0:
                ax.set_ylabel(f"{class_name}\nPC2 (centered)")
            else:
                ax.set_ylabel("PC2 (centered)")
            ax.set_xlabel("PC1 (centered)")
            ax.grid(alpha=0.2)
            try:
                ax.set_aspect("equal", adjustable="box")
            except Exception:
                pass

    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=2.0, label="Healthy sessions"),
        plt.Line2D([0], [0], color="#d62728", lw=2.8, label=suspicious_session),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.suptitle(
        f"Session PCA health check | tag={tag or '[default]'} | "
        f"healthy={','.join(sessions_healthy)} | suspicious={suspicious_session}",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    ensure_dir(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return counts_lines


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight PCA health check for one suspicious session.")
    parser.add_argument("--healthy-sessions", nargs="+", default=DEFAULT_HEALTHY)
    parser.add_argument("--suspicious-session", default=DEFAULT_SUSPICIOUS)
    parser.add_argument(
        "--pipeline-tags",
        nargs="+",
        default=DEFAULT_PIPELINE_TAGS,
        help="Structured-data suffixes to compare, for example '' and '__no_bad_channels'.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional explicit class names. If omitted, the script selects common well-populated classes.",
    )
    parser.add_argument("--min-trials-per-session", type=int, default=12)
    parser.add_argument("--max-classes", type=int, default=4)
    parser.add_argument("--max-trials-per-session", type=int, default=20)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--smooth-sigma", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default="GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/session_pca_health",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sessions = args.healthy_sessions + [args.suspicious_session]

    for tag in args.pipeline_tags:
        counts = collect_class_counts(sessions, tag)
        if args.classes is None:
            selected_classes = pick_common_classes(counts, args.min_trials_per_session, args.max_classes)
            if not selected_classes:
                raise ValueError(
                    f"No common classes found for tag {tag!r} with "
                    f"min_trials_per_session={args.min_trials_per_session}."
                )
        else:
            selected_classes = args.classes

        print(f"[INFO] tag={tag!r} selected classes: {selected_classes}")

        safe_tag = "default" if tag == "" else tag.strip("_")
        out_path = os.path.join(args.out_dir, f"session_health_{safe_tag}.png")
        counts_lines = run_pipeline_variant(
            tag=tag,
            sessions_healthy=args.healthy_sessions,
            suspicious_session=args.suspicious_session,
            class_names=selected_classes,
            out_path=out_path,
            args=args,
        )
        print(f"[DONE] Saved figure to {out_path}")

        counts_path = os.path.join(args.out_dir, f"session_health_{safe_tag}_counts.csv")
        ensure_dir(counts_path)
        with open(counts_path, "w", encoding="utf-8") as fp:
            fp.write("pipeline_tag,class_name,session,n_trials\n")
            fp.write("\n".join(counts_lines))
            fp.write("\n")
        print(f"[DONE] Saved counts to {counts_path}")


if __name__ == "__main__":
    main()
