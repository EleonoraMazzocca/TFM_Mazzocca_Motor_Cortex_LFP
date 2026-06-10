"""
Beginner-friendly clustering explorer for CrossTaskClassification signals.

Main goals:
1) Global plot: all selected classes and all phases.
   - color = class
   - marker = phase
2) Single-class plot: only one class, to study phase separation.
3) Grid plot: one subplot per class, to compare phase separation class by class.

Embeddings:
- pca
- kpca     (Kernel PCA, nonlinear)
- isomap   (nonlinear manifold)
- tsne
- umap     (requires umap-learn installed)

2D and 3D are supported with --n-components 2 or 3.
"""

import argparse
import math
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import Isomap, TSNE

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from data_paths import CLASS_FILES
except ImportError:
    from GitHub_PreProcess_Pipeline.CrossTaskClassification.data_paths import CLASS_FILES


def ensure_dir(path):
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)


def trial_phase_to_vector(phase_signal, feature_mode):
    """
    Convert one phase signal (channels x time) to a vector.

    feature_mode:
    - psd: mean(abs(signal)) per channel (fast, interpretable)
    - raw: flatten channels x time (high dimensional, slower)
    """
    x = phase_signal.astype(np.float32, copy=False)
    if feature_mode == "psd":
        x = np.mean(np.abs(x), axis=1)
    else:
        x = x.reshape(-1)
    return x.astype(np.float32, copy=False)


def load_trials_for_class(class_name, max_trials_per_class, rng):
    arr = np.load(CLASS_FILES[class_name], mmap_mode="r")  # (trials, 3, channels, time)
    n_trials = arr.shape[0]
    if max_trials_per_class > 0 and n_trials > max_trials_per_class:
        idx = np.sort(rng.choice(n_trials, size=max_trials_per_class, replace=False))
        return np.array(arr[idx], copy=True)
    return np.array(arr, copy=True)


def build_points_for_classes(class_names, max_trials_per_class, feature_mode, rng):
    """
    Build:
    - X: feature matrix for embedding
    - class_labels: class name per point
    - phase_labels: phase id (0/1/2) per point
    - class_to_trials: trial tensors, used for single/grid plots
    """
    features = []
    class_labels = []
    phase_labels = []
    class_to_trials = {}

    for class_name in class_names:
        trials = load_trials_for_class(class_name, max_trials_per_class, rng)
        class_to_trials[class_name] = trials
        print(f"[INFO] Loaded {class_name}: {trials.shape}")

        for trial in trials:
            for phase in (0, 1, 2):
                features.append(trial_phase_to_vector(trial[phase], feature_mode))
                class_labels.append(class_name)
                phase_labels.append(phase)

    return np.stack(features, axis=0), np.array(class_labels), np.array(phase_labels), class_to_trials


def compute_embedding(X, embedding, n_components, seed, kpca_kernel, isomap_neighbors):
    """
    Return:
    - Z: embedded coordinates
    - axis_labels: labels for axes
    """
    emb = embedding.lower()

    if emb == "pca":
        model = PCA(n_components=n_components, random_state=seed)
        Z = model.fit_transform(X)
        axis_labels = []
        for i in range(n_components):
            axis_labels.append(f"PC{i+1} ({model.explained_variance_ratio_[i]*100:.1f}% var)")
        return Z, axis_labels

    if emb == "kpca":
        model = KernelPCA(n_components=n_components, kernel=kpca_kernel, random_state=seed)
        Z = model.fit_transform(X)
        return Z, [f"KPCA{i+1}" for i in range(n_components)]

    if emb == "isomap":
        model = Isomap(n_components=n_components, n_neighbors=isomap_neighbors)
        Z = model.fit_transform(X)
        return Z, [f"ISOMAP{i+1}" for i in range(n_components)]

    if emb == "tsne":
        perplexity = min(30, max(1, X.shape[0] - 1))
        model = TSNE(
            n_components=n_components,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        Z = model.fit_transform(X)
        return Z, [f"t-SNE {i+1}" for i in range(n_components)]

    if emb == "umap":
        try:
            import umap  # type: ignore
        except Exception as exc:
            raise RuntimeError("UMAP is not installed. Run: python -m pip install umap-learn") from exc
        model = umap.UMAP(n_components=n_components, random_state=seed)
        Z = model.fit_transform(X)
        return Z, [f"UMAP {i+1}" for i in range(n_components)]

    raise ValueError(f"Unknown embedding: {embedding}")


def _create_axis(n_components, figsize):
    if n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        return fig, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _scatter_points(ax, Z, mask, n_components, color, marker, size, alpha):
    if n_components == 3:
        ax.scatter(
            Z[mask, 0], Z[mask, 1], Z[mask, 2],
            s=size, alpha=alpha, c=[color], marker=marker, linewidths=0
        )
    else:
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=size, alpha=alpha, c=[color], marker=marker, linewidths=0
        )


def _set_axis_labels(ax, axis_labels, n_components):
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    if n_components == 3:
        ax.set_zlabel(axis_labels[2])


def plot_global_map(
    Z,
    class_labels,
    phase_labels,
    class_names,
    axis_labels,
    title,
    out_path,
    n_components,
    color_by="class",
    marker_by="phase",
):
    phase_to_color = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    phase_to_marker = {0: "o", 1: "s", 2: "^"}
    class_marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8", "p", "d"]
    cmap = plt.get_cmap("tab20")
    class_to_color = {name: cmap(i % 20) for i, name in enumerate(class_names)}
    class_to_marker = {name: class_marker_cycle[i % len(class_marker_cycle)] for i, name in enumerate(class_names)}

    fig, ax = _create_axis(n_components, (10, 8))
    for phase in (0, 1, 2):
        phase_mask = phase_labels == phase
        for class_name in class_names:
            mask = phase_mask & (class_labels == class_name)
            if np.any(mask):
                if color_by == "phase":
                    point_color = phase_to_color[phase]
                else:
                    point_color = class_to_color[class_name]

                if marker_by == "class":
                    point_marker = class_to_marker[class_name]
                else:
                    point_marker = phase_to_marker[phase]

                _scatter_points(ax, Z, mask, n_components, point_color, point_marker, 10, 0.5)

    ax.set_title(title)
    _set_axis_labels(ax, axis_labels, n_components)
    ax.grid(alpha=0.2)

    if color_by == "phase":
        class_handles = [
            plt.Line2D([0], [0], marker=class_to_marker[c], color="black", linestyle="None", label=c, markersize=6)
            for c in class_names
        ]
        phase_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=phase_to_color[p], label=f"phase {p}", markersize=6)
            for p in (0, 1, 2)
        ]
    else:
        class_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=class_to_color[c], label=c, markersize=6)
            for c in class_names
        ]
        phase_handles = [
            plt.Line2D([0], [0], marker=phase_to_marker[p], color="black", linestyle="None", label=f"phase {p}", markersize=6)
            for p in (0, 1, 2)
        ]
    leg1 = ax.legend(handles=class_handles, title="Class", loc="upper right", fontsize=7, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=phase_handles, title="Phase", loc="lower right", fontsize=8, frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_single_class_phases(trials, class_name, args, out_path):
    X = []
    y_phase = []
    for trial in trials:
        for phase in (0, 1, 2):
            X.append(trial_phase_to_vector(trial[phase], args.feature_mode))
            y_phase.append(phase)
    X = np.stack(X, axis=0)
    y_phase = np.array(y_phase)

    Z, axis_labels = compute_embedding(
        X,
        args.embedding,
        args.n_components,
        args.seed,
        args.kpca_kernel,
        args.isomap_neighbors,
    )

    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    markers = {0: "o", 1: "s", 2: "^"}

    fig, ax = _create_axis(args.n_components, (8, 6))
    for phase in (0, 1, 2):
        mask = y_phase == phase
        _scatter_points(ax, Z, mask, args.n_components, colors[phase], markers[phase], 12, 0.6)

    ax.set_title(f"{class_name}: phase separation ({args.embedding.upper()}, {args.n_components}D)")
    _set_axis_labels(ax, axis_labels, args.n_components)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker=markers[p], color=colors[p], linestyle="None", label=f"phase {p}", markersize=6)
            for p in (0, 1, 2)
        ]
    )
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_subplots_per_class(class_to_trials, args, out_path):
    class_names = list(class_to_trials.keys())
    n = len(class_names)
    ncols = 4
    nrows = int(math.ceil(n / ncols))

    subplot_kw = {"projection": "3d"} if args.n_components == 3 else None
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), subplot_kw=subplot_kw)
    axes = np.array(axes).reshape(nrows, ncols)

    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    markers = {0: "o", 1: "s", 2: "^"}

    for idx, class_name in enumerate(class_names):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        trials = class_to_trials[class_name]

        X = []
        y_phase = []
        for trial in trials:
            for phase in (0, 1, 2):
                X.append(trial_phase_to_vector(trial[phase], args.feature_mode))
                y_phase.append(phase)
        X = np.stack(X, axis=0)
        y_phase = np.array(y_phase)

        Z, _ = compute_embedding(
            X,
            args.embedding,
            args.n_components,
            args.seed,
            args.kpca_kernel,
            args.isomap_neighbors,
        )
        for phase in (0, 1, 2):
            mask = y_phase == phase
            _scatter_points(ax, Z, mask, args.n_components, colors[phase], markers[phase], 8, 0.5)

        ax.set_title(class_name, fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        if args.n_components == 3:
            ax.set_zticks([])

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    handles = [
        plt.Line2D([0], [0], marker=markers[p], color=colors[p], linestyle="None", label=f"phase {p}", markersize=6)
        for p in (0, 1, 2)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.suptitle(f"Per-Class Phase Separation ({args.embedding.upper()}, {args.n_components}D)", y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Beginner-friendly class/phase clustering plots.")
    parser.add_argument("--classes", nargs="+", default=list(CLASS_FILES.keys()), help="Class keys to load.")
    parser.add_argument(
        "--embedding",
        choices=["pca", "kpca", "isomap", "tsne", "umap"],
        default="tsne",
        help="Embedding method.",
    )
    parser.add_argument("--n-components", type=int, choices=[2, 3], default=2, help="Embedding dimensions (2D or 3D).")
    parser.add_argument("--feature-mode", choices=["psd", "raw"], default="psd", help="Feature extraction mode.")
    parser.add_argument("--max-trials-per-class", type=int, default=150, help="Cap per class for speed/memory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-class", default=None, help="Optional class key for one-class phase plot.")
    parser.add_argument("--kpca-kernel", choices=["rbf", "poly", "sigmoid", "cosine", "linear"], default="rbf")
    parser.add_argument("--isomap-neighbors", type=int, default=12)
    parser.add_argument(
        "--global-color-by",
        choices=["class", "phase"],
        default="class",
        help="Global plot color encoding.",
    )
    parser.add_argument(
        "--global-marker-by",
        choices=["phase", "class"],
        default="phase",
        help="Global plot marker encoding.",
    )
    parser.add_argument("--out-global", default="GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/clusters_global.png")
    parser.add_argument("--out-single", default="GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/clusters_single_class.png")
    parser.add_argument("--out-grid", default="GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/clusters_per_class_grid.png")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    for c in args.classes:
        if c not in CLASS_FILES:
            raise ValueError(f"Unknown class key: {c}")
    if args.single_class is not None and args.single_class not in CLASS_FILES:
        raise ValueError(f"Unknown --single-class key: {args.single_class}")

    X, class_labels, phase_labels, class_to_trials = build_points_for_classes(
        class_names=args.classes,
        max_trials_per_class=args.max_trials_per_class,
        feature_mode=args.feature_mode,
        rng=rng,
    )

    Z, axis_labels = compute_embedding(
        X,
        args.embedding,
        args.n_components,
        args.seed,
        args.kpca_kernel,
        args.isomap_neighbors,
    )
    ensure_dir(args.out_global)
    plot_global_map(
        Z=Z,
        class_labels=class_labels,
        phase_labels=phase_labels,
        class_names=args.classes,
        axis_labels=axis_labels,
        title=(
            f"Global Clusters ({args.embedding.upper()}, {args.n_components}D): "
            f"color={args.global_color_by}, marker={args.global_marker_by}"
        ),
        out_path=args.out_global,
        n_components=args.n_components,
        color_by=args.global_color_by,
        marker_by=args.global_marker_by,
    )
    print(f"[INFO] Saved global plot: {args.out_global}")

    single_class = args.single_class if args.single_class is not None else args.classes[0]
    ensure_dir(args.out_single)
    plot_single_class_phases(class_to_trials[single_class], single_class, args, args.out_single)
    print(f"[INFO] Saved single-class plot: {args.out_single}")

    ensure_dir(args.out_grid)
    plot_subplots_per_class(class_to_trials, args, args.out_grid)
    print(f"[INFO] Saved grid plot: {args.out_grid}")


if __name__ == "__main__":
    main()
