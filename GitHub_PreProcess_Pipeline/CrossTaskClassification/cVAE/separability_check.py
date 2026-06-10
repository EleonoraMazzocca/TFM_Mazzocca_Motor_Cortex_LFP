"""Step 0 — Separability and independence verification.

Checks two prerequisites for the compositional generation claim:
  1. Separability: do (phase, grip, hand) combinations form coherent clusters
     in the transformer's learned representation space?
  2. Independence: are grip, hand, and phase encoded as approximately orthogonal
     factors (grip accuracy doesn't depend on hand/phase condition, and vice versa)?

Usage:
    python separability_check.py \\
        --checkpoint_reach  results/specialist_reach_per_channel/checkpoint.pt \\
        --checkpoint_prereach results/specialist_prereach_per_channel/checkpoint.pt \\
        --checkpoint_grasp  results/specialist_grasp_per_channel/checkpoint.pt \\
        --data_dir /path/to/mua_files

    # Reach only (minimum required):
    python separability_check.py \\
        --checkpoint_reach results/specialist_reach_per_channel/checkpoint.pt \\
        --data_dir /path/to/mua_files

    # Dry run (subsample to 500 trials):
    python separability_check.py \\
        --checkpoint_reach results/specialist_reach_per_channel/checkpoint.pt \\
        --data_dir /path/to/mua_files --dry_run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mutual_info_score, normalized_mutual_info_score
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_TRANSFORMER = _HERE.parent / "transformer"
if str(_TRANSFORMER) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER))

from data import (
    AREA_SLICES, N_AREAS, GRIP_TO_ID, HAND_TO_ID, ANGLE_TO_ID,
    PHASE_NAMES, MAX_AREA_CHANNELS,
)
from specialist_data import (
    LFPSpecialistDataset,
    compute_specialist_norm_stats,
    extract_phase_area_features,
)
from specialist_model import LFPSpecialistTransformer
from cvae_data import N_REAL_CHANNELS, N_TIMEPOINTS, ID_TO_GRIP, ID_TO_HAND

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CLASS_STEM = re.compile(
    r"^(power|precision)_unimanual_(left|right)_(0|45|90|135)_degrees_mua_200_500$"
)
_BROADBAND_STEM = re.compile(
    r"^(power|precision)_unimanual_(left|right)_(0|45|90|135)_degrees$"
)
AREA_NAMES   = ["PMvR", "M1", "PMdR", "PMdL"]
GRIP_NAMES   = ["power", "precision"]
HAND_NAMES   = ["left", "right"]
PHASE_COLORS = ["#4C72B0", "#DD8452", "#55A868"]   # prereach, reach, grasp
GRIP_COLORS  = ["#4C72B0", "#DD8452"]               # power, precision
HAND_COLORS  = ["#4C72B0", "#DD8452"]               # left, right
GH_COLORS    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # 4 grip×hand combos
FULL_COLORS  = [                                    # 12 full combo colors
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 0: separability and independence check on transformer representations."
    )
    p.add_argument("--checkpoint_reach",    type=str, required=True,
                   help="Path to reach specialist checkpoint (required).")
    p.add_argument("--checkpoint_prereach", type=str, default=None,
                   help="Path to prereach specialist checkpoint (optional).")
    p.add_argument("--checkpoint_grasp",    type=str, default=None,
                   help="Path to grasp specialist checkpoint (optional).")
    p.add_argument("--data_dir",   type=str, required=True,
                   help="Directory containing *_mua_200_500.npy files.")
    p.add_argument("--broadband_data_dir", type=str, default=None,
                   help="Optional directory containing unfiltered *_degrees.npy segmented files.")
    p.add_argument("--out_dir",    type=str, default="results/separability_check/",
                   help="Output directory for plots and summary.")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--no_plot",    action="store_true")
    p.add_argument("--dry_run",    action="store_true",
                   help="Subsample to 500 trials for speed.")
    # Model architecture defaults (should match the checkpoint)
    p.add_argument("--d_model",        type=int, default=64)
    p.add_argument("--n_heads",        type=int, default=4)
    p.add_argument("--n_layers",       type=int, default=2)
    p.add_argument("--feedforward_dim",type=int, default=128)
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--fs",             type=float, default=1000.0,
                   help="Sampling rate of MU signal in Hz (default: 1000).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading — all trials for all 16 class files
# ---------------------------------------------------------------------------

def load_all_trials(data_dir: Path) -> dict:
    """Load all MU class files and return flat index arrays.

    Returns dict with:
      file_paths  list of path strings
      file_idx    (N,) int — maps trial to file
      trial_idx   (N,) int
      y_grip      (N,) int
      y_hand      (N,) int
      y_angle     (N,) int
    """
    file_paths, file_idx_l, trial_idx_l = [], [], []
    y_grip_l, y_hand_l, y_angle_l = [], [], []

    for path in sorted(data_dir.glob("*_mua_200_500.npy")):
        if "bimanual" in path.name:
            continue
        m = _CLASS_STEM.match(path.stem)
        if m is None:
            continue
        gname, hname, aname = m.groups()
        g, h, a = GRIP_TO_ID[gname], HAND_TO_ID[hname], ANGLE_TO_ID[aname]

        arr = np.load(str(path), mmap_mode="r")
        n   = arr.shape[0]
        fi  = len(file_paths)
        file_paths.append(str(path))
        file_idx_l.append(np.full(n, fi, dtype=np.int32))
        trial_idx_l.append(np.arange(n, dtype=np.int32))
        y_grip_l.append(np.full(n, g, dtype=np.int64))
        y_hand_l.append(np.full(n, h, dtype=np.int64))
        y_angle_l.append(np.full(n, a, dtype=np.int64))

    return {
        "file_paths": file_paths,
        "file_idx":   np.concatenate(file_idx_l),
        "trial_idx":  np.concatenate(trial_idx_l),
        "y_grip":     np.concatenate(y_grip_l),
        "y_hand":     np.concatenate(y_hand_l),
        "y_angle":    np.concatenate(y_angle_l),
    }


def load_broadband_trials(data_dir: Path) -> dict:
    """Load all unfiltered/broadband class files and return flat index arrays."""
    file_paths, file_idx_l, trial_idx_l = [], [], []
    y_grip_l, y_hand_l, y_angle_l = [], [], []

    for path in sorted(data_dir.glob("*_degrees.npy")):
        if "bimanual" in path.name or "_mua_" in path.name:
            continue
        m = _BROADBAND_STEM.match(path.stem)
        if m is None:
            continue
        gname, hname, aname = m.groups()
        g, h, a = GRIP_TO_ID[gname], HAND_TO_ID[hname], ANGLE_TO_ID[aname]

        arr = np.load(str(path), mmap_mode="r")
        n   = arr.shape[0]
        fi  = len(file_paths)
        file_paths.append(str(path))
        file_idx_l.append(np.full(n, fi, dtype=np.int32))
        trial_idx_l.append(np.arange(n, dtype=np.int32))
        y_grip_l.append(np.full(n, g, dtype=np.int64))
        y_hand_l.append(np.full(n, h, dtype=np.int64))
        y_angle_l.append(np.full(n, a, dtype=np.int64))

    if not file_paths:
        raise ValueError(f"No unfiltered *_degrees.npy files found in {data_dir}")

    return {
        "file_paths": file_paths,
        "file_idx":   np.concatenate(file_idx_l),
        "trial_idx":  np.concatenate(trial_idx_l),
        "y_grip":     np.concatenate(y_grip_l),
        "y_hand":     np.concatenate(y_hand_l),
        "y_angle":    np.concatenate(y_angle_l),
    }


# ---------------------------------------------------------------------------
# Model loading and embedding extraction
# ---------------------------------------------------------------------------

def _try_load_arch_from_summary(checkpoint_path: str) -> dict:
    """Try reading model architecture from summary.json next to the checkpoint."""
    summary_path = Path(checkpoint_path).parent / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        s = json.loads(summary_path.read_text())
        m = s.get("model", {})
        return {
            "d_model":         m.get("d_model", None),
            "n_heads":         m.get("n_heads", None),
            "n_layers":        m.get("n_layers", None),
            "feedforward_dim": m.get("feedforward_dim", None),
        }
    except Exception:
        return {}


def load_specialist(
    checkpoint_path: str,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    feedforward_dim: int = 128,
    device: torch.device | None = None,
) -> tuple[LFPSpecialistTransformer, bool, int | str]:
    """Load a per-channel specialist checkpoint into eval mode."""
    if device is None:
        device = torch.device("cpu")

    # Try to override architecture from summary.json
    arch = _try_load_arch_from_summary(checkpoint_path)
    d_model        = arch.get("d_model")        or d_model
    n_heads        = arch.get("n_heads")        or n_heads
    n_layers       = arch.get("n_layers")       or n_layers
    feedforward_dim= arch.get("feedforward_dim")or feedforward_dim

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    input_weight = state.get("input_proj.weight")
    has_time_embedding = "time_embedding.weight" in state
    use_per_channel = not has_time_embedding
    input_dim = int(input_weight.shape[1]) if input_weight is not None else (
        MAX_AREA_CHANNELS if use_per_channel else 1
    )
    n_bins = 1

    model = LFPSpecialistTransformer(
        use_per_channel  = use_per_channel,
        input_dim        = input_dim,
        d_model          = d_model,
        n_heads          = n_heads,
        n_layers         = n_layers,
        feedforward_dim  = feedforward_dim,
        dropout          = 0.0,                 # eval mode — dropout disabled anyway
        n_bins           = n_bins,
        n_angle_classes  = 4,
    )

    model.load_state_dict(state)
    model.to(device).eval()
    mode = "per_channel" if use_per_channel else "nbins1"
    print(f"  Loaded specialist from {checkpoint_path}  d_model={d_model}  mode={mode}")
    return model, use_per_channel, n_bins


def extract_embeddings(
    model: LFPSpecialistTransformer,
    data: dict,
    phase_idx: int,
    norm_stats: dict,
    use_per_channel: bool,
    n_bins: int | str,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run all trials through specialist and collect pooled embeddings.

    Intercepts the output of model.norm (LayerNorm after mean pooling) via a
    forward hook — this is the representation that feeds the classification heads.

    Returns (embeddings (N, d_model), grip_logits (N,), hand_logits (N,)).
    """
    if device is None:
        device = torch.device("cpu")

    # Build a minimal data dict compatible with LFPSpecialistDataset
    n = len(data["y_grip"])
    ds_dict = {
        "file_paths": np.array(data["file_paths"], dtype=object),
        "file_idx":   data["file_idx"].astype(np.int16),
        "trial_idx":  data["trial_idx"].astype(np.int32),
        "y_grip":     data["y_grip"],
        "y_hand":     data["y_hand"],
        "y_angle":    data["y_angle"],
        "is_heldout": np.zeros(n, dtype=bool),
        "n_channels": np.array(256, dtype=np.int64),
    }

    ds = LFPSpecialistDataset(
        ds_dict, phase_idx, norm_stats=norm_stats,
        n_bins=n_bins, use_per_channel=use_per_channel,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    captured_embs: list[torch.Tensor] = []
    grip_logits_l: list[torch.Tensor] = []
    hand_logits_l: list[torch.Tensor] = []

    def _hook(module, inp, out):
        captured_embs.append(out.detach().cpu())

    handle = model.norm.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        for x, *_ in loader:
            x = x.to(device)
            lg, lh, _ = model(x)
            grip_logits_l.append(lg.cpu())
            hand_logits_l.append(lh.cpu())
    handle.remove()

    emb  = torch.cat(captured_embs, dim=0).numpy()       # (N, d_model)
    grip = torch.cat(grip_logits_l, dim=0).numpy()        # (N, 2)
    hand = torch.cat(hand_logits_l, dim=0).numpy()        # (N, 2)
    return emb, grip, hand


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def inter_intra_ratio(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean inter-class / mean intra-class euclidean distance."""
    classes = np.unique(labels)
    intra_dists, inter_dists = [], []
    for c in classes:
        idx = np.where(labels == c)[0]
        others = np.where(labels != c)[0]
        if len(idx) > 1:
            from sklearn.metrics.pairwise import euclidean_distances
            intra_dists.append(euclidean_distances(X[idx]).mean())
        if len(others) > 0:
            from sklearn.metrics.pairwise import euclidean_distances
            inter_dists.append(euclidean_distances(X[idx], X[others]).mean())
    if not intra_dists or not inter_dists:
        return float("nan")
    return float(np.mean(inter_dists) / max(np.mean(intra_dists), 1e-10))


def _label_combo(g: np.ndarray, h: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Encode (grip, hand, phase) → unique int in [0, 11]."""
    return g * 6 + h * 3 + p


# ---------------------------------------------------------------------------
# Part A: Representation extraction (main function)
# ---------------------------------------------------------------------------

def collect_all_embeddings(
    checkpoints: dict[int, str],   # {phase_idx: path}
    data: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Extract embeddings from all available phase checkpoints.

    Returns a dict with:
      embeddings  (N, d_model)  pooled transformer representations
      y_grip      (N,)
      y_hand      (N,)
      y_phase     (N,)  — which specialist produced this embedding (phase_idx)
      y_angle     (N,)
      y_griphand  (N,)  grip*2 + hand
      y_combo     (N,)  grip*6 + hand*3 + phase (0-11)
      grip_logits (N,)  scalar = precision_logit - power_logit
      hand_logits (N,)  scalar = right_logit - left_logit
      grip_pred   (N,)
      hand_pred   (N,)
    """
    all_emb, all_grip, all_hand, all_angle = [], [], [], []
    all_phase, all_grip_logit, all_hand_logit = [], [], []
    all_grip_pred, all_hand_pred = [], []

    for phase_idx, ckpt_path in sorted(checkpoints.items()):
        print(f"\n  Extracting {PHASE_NAMES[phase_idx].upper()} embeddings ...")
        model, use_per_channel, n_bins = load_specialist(
            ckpt_path,
            d_model         = args.d_model,
            n_heads         = args.n_heads,
            n_layers        = args.n_layers,
            feedforward_dim = args.feedforward_dim,
            device          = device,
        )

        # Compute norm stats from all trials (just for this phase)
        # We compute on all trials since we're doing analysis, not training
        norm_stats = compute_specialist_norm_stats(
            {
                "file_paths": np.array(data["file_paths"], dtype=object),
                "file_idx":   data["file_idx"].astype(np.int16),
                "trial_idx":  data["trial_idx"].astype(np.int32),
                "y_grip":     data["y_grip"],
                "y_hand":     data["y_hand"],
                "y_angle":    data["y_angle"],
                "is_heldout": np.zeros(len(data["y_grip"]), dtype=bool),
                "n_channels": np.array(256, dtype=np.int64),
            },
            phase_idx=phase_idx,
            n_bins=n_bins,
            use_per_channel=use_per_channel,
        )

        emb, grip_log, hand_log = extract_embeddings(
            model, data, phase_idx, norm_stats, use_per_channel, n_bins,
            batch_size=args.batch_size, device=device,
        )

        n = len(emb)
        all_emb.append(emb)
        all_grip.append(data["y_grip"])
        all_hand.append(data["y_hand"])
        all_angle.append(data["y_angle"])
        all_phase.append(np.full(n, phase_idx, dtype=np.int64))
        # Scalar logit: positive = right/precision, negative = left/power
        all_grip_logit.append(grip_log[:, 1] - grip_log[:, 0])   # prec - pow
        all_hand_logit.append(hand_log[:, 1] - hand_log[:, 0])   # right - left
        all_grip_pred.append(np.argmax(grip_log, axis=1))
        all_hand_pred.append(np.argmax(hand_log, axis=1))

        del model

    emb   = np.concatenate(all_emb,   axis=0)
    grip  = np.concatenate(all_grip)
    hand  = np.concatenate(all_hand)
    angle = np.concatenate(all_angle)
    phase = np.concatenate(all_phase)

    return {
        "embeddings":   emb,
        "y_grip":       grip,
        "y_hand":       hand,
        "y_angle":      angle,
        "y_phase":      phase,
        "y_griphand":   grip * 2 + hand,
        "y_combo":      _label_combo(grip, hand, phase),
        "grip_logits":  np.concatenate(all_grip_logit),
        "hand_logits":  np.concatenate(all_hand_logit),
        "grip_pred":    np.concatenate(all_grip_pred),
        "hand_pred":    np.concatenate(all_hand_pred),
    }


# ---------------------------------------------------------------------------
# Part B: PCA / UMAP visualization
# ---------------------------------------------------------------------------

def _project(X: np.ndarray, n_components: int = 3) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


def _umap_or_pca(X: np.ndarray, seed: int = 42) -> tuple[np.ndarray, str]:
    """Attempt UMAP, fall back to PCA if umap-learn is not installed."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(X), "UMAP"
    except ImportError:
        print("  WARNING: umap-learn not installed — using PCA instead of UMAP.")
        pca = PCA(n_components=2, random_state=seed)
        return pca.fit_transform(X), "PCA(fallback)"


def _tsne_or_none(X: np.ndarray, seed: int = 42) -> np.ndarray | None:
    """Compute t-SNE after PCA pre-reduction; return None if unavailable/failing."""
    try:
        from sklearn.manifold import TSNE
        X_pre = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=seed).fit_transform(X)
        perplexity = min(30, max(5, (len(X_pre) - 1) // 3))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(X_pre)
    except Exception as e:
        print(f"  WARNING: t-SNE failed/skipped: {e}")
        return None


def _scatter2d(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    colors: list[str],
    label_names: list[str],
    title: str,
) -> None:
    uniq = sorted(np.unique(labels))
    for pos, lab in enumerate(uniq):
        name = label_names[pos] if pos < len(label_names) else str(lab)
        color = colors[pos % len(colors)]
        mask = labels == lab
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=name, alpha=0.5, s=8, linewidths=0,
        )
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_xlabel("PC1" if "PCA" in title else "D1")
    ax.set_ylabel("PC2" if "PCA" in title else "D2")


def plot_embeddings(
    rep: dict,
    pca_coords: np.ndarray,   # (N, 2)
    umap_coords: np.ndarray,  # (N, 2)
    umap_label: str,
    out_dir: Path,
    no_plot: bool,
    tsne_coords: np.ndarray | None = None,
) -> None:
    """Produce five plot sets (Part B) and save to out_dir."""
    if no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots.")
        return

    phase_indices = sorted(np.unique(rep["y_phase"]))

    def _save_panel_plot(coords, labels, colors, names, title, fname, per_phase=True):
        if per_phase and len(phase_indices) > 1:
            ncols = len(phase_indices) + 1
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
            axes_l = axes[0]
            panel_masks = [(PHASE_NAMES[ph], rep["y_phase"] == ph) for ph in phase_indices]
            panel_masks.append(("combined", np.ones(len(labels), dtype=bool)))
            for ax, (panel_title, mask) in zip(axes_l, panel_masks):
                _scatter2d(ax, coords[mask], labels[mask], colors, names, f"{title} — {panel_title}")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            _scatter2d(ax, coords, labels, colors, names, title)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    combo_names = ["power+left", "power+right", "precision+left", "precision+right"]
    phase_label_names = [PHASE_NAMES[ph] for ph in phase_indices]
    phase_labels_compact = np.searchsorted(phase_indices, rep["y_phase"])
    full_unique = sorted(np.unique(rep["y_combo"]))
    full_label_names = [
        f"{PHASE_NAMES[c % 3]}+{GRIP_NAMES[c // 6]}+{HAND_NAMES[(c // 3) % 2]}"
        for c in full_unique
    ]
    full_labels_compact = np.array([full_unique.index(c) for c in rep["y_combo"]])

    plot_sets = [
        (rep["y_grip"],       GRIP_COLORS,  GRIP_NAMES,       "by grip",          "by_grip", True),
        (rep["y_hand"],       HAND_COLORS,  HAND_NAMES,       "by hand",          "by_hand", True),
        (rep["y_griphand"],   GH_COLORS,    combo_names,      "by grip+hand",     "by_combo_griphand", True),
        (phase_labels_compact,PHASE_COLORS, phase_label_names,"by phase",         "by_phase", False),
        (full_labels_compact, FULL_COLORS,  full_label_names, "by full combo",    "by_combo_full", False),
    ]

    for labels, colors, names, title, stem, per_phase in plot_sets:
        _save_panel_plot(pca_coords, labels, colors, names, f"PCA — {title}", f"pca_{stem}.png", per_phase)
        _save_panel_plot(umap_coords, labels, colors, names, f"{umap_label} — {title}", f"umap_{stem}.png", per_phase)
        if tsne_coords is not None:
            _save_panel_plot(tsne_coords, labels, colors, names, f"t-SNE — {title}", f"tsne_{stem}.png", per_phase)

    print(f"  Saved embedding plots to {out_dir}")


# ---------------------------------------------------------------------------
# Model-free segmented-data separability
# ---------------------------------------------------------------------------

def collect_model_free_amplitude_features(
    data: dict,
    label: str,
) -> dict:
    """Build a model-free per-channel amplitude representation from segmented files.

    Each sample is one trial x phase with features:
      mean(abs(signal[channel, :])) for the 256 real channels.

    This is intentionally transformer-free and works for both MU-filtered and
    unfiltered/broadband segmented class files.
    """
    print(f"\n  Extracting model-free amplitude features: {label}")
    file_cache: dict[str, np.ndarray] = {}
    features, y_grip, y_hand, y_angle, y_phase = [], [], [], [], []

    for i in range(len(data["y_grip"])):
        fp = data["file_paths"][data["file_idx"][i]]
        if fp not in file_cache:
            file_cache[fp] = np.load(fp, mmap_mode="r")
        sample = file_cache[fp][int(data["trial_idx"][i])]  # (phase, channel, time)
        for ph in range(len(PHASE_NAMES)):
            amp = np.mean(
                np.abs(sample[ph, :N_REAL_CHANNELS, :].astype(np.float32, copy=False)),
                axis=1,
            )
            features.append(amp)
            y_grip.append(data["y_grip"][i])
            y_hand.append(data["y_hand"][i])
            y_angle.append(data["y_angle"][i])
            y_phase.append(ph)

    emb = np.asarray(features, dtype=np.float32)
    grip = np.asarray(y_grip, dtype=np.int64)
    hand = np.asarray(y_hand, dtype=np.int64)
    phase = np.asarray(y_phase, dtype=np.int64)
    angle = np.asarray(y_angle, dtype=np.int64)
    return {
        "embeddings": emb,
        "y_grip": grip,
        "y_hand": hand,
        "y_angle": angle,
        "y_phase": phase,
        "y_griphand": grip * 2 + hand,
        "y_combo": _label_combo(grip, hand, phase),
    }


def run_model_free_separability(
    data: dict,
    label: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Run PCA/UMAP, silhouette, and distance ratios on direct segmented data."""
    print("\n" + "=" * 60)
    print(f"  Model-free separability — {label}")
    print("=" * 60)

    rep = collect_model_free_amplitude_features(data, label)
    phase_indices = list(range(len(PHASE_NAMES)))
    phase_names = [PHASE_NAMES[i] for i in phase_indices]

    pca_coords = _project(rep["embeddings"], n_components=2)
    umap_coords, umap_label = _umap_or_pca(rep["embeddings"], seed=args.seed)
    tsne_coords = None if args.no_plot else _tsne_or_none(rep["embeddings"], seed=args.seed)

    plot_dir = out_dir / f"model_free_{label}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plot:
        plot_embeddings(rep, pca_coords, umap_coords, umap_label, plot_dir, args.no_plot, tsne_coords)

    sil = compute_silhouette_scores(rep, phase_indices)
    ratios = compute_distance_ratios(rep, phase_indices)
    probes = compute_linear_probe_scores(rep, phase_indices, seed=args.seed)

    print(f"\nModel-free silhouette scores ({label}):")
    _print_silhouette_table(sil, phase_names)
    print(f"\nModel-free distance ratios ({label}):")
    _print_ratio_table(ratios, phase_names)
    print(f"\nModel-free linear probes ({label}):")
    _print_probe_table(probes, phase_names)

    return {
        "feature": "per_channel_mean_abs_amplitude",
        "n_samples": int(len(rep["embeddings"])),
        "input_dim": int(rep["embeddings"].shape[1]),
        "silhouette_scores": sil,
        "distance_ratios": ratios,
        "linear_probe_scores": probes,
        "plot_dir": str(plot_dir),
    }


# ---------------------------------------------------------------------------
# Part C: Silhouette scores
# ---------------------------------------------------------------------------

def compute_silhouette_scores(rep: dict, phase_indices: list[int]) -> dict:
    """Compute silhouette scores per phase and on the combined embedding."""
    results: dict = {}
    emb  = rep["embeddings"]
    cols = ["grip", "hand", "phase", "griphand", "full_combo"]

    for ph in phase_indices:
        mask = rep["y_phase"] == ph
        if mask.sum() < 4:
            continue
        sub = emb[mask]
        ph_name = PHASE_NAMES[ph]
        results.setdefault(ph_name, {})
        for col, labels in [
            ("grip",      rep["y_grip"][mask]),
            ("hand",      rep["y_hand"][mask]),
            ("griphand",  rep["y_griphand"][mask]),
            ("full_combo",rep["y_combo"][mask]),
        ]:
            try:
                results[ph_name][col] = float(silhouette_score(sub, labels))
            except Exception:
                results[ph_name][col] = float("nan")

    # Combined embedding
    results["combined"] = {}
    for col, labels in [
        ("grip",      rep["y_grip"]),
        ("hand",      rep["y_hand"]),
        ("phase",     rep["y_phase"]),
        ("griphand",  rep["y_griphand"]),
        ("full_combo",rep["y_combo"]),
    ]:
        try:
            results["combined"][col] = float(silhouette_score(emb, labels))
        except Exception:
            results["combined"][col] = float("nan")

    return results


def _print_silhouette_table(sil: dict, phase_names: list[str]) -> None:
    cols    = ["grip", "hand", "phase", "griphand", "full_combo"]
    phases  = phase_names + ["combined"]
    W = 10
    header  = f"{'':>15}" + "".join(f"{p:>{W}}" for p in phases)
    print("\nSilhouette scores:")
    print(header)
    print("  " + "-" * (15 + W * len(phases)))
    for col in cols:
        row = f"  {col:<13}"
        for ph in phases:
            v = sil.get(ph, {}).get(col, float("nan"))
            row += f"{v:>{W}.3f}" if not np.isnan(v) else f"{'—':>{W}}"
        print(row)
    print("  Interpretation: >0.3 = clear, 0.1-0.3 = moderate, <0.0 = no structure")


# ---------------------------------------------------------------------------
# Part D: Within-class vs between-class distances
# ---------------------------------------------------------------------------

def compute_distance_ratios(rep: dict, phase_indices: list[int]) -> dict:
    """Compute inter/intra euclidean distance ratios."""
    from sklearn.metrics.pairwise import euclidean_distances
    emb    = rep["embeddings"]
    results: dict = {}

    def _ratio(X, labels):
        classes = np.unique(labels)
        intra, inter = [], []
        for c in classes:
            idx  = np.where(labels == c)[0]
            rest = np.where(labels != c)[0]
            if len(idx) > 1:
                intra.append(euclidean_distances(X[idx]).mean())
            if len(rest) > 0:
                inter.append(euclidean_distances(X[idx], X[rest]).mean())
        if not intra or not inter:
            return float("nan")
        return float(np.mean(inter) / max(np.mean(intra), 1e-10))

    for ph in phase_indices:
        mask    = rep["y_phase"] == ph
        sub     = emb[mask]
        ph_name = PHASE_NAMES[ph]
        results.setdefault(ph_name, {})
        for col, labels in [
            ("grip",      rep["y_grip"][mask]),
            ("hand",      rep["y_hand"][mask]),
            ("griphand",  rep["y_griphand"][mask]),
            ("full_combo",rep["y_combo"][mask]),
        ]:
            results[ph_name][col] = _ratio(sub, labels)

    results["combined"] = {}
    for col, labels in [
        ("grip",       rep["y_grip"]),
        ("hand",       rep["y_hand"]),
        ("phase",      rep["y_phase"]),
        ("griphand",   rep["y_griphand"]),
        ("full_combo", rep["y_combo"]),
    ]:
        results["combined"][col] = _ratio(emb, labels)

    return results


def compute_linear_probe_scores(rep: dict, phase_indices: list[int], seed: int = 42) -> dict:
    """Cross-validated linear decoding accuracy for each label type.

    This complements silhouette/distance ratios. A factor can be linearly
    decodable even when it does not form compact unsupervised clusters.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = rep["embeddings"]
    label_map = {
        "grip": rep["y_grip"],
        "hand": rep["y_hand"],
        "phase": rep["y_phase"],
        "griphand": rep["y_griphand"],
        "full_combo": rep["y_combo"],
    }

    def _score(Xi: np.ndarray, yi: np.ndarray) -> dict:
        classes, counts = np.unique(yi, return_counts=True)
        if len(classes) < 2 or counts.min() < 3:
            return {"balanced_accuracy_mean": float("nan"), "balanced_accuracy_std": float("nan"), "chance": float("nan")}
        n_splits = int(min(5, counts.min()))
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, Xi, yi, cv=cv, scoring="balanced_accuracy")
        return {
            "balanced_accuracy_mean": float(np.mean(scores)),
            "balanced_accuracy_std": float(np.std(scores)),
            "chance": float(1.0 / len(classes)),
        }

    results: dict = {"combined": {}}
    for name, labels in label_map.items():
        results["combined"][name] = _score(X, labels)

    for ph in phase_indices:
        mask = rep["y_phase"] == ph
        ph_name = PHASE_NAMES[ph]
        results[ph_name] = {}
        for name in ("grip", "hand", "griphand", "full_combo"):
            results[ph_name][name] = _score(X[mask], label_map[name][mask])
    return results


def _print_probe_table(probes: dict, phase_names: list[str]) -> None:
    cols = ["grip", "hand", "phase", "griphand", "full_combo"]
    phases = phase_names + ["combined"]
    W = 14
    print("\nLinear-probe balanced accuracy:")
    print(f"{'':>15}" + "".join(f"{p:>{W}}" for p in phases))
    print("  " + "-" * (15 + W * len(phases)))
    for col in cols:
        row = f"  {col:<13}"
        for ph in phases:
            d = probes.get(ph, {}).get(col, {})
            v = d.get("balanced_accuracy_mean", float("nan"))
            row += f"{v:>{W}.3f}" if not np.isnan(v) else f"{'—':>{W}}"
        print(row)
    print("  Interpretation: compare against chance; high probe accuracy can coexist with weak silhouette.")


def _print_ratio_table(ratios: dict, phase_names: list[str]) -> None:
    cols   = ["grip", "hand", "phase", "griphand", "full_combo"]
    phases = phase_names + ["combined"]
    W = 10
    print("\nDistance ratios (inter/intra euclidean):")
    print(f"{'':>15}" + "".join(f"{p:>{W}}" for p in phases))
    print("  " + "-" * (15 + W * len(phases)))
    for col in cols:
        row = f"  {col:<13}"
        for ph in phases:
            v = ratios.get(ph, {}).get(col, float("nan"))
            row += f"{v:>{W}.2f}" if not np.isnan(v) else f"{'—':>{W}}"
        print(row)
    print("  Interpretation: ratio > 1.5 = good separability for cVAE")


# ---------------------------------------------------------------------------
# Part E: PSD consistency (model-free)
# ---------------------------------------------------------------------------

def compute_psd_consistency(
    data: dict,
    phase_indices: list[int],
    out_dir: Path,
    no_plot: bool,
    fs: float = 1000.0,
) -> dict:
    """Compute PSD per (phase, grip, hand) combination and coefficient of variation.

    Loads raw MU waveforms, runs welch PSD per channel per trial, averages across
    channels, then computes mean ± std across trials within each combination.
    """
    print("\nPart E: PSD consistency check ...")
    n_all    = len(data["y_grip"])
    file_cache: dict[str, np.ndarray] = {}

    def _load(fp):
        if fp not in file_cache:
            file_cache[fp] = np.load(fp, mmap_mode="r")
        return file_cache[fp]

    results: dict = {}
    combo_names = {
        (0, 0): "power+left", (0, 1): "power+right",
        (1, 0): "prec+left",  (1, 1): "prec+right",
    }

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        CAN_PLOT = not no_plot
    except ImportError:
        CAN_PLOT = False

    GH_PLOT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ph in phase_indices:
        ph_name  = PHASE_NAMES[ph]
        psd_by_combo: dict = {}

        for (g, h), cname in combo_names.items():
            mask  = (data["y_grip"] == g) & (data["y_hand"] == h)
            idx   = np.where(mask)[0]
            psds  = []
            for i in idx:
                fp  = data["file_paths"][data["file_idx"][i]]
                arr = _load(fp)
                sig = arr[int(data["trial_idx"][i]), ph, :N_REAL_CHANNELS, :]   # (256, 500)
                sig = sig.astype(np.float32)
                # PSD per channel, then average across channels
                trial_psds = []
                for ch in range(N_REAL_CHANNELS):
                    if np.all(sig[ch] == 0):
                        continue  # skip bad channels
                    f, Pxx = welch(sig[ch], fs=fs)
                    trial_psds.append(Pxx)
                if trial_psds:
                    psds.append(np.mean(trial_psds, axis=0))

            if not psds:
                continue
            psds = np.array(psds)  # (n_trials, n_freqs)
            psd_by_combo[cname] = {
                "mean": psds.mean(axis=0).tolist(),
                "std":  psds.std(axis=0).tolist(),
                "freqs": f.tolist(),
            }
            # Coefficient of variation (std/mean) averaged over frequencies and trials
            cv = float(np.mean(psds.std(axis=0) / (psds.mean(axis=0) + 1e-30)))
            psd_by_combo[cname]["cv"] = cv

        results[ph_name] = psd_by_combo

        if CAN_PLOT:
            fig, ax = plt.subplots(figsize=(8, 4))
            for (g, h), cname, color in zip(
                [(0,0),(0,1),(1,0),(1,1)], combo_names.values(), GH_PLOT_COLORS
            ):
                if cname not in psd_by_combo:
                    continue
                d   = psd_by_combo[cname]
                f_  = np.array(d["freqs"])
                m_  = np.array(d["mean"])
                s_  = np.array(d["std"])
                ax.plot(f_, m_, label=cname, color=color)
                ax.fill_between(f_, m_ - s_, m_ + s_, alpha=0.2, color=color)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")
            ax.set_xlim(0, min(500, fs / 2))
            ax.set_title(f"PSD consistency — {ph_name}")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"psd_consistency_{ph_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Print CV table
    print("\nPSD consistency (CV per combination, lower = more consistent):")
    print(f"  {'':>18}" + "".join(f"{PHASE_NAMES[ph]:>12}" for ph in phase_indices))
    print("  " + "-" * (18 + 12 * len(phase_indices)))
    for (g, h), cname in combo_names.items():
        row = f"  {cname:<18}"
        for ph in phase_indices:
            ph_name = PHASE_NAMES[ph]
            cv = results.get(ph_name, {}).get(cname, {}).get("cv", float("nan"))
            row += f"{cv:>12.3f}" if not np.isnan(cv) else f"{'N/A':>12}"
        print(row)
    print("  Target: CV < 0.3 for meaningful generation comparison")

    return results


# ---------------------------------------------------------------------------
# Part F: Independence tests
# ---------------------------------------------------------------------------

def run_independence_tests(rep: dict, phase_indices: list[int]) -> dict:
    """Run all five independence sub-tests (F1–F5)."""
    print("\nPart F: Independence tests ...")
    out: dict = {}

    grip_pred  = rep["grip_pred"]
    hand_pred  = rep["hand_pred"]
    grip_label = rep["y_grip"]
    hand_label = rep["y_hand"]
    phase      = rep["y_phase"]
    emb        = rep["embeddings"]

    # ---- F1: Accuracy breakdown tables ----
    print("\n  F1 — Accuracy breakdown:")
    ph_names = [PHASE_NAMES[i] for i in phase_indices]

    grip_acc_table = {}   # {ph_name: {grip_val: {hand_val: acc}}}
    hand_acc_table = {}

    for ph in phase_indices:
        ph_name = PHASE_NAMES[ph]
        ph_mask = (phase == ph)
        grip_acc_table[ph_name] = {}
        hand_acc_table[ph_name] = {}
        for g in (0, 1):
            grip_acc_table[ph_name][g] = {}
            for h in (0, 1):
                mask = ph_mask & (grip_label == g) & (hand_label == h)
                if mask.sum() == 0:
                    grip_acc_table[ph_name][g][h] = float("nan")
                else:
                    grip_acc_table[ph_name][g][h] = float((grip_pred[mask] == grip_label[mask]).mean())
        for h in (0, 1):
            hand_acc_table[ph_name][h] = {}
            for g in (0, 1):
                mask = ph_mask & (grip_label == g) & (hand_label == h)
                if mask.sum() == 0:
                    hand_acc_table[ph_name][h][g] = float("nan")
                else:
                    hand_acc_table[ph_name][h][g] = float((hand_pred[mask] == hand_label[mask]).mean())

    out["F1_grip_acc_table"] = grip_acc_table
    out["F1_hand_acc_table"] = hand_acc_table

    # Row/column variances
    for g in (0, 1):
        vals = [grip_acc_table.get(ph, {}).get(g, {}).get(h, float("nan"))
                for ph in ph_names for h in (0, 1)]
        vals = [v for v in vals if not np.isnan(v)]
        var  = max(vals) - min(vals) if len(vals) > 1 else float("nan")
        print(f"    Grip acc row-var (grip={GRIP_NAMES[g]}): max-min={var:.3f}  (<0.05 = hand-independent)")
    out["F1_grip_row_variance"] = {
        GRIP_NAMES[g]: float(max(
            [grip_acc_table.get(ph, {}).get(g, {}).get(h, float("nan"))
             for ph in ph_names for h in (0, 1)
             if not np.isnan(grip_acc_table.get(ph, {}).get(g, {}).get(h, float("nan")))],
            default=0.0
        ) - min(
            [grip_acc_table.get(ph, {}).get(g, {}).get(h, float("nan"))
             for ph in ph_names for h in (0, 1)
             if not np.isnan(grip_acc_table.get(ph, {}).get(g, {}).get(h, float("nan")))],
            default=0.0
        ))
        for g in (0, 1)
    }

    # ---- F2: Logit correlation ----
    print("\n  F2 — Grip-hand logit correlation:")
    gl = rep["grip_logits"]
    hl = rep["hand_logits"]
    for ph in phase_indices:
        mask = (phase == ph)
        r, p = pearsonr(gl[mask], hl[mask])
        print(f"    {PHASE_NAMES[ph]:<12}: r={r:+.3f}  p={p:.4f}")
        out[f"F2_corr_{PHASE_NAMES[ph]}"] = {"r": float(r), "p": float(p)}
    r_all, p_all = pearsonr(gl, hl)
    print(f"    {'combined':<12}: r={r_all:+.3f}  p={p_all:.4f}")
    out["F2_corr_combined"] = {"r": float(r_all), "p": float(p_all)}
    if len(np.unique(phase)) > 1:
        r_gp, p_gp = pearsonr(gl, phase)
        r_hp, p_hp = pearsonr(hl, phase)
        print(f"    grip↔phase : r={r_gp:+.3f}  p={p_gp:.4f}")
        print(f"    hand↔phase : r={r_hp:+.3f}  p={p_hp:.4f}")
        out["F2_corr_grip_phase"] = {"r": float(r_gp), "p": float(p_gp)}
        out["F2_corr_hand_phase"] = {"r": float(r_hp), "p": float(p_hp)}
    print("    Interpretation: |r| < 0.1 = independent encoding")

    # ---- F3: Mutual information ----
    print("\n  F3 — Mutual information (normalized, 0=independent 1=redundant):")
    mi_gh  = normalized_mutual_info_score(grip_pred, hand_pred)
    mi_gph = normalized_mutual_info_score(grip_pred, phase)
    mi_hph = normalized_mutual_info_score(hand_pred, phase)
    print(f"    grip ↔ hand:  {mi_gh:.3f}")
    print(f"    grip ↔ phase: {mi_gph:.3f}")
    print(f"    hand ↔ phase: {mi_hph:.3f}")
    print("    Interpretation: NMI < 0.05 = approximately independent")
    out["F3_nmi"] = {"grip_hand": float(mi_gh), "grip_phase": float(mi_gph), "hand_phase": float(mi_hph)}

    # ---- F4: Phase stability (CV across phases) ----
    if len(phase_indices) > 1:
        print("\n  F4 — Phase stability of encoding (CV across phases, lower = more stable):")
        out["F4_phase_stability"] = {}
        for g in (0, 1):
            for h in (0, 1):
                combo = f"{GRIP_NAMES[g]}+{HAND_NAMES[h]}"
                g_accs = [grip_acc_table.get(PHASE_NAMES[ph], {}).get(g, {}).get(h, float("nan"))
                          for ph in phase_indices]
                h_accs = [hand_acc_table.get(PHASE_NAMES[ph], {}).get(h, {}).get(g, float("nan"))
                          for ph in phase_indices]
                g_accs = [v for v in g_accs if not np.isnan(v)]
                h_accs = [v for v in h_accs if not np.isnan(v)]
                cv_g = float(np.std(g_accs) / max(np.mean(g_accs), 1e-8)) if g_accs else float("nan")
                cv_h = float(np.std(h_accs) / max(np.mean(h_accs), 1e-8)) if h_accs else float("nan")
                print(f"    {combo:<20} grip_acc_CV={cv_g:.3f}  hand_acc_CV={cv_h:.3f}")
                out["F4_phase_stability"][combo] = {"cv_grip_acc": cv_g, "cv_hand_acc": cv_h}
        print("    Interpretation: CV < 0.10 = phase-stable encoding")

    # ---- F5: Cross-phase latent distances ----
    if len(phase_indices) > 1:
        print("\n  F5 — Cross-phase centroid distances:")
        out["F5_cross_phase"] = {}
        centroids: dict = {}   # {(g, h, ph): centroid (d_model,)}
        for g in (0, 1):
            for h in (0, 1):
                for ph in phase_indices:
                    mask = (rep["y_grip"] == g) & (rep["y_hand"] == h) & (phase == ph)
                    if mask.sum() > 0:
                        centroids[(g, h, ph)] = emb[mask].mean(axis=0)

        within_combo_dists = []
        for g in (0, 1):
            for h in (0, 1):
                combo = f"{GRIP_NAMES[g]}+{HAND_NAMES[h]}"
                dists = {}
                ph_list = sorted(phase_indices)
                for i, ph1 in enumerate(ph_list):
                    for ph2 in ph_list[i+1:]:
                        key = f"{PHASE_NAMES[ph1]}→{PHASE_NAMES[ph2]}"
                        if (g, h, ph1) in centroids and (g, h, ph2) in centroids:
                            d = float(np.linalg.norm(centroids[(g,h,ph1)] - centroids[(g,h,ph2)]))
                            dists[key] = d
                            within_combo_dists.append(d)
                out["F5_cross_phase"][combo] = dists
                row = f"    {combo:<20}" + "".join(f"{v:.3f}  " for v in dists.values())
                print(row)

        # Between-combination distances (same phase)
        between_combo_dists = []
        for ph in phase_indices:
            ph_combos = [(g, h) for g in (0, 1) for h in (0, 1)
                         if (g, h, ph) in centroids]
            for i in range(len(ph_combos)):
                for j in range(i+1, len(ph_combos)):
                    g1,h1 = ph_combos[i]; g2,h2 = ph_combos[j]
                    d = float(np.linalg.norm(centroids[(g1,h1,ph)] - centroids[(g2,h2,ph)]))
                    between_combo_dists.append(d)

        mean_within  = float(np.mean(within_combo_dists)) if within_combo_dists else float("nan")
        mean_between = float(np.mean(between_combo_dists)) if between_combo_dists else float("nan")
        print(f"\n    Mean within-combination cross-phase dist:  {mean_within:.3f}")
        print(f"    Mean between-combination same-phase dist:  {mean_between:.3f}")
        if not np.isnan(mean_within) and not np.isnan(mean_between):
            verdict = "PHASE-STABLE" if mean_within < mean_between else "PHASE-DEPENDENT"
            print(f"    → {verdict} encoding")
        out["F5_mean_within_combo"] = mean_within
        out["F5_mean_between_combo"] = mean_between

    return out


# ---------------------------------------------------------------------------
# Independence verdict
# ---------------------------------------------------------------------------

def print_verdict(sil: dict, ratios: dict, independence: dict) -> dict:
    """Derive and print the PROCEED/CAUTION/REVISE verdict."""
    print("\n" + "=" * 72)
    print("  INDEPENDENCE AND SEPARABILITY SUMMARY")
    print("=" * 72)

    def _sil(key):  return sil.get("combined", {}).get(key, float("nan"))
    def _rat(key):  return ratios.get("combined", {}).get(key, float("nan"))

    grip_sil = _sil("grip"); hand_sil = _sil("hand"); phase_sil = _sil("phase")
    grip_rat = _rat("grip"); hand_rat = _rat("hand")
    r_gh = independence.get("F2_corr_combined", {}).get("r", float("nan"))
    r_gp = independence.get("F2_corr_grip_phase", {}).get("r", float("nan"))
    r_hp = independence.get("F2_corr_hand_phase", {}).get("r", float("nan"))
    nmi  = independence.get("F3_nmi", {})

    def _sep(s):
        if np.isnan(s): return "UNKNOWN"
        if s > 0.3:     return "SEPARABLE"
        if s > 0.1:     return "WEAK"
        return "NOT SEPARABLE"

    def _phase_dom(s):
        if np.isnan(s): return "UNKNOWN"
        if s > 0.5:     return "DOMINANT"
        if s > 0.2:     return "MODERATE"
        return "WEAK"

    print(f"\n  Silhouette  | grip={grip_sil:.3f}  hand={hand_sil:.3f}  phase={phase_sil:.3f}")
    print(f"  Dist-ratio  | grip={grip_rat:.2f}  hand={hand_rat:.2f}")
    print(f"  Logit corr  | grip↔hand r={r_gh:.3f}  grip↔phase r={r_gp:.3f}  hand↔phase r={r_hp:.3f}")
    print(f"  NMI         | grip↔hand={nmi.get('grip_hand', float('nan')):.3f}  "
          f"grip↔phase={nmi.get('grip_phase', float('nan')):.3f}  "
          f"hand↔phase={nmi.get('hand_phase', float('nan')):.3f}")
    print()

    grip_verdict = _sep(grip_sil)
    hand_verdict = _sep(hand_sil)
    phase_verdict = _phase_dom(phase_sil)

    print(f"  GRIP:  [{grip_verdict}]")
    print(f"  HAND:  [{hand_verdict}]")
    print(f"  PHASE: [{phase_verdict}]")

    # Overall recommendation
    issues = []
    if "NOT" in grip_verdict:  issues.append("grip not separable")
    if "NOT" in hand_verdict:  issues.append("hand not separable")
    if abs(r_gh) > 0.2:        issues.append(f"grip-hand logit correlation high (r={r_gh:.2f})")
    if nmi.get("grip_hand", 0) > 0.10: issues.append("grip-hand NMI elevated")
    if phase_verdict == "DOMINANT": issues.append("phase dominates representation")

    print()
    if not issues:
        print("  OVERALL RECOMMENDATION:")
        print("  → PROCEED WITH CVAE")
        print("    Grip and hand separable, approximately independent.")
        recommendation = "PROCEED"
    elif len(issues) <= 2:
        print("  OVERALL RECOMMENDATION:")
        print("  → PROCEED WITH CAUTION")
        for iss in issues:
            print(f"    Issue: {iss}")
        recommendation = "CAUTION"
    else:
        print("  OVERALL RECOMMENDATION:")
        print("  → REVISE APPROACH")
        for iss in issues:
            print(f"    Issue: {iss}")
        recommendation = "REVISE"
    print("=" * 72)

    return {
        "grip":  {"silhouette": grip_sil,  "dist_ratio": grip_rat,  "verdict": grip_verdict},
        "hand":  {"silhouette": hand_sil,  "dist_ratio": hand_rat,  "verdict": hand_verdict},
        "phase": {"silhouette": phase_sil, "verdict": phase_verdict},
        "logit_corr_grip_hand": float(r_gh),
        "nmi": nmi,
        "recommendation": recommendation,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve available checkpoints
    checkpoints: dict[int, str] = {}
    for phase_idx, path_str in [
        (0, args.checkpoint_prereach),
        (1, args.checkpoint_reach),
        (2, args.checkpoint_grasp),
    ]:
        if path_str and Path(path_str).exists():
            checkpoints[phase_idx] = path_str
        elif path_str:
            print(f"WARNING: checkpoint not found: {path_str}")

    if 1 not in checkpoints:
        sys.exit("Error: --checkpoint_reach must exist.")

    print(f"\nAvailable checkpoints: {[PHASE_NAMES[i] for i in sorted(checkpoints)]}")
    if set(checkpoints) == {1}:
        print("NOTE: only the reach checkpoint is available; cross-phase plots, phase silhouette, "
              "phase-stability tests, and cross-phase centroid tests are limited or omitted.")

    # ---- Load all trials ----
    data_dir = Path(args.data_dir)
    data     = load_all_trials(data_dir)
    n_total  = len(data["y_grip"])
    print(f"Loaded {n_total} trials from {len(data['file_paths'])} class files.")

    broadband_data = None
    broadband_data_dir = Path(args.broadband_data_dir) if args.broadband_data_dir else None
    if broadband_data_dir is not None:
        broadband_data = load_broadband_trials(broadband_data_dir)
        print(
            f"Loaded {len(broadband_data['y_grip'])} broadband trials "
            f"from {len(broadband_data['file_paths'])} class files."
        )

    if args.dry_run and n_total > 500:
        rng  = np.random.default_rng(args.seed)
        keep = rng.choice(n_total, size=500, replace=False)
        keep.sort()
        print(f"[dry-run] Subsampling to 500 trials.")
        for key in ("file_idx", "trial_idx", "y_grip", "y_hand", "y_angle"):
            data[key] = data[key][keep]

    if args.dry_run and broadband_data is not None and len(broadband_data["y_grip"]) > 500:
        rng = np.random.default_rng(args.seed + 1)
        keep = rng.choice(len(broadband_data["y_grip"]), size=500, replace=False)
        keep.sort()
        print("[dry-run] Subsampling broadband data to 500 trials.")
        for key in ("file_idx", "trial_idx", "y_grip", "y_hand", "y_angle"):
            broadband_data[key] = broadband_data[key][keep]

    # ---- Model-free segmented data checks ----
    print("\n" + "=" * 60)
    print("  Model-free segmented-data checks")
    print("=" * 60)
    model_free_results = {
        "mu": run_model_free_separability(data, "mu", out_dir, args)
    }
    if broadband_data is not None:
        model_free_results["broadband"] = run_model_free_separability(
            broadband_data, "broadband", out_dir, args
        )

    # ---- Part A: Collect embeddings ----
    print("\n" + "=" * 60)
    print("  Part A — Representation extraction")
    print("=" * 60)
    rep = collect_all_embeddings(checkpoints, data, args, device)
    phase_indices = sorted(checkpoints.keys())
    phase_names   = [PHASE_NAMES[i] for i in phase_indices]

    emb_dim = rep["embeddings"].shape[1]
    print(f"\nEmbedding dimension: {emb_dim}  |  Total samples: {len(rep['embeddings'])}")

    # ---- Part B: PCA / UMAP visualization ----
    print("\n" + "=" * 60)
    print("  Part B — Visualization (PCA + UMAP/fallback)")
    print("=" * 60)
    pca_coords   = _project(rep["embeddings"], n_components=2)
    umap_coords, umap_label = _umap_or_pca(rep["embeddings"], seed=args.seed)
    tsne_coords = None if args.no_plot else _tsne_or_none(rep["embeddings"], seed=args.seed)

    if not args.no_plot:
        plot_embeddings(rep, pca_coords, umap_coords, umap_label, out_dir, args.no_plot, tsne_coords)

    # ---- Part C: Silhouette scores ----
    print("\n" + "=" * 60)
    print("  Part C — Silhouette scores")
    print("=" * 60)
    sil = compute_silhouette_scores(rep, phase_indices)
    _print_silhouette_table(sil, phase_names)

    # ---- Part D: Distance ratios ----
    print("\n" + "=" * 60)
    print("  Part D — Within-class vs between-class distances")
    print("=" * 60)
    ratios = compute_distance_ratios(rep, phase_indices)
    _print_ratio_table(ratios, phase_names)

    # ---- Part D2: Linear probe separability ----
    print("\n" + "=" * 60)
    print("  Part D2 — Linear-probe separability")
    print("=" * 60)
    probe_scores = compute_linear_probe_scores(rep, phase_indices, seed=args.seed)
    _print_probe_table(probe_scores, phase_names)

    # ---- Part E: PSD consistency ----
    print("\n" + "=" * 60)
    print("  Part E — PSD consistency (model-free)")
    print("=" * 60)
    psd_results = compute_psd_consistency(
        data, phase_indices, out_dir, args.no_plot, fs=args.fs
    )

    # ---- Part F: Independence tests ----
    print("\n" + "=" * 60)
    print("  Part F — Independence tests")
    print("=" * 60)
    independence = run_independence_tests(rep, phase_indices)

    # ---- Verdict ----
    verdict = print_verdict(sil, ratios, independence)

    # ---- Save summary ----
    summary = {
        "separability": {
            "model_free":          model_free_results,
            "silhouette_scores": sil,
            "distance_ratios":   ratios,
            "linear_probe_scores": probe_scores,
            "psd_consistency":   psd_results,
            "embedding_dim":     emb_dim,
            "n_trials":          int(len(rep["embeddings"])),
            "phase_indices":     phase_indices,
        },
        "independence": independence,
        "verdict": verdict,
        "args": {
            "checkpoints":  {PHASE_NAMES[i]: p for i, p in checkpoints.items()},
            "data_dir":     str(data_dir),
            "broadband_data_dir": str(broadband_data_dir) if broadband_data_dir else None,
            "seed":         args.seed,
            "dry_run":      args.dry_run,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nAll results saved to {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
