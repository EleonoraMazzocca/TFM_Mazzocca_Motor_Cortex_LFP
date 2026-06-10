"""Train and evaluate a raw-waveform phase-specialist LFP transformer.

Usage:
    python run_rawwave.py --phase reach --broadband_data_dir /path/to/broadband
    python run_rawwave.py --phase reach --broadband_data_dir /path/to/bb --no_heldout
    python run_rawwave.py --phase reach --broadband_data_dir /path/to/bb --dry_run --no_heldout --no_plot
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from data import (
    ANGLE_TO_ID, GRIP_TO_ID, HAND_TO_ID,
    PHASE_NAMES,
)
from rawwave_data import (
    AREA_IDX_256, AREA_SIZES_REAL, N_AREAS, N_REAL_CHANNELS, N_TIMEPOINTS, FS,
    compute_rawwave_norm_stats,
    extract_and_cache_rawwave,
    load_rawwave_dataset,
    LFPRawwaveDataset,
    PermutedRawwaveDataset,
)
from rawwave_model import LFPRawwaveTransformer

AREA_NAMES = ["PMvR", "M1", "PMdR", "PMdL"]
HEAD_NAMES = ["grip", "hand", "angle"]
HEAD_CLASS_NAMES: dict[str, list[str]] = {
    "grip":  ["power", "precision"],
    "hand":  ["left", "right"],
    "angle": ["0°", "45°", "90°", "135°"],
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a raw-waveform phase-specialist LFP transformer.")
    p.add_argument("--phase", choices=PHASE_NAMES, required=True)
    p.add_argument("--broadband_data_dir", type=str, required=True,
                   help="Directory containing broadband *_degrees.npy files.")
    split_group = p.add_mutually_exclusive_group()
    split_group.add_argument("--heldout", type=str, default="precision_right_135",
                             help="Held-out combination, e.g. 'precision_right_135'.")
    split_group.add_argument("--no_heldout", action="store_true",
                             help="Standard stratified 80/10/10 split; no combination held out.")
    p.add_argument("--norm_mode", choices=["per_channel", "per_timepoint"],
                   default="per_channel",
                   help="per_channel preserves waveform shape (default); "
                        "per_timepoint removes mean waveform (follow-up control only).")
    p.add_argument("--n_permutations",   type=int, default=100,
                   help="Permutation test iterations (heldout mode).")
    p.add_argument("--n_perm_noheldout", type=int, default=20,
                   help="Lightweight permutation check iterations (no_heldout mode).")
    p.add_argument("--epochs",        type=int,   default=40)
    p.add_argument("--patience",      type=int,   default=8)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--d_model",       type=int,   default=32)
    p.add_argument("--n_heads",       type=int,   default=4)
    p.add_argument("--n_layers",      type=int,   default=2)
    p.add_argument("--feedforward_dim", type=int, default=64)
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--n_angle_classes", type=int, default=4,
                   help="Only 4 supported; any other value raises an error.")
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--cache_dir",     type=str,   default="/tmp/lfp_rawwave_cache")
    p.add_argument("--out_dir",       type=str,   default=None)
    p.add_argument("--device",        type=str,   default=None,
                   help="cuda | cpu (default: auto-detect).")
    p.add_argument("--no_plot",  action="store_true")
    p.add_argument("--dry_run",  action="store_true",
                   help="2 epochs, patience=1, 3 permutations, 2 perm-check iterations.")
    return p.parse_args()


def _resolve_heldout(heldout_str: str) -> tuple[int, int, int]:
    parts = heldout_str.strip().split("_")
    if len(parts) != 3:
        raise ValueError(f"--heldout must be 'grip_hand_angle', got {heldout_str!r}")
    gn, hn, an = parts
    for name, mapping in [("grip", GRIP_TO_ID), ("hand", HAND_TO_ID), ("angle", ANGLE_TO_ID)]:
        val = locals()[f"{name[0]}n"]
        if val not in mapping:
            raise ValueError(f"--heldout: invalid {name} {val!r}")
    return GRIP_TO_ID[gn], HAND_TO_ID[hn], ANGLE_TO_ID[an]


# ---------------------------------------------------------------------------
# Data splits (return index arrays into the full feature array)
# ---------------------------------------------------------------------------

def _compositional_split_indices(
    data: dict, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(data["y_grip"]))
    heldout_idx   = all_idx[data["is_heldout"]]
    remaining_idx = all_idx[~data["is_heldout"]]

    remaining_combo = np.array([
        f"{data['y_grip'][i]}_{data['y_hand'][i]}_{data['y_angle'][i]}"
        for i in remaining_idx
    ])
    ho_val_idx, ho_test_idx = train_test_split(
        heldout_idx, test_size=0.5, random_state=seed, shuffle=True,
    )
    rem_train_idx, rem_temp_idx = train_test_split(
        remaining_idx, test_size=0.15, random_state=seed, shuffle=True,
        stratify=remaining_combo,
    )
    rem_temp_combo = np.array([
        f"{data['y_grip'][i]}_{data['y_hand'][i]}_{data['y_angle'][i]}"
        for i in rem_temp_idx
    ])
    rem_val_idx, rem_test_idx = train_test_split(
        rem_temp_idx, test_size=0.5, random_state=seed, shuffle=True,
        stratify=rem_temp_combo,
    )

    train_idx        = rem_train_idx
    val_idx          = np.concatenate([rem_val_idx, ho_val_idx])
    seen_test_idx    = rem_test_idx
    heldout_test_idx = ho_test_idx

    print(
        "Split sizes | "
        f"train={len(train_idx)} val={len(val_idx)} "
        f"seen_test={len(seen_test_idx)} heldout_test={len(heldout_test_idx)}"
    )
    return train_idx, val_idx, seen_test_idx, heldout_test_idx


def _standard_split_indices(
    data: dict, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data["y_grip"])
    combo = data["y_grip"] * 8 + data["y_hand"] * 4 + data["y_angle"]
    all_idx = np.arange(n)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_rel, temp_idx = next(sss.split(all_idx, combo))
    train_idx = all_idx[train_rel]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(sss2.split(temp_idx, combo[temp_idx]))
    val_idx  = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    print(f"Split sizes | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = total_correct = total_n = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y_grip, y_hand, y_angle in loader:
            x       = x.to(device)
            y_grip  = y_grip.to(device)
            y_hand  = y_hand.to(device)
            y_angle = y_angle.to(device)

            lg, lh, la = model(x)
            loss = (
                nn.functional.cross_entropy(lg, y_grip,  label_smoothing=0.1)
                + nn.functional.cross_entropy(lh, y_hand,  label_smoothing=0.1)
                + nn.functional.cross_entropy(la, y_angle, label_smoothing=0.1)
            )
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            n = x.shape[0]
            total_loss += loss.item() * n
            total_correct += (
                (lg.argmax(1) == y_grip).float()
                + (lh.argmax(1) == y_hand).float()
                + (la.argmax(1) == y_angle).float()
            ).sum().item() / 3.0
            total_n += n

    return total_loss / total_n, total_correct / total_n


def train_model(
    model: nn.Module,
    train_ds: LFPRawwaveDataset,
    val_ds: LFPRawwaveDataset,
    config: dict,
    save_path: str | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Training on {device}")
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-3),
    )
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    for epoch in range(config["epochs"]):
        tl, ta = _run_epoch(model, train_loader, device, optimizer=optimizer)
        vl, va = _run_epoch(model, val_loader,   device, optimizer=None)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        if verbose:
            print(
                f"Epoch {epoch + 1:02d}/{config['epochs']} | "
                f"train_loss={tl:.4f}  train_acc={ta:.4f} | "
                f"val_loss={vl:.4f}  val_acc={va:.4f}"
            )

        if vl < best_val_loss:
            best_val_loss = vl
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path:
        torch.save({"model_state": best_state, "config": config, "history": history}, save_path)
        if verbose:
            print(f"Saved checkpoint to {save_path}")

    return model, history, device


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    dataset: LFPRawwaveDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, dict]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds  = {h: [] for h in HEAD_NAMES}
    all_labels = {h: [] for h in HEAD_NAMES}

    with torch.no_grad():
        for x, y_grip, y_hand, y_angle in loader:
            x = x.to(device)
            lg, lh, la = model(x)
            all_preds["grip"].append(lg.argmax(1).cpu().numpy())
            all_preds["hand"].append(lh.argmax(1).cpu().numpy())
            all_preds["angle"].append(la.argmax(1).cpu().numpy())
            all_labels["grip"].append(y_grip.numpy())
            all_labels["hand"].append(y_hand.numpy())
            all_labels["angle"].append(y_angle.numpy())

    preds  = {h: np.concatenate(v) for h, v in all_preds.items()}
    labels = {h: np.concatenate(v) for h, v in all_labels.items()}
    results: dict[str, dict] = {}
    for head in HEAD_NAMES:
        yt, yp = labels[head], preds[head]
        n_cls = len(HEAD_CLASS_NAMES[head])
        results[head] = {
            "accuracy":         float((yt == yp).mean()),
            "confusion_matrix": confusion_matrix(yt, yp, labels=np.arange(n_cls)),
            "y_true": yt,
            "y_pred": yp,
        }
    return results


# ---------------------------------------------------------------------------
# Permutation test — heldout mode (full, 100 iterations)
# ---------------------------------------------------------------------------

def run_permutation_test(
    train_ds: LFPRawwaveDataset,
    val_ds: LFPRawwaveDataset,
    heldout_test_ds: LFPRawwaveDataset,
    model_kwargs: dict,
    train_config: dict,
    n_permutations: int,
    seed: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 10,
) -> dict[str, list[float]]:
    null: dict[str, list[float]] = {h: [] for h in HEAD_NAMES}
    start_i = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            saved   = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            null    = saved["null_distributions"]
            start_i = saved["completed"]
            print(f"  Resuming from iteration {start_i}")
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    for _ in range(start_i * 6):
        rng.integers(0, 1)

    t0 = time.time()
    for i in range(start_i, n_permutations):
        perm_train = PermutedRawwaveDataset(train_ds, rng)
        perm_val   = PermutedRawwaveDataset(val_ds,   rng)

        torch.manual_seed(seed + i + 1)
        model_perm = LFPRawwaveTransformer(**model_kwargs).to(device)
        model_perm, _, _ = train_model(
            model_perm, perm_train, perm_val,
            train_config, save_path=None, device=device, verbose=False,
        )
        perm_res = evaluate_model(model_perm, heldout_test_ds, device, train_config["batch_size"])
        for h in HEAD_NAMES:
            null[h].append(perm_res[h]["accuracy"])
        del model_perm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed   = time.time() - t0
        avg       = elapsed / (i - start_i + 1)
        remaining = avg * (n_permutations - i - 1)
        print(
            f"  [{i + 1:>{len(str(n_permutations))}}/{n_permutations}]  "
            f"elapsed={elapsed:.0f}s  est_remaining={remaining:.0f}s",
            end="\r", flush=True,
        )
        if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
            checkpoint_path.write_text(
                json.dumps({"completed": i + 1, "null_distributions": null}, indent=2),
                encoding="utf-8",
            )

    print(f"\n  {n_permutations}/{n_permutations} permutations done ({time.time() - t0:.0f}s).")
    return null


# ---------------------------------------------------------------------------
# Permutation check — no_heldout mode (lightweight, 20 iterations)
# ---------------------------------------------------------------------------

def run_permutation_check_noheldout(
    train_ds: LFPRawwaveDataset,
    val_ds: LFPRawwaveDataset,
    test_ds: LFPRawwaveDataset,
    model_kwargs: dict,
    train_config: dict,
    n_iterations: int,
    seed: int,
    device: torch.device,
) -> dict[str, list[float]]:
    null: dict[str, list[float]] = {h: [] for h in HEAD_NAMES}
    rng = np.random.default_rng(seed)
    t0  = time.time()

    for i in range(n_iterations):
        perm_train = PermutedRawwaveDataset(train_ds, rng)
        perm_val   = PermutedRawwaveDataset(val_ds,   rng)

        torch.manual_seed(seed + i + 1)
        model_perm = LFPRawwaveTransformer(**model_kwargs).to(device)
        model_perm, _, _ = train_model(
            model_perm, perm_train, perm_val,
            train_config, save_path=None, device=device, verbose=False,
        )
        perm_res = evaluate_model(model_perm, test_ds, device, train_config["batch_size"])
        for h in HEAD_NAMES:
            null[h].append(perm_res[h]["accuracy"])
        del model_perm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed   = time.time() - t0
        avg       = elapsed / (i + 1)
        remaining = avg * (n_iterations - i - 1)
        print(
            f"  [{i + 1}/{n_iterations}]  "
            f"elapsed={elapsed:.0f}s  est_remaining={remaining:.0f}s",
            end="\r", flush=True,
        )

    print(f"\n  {n_iterations}/{n_iterations} permutation check done ({time.time() - t0:.0f}s).")
    return null


# ---------------------------------------------------------------------------
# Attention analysis
# ---------------------------------------------------------------------------

def collect_attention_weights(
    model: LFPRawwaveTransformer,
    dataset: LFPRawwaveDataset,
    device: torch.device,
    batch_size: int = 64,
) -> list[np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    layer_batches: list[list[np.ndarray]] | None = None

    with torch.no_grad():
        for x, *_ in loader:
            x = x.to(device)
            model(x)
            batch_weights = model.get_layer_attention_weights()
            if layer_batches is None:
                layer_batches = [[] for _ in batch_weights]
            for i, w in enumerate(batch_weights):
                layer_batches[i].append(w.cpu().numpy())

    if layer_batches is None:
        return []
    return [np.concatenate(b, axis=0) for b in layer_batches]


def compute_area_importance(
    layer_weights: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-area importance from last attention layer.

    All 256 tokens are real — no padding filter needed.

    Returns:
        importance  — (N_AREAS,) normalized
        attn_matrix — (N_AREAS, N_AREAS) area-level mean attention
    """
    attn      = layer_weights[-1]             # (n_samples, n_heads, 256, 256)
    attn_full = attn.mean(axis=(0, 1))        # (256, 256)
    col_mean  = attn_full.mean(axis=0)        # (256,) attention received per channel

    importance  = np.zeros(N_AREAS, dtype=np.float32)
    attn_matrix = np.zeros((N_AREAS, N_AREAS), dtype=np.float32)
    start_i = 0
    for i, sz_i in enumerate(AREA_SIZES_REAL):
        importance[i] = col_mean[start_i:start_i + sz_i].mean()
        start_j = 0
        for j, sz_j in enumerate(AREA_SIZES_REAL):
            attn_matrix[i, j] = attn_full[
                start_i:start_i + sz_i,
                start_j:start_j + sz_j,
            ].mean()
            start_j += sz_j
        start_i += sz_i

    importance = importance / (importance.sum() + 1e-10)
    return importance, attn_matrix


# ---------------------------------------------------------------------------
# Input gradient saliency — per-channel and per-area
# ---------------------------------------------------------------------------

def compute_channel_saliency(
    model: LFPRawwaveTransformer,
    dataset: LFPRawwaveDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, dict]:
    """Mean absolute input gradient per channel and per area, for each head.

    Uses plain |grad| (not |grad × input|) so scores are comparable across
    channels with different amplitude scales.

    Returns per head: {"imp_channel": (256,), "imp_area": (N_AREAS,)} normalized.
    """
    model.eval()
    criterion  = nn.CrossEntropyLoss()
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    head_fns   = {
        "grip":  model.head_grip,
        "hand":  model.head_hand,
        "angle": model.head_angle,
    }
    accum_ch      = {h: np.zeros(N_REAL_CHANNELS, dtype=np.float64) for h in HEAD_NAMES}
    n_samples_total = 0

    for x, y_grip, y_hand, y_angle in loader:
        x        = x.to(device)
        batch_n  = x.shape[0]
        targets  = {
            "grip":  y_grip.to(device),
            "hand":  y_hand.to(device),
            "angle": y_angle.to(device),
        }

        for head in HEAD_NAMES:
            x_inp = x.detach().requires_grad_(True)
            with torch.enable_grad():
                tok    = model.input_proj(x_inp)                              # (batch, 256, d_model)
                tok    = tok + model.area_embedding(model.area_idx)[None, :, :]
                for layer in model.layers:
                    tok = layer(tok)
                pooled = model.norm(tok.mean(dim=1))
                logits = head_fns[head](pooled)
                loss   = criterion(logits, targets[head])
                model.zero_grad()
                loss.backward()

            if x_inp.grad is not None:
                # sum over batch and timepoints → (256,); divide by totals at the end
                accum_ch[head] += x_inp.grad.abs().sum(dim=(0, 2)).detach().cpu().numpy()

        n_samples_total += batch_n

    result: dict[str, dict] = {}
    for head in HEAD_NAMES:
        denom  = max(n_samples_total * N_TIMEPOINTS, 1)
        ch_imp = (accum_ch[head] / denom).astype(np.float32)
        imp_area = np.zeros(N_AREAS, dtype=np.float32)
        start = 0
        for a, sz in enumerate(AREA_SIZES_REAL):
            imp_area[a] = ch_imp[start:start + sz].mean()
            start += sz
        imp_area = imp_area / (imp_area.sum() + 1e-10)
        result[head] = {"imp_channel": ch_imp, "imp_area": imp_area}
    return result


# ---------------------------------------------------------------------------
# Filter frequency analysis
# ---------------------------------------------------------------------------

def analyze_filter_frequencies(
    model: LFPRawwaveTransformer,
    fs: float = FS,
    out_dir: Path | None = None,
    norm_mode: str = "per_channel",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """FFT of learned temporal filters in input_proj.weight.

    Returns (freqs, mean_spectrum, dominant_freqs, peak_freq, note_str).
    Interpretation is approximate for per_channel normalization
    (constant scale factor per channel); exact for per_timepoint normalization.
    """
    weights         = model.input_proj.weight.detach().cpu().numpy()  # (d_model, 500)
    n_filters       = weights.shape[0]
    freqs           = np.fft.rfftfreq(N_TIMEPOINTS, d=1.0 / fs)       # (251,)
    spectra         = np.abs(np.fft.rfft(weights, axis=1))             # (d_model, 251)
    mean_spectrum   = spectra.mean(axis=0)                             # (251,)
    dominant_freqs  = freqs[spectra.argmax(axis=1)]                    # (d_model,)
    peak_freq       = float(freqs[mean_spectrum.argmax()])

    if norm_mode == "per_channel":
        note_str = ("approximate — per_channel normalization "
                    "(constant scale factor per channel, waveform shape preserved)")
    else:
        note_str = "exact — per_timepoint normalization"

    dom_str = "  ".join(f"{f:.1f}" for f in sorted(dominant_freqs))
    print(f"\nFilter frequency analysis (input_proj.weight, {n_filters} filters):")
    print(f"  Dominant frequencies (Hz): {dom_str}")
    print(f"  Mean spectrum peak: {peak_freq:.1f} Hz")
    print(f"  Note: {note_str}")

    if out_dir is not None:
        plt = _get_plt()
        if plt is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            ax = axes[0]
            ax.plot(freqs, mean_spectrum)
            ax.axvline(peak_freq, color="red", linestyle="--", alpha=0.7,
                       label=f"peak={peak_freq:.1f} Hz")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Mean |FFT| of learned filters")
            ax.set_title(f"Mean spectrum across {n_filters} learned temporal filters")
            ax.legend()
            ax = axes[1]
            ax.hist(dominant_freqs, bins=20, color="#4C72B0", edgecolor="white")
            ax.set_xlabel("Dominant frequency (Hz)")
            ax.set_ylabel("# filters")
            ax.set_title(f"Distribution of dominant frequencies ({n_filters} filters)")
            plt.tight_layout()
            save_path = str(out_dir / "filter_frequency_analysis.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {save_path}")

    return freqs, mean_spectrum, dominant_freqs, peak_freq, note_str


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_saliency_table(
    saliency: dict[str, dict],
    split_label: str = "seen test",
) -> None:
    W = 7
    col_header = "".join(f"{a:>{W}}" for a in AREA_NAMES)
    print(f"\n  Area importance (input gradient saliency, |grad|, {split_label}):")
    print(f"    {'Head':<8s}{col_header}")
    print("    " + "-" * (8 + W * len(AREA_NAMES)))
    for head in HEAD_NAMES:
        imp = saliency[head]["imp_area"]
        row = f"    {head.upper():<8s}" + "".join(f"{v:>{W}.3f}" for v in imp)
        print(row)


def _get_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return None


def plot_training_history(history: dict, save_path: str) -> None:
    plt = _get_plt()
    if plt is None:
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [
        (axes[0], "loss", "Loss"),
        (axes[1], "acc",  "Accuracy (avg across heads)"),
    ]:
        ax.plot(epochs, history[f"train_{key}"], label="train")
        ax.plot(epochs, history[f"val_{key}"],   label="val")
        ax.set_xlabel("Epoch"); ax.set_title(title); ax.legend()
        if key == "acc":
            ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_confusion_matrix(cm, target_names, title, save_path) -> None:
    plt = _get_plt()
    if plt is None:
        return
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(target_names))); ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticks(range(len(target_names))); ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n{cm_norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.6 else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_saliency_importance(
    saliency: dict[str, dict],
    phase_name: str,
    save_path: str,
    split_label: str = "seen test",
) -> None:
    plt = _get_plt()
    if plt is None:
        return
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, head in zip(axes, HEAD_NAMES):
        imp  = saliency[head]["imp_area"]
        bars = ax.bar(AREA_NAMES, imp, color=colors)
        ax.set_xlabel("Brain area")
        ax.set_ylabel("Normalized |grad| importance")
        ax.set_title(f"{phase_name} — {head}\n(input gradient saliency, {split_label})")
        ax.set_ylim(0, min(1.0, imp.max() * 1.45 + 0.05))
        for bar, v in zip(bars, imp):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_attention_analysis(importance, attn_matrix, area_names, title_prefix, save_path) -> None:
    plt = _get_plt()
    if plt is None:
        return
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    bars = ax.bar(area_names, importance, color=colors)
    ax.set_ylabel("Normalized attention received")
    ax.set_title(f"{title_prefix} — Area importance")
    ax.set_ylim(0, min(1.0, importance.max() * 1.4 + 0.05))
    for bar, v in zip(bars, importance):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax = axes[1]
    vmax = max(attn_matrix.max(), 1e-6)
    im   = ax.imshow(attn_matrix, cmap="Blues", vmin=0.0, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(area_names))); ax.set_xticklabels(area_names)
    ax.set_yticks(range(len(area_names))); ax.set_yticklabels(area_names)
    ax.set_xlabel("Key area"); ax.set_ylabel("Query area")
    ax.set_title(f"{title_prefix} — Attention matrix")
    for i in range(len(area_names)):
        for j in range(len(area_names)):
            ax.text(j, i, f"{attn_matrix[i,j]:.2f}", ha="center", va="center",
                    color="white" if attn_matrix[i, j] > vmax * 0.65 else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.n_angle_classes != 4:
        sys.exit("Error: --n_angle_classes != 4 is not supported in rawwave specialist.")

    if args.no_heldout:
        heldout_grip = heldout_hand = heldout_angle = -1
        heldout_label = "none"
    else:
        try:
            heldout_grip, heldout_hand, heldout_angle = _resolve_heldout(args.heldout)
        except ValueError as exc:
            sys.exit(f"Error: {exc}")
        heldout_label = args.heldout

    if args.dry_run:
        args.epochs          = 2
        args.patience        = 1
        args.n_permutations  = 3
        args.n_perm_noheldout = 2

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cpu":
        print(
            "WARNING: 256-token attention on CPU is slow. GPU strongly recommended.\n"
            "Estimated time per phase on CPU: 4-8h. On GPU: 5-10 min."
        )

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.no_heldout:
        out_dir = _HERE / "results" / f"rawwave_{args.phase}_no_heldout"
    else:
        out_dir = _HERE / "results" / f"rawwave_{args.phase}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    phase_idx  = PHASE_NAMES.index(args.phase)
    input_desc = "rawwave(500tp×256ch)"

    print(f"\n{'='*60}")
    print(f"  Rawwave specialist: {args.phase.upper()}")
    if args.dry_run:
        print(
            f"  [dry-run] phase={args.phase.upper()} input={input_desc} d_model={args.d_model}\n"
            f"            norm={args.norm_mode} device={device}"
        )
    if args.no_heldout:
        print(f"  Mode: standard 80/10/10 split (no held-out combination)")
    else:
        print(f"  Held-out combination: {heldout_label}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data = load_rawwave_dataset(
        data_dir=args.broadband_data_dir,
        heldout_grip=max(heldout_grip, 0),
        heldout_hand=max(heldout_hand, 0),
        heldout_angle=max(heldout_angle, 0),
    )
    print(f"Dataset: {len(data['y_grip'])} trials")

    features = extract_and_cache_rawwave(
        data, phase_idx,
        cache_dir=Path(args.cache_dir),
        data_dir=args.broadband_data_dir,
    )

    if args.no_heldout:
        train_idx, val_idx, test_idx = _standard_split_indices(data, args.seed)
        seen_test_idx = heldout_test_idx = None
    else:
        train_idx, val_idx, seen_test_idx, heldout_test_idx = _compositional_split_indices(
            data, args.seed
        )
        test_idx = None

    norm_stats = compute_rawwave_norm_stats(features, train_idx, norm_mode=args.norm_mode)
    np.savez_compressed(
        out_dir / "normalization_stats.npz",
        mu=norm_stats["mu"], sigma=norm_stats["sigma"],
    )

    train_ds = LFPRawwaveDataset(features, train_idx, data, norm_stats)
    val_ds   = LFPRawwaveDataset(features, val_idx,   data, norm_stats)
    if args.no_heldout:
        test_ds         = LFPRawwaveDataset(features, test_idx,         data, norm_stats)
        seen_test_ds    = heldout_test_ds = None
        print(f"Datasets: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    else:
        test_ds         = None
        seen_test_ds    = LFPRawwaveDataset(features, seen_test_idx,    data, norm_stats)
        heldout_test_ds = LFPRawwaveDataset(features, heldout_test_idx, data, norm_stats)
        print(f"Datasets: train={len(train_ds)}  val={len(val_ds)}  "
              f"seen_test={len(seen_test_ds)}  heldout_test={len(heldout_test_ds)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_kwargs = dict(
        n_angle_classes=4,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    model    = LFPRawwaveTransformer(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}  (input={input_desc})")

    train_config = {
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "epochs":       args.epochs,
        "patience":     args.patience,
        "weight_decay": args.weight_decay,
    }

    summary_path = out_dir / "summary.json"

    def _save_summary(s: dict) -> None:
        summary_path.write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"Saved {summary_path}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print()
    model, history, device = train_model(
        model, train_ds, val_ds, train_config,
        save_path=str(out_dir / "checkpoint.pt"),
        device=device, verbose=True,
    )
    if not args.no_plot:
        plot_training_history(history, str(out_dir / "training_curves.png"))

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    if args.no_heldout:
        print("\nEvaluating on test set (standard split)...")
        test_results    = evaluate_model(model, test_ds,    device, args.batch_size)
        seen_results    = heldout_results = None
        print(f"\n  Test (standard 80-10-10 split):")
        for h in HEAD_NAMES:
            print(f"    {h:6s}: {test_results[h]['accuracy']:.4f}")
        eval_pairs = [("test", test_results)]
    else:
        print("\nEvaluating on seen combinations...")
        seen_results    = evaluate_model(model, seen_test_ds,    device, args.batch_size)
        print("Evaluating on held-out combination...")
        heldout_results = evaluate_model(model, heldout_test_ds, device, args.batch_size)
        test_results    = None
        for split_name, results in [("Seen", seen_results), ("Held-out", heldout_results)]:
            print(f"\n  {split_name}:")
            for h in HEAD_NAMES:
                print(f"    {h:6s}: {results[h]['accuracy']:.4f}")
        real_acc   = {h: heldout_results[h]["accuracy"] for h in HEAD_NAMES}
        eval_pairs = [("seen", seen_results), ("heldout", heldout_results)]

    for split_name, results in eval_pairs:
        for h in HEAD_NAMES:
            np.save(out_dir / f"{split_name}_{h}_confusion_matrix.npy",
                    results[h]["confusion_matrix"])
    if not args.no_plot:
        for split_name, results in eval_pairs:
            for h in HEAD_NAMES:
                plot_confusion_matrix(
                    results[h]["confusion_matrix"],
                    target_names=HEAD_CLASS_NAMES[h],
                    title=f"{args.phase} — {h} ({split_name})",
                    save_path=str(out_dir / f"{split_name}_{h}_confusion_matrix.png"),
                )

    _model_meta = {
        "n_tokens": N_REAL_CHANNELS, "n_timepoints": N_TIMEPOINTS,
        "d_model": args.d_model, "n_heads": args.n_heads,
        "n_layers": args.n_layers, "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout, "weight_decay": args.weight_decay,
        "n_params": n_params,
    }
    if args.no_heldout:
        summary: dict = {
            "phase": args.phase, "phase_idx": phase_idx,
            "mode": "no_heldout", "input_mode": input_desc, "norm_mode": args.norm_mode,
            "n_angle_classes": 4,
            "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds),
            "test_accuracy": {h: test_results[h]["accuracy"] for h in HEAD_NAMES},
            "no_heldout_permutation_check": None,
            "area_importance": None, "attention_matrix": None,
            "channel_importance_gradient_saliency": None,
            "filter_frequency_analysis": None,
            "model": _model_meta, "config": train_config, "seed": args.seed,
        }
    else:
        summary = {
            "phase": args.phase, "phase_idx": phase_idx,
            "mode": "heldout", "heldout_label": heldout_label,
            "input_mode": input_desc, "norm_mode": args.norm_mode,
            "n_angle_classes": 4,
            "train_size": len(train_ds), "val_size": len(val_ds),
            "seen_test_size": len(seen_test_ds), "heldout_test_size": len(heldout_test_ds),
            "seen_accuracy":    {h: seen_results[h]["accuracy"]    for h in HEAD_NAMES},
            "heldout_accuracy": {h: heldout_results[h]["accuracy"] for h in HEAD_NAMES},
            "p_values": None, "permutation_test": None,
            "area_importance_seen": None, "area_importance_heldout": None,
            "attention_matrix_seen": None, "attention_matrix_heldout": None,
            "channel_importance_gradient_saliency": None,
            "filter_frequency_analysis": None,
            "model": _model_meta, "config": train_config,
            "seed": args.seed, "n_permutations": args.n_permutations,
        }
    print()
    _save_summary(summary)

    # ------------------------------------------------------------------
    # Permutation test (heldout mode) / check (no_heldout mode)
    # ------------------------------------------------------------------
    if not args.no_heldout:
        perm_ckpt = out_dir / "null_distributions_checkpoint.json"
        print(f"\nRunning {args.n_permutations} permutation iterations ...")
        null_distributions = run_permutation_test(
            train_ds=train_ds, val_ds=val_ds, heldout_test_ds=heldout_test_ds,
            model_kwargs=model_kwargs, train_config=train_config,
            n_permutations=args.n_permutations, seed=args.seed + 1,
            device=device, checkpoint_path=perm_ckpt, checkpoint_every=10,
        )
        p_values = {
            h: float((np.array(null_distributions[h]) >= real_acc[h]).mean())
            if null_distributions[h] else 1.0
            for h in HEAD_NAMES
        }
        print("\n  Permutation test results (held-out):")
        for h in HEAD_NAMES:
            sig = " *" if p_values[h] < 0.05 else ""
            print(f"    {h:6s}: acc={real_acc[h]:.4f}  p={p_values[h]:.4f}{sig}")
        summary["p_values"] = p_values
        summary["permutation_test"] = {
            "method": "retrain on shuffled training labels, evaluate on real held-out labels",
            "n_permutations": args.n_permutations,
            "null_distributions": {h: null_distributions[h] for h in HEAD_NAMES},
        }
        _save_summary(summary)

    else:
        n_perm = args.n_perm_noheldout
        print(f"\nRunning {n_perm} no-heldout permutation check iterations ...")
        null_nh = run_permutation_check_noheldout(
            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            model_kwargs=model_kwargs, train_config=train_config,
            n_iterations=n_perm, seed=args.seed + 1, device=device,
        )
        real_test_acc = {h: test_results[h]["accuracy"] for h in HEAD_NAMES}
        print(f"\nNo-heldout permutation check ({n_perm} iterations):")
        perm_check: dict = {"n_iterations": n_perm}
        for h in HEAD_NAMES:
            null_arr = np.array(null_nh[h])
            print(f"  {h:6s}: real={real_test_acc[h]:.4f}  "
                  f"null_mean={null_arr.mean():.4f}  null_std={null_arr.std():.4f}")
            perm_check[h] = {
                "real":      real_test_acc[h],
                "null_mean": float(null_arr.mean()),
                "null_std":  float(null_arr.std()),
            }
        summary["no_heldout_permutation_check"] = perm_check
        _save_summary(summary)

    # ------------------------------------------------------------------
    # Attention + saliency analysis
    # ------------------------------------------------------------------
    eval_ds_for_attn = test_ds if args.no_heldout else seen_test_ds
    attn_split_label = "test" if args.no_heldout else "seen test"

    print(f"\nExtracting attention weights ({attn_split_label})...")
    layer_attn = collect_attention_weights(model, eval_ds_for_attn, device, args.batch_size)
    importance, attn_matrix = compute_area_importance(layer_attn)

    print(f"\n  Area importance — {args.phase} ({attn_split_label}, normalized):")
    for name, imp in zip(AREA_NAMES, importance):
        bar = "█" * int(imp * 40)
        print(f"    {name:6s}: {imp:.4f}  {bar}")

    print(f"\nComputing input gradient saliency ({attn_split_label})...")
    saliency = compute_channel_saliency(model, eval_ds_for_attn, device, args.batch_size)
    _print_saliency_table(saliency, split_label=attn_split_label)

    if not args.no_heldout:
        print("\nExtracting attention weights (held-out test)...")
        heldout_layer_attn = collect_attention_weights(
            model, heldout_test_ds, device, args.batch_size
        )
        heldout_importance, heldout_attn_matrix = compute_area_importance(heldout_layer_attn)

    # ------------------------------------------------------------------
    # Filter frequency analysis
    # ------------------------------------------------------------------
    print(f"\nAnalyzing learned filter frequencies ...")
    plot_out = out_dir if not args.no_plot else None
    freqs, mean_spectrum, dominant_freqs, peak_freq, freq_note = analyze_filter_frequencies(
        model, fs=FS, out_dir=plot_out, norm_mode=args.norm_mode,
    )

    # ------------------------------------------------------------------
    # Update summary
    # ------------------------------------------------------------------
    attn_note = (
        "attn_matrix[i][j] = attention from area i (query) to area j (key), "
        "averaged over samples and heads (last layer)"
    )
    if args.no_heldout:
        summary["area_importance"] = dict(zip(AREA_NAMES, importance.tolist()))
        summary["attention_matrix"] = {
            "areas": AREA_NAMES, "matrix": attn_matrix.tolist(), "note": attn_note,
        }
    else:
        summary["area_importance_seen"]     = dict(zip(AREA_NAMES, importance.tolist()))
        summary["area_importance_heldout"]  = dict(zip(AREA_NAMES, heldout_importance.tolist()))
        summary["attention_matrix_seen"]    = {
            "areas": AREA_NAMES, "matrix": attn_matrix.tolist(), "note": attn_note,
        }
        summary["attention_matrix_heldout"] = {
            "areas": AREA_NAMES, "matrix": heldout_attn_matrix.tolist(), "note": attn_note,
        }

    summary["channel_importance_gradient_saliency"] = {
        head: {
            "areas":      AREA_NAMES,
            "importance": saliency[head]["imp_area"].tolist(),
            "split":      attn_split_label,
        }
        for head in HEAD_NAMES
    }
    summary["filter_frequency_analysis"] = {
        "dominant_freqs_hz":     dominant_freqs.tolist(),
        "mean_spectrum_peak_hz": peak_freq,
        "note": freq_note,
    }
    _save_summary(summary)

    if not args.no_plot:
        plot_saliency_importance(
            saliency, args.phase,
            save_path=str(out_dir / f"saliency_{attn_split_label.replace(' ', '_')}.png"),
            split_label=attn_split_label,
        )
        plot_attention_analysis(
            importance=importance, attn_matrix=attn_matrix,
            area_names=AREA_NAMES, title_prefix=f"{args.phase} ({attn_split_label})",
            save_path=str(out_dir / f"attention_{attn_split_label.replace(' ', '_')}.png"),
        )
        if not args.no_heldout:
            plot_attention_analysis(
                importance=heldout_importance, attn_matrix=heldout_attn_matrix,
                area_names=AREA_NAMES, title_prefix=f"{args.phase} (heldout)",
                save_path=str(out_dir / "attention_heldout.png"),
            )

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
