"""Train and evaluate a phase-specialist LFP transformer.

Usage:
    python run_specialists.py --phase prereach
    python run_specialists.py --phase reach --n_permutations 100
    python run_specialists.py --phase grasp --out_dir results/my_grasp_run
    python run_specialists.py --phase prereach --dry_run   # 2 epochs, 3 perms
    python run_specialists.py --phase grasp --n_bins 10 --angles binary --heldout precision_right_135
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
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from data import (
    ANGLE_TO_ID,
    GRIP_TO_ID,
    HAND_TO_ID,
    ID_TO_ANGLE,
    ID_TO_GRIP,
    ID_TO_HAND,
    MAX_AREA_CHANNELS,
    PHASE_NAMES,
    load_dataset,
    make_compositional_split,
)
from specialist_data import (
    LFPSpecialistDataset,
    N_TIMEPOINTS,
    PermutedLabelDataset,
    compute_specialist_norm_stats,
)
from specialist_model import LFPSpecialistTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AREA_NAMES = ["PMvR", "M1", "PMdR", "PMdL"]
GRIP_NAMES = ["power", "precision"]
HAND_NAMES = ["left", "right"]
ANGLE_NAMES_ALL = ["0°", "45°", "90°", "135°"]
HEAD_NAMES = ["grip", "hand", "angle"]
HEAD_CLASS_NAMES: dict[str, list[str]] = {
    "grip": GRIP_NAMES,
    "hand": HAND_NAMES,
    "angle": ANGLE_NAMES_ALL[:],  # mutable copy; updated to ["0°","135°"] in binary mode
}


# ---------------------------------------------------------------------------
# Argument parsing helpers
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


def _validate_heldout(heldout: str) -> tuple[int, int, int]:
    """Parse and validate 'precision_right_135' → (grip_id, hand_id, angle_id).

    Format: {grip}_{hand}_{angle} where grip ∈ {power, precision},
    hand ∈ {left, right}, angle ∈ {0, 45, 90, 135}.

    If the data directory is mounted, also checks the class file exists.
    """
    parts = heldout.strip().split("_")
    if len(parts) != 3:
        raise ValueError(
            f"--heldout must be 'grip_hand_angle', e.g. 'precision_right_135'. Got: {heldout!r}"
        )
    grip_str, hand_str, angle_str = parts
    if grip_str not in GRIP_TO_ID:
        raise ValueError(
            f"--heldout: invalid grip {grip_str!r}. Must be one of {sorted(GRIP_TO_ID)}"
        )
    if hand_str not in HAND_TO_ID:
        raise ValueError(
            f"--heldout: invalid hand {hand_str!r}. Must be one of {sorted(HAND_TO_ID)}"
        )
    if angle_str not in ANGLE_TO_ID:
        raise ValueError(
            f"--heldout: invalid angle {angle_str!r}. Must be one of {sorted(ANGLE_TO_ID)}"
        )

    # Best-effort file-existence check (skipped if data drive not mounted)
    try:
        from data_paths import SEPARATED_CLASSES_DIR
        expected = (
            SEPARATED_CLASSES_DIR
            / f"{grip_str}_unimanual_{hand_str}_{angle_str}_degrees_mua_200_500.npy"
        )
        if SEPARATED_CLASSES_DIR.exists() and not expected.exists():
            raise ValueError(f"--heldout: class file not found: {expected}")
    except ImportError:
        pass

    return GRIP_TO_ID[grip_str], HAND_TO_ID[hand_str], ANGLE_TO_ID[angle_str]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a phase-specialist LFP transformer.")
    parser.add_argument("--phase", choices=PHASE_NAMES, required=True,
                        help="Which movement phase this specialist sees.")
    # Held-out combination — accept as a single string (preferred) or separate ints
    parser.add_argument("--heldout", type=str, default=None,
                        help="Held-out combination string, e.g. 'precision_right_135'. "
                             "Overrides --heldout_grip/hand/angle when provided. "
                             "Format: {grip}_{hand}_{angle}.")
    parser.add_argument("--heldout_grip", type=int, default=1,
                        help="Grip label (0=power, 1=precision) held out of training.")
    parser.add_argument("--heldout_hand", type=int, default=1,
                        help="Hand label (0=left, 1=right) held out of training.")
    parser.add_argument("--heldout_angle", type=int, default=3,
                        help="Angle label (0=0°,1=45°,2=90°,3=135°) held out of training.")
    # Temporal resolution
    parser.add_argument("--n_bins", type=_n_bins_type, default=1,
                        help="Temporal bins per area token. "
                             "1=single area-average (default), 5/10/20=multi-bin, "
                             "raw=500 individual timepoints (CPU-expensive).")
    # Angle subset
    parser.add_argument("--angles", choices=["all", "binary"], default="all",
                        help="all: use all 4 angles (default); "
                             "binary: keep only 0° and 135°, chance level = 0.5.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_permutations", type=int, default=1000,
                        help="Number of retrain permutation iterations per specialist. "
                             "Each iteration trains a fresh model on shuffled labels. "
                             "Use 100 for a fast overnight CPU run.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Defaults to results/specialist_{phase}/")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for caching the loaded dataset.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--feedforward_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--per_channel", action="store_true",
                        help="Use per-channel features (96 channels per area, no temporal binning) "
                             "instead of area-averaged temporal bins. --n_bins is ignored.")
    parser.add_argument("--no_heldout", action="store_true",
                        help="Standard stratified 80/10/10 split; no combination held out. "
                             "Skips permutation test. Output: results/specialist_{phase}_no_heldout/")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip all matplotlib figure generation.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 2 epochs (patience=1) and 3 permutations to verify the full "
                             "pipeline and output file structure without committing to a full run.")
    return parser.parse_args()


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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = total_correct = total_n = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y_grip, y_hand, y_angle in loader:
            x = x.to(device)
            y_grip = y_grip.to(device)
            y_hand = y_hand.to(device)
            y_angle = y_angle.to(device)

            lg, lh, la = model(x)
            loss = (
                criterion(lg, y_grip)
                + criterion(lh, y_hand)
                + criterion(la, y_angle)
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


def train_specialist(
    model: nn.Module,
    train_ds,
    val_ds,
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

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=1e-4
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(config["epochs"]):
        tl, ta = _run_epoch(model, train_loader, device, optimizer=optimizer)
        vl, va = _run_epoch(model, val_loader, device, optimizer=None)

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
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best val_loss={best_val_loss:.4f})"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path:
        torch.save(
            {"model_state": best_state, "config": config, "history": history},
            save_path,
        )
        if verbose:
            print(f"Saved checkpoint to {save_path}")

    return model, history, device


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_specialist(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, dict]:
    """Returns per-head dict with accuracy, confusion_matrix, y_true, y_pred."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds: dict[str, list] = {h: [] for h in HEAD_NAMES}
    all_labels: dict[str, list] = {h: [] for h in HEAD_NAMES}

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

    preds = {h: np.concatenate(v) for h, v in all_preds.items()}
    labels = {h: np.concatenate(v) for h, v in all_labels.items()}

    results: dict[str, dict] = {}
    for head in HEAD_NAMES:
        y_true, y_pred = labels[head], preds[head]
        n_classes = len(HEAD_CLASS_NAMES[head])
        results[head] = {
            "accuracy": float((y_true == y_pred).mean()),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    return results


# ---------------------------------------------------------------------------
# Permutation test — shuffle TRAINING labels and retrain from scratch
# ---------------------------------------------------------------------------

def run_permutation_test_retrain(
    train_ds: LFPSpecialistDataset,
    val_ds: LFPSpecialistDataset,
    heldout_test_ds: LFPSpecialistDataset,
    model_kwargs: dict,
    train_config: dict,
    n_permutations: int,
    seed: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 10,
) -> dict[str, list[float]]:
    """Permutation test via label-shuffle + retrain.

    For each iteration:
      1. Independently shuffle y_grip, y_hand, y_angle in both train_ds and
         val_ds (val labels are also shuffled so early stopping cannot
         accidentally exploit real signal).
      2. Train a fresh model instance on the shuffled labels.
      3. Evaluate on the REAL held-out test labels (never shuffled).
      4. Record per-head accuracy.

    The null distribution is centred around chance level.  p-value is the
    fraction of null accuracies >= the real model's accuracy.

    checkpoint_path: if given, the null distributions are written to this JSON
        file every checkpoint_every iterations so a crash does not lose progress.

    Returns null_distributions: {head: list of n_permutations accuracies}.
    """
    # Resume from checkpoint if one exists
    null: dict[str, list[float]] = {h: [] for h in HEAD_NAMES}
    start_i = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            null = saved["null_distributions"]
            start_i = saved["completed"]
            print(f"  Resuming permutation test from iteration {start_i} "
                  f"(checkpoint: {checkpoint_path})")
        except Exception:
            pass  # corrupt checkpoint — start fresh

    rng = np.random.default_rng(seed)
    # Advance RNG past already-completed iterations so we don't repeat them
    for _ in range(start_i * 6):  # 3 heads × 2 datasets = 6 permutation calls per iteration
        rng.integers(0, 1)

    t0 = time.time()
    for i in range(start_i, n_permutations):
        perm_train = PermutedLabelDataset(train_ds, rng)
        perm_val = PermutedLabelDataset(val_ds, rng)

        torch.manual_seed(seed + i + 1)
        model_perm = LFPSpecialistTransformer(**model_kwargs).to(device)
        model_perm, _, _ = train_specialist(
            model_perm, perm_train, perm_val,
            train_config, save_path=None, device=device, verbose=False,
        )

        perm_results = evaluate_specialist(
            model_perm, heldout_test_ds, device, train_config["batch_size"]
        )
        for h in HEAD_NAMES:
            null[h].append(perm_results[h]["accuracy"])

        del model_perm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        avg = elapsed / (i - start_i + 1)
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

    print(f"\n  {n_permutations}/{n_permutations} permutations complete "
          f"({time.time() - t0:.0f}s total).")
    return null


# ---------------------------------------------------------------------------
# Attention analysis
# ---------------------------------------------------------------------------

def collect_attention_weights(
    model: LFPSpecialistTransformer,
    dataset,
    device: torch.device,
    batch_size: int = 64,
) -> list[np.ndarray]:
    """Run inference and collect per-layer attention weights from every sample.

    Returns a list (one element per transformer layer) of arrays shaped
    (n_samples, n_heads, n_areas, n_areas).

    attn[s, h, i, j] = attention weight from area i (query) to area j (key)
    for sample s and head h.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    layer_batches: list[list[np.ndarray]] | None = None

    with torch.no_grad():
        for x, *_ in loader:
            x = x.to(device)
            model(x)  # forward pass populates last_attn_weights in each layer
            batch_weights = model.get_layer_attention_weights()

            if layer_batches is None:
                layer_batches = [[] for _ in batch_weights]
            for i, w in enumerate(batch_weights):
                layer_batches[i].append(w.cpu().numpy())

    if layer_batches is None:
        return []
    return [np.concatenate(batches, axis=0) for batches in layer_batches]
    # Each element: (n_samples, n_heads, n_areas, n_areas)


def compute_area_importance(
    layer_weights: list[np.ndarray],
    n_areas: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-area importance from the last transformer layer.

    The attention tensor is (n_samples, n_heads, n_tokens, n_tokens) where
    n_tokens = n_areas * n_bins.  Token order is area-first: token k represents
    (area k//n_bins, bin k%n_bins).  Time bins are aggregated out to recover
    per-area summaries compatible with existing plots.

    Returns:
        importance  — (n_areas,) normalized to sum to 1
        attn_matrix — (n_areas, n_areas) mean attention, area-aggregated
    """
    attn = layer_weights[-1]       # (n_samples, n_heads, n_tokens, n_tokens)
    attn_full = attn.mean(axis=(0, 1))  # (n_tokens, n_tokens)

    # Column mean = attention received per token; reshape to (n_areas, n_bins), avg over bins
    imp_per_token = attn_full.mean(axis=0)                          # (n_tokens,)
    importance = imp_per_token.reshape(n_areas, n_bins).mean(axis=1)  # (n_areas,)
    importance = importance / (importance.sum() + 1e-10)

    # Area-level attention matrix: average over (query_bin, key_bin) blocks
    # Reshape to (n_areas, n_bins, n_areas, n_bins), then mean over both bin dims
    attn_matrix = attn_full.reshape(n_areas, n_bins, n_areas, n_bins).mean(axis=(1, 3))
    return importance, attn_matrix


def compute_head_area_importance(
    model: LFPSpecialistTransformer,
    dataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """GradCAM-style per-head area importance.

    For each classification head (grip / hand / angle):
      1. Replay the forward pass with gradient tracking enabled.
      2. Retain the gradient on the last transformer layer's token activations.
      3. Backpropagate only that head's cross-entropy loss.
      4. Importance per area = mean_d_model( |grad × activation| ), averaged over samples.

    This gives three independent (n_areas,) importance vectors — one per head —
    reflecting which brain areas matter most for each classification task.

    Returns {"grip": (n_areas,), "hand": (n_areas,), "angle": (n_areas,)},
    each normalized to sum to 1.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    accum: dict[str, np.ndarray] = {h: np.zeros(len(AREA_NAMES), dtype=np.float64) for h in HEAD_NAMES}
    n_batches = 0

    head_logit_fns = {
        "grip": model.head_grip,
        "hand": model.head_hand,
        "angle": model.head_angle,
    }

    for x, y_grip, y_hand, y_angle in loader:
        x = x.to(device)
        targets = {
            "grip": y_grip.to(device),
            "hand": y_hand.to(device),
            "angle": y_angle.to(device),
        }

        for head in HEAD_NAMES:
            with torch.enable_grad():
                batch_sz = x.shape[0]
                if model.use_per_channel:
                    # x: (batch, n_areas, input_dim)
                    tok = model.input_proj(x.detach())                    # (batch, n_areas, d_model)
                    tok = tok + model.area_embedding(model.area_idx)[None, :, :]
                else:
                    x_flat = x.detach().reshape(batch_sz, model.n_tokens, 1)
                    tok = model.input_proj(x_flat)                        # (batch, n_tokens, d_model)
                    tok = tok + model.area_embedding(model.area_idx)[None, :, :]
                    tok = tok + model.time_embedding(model.bin_idx)[None, :, :]
                for layer in model.layers:
                    tok = layer(tok)
                tok.retain_grad()

                pooled = model.norm(tok.mean(dim=1))
                logits = head_logit_fns[head](pooled)
                loss = criterion(logits, targets[head])

                model.zero_grad()
                loss.backward()

            if tok.grad is not None:
                imp_tokens = (tok.grad * tok.detach()).abs().mean(dim=-1)   # (batch, n_tokens)
                if model.use_per_channel:
                    imp_areas = imp_tokens                                   # (batch, n_areas)
                else:
                    imp_areas = imp_tokens.reshape(
                        batch_sz, model.n_areas, model.n_bins
                    ).mean(dim=-1)                                           # (batch, n_areas)
                accum[head] += imp_areas.detach().cpu().numpy().mean(axis=0)

        n_batches += 1

    result: dict[str, np.ndarray] = {}
    for head in HEAD_NAMES:
        imp = accum[head] / max(n_batches, 1)
        result[head] = (imp / (imp.sum() + 1e-10)).astype(np.float32)

    return result


def compute_class_area_importance(
    layer_weights: list[np.ndarray],
    eval_results: dict[str, dict],
    n_areas: int,
    n_bins: int,
) -> dict[str, np.ndarray]:
    """Per-class, per-area attention importance for the held-out set.

    Groups each sample's attention profile by its TRUE label for each head,
    then averages within group.  This shows whether different conditions (e.g.
    angle = 135° vs 0°) route attention differently across brain areas.

    The attention tensor is (n_samples, n_heads, n_tokens, n_tokens); time bins
    are aggregated out before grouping by class.

    Returns dict with keys "grip", "hand", "angle".
    Value: (n_classes, n_areas) — normalized attention received per area, per true class.
    """
    attn = layer_weights[-1]  # (n_samples, n_heads, n_tokens, n_tokens)
    n_samples = attn.shape[0]

    attn_mean_heads = attn.mean(axis=1)                  # (n_samples, n_tokens, n_tokens)
    per_sample_token = attn_mean_heads.mean(axis=1)      # (n_samples, n_tokens) — col mean

    # Aggregate token-level importance to area-level by averaging over bins
    per_sample_imp = per_sample_token.reshape(
        n_samples, n_areas, n_bins
    ).mean(axis=-1)                                      # (n_samples, n_areas)

    row_sums = per_sample_imp.sum(axis=1, keepdims=True)
    per_sample_imp = per_sample_imp / (row_sums + 1e-10)

    # Use HEAD_CLASS_NAMES so binary-mode angle (2 classes) is handled correctly
    n_classes_per_head = {h: len(HEAD_CLASS_NAMES[h]) for h in HEAD_NAMES}
    results: dict[str, np.ndarray] = {}
    for head in HEAD_NAMES:
        y_true = eval_results[head]["y_true"]
        n_c = n_classes_per_head[head]
        class_imp = np.zeros((n_c, n_areas), dtype=np.float32)
        for c in range(n_c):
            mask = y_true == c
            if mask.sum() > 0:
                group = per_sample_imp[mask].mean(axis=0)
                class_imp[c] = group / (group.sum() + 1e-10)
        results[head] = class_imp  # (n_classes, n_areas)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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
        (axes[1], "acc", "Accuracy (avg across heads)"),
    ]:
        ax.plot(epochs, history[f"train_{key}"], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="val")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        if key == "acc":
            ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: list[str],
    title: str,
    save_path: str,
) -> None:
    plt = _get_plt()
    if plt is None:
        return
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i, j]}\n{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.6 else "black",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_permutation_test(
    null_accs: list[float],
    real_acc: float,
    p_val: float,
    head: str,
    phase_name: str,
    save_path: str,
) -> None:
    plt = _get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_accs, bins=30, color="#4C72B0", alpha=0.75, label="Null (retrained on shuffled labels)")
    ax.axvline(
        real_acc, color="#C44E52", linewidth=2.5,
        label=f"Real accuracy = {real_acc:.3f}",
    )
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")
    ax.set_title(f"{phase_name} — {head}  (p = {p_val:.4f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_attention_analysis(
    importance: np.ndarray,
    attn_matrix: np.ndarray,
    area_names: list[str],
    title_prefix: str,
    save_path: str,
) -> None:
    """Bar chart of area importance + attention heatmap side-by-side."""
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
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
        )

    ax = axes[1]
    vmax = max(attn_matrix.max(), 1e-6)
    im = ax.imshow(attn_matrix, cmap="Blues", vmin=0.0, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(area_names)))
    ax.set_yticks(range(len(area_names)))
    ax.set_xticklabels(area_names)
    ax.set_yticklabels(area_names)
    ax.set_xlabel("Key area (attends to →)")
    ax.set_ylabel("← Query area (attends from)")
    ax.set_title(f"{title_prefix} — Attention matrix (last layer)")
    for i in range(len(area_names)):
        for j in range(len(area_names)):
            ax.text(
                j, i, f"{attn_matrix[i, j]:.2f}",
                ha="center", va="center",
                color="white" if attn_matrix[i, j] > vmax * 0.65 else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_class_area_attention(
    class_importance: dict[str, np.ndarray],
    area_names: list[str],
    phase_name: str,
    save_path: str,
    split_label: str = "seen test",
) -> None:
    """3-panel heatmap: one per head, rows = true classes, columns = brain areas."""
    plt = _get_plt()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, head in zip(axes, HEAD_NAMES):
        mat = class_importance[head]   # (n_classes, n_areas)
        cnames = HEAD_CLASS_NAMES[head]
        vmax = max(mat.max(), 1e-6)

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(area_names)))
        ax.set_yticks(range(len(cnames)))
        ax.set_xticklabels(area_names)
        ax.set_yticklabels(cnames)
        ax.set_xlabel("Brain area")
        ax.set_ylabel("True class")
        ax.set_title(f"{phase_name} — {head}\nattention by true class ({split_label})")
        for i in range(len(cnames)):
            for j in range(len(area_names)):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if mat[i, j] > vmax * 0.7 else "black",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_head_area_importance(
    head_importance: dict[str, np.ndarray],
    area_names: list[str],
    phase_name: str,
    save_path: str,
    split_label: str = "seen test",
) -> None:
    """3-panel bar chart — one panel per classification head (GradCAM importance)."""
    plt = _get_plt()
    if plt is None:
        return

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ax, head in zip(axes, HEAD_NAMES):
        imp = head_importance[head]
        bars = ax.bar(area_names, imp, color=colors)
        ax.set_xlabel("Brain area")
        ax.set_ylabel("Normalized GradCAM importance")
        ax.set_title(f"{phase_name} — {head}\n(GradCAM, {split_label})")
        ax.set_ylim(0, min(1.0, imp.max() * 1.45 + 0.05))
        for bar, v in zip(bars, imp):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Terminal reporting helpers
# ---------------------------------------------------------------------------

def _print_class_attention_table(
    class_importance: dict[str, np.ndarray],
    split_label: str = "seen test",
) -> None:
    W = 7
    print(f"\n  Class-conditional area attention ({split_label}, last attn layer):")
    for head in HEAD_NAMES:
        mat = class_importance[head]
        cnames = HEAD_CLASS_NAMES[head]
        header = f"    {'':12s}" + "".join(f"{a:>{W}}" for a in AREA_NAMES)
        print(f"\n  {head.upper()}:")
        print(header)
        for c, cname in enumerate(cnames):
            row = f"    {cname:<12s}" + "".join(f"{mat[c, j]:>{W}.3f}" for j in range(len(AREA_NAMES)))
            print(row)


def _print_head_area_importance_table(
    head_importance: dict[str, np.ndarray],
    split_label: str = "seen test",
) -> None:
    W = 8
    print(f"\n  Head-specific area importance (GradCAM, |grad × activation|, {split_label}):")
    header = f"    {'Head':8s}" + "".join(f"{a:>{W}}" for a in AREA_NAMES)
    print(header)
    print("    " + "-" * (8 + W * len(AREA_NAMES)))
    for head in HEAD_NAMES:
        imp = head_importance[head]
        row = f"    {head.upper():<8s}" + "".join(f"{v:>{W}.3f}" for v in imp)
        print(row)


# ---------------------------------------------------------------------------
# Standard split (--no_heldout mode)
# ---------------------------------------------------------------------------

def _subset_data(data: dict, idx: np.ndarray) -> dict:
    return {
        "file_paths": data["file_paths"],
        "n_channels": data["n_channels"],
        "file_idx": data["file_idx"][idx],
        "trial_idx": data["trial_idx"][idx],
        "y_grip": data["y_grip"][idx],
        "y_hand": data["y_hand"][idx],
        "y_angle": data["y_angle"][idx],
        "is_heldout": data["is_heldout"][idx],
    }


def make_standard_split(
    data: dict,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """Stratified 80/10/10 split with all 16 combinations present in every split."""
    from sklearn.model_selection import StratifiedShuffleSplit

    n = len(data["y_grip"])
    combo_idx = data["y_grip"] * 8 + data["y_hand"] * 4 + data["y_angle"]
    all_idx = np.arange(n)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, temp_idx = next(sss.split(all_idx, combo_idx))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(sss2.split(temp_idx, combo_idx[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    print(
        "Split sizes | "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )
    return _subset_data(data, train_idx), _subset_data(data, val_idx), _subset_data(data, test_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # -- Resolve --heldout string (overrides separate integer flags)
    if args.heldout is not None:
        if args.no_heldout:
            print("WARNING: --heldout ignored when --no_heldout is set")
        else:
            try:
                grip_id, hand_id, angle_id = _validate_heldout(args.heldout)
            except ValueError as exc:
                sys.exit(f"Error: {exc}")
            args.heldout_grip = grip_id
            args.heldout_hand = hand_id
            args.heldout_angle = angle_id

    # -- Validate binary mode + held-out angle compatibility
    if not args.no_heldout and args.angles == "binary" and args.heldout_angle not in {0, 3}:
        sys.exit(
            f"Error: --angles binary requires held-out angle to be 0° or 135°, "
            f"got {ID_TO_ANGLE[args.heldout_angle]}°"
        )

    # -- Warn about expensive raw mode
    if args.n_bins == "raw":
        print(
            "WARNING: n_bins='raw' uses 500 temporal bins. "
            "Training will be significantly slower on CPU."
        )

    if args.per_channel and args.n_bins != 1:
        print(f"WARNING: --n_bins ignored when --per_channel is set (got n_bins={args.n_bins}).")

    if args.dry_run:
        args.epochs = 2
        args.patience = 1
        args.n_permutations = 3
        heldout_str = (
            args.heldout if args.heldout is not None
            else f"{ID_TO_GRIP[args.heldout_grip]}_{ID_TO_HAND[args.heldout_hand]}_{ID_TO_ANGLE[args.heldout_angle]}"
        )
        input_desc = f"per_channel({MAX_AREA_CHANNELS})" if args.per_channel else f"n_bins={args.n_bins}"
        print(
            f"[dry-run] phase={args.phase.upper()} input={input_desc} "
            f"angles={args.angles} heldout={heldout_str}"
        )

    phase_idx = PHASE_NAMES.index(args.phase)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.no_heldout:
        out_dir = _HERE / "results" / f"specialist_{args.phase}_no_heldout"
    elif args.per_channel:
        out_dir = _HERE / "results" / f"specialist_{args.phase}_per_channel"
    else:
        out_dir = _HERE / "results" / f"specialist_{args.phase}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    heldout_label = (
        f"{ID_TO_GRIP[args.heldout_grip]}_"
        f"{ID_TO_HAND[args.heldout_hand]}_"
        f"{ID_TO_ANGLE[args.heldout_angle]}"
    )

    print(f"\n{'='*60}")
    print(f"  Phase specialist: {args.phase.upper()}")
    print(f"  n_bins: {args.n_bins}  |  angles: {args.angles}")
    if args.no_heldout:
        print(f"  Mode: standard 80/10/10 split (no held-out combination)")
    else:
        print(f"  Held-out combination: {heldout_label}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data = load_dataset(
        cache_dir=args.cache_dir,
        heldout_grip=args.heldout_grip,
        heldout_hand=args.heldout_hand,
        heldout_angle=args.heldout_angle,
    )

    # -- Binary angle filtering: keep only 0° (id=0) and 135° (id=3)
    if args.angles == "binary":
        binary_mask = (data["y_angle"] == 0) | (data["y_angle"] == 3)
        data = {
            "file_paths": data["file_paths"],
            "n_channels": data["n_channels"],
            **{k: data[k][binary_mask]
               for k in ("file_idx", "trial_idx", "y_grip", "y_hand", "y_angle", "is_heldout")},
        }
        # Remap angle labels: 0→0, 135(3)→1
        data["y_angle"] = np.where(
            data["y_angle"] == 3, np.int64(1), data["y_angle"]
        )
        n_angle_classes = 2
        HEAD_CLASS_NAMES["angle"] = ["0°", "135°"]
        print(f"Binary mode: kept {binary_mask.sum()} trials (0° and 135° only), "
              f"remapped angle labels 0→0, 135→1")
    else:
        n_angle_classes = 4
        HEAD_CLASS_NAMES["angle"] = ANGLE_NAMES_ALL[:]

    print(f"Dataset: {len(data['y_grip'])} trials, {int(data['n_channels'])} channels")

    if args.no_heldout:
        train_data, val_data, test_data = make_standard_split(data, seed=args.seed)
        seen_test_data = heldout_test_data = None
    else:
        train_data, val_data, seen_test_data, heldout_test_data, _ = make_compositional_split(
            data, seed=args.seed,
        )

    # -- n_bins: compute actual_bins for model input_dim
    actual_bins: int = N_TIMEPOINTS if args.n_bins == "raw" else int(args.n_bins)

    if args.per_channel:
        print(f"\nComputing {args.phase} normalization stats from training set "
              f"(per_channel mode, {MAX_AREA_CHANNELS} channels per area)...")
    else:
        print(f"\nComputing {args.phase} normalization stats from training set "
              f"(n_bins={args.n_bins}, actual_bins={actual_bins})...")
    norm_stats = compute_specialist_norm_stats(
        train_data, phase_idx, args.n_bins, use_per_channel=args.per_channel
    )

    train_ds = LFPSpecialistDataset(train_data, phase_idx, norm_stats, args.n_bins, args.per_channel)
    val_ds = LFPSpecialistDataset(val_data, phase_idx, norm_stats, args.n_bins, args.per_channel)
    if args.no_heldout:
        test_ds = LFPSpecialistDataset(test_data, phase_idx, norm_stats, args.n_bins, args.per_channel)
        seen_test_ds = heldout_test_ds = None
        print(f"Split: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    else:
        test_ds = None
        seen_test_ds = LFPSpecialistDataset(seen_test_data, phase_idx, norm_stats, args.n_bins, args.per_channel)
        heldout_test_ds = LFPSpecialistDataset(heldout_test_data, phase_idx, norm_stats, args.n_bins, args.per_channel)
        print(
            f"Split: train={len(train_ds)}  val={len(val_ds)}  "
            f"seen_test={len(seen_test_ds)}  heldout_test={len(heldout_test_ds)}"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_kwargs = dict(
        n_bins=actual_bins,
        n_angle_classes=n_angle_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        use_per_channel=args.per_channel,
        input_dim=MAX_AREA_CHANNELS if args.per_channel else 1,
    )
    model = LFPSpecialistTransformer(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input_mode = "per_channel" if args.per_channel else f"per_channel=False, n_bins={actual_bins}"
    print(f"Trainable parameters: {n_params:,}  (input={input_mode}, n_tokens={model.n_tokens}, n_angle_classes={n_angle_classes})")

    train_config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
    }

    summary_path = out_dir / "summary.json"

    def _save_summary(s: dict) -> None:
        summary_path.write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"Saved {summary_path}")

    # ------------------------------------------------------------------
    # Train real model
    # ------------------------------------------------------------------
    print()
    model, history, device = train_specialist(
        model, train_ds, val_ds, train_config,
        save_path=str(out_dir / "checkpoint.pt"),
        verbose=True,
    )

    # Save normalization stats and training curve immediately after training
    np.savez_compressed(out_dir / "normalization_stats.npz", **norm_stats)
    if not args.no_plot:
        plot_training_history(history, str(out_dir / "training_curves.png"))

    # ------------------------------------------------------------------
    # Evaluate real model
    # ------------------------------------------------------------------
    if args.no_heldout:
        print("\nEvaluating on test set (standard split)...")
        test_results = evaluate_specialist(model, test_ds, device, args.batch_size)
        seen_results = heldout_results = None
        print(f"\n  Test (standard 80-20 split):")
        for h in HEAD_NAMES:
            print(f"    {h:6s}: {test_results[h]['accuracy']:.4f}")
        eval_pairs = [("test", test_results)]
    else:
        print("\nEvaluating on seen combinations...")
        seen_results = evaluate_specialist(model, seen_test_ds, device, args.batch_size)
        print("Evaluating on held-out combination...")
        heldout_results = evaluate_specialist(model, heldout_test_ds, device, args.batch_size)
        test_results = None
        for split_name, results in [("Seen", seen_results), ("Held-out", heldout_results)]:
            print(f"\n  {split_name}:")
            for h in HEAD_NAMES:
                print(f"    {h:6s}: {results[h]['accuracy']:.4f}")
        real_acc = {h: heldout_results[h]["accuracy"] for h in HEAD_NAMES}
        eval_pairs = [("seen", seen_results), ("heldout", heldout_results)]

    # Save confusion matrices immediately after evaluation
    for split_name, results in eval_pairs:
        for h in HEAD_NAMES:
            np.save(
                out_dir / f"{split_name}_{h}_confusion_matrix.npy",
                results[h]["confusion_matrix"],
            )
    if not args.no_plot:
        for split_name, results in eval_pairs:
            for h in HEAD_NAMES:
                plot_confusion_matrix(
                    results[h]["confusion_matrix"],
                    target_names=HEAD_CLASS_NAMES[h],
                    title=f"{args.phase} — {h} ({split_name})",
                    save_path=str(out_dir / f"{split_name}_{h}_confusion_matrix.png"),
                )

    # Write a partial summary so results are not lost if the permutation test crashes
    _model_meta = {
        "n_bins": actual_bins,
        "n_tokens": model.n_tokens,
        "n_angle_classes": n_angle_classes,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
        "n_params": n_params,
    }
    if args.no_heldout:
        summary: dict = {
            "phase": args.phase,
            "phase_idx": phase_idx,
            "mode": "no_heldout",
            "input_mode": input_mode,
            "n_bins": args.n_bins,
            "angles": args.angles,
            "n_angle_classes": n_angle_classes,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "test_accuracy": {h: test_results[h]["accuracy"] for h in HEAD_NAMES},
            "area_importance": None,
            "attention_matrix": None,
            "class_area_attention": None,
            "head_area_importance_gradcam": None,
            "model": _model_meta,
            "config": train_config,
            "seed": args.seed,
        }
    else:
        summary = {
            "phase": args.phase,
            "phase_idx": phase_idx,
            "heldout_label": heldout_label,
            "input_mode": input_mode,
            "n_bins": args.n_bins,
            "angles": args.angles,
            "n_angle_classes": n_angle_classes,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "seen_test_size": len(seen_test_ds),
            "heldout_test_size": len(heldout_test_ds),
            "seen_accuracy": {h: seen_results[h]["accuracy"] for h in HEAD_NAMES},
            "heldout_accuracy": {h: heldout_results[h]["accuracy"] for h in HEAD_NAMES},
            "p_values": None,
            "area_importance_seen": None,
            "area_importance_heldout": None,
            "attention_matrix_seen": None,
            "attention_matrix_heldout": None,
            "class_area_attention_seen": None,
            "head_area_importance_gradcam": None,
            "permutation_test": None,
            "model": _model_meta,
            "config": train_config,
            "seed": args.seed,
            "n_permutations": args.n_permutations,
        }
    print()
    _save_summary(summary)

    # ------------------------------------------------------------------
    # Permutation test — skipped in no_heldout mode
    # ------------------------------------------------------------------
    if not args.no_heldout:
        perm_checkpoint = out_dir / "null_distributions_checkpoint.json"
        print(f"\nRunning {args.n_permutations} permutation iterations "
              f"(retrain on shuffled labels each time)...")
        null_distributions = run_permutation_test_retrain(
            train_ds=train_ds,
            val_ds=val_ds,
            heldout_test_ds=heldout_test_ds,
            model_kwargs=model_kwargs,
            train_config=train_config,
            n_permutations=args.n_permutations,
            seed=args.seed + 1,
            device=device,
            checkpoint_path=perm_checkpoint,
            checkpoint_every=10,
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

        if not args.no_plot:
            for h in HEAD_NAMES:
                plot_permutation_test(
                    null_accs=null_distributions[h],
                    real_acc=real_acc[h],
                    p_val=p_values[h],
                    head=h,
                    phase_name=args.phase,
                    save_path=str(out_dir / f"permutation_{h}.png"),
                )

    # ------------------------------------------------------------------
    # Attention analysis
    # ------------------------------------------------------------------
    if args.no_heldout:
        print("\nExtracting attention weights (test set)...")
        test_layer_attn = collect_attention_weights(model, test_ds, device, args.batch_size)
        test_importance, test_attn_matrix = compute_area_importance(test_layer_attn, model.n_areas, model.n_bins)

        print(f"\n  Area importance — {args.phase} (test, normalized attention received):")
        for name, imp in zip(AREA_NAMES, test_importance):
            bar = "█" * int(imp * 40)
            print(f"    {name:6s}: {imp:.4f}  {bar}")

        class_area_attention = compute_class_area_importance(test_layer_attn, test_results, model.n_areas, model.n_bins)
        _print_class_attention_table(class_area_attention, split_label="test")

        print("\nComputing head-specific area importance (GradCAM, test)...")
        head_importance = compute_head_area_importance(model, test_ds, device, args.batch_size)
        _print_head_area_importance_table(head_importance, split_label="test")

        summary["area_importance"] = dict(zip(AREA_NAMES, test_importance.tolist()))
        summary["attention_matrix"] = {
            "areas": AREA_NAMES,
            "matrix": test_attn_matrix.tolist(),
            "note": "attn_matrix[i][j] = attention from area i (query) to area j (key), "
                    "averaged over test samples and heads (last transformer layer)",
        }
        summary["class_area_attention"] = {
            head: {
                "classes": HEAD_CLASS_NAMES[head],
                "areas": AREA_NAMES,
                "matrix": class_area_attention[head].tolist(),
            }
            for head in HEAD_NAMES
        }
        summary["head_area_importance_gradcam"] = {
            head: {
                "areas": AREA_NAMES,
                "importance": head_importance[head].tolist(),
            }
            for head in HEAD_NAMES
        }
        _save_summary(summary)

        if not args.no_plot:
            plot_attention_analysis(
                importance=test_importance,
                attn_matrix=test_attn_matrix,
                area_names=AREA_NAMES,
                title_prefix=f"{args.phase} (test)",
                save_path=str(out_dir / "attention_test.png"),
            )
            plot_class_area_attention(
                class_importance=class_area_attention,
                area_names=AREA_NAMES,
                phase_name=args.phase,
                save_path=str(out_dir / "attention_by_class_test.png"),
                split_label="test",
            )
            plot_head_area_importance(
                head_importance=head_importance,
                area_names=AREA_NAMES,
                phase_name=args.phase,
                save_path=str(out_dir / "head_area_importance_gradcam.png"),
                split_label="test",
            )

    else:
        print("\nExtracting attention weights (seen test)...")
        seen_layer_attn = collect_attention_weights(model, seen_test_ds, device, args.batch_size)
        seen_importance, seen_attn_matrix = compute_area_importance(seen_layer_attn, model.n_areas, model.n_bins)

        print("Extracting attention weights (held-out test)...")
        heldout_layer_attn = collect_attention_weights(model, heldout_test_ds, device, args.batch_size)
        heldout_importance, heldout_attn_matrix = compute_area_importance(heldout_layer_attn, model.n_areas, model.n_bins)

        print(f"\n  Area importance — {args.phase} (held-out, normalized attention received):")
        for name, imp in zip(AREA_NAMES, heldout_importance):
            bar = "█" * int(imp * 40)
            print(f"    {name:6s}: {imp:.4f}  {bar}")

        class_area_attention = compute_class_area_importance(seen_layer_attn, seen_results, model.n_areas, model.n_bins)
        _print_class_attention_table(class_area_attention)

        print("\nComputing head-specific area importance (GradCAM, seen test)...")
        head_importance = compute_head_area_importance(model, seen_test_ds, device, args.batch_size)
        _print_head_area_importance_table(head_importance)

        summary["area_importance_seen"] = dict(zip(AREA_NAMES, seen_importance.tolist()))
        summary["area_importance_heldout"] = dict(zip(AREA_NAMES, heldout_importance.tolist()))
        summary["attention_matrix_seen"] = {
            "areas": AREA_NAMES,
            "matrix": seen_attn_matrix.tolist(),
            "note": "attn_matrix[i][j] = attention from area i (query) to area j (key), "
                    "averaged over test samples and heads (last transformer layer)",
        }
        summary["attention_matrix_heldout"] = {
            "areas": AREA_NAMES,
            "matrix": heldout_attn_matrix.tolist(),
            "note": "attn_matrix[i][j] = attention from area i (query) to area j (key), "
                    "averaged over test samples and heads (last transformer layer)",
        }
        summary["class_area_attention_seen"] = {
            head: {
                "classes": HEAD_CLASS_NAMES[head],
                "areas": AREA_NAMES,
                "matrix": class_area_attention[head].tolist(),
                "note": "matrix[c][j] = normalized mean attention received by area j "
                        "when true class is c. Computed on seen_test (all class values "
                        "present); held-out test contains only one combination so "
                        "per-class breakdown is not meaningful there.",
            }
            for head in HEAD_NAMES
        }
        summary["head_area_importance_gradcam"] = {
            head: {
                "areas": AREA_NAMES,
                "importance": head_importance[head].tolist(),
                "note": "GradCAM importance: mean_d_model(|grad × activation|) for last "
                        "transformer layer tokens, backpropagating each head's loss "
                        "independently. Computed on seen_test. Normalized to sum to 1.",
            }
            for head in HEAD_NAMES
        }
        _save_summary(summary)

        if not args.no_plot:
            for split_name, importance, attn_matrix in [
                ("seen", seen_importance, seen_attn_matrix),
                ("heldout", heldout_importance, heldout_attn_matrix),
            ]:
                plot_attention_analysis(
                    importance=importance,
                    attn_matrix=attn_matrix,
                    area_names=AREA_NAMES,
                    title_prefix=f"{args.phase} ({split_name})",
                    save_path=str(out_dir / f"attention_{split_name}.png"),
                )

            plot_class_area_attention(
                class_importance=class_area_attention,
                area_names=AREA_NAMES,
                phase_name=args.phase,
                save_path=str(out_dir / "attention_by_class_seen.png"),
            )

            plot_head_area_importance(
                head_importance=head_importance,
                area_names=AREA_NAMES,
                phase_name=args.phase,
                save_path=str(out_dir / "head_area_importance_gradcam.png"),
            )

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
