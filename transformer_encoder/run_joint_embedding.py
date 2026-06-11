"""Train a joint phase/grip/hand channel-token transformer.

This is the executable entry point for the transformer stage.

It loads separated class files, expands each trial into phase-level samples,
extracts MU or six-band channel-token features, trains one shared encoder with
three heads (phase, grip, hand), and writes both evaluation outputs and pooled
embeddings for the downstream embedding cVAE.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformer_encoder.joint_embedding_data import (
    AREA_SIZES,
    BAND_NAMES_6,
    CHANNEL_VALID,
    INPUT_MODES,
    JointEmbeddingDataset,
    MAX_AREA_CHANNELS,
    N_AREAS,
    N_TOKENS,
    PADDING_MASK,
    PermutedJointDataset,
    compute_norm_stats,
    extract_and_cache_features,
    load_joint_trials,
    phase_expand,
    subset_flat,
)
from transformer_encoder.joint_embedding_model import JointFactorTransformer

from transformer_encoder.joint_embedding_data import AREA_NAMES, GRIP_TO_ID, HAND_TO_ID, PHASE_NAMES


HEADS = ("phase", "grip", "hand")
CLASS_NAMES = {
    "phase": PHASE_NAMES,
    "grip": ["power", "precision"],
    "hand": ["left", "right"],
}


def expected_heldout_run_name(args: argparse.Namespace) -> str:
    return f"transformer_heldout_{args.heldout_phase}_{args.heldout_grip}_{args.heldout_hand}"


def validate_output_dir(args: argparse.Namespace, out_dir: Path) -> None:
    """Prevent saving a held-out checkpoint into another combo's folder."""
    if not args.heldout:
        return

    expected = expected_heldout_run_name(args)
    actual = out_dir.name
    if actual != expected and not actual.startswith(f"{expected}_"):
        raise SystemExit(
            "Refusing to write held-out checkpoint to mismatched --out_dir.\n"
            f"  heldout combo: {args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}\n"
            f"  expected folder name: {expected} or {expected}_...\n"
            f"  got folder name: {actual}\n"
            "Use a matching --out_dir so checkpoint metadata and result files cannot diverge."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--input_mode", choices=INPUT_MODES, default="mu")
    p.add_argument("--cache_dir", type=str, default="/tmp/lfp_joint_embedding_cache")
    p.add_argument("--out_dir", type=str, default=str(_HERE / "results" / "joint_mu"))
    split = p.add_mutually_exclusive_group()
    split.add_argument(
        "--heldout",
        action="store_true",
        help="Diagnostic only: hold out one phase/grip/hand combo from joint-transformer training.",
    )
    split.add_argument(
        "--no_heldout",
        action="store_true",
        help="Normal stratified train/val/test split. This is the Step 1b default.",
    )
    p.add_argument("--heldout_phase", choices=PHASE_NAMES, default="grasp")
    p.add_argument("--heldout_grip", choices=["power", "precision"], default="precision")
    p.add_argument("--heldout_hand", choices=["left", "right"], default="right")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--feedforward_dim", type=int, default=128)
    p.add_argument("--label_smoothing", type=float, default=0.15)
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--loss_weight_phase", type=float, default=1.0)
    p.add_argument("--loss_weight_grip",  type=float, default=1.0)
    p.add_argument("--loss_weight_hand",  type=float, default=1.0)
    p.add_argument("--n_permutations", type=int, default=20)
    p.add_argument(
        "--permutation_epochs",
        type=int,
        default=8,
        help="Max epochs for each permutation retrain. Use --full_permutation_training to match main training.",
    )
    p.add_argument(
        "--permutation_patience",
        type=int,
        default=2,
        help="Early-stopping patience for each permutation retrain.",
    )
    p.add_argument(
        "--full_permutation_training",
        action="store_true",
        help="Use the main --epochs/--patience for each permutation retrain.",
    )
    p.add_argument(
        "--skip_permutation",
        action="store_true",
        help="Skip retrain permutation baselines.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def _heldout_split_indices(
    flat: dict,
    heldout_phase: int,
    heldout_grip: int,
    heldout_hand: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(flat["y_phase"]))
    heldout_mask = (
        (flat["y_phase"] == heldout_phase)
        & (flat["y_grip"] == heldout_grip)
        & (flat["y_hand"] == heldout_hand)
    )
    heldout_idx = all_idx[heldout_mask]
    remaining_idx = all_idx[~heldout_mask]
    if len(heldout_idx) < 2:
        raise ValueError("Held-out phase/grip/hand combination has fewer than two samples.")

    strat_remaining = (
        flat["y_phase"][remaining_idx].astype(np.int64) * 16
        + flat["y_grip"][remaining_idx].astype(np.int64) * 8
        + flat["y_hand"][remaining_idx].astype(np.int64) * 4
        + flat["y_angle"][remaining_idx].astype(np.int64)
    )
    train_idx, temp_idx = _train_test_split_maybe_stratified(
        remaining_idx,
        test_size=0.2,
        random_state=seed,
        stratify=strat_remaining,
    )
    strat_temp = (
        flat["y_phase"][temp_idx].astype(np.int64) * 16
        + flat["y_grip"][temp_idx].astype(np.int64) * 8
        + flat["y_hand"][temp_idx].astype(np.int64) * 4
        + flat["y_angle"][temp_idx].astype(np.int64)
    )
    seen_val_idx, seen_test_idx = _train_test_split_maybe_stratified(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=strat_temp,
    )
    # Strict zero-shot protocol: held-out samples must not influence training,
    # validation, early stopping, or hyperparameter/model selection.
    # The entire held-out combination is reserved for final evaluation only.
    val_idx = seen_val_idx
    held_test_idx = heldout_idx
    return np.sort(train_idx), np.sort(val_idx), np.sort(seen_test_idx), np.sort(held_test_idx)


def _train_test_split_maybe_stratified(
    idx: np.ndarray,
    test_size: float,
    random_state: int,
    stratify: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if stratify is not None:
        _, counts = np.unique(stratify, return_counts=True)
        if len(counts) == 0 or counts.min() < 2:
            stratify = None
    return train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )


def _normal_split_indices(
    flat_data: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(flat_data["y_grip"]))
    strat = (
        flat_data["y_phase"].astype(np.int64) * 32
        + flat_data["y_grip"].astype(np.int64) * 16
        + flat_data["y_hand"].astype(np.int64) * 8
        + flat_data["y_angle"].astype(np.int64)
    )
    train_idx, temp_idx = _train_test_split_maybe_stratified(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=strat,
    )
    strat_temp = (
        flat_data["y_phase"][temp_idx].astype(np.int64) * 32
        + flat_data["y_grip"][temp_idx].astype(np.int64) * 16
        + flat_data["y_hand"][temp_idx].astype(np.int64) * 8
        + flat_data["y_angle"][temp_idx].astype(np.int64)
    )
    val_idx, test_idx = _train_test_split_maybe_stratified(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=strat_temp,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float,
    optimizer: torch.optim.Optimizer | None = None,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0.0
    total_n = 0
    w_phase, w_grip, w_hand = loss_weights
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y_phase, y_grip, y_hand, _ in loader:
            x = x.to(device)
            y_phase = y_phase.to(device)
            y_grip = y_grip.to(device)
            y_hand = y_hand.to(device)
            lp, lg, lh = model(x)
            loss = (
                w_phase * nn.functional.cross_entropy(lp, y_phase, label_smoothing=label_smoothing)
                + w_grip  * nn.functional.cross_entropy(lg, y_grip,  label_smoothing=label_smoothing)
                + w_hand  * nn.functional.cross_entropy(lh, y_hand,  label_smoothing=label_smoothing)
            )
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n = x.shape[0]
            total_loss += float(loss.item()) * n
            total_correct += (
                (lp.argmax(1) == y_phase).float()
                + (lg.argmax(1) == y_grip).float()
                + (lh.argmax(1) == y_hand).float()
            ).sum().item() / 3.0
            total_n += n
    return total_loss / total_n, total_correct / total_n


def train_model(
    model: JointFactorTransformer,
    train_ds: JointEmbeddingDataset,
    val_ds: JointEmbeddingDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[JointFactorTransformer, dict]:
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = getattr(args, "warmup_epochs", 5)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: 1.0 if warmup_epochs <= 0 else min(1.0, (ep + 1) / warmup_epochs))
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(args.epochs):
        lw = (args.loss_weight_phase, args.loss_weight_grip, args.loss_weight_hand)
        tl, ta = _run_epoch(model, train_loader, device, args.label_smoothing, optimizer, loss_weights=lw)
        vl, va = _run_epoch(model, val_loader, device, args.label_smoothing, loss_weights=lw)
        scheduler.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"train_loss={tl:.4f} train_acc={ta:.4f} | "
            f"val_loss={vl:.4f} val_acc={va:.4f}"
        )
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch + 1} (best val_loss={best_val:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history


def _train_model_quiet(
    model: JointFactorTransformer,
    train_ds,
    val_ds,
    args: argparse.Namespace,
    device: torch.device,
) -> JointFactorTransformer:
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = getattr(args, "warmup_epochs", 5)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: 1.0 if warmup_epochs <= 0 else min(1.0, (ep + 1) / warmup_epochs))
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience_ctr = 0
    for _ in range(args.epochs):
        _run_epoch(model, train_loader, device, args.label_smoothing, optimizer)
        vl, _ = _run_epoch(model, val_loader, device, args.label_smoothing, loss_weights=(1.0, 1.0, 1.0))
        scheduler.step()
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                break
    model.load_state_dict(best_state)
    return model


def evaluate_model(
    model: JointFactorTransformer,
    dataset: JointEmbeddingDataset,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = {h: [] for h in HEADS}
    labels = {h: [] for h in HEADS}
    with torch.no_grad():
        for x, y_phase, y_grip, y_hand, _ in loader:
            lp, lg, lh = model(x.to(device))
            preds["phase"].append(lp.argmax(1).cpu().numpy())
            preds["grip"].append(lg.argmax(1).cpu().numpy())
            preds["hand"].append(lh.argmax(1).cpu().numpy())
            labels["phase"].append(y_phase.numpy())
            labels["grip"].append(y_grip.numpy())
            labels["hand"].append(y_hand.numpy())

    out = {}
    for head in HEADS:
        y_true = np.concatenate(labels[head])
        y_pred = np.concatenate(preds[head])
        out[head] = {
            "accuracy": float((y_true == y_pred).mean()),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
                labels=np.arange(len(CLASS_NAMES[head])),
            ).tolist(),
        }
    return out


def run_permutation_test(
    train_ds: JointEmbeddingDataset,
    val_ds: JointEmbeddingDataset,
    test_ds: JointEmbeddingDataset,
    model_kwargs: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[float]]:
    null = {head: [] for head in HEADS}
    rng = np.random.default_rng(args.seed + 1)
    perm_args = copy.copy(args)
    if not args.full_permutation_training:
        perm_args.epochs = min(args.epochs, args.permutation_epochs)
        perm_args.patience = min(args.patience, args.permutation_patience)
    print(
        f"\nRunning {args.n_permutations} permutation iterations "
        f"(epochs={perm_args.epochs}, patience={perm_args.patience}) ..."
    )
    for i in range(args.n_permutations):
        perm_train = PermutedJointDataset(train_ds, rng)
        perm_val = PermutedJointDataset(val_ds, rng)
        torch.manual_seed(args.seed + i + 1000)
        model_perm = JointFactorTransformer(**model_kwargs).to(device)
        model_perm = _train_model_quiet(model_perm, perm_train, perm_val, perm_args, device)
        res = evaluate_model(model_perm, test_ds, device, args.batch_size)
        for head in HEADS:
            null[head].append(float(res[head]["accuracy"]))
        print(
            f"  [{i + 1:>{len(str(args.n_permutations))}}/{args.n_permutations}] "
            + " ".join(f"{head}={res[head]['accuracy']:.3f}" for head in HEADS),
            flush=True,
        )
        del model_perm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"  {args.n_permutations}/{args.n_permutations} permutations done.")
    return null


def collect_embeddings(
    model: JointFactorTransformer,
    dataset: JointEmbeddingDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    with torch.no_grad():
        for x, *_ in loader:
            chunks.append(model.extract_embedding(x.to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not available; skipping PNG outputs.")
        return None


def plot_training_history(history: dict, save_path: Path) -> None:
    plt = _get_plt()
    if plt is None:
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Accuracy (avg phase/grip/hand)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_confusion_matrix(cm: np.ndarray, target_names: list[str], title: str, save_path: Path) -> None:
    plt = _get_plt()
    if plt is None:
        return
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticks(range(len(target_names)))
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.6 else "black",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Plots and diagnostics
# ---------------------------------------------------------------------------


def plot_permutation_test(
    observed: float,
    null_distribution: list[float],
    head: str,
    save_path: Path,
) -> None:
    plt = _get_plt()
    if plt is None:
        return
    null = np.asarray(null_distribution, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=min(20, max(5, len(null))), alpha=0.75, color="#8C8C8C")
    ax.axvline(observed, color="#C44E52", linewidth=2, label=f"observed={observed:.3f}")
    ax.set_xlabel("Accuracy under permuted labels")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation baseline: {head}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def collect_attention_weights(
    model: JointFactorTransformer,
    dataset: JointEmbeddingDataset,
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    layer_batches: list[list[np.ndarray]] | None = None
    with torch.no_grad():
        for x, *_ in loader:
            model(x.to(device))
            weights = model.get_layer_attention_weights()
            if layer_batches is None:
                layer_batches = [[] for _ in weights]
            for i, w in enumerate(weights):
                layer_batches[i].append(w.cpu().numpy())
    if layer_batches is None:
        return []
    return [np.concatenate(chunks, axis=0) for chunks in layer_batches]


def compute_area_importance(layer_weights: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    attn = layer_weights[-1]
    attn_full = attn.mean(axis=(0, 1))
    nonpad_rows = ~PADDING_MASK
    col_mean = attn_full[nonpad_rows, :].mean(axis=0)
    importance = np.zeros(N_AREAS, dtype=np.float32)
    for ai, size in enumerate(AREA_SIZES):
        t0 = ai * MAX_AREA_CHANNELS
        importance[ai] = col_mean[t0 : t0 + size].mean()
    importance = importance / (importance.sum() + 1e-10)

    attn_matrix = np.zeros((N_AREAS, N_AREAS), dtype=np.float32)
    for i, si in enumerate(AREA_SIZES):
        for j, sj in enumerate(AREA_SIZES):
            attn_matrix[i, j] = attn_full[
                i * MAX_AREA_CHANNELS : i * MAX_AREA_CHANNELS + si,
                j * MAX_AREA_CHANNELS : j * MAX_AREA_CHANNELS + sj,
            ].mean()
    return importance, attn_matrix


def plot_attention_analysis(
    importance: np.ndarray,
    attn_matrix: np.ndarray,
    title_prefix: str,
    save_path: Path,
) -> None:
    plt = _get_plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(AREA_NAMES, importance, color="#4C72B0")
    axes[0].set_ylim(0, min(1.0, float(importance.max()) * 1.35 + 0.05))
    axes[0].set_ylabel("Normalized attention received")
    axes[0].set_title(f"{title_prefix}: area importance")
    im = axes[1].imshow(attn_matrix, cmap="viridis")
    axes[1].set_xticks(range(N_AREAS))
    axes[1].set_xticklabels(AREA_NAMES, rotation=45, ha="right")
    axes[1].set_yticks(range(N_AREAS))
    axes[1].set_yticklabels(AREA_NAMES)
    axes[1].set_xlabel("Key area")
    axes[1].set_ylabel("Query area")
    axes[1].set_title("Area attention matrix")
    fig.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def compute_head_area_importance(
    model: JointFactorTransformer,
    dataset: JointEmbeddingDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    accum = {head: np.zeros(N_AREAS, dtype=np.float64) for head in HEADS}
    ch_mask = torch.tensor(CHANNEL_VALID, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)

    for x, y_phase, y_grip, y_hand, _ in loader:
        targets = {
            "phase": y_phase.to(device),
            "grip": y_grip.to(device),
            "hand": y_hand.to(device),
        }
        for head in HEADS:
            x_in = x.to(device).detach().requires_grad_(True)
            lp, lg, lh = model(x_in)
            logits = {"phase": lp, "grip": lg, "hand": lh}[head]
            loss = criterion(logits, targets[head])
            model.zero_grad()
            loss.backward()
            if x_in.grad is None:
                continue
            grad = x_in.grad.abs() * ch_mask
            for ai, size in enumerate(AREA_SIZES):
                accum[head][ai] += float(grad[:, ai, :size, :].mean().detach().cpu())

    result = {}
    for head in HEADS:
        vals = accum[head]
        result[head] = (vals / (vals.sum() + 1e-10)).astype(np.float32)
    return result


def plot_embedding_structure(
    embeddings: np.ndarray,
    y_phase: np.ndarray,
    y_grip: np.ndarray,
    y_hand: np.ndarray,
    out_dir: Path,
    seed: int = 42,
) -> None:
    """4 plots characterising how phase / grip / hand organise the embedding space."""
    from sklearn.decomposition import PCA

    plt = _get_plt()
    if plt is None:
        return

    GRIP_NAMES = ["power", "precision"]
    HAND_NAMES = ["left", "right"]

    # --- PCA ---
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(embeddings)
    ev = pca.explained_variance_ratio_

    # ordered conditions: phase × grip × hand
    combo_keys = [
        (ph, gr, ha)
        for ph in range(len(PHASE_NAMES))
        for gr in range(2)
        for ha in range(2)
    ]
    combo_label = {
        (ph, gr, ha): f"{PHASE_NAMES[ph][:3]}+{GRIP_NAMES[gr][:2]}+{HAND_NAMES[ha][0].upper()}"
        for ph, gr, ha in combo_keys
    }

    # ── Plot 1: phase ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    phase_colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ph, (name, col) in enumerate(zip(PHASE_NAMES, phase_colors)):
        m = y_phase == ph
        ax.scatter(coords[m, 0], coords[m, 1], c=col, label=name, alpha=0.35, s=8, linewidths=0)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.set_title("Embedding space — by phase")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_pca_phase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'embedding_pca_phase.png'}")

    # ── Plot 2: grip+hand (4 colours) ─────────────────────────────────────────
    gh_colors = ["#4C72B0", "#9AC4E8", "#DD8452", "#F4A261"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for gi, gr in enumerate(range(2)):
        for hi, ha in enumerate(range(2)):
            m = (y_grip == gr) & (y_hand == ha)
            ax.scatter(coords[m, 0], coords[m, 1], c=gh_colors[gi * 2 + hi],
                       label=f"{GRIP_NAMES[gr]}+{HAND_NAMES[ha]}", alpha=0.35, s=8, linewidths=0)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.set_title("Embedding space — by grip+hand")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_pca_grip_hand.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'embedding_pca_grip_hand.png'}")

    # ── Plot 3: full combo (up to 12 colours) ─────────────────────────────────
    cmap12 = plt.cm.get_cmap("tab20", 12)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, key in enumerate(combo_keys):
        ph, gr, ha = key
        m = (y_phase == ph) & (y_grip == gr) & (y_hand == ha)
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], c=[cmap12(i)],
                   label=combo_label[key], alpha=0.45, s=10, linewidths=0)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.set_title("Embedding space — all phase×grip×hand combinations")
    ax.legend(markerscale=3, fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_pca_full_combo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'embedding_pca_full_combo.png'}")

    # ── Plot 4: centroid distance matrix ──────────────────────────────────────
    present_keys = []
    centroids = []
    for key in combo_keys:
        ph, gr, ha = key
        m = (y_phase == ph) & (y_grip == gr) & (y_hand == ha)
        if m.any():
            present_keys.append(key)
            centroids.append(embeddings[m].mean(axis=0))
    centroids = np.stack(centroids)

    n = len(present_keys)
    dist_mat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    tick_labels = [combo_label[k] for k in present_keys]

    # phase block separators (dynamic — handles missing held-out condition)
    phase_counts = [sum(1 for k in present_keys if k[0] == ph) for ph in range(len(PHASE_NAMES))]
    separators = []
    cs = 0
    for count in phase_counts[:-1]:
        cs += count
        separators.append(cs - 0.5)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(dist_mat, cmap="viridis_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="L2 distance")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist_mat[i, j]:.1f}", ha="center", va="center",
                    fontsize=6, color="white" if dist_mat[i, j] < dist_mat.max() * 0.6 else "black")
    for sep in separators:
        ax.axhline(sep, color="white", linewidth=2)
        ax.axvline(sep, color="white", linewidth=2)
    ax.set_title("Centroid distance matrix (ordered: phase → grip+hand within phase)")
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_centroid_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'embedding_centroid_matrix.png'}")


def plot_head_area_importance(head_importance: dict[str, np.ndarray], save_path: Path) -> None:
    plt = _get_plt()
    if plt is None:
        return
    x = np.arange(N_AREAS)
    width = 0.24
    colors = {"phase": "#4C72B0", "grip": "#DD8452", "hand": "#55A868"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, head in zip((-width, 0.0, width), HEADS):
        ax.bar(x + offset, head_importance[head], width=width, label=head, color=colors[head])
    ax.set_xticks(x)
    ax.set_xticklabels(AREA_NAMES)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Normalized gradient saliency")
    ax.set_title("Head-wise area importance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def compute_class_area_importance(
    layer_weights: list[np.ndarray],
    dataset: JointEmbeddingDataset,
) -> dict[str, np.ndarray]:
    attn = layer_weights[-1]
    col_mean = attn.mean(axis=1)[:, ~PADDING_MASK, :].mean(axis=1)
    sample_area = np.zeros((attn.shape[0], N_AREAS), dtype=np.float32)
    for ai, size in enumerate(AREA_SIZES):
        t0 = ai * MAX_AREA_CHANNELS
        sample_area[:, ai] = col_mean[:, t0 : t0 + size].mean(axis=1)
    sample_area = sample_area / (sample_area.sum(axis=1, keepdims=True) + 1e-10)

    labels = {
        "phase": dataset.y_phase.numpy(),
        "grip": dataset.y_grip.numpy(),
        "hand": dataset.y_hand.numpy(),
    }
    out = {}
    for head in HEADS:
        matrix = np.zeros((len(CLASS_NAMES[head]), N_AREAS), dtype=np.float32)
        for cls_id in range(len(CLASS_NAMES[head])):
            mask = labels[head] == cls_id
            if mask.any():
                row = sample_area[mask].mean(axis=0)
                matrix[cls_id] = row / (row.sum() + 1e-10)
        out[head] = matrix
    return out


def plot_class_area_attention(
    class_importance: dict[str, np.ndarray],
    split_label: str,
    save_path: Path,
) -> None:
    plt = _get_plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    x = np.arange(N_AREAS)
    for ax, head in zip(axes, HEADS):
        matrix = class_importance[head]
        width = min(0.8 / max(matrix.shape[0], 1), 0.24)
        offsets = (np.arange(matrix.shape[0]) - (matrix.shape[0] - 1) / 2.0) * width
        for cls_id, offset in enumerate(offsets):
            ax.bar(x + offset, matrix[cls_id], width=width, label=CLASS_NAMES[head][cls_id])
        ax.set_xticks(x)
        ax.set_xticklabels(AREA_NAMES)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{head}: attention by true class")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Normalized attention received")
    fig.suptitle(f"Joint transformer class-conditional area attention ({split_label})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def save_evaluation_outputs(results: dict, prefix: str, out_dir: Path, make_plots: bool = True) -> None:
    for head in HEADS:
        cm = np.asarray(results[head]["confusion_matrix"], dtype=np.int64)
        np.save(out_dir / f"{prefix}_{head}_confusion_matrix.npy", cm)
        if make_plots:
            plot_confusion_matrix(
                cm,
                CLASS_NAMES[head],
                f"{prefix.replace('_', ' ')} {head}",
                out_dir / f"{prefix}_{head}_confusion_matrix.png",
            )


def _jsonify_history(history: dict) -> dict:
    return {k: [float(vv) for vv in v] for k, v in history.items()}


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def save_checkpoint(
    out_dir: Path,
    model: JointFactorTransformer,
    args: argparse.Namespace,
    stats: dict,
    history: dict,
) -> Path:
    checkpoint = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "norm_stats": stats,
            "history": history,
            "class_names": CLASS_NAMES,
        },
        checkpoint,
    )
    print(f"Saved checkpoint to {checkpoint}")
    return checkpoint


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.dry_run:
        args.epochs = min(args.epochs, 2)
        args.patience = 1
        args.n_permutations = min(args.n_permutations, 3)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    out_dir = Path(args.out_dir)
    validate_output_dir(args, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input_mode} trials from {args.data_dir}")
    trial_data = load_joint_trials(Path(args.data_dir), args.input_mode)
    flat = phase_expand(trial_data)
    if args.dry_run and len(flat["y_phase"]) > 600:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(flat["y_phase"]), size=600, replace=False))
        flat = subset_flat(flat, keep)
    features = extract_and_cache_features(flat, args.cache_dir)

    if args.heldout:
        train_idx, val_idx, test_idx, heldout_test_idx = _heldout_split_indices(
            flat,
            PHASE_NAMES.index(args.heldout_phase),
            GRIP_TO_ID[args.heldout_grip],
            HAND_TO_ID[args.heldout_hand],
            args.seed,
        )
    else:
        train_idx, val_idx, test_idx = _normal_split_indices(flat, args.seed)
        heldout_test_idx = None
    print(
        "Split sizes | "
        f"train={len(train_idx)} val={len(val_idx)} seen_test={len(test_idx)} "
        f"heldout_test={0 if heldout_test_idx is None else len(heldout_test_idx)}"
    )

    stats = compute_norm_stats(features, train_idx)
    train_ds = JointEmbeddingDataset(features, flat, train_idx, stats)
    val_ds = JointEmbeddingDataset(features, flat, val_idx, stats)
    test_ds = JointEmbeddingDataset(features, flat, test_idx, stats)
    heldout_ds = (
        None if heldout_test_idx is None
        else JointEmbeddingDataset(features, flat, heldout_test_idx, stats)
    )

    n_bands = 1 if args.input_mode == "mu" else len(BAND_NAMES_6)
    model_kwargs = {
        "n_bands": n_bands,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
    }
    model = JointFactorTransformer(
        n_bands=n_bands,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    ).to(device)
    model, history = train_model(model, train_ds, val_ds, args, device)

    seen_results = evaluate_model(model, test_ds, device, args.batch_size)
    heldout_results = (
        None if heldout_ds is None
        else evaluate_model(model, heldout_ds, device, args.batch_size)
    )
    classification_accuracy = {
        "seen_test": {head: float(seen_results[head]["accuracy"]) for head in HEADS},
        "heldout_test": None if heldout_results is None else {
            head: float(heldout_results[head]["accuracy"]) for head in HEADS
        },
    }

    np.savez_compressed(out_dir / "normalization_stats.npz", **stats)
    save_evaluation_outputs(seen_results, "seen", out_dir, make_plots=not args.no_plot)
    if heldout_results is not None:
        save_evaluation_outputs(heldout_results, "heldout", out_dir, make_plots=not args.no_plot)
    checkpoint = save_checkpoint(out_dir, model, args, stats, history)

    null_distributions = None
    p_values = None
    if not args.skip_permutation and args.n_permutations > 0:
        null_distributions = run_permutation_test(
            train_ds,
            val_ds,
            test_ds,
            model_kwargs,
            args,
            device,
        )
        p_values = {}
        for head in HEADS:
            observed = float(seen_results[head]["accuracy"])
            null = np.asarray(null_distributions[head], dtype=float)
            p_values[head] = float((np.sum(null >= observed) + 1) / (len(null) + 1))
            if not args.no_plot:
                plot_permutation_test(
                    observed,
                    null_distributions[head],
                    head,
                    out_dir / f"permutation_{head}.png",
                )

    area_importance_seen = None
    attention_matrix_seen = None
    area_importance_heldout = None
    attention_matrix_heldout = None
    head_area_importance = None
    class_area_attention_seen = None
    if not args.no_plot:
        plot_training_history(history, out_dir / "training_curves.png")
        print("\nExtracting attention weights (seen test)...")
        seen_attn = collect_attention_weights(model, test_ds, device, args.batch_size)
        if seen_attn:
            area_importance_seen, attention_matrix_seen = compute_area_importance(seen_attn)
            class_area_attention_seen = compute_class_area_importance(seen_attn, test_ds)
            plot_attention_analysis(
                area_importance_seen,
                attention_matrix_seen,
                "Joint transformer seen test",
                out_dir / "attention_seen.png",
            )
            plot_class_area_attention(
                class_area_attention_seen,
                "seen test",
                out_dir / "attention_by_class_seen.png",
            )
        if heldout_ds is not None:
            print("Extracting attention weights (held-out test)...")
            heldout_attn = collect_attention_weights(model, heldout_ds, device, args.batch_size)
            if heldout_attn:
                area_importance_heldout, attention_matrix_heldout = compute_area_importance(heldout_attn)
                plot_attention_analysis(
                    area_importance_heldout,
                    attention_matrix_heldout,
                    "Joint transformer held-out test",
                    out_dir / "attention_heldout.png",
                )
        print("Computing head-wise area saliency...")
        head_area_importance = compute_head_area_importance(model, test_ds, device, args.batch_size)
        plot_head_area_importance(head_area_importance, out_dir / "head_area_importance_gradcam.png")

    emb_seen = collect_embeddings(model, test_ds, device, args.batch_size)
    np.savez_compressed(
        out_dir / "seen_embeddings.npz",
        embeddings=emb_seen,
        y_phase=flat["y_phase"][test_idx],
        y_grip=flat["y_grip"][test_idx],
        y_hand=flat["y_hand"][test_idx],
        y_angle=flat["y_angle"][test_idx],
    )

    if not args.no_plot:
        print("\nCollecting all seen embeddings for structure analysis...")
        all_seen_idx = np.concatenate([train_idx, val_idx, test_idx])
        all_seen_ds = JointEmbeddingDataset(features, flat, all_seen_idx, stats)
        emb_all_seen = collect_embeddings(model, all_seen_ds, device, args.batch_size)
        plot_embedding_structure(
            emb_all_seen,
            flat["y_phase"][all_seen_idx],
            flat["y_grip"][all_seen_idx],
            flat["y_hand"][all_seen_idx],
            out_dir,
            seed=args.seed,
        )
    if heldout_ds is not None and heldout_test_idx is not None:
        emb_heldout = collect_embeddings(model, heldout_ds, device, args.batch_size)
        np.savez_compressed(
            out_dir / "heldout_embeddings.npz",
            embeddings=emb_heldout,
            y_phase=flat["y_phase"][heldout_test_idx],
            y_grip=flat["y_grip"][heldout_test_idx],
            y_hand=flat["y_hand"][heldout_test_idx],
            y_angle=flat["y_angle"][heldout_test_idx],
        )

    summary = {
        "input_mode": args.input_mode,
        "n_samples": int(len(flat["y_phase"])),
        "split_protocol": "strict_zero_shot" if args.heldout else "standard_stratified",
        "split_note": (
            "held-out phase/grip/hand samples are excluded from train and val; "
            "they are used only for final heldout_test evaluation"
        ) if args.heldout else "no held-out combination requested",
        "splits": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "seen_test": int(len(test_idx)),
            "heldout_test": int(0 if heldout_test_idx is None else len(heldout_test_idx)),
        },
        "heldout": {
            "phase": args.heldout_phase,
            "grip": args.heldout_grip,
            "hand": args.heldout_hand,
        } if args.heldout else None,
        "history": _jsonify_history(history),
        "classification_accuracy": classification_accuracy,
        "p_values": p_values,
        "permutation_test": None if null_distributions is None else {
            "n_permutations": args.n_permutations,
            "epochs_per_permutation": (
                args.epochs if args.full_permutation_training else min(args.epochs, args.permutation_epochs)
            ),
            "patience_per_permutation": (
                args.patience if args.full_permutation_training else min(args.patience, args.permutation_patience)
            ),
            "null_distributions": null_distributions,
            "note": "Each iteration retrains a fresh joint transformer with independently permuted phase, grip, and hand labels.",
        },
        "seen_test": seen_results,
        "heldout_test": heldout_results,
        "area_importance_seen": None if area_importance_seen is None else dict(zip(AREA_NAMES, area_importance_seen.tolist())),
        "area_importance_heldout": None if area_importance_heldout is None else dict(zip(AREA_NAMES, area_importance_heldout.tolist())),
        "attention_matrix_seen": None if attention_matrix_seen is None else {
            "matrix": attention_matrix_seen.tolist(),
            "note": "matrix[i][j] = attention from area i (query) to area j (key)",
        },
        "attention_matrix_heldout": None if attention_matrix_heldout is None else {
            "matrix": attention_matrix_heldout.tolist(),
            "note": "matrix[i][j] = attention from area i (query) to area j (key)",
        },
        "head_area_importance_gradcam": None if head_area_importance is None else {
            head: dict(zip(AREA_NAMES, vals.tolist()))
            for head, vals in head_area_importance.items()
        },
        "class_area_attention_seen": None if class_area_attention_seen is None else {
            head: {
                "class_names": CLASS_NAMES[head],
                "area_names": AREA_NAMES,
                "matrix": matrix.tolist(),
                "note": "matrix[c][j] = normalized mean attention received by area j for true class c",
            }
            for head, matrix in class_area_attention_seen.items()
        },
        "checkpoint": str(checkpoint),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nJoint embedding summary:")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"history"}}, indent=2))


if __name__ == "__main__":
    main()
