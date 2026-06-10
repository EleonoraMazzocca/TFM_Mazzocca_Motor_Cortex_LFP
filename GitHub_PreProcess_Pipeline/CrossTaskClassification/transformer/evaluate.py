from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


GRIP_NAMES = ["power", "precision"]
HAND_NAMES = ["left", "right"]
ANGLE_NAMES = ["0°", "45°", "90°", "135°"]

HEADS = {
    "grip": GRIP_NAMES,
    "hand": HAND_NAMES,
    "angle": ANGLE_NAMES,
}


def evaluate_model(
    model,
    dataset,
    batch_size: int = 64,
    device: torch.device | None = None,
    use_instruction: bool = False,
) -> dict[str, dict]:
    if use_instruction:
        # Safety: prevent accidentally evaluating with live instruction labels.
        # BalancedInstructionDataset sets is_test=True to zero all instructions.
        # LFPDataset and custom ablation wrappers must set is_test=True explicitly.
        assert getattr(dataset, "is_test", False), (
            "evaluate_model called with use_instruction=True but dataset.is_test is not True. "
            "Use BalancedInstructionDataset(is_test=True) to guarantee zero instructions at evaluation."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits: dict[str, list] = {"grip": [], "hand": [], "angle": []}
    all_labels: dict[str, list] = {"grip": [], "hand": [], "angle": []}

    with torch.no_grad():
        for batch in loader:
            x, y_grip, y_hand, y_angle, instr = batch
            x = x.to(device)
            instr = instr.to(device)
            if use_instruction:
                lg, lh, la = model(x, instr)
            else:
                lg, lh, la = model(x)
            all_logits["grip"].append(lg.cpu())
            all_logits["hand"].append(lh.cpu())
            all_logits["angle"].append(la.cpu())
            all_labels["grip"].append(y_grip)
            all_labels["hand"].append(y_hand)
            all_labels["angle"].append(y_angle)

    results: dict[str, dict] = {}
    for head, names in HEADS.items():
        logits = torch.cat(all_logits[head])
        labels = torch.cat(all_labels[head]).numpy()
        preds = logits.argmax(dim=1).numpy()
        cm = confusion_matrix(labels, preds, labels=np.arange(len(names)))
        results[head] = {
            "accuracy": float((preds == labels).mean()),
            "confusion_matrix": cm,
            "report": classification_report(
                labels,
                preds,
                labels=np.arange(len(names)),
                target_names=names,
                digits=4,
                zero_division=0,
            ),
            "y_true": labels,
            "y_pred": preds,
        }

    return results


def print_results(results: dict[str, dict], title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for head, names in HEADS.items():
        acc = results[head]["accuracy"]
        print(f"  {head:6s}  accuracy={acc:.4f}")
    print()
    for head in HEADS:
        print(f"--- {head} ---")
        print(results[head]["report"])


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: list[str],
    title: str = "Confusion Matrix",
    save_path: str | None = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping confusion matrix plot")
        return None

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
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    return fig


def plot_training_history(history: dict, save_path: str | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping training curves plot")
        return None

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (avg across heads)")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    return fig
