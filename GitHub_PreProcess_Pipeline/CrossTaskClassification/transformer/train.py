from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


DEFAULT_CONFIG = {
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 40,
    "patience": 8,
    "grad_clip_norm": 1.0,
}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float | None,
    use_instruction: bool = False,
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    is_train = optimizer is not None
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    if is_train:
        model.train()
    else:
        model.eval()

    for batch in loader:
        x, y_grip, y_hand, y_angle, instr = batch
        x = x.to(device)
        y_grip = y_grip.to(device)
        y_hand = y_hand.to(device)
        y_angle = y_angle.to(device)
        instr = instr.to(device)

        with torch.set_grad_enabled(is_train):
            if use_instruction:
                logits_grip, logits_hand, logits_angle = model(x, instr)
            else:
                logits_grip, logits_hand, logits_angle = model(x)
            loss = (
                criterion(logits_grip, y_grip)
                + criterion(logits_hand, y_hand)
                + criterion(logits_angle, y_angle)
            )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        n = x.shape[0]
        total_loss += loss.item() * n
        acc = (
            (logits_grip.argmax(1) == y_grip).float()
            + (logits_hand.argmax(1) == y_hand).float()
            + (logits_angle.argmax(1) == y_angle).float()
        ).sum().item() / 3.0
        total_correct += acc
        total_examples += n

    return total_loss / total_examples, total_correct / total_examples


def train_model(
    model: nn.Module,
    train_dataset,
    val_dataset,
    config: dict | None = None,
    save_path: str | None = None,
    use_instruction: bool = False,
) -> tuple[nn.Module, dict[str, list[float]]]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = model.to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(cfg["epochs"]):
        # Recompute balanced instruction masks at the start of every epoch.
        # BalancedInstructionDataset provides this method; LFPDataset does not.
        if hasattr(train_dataset, "reshuffle_masks"):
            train_dataset.reshuffle_masks()

        train_loss, train_acc = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=cfg["grad_clip_norm"],
            use_instruction=use_instruction,
        )
        val_loss, val_acc = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            grad_clip_norm=None,
            use_instruction=use_instruction,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:02d}/{cfg['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best val_loss={best_val_loss:.4f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path:
        torch.save(
            {
                "model_state": best_state,
                "config": cfg,
                "history": history,
            },
            save_path,
        )
        print(f"Saved checkpoint to {save_path}")

    return model, history
