from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))

from data import (
    ANGLE_TO_ID,
    AREA_FEATURE_DIM,
    HAND_TO_ID,
    N_AREAS,
    N_PHASES,
    PHASE_NAMES,
    extract_area_features,
    load_dataset,
)
from evaluate import ANGLE_NAMES, plot_confusion_matrix, plot_training_history
from train import DEFAULT_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a transformer to classify angle using only right-hand unimanual "
            "trials."
        )
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/transformer_right_hand_angle",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--feedforward_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_plot", action="store_true")
    return parser.parse_args()


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _subset(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    subset: dict[str, np.ndarray] = {
        "file_paths": data["file_paths"],
        "n_channels": data["n_channels"],
    }
    for key in ("file_idx", "trial_idx", "y_grip", "y_hand", "y_angle", "is_heldout"):
        subset[key] = data[key][idx]
    return subset


def _load_sample_from_data(
    data: dict[str, np.ndarray],
    idx: int,
    file_cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    file_path = data["file_paths"][data["file_idx"][idx]]
    if file_cache is not None and file_path in file_cache:
        features = file_cache[file_path]
    else:
        features = np.load(file_path, mmap_mode="r")
        if file_cache is not None:
            file_cache[file_path] = features
    sample = features[data["trial_idx"][idx], :, :, :]
    return np.mean(np.abs(sample.astype(np.float32)), axis=-1)


def _normalise(train_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_train = len(train_data["y_angle"])
    n_channels = int(train_data["n_channels"])
    n_phases = len(PHASE_NAMES)
    sum_x = np.zeros((n_phases, n_channels), dtype=np.float64)
    sum_x2 = np.zeros((n_phases, n_channels), dtype=np.float64)
    file_cache: dict[str, np.ndarray] = {}

    for idx in range(n_train):
        sample = _load_sample_from_data(train_data, idx, file_cache=file_cache).astype(np.float64)
        sum_x += sample
        sum_x2 += sample ** 2

    mu = sum_x / max(n_train, 1)
    variance = (sum_x2 / max(n_train, 1)) - mu ** 2
    sigma = np.sqrt(np.maximum(variance, 0.0))
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return {"mu": mu, "sigma": sigma}


def make_right_hand_angle_split(
    data: dict[str, np.ndarray],
    seed: int,
    val_size: float,
    test_size: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not 0.0 < val_size < 1.0:
        raise ValueError("--val_size must be between 0 and 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test_size must be between 0 and 1.")
    if val_size + test_size >= 1.0:
        raise ValueError("--val_size + --test_size must be less than 1.")

    right_hand_id = HAND_TO_ID["right"]
    all_idx = np.arange(len(data["y_angle"]))
    right_idx = all_idx[data["y_hand"] == right_hand_id]

    if len(right_idx) == 0:
        raise ValueError("No right-hand trials found.")

    strat_labels = np.array([str(data["y_angle"][i]) for i in right_idx])

    train_idx, test_idx = train_test_split(
        right_idx,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=strat_labels,
    )
    train_labels = np.array([str(data["y_angle"][i]) for i in train_idx])
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=relative_val_size,
        random_state=seed,
        shuffle=True,
        stratify=train_labels,
    )

    train_data = _subset(data, train_idx)
    val_data = _subset(data, val_idx)
    test_data = _subset(data, test_idx)
    norm_stats = _normalise(train_data)

    print(
        "Right-hand split sizes | "
        f"train={len(train_data['y_angle'])} "
        f"val={len(val_data['y_angle'])} "
        f"test={len(test_data['y_angle'])}"
    )
    return train_data, val_data, test_data, norm_stats


class RightHandAngleDataset(Dataset):
    def __init__(
        self,
        data: dict[str, np.ndarray],
        norm_stats: dict[str, np.ndarray] | None = None,
    ):
        self.file_paths = [os.fspath(path) for path in data["file_paths"].tolist()]
        self.file_idx = data["file_idx"]
        self.trial_idx = data["trial_idx"]
        self.y_angle = torch.tensor(data["y_angle"], dtype=torch.long)
        self.mu = None if norm_stats is None else torch.tensor(norm_stats["mu"], dtype=torch.float32)
        self.sigma = None if norm_stats is None else torch.tensor(norm_stats["sigma"], dtype=torch.float32)
        self._file_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.y_angle)

    def _get_features(self, file_path: str) -> np.ndarray:
        features = self._file_cache.get(file_path)
        if features is None:
            features = np.load(file_path, mmap_mode="r")
            self._file_cache[file_path] = features
        return features

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[self.file_idx[idx]]
        features = self._get_features(file_path)
        x = torch.from_numpy(
            np.mean(
                np.abs(features[self.trial_idx[idx], :, :, :].astype(np.float32)),
                axis=-1,
            )
        )
        if self.mu is not None and self.sigma is not None:
            zero_mask = x == 0.0
            x = (x - self.mu) / self.sigma
            x[zero_mask] = 0.0
        return x, self.y_angle[idx]


class RightHandAngleTransformer(nn.Module):
    def __init__(
        self,
        n_channels: int = 256,
        n_phases: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.phase_embedding = nn.Parameter(torch.zeros(1, n_phases, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head_angle = nn.Linear(d_model, len(ANGLE_TO_ID))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.phase_embedding
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head_angle(x)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float | None,
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    is_train = optimizer is not None
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    if is_train:
        model.train()
    else:
        model.eval()

    for x, y_angle in loader:
        x = x.to(device)
        y_angle = y_angle.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y_angle)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        n = x.shape[0]
        total_loss += loss.item() * n
        total_correct += int((logits.argmax(dim=1) == y_angle).sum().item())
        total_examples += n

    return total_loss / total_examples, total_correct / total_examples


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict,
    save_path: str | None = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = model.to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
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

    for epoch in range(config["epochs"]):
        train_loss, train_acc = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config["grad_clip_norm"],
        )
        val_loss, val_acc = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            grad_clip_norm=None,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:02d}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
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
                "config": config,
                "history": history,
            },
            save_path,
        )
        print(f"Saved checkpoint to {save_path}")

    return model, history


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y_angle in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y_angle)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(dim=1).numpy()
    cm = confusion_matrix(labels, preds, labels=np.arange(len(ANGLE_NAMES)))

    return {
        "accuracy": float((preds == labels).mean()),
        "confusion_matrix": cm,
        "report": classification_report(
            labels,
            preds,
            labels=np.arange(len(ANGLE_NAMES)),
            target_names=ANGLE_NAMES,
            digits=4,
            zero_division=0,
        ),
        "y_true": labels,
        "y_pred": preds,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(cache_dir=args.cache_dir)
    right_trials = int((data["y_hand"] == HAND_TO_ID["right"]).sum())
    print(
        "Dataset loaded: "
        f"trials={len(data['y_angle'])} "
        f"right_hand_trials={right_trials} "
        f"channels={int(data['n_channels'])}"
    )
    print(f"Using phases: {', '.join(PHASE_NAMES)}")

    train_data, val_data, test_data, norm_stats = make_right_hand_angle_split(
        data=data,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    train_ds = RightHandAngleDataset(train_data, norm_stats=norm_stats)
    val_ds = RightHandAngleDataset(val_data, norm_stats=norm_stats)
    test_ds = RightHandAngleDataset(test_data, norm_stats=norm_stats)

    model = RightHandAngleTransformer(
        n_channels=int(data["n_channels"]),
        n_phases=len(PHASE_NAMES),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    config = {
        **DEFAULT_CONFIG,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
    }
    checkpoint_path = out_dir / "checkpoint.pt"
    model, history = train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        save_path=str(checkpoint_path),
    )

    test_results = evaluate_model(model, test_ds, batch_size=args.batch_size)
    print("\nRight-hand angle results")
    print("========================")
    print(f"  accuracy={test_results['accuracy']:.4f}")
    print()
    print(test_results["report"])

    np.save(out_dir / "angle_confusion_matrix.npy", test_results["confusion_matrix"])
    np.savez_compressed(out_dir / "normalization_stats.npz", **norm_stats)
    _save_text(out_dir / "angle_report.txt", test_results["report"])

    summary = {
        "task": "right_hand_angle_classification",
        "phases": PHASE_NAMES,
        "hand_filter": "right",
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "test_accuracy": test_results["accuracy"],
        "config": config,
        "model": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "feedforward_dim": args.feedforward_dim,
            "dropout": args.dropout,
            "n_params": n_params,
        },
    }
    _save_text(out_dir / "summary.json", json.dumps(summary, indent=2))

    if not args.no_plot:
        plot_confusion_matrix(
            test_results["confusion_matrix"],
            target_names=ANGLE_NAMES,
            title="Angle - right hand only",
            save_path=str(out_dir / "angle_confusion_matrix.png"),
        )
        plot_training_history(history, save_path=str(out_dir / "training_curves.png"))


if __name__ == "__main__":
    main()
