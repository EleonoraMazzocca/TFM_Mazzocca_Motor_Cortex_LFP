import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with pip or conda before running this script."
    ) from exc

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required for masked-channel reconstruction. "
        "Install it first, for example from https://pytorch.org/get-started/locally/."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from data_paths import CLEANED_STRUCTURED_DIR
except ImportError:
    from GitHub_PreProcess_Pipeline.CrossTaskClassification.data_paths import CLEANED_STRUCTURED_DIR


PHASE_NAMES = ("PREREACH", "REACH", "GRASP")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_load_array(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    if path.suffix != ".npy":
        raise ValueError(f"Expected a .npy file, got {path}")
    array = np.load(path)
    if array.ndim != 4:
        raise ValueError(
            f"Expected array shape (n_trials, 3, channels, time), got {array.shape} from {path}"
        )
    return array


def _default_real_data_path() -> Optional[Path]:
    candidates = sorted(CLEANED_STRUCTURED_DIR.glob("data_*.npy"))
    return candidates[0] if candidates else None


def _default_manifest_path() -> Optional[Path]:
    candidate = Path("GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/reconstruction_manifest.json")
    return candidate if candidate.exists() else None


def generate_synthetic_lfp(
    n_trials: int,
    n_phases: int,
    n_channels: int,
    n_time: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0.0, 1.0, n_time, endpoint=False, dtype=np.float32)
    data = np.zeros((n_trials, n_phases, n_channels, n_time), dtype=np.float32)
    base_freqs = rng.uniform(2.0, 40.0, size=(n_trials, 3))
    coupling = rng.normal(0.0, 0.45, size=(n_channels, 3)).astype(np.float32)
    phase_offsets = rng.uniform(-0.8, 0.8, size=(n_phases, 3)).astype(np.float32)
    channel_bias = rng.normal(0.0, 0.15, size=(n_channels,)).astype(np.float32)

    for trial_idx in range(n_trials):
        latent_components = []
        for component_idx in range(3):
            freq = base_freqs[trial_idx, component_idx]
            waveform = (
                np.sin(2.0 * math.pi * freq * time_axis + rng.uniform(0.0, 2.0 * math.pi))
                + 0.5 * np.sin(2.0 * math.pi * (freq * 0.5) * time_axis + rng.uniform(0.0, 2.0 * math.pi))
            ).astype(np.float32)
            latent_components.append(waveform)
        latent_components = np.stack(latent_components, axis=0)

        for phase_idx in range(n_phases):
            phase_mix = latent_components + phase_offsets[phase_idx, :, None]
            shared_noise = rng.normal(0.0, 0.05, size=(n_time,)).astype(np.float32)
            for channel_idx in range(n_channels):
                waveform = (
                    coupling[channel_idx] @ phase_mix
                    + channel_bias[channel_idx]
                    + shared_noise
                    + rng.normal(0.0, 0.08, size=(n_time,)).astype(np.float32)
                )
                data[trial_idx, phase_idx, channel_idx] = waveform.astype(np.float32)
    return data


def flatten_trials_and_phases(data: np.ndarray, phases: Optional[Sequence[int]] = None) -> np.ndarray:
    if phases is not None:
        data = data[:, phases]
    n_trials, n_phases, n_channels, n_time = data.shape
    return data.reshape(n_trials * n_phases, n_channels, n_time)


def load_manifest(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "splits" not in manifest:
        raise ValueError(f"Manifest missing 'splits': {path}")
    return manifest


def load_samples_from_manifest(
    manifest: Dict[str, object],
    split_name: str,
    phases: Optional[Sequence[int]],
) -> np.ndarray:
    split_entries = manifest["splits"][split_name]
    by_path: Dict[str, List[int]] = {}
    for entry in split_entries:
        by_path.setdefault(entry["path"], []).append(int(entry["trial_index"]))

    split_arrays: List[np.ndarray] = []
    for path_str, trial_indices in sorted(by_path.items()):
        array = np.load(path_str, mmap_mode="r")
        selected = np.asarray(sorted(trial_indices), dtype=np.int64)
        subset = np.asarray(array[selected], dtype=np.float32)
        split_arrays.append(flatten_trials_and_phases(subset, phases=phases))

    if not split_arrays:
        raise ValueError(f"No samples found for split '{split_name}' in manifest.")
    return np.concatenate(split_arrays, axis=0)


def select_subset(data: np.ndarray, max_samples: int, max_channels: int, downsample: int) -> np.ndarray:
    limited = data[: min(max_samples, data.shape[0]), : min(max_channels, data.shape[1])]
    if downsample > 1:
        limited = limited[:, :, ::downsample]
    return limited.astype(np.float32, copy=False)


def split_indices(
    n_samples: int,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Expected train_frac > 0, val_frac > 0, and train_frac + val_frac < 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    train_end = int(round(n_samples * train_frac))
    val_end = train_end + int(round(n_samples * val_frac))
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("One of the splits is empty. Increase sample count or adjust split ratios.")
    return train_idx, val_idx, test_idx


@dataclass
class StandardizationStats:
    mean: np.ndarray
    std: np.ndarray


def compute_train_stats(train_data: np.ndarray) -> StandardizationStats:
    mean = train_data.mean(axis=(0, 2), keepdims=True)
    std = train_data.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return StandardizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_standardization(data: np.ndarray, stats: StandardizationStats) -> np.ndarray:
    return ((data - stats.mean) / stats.std).astype(np.float32)


class MaskedChannelDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        seed: int,
        allowed_mask_channels: Optional[Sequence[int]] = None,
        fixed_mask_channel: Optional[int] = None,
    ):
        if data.ndim != 3:
            raise ValueError(f"Expected shape (n_samples, channels, time), got {data.shape}")
        self.data = data.astype(np.float32, copy=False)
        self.n_samples, self.n_channels, self.n_time = self.data.shape
        rng = np.random.default_rng(seed)
        if allowed_mask_channels is None:
            allowed = np.arange(self.n_channels, dtype=np.int64)
        else:
            allowed = np.asarray(sorted(set(int(idx) for idx in allowed_mask_channels)), dtype=np.int64)
        if allowed.size == 0:
            raise ValueError("allowed_mask_channels cannot be empty.")
        if np.any(allowed < 0) or np.any(allowed >= self.n_channels):
            raise ValueError(f"allowed_mask_channels must be in [0, {self.n_channels - 1}]")

        if fixed_mask_channel is not None:
            if int(fixed_mask_channel) not in set(allowed.tolist()):
                raise ValueError("fixed_mask_channel must also be present in allowed_mask_channels.")
            self.masked_channel_indices = np.full(self.n_samples, int(fixed_mask_channel), dtype=np.int64)
        else:
            self.masked_channel_indices = rng.choice(allowed, size=self.n_samples, replace=True)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.data[index]
        masked_channel = int(self.masked_channel_indices[index])
        x_masked = sample.copy()
        x_masked[masked_channel] = 0.0
        y_target = sample[masked_channel].copy()
        return {
            "x_masked": torch.from_numpy(x_masked.T.copy()),
            "y_target": torch.from_numpy(y_target.copy()),
            "masked_channel_index": torch.tensor(masked_channel, dtype=torch.long),
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerReconstructor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
        channel_embed_dim: int,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.channel_embedding = nn.Embedding(input_channels, channel_embed_dim)
        self.output_head = nn.Sequential(
            nn.Linear(d_model + channel_embed_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, masked_channel_index: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        hidden = self.positional_encoding(hidden)
        hidden = self.encoder(hidden)
        channel_embedding = self.channel_embedding(masked_channel_index).unsqueeze(1).expand(-1, hidden.size(1), -1)
        hidden = torch.cat([hidden, channel_embedding], dim=-1)
        return self.output_head(hidden).squeeze(-1)


class MLPBaseline(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, channel_embed_dim: int, dropout: float):
        super().__init__()
        self.channel_embedding = nn.Embedding(input_channels, channel_embed_dim)
        self.network = nn.Sequential(
            nn.Linear(input_channels + channel_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, masked_channel_index: torch.Tensor) -> torch.Tensor:
        channel_embedding = self.channel_embedding(masked_channel_index).unsqueeze(1).expand(-1, x.size(1), -1)
        features = torch.cat([x, channel_embedding], dim=-1)
        return self.network(features).squeeze(-1)


class LinearBaseline(nn.Module):
    def __init__(self, input_channels: int, channel_embed_dim: int):
        super().__init__()
        self.channel_embedding = nn.Embedding(input_channels, channel_embed_dim)
        self.linear = nn.Linear(input_channels + channel_embed_dim, 1)

    def forward(self, x: torch.Tensor, masked_channel_index: torch.Tensor) -> torch.Tensor:
        channel_embedding = self.channel_embedding(masked_channel_index).unsqueeze(1).expand(-1, x.size(1), -1)
        features = torch.cat([x, channel_embedding], dim=-1)
        return self.linear(features).squeeze(-1)


def compute_batch_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((y_true - y_pred) ** 2).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    true_centered = y_true - y_true.mean(dim=1, keepdim=True)
    pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)
    numerator = (true_centered * pred_centered).sum(dim=1)
    denominator = torch.sqrt(
        (true_centered.square().sum(dim=1) * pred_centered.square().sum(dim=1)).clamp_min(1e-8)
    )
    correlation = (numerator / denominator).mean().item()
    return {"mse": mse, "mae": mae, "corr": correlation}


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        key: float(np.mean([metrics[key] for metrics in metric_list]))
        for key in ("mse", "mae", "corr")
    }


def evaluate_callable(
    prediction_fn,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    metric_list: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in data_loader:
            x_masked = batch["x_masked"].to(device)
            y_target = batch["y_target"].to(device)
            masked_channel_index = batch["masked_channel_index"].to(device)
            predictions = prediction_fn(x_masked, masked_channel_index)
            metric_list.append(compute_batch_metrics(y_target, predictions))
    return aggregate_metrics(metric_list)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    checkpoint_path: Path,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_mse": [], "val_mae": [], "val_corr": []}
    best_state = None
    best_metrics: Optional[Dict[str, float]] = None
    best_val = float("inf")
    epochs_without_improvement = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            x_masked = batch["x_masked"].to(device)
            y_target = batch["y_target"].to(device)
            masked_channel_index = batch["masked_channel_index"].to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(x_masked, masked_channel_index)
            loss = criterion(predictions, y_target)
            loss.backward()
            optimizer.step()

            batch_size = x_masked.size(0)
            running_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = running_loss / max(total_examples, 1)
        model.eval()
        val_metrics = evaluate_callable(lambda x, idx: model(x, idx), val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_corr"].append(val_metrics["corr"])

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.5f} | "
            f"val_mse={val_metrics['mse']:.5f} | val_mae={val_metrics['mae']:.5f} | "
            f"val_corr={val_metrics['corr']:.5f}"
        )

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, history, best_metrics


def collect_predictions(
    model_fn,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    xs, ys, preds, mask_indices = [], [], [], []
    batch_count = 0
    with torch.no_grad():
        for batch in data_loader:
            x_masked = batch["x_masked"].to(device)
            y_target = batch["y_target"].to(device)
            masked_channel_index = batch["masked_channel_index"].to(device)
            y_pred = model_fn(x_masked, masked_channel_index)
            xs.append(x_masked.cpu().numpy())
            ys.append(y_target.cpu().numpy())
            preds.append(y_pred.detach().cpu().numpy())
            mask_indices.append(masked_channel_index.cpu().numpy())
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break
    return {
        "x_masked": np.concatenate(xs, axis=0),
        "y_true": np.concatenate(ys, axis=0),
        "y_pred": np.concatenate(preds, axis=0),
        "masked_channel_index": np.concatenate(mask_indices, axis=0),
    }


def plot_training_history(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train MSE")
    axes[0].plot(epochs, history["val_mse"], label="Val MSE")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_mae"], label="Val MAE")
    axes[1].plot(epochs, history["val_corr"], label="Val Corr")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_example_reconstructions(
    example_bundle: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
    num_examples: int,
) -> None:
    model_names = list(example_bundle.keys())
    n_examples = min(num_examples, example_bundle[model_names[0]]["y_true"].shape[0])
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3.2 * n_examples), squeeze=False)

    for row_idx in range(n_examples):
        ax = axes[row_idx, 0]
        true_waveform = example_bundle[model_names[0]]["y_true"][row_idx]
        masked_channel = int(example_bundle[model_names[0]]["masked_channel_index"][row_idx])
        ax.plot(true_waveform, label="True", linewidth=2.0, color="black")
        for model_name in model_names:
            ax.plot(example_bundle[model_name]["y_pred"][row_idx], label=model_name, alpha=0.8)
        ax.set_title(f"Example {row_idx + 1} | masked channel={masked_channel}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Standardized amplitude")
        ax.legend(ncol=min(len(model_names) + 1, 4))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_phase_argument(phase_names: str) -> Optional[List[int]]:
    if phase_names.lower() == "all":
        return None
    phase_lookup = {name.lower(): idx for idx, name in enumerate(PHASE_NAMES)}
    selected = []
    for token in phase_names.split(","):
        token = token.strip().lower()
        if token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= len(PHASE_NAMES):
                raise ValueError(f"Invalid phase index: {idx}")
            selected.append(idx)
        elif token in phase_lookup:
            selected.append(phase_lookup[token])
        else:
            raise ValueError(f"Unknown phase token: {token}")
    return sorted(set(selected))


def parse_channel_argument(channel_spec: Optional[str]) -> Optional[List[int]]:
    if channel_spec is None:
        return None
    value = channel_spec.strip().lower()
    if value in {"", "all"}:
        return None
    channels = []
    for token in channel_spec.split(","):
        token = token.strip()
        if not token:
            continue
        channels.append(int(token))
    return sorted(set(channels))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Masked-channel reconstruction experiment for segmented LFP data.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Path to a reconstruction manifest built from separated class files.")
    parser.add_argument("--data-path", type=Path, default=None, help="Path to a .npy array shaped (n_trials, 3, channels, time).")
    parser.add_argument("--output-dir", type=Path, default=Path("GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/masked_channel_reconstruction"))
    parser.add_argument("--max-samples", type=int, default=1200, help="Maximum flattened phase samples after trial/phase expansion.")
    parser.add_argument("--max-channels", type=int, default=24, help="Maximum number of channels to keep.")
    parser.add_argument("--downsample", type=int, default=2, help="Keep every Nth time point.")
    parser.add_argument("--phases", type=str, default="all", help="Comma-separated phase names or indices, or 'all'.")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--channel-embed-dim", type=int, default=16)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--synthetic-trials", type=int, default=320)
    parser.add_argument("--synthetic-channels", type=int, default=24)
    parser.add_argument("--synthetic-time", type=int, default=300)
    parser.add_argument("--num-plot-examples", type=int, default=4)
    parser.add_argument("--allowed-mask-channels", type=str, default="all", help="Comma-separated channel indices eligible for masking, or 'all'.")
    parser.add_argument("--fixed-mask-channel", type=int, default=None, help="If set, always mask this exact channel index.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    device = infer_device()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_phases = parse_phase_argument(args.phases)
    allowed_mask_channels = parse_channel_argument(args.allowed_mask_channels)
    manifest_path = args.manifest_path or _default_manifest_path()

    raw_data = None
    if manifest_path is not None and manifest_path.exists():
        manifest = load_manifest(manifest_path)
        train_raw = load_samples_from_manifest(manifest, "train", phases=selected_phases)
        val_raw = load_samples_from_manifest(manifest, "val", phases=selected_phases)
        test_raw = load_samples_from_manifest(manifest, "test", phases=selected_phases)
        data_source = str(manifest_path)
    else:
        real_data_path = args.data_path or _default_real_data_path()
        real_data = _maybe_load_array(real_data_path) if real_data_path is not None else None

        if real_data is None:
            print("No manifest or real segmented LFP array found. Using synthetic demo data.")
            raw_data = generate_synthetic_lfp(
                n_trials=args.synthetic_trials,
                n_phases=3,
                n_channels=args.synthetic_channels,
                n_time=args.synthetic_time,
                seed=args.seed,
            )
            data_source = "synthetic"
        else:
            raw_data = real_data.astype(np.float32, copy=False)
            data_source = str(real_data_path)

        samples = flatten_trials_and_phases(raw_data, phases=selected_phases)
        samples = select_subset(
            data=samples,
            max_samples=args.max_samples,
            max_channels=args.max_channels,
            downsample=max(args.downsample, 1),
        )
        train_idx, val_idx, test_idx = split_indices(
            n_samples=samples.shape[0],
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        train_raw = samples[train_idx]
        val_raw = samples[val_idx]
        test_raw = samples[test_idx]

    train_raw = select_subset(
        data=train_raw,
        max_samples=args.max_samples,
        max_channels=args.max_channels,
        downsample=max(args.downsample, 1),
    )
    val_raw = select_subset(
        data=val_raw,
        max_samples=args.max_samples,
        max_channels=args.max_channels,
        downsample=max(args.downsample, 1),
    )
    test_raw = select_subset(
        data=test_raw,
        max_samples=args.max_samples,
        max_channels=args.max_channels,
        downsample=max(args.downsample, 1),
    )

    stats = compute_train_stats(train_raw)
    train_data = apply_standardization(train_raw, stats)
    val_data = apply_standardization(val_raw, stats)
    test_data = apply_standardization(test_raw, stats)

    train_dataset = MaskedChannelDataset(
        train_data,
        seed=args.seed,
        allowed_mask_channels=allowed_mask_channels,
        fixed_mask_channel=args.fixed_mask_channel,
    )
    val_dataset = MaskedChannelDataset(
        val_data,
        seed=args.seed + 1,
        allowed_mask_channels=allowed_mask_channels,
        fixed_mask_channel=args.fixed_mask_channel,
    )
    test_dataset = MaskedChannelDataset(
        test_data,
        seed=args.seed + 2,
        allowed_mask_channels=allowed_mask_channels,
        fixed_mask_channel=args.fixed_mask_channel,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    _, n_channels, n_time = train_data.shape
    print(
        f"Device: {device}\n"
        f"Data source: {data_source}\n"
        f"Train/val/test samples: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}\n"
        f"Per-sample shape after preprocessing: channels={n_channels}, time={n_time}\n"
        f"Maskable channels: {allowed_mask_channels if allowed_mask_channels is not None else 'all'}\n"
        f"Fixed mask channel: {args.fixed_mask_channel}"
    )

    transformer = TransformerReconstructor(
        input_channels=n_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        channel_embed_dim=args.channel_embed_dim,
    ).to(device)

    linear_baseline = LinearBaseline(
        input_channels=n_channels,
        channel_embed_dim=args.channel_embed_dim,
    ).to(device)

    mlp_baseline = MLPBaseline(
        input_channels=n_channels,
        hidden_dim=args.mlp_hidden_dim,
        channel_embed_dim=args.channel_embed_dim,
        dropout=args.dropout,
    ).to(device)

    print("\nTraining linear baseline...")
    linear_checkpoint = output_dir / "best_linear_baseline.pt"
    linear_baseline, linear_history, linear_best_val = train_model(
        model=linear_baseline,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=linear_checkpoint,
    )

    print("\nTraining MLP baseline...")
    mlp_checkpoint = output_dir / "best_mlp_baseline.pt"
    mlp_baseline, mlp_history, mlp_best_val = train_model(
        model=mlp_baseline,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=mlp_checkpoint,
    )

    print("\nTraining Transformer...")
    transformer_checkpoint = output_dir / "best_transformer.pt"
    transformer, transformer_history, transformer_best_val = train_model(
        model=transformer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=transformer_checkpoint,
    )

    print("\nEvaluating on test set...")
    linear_baseline.eval()
    mlp_baseline.eval()
    transformer.eval()
    results = {
        "linear_baseline": evaluate_callable(lambda x, idx: linear_baseline(x, idx), test_loader, device),
        "mlp_baseline": evaluate_callable(lambda x, idx: mlp_baseline(x, idx), test_loader, device),
        "transformer": evaluate_callable(lambda x, idx: transformer(x, idx), test_loader, device),
    }

    for model_name, metrics in results.items():
        print(
            f"{model_name:>16} | test_mse={metrics['mse']:.5f} | "
            f"test_mae={metrics['mae']:.5f} | test_corr={metrics['corr']:.5f}"
        )

    plot_training_history(linear_history, output_dir / "linear_training_history.png")
    plot_training_history(mlp_history, output_dir / "mlp_training_history.png")
    plot_training_history(transformer_history, output_dir / "transformer_training_history.png")

    example_bundle = {
        "Linear": collect_predictions(lambda x, idx: linear_baseline(x, idx), test_loader, device, max_batches=1),
        "MLP": collect_predictions(lambda x, idx: mlp_baseline(x, idx), test_loader, device, max_batches=1),
        "Transformer": collect_predictions(lambda x, idx: transformer(x, idx), test_loader, device, max_batches=1),
    }
    plot_example_reconstructions(
        example_bundle=example_bundle,
        output_path=output_dir / "example_reconstructions.png",
        num_examples=args.num_plot_examples,
    )

    config_summary = {
        "data_source": data_source,
        "raw_shape": list(raw_data.shape) if raw_data is not None else None,
        "selected_phases": selected_phases if selected_phases is not None else [0, 1, 2],
        "processed_shape": {
            "train": list(train_raw.shape),
            "val": list(val_raw.shape),
            "test": list(test_raw.shape),
        },
        "split_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "masking": {
            "allowed_mask_channels": allowed_mask_channels,
            "fixed_mask_channel": args.fixed_mask_channel,
        },
        "standardization": {
            "mean_shape": list(stats.mean.shape),
            "std_shape": list(stats.std.shape),
        },
        "best_val": {
            "linear_baseline": linear_best_val,
            "mlp_baseline": mlp_best_val,
            "transformer": transformer_best_val,
        },
        "test_results": results,
    }
    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as handle:
        json.dump(config_summary, handle, indent=2)


if __name__ == "__main__":
    main()
