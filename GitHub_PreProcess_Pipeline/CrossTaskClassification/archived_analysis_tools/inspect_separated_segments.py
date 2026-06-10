import argparse
from pathlib import Path
from typing import Dict, Iterable, List
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from data_paths import CLASS_FILE_NAMES, SEPARATED_CLASSES_DIR, with_class_tag
except ImportError:
    from GitHub_PreProcess_Pipeline.CrossTaskClassification.data_paths import (
        CLASS_FILE_NAMES,
        SEPARATED_CLASSES_DIR,
        with_class_tag,
    )


PHASE_NAMES = ("PREREACH", "REACH", "GRASP")


def resolve_paths(class_names: List[str] | None, limit: int | None) -> List[Path]:
    if class_names:
        paths = []
        for class_name in class_names:
            if class_name in CLASS_FILE_NAMES:
                paths.append(SEPARATED_CLASSES_DIR / with_class_tag(CLASS_FILE_NAMES[class_name]))
            else:
                path = Path(class_name)
                paths.append(path if path.is_absolute() else SEPARATED_CLASSES_DIR / path)
    else:
        paths = sorted(SEPARATED_CLASSES_DIR.glob("*.npy"))

    if limit is not None:
        paths = paths[:limit]
    return paths


def zero_channel_mask(array: np.ndarray) -> np.ndarray:
    return np.all(array == 0, axis=(0, 1, 3))


def phase_zero_channel_mask(array: np.ndarray) -> np.ndarray:
    return np.all(array == 0, axis=(0, 3))


def summarize_array(path: Path) -> Dict[str, object]:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 4:
        raise ValueError(f"Expected shape (N, 3, channels, time), got {array.shape} for {path}")

    n_trials, n_phases, n_channels, n_time = array.shape
    all_zero_channels = zero_channel_mask(array)
    phase_zero = phase_zero_channel_mask(array)

    summary = {
        "path": path,
        "shape": array.shape,
        "dtype": str(array.dtype),
        "n_trials": int(n_trials),
        "n_phases": int(n_phases),
        "n_channels": int(n_channels),
        "n_time": int(n_time),
        "all_zero_channel_count": int(all_zero_channels.sum()),
        "all_zero_channel_indices": np.flatnonzero(all_zero_channels).tolist(),
        "phase_zero_counts": {
            PHASE_NAMES[phase_idx] if phase_idx < len(PHASE_NAMES) else str(phase_idx): int(phase_zero[phase_idx].sum())
            for phase_idx in range(n_phases)
        },
        "value_min": int(array.min()),
        "value_max": int(array.max()),
        "mean_abs": float(np.mean(np.abs(array.astype(np.float32)))),
    }
    return summary


def print_summary(summary: Dict[str, object], show_indices: bool) -> None:
    print("=" * 80)
    print(f"File: {Path(summary['path']).name}")
    print(f"Shape: {summary['shape']} | dtype: {summary['dtype']}")
    print(
        f"Trials={summary['n_trials']} | phases={summary['n_phases']} | "
        f"channels={summary['n_channels']} | time={summary['n_time']}"
    )
    print(
        f"Value range: [{summary['value_min']}, {summary['value_max']}] | "
        f"mean(|x|)={summary['mean_abs']:.3f}"
    )
    print(f"All-zero channels across all trials/phases: {summary['all_zero_channel_count']}")
    print("All-zero channels per phase:")
    for phase_name, count in summary["phase_zero_counts"].items():
        print(f"  {phase_name}: {count}")
    if show_indices:
        print(f"All-zero channel indices: {summary['all_zero_channel_indices']}")


def aggregate_zero_channel_histories(summaries: Iterable[Dict[str, object]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for summary in summaries:
        for channel_idx in summary["all_zero_channel_indices"]:
            counts[channel_idx] = counts.get(channel_idx, 0) + 1
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect separated segment arrays shaped (N, 3, 256, 500).")
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional class names from data_paths.py or explicit .npy filenames inside Separated_Data/classes.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Inspect only the first N files.")
    parser.add_argument("--show-indices", action="store_true", help="Print the indices of all-zero channels.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = resolve_paths(args.class_names, args.limit)
    if not paths:
        raise ValueError(f"No .npy files found in {SEPARATED_CLASSES_DIR}")

    summaries = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        summary = summarize_array(path)
        summaries.append(summary)
        print_summary(summary, show_indices=args.show_indices)

    print("=" * 80)
    print(f"Inspected {len(summaries)} files.")
    channel_hist = aggregate_zero_channel_histories(summaries)
    if channel_hist:
        most_common = sorted(channel_hist.items(), key=lambda item: (-item[1], item[0]))[:15]
        print("Most frequently all-zero channels across inspected files:")
        for channel_idx, count in most_common:
            print(f"  channel {channel_idx}: {count} files")
    else:
        print("No channels were all-zero across an entire inspected file.")


if __name__ == "__main__":
    main()
