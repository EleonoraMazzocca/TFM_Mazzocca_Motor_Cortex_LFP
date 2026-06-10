import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

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


def resolve_class_paths(
    class_names: Sequence[str] | None,
    include_tagged_variants: bool,
) -> Dict[str, Path]:
    if class_names:
        resolved = {}
        for class_name in class_names:
            if class_name in CLASS_FILE_NAMES:
                resolved[class_name] = SEPARATED_CLASSES_DIR / with_class_tag(CLASS_FILE_NAMES[class_name])
            else:
                path = Path(class_name)
                label = path.stem.upper()
                resolved[label] = path if path.is_absolute() else SEPARATED_CLASSES_DIR / path
        return resolved

    suffix = ".npy"
    paths = sorted(SEPARATED_CLASSES_DIR.glob(f"*{suffix}"))
    resolved = {}
    for path in paths:
        stem = path.stem
        if not include_tagged_variants and "_mua_" in stem:
            continue
        resolved[stem.upper()] = path
    return resolved


def split_trial_indices(
    n_trials: int,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_trials)
    train_end = int(round(n_trials * train_frac))
    val_end = train_end + int(round(n_trials * val_frac))
    train_idx = indices[:train_end].tolist()
    val_idx = indices[train_end:val_end].tolist()
    test_idx = indices[val_end:].tolist()
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError(
            f"Invalid split for n_trials={n_trials}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_manifest(
    class_paths: Dict[str, Path],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, object]:
    manifest = {
        "version": 1,
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "classes": {},
        "splits": {"train": [], "val": [], "test": []},
    }

    for offset, (class_name, path) in enumerate(sorted(class_paths.items())):
        if not path.exists():
            raise FileNotFoundError(f"Missing class file: {path}")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 4:
            raise ValueError(f"Expected shape (n_trials, 3, channels, time), got {array.shape} from {path}")
        n_trials, n_phases, n_channels, n_time = array.shape
        split_indices = split_trial_indices(
            n_trials=n_trials,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed + offset,
        )
        manifest["classes"][class_name] = {
            "path": str(path),
            "shape": [int(n_trials), int(n_phases), int(n_channels), int(n_time)],
        }
        for split_name, trial_indices in split_indices.items():
            for trial_index in trial_indices:
                manifest["splits"][split_name].append(
                    {
                        "class_name": class_name,
                        "path": str(path),
                        "trial_index": int(trial_index),
                    }
                )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reproducible train/val/test manifest for channel-reconstruction experiments.")
    parser.add_argument("--class-names", nargs="*", default=None, help="Optional class names from data_paths.py or explicit filenames.")
    parser.add_argument("--include-tagged-variants", action="store_true", help="Include tagged files like *_mua_200_500.npy when auto-discovering.")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/reconstruction_manifest.json"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.train_frac <= 0 or args.val_frac <= 0 or args.train_frac + args.val_frac >= 1:
        raise ValueError("Expected train_frac > 0, val_frac > 0, and train_frac + val_frac < 1")

    class_paths = resolve_class_paths(
        class_names=args.class_names,
        include_tagged_variants=args.include_tagged_variants,
    )
    if not class_paths:
        raise ValueError(f"No class files found in {SEPARATED_CLASSES_DIR}")

    manifest = build_manifest(
        class_paths=class_paths,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Saved manifest to {args.output_path}")
    for split_name, entries in manifest["splits"].items():
        print(f"{split_name}: {len(entries)} trials")


if __name__ == "__main__":
    main()
