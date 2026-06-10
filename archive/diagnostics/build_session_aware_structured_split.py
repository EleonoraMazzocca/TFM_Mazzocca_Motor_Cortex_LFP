import argparse
import itertools
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from preprocess_pipeline.data_paths import CLEANED_STRUCTURED_DIR
except ImportError:
    from .data_paths import CLEANED_STRUCTURED_DIR


DEFAULT_SESSIONS = [
    "20180531Y",
    "20180601Y",
    "20180606Y",
    "20180607Y",
    "20180608Y",
    "20180612Y",
    "20180613Y",
    "20180614Y",
    "20180615Y",
    "20180618Y",
    "20180619Y",
]

PHASE_NAMES = ["PREREACH", "REACH", "GRASP"]


def load_session_structured(session: str, tag: str) -> Tuple[np.ndarray, Dict[str, List[object]]]:
    """
    Load one session from the structured preprocessing output.

    data shape:
    (n_trials, 3, channels, time)
    """
    data_path = CLEANED_STRUCTURED_DIR / f"data_{session}{tag}.npy"
    info_path = CLEANED_STRUCTURED_DIR / f"info_{session}{tag}.pkl"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info file: {info_path}")

    data = np.load(data_path, mmap_mode="r")
    with open(info_path, "rb") as handle:
        info = pickle.load(handle)
    return data, info


def parse_trial_class_name(info: Dict[str, List[object]], trial_index: int) -> str:
    """
    Convert the stored metadata into the same human-readable class names
    used elsewhere in the repo.
    """
    precision_power = int(info["Precision/Power"][trial_index])
    uni_bi = int(info["Unimanual/Bimanual"][trial_index])
    left_angle = str(info["LeftAngle"][trial_index])
    right_angle = str(info["RightAngle"][trial_index])

    if precision_power == 0 and uni_bi == 1:
        if left_angle == "45" and right_angle == "45":
            return "PRECISION_BIMANUAL_45"
        if left_angle == "135" and right_angle == "135":
            return "PRECISION_BIMANUAL_135"
        if left_angle == "45" and right_angle == "135":
            return "PRECISION_BIMANUAL_45_135"
        if left_angle == "135" and right_angle == "45":
            return "PRECISION_BIMANUAL_135_45"
        return "UNKNOWN"

    prefix = "PRECISION" if precision_power == 0 else "POWER"
    if left_angle == "-1":
        return f"{prefix}_UNIMANUAL_R_{right_angle}"
    if right_angle == "-1":
        return f"{prefix}_UNIMANUAL_L_{left_angle}"
    return "UNKNOWN"


def build_entries_for_session(session: str, tag: str) -> List[Dict[str, object]]:
    """
    Build one entry per trial.

    We keep:
    - session id
    - trial index
    - class label
    - path to the structured data

    We do not split here.
    """
    data, info = load_session_structured(session=session, tag=tag)

    entries: List[Dict[str, object]] = []
    n_trials = data.shape[0]

    for trial_index in range(n_trials):
        entry = {
            "session": session,
            "trial_index": int(trial_index),
            "class_name": parse_trial_class_name(info, trial_index),
            "precision_power": int(info["Precision/Power"][trial_index]),
            "unimanual_bimanual": int(info["Unimanual/Bimanual"][trial_index]),
            "left_angle": str(info["LeftAngle"][trial_index]),
            "right_angle": str(info["RightAngle"][trial_index]),
            "data_path": str(CLEANED_STRUCTURED_DIR / f"data_{session}{tag}.npy"),
            "info_path": str(CLEANED_STRUCTURED_DIR / f"info_{session}{tag}.pkl"),
            "phases": PHASE_NAMES,
        }
        entries.append(entry)

    return entries


def entry_matches_holdout_combination(
    entry: Dict[str, object],
    holdout_right_angle: str | None,
    holdout_precision_power: str | None,
) -> bool:
    """
    Decide whether one entry belongs to the held-out compositional combination.

    For now we keep this rule intentionally simple:
    - if holdout_right_angle is set, any trial whose right hand angle matches
      that value belongs to the held-out pool
    - if holdout_precision_power is set, the trial must also match that label

    Example:
    --holdout-right-angle 135 --holdout-precision-power PRECISION
    """
    if holdout_right_angle is None and holdout_precision_power is None:
        return False

    angle_matches = True
    label_matches = True

    if holdout_right_angle is not None:
        angle_matches = str(entry["right_angle"]) == str(holdout_right_angle)

    if holdout_precision_power is not None:
        label_matches = str(entry["class_name"]).startswith(str(holdout_precision_power).upper())

    return angle_matches and label_matches


def count_classes(entries: List[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in entries:
        class_name = str(entry["class_name"])
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def split_entries_randomly(
    entries: List[Dict[str, object]],
    first_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Split one list into two parts.

    We use this only inside the held-out combination pool:
    - first part -> validation
    - second part -> test
    """
    rng = np.random.default_rng(seed)
    if not entries:
        return [], []

    shuffled_indices = rng.permutation(len(entries))
    first_size = int(round(len(entries) * first_fraction))

    first_split: List[Dict[str, object]] = []
    second_split: List[Dict[str, object]] = []

    for position, entry_index in enumerate(shuffled_indices):
        entry = entries[int(entry_index)]
        if position < first_size:
            first_split.append(entry)
        else:
            second_split.append(entry)

    return first_split, second_split


def build_manifest(
    sessions: List[str],
    tag: str,
    val_fraction: float,
    test_fraction: float,
    holdout_right_angle: str | None,
    holdout_precision_power: str | None,
    seed: int,
) -> Dict[str, object]:
    entries_by_session: Dict[str, List[Dict[str, object]]] = {}
    for session in sessions:
        entries_by_session[session] = build_entries_for_session(session=session, tag=tag)

    all_entries: List[Dict[str, object]] = []
    for session in sessions:
        all_entries.extend(entries_by_session[session])

    train_entries: List[Dict[str, object]] = []
    holdout_entries: List[Dict[str, object]] = []

    for entry in all_entries:
        if entry_matches_holdout_combination(
            entry=entry,
            holdout_right_angle=holdout_right_angle,
            holdout_precision_power=holdout_precision_power,
        ):
            holdout_entries.append(entry)
        else:
            train_entries.append(entry)

    if not holdout_entries:
        raise ValueError(
            "No entries matched the held-out combination. "
            "Try changing --holdout-right-angle or inspect the dataset labels first."
        )

    holdout_total = val_fraction + test_fraction
    if holdout_total <= 0:
        raise ValueError("val_fraction + test_fraction must be > 0 for the held-out combination split.")
    val_share_inside_holdout = val_fraction / holdout_total
    val_entries, test_entries = split_entries_randomly(
        entries=holdout_entries,
        first_fraction=val_share_inside_holdout,
        seed=seed,
    )

    manifest = {
        "version": 1,
        "source": "structured_session_files",
        "split_type": "compositional_holdout",
        "tag": tag,
        "validation_holdout_rule": {
            "holdout_right_angle": holdout_right_angle,
            "holdout_precision_power": holdout_precision_power,
        },
        "fractions": {
            "train": None,
            "val": val_fraction,
            "test": test_fraction,
        },
        "sessions_used": sorted(sessions),
        "splits": {
            "train": train_entries,
            "val": val_entries,
            "test": test_entries,
        },
        "counts": {},
    }

    manifest["counts"]["train"] = {
        "n_trials": len(train_entries),
        "sessions": sorted({entry["session"] for entry in train_entries}),
        "class_counts": count_classes(train_entries),
    }
    manifest["counts"]["val"] = {
        "n_trials": len(val_entries),
        "sessions": sorted({entry["session"] for entry in val_entries}),
        "class_counts": count_classes(val_entries),
    }
    manifest["counts"]["test"] = {
        "n_trials": len(test_entries),
        "sessions": sorted({entry["session"] for entry in test_entries}),
        "class_counts": count_classes(test_entries),
    }

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compositional holdout split for the full structured dataset."
    )
    parser.add_argument("--sessions", nargs="*", default=DEFAULT_SESSIONS)
    parser.add_argument("--tag", type=str, default="", help="Optional structured-data tag, for example _mua_200_500")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.8)
    parser.add_argument(
        "--holdout-right-angle",
        type=str,
        default="135",
        help="Held-out compositional rule. Example: 135 means all trials with right_angle=135 are excluded from train and split into val/test.",
    )
    parser.add_argument(
        "--holdout-precision-power",
        type=str,
        default="PRECISION",
        choices=["PRECISION", "POWER"],
        help="Optional extra filter for the held-out combination.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/session_aware_structured_split.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_fraction = args.val_fraction + args.test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError("val_fraction + test_fraction must sum to 1 for the held-out combination")

    manifest = build_manifest(
        sessions=args.sessions,
        tag=args.tag,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        holdout_right_angle=args.holdout_right_angle,
        holdout_precision_power=args.holdout_precision_power,
        seed=args.seed,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Saved compositional split to: {args.output_path}")
    print(
        "Holdout rule: "
        f"right_angle == {args.holdout_right_angle!r} "
        f"and label starts with {args.holdout_precision_power!r}"
    )
    print(f"All sessions used: {manifest['sessions_used']}")
    print("Counts:")
    for split_name, info in manifest["counts"].items():
        print(f"  {split_name}: n_trials={info['n_trials']} | sessions={info['sessions']}")


if __name__ == "__main__":
    main()
