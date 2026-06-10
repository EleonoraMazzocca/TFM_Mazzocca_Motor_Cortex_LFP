"""Build session-to-task mapping tables from structured preprocessed data."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from data_paths import CLEANED_META_DIR, CLEANED_STRUCTURED_DIR
except ImportError:
    from .data_paths import CLEANED_META_DIR, CLEANED_STRUCTURED_DIR


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


def task_label(info: dict[str, list], index: int) -> str:
    grip = "precision" if info["Precision/Power"][index] == 0 else "power"
    manuality = "bimanual" if info["Unimanual/Bimanual"][index] == 1 else "unimanual"
    left_angle = info["LeftAngle"][index]
    right_angle = info["RightAngle"][index]

    if manuality == "bimanual":
        return f"{grip}_{manuality}_{left_angle}_{right_angle}"
    if str(left_angle) == "-1":
        return f"{grip}_{manuality}_right_{right_angle}"
    return f"{grip}_{manuality}_left_{left_angle}"


def build_rows(sessions: list[str], tag: str) -> tuple[list[dict[str, object]], list[str]]:
    rows = []
    all_tasks = set()

    for session in sessions:
        info_path = CLEANED_STRUCTURED_DIR / f"info_{session}{tag}.pkl"
        data_path = CLEANED_STRUCTURED_DIR / f"data_{session}{tag}.npy"
        if not info_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"Missing structured files for {session} with tag {tag!r}")

        with open(info_path, "rb") as fp:
            info = pickle.load(fp)

        data = np.load(data_path, mmap_mode="r")
        raw_trials = None
        motorno_path = CLEANED_META_DIR / f"motorno_{session}.csv"
        if motorno_path.exists():
            with open(motorno_path, newline="") as fp:
                raw_trials = len(next(csv.reader(fp)))

        counts = Counter(task_label(info, i) for i in range(data.shape[0]))
        all_tasks.update(counts)
        rows.append(
            {
                "session": session,
                "raw_parameter_trials": raw_trials,
                "accepted_trials": int(data.shape[0]),
                "discarded_trials": None if raw_trials is None else raw_trials - int(data.shape[0]),
                "shape": "x".join(str(x) for x in data.shape),
                "task_counts": counts,
            }
        )

    return rows, sorted(all_tasks)


def write_summary(rows: list[dict[str, object]], tasks: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "session",
                "raw_parameter_trials",
                "accepted_trials",
                "discarded_trials",
                "shape",
                "n_tasks",
                *tasks,
            ]
        )
        for row in rows:
            counts = row["task_counts"]
            writer.writerow(
                [
                    row["session"],
                    row["raw_parameter_trials"],
                    row["accepted_trials"],
                    row["discarded_trials"],
                    row["shape"],
                    sum(1 for task in tasks if counts.get(task, 0)),
                    *(counts.get(task, 0) for task in tasks),
                ]
            )


def write_long(rows: list[dict[str, object]], tasks: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["session", "task", "accepted_trials"])
        for row in rows:
            counts = row["task_counts"]
            for task in tasks:
                count = counts.get(task, 0)
                if count:
                    writer.writerow([row["session"], task, count])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="Structured filename tag, e.g. _mua_200_500 or __no_bad_channels.")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/trial_task_mapping"))
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    args = parser.parse_args()

    rows, tasks = build_rows(args.sessions, args.tag)
    suffix = args.tag if args.tag else "_default"
    write_summary(rows, tasks, args.output_dir / f"trial_task_mapping{suffix}.csv")
    write_long(rows, tasks, args.output_dir / f"trial_task_mapping_long{suffix}.csv")

    print(f"Wrote {len(rows)} sessions and {len(tasks)} tasks to {args.output_dir}")
    print(f"Total accepted trials: {sum(row['accepted_trials'] for row in rows)}")


if __name__ == "__main__":
    main()
