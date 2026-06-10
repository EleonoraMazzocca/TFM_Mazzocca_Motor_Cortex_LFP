"""Standardize raw MATLAB LFP files into NumPy arrays.

This is the first preprocessing step. It extracts LFP1 and LFP2 from each raw
``lfp_data_<session>.mat`` file and writes small metadata needed by later steps.
"""
from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import mat73
import numpy as np

try:
    from preprocess_pipeline.data_paths import CLEANED_DATA_DIR, RAW_DATA_DIR
except ImportError:
    from .data_paths import CLEANED_DATA_DIR, RAW_DATA_DIR

BASE_FS = 4.8828125 * 10**3


def discover_sessions() -> list[str]:
    """Return session IDs available as raw LFP MATLAB files."""
    return sorted(
        path.stem.replace("lfp_data_", "")
        for path in RAW_DATA_DIR.glob("lfp_data_*.mat")
    )


def standardize_session(session: str, overwrite: bool = False) -> None:
    """Extract LFP blocks and metadata for one session."""
    raw_path = RAW_DATA_DIR / f"lfp_data_{session}.mat"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw file: {raw_path}")

    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_dir = CLEANED_DATA_DIR / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    lfp1_out = CLEANED_DATA_DIR / f"lfp1_data_{session}.npy"
    lfp2_out = CLEANED_DATA_DIR / f"lfp2_data_{session}.npy"
    meta_out = meta_dir / f"meta_{session}.pkl"

    if not overwrite and lfp1_out.exists() and lfp2_out.exists() and meta_out.exists():
        print(f"[skip] {session}: standardized outputs already exist")
        return

    print(f"[{session}] Loading LFP1 from {raw_path}")
    data_lfp1 = mat73.loadmat(raw_path, only_include=["lfp_data2/streams/LFP1"])
    lfp1 = data_lfp1["lfp_data2"]["streams"]["LFP1"]

    lfp1_fs = float(lfp1["fs"])
    lfp1_ratio = float(BASE_FS / lfp1_fs) if lfp1_fs != BASE_FS else 1.0
    meta = {"ratio": lfp1_ratio, "fs": lfp1_fs}

    np.save(lfp1_out, lfp1["data"])
    print(f"[{session}] Wrote {lfp1_out} shape={lfp1['data'].shape}")

    del data_lfp1, lfp1
    gc.collect()

    print(f"[{session}] Loading LFP2 from {raw_path}")
    data_lfp2 = mat73.loadmat(raw_path, only_include=["lfp_data2/streams/LFP2"])
    lfp2 = data_lfp2["lfp_data2"]["streams"]["LFP2"]

    np.save(lfp2_out, lfp2["data"])
    print(f"[{session}] Wrote {lfp2_out} shape={lfp2['data'].shape}")

    del data_lfp2, lfp2
    gc.collect()

    with meta_out.open("wb") as fp:
        pickle.dump(meta, fp)
    print(f"[{session}] Wrote {meta_out} metadata={meta}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract LFP1/LFP2 NumPy arrays and sampling metadata from raw MATLAB sessions."
    )
    parser.add_argument(
        "sessions",
        nargs="*",
        help="Session IDs such as 20180619Y. If omitted, all lfp_data_*.mat files are processed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate outputs even if lfp1/lfp2/meta files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = args.sessions or discover_sessions()
    if not sessions:
        raise SystemExit(f"No raw files found in {RAW_DATA_DIR}")

    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Cleaned data directory: {CLEANED_DATA_DIR}")
    print(f"Sessions to standardize ({len(sessions)}): {sessions}")

    for session in sessions:
        standardize_session(session, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
