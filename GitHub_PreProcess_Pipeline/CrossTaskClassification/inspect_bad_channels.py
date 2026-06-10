import argparse
import csv
import datetime
import pickle

import numpy as np
import scipy.signal as spsg

try:
    from data_paths import CLEANED_DATA_DIR, CLEANED_META_DIR
except ImportError:
    from .data_paths import CLEANED_DATA_DIR, CLEANED_META_DIR

name_file = "20180613Y"
block_to_inspect = "lfp2"


def as_scalar(value, name):
    array_value = np.asarray(value)
    if array_value.size != 1:
        raise ValueError(f"{name} must be a single scalar value, got shape {array_value.shape}")
    return array_value.reshape(-1)[0].item()


def bandpass_filtering(lfp, frq, lp=1.0, hp=500.0):
    nyquist = frq / 2.0
    b, a = spsg.iirfilter(3, [lp / nyquist, hp / nyquist], btype="bandpass", ftype="butter")
    for i in range(lfp.shape[0]):
        print(f"Bandpass channel {i} at {datetime.datetime.now()}")
        lfp[i] = spsg.filtfilt(b, a, lfp[i]).astype(lfp.dtype, copy=False)
    return lfp


def notch_filtering(lfp, frq, base_freq=60.0, quality=30.0):
    nyquist = frq / 2.0
    harmonics = np.arange(base_freq, nyquist, base_freq)
    for i in range(lfp.shape[0]):
        print(f"Notch channel {i} at {datetime.datetime.now()}")
        channel = lfp[i].astype(np.float32, copy=True)
        for harmonic in harmonics:
            b, a = spsg.iirnotch(harmonic, quality, fs=frq)
            channel = spsg.filtfilt(b, a, channel).astype(np.float32, copy=False)
        lfp[i] = np.clip(channel, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
    return lfp


def channelwise_std(lfp):
    std_values = np.empty(lfp.shape[0], dtype=np.float32)
    for ch in range(lfp.shape[0]):
        std_values[ch] = np.std(lfp[ch].astype(np.float32, copy=False), dtype=np.float32)
    return std_values


def std_bad_channels_with_reasons(lfp):
    all_channels = np.arange(lfp.shape[0])
    remaining_channels = all_channels.copy()
    std_all = channelwise_std(lfp)
    rejection_reasons = {int(ch): [] for ch in all_channels}
    sd_pk = np.inf
    iteration = 0

    while sd_pk > 5:
        sd_k = std_all[remaining_channels]
        m_k = np.median(sd_k)
        third_quartile = np.percentile(sd_k, 75)
        temp = np.std(sd_k)
        if sd_pk == temp:
            break
        sd_pk = temp

        bad_channels_k = []
        for ch in remaining_channels:
            sd_jk = float(std_all[ch])
            reasons = []
            if sd_jk < 10e-1:
                reasons.append(f"std<{10e-1}")
            if sd_jk > 100:
                reasons.append("std>100")
            if abs(sd_jk - m_k) > third_quartile:
                reasons.append(f"|std-median|>{third_quartile:.4f}")
            if reasons:
                bad_channels_k.append(ch)
                rejection_reasons[int(ch)].append(
                    {
                        "iteration": iteration,
                        "std": sd_jk,
                        "median": float(m_k),
                        "third_quartile": float(third_quartile),
                        "reasons": reasons,
                    }
                )
        remaining_channels = np.setdiff1d(remaining_channels, bad_channels_k)
        iteration += 1

    bad_channels = np.setdiff1d(all_channels, remaining_channels)
    return std_all, remaining_channels, bad_channels, rejection_reasons


def summarize_motor_classes(name_file):
    with open(CLEANED_META_DIR / f"motorno_{name_file}.csv", newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        motor_no = np.array(next(spamreader))
    unique_motor_no, motor_counts = np.unique(motor_no, return_counts=True)
    return dict(zip(unique_motor_no.tolist(), motor_counts.tolist()))


def inspect_block(name_file, block_name):
    data_path = CLEANED_DATA_DIR / f"{block_name}_data_{name_file}.npy"
    meta_path = CLEANED_META_DIR / f"meta_{name_file}.pkl"

    print("=" * 80)
    print(f"Inspecting {name_file} {block_name} from {data_path}")
    data = np.load(data_path)
    with open(meta_path, "rb") as fp:
        meta = pickle.load(fp)

    freq = float(as_scalar(meta["fs"], "fs"))
    ratio = as_scalar(meta["ratio"], "ratio")
    print(f"Loaded shape={data.shape}, dtype={data.dtype}, fs={freq}, ratio={ratio}")
    print(f"Motor classes: {summarize_motor_classes(name_file)}")

    data = bandpass_filtering(data, freq)
    data = notch_filtering(data, freq)
    std_all, remaining_channels, bad_channels, rejection_reasons = std_bad_channels_with_reasons(data)

    print("-" * 80)
    print(f"Remaining channels ({len(remaining_channels)}): {remaining_channels.tolist()}")
    print(f"Bad channels ({len(bad_channels)}): {bad_channels.tolist()}")
    print("Std summary:")
    print(f"  min={float(np.min(std_all)):.4f}")
    print(f"  median={float(np.median(std_all)):.4f}")
    print(f"  max={float(np.max(std_all)):.4f}")
    print("-" * 80)
    print("Per-channel details:")
    for ch, std_value in enumerate(std_all):
        details = rejection_reasons[ch]
        if details:
            latest = details[-1]
            reason_text = ", ".join(latest["reasons"])
            print(
                f"  ch {ch:3d}: std={float(std_value):9.4f} -> BAD "
                f"(iter={latest['iteration']}, median={latest['median']:.4f}, "
                f"q75={latest['third_quartile']:.4f}, reasons={reason_text})"
            )
        else:
            print(f"  ch {ch:3d}: std={float(std_value):9.4f} -> kept")


def main():
    parser = argparse.ArgumentParser(description="Inspect bad-channel rejection for one session and block.")
    parser.add_argument("name_file", nargs="?", default=name_file, help="Session name, for example 20180615Y")
    parser.add_argument(
        "--block",
        choices=["lfp1", "lfp2", "both"],
        default=block_to_inspect,
        help="Which block to inspect.",
    )
    args = parser.parse_args()

    blocks = ["lfp1", "lfp2"] if args.block == "both" else [args.block]
    for block_name in blocks:
        inspect_block(args.name_file, block_name)


name_file = "20180613Y"

if __name__ == "__main__":
    main()
