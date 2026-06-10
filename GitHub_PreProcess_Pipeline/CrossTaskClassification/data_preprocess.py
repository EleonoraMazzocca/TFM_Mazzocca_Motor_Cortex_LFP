import sys
import os
import builtins
import numpy as np
import pickle
import scipy.io as sio
import csv
import scipy.signal as spsg
import datetime
import traceback
import faulthandler
import gc

try:
    from meegkit.dss import dss_line
except ImportError:
    dss_line = None

try:
    from data_paths import CLEANED_DATA_DIR, CLEANED_META_DIR, PARAMETERS_DIR
except ImportError:
    from .data_paths import CLEANED_DATA_DIR, CLEANED_META_DIR, PARAMETERS_DIR

name_file = os.getenv("TFM_PREPROCESS_SESSION", "20180531Y")
#"20180531Y", --> done
#"20180601Y", --> done
#"20180606Y", --> done
#"20180607Y", --> done
#"20180608Y", --> done
#"20180612Y", --> done
#"20180613Y", --> done
#"20180614Y", --> done
#"20180615Y", --> done
#"20180618Y", --> done
#"20180619Y", --> done

LINE_NOISE_METHOD = os.getenv("TFM_LINE_NOISE_METHOD", "notch")
OUTPUT_TAG = os.getenv("TFM_PREPROCESS_OUTPUT_TAG", "")
SKIP_BAD_CHANNEL_REJECTION = os.getenv("TFM_SKIP_BAD_CHANNEL_REJECTION", "0").strip().lower() in {"1", "true", "yes", "y"}
BANDPASS_LOW_HZ = float(os.getenv("TFM_BANDPASS_LOW_HZ", "1.0"))
BANDPASS_HIGH_HZ = float(os.getenv("TFM_BANDPASS_HIGH_HZ", "500.0"))


LOG_SUFFIX = OUTPUT_TAG if OUTPUT_TAG else ""
LOG_PATH = os.path.abspath(f"data_preprocess_{name_file}{LOG_SUFFIX}.log")
LOG_FILE = open(LOG_PATH, "w", buffering=1)
faulthandler.enable(LOG_FILE, all_threads=True)

def tee_print(*args, **kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("flush", True)
    builtins.print(*args, **kwargs)
    file_kwargs = dict(kwargs)
    file_kwargs["file"] = LOG_FILE
    builtins.print(*args, **file_kwargs)

print = tee_print

def log_exception(exc_type, exc_value, exc_tb):
    print("\nUnhandled exception. Full traceback written below:")
    traceback.print_exception(exc_type, exc_value, exc_tb, file=LOG_FILE)
    LOG_FILE.flush()
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = log_exception

def as_scalar(value, name):
    array_value = np.asarray(value)
    if array_value.size != 1:
        raise ValueError(f"{name} must be a single scalar value, got shape {array_value.shape}")
    return array_value.reshape(-1)[0].item()

print("=" * 80)
print("Script version: notch-safe-v2")
print(f"Starting preprocessing at {datetime.datetime.now().isoformat()}")
print(f"Log file: {LOG_PATH}")
print(f"Python executable: {sys.executable}")
print(f"Session: {name_file}")
print(f"Output tag: {OUTPUT_TAG!r}")
print(f"Skip bad-channel rejection: {SKIP_BAD_CHANNEL_REJECTION}")
print(f"Bandpass range: {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz")

# Load data
lfp1_data = np.load(CLEANED_DATA_DIR / f"lfp1_data_{name_file}.npy")
lfp2_data = np.load(CLEANED_DATA_DIR / f"lfp2_data_{name_file}.npy")
print(f"Loaded lfp1_data with shape {lfp1_data.shape} and dtype {lfp1_data.dtype}")
print(f"Loaded lfp2_data with shape {lfp2_data.shape} and dtype {lfp2_data.dtype}")

# Load metadata
with open(CLEANED_META_DIR / f"meta_{name_file}.pkl", 'rb') as fp:
    meta = pickle.load(fp)
ratio = as_scalar(meta['ratio'], "ratio")
freq = float(as_scalar(meta['fs'], "fs"))
print(f"Loaded scalar metadata: ratio={ratio}, fs={freq}")

# -----------------------------------------------------------------------------
# This is the event data, you may change this to change the classification task
params = sio.loadmat(PARAMETERS_DIR / f"Params_{name_file}.mat")["TrialsParameters"]

motor_no = None
angle = None

# row 1 - MotorNo
#   MotorNo == "3-4": it is a bimanual precision, and you can get the corresponding hand orientations from the "Angle" variable.
#   MotorNo == "3": it is a unimanual precision with the lefthand.
#   MotorNo == "4": it is a unimanual precision with the right hand.
#   MotorNo == "1": it is a unimanual power with the left hand.
#   MotorNo == "2": it is a unimanual power with the right hand. 
# row 2 - Angle

with open(CLEANED_META_DIR / f"motorno_{name_file}.csv", newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    motor_no = np.array(next(spamreader))
    angle = np.array(next(spamreader))

dc_info = {"Precision/Power":[],"Unimanual/Bimanual":[],"LeftAngle":[],"RightAngle":[]}
unique_motor_no, motor_counts = np.unique(motor_no, return_counts=True)
print(f"Motor classes: {dict(zip(unique_motor_no.tolist(), motor_counts.tolist()))}")
for i, motor in enumerate(motor_no):
    precision_power = -1
    uni_bi = -1
    left_angle = -1
    right_angle = -1
    if motor == "3-4":
        precision_power = 0
        uni_bi = 1
        left_angle = angle[i].split()[0]
        right_angle = angle[i].split()[2]
    elif motor == "1":
        precision_power = 1
        uni_bi = 0
        left_angle = angle[i]
    elif motor == "2":
        precision_power = 1
        uni_bi = 0
        right_angle = angle[i]
    elif motor == "3":
        precision_power = 0
        uni_bi = 0
        left_angle = angle[i]
    elif motor == "4":
        precision_power = 0
        uni_bi = 0
        right_angle = angle[i]
    else:
        print(motor)
    dc_info["Precision/Power"].append(precision_power)
    dc_info["Unimanual/Bimanual"].append(uni_bi)
    dc_info["LeftAngle"].append(left_angle)
    dc_info["RightAngle"].append(right_angle)
print(
    "Metadata summary: "
    f"{len(dc_info['Precision/Power'])} trials, "
    f"unique LeftAngle values={sorted({str(x) for x in dc_info['LeftAngle']})[:10]}, "
    f"unique RightAngle values={sorted({str(x) for x in dc_info['RightAngle']})[:10]}"
)



#rh_indexes = np.where(rh_indexes == '4')
CueOn = params['TimeCueOn_samples'][0][0][0]
CueOff = params['TimeCueOff_samples'][0][0][0]
GraspStart = params['TimeGraspStart_samples'][0][0][0]
GraspEnd = params['TimeGraspEnd_samples'][0][0][0]
TrialEnd = params['TimeReward_samples'][0][0][0]
# -----------------------------------------------------------------------------
epoch_window = 500
total_channels = lfp1_data.shape[0] + lfp2_data.shape[0]
structured_data = np.zeros((len(CueOn), 3, total_channels, epoch_window), dtype=np.int16)
basefs = 4.8828125*10**3
discarded = set()

brain_areas = {"PMvR": (96, 96), "M1": (128, 128), "PMdR": (224, 224), "PMdL": (256, 256)}
back_mapping = [x for x in range(256)]
"""
---------------------------------------------------------------
PREPROCESSING
---------------------------------------------------------------
"""

############################################################
#                         Bandpass                         #
############################################################
#%%
def bandpass_filtering(lfp, frq, lp=1., hp=500.):
    nyquist = frq / 2.0
    if not (0 < lp < hp < nyquist):
        raise ValueError(
            f"Invalid bandpass range: lp={lp}, hp={hp}, nyquist={nyquist}. "
            "Expected 0 < lp < hp < nyquist."
        )
    b, a = spsg.iirfilter(3, [lp / nyquist, hp / nyquist], btype='bandpass', ftype='butter')
    for i, d in enumerate(lfp):
        print(i, end=" ")
        lfp[i] = spsg.filtfilt(b, a, d).astype(lfp.dtype, copy=False)

#%%
print("Bandpassing...")
print("\tProcessing LFP1 data...")
bandpass_filtering(lfp1_data, freq, lp=BANDPASS_LOW_HZ, hp=BANDPASS_HIGH_HZ)
#%%
print("\n\tProcessing LFP2 data...")
bandpass_filtering(lfp2_data, freq, lp=BANDPASS_LOW_HZ, hp=BANDPASS_HIGH_HZ)

#%%
############################################################
#                        ZapFilter                         #
############################################################
def safe_blocksize(signal_length, preferred=8192):
    if signal_length < 2:
        raise ValueError("Signal is too short for ZAP filtering.")
    return min(preferred, signal_length)

def zap_filtering_dss(lfp, frq, channels_per_batch=1, preferred_blocksize=2048):
    if dss_line is None:
        raise ImportError(
            "DSS line-noise removal requires meegkit. "
            "Install it or use TFM_LINE_NOISE_METHOD=notch."
        )
    filtered = np.empty_like(lfp, dtype=np.int16)
    blocksize = safe_blocksize(lfp.shape[1], preferred=preferred_blocksize)
    print(
        f"\tStarting DSS ZAP: channels={lfp.shape[0]}, samples={lfp.shape[1]}, "
        f"channels_per_batch={channels_per_batch}, blocksize={blocksize}, sfreq={frq}"
    )

    for batch_start in range(0, lfp.shape[0], channels_per_batch):
        batch_end = min(batch_start + channels_per_batch, lfp.shape[0])
        print(f"\tZAP channels {batch_start}:{batch_end} at {datetime.datetime.now()}")

        batch = lfp[batch_start:batch_end].T[:, :, np.newaxis].astype(np.float64, copy=False)
        print(f"\tBatch tensor shape before dss_line: {batch.shape}, dtype={batch.dtype}")
        try:
            batch, _ = dss_line(batch, fline=60.0, sfreq=frq, blocksize=blocksize)
        except Exception as exc:
            raise RuntimeError(
                f"ZAP filtering failed for channels {batch_start}:{batch_end} "
                f"with blocksize={blocksize} and sfreq={frq}"
            ) from exc

        batch = batch[:, :, 0].T
        filtered[batch_start:batch_end] = np.clip(batch, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

    return filtered

def notch_filtering(lfp, frq, base_freq=60.0, quality=30.0):
    nyquist = frq / 2.0
    harmonics = np.arange(base_freq, nyquist, base_freq)
    print(
        f"\tStarting notch line-noise removal: channels={lfp.shape[0]}, "
        f"samples={lfp.shape[1]}, sfreq={frq}, harmonics={harmonics.tolist()}"
    )

    for i in range(lfp.shape[0]):
        print(f"\tNotch filtering channel {i} at {datetime.datetime.now()}")
        channel = lfp[i].astype(np.float32, copy=True)
        for harmonic in harmonics:
            b, a = spsg.iirnotch(harmonic, quality, fs=frq)
            channel = spsg.filtfilt(b, a, channel).astype(np.float32, copy=False)
        lfp[i] = np.clip(channel, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
        del channel
        gc.collect()

    return lfp

def remove_line_noise(lfp, frq, method=LINE_NOISE_METHOD):
    if method == "notch":
        return notch_filtering(lfp, frq)
    if method == "dss":
        return zap_filtering_dss(lfp, frq)
    raise ValueError(f"Unsupported LINE_NOISE_METHOD: {method}")
#%%
print(f"Line-noise filtering method: {LINE_NOISE_METHOD}")
print("ZapFiltering...")
print(datetime.datetime.now())
print("\tProcessing LFP1 data...")
lfp1_data = remove_line_noise(lfp1_data, freq)
print(datetime.datetime.now())

#%%
np.save('lfp1_zap', lfp1_data)
#%%
#LFP2
print(datetime.datetime.now())
print("\tProcessing LFP2 data...")
lfp2_data = remove_line_noise(lfp2_data, freq)
print(datetime.datetime.now())

#%%
np.save('lfp2_zap', lfp2_data)
#%%
############################################################
#                  Bad Channel Rejection                   #
############################################################
# Automatic Iterative Standard Deviation method (Komosar, et al. 2022)
def channelwise_std(lfp):
    std_values = np.empty(lfp.shape[0], dtype=np.float32)
    for ch in range(lfp.shape[0]):
        print(f"\tComputing channel std {ch} at {datetime.datetime.now()}")
        std_values[ch] = np.std(lfp[ch].astype(np.float32, copy=False), dtype=np.float32)
    return std_values

def std_bad_channels(lfp):
    all_channels = np.arange(lfp.shape[0])
    remaining_channels = all_channels.copy()
    k = 0  # iteration counter
    sd_pk = np.inf  # std of all individual channel std's
    std_all = channelwise_std(lfp)
    while sd_pk > 5:
        print(f"\tBad-channel iteration {k} at {datetime.datetime.now()}")
        sd_k = std_all[remaining_channels] # std of each channel
        m_k = np.median(sd_k)  # median of channel std's
        third_quartile = np.percentile(sd_k, 75)
        temp = np.std(sd_k)
        if sd_pk == temp:  # if no channels are removed (not in paper)
            break
        sd_pk = temp
        bad_channels_k = []
        for ch in remaining_channels:
            sd_jk = std_all[ch]
            if sd_jk < 10e-1:
                bad_channels_k.append(ch)
            elif sd_jk > 100:
                bad_channels_k.append(ch)
            elif abs(sd_jk - m_k) > third_quartile:
                bad_channels_k.append(ch)
        remaining_channels = np.setdiff1d(remaining_channels, bad_channels_k)
        k+=1
    bad_channels = np.setdiff1d(all_channels, remaining_channels)
    print("\trejecting channels:", bad_channels)
    return remaining_channels, bad_channels

def process_indexes(bdict, indexes, offset=0):
    for i in indexes:
        for x in bdict:
            if i+offset < bdict[x][0]:
                bdict[x] = (bdict[x][0], bdict[x][1]-1)
            
def process_backmap(mapping, indexes, offset=0):
    for i in indexes:
        mapping[i+offset] = None
        for j in range(i+offset+1,len(mapping)):
            if mapping[j] is not None:
                mapping[j] -= 1

#%%

print(brain_areas)
print("Bad Channel Rejection")
if SKIP_BAD_CHANNEL_REJECTION:
    print("\tSkipping bad-channel rejection for both LFP blocks.")
else:
    print("\tProcessing LFP1 data...")
    remaining_channels, bad_channels = std_bad_channels(lfp1_data)
    process_backmap(back_mapping, bad_channels)
    process_indexes(brain_areas, bad_channels)
    lfp1_data[bad_channels,:] = 0.0
    #%%
    print("\tProcessing LFP2 data...")
    remaining_channels, bad_channels = std_bad_channels(lfp2_data)
    process_backmap(back_mapping, bad_channels, offset=128)
    process_indexes(brain_areas, bad_channels, offset=128)
    lfp2_data[bad_channels,:] = 0.0

#==============================
#          Epoching
#==============================
def extract_epoch(data_a, data_b, start_idx, end_idx, ratio, window_size=500):
    fstart = int(start_idx // ratio)
    fend = int(end_idx // ratio)
    duration = fend - fstart
    midpoint = duration // 2
    center = fstart + midpoint
    window_start = center - (window_size // 2)
    window_end = window_start + window_size
    max_samples = min(data_a.shape[1], data_b.shape[1])

    if duration < window_size + 1:
        return None, duration
    if window_start < 0 or window_end > max_samples:
        return None, duration

    epoch = np.concatenate(
        (data_a[:, window_start:window_end], data_b[:, window_start:window_end]),
        axis=0,
    )
    if epoch.shape != (data_a.shape[0] + data_b.shape[0], window_size):
        return None, duration
    return epoch, duration

for i in range(len(CueOn)):
    if i % 25 == 0:
        print(f"\tEpoching trial {i}/{len(CueOn)} at {datetime.datetime.now()}")
    cOn = CueOn[i]
    cOff = CueOff[i]
    gStart = GraspStart[i]
    gEnd = GraspEnd[i]
    tEnd = TrialEnd[i]

    # prereach
    prereach_epoch, duration = extract_epoch(lfp1_data, lfp2_data, cOn, cOff, ratio, window_size=epoch_window)
    if prereach_epoch is None:
        discarded.add(i)
    else:
        structured_data[i, 0] = prereach_epoch

    # reach
    reach_epoch, duration = extract_epoch(lfp1_data, lfp2_data, cOff, gStart, ratio, window_size=epoch_window)
    if reach_epoch is None:
        discarded.add(i)
    else:
        structured_data[i, 1] = reach_epoch

    # grasp
    grasp_epoch, duration = extract_epoch(lfp1_data, lfp2_data, gStart, gEnd, ratio, window_size=epoch_window)
    if grasp_epoch is None:
        discarded.add(i)
    else:
        structured_data[i, 2] = grasp_epoch

keep_mask = np.ones(len(CueOn), dtype=bool)
if discarded:
    keep_mask[list(discarded)] = False
structured_data = structured_data[keep_mask]
print(f"Discarded {len(discarded)} trials during epoching.")

for x in sorted(list(discarded), reverse=True):
    for k in dc_info.keys():
        dc_info[k].pop(x)

print(structured_data.shape)
print(len(dc_info["Precision/Power"]))

CLEANED_STRUCTURED_OUT = CLEANED_DATA_DIR / "structured"
CLEANED_STRUCTURED_OUT.mkdir(parents=True, exist_ok=True)

structured_stem = f"data_{name_file}{OUTPUT_TAG}"
info_stem = f"info_{name_file}{OUTPUT_TAG}.pkl"

np.save(CLEANED_STRUCTURED_OUT / structured_stem, structured_data)
with open(CLEANED_STRUCTURED_OUT / info_stem, 'wb') as fp:
    pickle.dump(dc_info, fp)
    print('Info succesfully saved to file')

print(f"Finished preprocessing at {datetime.datetime.now().isoformat()}")
LOG_FILE.close()

# %%
