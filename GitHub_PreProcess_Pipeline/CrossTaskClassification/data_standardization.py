import mat73
import numpy as np
import os
import pickle
import sys

try:
    from data_paths import CLEANED_DATA_DIR, PARAMETERS_DIR, RAW_DATA_DIR
except ImportError:
    from .data_paths import CLEANED_DATA_DIR, PARAMETERS_DIR, RAW_DATA_DIR

INPUT_PATH = RAW_DATA_DIR.parent
OUTPUT_PATH = CLEANED_DATA_DIR
name_file = "20180619Y"
# 20180531Y --> done
# 20180601Y --> done
# 20180606Y --> done
# 20180607Y --> done
# 20180608Y --> done
# 20180612Y --> done
# 20180613Y --> done
# 20180614Y --> done
# 20180615Y --> done
# 20180618Y --> done
# 20180619Y --> done

# Load only LFP1 and LFP2 data one at a time to avoid Out Of Memory errors
# 1. Load, process and save LFP1 data
print("Loading LFP1...")
data_lfp1 = mat73.loadmat(RAW_DATA_DIR / f"lfp_data_{name_file}.mat", only_include=['lfp_data2/streams/LFP1'])
lfp1 = data_lfp1["lfp_data2"]["streams"]["LFP1"]

# This is the clock frequency of the data from the Laboratory, you may change it to the sampling frequency of the LFP
basefs = 4.8828125*10**3

lfp1_fs = float(lfp1['fs'])
lfp1_ratio = float(basefs / lfp1_fs) if lfp1_fs != basefs else 1.0

meta = dict()
meta['ratio'] = lfp1_ratio
meta['fs'] = lfp1_fs

# Create output directories if they don't exist
os.makedirs(OUTPUT_PATH / "meta", exist_ok=True)

# Save LFP1 data
np.save(OUTPUT_PATH / f"lfp1_data_{name_file}", lfp1['data'])

# Delete LFP1 variables from memory
del data_lfp1
del lfp1
import gc
gc.collect()

# 2. Load and save LFP2 data
print("Loading LFP2...")
data_lfp2 = mat73.loadmat(RAW_DATA_DIR / f"lfp_data_{name_file}.mat", only_include=['lfp_data2/streams/LFP2'])
lfp2 = data_lfp2["lfp_data2"]["streams"]["LFP2"]

# Save LFP2 data
np.save(OUTPUT_PATH / f"lfp2_data_{name_file}", lfp2['data'])

# Delete LFP2 variables from memory
del data_lfp2
del lfp2
gc.collect()

with open(OUTPUT_PATH / "meta" / f"meta_{name_file}.pkl", 'wb') as fp:
    pickle.dump(meta, fp)
    print('dictionary saved successfully to file')
