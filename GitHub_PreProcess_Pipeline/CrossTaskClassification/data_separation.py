import pickle
import numpy as np
import os

try:
    from data_paths import CLEANED_STRUCTURED_DIR, SEPARATED_CLASSES_DIR, CLASS_FILE_NAMES, with_class_tag
except ImportError:
    from .data_paths import CLEANED_STRUCTURED_DIR, SEPARATED_CLASSES_DIR, CLASS_FILE_NAMES, with_class_tag

STRUCTURED_INPUT_TAG = os.getenv("TFM_STRUCTURED_INPUT_TAG", "")

data_files = [
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

# Separate into:
precision_bimanual_45_degrees = []
precision_bimanual_135_degrees = []
precision_bimanual_45_135_degrees = []
precision_bimanual_135_45_degrees = []
precision_unimanual_right_0_degrees = []
precision_unimanual_right_45_degrees = []
precision_unimanual_right_90_degrees = []
precision_unimanual_right_135_degrees = []
precision_unimanual_left_0_degrees = []
precision_unimanual_left_45_degrees = []
precision_unimanual_left_90_degrees = []
precision_unimanual_left_135_degrees = []
power_unimanual_right_0_degrees = []
power_unimanual_right_45_degrees = []
power_unimanual_right_90_degrees = []
power_unimanual_right_135_degrees = []
power_unimanual_left_0_degrees = []
power_unimanual_left_45_degrees = []
power_unimanual_left_90_degrees = []
power_unimanual_left_135_degrees = []

for file_name in data_files:
    # Load classes info
    with open(CLEANED_STRUCTURED_DIR / f"info_{file_name}{STRUCTURED_INPUT_TAG}.pkl", 'rb') as fp:
        info = pickle.load(fp)
    # Load data
    data = np.load(CLEANED_STRUCTURED_DIR / f"data_{file_name}{STRUCTURED_INPUT_TAG}.npy")
    
    print("="*50)
    print(f"Successfully loaded file: {file_name}")
    print("Data shape:", data.shape)
    print("First trial shape:", data[0].shape)
    print("-" * 50)
    
    print("Info type:", type(info))
    if hasattr(info, 'head'):
        print("Info head:\n", info.head())
    elif isinstance(info, dict):
        print("Info keys:", info.keys())
        for k in list(info.keys())[:5]:
            val = info[k]
            print(f" - {k}: {str(val)[:100]}")
    else:
        print("Info:", info)
    print("="*50)
    
    print("="*100)
    print("*"*100)
    print("="*100)
    print(len(info["Precision/Power"]))
    print(data.shape)
    
    # iterate over the data
    for i in range(len(info["Precision/Power"])):
        if info['Precision/Power'][i] == 0: # Precision
            if info['Unimanual/Bimanual'][i] == 0: # Unimanual
                if info['LeftAngle'][i] == -1: # Left doesn't have an angle
                    if info['RightAngle'][i] == '0':
                        precision_unimanual_right_0_degrees.append(data[i])
                    elif info['RightAngle'][i] == '45':
                        precision_unimanual_right_45_degrees.append(data[i])
                    elif info['RightAngle'][i] == '90':
                        precision_unimanual_right_90_degrees.append(data[i])
                    elif info['RightAngle'][i] == '135':
                        precision_unimanual_right_135_degrees.append(data[i])
                    else:
                        print(info['LeftAngle'][i], "and", info['RightAngle'][i])
                else: # Right doesn't have an angle
                    if info['LeftAngle'][i] == '0':
                        precision_unimanual_left_0_degrees.append(data[i])
                    elif info['LeftAngle'][i] == '45':
                        precision_unimanual_left_45_degrees.append(data[i])
                    elif info['LeftAngle'][i] == '90':
                        precision_unimanual_left_90_degrees.append(data[i])
                    elif info['LeftAngle'][i] == '135':
                        precision_unimanual_left_135_degrees.append(data[i])
                    else:
                        print(info['LeftAngle'][i], "and", info['RightAngle'][i])
            else: # Bimanual
                if info['LeftAngle'][i] == '45' and info['RightAngle'][i] == '45':
                    precision_bimanual_45_degrees.append(data[i])
                elif info['LeftAngle'][i] == '135' and info['RightAngle'][i] == '135':
                    precision_bimanual_135_degrees.append(data[i])
                elif info['LeftAngle'][i] == '45' and info['RightAngle'][i] == '135':
                    precision_bimanual_45_135_degrees.append(data[i])
                elif info['LeftAngle'][i] == '135' and info['RightAngle'][i] == '45':
                    precision_bimanual_135_45_degrees.append(data[i])
                else:
                    print(info['LeftAngle'][i], "and", info['RightAngle'][i])
        else: # Power
            if info['Unimanual/Bimanual'][i] == 0: # Unimanual
                if info['LeftAngle'][i] == -1: # Left doesn't have an angle
                    if info['RightAngle'][i] == '0':
                        power_unimanual_right_0_degrees.append(data[i])
                    elif info['RightAngle'][i] == '45':
                        power_unimanual_right_45_degrees.append(data[i])
                    elif info['RightAngle'][i] == '90':
                        power_unimanual_right_90_degrees.append(data[i])
                    elif info['RightAngle'][i] == '135':
                        power_unimanual_right_135_degrees.append(data[i])
                    else:
                        print(info['LeftAngle'][i], "and", info['RightAngle'][i])
                else: # Right doesn't have an angle
                    if info['LeftAngle'][i] == '0':
                        power_unimanual_left_0_degrees.append(data[i])
                    elif info['LeftAngle'][i] == '45':
                        power_unimanual_left_45_degrees.append(data[i])
                    elif info['LeftAngle'][i] == '90':
                        power_unimanual_left_90_degrees.append(data[i])
                    elif info['LeftAngle'][i] == '135':
                        power_unimanual_left_135_degrees.append(data[i])
                    else:
                        print(info['LeftAngle'][i], "and", info['RightAngle'][i])
            else:
                print("what 6")
                print(info['LeftAngle'][i], "and", info['RightAngle'][i])
    del data

precision_bimanual_45_degrees = np.array(precision_bimanual_45_degrees)
precision_bimanual_135_degrees = np.array(precision_bimanual_135_degrees)
precision_bimanual_45_135_degrees = np.array(precision_bimanual_45_135_degrees)
precision_bimanual_135_45_degrees = np.array(precision_bimanual_135_45_degrees)
precision_unimanual_right_0_degrees = np.array(precision_unimanual_right_0_degrees)
precision_unimanual_right_45_degrees = np.array(precision_unimanual_right_45_degrees)
precision_unimanual_right_90_degrees = np.array(precision_unimanual_right_90_degrees)
precision_unimanual_right_135_degrees = np.array(precision_unimanual_right_135_degrees)
precision_unimanual_left_0_degrees = np.array(precision_unimanual_left_0_degrees)
precision_unimanual_left_45_degrees = np.array(precision_unimanual_left_45_degrees)
precision_unimanual_left_90_degrees = np.array(precision_unimanual_left_90_degrees)
precision_unimanual_left_135_degrees = np.array(precision_unimanual_left_135_degrees)
power_unimanual_right_0_degrees = np.array(power_unimanual_right_0_degrees)
power_unimanual_right_45_degrees = np.array(power_unimanual_right_45_degrees)
power_unimanual_right_90_degrees = np.array(power_unimanual_right_90_degrees)
power_unimanual_right_135_degrees = np.array(power_unimanual_right_135_degrees)
power_unimanual_left_0_degrees = np.array(power_unimanual_left_0_degrees)
power_unimanual_left_45_degrees = np.array(power_unimanual_left_45_degrees)
power_unimanual_left_90_degrees = np.array(power_unimanual_left_90_degrees)
power_unimanual_left_135_degrees = np.array(power_unimanual_left_135_degrees)

print("/"*100)
print("/"*100)
print("/"*100)
print(precision_bimanual_45_degrees.shape)
print(precision_bimanual_135_degrees.shape)
print(precision_bimanual_45_135_degrees.shape)
print(precision_bimanual_135_45_degrees.shape)
print(precision_unimanual_right_0_degrees.shape)
print(precision_unimanual_right_45_degrees.shape)
print(precision_unimanual_right_90_degrees.shape)
print(precision_unimanual_right_135_degrees.shape)
print(precision_unimanual_left_0_degrees.shape)
print(precision_unimanual_left_45_degrees.shape)
print(precision_unimanual_left_90_degrees.shape)
print(precision_unimanual_left_135_degrees.shape)
print(power_unimanual_right_0_degrees.shape)
print(power_unimanual_right_45_degrees.shape)
print(power_unimanual_right_90_degrees.shape)
print(power_unimanual_right_135_degrees.shape)
print(power_unimanual_left_0_degrees.shape)
print(power_unimanual_left_45_degrees.shape)
print(power_unimanual_left_90_degrees.shape)
print(power_unimanual_left_135_degrees.shape)

SEPARATED_CLASSES_DIR.mkdir(parents=True, exist_ok=True)
arrays_to_save = {
    "PRECISION_BIMANUAL_45": precision_bimanual_45_degrees,
    "PRECISION_BIMANUAL_135": precision_bimanual_135_degrees,
    "PRECISION_BIMANUAL_45_135": precision_bimanual_45_135_degrees,
    "PRECISION_BIMANUAL_135_45": precision_bimanual_135_45_degrees,
    "PRECISION_UNIMANUAL_R_0": precision_unimanual_right_0_degrees,
    "PRECISION_UNIMANUAL_R_45": precision_unimanual_right_45_degrees,
    "PRECISION_UNIMANUAL_R_90": precision_unimanual_right_90_degrees,
    "PRECISION_UNIMANUAL_R_135": precision_unimanual_right_135_degrees,
    "PRECISION_UNIMANUAL_L_0": precision_unimanual_left_0_degrees,
    "PRECISION_UNIMANUAL_L_45": precision_unimanual_left_45_degrees,
    "PRECISION_UNIMANUAL_L_90": precision_unimanual_left_90_degrees,
    "PRECISION_UNIMANUAL_L_135": precision_unimanual_left_135_degrees,
    "POWER_UNIMANUAL_R_0": power_unimanual_right_0_degrees,
    "POWER_UNIMANUAL_R_45": power_unimanual_right_45_degrees,
    "POWER_UNIMANUAL_R_90": power_unimanual_right_90_degrees,
    "POWER_UNIMANUAL_R_135": power_unimanual_right_135_degrees,
    "POWER_UNIMANUAL_L_0": power_unimanual_left_0_degrees,
    "POWER_UNIMANUAL_L_45": power_unimanual_left_45_degrees,
    "POWER_UNIMANUAL_L_90": power_unimanual_left_90_degrees,
    "POWER_UNIMANUAL_L_135": power_unimanual_left_135_degrees,
}

for class_name, array in arrays_to_save.items():
    output_name = with_class_tag(CLASS_FILE_NAMES[class_name])
    np.save(SEPARATED_CLASSES_DIR / output_name, array)
