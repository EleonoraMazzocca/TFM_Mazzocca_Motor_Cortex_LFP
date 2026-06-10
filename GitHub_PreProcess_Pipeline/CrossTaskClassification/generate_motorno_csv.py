import csv
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

try:
    from data_paths import CLEANED_META_DIR, PARAMETERS_DIR
except ImportError:
    from .data_paths import CLEANED_META_DIR, PARAMETERS_DIR

INPUT_PATH = PARAMETERS_DIR
OUTPUT_PATH = CLEANED_META_DIR


def _load_trials_parameters(name_file):
    mat_path = INPUT_PATH / f"Params_{name_file}.mat"
    data = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    return data["TrialsParameters"]


def _build_motorno_and_angle(params):
    is_precision = np.asarray(params.isPrecision).squeeze().astype(float)
    is_right_hand = np.asarray(params.isRightHand).squeeze().astype(float)
    cond_code = np.asarray(params.CondCode).squeeze().astype(int)

    motor_no = []
    angle = []

    unimanual_angle_map = {
        1: "0",
        2: "45",
        3: "90",
        4: "135",
    }
    bimanual_angle_map = {
        1: "45 - 45",
        2: "45 - 135",
        3: "135 - 45",
        4: "135 - 135",
    }

######################################################################################    
# isPrecisionDef is a task-type code: 0=Bimanual, 1=Precision, 2=Power
# isRightHand is a hand-type code: 0=Left, 1=Right
# cond_code is the angle code: 1=0, 2=45, 3=90, 4=135

# MotorNo: 3-4=Bimanual, 3=Precision Left, 4=Precision Right, 1=Power Left, 2=Power Right
# - if isPrecisionDef == 0: bimanual --> MotorNo = 3-4
# - if isPrecisionDef == 1: precision --> MotorNo = 3 or 4 (if isRightHand == 0 --> 3, if isRightHand == 1 --> 4)
# - if isPrecisionDef == 2: power --> MotorNo = 1 or 2 (if isRightHand == 0 --> 1, if isRightHand == 1 --> 2)
######################################################################################

    for task_code, right_hand, cond in zip(is_precision, is_right_hand, cond_code):
        if task_code == 0:
            motor_no.append("3-4")
            angle.append(bimanual_angle_map[cond])
        elif task_code == 1:
            if np.isnan(right_hand):
                raise ValueError("Found precision trial with missing hand label.")
            motor_no.append("4" if int(right_hand) == 1 else "3")
            angle.append(unimanual_angle_map[cond])
        elif task_code == 2:
            if np.isnan(right_hand):
                raise ValueError("Found power trial with missing hand label.")
            motor_no.append("2" if int(right_hand) == 1 else "1")
            angle.append(unimanual_angle_map[cond])
        else:
            raise ValueError(f"Unexpected isPrecision code: {task_code}")

    return np.array(motor_no, dtype=str), np.array(angle, dtype=str)


def generate_csv(name_file):
    params = _load_trials_parameters(name_file)
    motor_no, angle = _build_motorno_and_angle(params)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_PATH / f"motorno_{name_file}.csv"
    with out_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"')
        writer.writerow(motor_no.tolist())
        writer.writerow(angle.tolist())

    print(f"Wrote {out_path}")
    print(f"Trials: {len(motor_no)}")
    print(f"Motor labels: {dict(zip(*np.unique(motor_no, return_counts=True)))}")


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(
            path.stem.replace("Params_", "")
            for path in INPUT_PATH.glob("Params_*.mat")
        )

    for name_file in files:
        generate_csv(name_file)


if __name__ == "__main__":
    main()
