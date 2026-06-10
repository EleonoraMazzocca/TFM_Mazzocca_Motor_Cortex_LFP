# TFM_Mazzocca_Motor_Cortex_LFP

Compositional generalization in motor-cortex local field potentials (LFP): decoding and generating brain states for unseen movement combinations.

This repository is organized as a research pipeline rather than as the original nested preprocessing folder. The layout follows the Cookiecutter idea of a clean, reproducible project skeleton, but keeps the thesis workflow visible at the top level.

## Project Layout

```text
preprocess_pipeline/          Raw-data standardization, cleaning, temporal phase segmentation, and class separation
baseline_linear_classifier/   Classical logistic-regression baselines
transformer_encoder/          Joint phase/grip/hand transformer encoder and condition-sentence utilities
cvae/                         Conditional VAE and MMD-cVAE generation experiments
configs/                      Condition-sentence configuration JSON files
docs/                         Original notes and migrated README material
scripts/                      Miscellaneous helper and verification scripts
archive/                      Legacy/exploratory code kept for reference
```

Generated outputs are intentionally excluded from the repository: raw data, cleaned arrays, model checkpoints, logs, plots, PDFs, `.npy`, `.npz`, and `.pt` files.

## Environment

Data paths are centralized in `preprocess_pipeline/data_paths.py` and default to:

```bash
TFM_DATA_ROOT=/mnt/temp_drive
```

The expected external data folder structure is:

```text
$TFM_DATA_ROOT/
  RawData/
  parametersY/
  Cleaned_Data/
  Separated_Data/
```

Override `TFM_DATA_ROOT` when running on another machine. The repository stores code only; the large input and output arrays live outside git.

## Data Flow

```text
raw LFP data
-> cleaning and standardization
-> temporal phase segmentation
-> class separation by grip/hand/angle
-> phase expansion for phase/grip/hand learning
-> transformer encoder
-> embedding cVAE / MMD-cVAE experiments
```

Phase segmentation is performed in `preprocess_pipeline/data_preprocess.py`:

- `prereach`: `CueOn -> CueOff`
- `reach`: `CueOff -> GraspStart`
- `grasp`: `GraspStart -> GraspEnd`

Each phase is represented as a centered 500-sample window. After concatenating LFP blocks along channels, structured trials have shape:

```text
(n_trials, 3, channels, 500)
```

The phase is initially encoded by axis 1: `0 = prereach`, `1 = reach`, `2 = grasp`.

Class separation groups full segmented trials into grip/hand/angle class files while preserving the phase axis. The joint transformer loader later expands each trial into three phase-level samples.

## Cleaning Pipeline: Order And Outputs

The cleaning pipeline converts the laboratory MATLAB recordings and event metadata into NumPy arrays that can be used by the baseline classifier, transformer, and cVAE experiments.

A typical session-level order is:

```text
1. data_standardization.py
2. generate_motorno_csv.py
3. inspect_bad_channels.py optional, diagnostic
4. data_preprocess.py
5. data_separation.py after all sessions have been preprocessed
```

`generate_motorno_csv.py` is not listed in the main question, but it is part of the practical flow because `data_preprocess.py` expects `motorno_<session>.csv` to exist.

### 1. `preprocess_pipeline/data_standardization.py`

Purpose: converts one raw MATLAB LFP recording into simple NumPy files. This is the first step because later scripts work with `.npy` arrays rather than loading the large `.mat` file repeatedly.

Intuition: it opens the raw recording, extracts the two LFP blocks (`LFP1` and `LFP2`), saves each as a separate array, and stores small metadata needed to align event times with the LFP sampling rate.

Current session selection: the script currently uses the `name_file` variable inside the file, for example `20180619Y`. To process another session, change that value before running.

Run:

```bash
python -m preprocess_pipeline.data_standardization
```

Inputs:

```text
$TFM_DATA_ROOT/RawData/lfp_data_<session>.mat
```

Expected outputs:

```text
$TFM_DATA_ROOT/Cleaned_Data/lfp1_data_<session>.npy
$TFM_DATA_ROOT/Cleaned_Data/lfp2_data_<session>.npy
$TFM_DATA_ROOT/Cleaned_Data/meta/meta_<session>.pkl
```

What the outputs represent:

- `lfp1_data_<session>.npy`: first LFP recording block, shaped approximately `(channels_lfp1, time_samples)`.
- `lfp2_data_<session>.npy`: second LFP recording block, shaped approximately `(channels_lfp2, time_samples)`.
- `meta_<session>.pkl`: small dictionary with sampling metadata, especially `fs` and `ratio`, used to convert behavioral event sample indices to LFP sample indices.

### 2. `preprocess_pipeline/generate_motorno_csv.py`

Purpose: extracts trial condition labels from the event parameter file and writes them in a small CSV format used by preprocessing.

Intuition: the behavioral file stores each trial condition with codes such as power/precision, left/right hand, and angle. This script creates a compact lookup table so every trial can later be assigned a class.

Run:

```bash
python -m preprocess_pipeline.generate_motorno_csv <session>
```

Expected output:

```text
$TFM_DATA_ROOT/Cleaned_Data/meta/motorno_<session>.csv
```

What the output represents:

- A two-row CSV containing the trial motor codes and angles.
- `data_preprocess.py` reads this file to build trial-level labels such as precision/power, unimanual/bimanual, left angle, and right angle.

### 3. `preprocess_pipeline/inspect_bad_channels.py` optional

Purpose: diagnostic tool for understanding bad-channel rejection before running the full cleaning step.

Intuition: it runs the same style of filtering and channel-variance checks on one session/block and prints which channels would be rejected and why. It is useful when a session behaves strangely or when you want to justify/explain channel rejection.

Run examples:

```bash
python -m preprocess_pipeline.inspect_bad_channels 20180613Y --block lfp1
python -m preprocess_pipeline.inspect_bad_channels 20180613Y --block both
```

Inputs:

```text
$TFM_DATA_ROOT/Cleaned_Data/lfp1_data_<session>.npy
$TFM_DATA_ROOT/Cleaned_Data/lfp2_data_<session>.npy
$TFM_DATA_ROOT/Cleaned_Data/meta/meta_<session>.pkl
$TFM_DATA_ROOT/Cleaned_Data/meta/motorno_<session>.csv
```

Expected outputs:

- No main data file is produced.
- The script prints a per-channel report to the terminal: kept channels, rejected channels, standard deviation summaries, and rejection reasons.

How to interpret it:

- This is not required to produce the dataset.
- It is a quality-control/debugging script to inspect why channels may be removed or zeroed during preprocessing.

### 4. `preprocess_pipeline/data_preprocess.py`

Purpose: performs the main signal cleaning and temporal segmentation into movement phases.

Intuition: this is where the continuous LFP traces become trial-level examples. For each trial, the script filters the signal, removes line noise, optionally rejects bad channels, then cuts the trial into prereach, reach, and grasp windows.

Session selection and useful environment variables:

```bash
TFM_PREPROCESS_SESSION=20180619Y python -m preprocess_pipeline.data_preprocess
```

Optional settings:

```bash
TFM_LINE_NOISE_METHOD=notch       # default, no extra meegkit dependency
TFM_LINE_NOISE_METHOD=dss         # DSS/ZAP method if meegkit is installed
TFM_SKIP_BAD_CHANNEL_REJECTION=1  # keep all channels, useful for controlled comparisons
TFM_PREPROCESS_OUTPUT_TAG=_mua_200_500
TFM_BANDPASS_LOW_HZ=200
TFM_BANDPASS_HIGH_HZ=500
```

Inputs:

```text
$TFM_DATA_ROOT/Cleaned_Data/lfp1_data_<session>.npy
$TFM_DATA_ROOT/Cleaned_Data/lfp2_data_<session>.npy
$TFM_DATA_ROOT/Cleaned_Data/meta/meta_<session>.pkl
$TFM_DATA_ROOT/Cleaned_Data/meta/motorno_<session>.csv
$TFM_DATA_ROOT/parametersY/Params_<session>.mat
```

Expected outputs:

```text
$TFM_DATA_ROOT/Cleaned_Data/structured/data_<session><tag>.npy
$TFM_DATA_ROOT/Cleaned_Data/structured/info_<session><tag>.pkl
data_preprocess_<session><tag>.log
```

The `.log` file is written in the current working directory and records progress, filtering settings, discarded trials, and any traceback if the run fails.

What the outputs represent:

- `data_<session><tag>.npy`: the cleaned, segmented neural data for one session. Shape:

```text
(n_valid_trials, 3, channels, 500)
```

- Axis 0: valid trials after short/invalid trials are discarded.
- Axis 1: phase index: `0 = prereach`, `1 = reach`, `2 = grasp`.
- Axis 2: concatenated LFP channels from `lfp1` and `lfp2`, usually 256 before/with zeroed bad channels.
- Axis 3: 500 time samples centered within the event interval.

- `info_<session><tag>.pkl`: trial metadata aligned with the kept trials. It contains labels such as:

```text
Precision/Power
Unimanual/Bimanual
LeftAngle
RightAngle
```

These labels are used by `data_separation.py` to group trials into movement classes.

### 5. `preprocess_pipeline/data_separation.py`

Purpose: combines all session-level structured files and groups trials by movement class.

Intuition: after preprocessing, each session is still stored separately. This script pools sessions and creates one file per task condition, such as precision/right/135 degrees or power/left/45 degrees.

Run:

```bash
python -m preprocess_pipeline.data_separation
```

Optional input tag:

```bash
TFM_STRUCTURED_INPUT_TAG=_mua_200_500 python -m preprocess_pipeline.data_separation
```

Inputs:

```text
$TFM_DATA_ROOT/Cleaned_Data/structured/data_<session><tag>.npy
$TFM_DATA_ROOT/Cleaned_Data/structured/info_<session><tag>.pkl
```

Expected outputs:

```text
$TFM_DATA_ROOT/Separated_Data/classes/precision_unimanual_right_135_degrees.npy
$TFM_DATA_ROOT/Separated_Data/classes/power_unimanual_left_45_degrees.npy
...
```

There are separate class files for bimanual precision conditions, unimanual precision conditions, and unimanual power conditions. The exact canonical names are defined in `preprocess_pipeline/data_paths.py`.

Expected structure of each class file:

```text
(n_trials_for_this_class, 3, channels, 500)
```

Important: class separation does not remove or flatten the phase axis. Each saved class file still contains all three phase windows per trial. The phase axis is flattened later by `transformer_encoder/joint_embedding_data.py` through `phase_expand()`.

## Baseline And Diagnostic Scripts

### `baseline_linear_classifier/data_classification.py`

Purpose: broad classical baseline over many older task definitions.

Intuition: this script asks many decoding questions, for example power vs precision, left vs right, angle classification, phase classification, and unimanual vs bimanual. It converts each trial into compact per-channel mean-absolute-amplitude features and trains logistic regression models.

Status in this project:

- Useful as an older broad baseline and exploratory sanity check.
- Not the cleanest direct comparison to the current transformer, because the current transformer predicts `phase`, `grip`, and `hand` using phase-expanded samples.

Main outputs:

```text
scores_all_partial.npy
scores_all.npy
baseline_linear_classifier/logs/data_classification_<timestamp>.log
baseline_linear_classifier/logs/data_classification_<timestamp>.txt
baseline_linear_classifier/logs/data_classification_latest.txt
baseline_linear_classifier/logs/data_classification_confusion_matrices/<timestamp>/...
```

The score arrays store numerical results. The text files are human-readable summaries. The confusion-matrix folder contains CSV/PNG files for individual task results.

### `baseline_linear_classifier/run_classifier2_compositional.py`

Purpose: compositional logistic-regression baseline for the older `grip/hand/angle` setup.

Intuition: it loads the separated unimanual class files, extracts simple per-channel amplitude features, and trains separate logistic classifiers for each movement phase. It holds out a compositional condition, originally precision + right + 135 degrees, to test whether the linear model generalizes to a missing combination.

Status in this project:

- Useful as a baseline for grip/hand/angle compositional decoding.
- Not yet fully matched to the newer transformer target set (`phase`, `grip`, `hand`). A stricter comparison should reuse the transformer phase-expanded samples and train linear heads for `phase`, `grip`, and `hand`.

Default outputs:

```text
baseline_linear_classifier/logs/classifier2/all_results.json
baseline_linear_classifier/logs/classifier2/split_balance_report.txt
baseline_linear_classifier/logs/classifier2/accuracy_summary.png
baseline_linear_classifier/logs/classifier2/<phase>/grip_results.json
baseline_linear_classifier/logs/classifier2/<phase>/hand_results.json
baseline_linear_classifier/logs/classifier2/<phase>/angle_results.json
baseline_linear_classifier/logs/classifier2/<phase>/confusion_matrices.png
```

### `scripts/leakage_verification_tool.py`

Purpose: static audit tool for possible train/test leakage risks.

Intuition: it reads pipeline scripts as text and checks for patterns that could make results optimistic, such as splitting after data have already been pooled in a way that may mix sessions or related trials.

Status in this project:

- Diagnostic only; it does not create data for the model.
- You likely did not use it as part of the main experiment. It was added as a sanity-check/reporting helper when we were worried about leakage in the older baseline pipeline.

Example:

```bash
python -m scripts.leakage_verification_tool \
  --classification baseline_linear_classifier/data_classification.py \
  --standardization preprocess_pipeline/data_standardization.py \
  --preprocess preprocess_pipeline/data_preprocess.py \
  --extra-scripts preprocess_pipeline/build_session_aware_structured_split.py \
  --output outputs/leakage_verification_report.txt
```

Output:

```text
outputs/leakage_verification_report.txt
```

If `--output` is omitted, the report is printed to the terminal only.

### `preprocess_pipeline/build_session_aware_structured_split.py`

Purpose: creates a JSON manifest for a session-aware/compositional split from the structured session files.

Intuition: instead of immediately pooling everything into class files, this script records which session and trial each example came from, then creates train/validation/test entries according to a holdout rule. This is useful when you want stricter split control and want to avoid accidental mixing across sessions or conditions.

Status in this project:

- It is a helper for stricter future experiments.
- It does not appear to be the split actually used by the current `transformer_encoder/run_joint_embedding.py`, which builds its own split after loading separated class files and applying `phase_expand()`.
- Keep it for reproducibility/future cleanup, but do not describe it as part of the main transformer/cVAE run unless you explicitly use it.

Run:

```bash
python -m preprocess_pipeline.build_session_aware_structured_split
```

Default output:

```text
outputs/session_aware_structured_split.json
```

The JSON contains entries for train/validation/test splits. Each entry records the session, trial index, class name, angles, paths to the structured data and metadata, and available phases.

## Transformer Encoder

The current transformer code lives in `transformer_encoder/`. `run_joint_embedding.py` trains a joint encoder to predict:

- phase
- grip
- hand

Feature extraction supports:

- `mu`: mean absolute amplitude per channel, arranged as `4 areas x 96 channel slots x 1 band`
- `broadband6`: six band-amplitude features, arranged as `4 areas x 96 channel slots x 6 bands`

## cVAE

The generation experiments live in `cvae/`, including embedding-space cVAE, MMD/cVAE diagnostics, latent ablation, and evaluation scripts.
