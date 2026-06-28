# TFM_Mazzocca_Motor_Cortex_LFP

Compositional generalization in motor-cortex local field potentials (LFP): decoding movement factors and generating transformer-embedding representations for unseen movement combinations.

This repository is organized as a research pipeline rather than as the original nested preprocessing folder. The layout follows the Cookiecutter idea of a clean, reproducible project skeleton, but keeps the thesis workflow visible at the top level.

## Project Layout

```text
preprocess_pipeline/          Raw-data standardization, cleaning, temporal phase segmentation, and class separation
baseline_linear_classifier/   Classical logistic-regression baselines
transformer_encoder/          Joint phase/grip/hand transformer encoder and condition-sentence utilities
cvae/                         Conditional VAE and MMD-cVAE generation experiments
configs/                      Condition-sentence configuration JSON files
scripts/                      Active end-to-end pipeline runner
```

Generated outputs and local notes are intentionally excluded from the repository: raw data, cleaned arrays, model checkpoints, logs, plots, PDFs, `.npy`, `.npz`, `.pt` files, and `docs/` drafts.

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

Session selection: pass one or more session IDs, or pass no session IDs to process every `lfp_data_*.mat` file found in `RawData/`. Existing outputs are skipped unless `--overwrite` is used.

Run one session:

```bash
python -m preprocess_pipeline.data_standardization 20180619Y
```

Run all raw sessions:

```bash
python -m preprocess_pipeline.data_standardization
```

Force regeneration:

```bash
python -m preprocess_pipeline.data_standardization --overwrite
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

### `baseline_linear_classifier/run_linear_phase_grip_hand.py`

Purpose: direct logistic-regression baseline for the current transformer task.

Intuition: this script uses the same separated class files as the transformer, expands each trial into one sample per phase, extracts the same MU or six-band features, and trains three independent linear heads for `phase`, `grip`, and `hand`. This gives a simple baseline for asking whether the transformer architecture improves over linear decoding on the same inputs and split.

Example, six-band held-out comparison:

```bash
python -m baseline_linear_classifier.run_linear_phase_grip_hand \
  --data_dir data/classes \
  --input_mode broadband6 \
  --heldout \
  --heldout_phase grasp \
  --heldout_grip precision \
  --heldout_hand right \
  --out_dir outputs/broadband6/linear_heldout_grasp_precision_right
```

Example, MU held-out comparison:

```bash
python -m baseline_linear_classifier.run_linear_phase_grip_hand \
  --data_dir data/classes \
  --input_mode mu \
  --heldout \
  --heldout_phase grasp \
  --heldout_grip precision \
  --heldout_hand right \
  --out_dir outputs/mu/linear_heldout_grasp_precision_right
```

Main outputs:

```text
outputs/.../summary.json
outputs/.../phase_results.json
outputs/.../grip_results.json
outputs/.../hand_results.json
outputs/.../confusion_<head>_<split>.png
outputs/.../normalization_stats.npz
```

The reported splits are `seen_test` for combinations available during training and `heldout_test` for the held-out phase/grip/hand combination.

## Condition-Sentence Search

Sentence conditioning represents each of the 12 phase/grip/hand combinations
with a low-dimensional sentence embedding. The search utilities do not train a
cVAE; they only compare candidate condition sentences and export the selected
condition table as NumPy arrays.

### Fixed thesis condition set

`evaluate_condition_sentences.py` compares the built-in sentence strategies
and exports the selected instructional Option D as a five-dimensional PCA
table:

```bash
python -m cvae.condition_label.evaluate_condition_sentences \
  --out_dir outputs/sentence_eval
```

Main outputs:

```text
outputs/sentence_eval/condition_vectors_D_pca5.npy
outputs/sentence_eval/condition_keys_D_pca5.npy
outputs/sentence_eval/condition_linear_recoverability.json
```

The recoverability file records a descriptive multinomial logistic-regression
test across PCA dimensions. The classifier is fitted and evaluated on the same
12 condition vectors, so this measures whether the condition representation is
trivially linearly recoverable; it is not a cross-validated accuracy estimate.

These are the files used by the active pipeline's `--condition_type sentence`
mode.

### Search generated templates

`search_condition_sentences.py` builds combinations from the phase, grip, hand,
and sentence templates defined in the script. It embeds each 12-sentence set,
tests PCA dimensions, scores factor recoverability and held-out behavior, and
exports the highest-ranked table:

```bash
python -m cvae.condition_label.search_condition_sentences \
  --out_dir outputs/sentence_search \
  --dims 2 3 4 5 \
  --top_k 20
```

Main outputs include:

```text
outputs/sentence_search/sentence_search_top.json
outputs/sentence_search/best_sentences.txt
outputs/sentence_search/condition_vectors_best_pca<D>.npy
outputs/sentence_search/condition_keys_best_pca<D>.npy
```

### Score the JSON candidate sets

The files in `configs/` are optional, manually curated candidate collections.
They are not loaded automatically by `run_active_pipeline.sh`. Score one with:

```bash
python -m cvae.condition_label.score_candidates \
  configs/conditions_parallel.json \
  --cross \
  --out_dir outputs/sentence_candidates/parallel
```

The same command can be used with:

```text
configs/conditions_codex_targeted.json
configs/conditions_dominance_targeted.json
```

`--cross` evaluates both MiniLM and MPNet; omit it for MiniLM only. The command
ranks candidates across PCA dimensions and saves the best sentence set, key
order, and condition-vector table in the requested output directory. To use a
selected table in training, pass the exported vector and key-order `.npy` files
to `run_active_pipeline.sh` through `--sentence_condition_path` and
`--sentence_key_order_path`.

## Transformer Encoder

The current transformer code lives in `transformer_encoder/`. The only transformer file you normally execute is `run_joint_embedding.py`; the other files are imported helpers.

Execution sequence:

```text
1. Make sure class files exist in data/classes/
   - MU files:        *_mua_200_500.npy
   - broadband files: *_degrees.npy

2. Train the joint transformer:
   python -m transformer_encoder.run_joint_embedding ...

3. Use the transformer outputs downstream:
   - checkpoint.pt for cvae/run_embedding_cvae.py
   - seen_embeddings.npz and heldout_embeddings.npz for optional evaluation
   - summary.json and confusion matrices for reporting
```

Main transformer files:

```text
transformer_encoder/run_joint_embedding.py   executable training/evaluation script
transformer_encoder/joint_embedding_data.py  loads class files, phase-expands trials, extracts/cache features
transformer_encoder/joint_embedding_model.py neural network architecture
transformer_encoder/attention.py             attention layer that stores attention weights for diagnostics
```

`run_joint_embedding.py` trains a joint encoder to predict:

- phase
- grip
- hand

Feature extraction supports:

- `mu`: mean absolute amplitude per channel, arranged as `4 areas x 96 channel slots x 1 band`
- `broadband6`: six band-amplitude features, arranged as `4 areas x 96 channel slots x 6 bands`

Held-out experiments use a zero-shot combination split:

```text
train:        seen phase/grip/hand combinations only
validation:   seen phase/grip/hand combinations only
seen_test:    seen phase/grip/hand combinations only
heldout_test: all phase-level samples from the held-out phase/grip/hand combination
```

Held-out samples are not used for early stopping, hyperparameter selection, or model selection. In code summaries this is recorded as `split_protocol: strict_zero_shot`. Interpreted carefully, this means zero-shot with respect to the held-out phase x grip x hand label combination; it is not a leave-trial-out or leave-session-out protocol, because trials are expanded into phase-level samples before splitting.

Example MU run:

```bash
python -m transformer_encoder.run_joint_embedding \
  --data_dir data/classes \
  --input_mode mu \
  --heldout \
  --heldout_phase grasp \
  --heldout_grip precision \
  --heldout_hand right \
  --skip_permutation \
  --out_dir outputs/mu/transformer_heldout_grasp_precision_right
```

Example six-band run:

```bash
python -m transformer_encoder.run_joint_embedding \
  --data_dir data/classes \
  --input_mode broadband6 \
  --heldout \
  --heldout_phase grasp \
  --heldout_grip precision \
  --heldout_hand right \
  --skip_permutation \
  --out_dir outputs/broadband6/transformer_heldout_grasp_precision_right
```

Use `--skip_permutation` while iterating. Remove it only when you want the slower shuffled-label sanity check.

Transformer default hyperparameters in practice:

- `batch_size=64`: smaller than the cVAE because transformer training is more memory-intensive per sample.
- `lr=3e-4`: conservative default for attention-based models.
- `n_layers=2`: two transformer encoder blocks.
- `d_model=64`: size of each channel-token embedding.
- `feedforward_dim=128`: width of the MLP inside each transformer block.
- `dropout=0.35`: regularization inside the transformer.

## cVAE

The active generation path is the embedding-space cVAE:

```text
transformer_encoder/run_joint_embedding.py
-> cvae/run_embedding_cvae.py
-> cvae/embedding_cvae_pipeline.py
```

`run_embedding_cvae.py` is the user-facing wrapper. It validates the held-out condition and checkpoint metadata, then calls `embedding_cvae_pipeline.py`. The cVAE trains on pooled joint-transformer embeddings, not directly on LFP waveforms.

Example MU run:

```bash
python -m cvae.run_embedding_cvae \
  --data_dir data/classes \
  --joint_checkpoint outputs/mu/transformer_heldout_grasp_precision_right/checkpoint.pt \
  --input_mode mu \
  --heldout_phase grasp \
  --heldout_grip precision \
  --heldout_hand right \
  --out_dir outputs/mu/cvae_grasp_precision_right
```

Example six-band run:

```bash
python -m cvae.run_embedding_cvae \
  --data_dir data/classes \
  --joint_checkpoint outputs/broadband6/transformer_heldout_grasp_precision_right/checkpoint.pt \
  --input_mode broadband6 \
  --heldout_phase grasp \
  --heldout_grip precision \
  --heldout_hand right \
  --out_dir outputs/broadband6/cvae_grasp_precision_right
```

To rerun diagnostics from an existing cVAE output directory:

```bash
python -m cvae.run_embedding_cvae \
  --diag_only \
  --out_dir outputs/broadband6/cvae_grasp_precision_right
```

### One-command active pipeline

For the current active workflow, the repository also provides:

```text
scripts/run_active_pipeline.sh
```

This wrapper runs the main stages in order:

```text
linear baseline
-> joint transformer
-> cVAE
-> optional standalone cVAE evaluation
-> optional latent ablation
```

Default run:

```bash
bash scripts/run_active_pipeline.sh
```

This defaults to:

- `input_mode=broadband6`
- held-out `grasp + precision + right`
- one-hot conditioning
- `outputs/broadband6/...`

The script also writes a run summary with the key artifact paths, for example:

```text
outputs/broadband6/grasp_precision_right/run_summary_onehot.txt
```

That summary records the output directories and main files such as:

- per-seed linear baseline output directories
- per-seed transformer output directories and `checkpoint.pt` files
- per-seed cVAE output directories and `checkpoint.pt` files
- seen and held-out embedding files
- cVAE normalization statistics

To include the standalone evaluation and latent ablation:

```bash
bash scripts/run_active_pipeline.sh --with_eval --with_ablation
```

To run both cVAE loss variants, add:

```bash
bash scripts/run_active_pipeline.sh --with_mmd
```

If you want sentence conditioning, first generate the selected Option D condition table:

```bash
python -m cvae.condition_label.evaluate_condition_sentences \
  --out_dir outputs/sentence_eval
```

This writes:

```text
outputs/sentence_eval/condition_vectors_D_pca5.npy
outputs/sentence_eval/condition_keys_D_pca5.npy
```

Then run the full sentence-conditioned broadband6 pipeline with both ELBO and MMD:

```bash
bash scripts/run_active_pipeline.sh \
  --input_mode broadband6 \
  --heldout_phase grasp \
  --heldout_grip precision \
  --heldout_hand right \
  --condition_type sentence \
  --sentence_condition_path outputs/sentence_eval/condition_vectors_D_pca5.npy \
  --sentence_key_order_path outputs/sentence_eval/condition_keys_D_pca5.npy \
  --with_mmd \
  --with_eval \
  --with_ablation
```

cVAE default hyperparameters in practice:

- `batch_size=128`: larger than the transformer because the cVAE sees pooled embeddings and uses only MLPs, so it is cheaper per sample.
- `hidden_dims=128 64 32`: three hidden layers in the encoder MLP, mirrored in reverse in the decoder.
- `latent_dim=32`: size of the stochastic latent code `z`.
- `lr=1e-3`: higher than the transformer because this is a smaller dense model in an easier embedding space.
- `beta_max=1.0`: final weight on the KL term in the ELBO loss.
- `beta_anneal_epochs=10`: linearly ramps the KL weight up over the first 10 epochs.
- `noise_scale=0.1`: if `--denoising_aug` is enabled, adds Gaussian noise at 10% of each sample's embedding standard deviation.
- `dropout=0.2`: regularization between hidden layers in the cVAE MLP.

Active shared helpers:

```text
cvae/conditioning/       condition encodings: onehot.py and sentence.py
cvae/training.py         cVAE training loop, MMD-VAE loss path, augmentation helpers
cvae/metrics.py          shared metrics such as compute_mmd()
cvae/cvae_model.py       cVAE model definition
```

## Main Libraries

- `PyTorch`: transformer and cVAE models, losses, optimizers, datasets, and training loops.
- `NumPy`: array processing, normalization statistics, cached embeddings, and saved analysis artifacts.
- `scikit-learn`: train/validation/test splitting, PCA, pairwise distances, and a few condition-label analysis helpers. Examples: `sklearn.decomposition.PCA`.
- `SciPy`: statistical tests, Procrustes alignment, and some signal-processing utilities. Examples: `scipy.stats.wasserstein_distance`, `scipy.stats.kstest`, `scipy.linalg.orthogonal_procrustes`.
- `matplotlib`: diagnostic figures and summary plots for transformer, cVAE, and condition-label evaluation.
