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

## Main Components

### Preprocess Pipeline

The preprocessing code lives in `preprocess_pipeline/` and contains the original standardization, metadata, phase segmentation, bad-channel inspection, and class-separation utilities.

### Baseline Linear Classifier

The existing linear baselines live in `baseline_linear_classifier/`. Note that `run_classifier2_compositional.py` is still the older grip/hand/angle logistic baseline run separately per phase. A stricter comparison to the current transformer should use the same phase-expanded samples and heads: `phase`, `grip`, and `hand`.

### Transformer Encoder

The current transformer code lives in `transformer_encoder/`. `run_joint_embedding.py` trains a joint encoder to predict:

- phase
- grip
- hand

Feature extraction supports:

- `mu`: mean absolute amplitude per channel, arranged as `4 areas x 96 channel slots x 1 band`
- `broadband6`: six band-amplitude features, arranged as `4 areas x 96 channel slots x 6 bands`

### cVAE

The generation experiments live in `cvae/`, including embedding-space cVAE, MMD/cVAE diagnostics, latent ablation, and evaluation scripts.

## Environment

Data paths are centralized in `preprocess_pipeline/data_paths.py` and default to:

```bash
TFM_DATA_ROOT=/mnt/temp_drive
```

Override this variable when running on another machine.
