# CrossTaskClassification

## Data Mount

This project expects the shared data volume at:

```bash
/mnt/temp_drive
```

Expected folders:

```bash
/mnt/temp_drive/RawData
/mnt/temp_drive/parametersY
/mnt/temp_drive/Cleaned_Data
/mnt/temp_drive/Separated_Data
```

All scripts in this folder read the shared root from `data_paths.py`.

Default:

```bash
TFM_DATA_ROOT=/mnt/temp_drive
```

Optional override:

```bash
export TFM_DATA_ROOT=/some/other/mountpoint
```

## Main Workflow

Top-level scripts in this folder are the maintained pipeline:

- `data_preprocess.py`
- `data_separation.py`
- `data_standardization.py`
- `data_classification.py`
- `run_classifier2_compositional.py`
- `build_session_aware_structured_split.py`
- `leakage_verification_tool.py`
- `inspect_bad_channels.py`

Exploratory and one-off analysis scripts were moved to
`archived_analysis_tools/` to keep this directory focused.

## Run Classification

Main baseline:

```bash
python GitHub_PreProcess_Pipeline/CrossTaskClassification/data_classification.py
```

Outputs:
- timestamped console log in `GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/data_classification_<timestamp>.log`
- timestamped plain-text summary in `GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/data_classification_<timestamp>.txt`
- latest plain-text summary in `GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/data_classification_latest.txt`
- NumPy checkpoint/output files in the working directory: `scores_all_partial.npy` and `scores_all.npy`

Compositional baseline:

```bash
python GitHub_PreProcess_Pipeline/CrossTaskClassification/run_classifier2_compositional.py
```

Outputs go to `GitHub_PreProcess_Pipeline/CrossTaskClassification/logs/classifier2/`.

## Session-Aware Split Helper

Use:

```bash
python GitHub_PreProcess_Pipeline/CrossTaskClassification/build_session_aware_structured_split.py
```

This helper exists for session-aware experiments and to avoid mixing trials from
the same recording session across splits.

## Leakage Verification

Use:

```bash
python GitHub_PreProcess_Pipeline/CrossTaskClassification/leakage_verification_tool.py \
  --classification GitHub_PreProcess_Pipeline/CrossTaskClassification/data_classification.py \
  --standardization GitHub_PreProcess_Pipeline/CrossTaskClassification/data_standardization.py \
  --preprocess GitHub_PreProcess_Pipeline/CrossTaskClassification/data_preprocess.py \
  --extra-scripts GitHub_PreProcess_Pipeline/CrossTaskClassification/build_session_aware_structured_split.py
```

This generates a static report about likely leakage risks in the current
pipeline code.

## Archived Tools

Archived exploratory scripts now live in
`GitHub_PreProcess_Pipeline/CrossTaskClassification/archived_analysis_tools/`.

That folder includes:
- PCA trajectory plotting utilities
- signal similarity / clustering inspection
- masked-channel reconstruction experiments
- session PCA health checks
- separated-segment inspection

