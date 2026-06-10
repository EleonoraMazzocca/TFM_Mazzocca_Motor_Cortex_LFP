# Masked-Instruction Transformer — Implementation Plan

## Overview

The core idea is to train a bimodal transformer that receives both LFP signals and an
instruction vector encoding the motor combination (grip × hand × angle).  To prevent the
model from taking a shortcut (ignoring LFP and just copying the instruction label), a
balanced per-class masking scheme zeros a fraction of instructions every epoch, forcing
the model to rely on LFP when no instruction is available.  At test time the instruction
is **always** the zero vector, so evaluation measures pure LFP decoding ability.

**Why the old run 1 (LFPInstructionTransformer, additive fusion, always-present
instruction) was discarded:** the model simply learned to copy the one-hot embedding
to the output heads.  Held-out accuracy was high during training (instruction was
always present) but collapsed at test time when the instruction was zeroed.  The new
design prevents this by construction.

**Primary scientific question:** does training with a partially-available instruction
improve the LFP encoder's compositional generalisation, measured by held-out angle
accuracy when the instruction is absent at test time?

---

## Files

```
transformer/
├── run_transformer.py       ← entry point; all 7 runs use this script
├── model.py                 ← LFPTransformerClassifier + LFPInstructionTransformer
├── data.py                  ← BalancedInstructionDataset (+ legacy LFPDataset)
├── train.py                 ← epoch loop with reshuffle_masks call
├── evaluate.py              ← evaluate_model with is_test safety check
├── instruction_encoding.py  ← all encoding logic (onehot / bow / minilm / none)
└── instruction_embedding.py ← legacy one-hot encoder (used by ablation scripts)
```

---

## Architecture

### Baseline: `LFPTransformerClassifier`

```
forward(x: (B, 3, 4, 96))
  → input_proj              # (B, 3, 4, d_model)
  → + phase_emb + area_emb
  → reshape → (B, 12, d_model)
  → TransformerEncoder (n_layers)
  → mean pool → LayerNorm   # (B, d_model)
  → head_grip, head_hand, head_angle
```

### Instruction model: `LFPInstructionTransformer`

```
forward(x: (B, 3, 4, 96), instr: (B, instruction_dim))
  → [LFP pathway, identical to baseline] → pooled: (B, d_model)
  → instruction_proj(instr) → relu          # (B, instruction_proj_dim=32)
  → cat([pooled, instr_out], dim=1)          # (B, d_model + 32)
  → head_grip, head_hand, head_angle
```

**Why concat not add:** additive fusion (old design) let the instruction directly shift
the LFP representation in the same vector space, making it trivial for the model to
ignore LFP.  Concat keeps the two modalities separated until the classification heads,
giving the LFP pathway a clear gradient path even when instruction=0.

**Why instruction_proj_dim=32 not d_model=64:** projecting 8-dim onehot to 64-dim would
massively over-parameterise the instruction path and make shortcut learning easier.

---

## Dataset: `BalancedInstructionDataset`

| Split | is_test | mask_prob | Effect |
|-------|---------|-----------|--------|
| train | False | 0.5 or 0.7 | per-class masking reshuffled every epoch |
| val | True | 1.0 | instruction always zero |
| seen_test | True | 1.0 | instruction always zero |
| heldout_test | True | 1.0 | instruction always zero |

**Mask reshuffle (per epoch, in train.py):**
For each unique (grip, hand, angle) combination, randomly select
`floor(len * mask_prob)` samples to have their instruction zeroed.
Re-randomised every epoch — not seeded — so the model cannot memorise which samples
are masked.

---

## Experiment Table

Run the baseline first; every other run is compared against it.

| Run | Command flags | `--out_dir` | Key question |
|-----|--------------|------------|-------------|
| 0 | `--encoding none` | `results/run0_baseline` | Baseline floor |
| 1a | `--encoding onehot --mask_prob 0.5` | `results/run1a_onehot_50` | Does structured label help at 50/50? |
| 1b | `--encoding onehot --mask_prob 0.7` | `results/run1b_onehot_70` | Does more masking force better LFP use? |
| 2a | `--encoding bow --mask_prob 0.5` | `results/run2a_bow_50` | Does word co-occurrence add anything? |
| 2b | `--encoding bow --mask_prob 0.7` | `results/run2b_bow_70` | |
| 3a | `--encoding minilm --mask_prob 0.5` | `results/run3a_minilm_50` | Does semantic structure improve LFP encoder? |
| 3b | `--encoding minilm --mask_prob 0.7` | `results/run3b_minilm_70` | |

**Example commands** (run from `GitHub_PreProcess_Pipeline/CrossTaskClassification/`):

```bash
# Run 0 — baseline
python transformer/run_transformer.py \
    --cache_dir /tmp/lfp_cache \
    --out_dir results/run0_baseline \
    --encoding none --seed 42

# Run 1a — onehot, 50% masking
python transformer/run_transformer.py \
    --cache_dir /tmp/lfp_cache \
    --out_dir results/run1a_onehot_50 \
    --encoding onehot --mask_prob 0.5 --seed 42

# Run 1b — onehot, 70% masking
python transformer/run_transformer.py \
    --cache_dir /tmp/lfp_cache \
    --out_dir results/run1b_onehot_70 \
    --encoding onehot --mask_prob 0.7 --seed 42

# Run 2a / 2b — bow
python transformer/run_transformer.py \
    --cache_dir /tmp/lfp_cache \
    --out_dir results/run2a_bow_50 \
    --encoding bow --mask_prob 0.5 --seed 42

# Run 3a / 3b — minilm (requires: pip install sentence-transformers)
python transformer/run_transformer.py \
    --cache_dir /tmp/lfp_cache \
    --out_dir results/run3a_minilm_50 \
    --encoding minilm --mask_prob 0.5 --seed 42
```

---

## Acceptance Tests

Run these before launching any training:

```python
from instruction_encoding import encode_onehot, encode_bow, get_instruction_dim

# 1. Encoding correctness
assert encode_onehot("power_left_0").tolist()        == [1,0,1,0,1,0,0,0]
assert encode_onehot("precision_right_135").tolist() == [0,1,0,1,0,0,0,1]

v = encode_bow("precision_right_135")
assert v[1] == 1 and v[3] == 1 and v[7] == 1 and v[8] == 1 and v.sum() == 4

assert get_instruction_dim("onehot") == 8
assert get_instruction_dim("bow")    == 9
assert get_instruction_dim("minilm") == 384
assert get_instruction_dim("none")   == 0

# 2. Balanced masking within 5% of target rate
import numpy as np
from data import make_compositional_split, load_dataset, BalancedInstructionDataset

data = load_dataset(...)
train_data, *_, norm_stats = make_compositional_split(data)
ds = BalancedInstructionDataset(train_data, norm_stats, encoding="onehot", mask_prob=0.5)
for combo in np.unique(ds._combo_keys):
    idx = np.where(ds._combo_keys == combo)[0]
    rate = ds._mask[idx].mean()
    assert abs(rate - 0.5) < 0.05, f"mask rate {rate:.3f} off target for combo {combo}"

# 3. Masks reshuffle between calls
before = ds._mask.copy()
ds.reshuffle_masks()
assert not np.array_equal(before, ds._mask)

# 4. Test dataset always zeros instruction
test_ds = BalancedInstructionDataset(train_data, norm_stats, encoding="onehot", is_test=True)
for i in range(len(test_ds)):
    _, _, _, _, instr = test_ds[i]
    assert instr.sum() == 0.0

# 5. Model forward: live vs zero instruction must differ
import torch
from model import LFPInstructionTransformer

model = LFPInstructionTransformer(instruction_dim=8)
lfp   = torch.randn(4, 3, 4, 96)
live  = torch.randn(4, 8)
zero  = torch.zeros(4, 8)
_, _, out_live = model(lfp, live)
_, _, out_zero = model(lfp, zero)
assert not torch.allclose(out_live, out_zero), "concat fusion not connected to heads"
```

---

## Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `heldout angle` same for all runs | LFP mean-amplitude lacks compositional angle signal | Switch to PSD features or temporal input |
| `seen acc` drops vs baseline | Instruction hurting LFP learning | Increase mask_prob; check instruction_proj weight norms |
| `live ≈ zero` in acceptance test 5 | Fusion not connected | Verify concat dim in head definitions |
| `evaluate_model` AssertionError on ablation scripts | Ablation datasets lack `is_test=True` | Update `ZeroedInstructionDataset` etc. to set `is_test = True` |

---

## Key Comparisons

| Comparison | What it proves |
|---|---|
| Run 1a/1b vs Run 0 | Does any instruction encoding help LFP decoding? |
| Run 1a vs 1b | Does higher masking rate force better LFP use? |
| Run 2 vs Run 1 | Does word-level structure matter beyond indicator bits? |
| Run 3 vs Run 1 | Does semantic sentence embedding improve compositional angle accuracy? |

The single number that decides all comparisons: **held-out angle accuracy**.
