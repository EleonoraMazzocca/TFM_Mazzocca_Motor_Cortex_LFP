#!/bin/bash
set -e

cd /home/ryaan/Documents/TFM_Eleonora/GitHub_PreProcess_Pipeline/CrossTaskClassification
mkdir -p transformer/results

echo "=============================="
echo "Starting all runs: $(date)"
echo "=============================="

# Install MiniLM dependency if missing
python -c "import sentence_transformers" 2>/dev/null || {
    echo "Installing sentence-transformers..."
    pip install sentence-transformers -q
}

CACHE=/tmp/lfp_cache
R=transformer/results

for ENC_MASK in \
    "none:0.5:run0_baseline" \
    "onehot:0.5:run1a_onehot_50" \
    "onehot:0.7:run1b_onehot_70" \
    "bow:0.5:run2a_bow_50" \
    "bow:0.7:run2b_bow_70" \
    "minilm:0.5:run3a_minilm_50" \
    "minilm:0.7:run3b_minilm_70"
do
    ENC=$(echo $ENC_MASK | cut -d: -f1)
    MASK=$(echo $ENC_MASK | cut -d: -f2)
    DIR=$(echo $ENC_MASK | cut -d: -f3)
    echo ""
    echo "=============================="
    echo "=== $DIR  [$(date +%H:%M:%S)] ==="
    echo "=============================="
    python transformer/run_transformer.py \
        --cache_dir $CACHE \
        --out_dir $R/$DIR \
        --encoding $ENC \
        --mask_prob $MASK \
        --seed 42
done

echo ""
echo "=============================="
echo "All runs done: $(date)"
echo "=============================="
echo ""
echo "--- heldout angle accuracy ---"
for D in $R/run*/; do
    python -c "
import json
d = json.load(open('${D}summary.json'))
enc  = d.get('encoding', 'none')
mask = d.get('mask_prob', '-')
ha   = d['heldout_accuracy']['angle']
sa   = d['seen_accuracy']['angle']
print(f'  {enc:<8} mask={mask}  seen={sa:.4f}  heldout={ha:.4f}  ->  ${D}')
"
done
