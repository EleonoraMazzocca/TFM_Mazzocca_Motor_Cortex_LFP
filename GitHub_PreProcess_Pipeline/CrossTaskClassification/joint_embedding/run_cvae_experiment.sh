#!/bin/bash
# Run all cVAE experiment combinations overnight:
#   3 baseline seeds  (no augmentation)
#   3 seeds × 3 dropout values = 9 augmented runs
# Total: 12 training runs, then compare_only plots.
#
# Usage (from joint_embedding/ directory):
#   chmod +x run_cvae_experiment.sh
#   nohup bash run_cvae_experiment.sh > experiment.log 2>&1 &
#   tail -f experiment.log        # follow progress
#
# Or inside a tmux/screen session:
#   bash run_cvae_experiment.sh 2>&1 | tee experiment.log

DATA="/home/eleonora/TFM_Eleonora/data/classes"
CKPT="/home/eleonora/TFM_Eleonora/CrossTaskClassification/joint_embedding/results/broadband6/transformer_heldout_grasp_precision_right/checkpoint.pt"
BASE="/home/eleonora/TFM_Eleonora/CrossTaskClassification/joint_embedding/results/broadband6"

SEEDS=(42 123 7)
N_DROPS=(2 10 20)
FAILED=()

# Args shared by every run.
COMMON=(
    --data_dir      "$DATA"
    --input_mode    broadband6
    --joint_checkpoint "$CKPT"
    --heldout_phase grasp
    --heldout_grip  precision
    --heldout_hand  right
    --latent_dim    64
    --hidden_dims   256 128 64
    --no_early_stopping
    --split_seed    42
)

run_one() {
    local label="$1"; shift
    echo ""
    echo "============================================================"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  START: ${label}"
    echo "============================================================"
    if python run_embedding_cvae.py "$@"; then
        echo "  -> OK: ${label}"
    else
        echo "  -> FAILED: ${label}"
        FAILED+=("${label}")
    fi
}

# ── Augmented runs ───────────────────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    for n in "${N_DROPS[@]}"; do
        run_one "aug n_drop=${n} seed=${seed}" \
            "${COMMON[@]}" \
            --seed              "$seed" \
            --denoising_aug \
            --aug_n_dropout_dims "$n" \
            --out_dir           "${BASE}/cvae_grasp_precision_right_aug_n${n}_seed${seed}"
    done
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  $(date '+%Y-%m-%d %H:%M:%S')  ALL RUNS DONE"
echo "============================================================"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED runs (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "    - ${f}"; done
    exit 1
else
    echo "  All 9 runs completed successfully."
fi

# ── Comparison plots ─────────────────────────────────────────────────────────
# Uncomment to generate plots automatically after all runs finish.
# python run_embedding_cvae.py \
#     --compare_only \
#     --baseline_dirs \
#         "${BASE}/cvae_grasp_precision_right_seed42" \
#         "${BASE}/cvae_grasp_precision_right_seed123" \
#         "${BASE}/cvae_grasp_precision_right_seed7" \
#     --aug_dirs \
#         "${BASE}/cvae_grasp_precision_right_aug_n2_seed42" \
#         "${BASE}/cvae_grasp_precision_right_aug_n2_seed123" \
#         "${BASE}/cvae_grasp_precision_right_aug_n2_seed7" \
#         "${BASE}/cvae_grasp_precision_right_aug_n10_seed42" \
#         "${BASE}/cvae_grasp_precision_right_aug_n10_seed123" \
#         "${BASE}/cvae_grasp_precision_right_aug_n10_seed7" \
#         "${BASE}/cvae_grasp_precision_right_aug_n20_seed42" \
#         "${BASE}/cvae_grasp_precision_right_aug_n20_seed123" \
#         "${BASE}/cvae_grasp_precision_right_aug_n20_seed7" \
#     --out_dir "${BASE}/comparison_grasp_precision_right"
