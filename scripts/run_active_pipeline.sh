#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="data/classes"
INPUT_MODE="broadband6"
HELDOUT_PHASE="grasp"
HELDOUT_GRIP="precision"
HELDOUT_HAND="right"
OUTPUT_ROOT="outputs"
DEVICE="auto"
SEEDS=(42 43 44)

RUN_LINEAR=1
RUN_TRANSFORMER=1
RUN_CVAE=1
RUN_EVAL=0
RUN_ABLATION=0
SKIP_PERMUTATION=1
RUN_CVAE_ELBO=1
RUN_CVAE_MMD=0
LAMBDA_MMD="0.1"

CONDITION_TYPE="onehot"
SENTENCE_CONDITION_PATH=""
SENTENCE_KEY_ORDER_PATH=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_active_pipeline.sh [options]

Options:
  --data_dir PATH
  --input_mode {mu|broadband6}
  --heldout_phase {prereach|reach|grasp}
  --heldout_grip {power|precision}
  --heldout_hand {left|right}
  --output_root PATH
  --device {auto|cpu|cuda}
  --seed INT
  --seeds INT [INT ...]

  --condition_type {onehot|sentence}
  --sentence_condition_path PATH
  --sentence_key_order_path PATH

  --with_eval
  --with_ablation
  --with_mmd
  --mmd_only
  --lambda_mmd FLOAT
  --no_linear
  --no_transformer
  --no_cvae
  --run_permutation
  --help

Notes:
  - By default this runs three seeds: 42 43 44.
  - Outputs are grouped under outputs/<input_mode>/<phase>_<grip>_<hand>/.
  - By default the cVAE stage runs ELBO only.
  - --with_mmd adds a second cVAE run with --mmd_loss.
  - The default MMD weight is lambda_mmd=0.1.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --input_mode)
            INPUT_MODE="$2"
            shift 2
            ;;
        --heldout_phase)
            HELDOUT_PHASE="$2"
            shift 2
            ;;
        --heldout_grip)
            HELDOUT_GRIP="$2"
            shift 2
            ;;
        --heldout_hand)
            HELDOUT_HAND="$2"
            shift 2
            ;;
        --output_root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEEDS=("$2")
            shift 2
            ;;
        --seeds)
            shift
            SEEDS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SEEDS+=("$1")
                shift
            done
            if [[ ${#SEEDS[@]} -eq 0 ]]; then
                echo "--seeds requires at least one integer." >&2
                exit 1
            fi
            ;;
        --condition_type)
            CONDITION_TYPE="$2"
            shift 2
            ;;
        --sentence_condition_path)
            SENTENCE_CONDITION_PATH="$2"
            shift 2
            ;;
        --sentence_key_order_path)
            SENTENCE_KEY_ORDER_PATH="$2"
            shift 2
            ;;
        --with_eval)
            RUN_EVAL=1
            shift
            ;;
        --with_ablation)
            RUN_ABLATION=1
            shift
            ;;
        --with_mmd)
            RUN_CVAE_MMD=1
            shift
            ;;
        --mmd_only)
            RUN_CVAE_ELBO=0
            RUN_CVAE_MMD=1
            shift
            ;;
        --lambda_mmd)
            LAMBDA_MMD="$2"
            shift 2
            ;;
        --no_linear)
            RUN_LINEAR=0
            shift
            ;;
        --no_transformer)
            RUN_TRANSFORMER=0
            shift
            ;;
        --no_cvae)
            RUN_CVAE=0
            shift
            ;;
        --run_permutation)
            SKIP_PERMUTATION=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$CONDITION_TYPE" == "sentence" ]]; then
    if [[ -z "$SENTENCE_CONDITION_PATH" || -z "$SENTENCE_KEY_ORDER_PATH" ]]; then
        echo "Sentence conditioning requires both --sentence_condition_path and --sentence_key_order_path." >&2
        exit 1
    fi
fi

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

MODE_OUT_DIR="${OUTPUT_ROOT}/${INPUT_MODE}"
COMBO_LABEL="${HELDOUT_PHASE}_${HELDOUT_GRIP}_${HELDOUT_HAND}"
COMBO_OUT_DIR="${MODE_OUT_DIR}/${COMBO_LABEL}"
MASTER_SUMMARY_PATH="${COMBO_OUT_DIR}/run_summary_${CONDITION_TYPE}.txt"

mkdir -p "$COMBO_OUT_DIR"

{
    echo "Active experiment pipeline"
    echo "timestamp_start: $(timestamp)"
    echo "root_dir: $ROOT_DIR"
    echo "data_dir: $DATA_DIR"
    echo "input_mode: $INPUT_MODE"
    echo "heldout_phase: $HELDOUT_PHASE"
    echo "heldout_grip: $HELDOUT_GRIP"
    echo "heldout_hand: $HELDOUT_HAND"
    echo "condition_type: $CONDITION_TYPE"
    echo "sentence_condition_path: ${SENTENCE_CONDITION_PATH:-N/A}"
    echo "sentence_key_order_path: ${SENTENCE_KEY_ORDER_PATH:-N/A}"
    echo "device: $DEVICE"
    echo "seeds: ${SEEDS[*]}"
    echo "run_linear: $RUN_LINEAR"
    echo "run_transformer: $RUN_TRANSFORMER"
    echo "run_cvae: $RUN_CVAE"
    echo "run_eval: $RUN_EVAL"
    echo "run_ablation: $RUN_ABLATION"
    echo "skip_permutation: $SKIP_PERMUTATION"
    echo "run_cvae_elbo: $RUN_CVAE_ELBO"
    echo "run_cvae_mmd: $RUN_CVAE_MMD"
    echo "lambda_mmd: $LAMBDA_MMD"
    echo "combo_out_dir: $COMBO_OUT_DIR"
} > "$MASTER_SUMMARY_PATH"

echo "============================================================"
echo "Active experiment pipeline started: $(timestamp)"
echo "Root:          $ROOT_DIR"
echo "Data dir:      $DATA_DIR"
echo "Input mode:    $INPUT_MODE"
echo "Held-out:      ${HELDOUT_PHASE}+${HELDOUT_GRIP}+${HELDOUT_HAND}"
echo "Condition:     $CONDITION_TYPE"
echo "Seeds:         ${SEEDS[*]}"
echo "Output root:   $COMBO_OUT_DIR"
echo "Summary:       $MASTER_SUMMARY_PATH"
echo "============================================================"

run_cvae_variant() {
    local seed="$1"
    local loss_name="$2"
    local out_dir="$3"
    shift 3
    local extra_flags=("$@")
    local transformer_dir="${COMBO_OUT_DIR}/seed${seed}/transformer_heldout_${COMBO_LABEL}"

    echo
    echo "[3/5] cVAE (${loss_name}, seed=${seed}) -> $out_dir"
    local cvae_cmd=(
        python -m cvae.run_embedding_cvae
        --data_dir "$DATA_DIR"
        --joint_checkpoint "${transformer_dir}/checkpoint.pt"
        --input_mode "$INPUT_MODE"
        --heldout_phase "$HELDOUT_PHASE"
        --heldout_grip "$HELDOUT_GRIP"
        --heldout_hand "$HELDOUT_HAND"
        --condition_type "$CONDITION_TYPE"
        --seed "$seed"
        --device "$DEVICE"
        --out_dir "$out_dir"
    )
    if [[ "$CONDITION_TYPE" == "sentence" ]]; then
        cvae_cmd+=(
            --sentence_condition_path "$SENTENCE_CONDITION_PATH"
            --sentence_key_order_path "$SENTENCE_KEY_ORDER_PATH"
        )
    fi
    if [[ ${#extra_flags[@]} -gt 0 ]]; then
        cvae_cmd+=("${extra_flags[@]}")
    fi
    "${cvae_cmd[@]}"
}

run_eval_variant() {
    local seed="$1"
    local loss_name="$2"
    local cvae_dir="$3"
    local eval_dir="$4"
    local transformer_dir="${COMBO_OUT_DIR}/seed${seed}/transformer_heldout_${COMBO_LABEL}"

    echo
    echo "[4/5] Standalone cVAE evaluation (${loss_name}, seed=${seed}) -> $eval_dir"
    python -m cvae.evaluate_generation \
        --joint_checkpoint "${transformer_dir}/checkpoint.pt" \
        --cvae_checkpoint "$cvae_dir/checkpoint.pt" \
        --seen_embeddings "${transformer_dir}/seen_embeddings.npz" \
        --heldout_embeddings "${transformer_dir}/heldout_embeddings.npz" \
        --cvae_norm_stats "$cvae_dir/normalization_stats.npz" \
        --heldout_phase "$HELDOUT_PHASE" \
        --heldout_grip "$HELDOUT_GRIP" \
        --heldout_hand "$HELDOUT_HAND" \
        --seed "$seed" \
        --device "$DEVICE" \
        --out_dir "$eval_dir"
}

run_ablation_variant() {
    local seed="$1"
    local loss_name="$2"
    local cvae_dir="$3"

    echo
    echo "[5/5] Latent ablation (${loss_name}, seed=${seed}) -> $cvae_dir"
    python -m cvae.latent_ablation_cvae \
        --run_dirs "$cvae_dir" \
        --data_dir "$DATA_DIR" \
        --input_mode "$INPUT_MODE" \
        --generation_seed "$seed" \
        --device "$DEVICE"
}

for SEED in "${SEEDS[@]}"; do
    SEED_OUT_DIR="${COMBO_OUT_DIR}/seed${SEED}"
    LINEAR_OUT_DIR="${SEED_OUT_DIR}/linear"
    TRANSFORMER_OUT_DIR="${SEED_OUT_DIR}/transformer_heldout_${COMBO_LABEL}"
    CVAE_ELBO_OUT_DIR="${SEED_OUT_DIR}/cvae_${COMBO_LABEL}_${CONDITION_TYPE}_elbo"
    CVAE_MMD_OUT_DIR="${SEED_OUT_DIR}/cvae_${COMBO_LABEL}_${CONDITION_TYPE}_mmd"
    EVAL_ELBO_OUT_DIR="${SEED_OUT_DIR}/evaluation_${CONDITION_TYPE}_elbo"
    EVAL_MMD_OUT_DIR="${SEED_OUT_DIR}/evaluation_${CONDITION_TYPE}_mmd"
    RUN_SUMMARY_PATH="${SEED_OUT_DIR}/run_summary_${CONDITION_TYPE}.txt"

    mkdir -p "$SEED_OUT_DIR"

    {
        echo "seed: $SEED"
        echo "timestamp_start: $(timestamp)"
        echo "linear_out_dir: $LINEAR_OUT_DIR"
        echo "transformer_out_dir: $TRANSFORMER_OUT_DIR"
        echo "transformer_checkpoint: $TRANSFORMER_OUT_DIR/checkpoint.pt"
        echo "cvae_elbo_out_dir: $CVAE_ELBO_OUT_DIR"
        echo "cvae_mmd_out_dir: $CVAE_MMD_OUT_DIR"
        echo "cvae_elbo_checkpoint: $CVAE_ELBO_OUT_DIR/checkpoint.pt"
        echo "cvae_mmd_checkpoint: $CVAE_MMD_OUT_DIR/checkpoint.pt"
        echo "eval_elbo_out_dir: $EVAL_ELBO_OUT_DIR"
        echo "eval_mmd_out_dir: $EVAL_MMD_OUT_DIR"
        echo "seen_embeddings: $TRANSFORMER_OUT_DIR/seen_embeddings.npz"
        echo "heldout_embeddings: $TRANSFORMER_OUT_DIR/heldout_embeddings.npz"
        echo "cvae_elbo_norm_stats: $CVAE_ELBO_OUT_DIR/normalization_stats.npz"
        echo "cvae_mmd_norm_stats: $CVAE_MMD_OUT_DIR/normalization_stats.npz"
    } > "$RUN_SUMMARY_PATH"

    {
        echo
        echo "seed${SEED}:"
        echo "  summary: $RUN_SUMMARY_PATH"
        echo "  linear_out_dir: $LINEAR_OUT_DIR"
        echo "  transformer_out_dir: $TRANSFORMER_OUT_DIR"
        echo "  cvae_elbo_out_dir: $CVAE_ELBO_OUT_DIR"
        echo "  cvae_mmd_out_dir: $CVAE_MMD_OUT_DIR"
        echo "  eval_elbo_out_dir: $EVAL_ELBO_OUT_DIR"
        echo "  eval_mmd_out_dir: $EVAL_MMD_OUT_DIR"
    } >> "$MASTER_SUMMARY_PATH"

    echo
    echo "------------------------------------------------------------"
    echo "Running seed ${SEED}"
    echo "Seed folder: $SEED_OUT_DIR"
    echo "Seed summary: $RUN_SUMMARY_PATH"
    echo "------------------------------------------------------------"

    if [[ "$RUN_LINEAR" -eq 1 ]]; then
        echo
        echo "[1/5] Linear baseline (seed=${SEED}) -> $LINEAR_OUT_DIR"
        python -m baseline_linear_classifier.run_linear_phase_grip_hand \
            --data_dir "$DATA_DIR" \
            --input_mode "$INPUT_MODE" \
            --heldout \
            --heldout_phase "$HELDOUT_PHASE" \
            --heldout_grip "$HELDOUT_GRIP" \
            --heldout_hand "$HELDOUT_HAND" \
            --seed "$SEED" \
            --out_dir "$LINEAR_OUT_DIR"
    fi

    if [[ "$RUN_TRANSFORMER" -eq 1 ]]; then
        echo
        echo "[2/5] Transformer (seed=${SEED}) -> $TRANSFORMER_OUT_DIR"
        transformer_cmd=(
            python -m transformer_encoder.run_joint_embedding
            --data_dir "$DATA_DIR"
            --input_mode "$INPUT_MODE"
            --heldout
            --heldout_phase "$HELDOUT_PHASE"
            --heldout_grip "$HELDOUT_GRIP"
            --heldout_hand "$HELDOUT_HAND"
            --seed "$SEED"
            --device "$DEVICE"
            --out_dir "$TRANSFORMER_OUT_DIR"
        )
        if [[ "$SKIP_PERMUTATION" -eq 1 ]]; then
            transformer_cmd+=(--skip_permutation)
        fi
        "${transformer_cmd[@]}"
    fi

    if [[ "$RUN_CVAE" -eq 1 ]]; then
        if [[ "$RUN_CVAE_ELBO" -eq 1 ]]; then
            run_cvae_variant "$SEED" "elbo" "$CVAE_ELBO_OUT_DIR"
        fi
        if [[ "$RUN_CVAE_MMD" -eq 1 ]]; then
            run_cvae_variant "$SEED" "mmd" "$CVAE_MMD_OUT_DIR" --mmd_loss --lambda_mmd "$LAMBDA_MMD"
        fi
    fi

    if [[ "$RUN_EVAL" -eq 1 ]]; then
        if [[ "$RUN_CVAE_ELBO" -eq 1 ]]; then
            run_eval_variant "$SEED" "elbo" "$CVAE_ELBO_OUT_DIR" "$EVAL_ELBO_OUT_DIR"
        fi
        if [[ "$RUN_CVAE_MMD" -eq 1 ]]; then
            run_eval_variant "$SEED" "mmd" "$CVAE_MMD_OUT_DIR" "$EVAL_MMD_OUT_DIR"
        fi
    fi

    if [[ "$RUN_ABLATION" -eq 1 ]]; then
        if [[ "$RUN_CVAE_ELBO" -eq 1 ]]; then
            run_ablation_variant "$SEED" "elbo" "$CVAE_ELBO_OUT_DIR"
        fi
        if [[ "$RUN_CVAE_MMD" -eq 1 ]]; then
            run_ablation_variant "$SEED" "mmd" "$CVAE_MMD_OUT_DIR"
        fi
    fi

    {
        echo "timestamp_end: $(timestamp)"
        echo "status: completed"
    } >> "$RUN_SUMMARY_PATH"
done

{
    echo
    echo "timestamp_end: $(timestamp)"
    echo "status: completed"
} >> "$MASTER_SUMMARY_PATH"

echo
echo "Pipeline finished: $(timestamp)"
echo "Combination folder: $COMBO_OUT_DIR"
echo "Master summary:     $MASTER_SUMMARY_PATH"
