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
SEED="42"

RUN_LINEAR=1
RUN_TRANSFORMER=1
RUN_CVAE=1
RUN_EVAL=0
RUN_ABLATION=0
SKIP_PERMUTATION=1
RUN_CVAE_ELBO=1
RUN_CVAE_MMD=0

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

  --condition_type {onehot|sentence}
  --sentence_condition_path PATH
  --sentence_key_order_path PATH

  --with_eval
  --with_ablation
  --with_mmd
  --mmd_only
  --no_linear
  --no_transformer
  --no_cvae
  --run_permutation
  --help

Notes:
  - By default this runs: linear baseline -> transformer -> cVAE.
  - By default the cVAE stage runs ELBO only.
  - The cVAE stage already includes its main diagnostics and held-out generation summary.
  - --with_eval adds the standalone cvae.evaluate_generation.py pass.
  - --with_ablation adds the latent-ablation diagnostic after cVAE training.
  - --with_mmd adds a second cVAE run with --mmd_loss.
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
            SEED="$2"
            shift 2
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

MODE_OUT_DIR="${OUTPUT_ROOT}/${INPUT_MODE}"
LINEAR_OUT_DIR="${MODE_OUT_DIR}/linear_heldout_${HELDOUT_PHASE}_${HELDOUT_GRIP}_${HELDOUT_HAND}"
TRANSFORMER_OUT_DIR="${MODE_OUT_DIR}/transformer_heldout_${HELDOUT_PHASE}_${HELDOUT_GRIP}_${HELDOUT_HAND}"
CVAE_STEM="${MODE_OUT_DIR}/cvae_${HELDOUT_PHASE}_${HELDOUT_GRIP}_${HELDOUT_HAND}_${CONDITION_TYPE}"
CVAE_ELBO_OUT_DIR="${CVAE_STEM}_elbo"
CVAE_MMD_OUT_DIR="${CVAE_STEM}_mmd"
EVAL_ELBO_OUT_DIR="${MODE_OUT_DIR}/evaluation_${HELDOUT_PHASE}_${HELDOUT_GRIP}_${HELDOUT_HAND}_${CONDITION_TYPE}_elbo"
EVAL_MMD_OUT_DIR="${MODE_OUT_DIR}/evaluation_${HELDOUT_PHASE}_${HELDOUT_GRIP}_${HELDOUT_HAND}_${CONDITION_TYPE}_mmd"
RUN_SUMMARY_PATH="${MODE_OUT_DIR}/run_summary_${HELDOUT_PHASE}_${HELDOUT_GRIP}_${HELDOUT_HAND}_${CONDITION_TYPE}.txt"

mkdir -p "$MODE_OUT_DIR"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

echo "============================================================"

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
    echo "seed: $SEED"
    echo "device: $DEVICE"
    echo "run_linear: $RUN_LINEAR"
    echo "run_transformer: $RUN_TRANSFORMER"
    echo "run_cvae: $RUN_CVAE"
    echo "run_eval: $RUN_EVAL"
    echo "run_ablation: $RUN_ABLATION"
    echo "skip_permutation: $SKIP_PERMUTATION"
    echo "run_cvae_elbo: $RUN_CVAE_ELBO"
    echo "run_cvae_mmd: $RUN_CVAE_MMD"
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

echo "Run summary will be written to: $RUN_SUMMARY_PATH"
echo "Active experiment pipeline started: $(timestamp)"
echo "Root:          $ROOT_DIR"
echo "Data dir:      $DATA_DIR"
echo "Input mode:    $INPUT_MODE"
echo "Held-out:      ${HELDOUT_PHASE}+${HELDOUT_GRIP}+${HELDOUT_HAND}"
echo "Condition:     $CONDITION_TYPE"
echo "Output root:   $MODE_OUT_DIR"
echo "============================================================"

if [[ "$RUN_LINEAR" -eq 1 ]]; then
    echo
    echo "[1/5] Linear baseline -> $LINEAR_OUT_DIR"
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
    echo "[2/5] Transformer -> $TRANSFORMER_OUT_DIR"
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

run_cvae_variant() {
    local loss_name="$1"
    local out_dir="$2"
    shift 2
    local extra_flags=("$@")

    echo
    echo "[3/5] cVAE (${loss_name}) -> $out_dir"
    local cvae_cmd=(
        python -m cvae.run_embedding_cvae
        --data_dir "$DATA_DIR"
        --joint_checkpoint "$TRANSFORMER_OUT_DIR/checkpoint.pt"
        --input_mode "$INPUT_MODE"
        --heldout_phase "$HELDOUT_PHASE"
        --heldout_grip "$HELDOUT_GRIP"
        --heldout_hand "$HELDOUT_HAND"
        --condition_type "$CONDITION_TYPE"
        --seed "$SEED"
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
    local loss_name="$1"
    local cvae_dir="$2"
    local eval_dir="$3"

    echo
    echo "[4/5] Standalone cVAE evaluation (${loss_name}) -> $eval_dir"
    python -m cvae.evaluate_generation \
        --joint_checkpoint "$TRANSFORMER_OUT_DIR/checkpoint.pt" \
        --cvae_checkpoint "$cvae_dir/checkpoint.pt" \
        --seen_embeddings "$TRANSFORMER_OUT_DIR/seen_embeddings.npz" \
        --heldout_embeddings "$TRANSFORMER_OUT_DIR/heldout_embeddings.npz" \
        --cvae_norm_stats "$cvae_dir/normalization_stats.npz" \
        --heldout_phase "$HELDOUT_PHASE" \
        --heldout_grip "$HELDOUT_GRIP" \
        --heldout_hand "$HELDOUT_HAND" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --out_dir "$eval_dir"
}

run_ablation_variant() {
    local loss_name="$1"
    local cvae_dir="$2"

    echo
    echo "[5/5] Latent ablation (${loss_name}) -> $cvae_dir"
    python -m cvae.latent_ablation_cvae \
        --run_dirs "$cvae_dir" \
        --data_dir "$DATA_DIR" \
        --input_mode "$INPUT_MODE" \
        --generation_seed "$SEED" \
        --device "$DEVICE"
}

if [[ "$RUN_CVAE" -eq 1 ]]; then
    if [[ "$RUN_CVAE_ELBO" -eq 1 ]]; then
        run_cvae_variant "elbo" "$CVAE_ELBO_OUT_DIR"
    fi
    if [[ "$RUN_CVAE_MMD" -eq 1 ]]; then
        run_cvae_variant "mmd" "$CVAE_MMD_OUT_DIR" --mmd_loss
    fi
fi

if [[ "$RUN_EVAL" -eq 1 ]]; then
    if [[ "$RUN_CVAE_ELBO" -eq 1 ]]; then
        run_eval_variant "elbo" "$CVAE_ELBO_OUT_DIR" "$EVAL_ELBO_OUT_DIR"
    fi
    if [[ "$RUN_CVAE_MMD" -eq 1 ]]; then
        run_eval_variant "mmd" "$CVAE_MMD_OUT_DIR" "$EVAL_MMD_OUT_DIR"
    fi
fi

if [[ "$RUN_ABLATION" -eq 1 ]]; then
    if [[ "$RUN_CVAE_ELBO" -eq 1 ]]; then
        run_ablation_variant "elbo" "$CVAE_ELBO_OUT_DIR"
    fi
    if [[ "$RUN_CVAE_MMD" -eq 1 ]]; then
        run_ablation_variant "mmd" "$CVAE_MMD_OUT_DIR"
    fi
fi

echo
echo "Pipeline finished: $(timestamp)"
echo "Linear:      $LINEAR_OUT_DIR"
echo "Transformer: $TRANSFORMER_OUT_DIR"
if [[ "$RUN_CVAE_ELBO" -eq 1 ]]; then
    echo "cVAE ELBO:   $CVAE_ELBO_OUT_DIR"
fi
if [[ "$RUN_CVAE_MMD" -eq 1 ]]; then
    echo "cVAE MMD:    $CVAE_MMD_OUT_DIR"
fi
if [[ "$RUN_EVAL" -eq 1 ]]; then
    if [[ "$RUN_CVAE_ELBO" -eq 1 ]]; then
        echo "Eval ELBO:   $EVAL_ELBO_OUT_DIR"
    fi
    if [[ "$RUN_CVAE_MMD" -eq 1 ]]; then
        echo "Eval MMD:    $EVAL_MMD_OUT_DIR"
    fi
fi
echo "Summary:     $RUN_SUMMARY_PATH"

{
    echo "timestamp_end: $(timestamp)"
    echo "status: completed"
} >> "$RUN_SUMMARY_PATH"
