#!/bin/bash
# run_control_full.sh
# Runs all missing vqa_1k_control inference and fills pronominalized.
# InternVL excluded (too large for 2×TITAN RTX 24GB).
#
# Models covered:
#   Existing (need control blind / completion):
#     Qwen3-VL-2B/4B/8B-Instruct, LLaVA-1.5-7B, LLaVA-1.6-Mistral-7B,
#     LLaVA-1.6-Vicuna-7B, LLaVA-1.6-Vicuna-13B (control_blind missing)
#   New additions (all conditions):
#     Molmo-7B-D, Llama-3.2-11B-Vision, Idefics3-8B, MiniCPM-V-2_6
#
# Phases:
#   1. GPT extraction — adds 'pronominalized' to vqa1k_control.jsonl
#   2. Missing control inference (sequential per model)
#   3. Fill pronominalized into all existing control logit files
#
# Usage:
#   bash scripts/run_control_full.sh            # full run
#   bash scripts/run_control_full.sh --skip-gpt # skip GPT extraction (already done)
#   bash scripts/run_control_full.sh --phase 2  # only phase 2
#   bash scripts/run_control_full.sh --phase 3  # only phase 3 (pronominalized fill)

set -e
cd "$(dirname "$0")/.."

GPU="0,1"
LOG_DIR="logs/control_inference"
mkdir -p "$LOG_DIR"

SKIP_GPT=false
PHASE_ONLY=""
for arg in "$@"; do
    case $arg in
        --skip-gpt) SKIP_GPT=true ;;
        --phase=*) PHASE_ONLY="${arg#*=}" ;;
        --phase) shift; PHASE_ONLY="$1" ;;
    esac
done

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/run.log"; }

run_inference() {
    local MODEL=$1
    local CONDITION=$2
    local EXTRA_ARGS="${3:-}"
    local MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
    local LOGFILE="$LOG_DIR/${MODEL_SHORT}${CONDITION//\//_}.log"

    log "▶ $MODEL_SHORT | condition=$CONDITION $EXTRA_ARGS"
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" \
        --condition "$CONDITION" \
        --dataset "vqa_1k" \
        $EXTRA_ARGS \
        > "$LOGFILE" 2>&1
    if [ $? -eq 0 ]; then
        log "✓ $MODEL_SHORT | condition=$CONDITION"
    else
        log "✗ FAILED $MODEL_SHORT | condition=$CONDITION — see $LOGFILE"
    fi
}

run_fill() {
    local MODEL=$1
    local CONDITION=$2
    local MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
    local LOGFILE="$LOG_DIR/${MODEL_SHORT}${CONDITION//\//_}_fill_pron.log"

    log "▶ fill pronominalized: $MODEL_SHORT | condition=$CONDITION"
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" \
        --condition "$CONDITION" \
        --dataset "vqa_1k" \
        --fill_missing \
        --control_types "pronominalized" \
        > "$LOGFILE" 2>&1
    if [ $? -eq 0 ]; then
        log "✓ fill pron: $MODEL_SHORT | condition=$CONDITION"
    else
        log "✗ FAILED fill pron: $MODEL_SHORT | condition=$CONDITION — see $LOGFILE"
    fi
}

log "=== run_control_full.sh started ==="
log "GPU: $GPU | skip-gpt: $SKIP_GPT | phase: ${PHASE_ONLY:-all}"
if [ -z "$PHASE_ONLY" ] || [ "$PHASE_ONLY" = "2" ]; then
    log "=== PHASE 2: Missing control inference ==="

    # ── Existing models: fill gaps ────────────────────────────────────────────

    # Qwen3-VL-2B — all three conditions missing
    run_inference "Qwen/Qwen3-VL-2B-Instruct" "_control"
    run_inference "Qwen/Qwen3-VL-2B-Instruct" "_control_blind"
    run_inference "Qwen/Qwen3-VL-2B-Instruct" "_control_inst_blind"

    # Qwen3-VL-4B — control and control_blind missing (inst_blind done)
    run_inference "Qwen/Qwen3-VL-4B-Instruct" "_control"
    run_inference "Qwen/Qwen3-VL-4B-Instruct" "_control_blind"

    # Qwen3-VL-8B — inst_blind incomplete (994/1000), resume it
    run_inference "Qwen/Qwen3-VL-8B-Instruct" "_control_inst_blind" "--resume"

    # LLaVA-1.6-Vicuna-13B — control and inst_blind exist, blind missing
    run_inference "llava-hf/llava-v1.6-vicuna-13b-hf" "_control_blind"

    # ── New model family 1: Molmo-7B-D (AllenAI) ~15GB ───────────────────────
    # Qwen2 backbone; architecturally distinct from all current models
    run_inference "allenai/Molmo-7B-D-0924" "_control"
    run_inference "allenai/Molmo-7B-D-0924" "_control_blind"
    run_inference "allenai/Molmo-7B-D-0924" "_control_inst_blind"

    # ── New model family 2: Llama-3.2-11B-Vision (Meta) ~22GB ────────────────
    # Cross-attention visual adapter; decouples language from vision
    run_inference "meta-llama/Llama-3.2-11B-Vision-Instruct" "_control"
    run_inference "meta-llama/Llama-3.2-11B-Vision-Instruct" "_control_blind"
    run_inference "meta-llama/Llama-3.2-11B-Vision-Instruct" "_control_inst_blind"

    # ── New model family 3: Idefics3-8B (HuggingFace) ~17GB ──────────────────
    # Llama-3-8B backbone; minimal visual processing → stronger language prior
    run_inference "HuggingFaceM4/Idefics3-8B-Llama3" "_control"
    run_inference "HuggingFaceM4/Idefics3-8B-Llama3" "_control_blind"
    run_inference "HuggingFaceM4/Idefics3-8B-Llama3" "_control_inst_blind"

    # ── New model family 4: MiniCPM-V-2_6 (OpenBMB) ~17GB ────────────────────
    # Qwen2-7B backbone + compressed resampler; efficiency-focused design
    run_inference "openbmb/MiniCPM-V-2_6" "_control"
    run_inference "openbmb/MiniCPM-V-2_6" "_control_blind"
    run_inference "openbmb/MiniCPM-V-2_6" "_control_inst_blind"

    log "=== PHASE 2 complete ==="
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: Fill pronominalized into all control files
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "$PHASE_ONLY" ] || [ "$PHASE_ONLY" = "3" ]; then
    log "=== PHASE 3: Fill pronominalized ==="

    ALL_MODELS=(
        # Existing LLaVA family
        "llava-hf/llava-1.5-7b-hf"
        "llava-hf/llava-v1.6-mistral-7b-hf"
        "llava-hf/llava-v1.6-vicuna-7b-hf"
        "llava-hf/llava-v1.6-vicuna-13b-hf"
        # Existing Qwen3-VL family
        "Qwen/Qwen3-VL-2B-Instruct"
        "Qwen/Qwen3-VL-4B-Instruct"
        "Qwen/Qwen3-VL-8B-Instruct"
        # New models
        "allenai/Molmo-7B-D-0924"
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
        "HuggingFaceM4/Idefics3-8B-Llama3"
        "openbmb/MiniCPM-V-2_6"
    )

    for MODEL in "${ALL_MODELS[@]}"; do
        run_fill "$MODEL" "_control"
        run_fill "$MODEL" "_control_blind"
        run_fill "$MODEL" "_control_inst_blind"
    done

    log "=== PHASE 3 complete ==="
fi

log "=== run_control_full.sh finished ==="
