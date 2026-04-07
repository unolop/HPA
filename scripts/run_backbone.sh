#!/bin/bash
# run_backbone.sh — Run standalone LLM backbone models (text-only) on control questions.
#
# Each backbone is the LLM component used to build a VLM:
#
#   Backbone model              HF ID                               VLM(s) that use it
#   ─────────────────────────────────────────────────────────────────────────────────
#   Qwen3-0.6B                  Qwen/Qwen3-0.6B                     InternVL3.5-1B
#   Qwen3-1.7B                  Qwen/Qwen3-1.7B                     InternVL3.5-2B, Qwen3-VL-2B
#   Qwen3-4B                    Qwen/Qwen3-4B                       Qwen3-VL-4B
#   Qwen3-8B                    Qwen/Qwen3-8B                       InternVL3.5-8B, Qwen3-VL-8B
#   Qwen3-32B                   Qwen/Qwen3-32B                      Qwen3-VL-32B
#   vicuna-7b-v1.5              lmsys/vicuna-7b-v1.5                llava-1.5-7b, llava-v1.6-vicuna-7b
#   vicuna-13b-v1.5             lmsys/vicuna-13b-v1.5               llava-v1.6-vicuna-13b
#   Mistral-7B-Instruct-v0.2    mistralai/Mistral-7B-Instruct-v0.2  llava-v1.6-mistral-7b
#
# Results go to evaluation/logits/backbone/ (distinct from lm_decoder and pretrained).
#
# GPU 0: vicuna-7b, Mistral-7B, Qwen3-0.6B, Qwen3-1.7B
# GPU 1: Qwen3-4B, Qwen3-8B
# Serial (both GPUs): vicuna-13b, Qwen3-32B
#
# Usage:
#   bash scripts/run_backbone.sh           # full parallel run
#   bash scripts/run_backbone.sh --gpu0    # only GPU 0 worker
#   bash scripts/run_backbone.sh --gpu1    # only GPU 1 worker
#   bash scripts/run_backbone.sh --serial  # only large models (both GPUs)

set -e
cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG_DIR="logs/backbone"
SAVEDIR="evaluation/logits/backbone"
mkdir -p "$LOG_DIR"

ONLY=""
for arg in "$@"; do
    case $arg in
        --gpu0)   ONLY="gpu0"   ;;
        --gpu1)   ONLY="gpu1"   ;;
        --serial) ONLY="serial" ;;
    esac
done

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/run.log"; }

run_backbone() {
    local GPU=$1; local MODEL=$2; local CONDITION=$3
    local SHORT=$(echo "$MODEL" | sed 's|.*/||')
    local LOGFILE="$LOG_DIR/${SHORT}${CONDITION}.log"
    local EXTRA=""
    # Disable chain-of-thought for Qwen3 backbone models
    [[ "$MODEL" == *"Qwen3"* ]] && EXTRA="--template_type qwen3_nothinking"
    # Swift can't auto-detect model_type for non-VLM models — pass explicitly
    [[ "$MODEL" == *"vicuna"* ]]  && EXTRA="$EXTRA --swift_model_type llama"
    [[ "$MODEL" == *"Mistral"* ]] && EXTRA="$EXTRA --swift_model_type mistral"
    log "▶ [GPU$GPU] $SHORT | $CONDITION $EXTRA"
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" \
        --model_type lm \
        --condition "$CONDITION" \
        --dataset "vqa_1k" \
        --savedir "$SAVEDIR" \
        --resume \
        $EXTRA \
        > "$LOGFILE" 2>&1 \
        && log "✓ [GPU$GPU] $SHORT | $CONDITION" \
        || { log "✗ FAILED [GPU$GPU] $SHORT | $CONDITION — see $LOGFILE"; return 1; }
}

# ── GPU 0: small/medium LLM backbones ────────────────────────────────────────
worker_gpu0() {
    log "=== GPU 0: vicuna-7b, Mistral-7B, Qwen3-0.6B, Qwen3-1.7B ==="
    for MODEL in \
        "lmsys/vicuna-7b-v1.5" \
        "mistralai/Mistral-7B-Instruct-v0.2" \
        "Qwen/Qwen3-0.6B" \
        "Qwen/Qwen3-1.7B"; do
        for COND in _control _control_blind _control_inst_blind; do
            run_backbone 0 "$MODEL" "$COND"
        done
    done
    log "=== GPU 0 done ==="
}

# ── GPU 1: medium Qwen3 backbones ─────────────────────────────────────────────
worker_gpu1() {
    log "=== GPU 1: Qwen3-4B, Qwen3-8B ==="
    for MODEL in \
        "Qwen/Qwen3-4B" \
        "Qwen/Qwen3-8B"; do
        for COND in _control _control_blind _control_inst_blind; do
            run_backbone 1 "$MODEL" "$COND"
        done
    done
    log "=== GPU 1 done ==="
}

# ── Serial: large backbones needing both GPUs ─────────────────────────────────
worker_serial() {
    log "=== Serial: vicuna-13b, Qwen3-32B (GPU 0,1) ==="
    for MODEL in \
        "lmsys/vicuna-13b-v1.5" \
        "Qwen/Qwen3-32B"; do
        for COND in _control _control_blind _control_inst_blind; do
            SHORT=$(echo "$MODEL" | sed 's|.*/||')
            LOGFILE="$LOG_DIR/${SHORT}${COND}.log"
            log "▶ [GPU0,1] $SHORT | $COND"
            EXTRA_SERIAL=""
            [[ "$MODEL" == *"Qwen3"* ]]   && EXTRA_SERIAL="--template_type qwen3_nothinking"
            [[ "$MODEL" == *"vicuna"* ]]   && EXTRA_SERIAL="$EXTRA_SERIAL --swift_model_type llama"
            CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
                --model "$MODEL" \
                --model_type lm \
                --condition "$COND" \
                --dataset "vqa_1k" \
                --savedir "$SAVEDIR" \
                --resume \
                $EXTRA_SERIAL \
                > "$LOGFILE" 2>&1 \
                && log "✓ [GPU0,1] $SHORT | $COND" \
                || log "✗ FAILED [GPU0,1] $SHORT | $COND — see $LOGFILE"
        done
    done
    log "=== Serial done ==="
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "=== run_backbone.sh started (only=${ONLY:-all}) ==="

case "$ONLY" in
    gpu0)   worker_gpu0 ;;
    gpu1)   worker_gpu1 ;;
    serial) worker_serial ;;
    *)
        worker_gpu0 &
        PID0=$!
        worker_gpu1 &
        PID1=$!
        log "GPU 0 PID=$PID0 | GPU 1 PID=$PID1"
        wait $PID0 && log "✓ GPU 0 done" || log "✗ GPU 0 FAILED"
        wait $PID1 && log "✓ GPU 1 done" || log "✗ GPU 1 FAILED"
        worker_serial
        ;;
esac

log "=== run_backbone.sh finished ==="
