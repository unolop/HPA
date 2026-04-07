#!/bin/bash
# run_lm_decoder.sh — Run all VLMs in text-only (decoder-only) mode.
#
# Uses --model_type lm: full VLM is loaded but no image is passed.
# Results go to evaluation/logits/lm_decoder/ (separate from VLM results).
#
# GPU 0: LLaVA family (llava-1.5-7b, mistral-7b, vicuna-7b)
# GPU 1: Qwen3-VL (2B, 4B, 8B) + InternVL3.5 (1B, 2B, 8B)
# Serial (both GPUs): llava-vicuna-13b, Qwen3-VL-32B
#
# Usage:
#   bash scripts/run_lm_decoder.sh           # full parallel run
#   bash scripts/run_lm_decoder.sh --gpu0    # only GPU 0 worker
#   bash scripts/run_lm_decoder.sh --gpu1    # only GPU 1 worker
#   bash scripts/run_lm_decoder.sh --serial  # only large models (both GPUs)

set -e
cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG_DIR="logs/lm_decoder"
SAVEDIR="evaluation/logits/lm_decoder"
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

run_lm() {
    local GPU=$1; local MODEL=$2; local CONDITION=$3; local EXTRA="${4:-}"
    local SHORT=$(echo "$MODEL" | sed 's|.*/||')
    local LOGFILE="$LOG_DIR/${SHORT}${CONDITION}.log"
    log "▶ [GPU$GPU] $SHORT | $CONDITION"
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

# ── GPU 0: LLaVA family ───────────────────────────────────────────────────────
worker_gpu0() {
    log "=== GPU 0: LLaVA family ==="
    for MODEL in \
        "llava-hf/llava-1.5-7b-hf" \
        "llava-hf/llava-v1.6-mistral-7b-hf" \
        "llava-hf/llava-v1.6-vicuna-7b-hf"; do
        for COND in _control _control_blind _control_inst_blind; do
            run_lm 0 "$MODEL" "$COND"
        done
    done
    log "=== GPU 0 done ==="
}

# ── GPU 1: Qwen3-VL + InternVL3.5 (small/medium) ────────────────────────────
worker_gpu1() {
    log "=== GPU 1: Qwen3-VL + InternVL3.5 ==="
    for MODEL in \
        "Qwen/Qwen3-VL-2B-Instruct" \
        "Qwen/Qwen3-VL-4B-Instruct" \
        "Qwen/Qwen3-VL-8B-Instruct" \
        "OpenGVLab/InternVL3_5-1B" \
        "OpenGVLab/InternVL3_5-2B" \
        "OpenGVLab/InternVL3_5-8B"; do
        EXTRA=""
        [[ "$MODEL" == *"InternVL"* ]] && EXTRA="--attn_impl eager"
        for COND in _control _control_blind _control_inst_blind; do
            run_lm 1 "$MODEL" "$COND" "$EXTRA"
        done
    done
    log "=== GPU 1 done ==="
}

# ── Serial: large models needing both GPUs ────────────────────────────────────
worker_serial() {
    log "=== Serial: large models (GPU 0,1) ==="
    for MODEL in \
        "llava-hf/llava-v1.6-vicuna-13b-hf" \
        "Qwen/Qwen3-VL-32B-Instruct"; do
        for COND in _control _control_blind _control_inst_blind; do
            SHORT=$(echo "$MODEL" | sed 's|.*/||')
            LOGFILE="$LOG_DIR/${SHORT}${COND}.log"
            log "▶ [GPU0,1] $SHORT | $COND"
            CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
                --model "$MODEL" \
                --model_type lm \
                --condition "$COND" \
                --dataset "vqa_1k" \
                --savedir "$SAVEDIR" \
                --resume \
                > "$LOGFILE" 2>&1 \
                && log "✓ [GPU0,1] $SHORT | $COND" \
                || log "✗ FAILED [GPU0,1] $SHORT | $COND — see $LOGFILE"
        done
    done
    log "=== Serial done ==="
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "=== run_lm_decoder.sh started (only=${ONLY:-all}) ==="

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

log "=== run_lm_decoder.sh finished ==="
