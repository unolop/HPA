#!/bin/bash
# GPU 1: Qwen2.5-7B-Instruct — blind + inst_blind
cd "$(dirname "$0")/.."

LOG="logs/qwen25_gpu1.log"; mkdir -p logs
PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
SAVEDIR="evaluation/logits/backbone_nothink"
MODEL="Qwen/Qwen2.5-7B-Instruct"
SAVE="Qwen2.5-7B-Instruct"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run() {
    local COND="$1"
    local OUT="$SAVEDIR/pretrained/$SAVE/vqa_1k${COND}.jsonl"
    if [ -f "$OUT" ] && [ "$(wc -l < $OUT)" -ge 1000 ]; then
        log "SKIP (done): $SAVE $COND"; return
    fi
    log "▶ [GPU 1] $SAVE $COND"
    mkdir -p "$SAVEDIR/pretrained/$SAVE"
    CUDA_VISIBLE_DEVICES=1 HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model "$MODEL" --model_type lm --dataset vqa_1k \
        --condition "$COND" --savedir "$SAVEDIR" \
        >> "$LOG" 2>&1 \
        && log "✓ $SAVE $COND" \
        || { log "✗ FAILED $SAVE $COND"; exit 1; }
}

log "=== run_qwen25_gpu1.sh started ==="
run "_control_blind"
run "_control_inst_blind"
log "=== run_qwen25_gpu1.sh complete ==="
