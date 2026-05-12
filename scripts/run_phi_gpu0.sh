#!/bin/bash
# GPU 0: Phi-3.5-mini-instruct — blind + inst_blind
cd "$(dirname "$0")/.."

LOG="logs/phi_gpu0.log"; mkdir -p logs
PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
SAVEDIR="evaluation/logits/backbone_nothink"
MODEL="microsoft/Phi-3.5-mini-instruct"
SAVE="Phi-3.5-mini-instruct"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run() {
    local COND="$1"
    local OUT="$SAVEDIR/pretrained/$SAVE/vqa_1k${COND}.jsonl"
    if [ -f "$OUT" ] && [ "$(wc -l < $OUT)" -ge 1000 ]; then
        log "SKIP (done): $SAVE $COND"; return
    fi
    log "▶ [GPU 0] $SAVE $COND"
    mkdir -p "$SAVEDIR/pretrained/$SAVE"
    CUDA_VISIBLE_DEVICES=0 HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model "$MODEL" --model_type lm --dataset vqa_1k \
        --condition "$COND" --savedir "$SAVEDIR" \
        >> "$LOG" 2>&1 \
        && log "✓ $SAVE $COND" \
        || { log "✗ FAILED $SAVE $COND"; exit 1; }
}

log "=== run_phi_gpu0.sh started ==="
run "_control_blind"
run "_control_inst_blind"
log "=== run_phi_gpu0.sh complete ==="
