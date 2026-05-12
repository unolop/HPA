#!/bin/bash
# GPU 1: vicuna-7b inst_blind (runs in parallel with gpu0 Qwen3 small models)
#
#   nohup bash scripts/run_inst_blind_gpu0.sh > logs/inst_blind_gpu0.log 2>&1 &
#   nohup bash scripts/run_inst_blind_gpu1.sh > logs/inst_blind_gpu1.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/inst_blind_gpu1.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

BB_DIR="evaluation/logits/backbone"
HF_CACHE="/home/david/Desktop/yuna/.cache/hf"
COND="_control_inst_blind"

log "=== run_inst_blind_gpu1.sh started ==="

OUT="$BB_DIR/pretrained/vicuna-7b-v1.5/vqa_1k${COND}.jsonl"
if [ -f "$OUT" ] && [ $(wc -l < "$OUT") -ge 1000 ]; then
    log "SKIP (done): vicuna-7b-v1.5"
else
    log "▶ [GPU 1] vicuna-7b-v1.5 | $COND"
    CUDA_VISIBLE_DEVICES=1 \
    TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" \
    python evaluation/inference.py \
        --model      "lmsys/vicuna-7b-v1.5" \
        --model_type lm \
        --dataset    vqa_1k \
        --condition  "$COND" \
        --savedir    "$BB_DIR" \
        --swift_model_type llama \
        >> "$LOG" 2>&1 \
        && log "✓ vicuna-7b-v1.5" \
        || { log "✗ FAILED vicuna-7b-v1.5"; exit 1; }
fi

log "=== run_inst_blind_gpu1.sh complete — GPU 1 now free for gpu0 large models ==="
