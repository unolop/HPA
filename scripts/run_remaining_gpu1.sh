#!/bin/bash
# GPU 1: vicuna-7b blind
cd "$(dirname "$0")/.."
source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/remaining_gpu1.log"; mkdir -p logs
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

BB="evaluation/logits/backbone"
CACHE="/home/david/Desktop/yuna/.cache/hf"

log "=== run_remaining_gpu1.sh started ==="

OUT="$BB/pretrained/vicuna-7b-v1.5/vqa_1k_control_blind.jsonl"
if [ -f "$OUT" ] && [ "$(wc -l < $OUT)" -ge 1000 ]; then
    log "SKIP (done): vicuna-7b blind"
else
    log "▶ [GPU 1] vicuna-7b blind"
    CUDA_VISIBLE_DEVICES=1 HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    /home/david/miniconda3/envs/zero/bin/python evaluation/inference.py \
        --model "lmsys/vicuna-7b-v1.5" --model_type lm --dataset vqa_1k \
        --condition "_control_blind" --savedir "$BB" \
        --swift_model_type llama >> "$LOG" 2>&1 \
        && log "✓ vicuna-7b blind" || { log "✗ FAILED vicuna-7b blind"; exit 1; }
fi

log "=== run_remaining_gpu1.sh complete ==="
