#!/bin/bash
cd "$(dirname "$0")/.."
source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache"
DATASET="vqa_1k"
LOG="logs/run_qwen25_base.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== run_qwen25_base.sh started ==="

log "=== [1/2] Qwen2.5-7B blind ==="
CUDA_VISIBLE_DEVICES=0 HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
$PYTHON evaluation/inference.py \
    --model "Qwen/Qwen2.5-7B" --model_type lm \
    --dataset "$DATASET" --condition "_control_blind" \
    --savedir "evaluation/logits/backbone" \
    >> "$LOG" 2>&1 \
    && log "✓ Qwen2.5-7B blind" \
    || { log "✗ FAILED Qwen2.5-7B blind"; exit 1; }

log "=== [2/2] Qwen2.5-7B inst_blind ==="
CUDA_VISIBLE_DEVICES=0 HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
$PYTHON evaluation/inference.py \
    --model "Qwen/Qwen2.5-7B" --model_type lm \
    --dataset "$DATASET" --condition "_control_inst_blind" \
    --savedir "evaluation/logits/backbone" \
    >> "$LOG" 2>&1 \
    && log "✓ Qwen2.5-7B inst_blind" \
    || { log "✗ FAILED Qwen2.5-7B inst_blind"; exit 1; }

log "=== run_qwen25_base.sh complete ==="
