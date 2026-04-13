#!/bin/bash
# Resume all remaining runs: lm_decoder 32B → backbone Qwen3-32B inst_blind → push condition
# All use --resume so already-processed questions are skipped.
# Run with: nohup bash scripts/run_remaining.sh > logs/remaining_run.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/remaining_run.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== run_remaining.sh started ==="

# 1. lm_decoder: Qwen3-VL-32B blind + inst_blind (resumes from 330/1000)
log "--- lm_decoder Qwen3-VL-32B ---"
for COND in _control_blind _control_inst_blind; do
    log "▶ [GPU0,1] Qwen3-VL-32B-Instruct | $COND"
    CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
        --model "Qwen/Qwen3-VL-32B-Instruct" --model_type lm \
        --condition "$COND" --dataset "vqa_1k" \
        --savedir "evaluation/logits/lm_decoder" --resume \
        >> "$LOG" 2>&1 \
        && log "✓ Qwen3-VL-32B $COND" || log "✗ FAILED Qwen3-VL-32B $COND"
done

# 2. backbone: Qwen3-32B inst_blind only
log "--- backbone Qwen3-32B inst_blind ---"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "Qwen/Qwen3-32B" --model_type lm \
    --condition "_control_inst_blind" --dataset "vqa_1k" \
    --savedir "evaluation/logits/backbone" --resume \
    --template_type qwen3_nothinking \
    >> "$LOG" 2>&1 \
    && log "✓ Qwen3-32B inst_blind" || log "✗ FAILED Qwen3-32B inst_blind"

# 3. push condition for all models
log "--- push condition ---"
bash scripts/run_push_condition.sh >> "$LOG" 2>&1 \
    && log "✓ push done" || log "✗ push FAILED"

log "=== run_remaining.sh done ==="
