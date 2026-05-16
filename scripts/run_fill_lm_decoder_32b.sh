#!/bin/bash
# Fill the single remaining missing file:
#   lm_decoder/Qwen3-VL-32B-Instruct / vqa_1k_control_inst_blind  (0/1000)
#
# Uses both GPUs (32B model needs ~60GB).
# Run: nohup bash scripts/run_fill_lm_decoder_32b.sh > logs/fill_lm_decoder_32b.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/fill_lm_decoder_32b.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== run_fill_lm_decoder_32b.sh started ==="

CUDA_VISIBLE_DEVICES=0,1 \
    python evaluation/inference.py \
        --model       "Qwen/Qwen3-VL-32B-Instruct" \
        --model_type  lm \
        --dataset     vqa_1k \
        --condition   _control_inst_blind \
        --savedir     "evaluation/logits/lm_decoder" \
        --template_type qwen3_nothinking \
        >> "$LOG" 2>&1 \
    && log "✓ lm_decoder Qwen3-VL-32B-Instruct | _control_inst_blind" \
    || log "✗ FAILED lm_decoder Qwen3-VL-32B-Instruct | _control_inst_blind"

log "=== done ==="
