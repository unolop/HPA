#!/bin/bash
# Fill missing answers for Qwen3-VL-32B-Instruct (pretrained VLM tier)
# Uses --fill_missing to patch only the absent question_ids in each file.
# Run with: nohup bash scripts/run_fill_32b.sh > logs/fill_32b.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/fill_32b.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== fill_32b.sh started ==="

MODEL="Qwen/Qwen3-VL-32B-Instruct"
SAVEDIR="evaluation/logits/pretrained"
COMMON="--model $MODEL --model_type vlm --dataset vqa_1k \
        --savedir $SAVEDIR --template_type qwen3_nothinking --fill_missing"

# Order: most missing first to validate early
# 1. control (957 missing)
log "--- [1/4] _control (957 missing) ---"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    $COMMON --condition _control \
    >> "$LOG" 2>&1 \
    && log "✓ _control" || { log "✗ FAILED _control"; exit 1; }

# 2. original / vqa_1k with image (393 missing)
log "--- [2/4] no condition / vqa_1k.jsonl (393 missing) ---"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    $COMMON --condition "" \
    >> "$LOG" 2>&1 \
    && log "✓ original" || { log "✗ FAILED original"; exit 1; }

# 3. blind (112 missing)
log "--- [3/4] _control_blind (112 missing) ---"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    $COMMON --condition _control_blind \
    >> "$LOG" 2>&1 \
    && log "✓ _control_blind" || { log "✗ FAILED _control_blind"; exit 1; }

# 4. inst_blind (26 missing)
log "--- [4/4] _control_inst_blind (26 missing) ---"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    $COMMON --condition _control_inst_blind \
    >> "$LOG" 2>&1 \
    && log "✓ _control_inst_blind" || { log "✗ FAILED _control_inst_blind"; exit 1; }

log "=== fill_32b.sh done ==="
