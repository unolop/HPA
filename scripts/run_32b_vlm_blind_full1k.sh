#!/bin/bash
# Run Qwen3-VL-32B VLM blind inference on the full 1000-question dataset.
# Uses --fill_missing to skip the 113 study questions already done,
# adding the remaining ~887 questions using 4-bit quantization.
#
# Usage:
#   nohup bash scripts/run_32b_vlm_blind_full1k.sh >> logs/run_32b_vlm_blind_full1k.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/run_32b_vlm_blind_full1k.log"
PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()       { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== VLM blind inference on full 1k (4-bit, fill missing) ==="
log "Current entries in output JSONL: $(wc -l < evaluation/logits/vlm/pretrained/Qwen3-VL-32B-Instruct/vqa_1k_control_blind.jsonl 2>/dev/null || echo 0)"

CUDA_VISIBLE_DEVICES=0,1 HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
$PYTHON evaluation/inference.py \
    --model "Qwen/Qwen3-VL-32B-Instruct" \
    --model_type vlm \
    --dataset vqa_1k \
    --condition _control_blind \
    --savedir "evaluation/logits/vlm" \
    --template_type qwen3_nothinking \
    --fill_missing \
    --quantization_bit 4 \
    2>&1 | tee -a "$LOG"

log "=== Done. Final line count: $(wc -l < evaluation/logits/vlm/pretrained/Qwen3-VL-32B-Instruct/vqa_1k_control_blind.jsonl) ==="
