#!/bin/bash
# Image-type ablation: run Qwen3-VL-4B on blank/gray/white/noise images.
#
# Goal: test whether model behaviour changes across uninformative image types.
# Uses plain vqa_1k + _blind (no control variants — degradation curve not needed here).
# Results go to evaluation/logits/ablation/ (separate from main results).
#
# Output files (one per image type):
#   evaluation/logits/ablation/pretrained/Qwen3-VL-4B-Instruct/vqa_1k_blind.jsonl       (blank)
#   evaluation/logits/ablation/pretrained/Qwen3-VL-4B-Instruct/vqa_1k_blind_gray.jsonl
#   evaluation/logits/ablation/pretrained/Qwen3-VL-4B-Instruct/vqa_1k_blind_white.jsonl
#   evaluation/logits/ablation/pretrained/Qwen3-VL-4B-Instruct/vqa_1k_blind_noise.jsonl
#
# Usage:
#   nohup bash scripts/run_ablation_image_type.sh > logs/ablation_image_type.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/ablation_image_type.log"
mkdir -p logs evaluation/logits/ablation

PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
MODEL="Qwen/Qwen3-VL-4B-Instruct"
SDIR="evaluation/logits/ablation"
DATASET="vqa_1k"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run_ablation() {
    local GPUS="$1" IMG_TYPE="$2"
    local SUFFIX=""
    [ "$IMG_TYPE" != "blank" ] && SUFFIX="_${IMG_TYPE}"
    local OUT="$SDIR/pretrained/Qwen3-VL-4B-Instruct/${DATASET}_blind${SUFFIX}.jsonl"
    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 1000 ]; then
        log "SKIP (complete): image=$IMG_TYPE"; return
    fi
    log "▶ [GPU $GPUS] image_override=$IMG_TYPE"
    CUDA_VISIBLE_DEVICES="$GPUS" HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model "$MODEL" --model_type vlm \
        --dataset "$DATASET" --condition _blind \
        --savedir "$SDIR" \
        --image_override "$IMG_TYPE" \
        --template_type qwen3_nothinking \
        >> "$LOG" 2>&1 \
        && log "✓ image_override=$IMG_TYPE" \
        || { log "✗ FAILED image_override=$IMG_TYPE"; exit 1; }
}

log "=== run_ablation_image_type.sh started (Qwen3-VL-4B) ==="

run_ablation 0 blank
run_ablation 0 gray
run_ablation 0 white
run_ablation 0 noise

log "=== run_ablation_image_type.sh complete ==="
