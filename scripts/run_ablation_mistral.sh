#!/bin/bash
# Image-type ablation for LLaVA-Mistral-7B on vqa_1k_control _blind (variant C).
# Runs gray, white, noise in sequence on GPU 0.
#
# Output files:
#   evaluation/logits/vlm/pretrained/llava-v1.6-mistral-7b-hf/vqa_1k_control_blind_gray.jsonl
#   evaluation/logits/vlm/pretrained/llava-v1.6-mistral-7b-hf/vqa_1k_control_blind_white.jsonl
#   evaluation/logits/vlm/pretrained/llava-v1.6-mistral-7b-hf/vqa_1k_control_blind_noise.jsonl
#
# Usage:
#   nohup bash scripts/run_ablation_mistral.sh > logs/ablation_mistral.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/ablation_mistral.log"
mkdir -p logs

PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
MODEL="llava-hf/llava-v1.6-mistral-7b-hf"
SDIR="evaluation/logits/vlm"
DATASET="vqa_1k_control"
COND="_blind"
GPU=0

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run_ablation() {
    local IMG_TYPE="$1"
    local OUT="$SDIR/pretrained/llava-v1.6-mistral-7b-hf/${DATASET}${COND}_${IMG_TYPE}.jsonl"
    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 1000 ]; then
        log "SKIP (complete): image=$IMG_TYPE"; return
    fi
    log "▶ [GPU $GPU] image_override=$IMG_TYPE"
    CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model "$MODEL" \
        --model_type vlm \
        --dataset "$DATASET" \
        --condition "$COND" \
        --savedir "$SDIR" \
        --image_override "$IMG_TYPE" \
        --quantization_bit 8 \
        >> "$LOG" 2>&1 \
        && log "✓ image_override=$IMG_TYPE" \
        || { log "✗ FAILED image_override=$IMG_TYPE"; exit 1; }
}

log "=== run_ablation_mistral.sh started (LLaVA-Mistral-7B) ==="

run_ablation gray
run_ablation white
run_ablation noise

log "=== run_ablation_mistral.sh complete ==="
