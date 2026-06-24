#!/bin/bash
# Image-type ablation for InternVL3.5-8B and LLaVA-1.5-7B
# Runs gray, white, noise in sequence on GPU 0 (24GB TITAN RTX, fp16).
#
# These extend the image ablation study to cover all 3 VLM families:
#   - Qwen3-VL (already done: 4B, 8B)
#   - LLaVA-Mistral (already done)
#   - InternVL (NEW)
#   - LLaVA-1.5 (NEW)
#
# Output files:
#   evaluation/logits/vlm/pretrained/InternVL3_5-8B/vqa_1k_control_blind_{gray,white,noise}.jsonl
#   evaluation/logits/vlm/pretrained/llava-1.5-7b-hf/vqa_1k_control_blind_{gray,white,noise}.jsonl
#
# Usage:
#   nohup bash scripts/run_ablation_internvl_llava15.sh > logs/ablation_internvl_llava15.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/ablation_internvl_llava15.log"
mkdir -p logs

PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
SDIR="evaluation/logits/vlm"
DATASET="vqa_1k_control"
COND="_blind"
GPU=0

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run_ablation() {
    local MODEL="$1" MODEL_SHORT="$2" IMG_TYPE="$3" EXTRA="$4"
    local OUT="$SDIR/pretrained/${MODEL_SHORT}/${DATASET}${COND}_${IMG_TYPE}.jsonl"
    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 1000 ]; then
        log "SKIP (complete): ${MODEL_SHORT} image=$IMG_TYPE"; return
    fi
    log "▶ [GPU $GPU] ${MODEL_SHORT} image_override=$IMG_TYPE"
    CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model "$MODEL" \
        --model_type vlm \
        --dataset "$DATASET" \
        --condition "$COND" \
        --savedir "$SDIR" \
        --image_override "$IMG_TYPE" \
        $EXTRA \
        >> "$LOG" 2>&1 \
        && log "✓ ${MODEL_SHORT} image_override=$IMG_TYPE" \
        || { log "✗ FAILED ${MODEL_SHORT} image_override=$IMG_TYPE"; }
}

log "=== run_ablation_internvl_llava15.sh started ==="

# ── InternVL3.5-8B ──────────────────────────────────────────────────────────
log "--- InternVL3.5-8B ---"
run_ablation "OpenGVLab/InternVL3_5-8B" "InternVL3_5-8B" gray "--attn_impl eager"
run_ablation "OpenGVLab/InternVL3_5-8B" "InternVL3_5-8B" white "--attn_impl eager"
run_ablation "OpenGVLab/InternVL3_5-8B" "InternVL3_5-8B" noise "--attn_impl eager"

# ── LLaVA-1.5-7B ───────────────────────────────────────────────────────────
log "--- LLaVA-1.5-7B ---"
run_ablation "llava-hf/llava-1.5-7b-hf" "llava-1.5-7b-hf" gray ""
run_ablation "llava-hf/llava-1.5-7b-hf" "llava-1.5-7b-hf" white ""
run_ablation "llava-hf/llava-1.5-7b-hf" "llava-1.5-7b-hf" noise ""

log "=== run_ablation_internvl_llava15.sh complete ==="
