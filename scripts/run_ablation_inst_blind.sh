#!/bin/bash
# Image-type ablation with instruction (inst_blind) for all 4 VLM families.
# Runs gray, white, noise on GPU 1 sequentially.
# Blank (default) inst_blind already exists for all models.
#
# Models:
#   - InternVL3.5-8B
#   - Qwen3-VL-8B
#   - LLaVA-Mistral-7B
#   - LLaVA-1.5-7B
#
# Output files:
#   evaluation/logits/vlm/pretrained/<model>/vqa_1k_control_inst_blind_{gray,white,noise}.jsonl
#
# Usage:
#   nohup bash scripts/run_ablation_inst_blind.sh > logs/ablation_inst_blind.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/ablation_inst_blind.log"
mkdir -p logs

PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
SDIR="evaluation/logits/vlm"
DATASET="vqa_1k_control"
COND="_inst_blind"
# GPU 0 (PCI 01:00.0) is down; address GPU 1 by PCI bus ID to bypass broken NVML
GPU_PCI="0000:02:00.0"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run_ablation() {
    local MODEL="$1" MODEL_SHORT="$2" IMG_TYPE="$3" EXTRA="$4"
    local OUT="$SDIR/pretrained/${MODEL_SHORT}/${DATASET}${COND}_${IMG_TYPE}.jsonl"
    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 1000 ]; then
        log "SKIP (complete): ${MODEL_SHORT} inst_blind image=$IMG_TYPE"; return
    fi
    log "▶ [GPU $GPU_PCI] ${MODEL_SHORT} condition=${COND} image_override=$IMG_TYPE"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_PCI" HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model "$MODEL" \
        --model_type vlm \
        --dataset "$DATASET" \
        --condition "$COND" \
        --savedir "$SDIR" \
        --image_override "$IMG_TYPE" \
        $EXTRA \
        >> "$LOG" 2>&1 \
        && log "✓ ${MODEL_SHORT} inst_blind image_override=$IMG_TYPE" \
        || { log "✗ FAILED ${MODEL_SHORT} inst_blind image_override=$IMG_TYPE"; }
}

log "=== run_ablation_inst_blind.sh started ==="

# ── InternVL3.5-8B ──────────────────────────────────────────────────────────
log "--- InternVL3.5-8B (inst_blind) ---"
run_ablation "OpenGVLab/InternVL3_5-8B" "InternVL3_5-8B" gray "--attn_impl eager"
run_ablation "OpenGVLab/InternVL3_5-8B" "InternVL3_5-8B" white "--attn_impl eager"
run_ablation "OpenGVLab/InternVL3_5-8B" "InternVL3_5-8B" noise "--attn_impl eager"

# ── Qwen3-VL-8B ────────────────────────────────────────────────────────────
log "--- Qwen3-VL-8B (inst_blind) ---"
run_ablation "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" gray "--template_type qwen3_nothinking"
run_ablation "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" white "--template_type qwen3_nothinking"
run_ablation "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" noise "--template_type qwen3_nothinking"

# ── LLaVA-Mistral-7B ───────────────────────────────────────────────────────
log "--- LLaVA-Mistral-7B (inst_blind) ---"
run_ablation "llava-hf/llava-v1.6-mistral-7b-hf" "llava-v1.6-mistral-7b-hf" gray ""
run_ablation "llava-hf/llava-v1.6-mistral-7b-hf" "llava-v1.6-mistral-7b-hf" white ""
run_ablation "llava-hf/llava-v1.6-mistral-7b-hf" "llava-v1.6-mistral-7b-hf" noise ""

# ── LLaVA-1.5-7B ───────────────────────────────────────────────────────────
log "--- LLaVA-1.5-7B (inst_blind) ---"
run_ablation "llava-hf/llava-1.5-7b-hf" "llava-1.5-7b-hf" gray ""
run_ablation "llava-hf/llava-1.5-7b-hf" "llava-1.5-7b-hf" white ""
run_ablation "llava-hf/llava-1.5-7b-hf" "llava-1.5-7b-hf" noise ""

log "=== run_ablation_inst_blind.sh complete ==="
