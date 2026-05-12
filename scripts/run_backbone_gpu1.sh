#!/bin/bash
# GPU 1: vicuna-7b — blind + inst_blind
#
# Run alongside run_backbone_gpu0.sh:
#   nohup bash scripts/run_backbone_gpu0.sh > logs/backbone_gpu0.log 2>&1 &
#   nohup bash scripts/run_backbone_gpu1.sh > logs/backbone_gpu1.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/backbone_gpu1.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== run_backbone_gpu1.sh started ==="

HF_CACHE="/home/david/Desktop/yuna/.cache/hf"
BB_DIR="evaluation/logits/backbone"

run_bb() {
    local GPUS="$1" MODEL_ID="$2" SAVE_NAME="$3" COND="$4"
    shift 4
    log "▶ [GPU $GPUS] $SAVE_NAME | $COND"
    CUDA_VISIBLE_DEVICES="$GPUS" \
    TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" \
    python evaluation/inference.py \
        --model      "$MODEL_ID" \
        --model_type lm \
        --dataset    vqa_1k \
        --condition  "$COND" \
        --savedir    "$BB_DIR" \
        "$@" \
        >> "$LOG" 2>&1 \
        && log "✓ $SAVE_NAME | $COND" \
        || { log "✗ FAILED $SAVE_NAME | $COND"; exit 1; }
}

# ── vicuna-7b (GPU 1 only) ────────────────────────────────────────────────────
log "=== vicuna-7b-v1.5 blind + inst_blind (GPU 1) ==="

for COND in _control_blind _control_inst_blind; do
    run_bb 1 "lmsys/vicuna-7b-v1.5" "vicuna-7b-v1.5" "$COND" \
        --swift_model_type llama
done

log "=== run_backbone_gpu1.sh complete ==="
log "    GPU 1 is now free. vicuna-13b and Qwen3-32B in gpu0 script will use both GPUs next."
