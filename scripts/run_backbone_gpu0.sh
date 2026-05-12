#!/bin/bash
# GPU 0: Qwen3 small backbone (0.6B, 1.7B, 4B, 8B) — nothinking, blind + inst_blind
# Then GPU 0+1: vicuna-13b + Qwen3-32B (large, need both GPUs) once gpu1 script finishes.
#
# Run alongside run_backbone_gpu1.sh:
#   nohup bash scripts/run_backbone_gpu0.sh > logs/backbone_gpu0.log 2>&1 &
#   nohup bash scripts/run_backbone_gpu1.sh > logs/backbone_gpu1.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/backbone_gpu0.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== run_backbone_gpu0.sh started ==="

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

# ── §1  Qwen3 small — nothinking re-run (GPU 0 only) ─────────────────────────
log "=== [1/3] Qwen3 0.6B → 8B nothinking (GPU 0) ==="

for SIZE in 0.6B 1.7B 4B 8B; do
    for COND in _control_blind _control_inst_blind; do
        run_bb 0 "Qwen/Qwen3-${SIZE}" "Qwen3-${SIZE}" "$COND" \
            --template_type qwen3_nothinking
    done
done

# ── §2  vicuna-13b (both GPUs — run after gpu1 script has finished) ───────────
log "=== [2/3] vicuna-13b (GPU 0,1) — waiting for gpu1 to finish first ==="
log "    Start time: $(date). Check gpu1 log before proceeding if running in parallel."

for COND in _control_blind _control_inst_blind; do
    run_bb 0,1 "lmsys/vicuna-13b-v1.5" "vicuna-13b-v1.5" "$COND" \
        --swift_model_type llama
done

# ── §3  Qwen3-32B backbone — inst_blind only (both GPUs) ─────────────────────
log "=== [3/3] Qwen3-32B backbone inst_blind (GPU 0,1) ==="

run_bb 0,1 "Qwen/Qwen3-32B" "Qwen3-32B" "_control_inst_blind" \
    --template_type qwen3_nothinking

log "=== run_backbone_gpu0.sh complete ==="
