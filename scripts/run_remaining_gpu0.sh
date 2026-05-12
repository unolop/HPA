#!/bin/bash
# GPU 0: Qwen3 small inst_blind (nothinking) → vicuna-13b blind → Qwen3-32B inst_blind
cd "$(dirname "$0")/.."
source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/remaining_gpu0.log"; mkdir -p logs
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

BB="evaluation/logits/backbone"
HF="evaluation/logits/backbone"
CACHE="/home/david/Desktop/yuna/.cache/hf"

run_bb() {
    local GPUS="$1" MODEL="$2" SAVE="$3" COND="$4"; shift 4
    local OUT="$BB/pretrained/$SAVE/vqa_1k${COND}.jsonl"
    if [ -f "$OUT" ] && [ "$(wc -l < $OUT)" -ge 1000 ]; then
        log "SKIP (done): $SAVE $COND"; return
    fi
    log "▶ [GPU $GPUS] $SAVE $COND"
    CUDA_VISIBLE_DEVICES="$GPUS" HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    /home/david/miniconda3/envs/zero/bin/python evaluation/inference.py \
        --model "$MODEL" --model_type lm --dataset vqa_1k \
        --condition "$COND" --savedir "$BB" "$@" >> "$LOG" 2>&1 \
        && log "✓ $SAVE $COND" || { log "✗ FAILED $SAVE $COND"; exit 1; }
}

log "=== run_remaining_gpu0.sh started ==="

log "--- Qwen3 small inst_blind nothinking (GPU 0) ---"
run_bb 0 "Qwen/Qwen3-0.6B" "Qwen3-0.6B" "_control_inst_blind" --template_type qwen3_nothinking
run_bb 0 "Qwen/Qwen3-4B"   "Qwen3-4B"   "_control_inst_blind" --template_type qwen3_nothinking
run_bb 0 "Qwen/Qwen3-8B"   "Qwen3-8B"   "_control_inst_blind" --template_type qwen3_nothinking

log "--- vicuna-13b blind (GPU 0,1) ---"
run_bb 0,1 "lmsys/vicuna-13b-v1.5" "vicuna-13b-v1.5" "_control_blind" --swift_model_type llama

log "--- Qwen3-32B inst_blind nothinking (GPU 0,1) ---"
run_bb 0,1 "Qwen/Qwen3-32B" "Qwen3-32B" "_control_inst_blind" --template_type qwen3_nothinking

log "=== run_remaining_gpu0.sh complete ==="
