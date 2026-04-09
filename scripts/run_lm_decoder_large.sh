#!/bin/bash
# Run lm_decoder for the two large models (vicuna-13b, Qwen3-VL-32B) using both GPUs.
# Run this independently after the parallel GPU0/GPU1 jobs are done.

set -e
cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG_DIR="logs/lm_decoder"
mkdir -p "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/chain.log"; }

log "=== lm_decoder large models (GPU0+1): vicuna-13b + Qwen3-VL-32B ==="

for MODEL in \
    "llava-hf/llava-v1.6-vicuna-13b-hf" \
    "Qwen/Qwen3-VL-32B-Instruct"; do
    SHORT=$(echo "$MODEL" | sed 's|.*/||')
    for COND in _control_blind _control_inst_blind; do
        LOGFILE="$LOG_DIR/${SHORT}${COND}.log"
        log "▶ [GPU0,1] $SHORT | $COND"
        CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
            --model "$MODEL" \
            --model_type lm \
            --condition "$COND" \
            --dataset "vqa_1k" \
            --savedir "evaluation/logits/lm_decoder" \
            --resume \
            > "$LOGFILE" 2>&1 \
            && log "✓ [GPU0,1] $SHORT | $COND" \
            || { log "✗ FAILED — see $LOGFILE"; cat "$LOGFILE" | tail -5; }
    done
done

log "=== Done ==="
