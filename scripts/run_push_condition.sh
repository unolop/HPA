#!/bin/bash
# Run _control_push condition for all models (VLM + lm_decoder).
# Push prompt: "Answer based on your language knowledge and common sense."
# Contrasts with inst_blind ("imagine an image exists") to isolate language prior.
#
# GPU1: Qwen3-VL + InternVL (parallel)
# GPU0: LLaVA 7b family (parallel, waits for GPU to free)
# Serial: vicuna-13b + Qwen3-VL-32B (both GPUs)

set -e
cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG_DIR="logs/push_condition"
mkdir -p "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/run.log"; }

COND="_control_push"

run_vlm() {
    local GPU=$1; local MODEL=$2; local EXTRA="${3:-}"
    local SHORT=$(echo "$MODEL" | sed 's|.*/||')
    local LOGFILE="$LOG_DIR/vlm_${SHORT}.log"
    log "▶ VLM [GPU$GPU] $SHORT"
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" \
        --model_type vlm \
        --condition "$COND" \
        --dataset "vqa_1k" \
        --savedir "evaluation/logits/pretrained" \
        --resume \
        $EXTRA \
        > "$LOGFILE" 2>&1 \
        && log "✓ VLM [GPU$GPU] $SHORT" \
        || { log "✗ FAILED VLM [GPU$GPU] $SHORT — see $LOGFILE"; return 1; }
}

run_lm() {
    local GPU=$1; local MODEL=$2; local EXTRA="${3:-}"
    local SHORT=$(echo "$MODEL" | sed 's|.*/||')
    local LOGFILE="$LOG_DIR/lm_${SHORT}.log"
    log "▶ LM  [GPU$GPU] $SHORT"
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" \
        --model_type lm \
        --condition "$COND" \
        --dataset "vqa_1k" \
        --savedir "evaluation/logits/lm_decoder" \
        --resume \
        $EXTRA \
        > "$LOGFILE" 2>&1 \
        && log "✓ LM  [GPU$GPU] $SHORT" \
        || { log "✗ FAILED LM  [GPU$GPU] $SHORT — see $LOGFILE"; return 1; }
}

# ── GPU 1: Qwen3-VL + InternVL ───────────────────────────────────────────────
worker_gpu1() {
    log "=== [GPU1] Qwen3-VL + InternVL ==="
    for MODEL in \
        "Qwen/Qwen3-VL-2B-Instruct" \
        "Qwen/Qwen3-VL-4B-Instruct" \
        "Qwen/Qwen3-VL-8B-Instruct" \
        "OpenGVLab/InternVL3_5-1B" \
        "OpenGVLab/InternVL3_5-2B" \
        "OpenGVLab/InternVL3_5-8B"; do
        EXTRA=""
        [[ "$MODEL" == *"InternVL"* ]] && EXTRA="--attn_impl eager"
        run_vlm 1 "$MODEL" "$EXTRA"
        run_lm  1 "$MODEL" "$EXTRA"
    done
    log "=== GPU1 done ==="
}

# ── GPU 0: LLaVA 7b family ────────────────────────────────────────────────────
worker_gpu0() {
    log "=== [GPU0] LLaVA 7b family ==="
    for MODEL in \
        "llava-hf/llava-1.5-7b-hf" \
        "llava-hf/llava-v1.6-mistral-7b-hf" \
        "llava-hf/llava-v1.6-vicuna-7b-hf"; do
        run_vlm 0 "$MODEL"
        run_lm  0 "$MODEL"
    done
    log "=== GPU0 done ==="
}

log "=== run_push_condition.sh started ==="

worker_gpu1 &
PID1=$!
worker_gpu0 &
PID0=$!

wait $PID1 && log "✓ GPU1 done" || log "✗ GPU1 FAILED"
wait $PID0 && log "✓ GPU0 done" || log "✗ GPU0 FAILED"

# ── Serial: large models (both GPUs) ─────────────────────────────────────────
log "=== Serial: vicuna-13b + Qwen3-VL-32B (GPU0+1) ==="
for MODEL in \
    "llava-hf/llava-v1.6-vicuna-13b-hf" \
    "Qwen/Qwen3-VL-32B-Instruct"; do
    SHORT=$(echo "$MODEL" | sed 's|.*/||')
    CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
        --model "$MODEL" \
        --model_type vlm \
        --condition "$COND" \
        --dataset "vqa_1k" \
        --savedir "evaluation/logits/pretrained" \
        --resume \
        > "$LOG_DIR/vlm_${SHORT}.log" 2>&1 \
        && log "✓ VLM [GPU0,1] $SHORT" || log "✗ FAILED VLM $SHORT"
    CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
        --model "$MODEL" \
        --model_type lm \
        --condition "$COND" \
        --dataset "vqa_1k" \
        --savedir "evaluation/logits/lm_decoder" \
        --resume \
        > "$LOG_DIR/lm_${SHORT}.log" 2>&1 \
        && log "✓ LM  [GPU0,1] $SHORT" || log "✗ FAILED LM  $SHORT"
done

log "=== All push condition runs done ==="
