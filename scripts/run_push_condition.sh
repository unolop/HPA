#!/bin/bash
# Run _control_push condition for all models: VLM, lm_decoder, and backbone.
# Push prompt: "Answer based on your language knowledge and common sense."
# Contrasts with inst_blind ("imagine an image exists") to isolate language prior.
#
# Note: backbone models skip _control (redundant — no image in any condition).
#
# GPU1: Qwen3-VL + InternVL (VLM + lm_decoder) + Qwen3-4B/8B backbone
# GPU0: LLaVA 7b (VLM + lm_decoder) + vicuna-7b/Mistral/Qwen3-0.6B/1.7B backbone
# Serial: vicuna-13b + Qwen3-VL-32B (VLM + lm_decoder) + Qwen3-32B backbone

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
    log "▶ VLM [GPU$GPU] $SHORT"
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" --model_type vlm --condition "$COND" \
        --dataset "vqa_1k" --savedir "evaluation/logits/pretrained" --resume $EXTRA \
        > "$LOG_DIR/vlm_${SHORT}.log" 2>&1 \
        && log "✓ VLM [GPU$GPU] $SHORT" \
        || { log "✗ FAILED VLM [GPU$GPU] $SHORT"; return 1; }
}

run_lm() {
    local GPU=$1; local MODEL=$2; local EXTRA="${3:-}"
    local SHORT=$(echo "$MODEL" | sed 's|.*/||')
    log "▶ LM  [GPU$GPU] $SHORT"
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" --model_type lm --condition "$COND" \
        --dataset "vqa_1k" --savedir "evaluation/logits/lm_decoder" --resume $EXTRA \
        > "$LOG_DIR/lm_${SHORT}.log" 2>&1 \
        && log "✓ LM  [GPU$GPU] $SHORT" \
        || { log "✗ FAILED LM  [GPU$GPU] $SHORT"; return 1; }
}

run_backbone() {
    local GPU=$1; local MODEL=$2
    local SHORT=$(echo "$MODEL" | sed 's|.*/||')
    local EXTRA=""
    [[ "$MODEL" == *"Qwen3"* ]]   && EXTRA="--template_type qwen3_nothinking"
    [[ "$MODEL" == *"vicuna"* ]]  && EXTRA="$EXTRA --swift_model_type llama"
    [[ "$MODEL" == *"Mistral"* ]] && EXTRA="$EXTRA --swift_model_type mistral"
    log "▶ BB  [GPU$GPU] $SHORT"
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" --model_type lm --condition "$COND" \
        --dataset "vqa_1k" --savedir "evaluation/logits/backbone" --resume $EXTRA \
        > "$LOG_DIR/bb_${SHORT}.log" 2>&1 \
        && log "✓ BB  [GPU$GPU] $SHORT" \
        || { log "✗ FAILED BB  [GPU$GPU] $SHORT"; return 1; }
}

# ── GPU 1: Qwen3-VL + InternVL + Qwen3-4B/8B backbone ───────────────────────
worker_gpu1() {
    log "=== [GPU1] Qwen3-VL + InternVL + Qwen3 backbones ==="
    for MODEL in \
        "Qwen/Qwen3-VL-2B-Instruct" \
        "Qwen/Qwen3-VL-4B-Instruct" \
        "Qwen/Qwen3-VL-8B-Instruct" \
        "OpenGVLab/InternVL3_5-1B" \
        "OpenGVLab/InternVL3_5-2B" \
        "OpenGVLab/InternVL3_5-8B"; do
        EXTRA=""; [[ "$MODEL" == *"InternVL"* ]] && EXTRA="--attn_impl eager"
        run_vlm 1 "$MODEL" "$EXTRA"
        run_lm  1 "$MODEL" "$EXTRA"
    done
    for MODEL in "Qwen/Qwen3-4B" "Qwen/Qwen3-8B"; do
        run_backbone 1 "$MODEL"
    done
    log "=== GPU1 done ==="
}

# ── GPU 0: LLaVA 7b + small backbones ────────────────────────────────────────
worker_gpu0() {
    log "=== [GPU0] LLaVA 7b + small backbones ==="
    for MODEL in \
        "llava-hf/llava-1.5-7b-hf" \
        "llava-hf/llava-v1.6-mistral-7b-hf" \
        "llava-hf/llava-v1.6-vicuna-7b-hf"; do
        run_vlm 0 "$MODEL"
        run_lm  0 "$MODEL"
    done
    for MODEL in \
        "lmsys/vicuna-7b-v1.5" \
        "mistralai/Mistral-7B-Instruct-v0.2" \
        "Qwen/Qwen3-0.6B" \
        "Qwen/Qwen3-1.7B"; do
        run_backbone 0 "$MODEL"
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
log "=== Serial: large models (GPU0+1) ==="
for MODEL in \
    "llava-hf/llava-v1.6-vicuna-13b-hf" \
    "Qwen/Qwen3-VL-32B-Instruct"; do
    SHORT=$(echo "$MODEL" | sed 's|.*/||')
    CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
        --model "$MODEL" --model_type vlm --condition "$COND" \
        --dataset "vqa_1k" --savedir "evaluation/logits/pretrained" --resume \
        > "$LOG_DIR/vlm_${SHORT}.log" 2>&1 \
        && log "✓ VLM [GPU0,1] $SHORT" || log "✗ FAILED VLM $SHORT"
    CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
        --model "$MODEL" --model_type lm --condition "$COND" \
        --dataset "vqa_1k" --savedir "evaluation/logits/lm_decoder" --resume \
        > "$LOG_DIR/lm_${SHORT}.log" 2>&1 \
        && log "✓ LM  [GPU0,1] $SHORT" || log "✗ FAILED LM  $SHORT"
done

# Qwen3-32B backbone (large)
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "Qwen/Qwen3-32B" --model_type lm --condition "$COND" \
    --dataset "vqa_1k" --savedir "evaluation/logits/backbone" --resume \
    --template_type qwen3_nothinking \
    > "$LOG_DIR/bb_Qwen3-32B.log" 2>&1 \
    && log "✓ BB  [GPU0,1] Qwen3-32B" || log "✗ FAILED BB  Qwen3-32B"

# vicuna-13b backbone
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "lmsys/vicuna-13b-v1.5" --model_type lm --condition "$COND" \
    --dataset "vqa_1k" --savedir "evaluation/logits/backbone" --resume \
    --swift_model_type llama \
    > "$LOG_DIR/bb_vicuna-13b.log" 2>&1 \
    && log "✓ BB  [GPU0,1] vicuna-13b" || log "✗ FAILED BB  vicuna-13b"

log "=== All push condition runs done ==="
