#!/bin/bash
# Parallel step-1 + serial 32B pipeline.
#
# GPU 0: waits for in-flight InternVL-1B (PID $1), then runs InternVL-2B
# GPU 1: InternVL-8B → Qwen3-VL-2B
# Both: join → Qwen3-VL-32B fill (steps 2-5 from run_missing.sh)
#
# Usage:
#   bash scripts/run_parallel_step1.sh <internvl1b_pid>

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

INTERNVL1B_PID="${1:?Usage: $0 <internvl1b_pid>}"
LOG="logs/missing.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

SAVEDIR="evaluation/logits"
DATASET="vqa_1k"

log "=== run_parallel_step1.sh started (watching PID $INTERNVL1B_PID) ==="

# ── GPU 0 worker: wait for InternVL-1B, then run InternVL-2B ─────────────────
gpu0_worker() {
    log "[GPU 0] Waiting for InternVL-1B (PID $INTERNVL1B_PID) to finish..."
    while kill -0 "$INTERNVL1B_PID" 2>/dev/null; do sleep 10; done
    log "[GPU 0] InternVL-1B done."

    log "▶ [GPU 0] InternVL3_5-2B | vqa_1k orig"
    CUDA_VISIBLE_DEVICES=0 python evaluation/inference.py \
        --model "OpenGVLab/InternVL3_5-2B" --model_type vlm \
        --dataset "$DATASET" --condition "" \
        --savedir "$SAVEDIR" --attn_impl eager \
        >> "$LOG" 2>&1 \
        && log "✓ [GPU 0] InternVL3_5-2B" \
        || { log "✗ FAILED InternVL3_5-2B"; exit 1; }
}

# ── GPU 1 worker: InternVL-8B → Qwen3-VL-2B ─────────────────────────────────
gpu1_worker() {
    log "▶ [GPU 1] InternVL3_5-8B | vqa_1k orig"
    CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
        --model "OpenGVLab/InternVL3_5-8B" --model_type vlm \
        --dataset "$DATASET" --condition "" \
        --savedir "$SAVEDIR" --attn_impl eager \
        >> "$LOG" 2>&1 \
        && log "✓ [GPU 1] InternVL3_5-8B" \
        || { log "✗ FAILED InternVL3_5-8B"; exit 1; }

    log "▶ [GPU 1] Qwen3-VL-2B | vqa_1k orig"
    CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
        --model "Qwen/Qwen3-VL-2B-Instruct" --model_type vlm \
        --dataset "$DATASET" --condition "" \
        --savedir "$SAVEDIR" --template_type qwen3_nothinking \
        >> "$LOG" 2>&1 \
        && log "✓ [GPU 1] Qwen3-VL-2B" \
        || { log "✗ FAILED Qwen3-VL-2B"; exit 1; }
}

# Run both workers in parallel, wait for both
gpu0_worker &
GPU0_PID=$!
gpu1_worker &
GPU1_PID=$!

wait $GPU0_PID || exit 1
wait $GPU1_PID || exit 1

log "=== Step 1 complete. Starting 32B fill (both GPUs) ==="

# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Pretrained Qwen3-VL-32B fill (all 4 conditions)
# ═════════════════════════════════════════════════════════════════════════════
log "=== [2/5] Qwen3-VL-32B pretrained fill ==="

COMMON_32B="--model Qwen/Qwen3-VL-32B-Instruct --model_type vlm
            --dataset $DATASET --savedir $SAVEDIR
            --template_type qwen3_nothinking --fill_missing"

for COND in _control "" _control_blind _control_inst_blind; do
    LABEL="${COND:-orig}"
    log "▶ [GPU 0,1] Qwen3-VL-32B | $LABEL"
    CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
        $COMMON_32B --condition "$COND" >> "$LOG" 2>&1 \
        && log "✓ Qwen3-VL-32B | $LABEL" \
        || { log "✗ FAILED Qwen3-VL-32B | $LABEL"; exit 1; }
done

# ═════════════════════════════════════════════════════════════════════════════
# Step 3: LM-decoder Qwen3-VL-32B
# ═════════════════════════════════════════════════════════════════════════════
log "=== [3/5] LM-decoder Qwen3-VL-32B ==="

log "▶ [GPU 0,1] lm Qwen3-VL-32B | _control_blind (fill)"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "Qwen/Qwen3-VL-32B-Instruct" --model_type lm \
    --dataset "$DATASET" --condition _control_blind \
    --savedir "$SAVEDIR/lm_decoder" --fill_missing \
    >> "$LOG" 2>&1 \
    && log "✓ lm Qwen3-VL-32B | _control_blind" \
    || { log "✗ FAILED lm Qwen3-VL-32B | _control_blind"; exit 1; }

log "▶ [GPU 0,1] lm Qwen3-VL-32B | _control_inst_blind"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "Qwen/Qwen3-VL-32B-Instruct" --model_type lm \
    --dataset "$DATASET" --condition _control_inst_blind \
    --savedir "$SAVEDIR/lm_decoder" \
    >> "$LOG" 2>&1 \
    && log "✓ lm Qwen3-VL-32B | _control_inst_blind" \
    || { log "✗ FAILED lm Qwen3-VL-32B | _control_inst_blind"; exit 1; }

# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Backbone Qwen3-32B inst_blind
# ═════════════════════════════════════════════════════════════════════════════
log "=== [4/5] Backbone Qwen3-32B _control_inst_blind ==="

CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "Qwen/Qwen3-32B" --model_type lm \
    --dataset "$DATASET" --condition _control_inst_blind \
    --savedir "$SAVEDIR/backbone" \
    --template_type qwen3_nothinking \
    >> "$LOG" 2>&1 \
    && log "✓ backbone Qwen3-32B | _control_inst_blind" \
    || { log "✗ FAILED backbone Qwen3-32B | _control_inst_blind"; exit 1; }

# ═════════════════════════════════════════════════════════════════════════════
# Step 5: Backbone vicuna-13b ctrl (needs both GPUs, 26GB bf16)
# ═════════════════════════════════════════════════════════════════════════════
log "=== [5/5] Backbone vicuna-13b-v1.5 _control ==="

CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "lmsys/vicuna-13b-v1.5" --model_type lm \
    --dataset "$DATASET" --condition _control \
    --savedir "$SAVEDIR/backbone" \
    --swift_model_type llama --template_type wizardlm2_moe \
    >> "$LOG" 2>&1 \
    && log "✓ backbone vicuna-13b | _control" \
    || { log "✗ FAILED backbone vicuna-13b | _control"; exit 1; }

# ═════════════════════════════════════════════════════════════════════════════
# Step 6: Backbone vicuna — rerun all conditions with correct template
# Previous runs used --swift_model_type llama without --template_type, which
# defaulted to [INST] format → EOS immediately → 100% empty answers.
# Fix: add --template_type wizardlm2_moe (USER: ASSISTANT: vicuna format).
# ═════════════════════════════════════════════════════════════════════════════
log "=== [6/6] Backbone vicuna: rerun all conditions with correct template ==="

VICUNA_COMMON="--model_type lm --dataset $DATASET --savedir $SAVEDIR/backbone
               --swift_model_type llama --template_type wizardlm2_moe"

for MODEL in "lmsys/vicuna-7b-v1.5" "lmsys/vicuna-13b-v1.5"; do
    SHORT=$(echo "$MODEL" | sed 's|.*/||')
    for COND in _control _control_blind _control_inst_blind; do
        log "▶ [GPU 0,1] backbone $SHORT | $COND"
        CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
            --model "$MODEL" $VICUNA_COMMON --condition "$COND" \
            >> "$LOG" 2>&1 \
            && log "✓ backbone $SHORT | $COND" \
            || { log "✗ FAILED backbone $SHORT | $COND"; exit 1; }
    done
done

log "=== run_parallel_step1.sh done ==="
