#!/bin/bash
# Run missing / re-run backbone (standalone LLM) inference WITHOUT thinking.
#
# What this covers:
#   [re-run with nothinking — existing data has empty <think> tags]
#     Qwen3-0.6B / 1.7B / 4B / 8B : blind + inst_blind (--overwrite)
#
#   [new — no data yet]
#     Qwen3-32B     : inst_blind only   (blind + control already done)
#     vicuna-7b-v1.5: blind + inst_blind
#     vicuna-13b-v1.5: blind + inst_blind
#
# GPU strategy:
#   Small models (0.6B–8B, vicuna-7b): 1 GPU each — run sequentially on GPU 0
#   Large models (vicuna-13b, Qwen3-32B): both GPUs
#
# Usage:
#   nohup bash scripts/run_backbone_missing.sh > logs/backbone_missing.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/backbone_missing.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== run_backbone_missing.sh started ==="

HF_CACHE="/home/david/Desktop/yuna/.cache/hf"
BB_DIR="evaluation/logits/backbone"

# ── Helper ─────────────────────────────────────────────────────────────────────
run_bb() {
    local GPUS="$1"
    local MODEL_ID="$2"   # HF model id, e.g. lmsys/vicuna-7b-v1.5
    local SAVE_NAME="$3"  # subfolder under BB_DIR/pretrained
    local COND="$4"       # _control_blind | _control_inst_blind
    shift 4
    local EXTRA="$*"

    log "▶ [GPU $GPUS] $SAVE_NAME | $COND"
    CUDA_VISIBLE_DEVICES="$GPUS" \
    TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" \
    python evaluation/inference.py \
        --model       "$MODEL_ID" \
        --model_type  lm \
        --dataset     vqa_1k \
        --condition   "$COND" \
        --savedir     "$BB_DIR" \
        $EXTRA \
        >> "$LOG" 2>&1 \
        && log "✓ $SAVE_NAME | $COND" \
        || { log "✗ FAILED $SAVE_NAME | $COND"; exit 1; }
}

# ══════════════════════════════════════════════════════════════════════════════
# §1  Re-run small Qwen3 backbone WITH nothinking  (single GPU, sequential)
#     These already have data but think tags are present — overwrite.
# ══════════════════════════════════════════════════════════════════════════════
log "=== [1/4] Qwen3 small backbone — nothinking re-run (GPU 0) ==="

for MODEL_SIZE in 0.6B 1.7B 4B 8B; do
    for COND in _control_blind _control_inst_blind; do
        run_bb 0 \
            "Qwen/Qwen3-${MODEL_SIZE}" \
            "Qwen3-${MODEL_SIZE}" \
            "$COND" \
            --template_type qwen3_nothinking
    done
done

# ══════════════════════════════════════════════════════════════════════════════
# §2  Qwen3-32B backbone — inst_blind only  (both GPUs)
#     blind + control already done; just need inst_blind
# ══════════════════════════════════════════════════════════════════════════════
log "=== [2/4] Qwen3-32B backbone — inst_blind (GPU 0,1) ==="

run_bb 0,1 \
    "Qwen/Qwen3-32B" \
    "Qwen3-32B" \
    _control_inst_blind \
    --template_type qwen3_nothinking

# ══════════════════════════════════════════════════════════════════════════════
# §3  vicuna-7b  (single GPU)
# ══════════════════════════════════════════════════════════════════════════════
log "=== [3/4] vicuna-7b-v1.5 — blind + inst_blind (GPU 0) ==="

for COND in _control_blind _control_inst_blind; do
    run_bb 0 \
        "lmsys/vicuna-7b-v1.5" \
        "vicuna-7b-v1.5" \
        "$COND" \
        --swift_model_type llama
done

# ══════════════════════════════════════════════════════════════════════════════
# §4  vicuna-13b  (~26 GB — both GPUs)
# ══════════════════════════════════════════════════════════════════════════════
log "=== [4/4] vicuna-13b-v1.5 — blind + inst_blind (GPU 0,1) ==="

for COND in _control_blind _control_inst_blind; do
    run_bb 0,1 \
        "lmsys/vicuna-13b-v1.5" \
        "vicuna-13b-v1.5" \
        "$COND" \
        --swift_model_type llama
done

log "=== All backbone inference complete ==="
