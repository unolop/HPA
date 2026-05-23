#!/bin/bash
# Wait for backbone/Qwen3-4B_think blind to reach 1000, then run all pending 32B jobs.
#
# Pending 32B jobs:
#   vlm/Qwen3-VL-32B        blind        (~888 → 1000)  fill
#   vlm/Qwen3-VL-32B        inst_blind   (~974 → 1000)  fill
#   vlm/Qwen3-VL-32B        control      (~137 → 1000)  fill
#   lm_decoder/Qwen3-VL-32B inst_blind   ( ~59 → 1000)  fill
#
# Usage:
#   nohup bash scripts/run_after_4b.sh > logs/run_after_4b.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/run_after_4b.log"
mkdir -p logs

PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
DATASET="vqa_1k"
TARGET_4B="evaluation/logits/backbone/pretrained/Qwen3-4B_think/vqa_1k_control_blind.jsonl"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

# ── Wait for Qwen3-4B blind to finish ────────────────────────────────────────
log "=== Waiting for backbone/Qwen3-4B_think blind to reach 1000 rows ==="
while true; do
    n=$(wc -l < "$TARGET_4B" 2>/dev/null || echo 0)
    if [ "$n" -ge 1000 ]; then
        log "Qwen3-4B blind complete ($n rows). Proceeding to 32B jobs."
        break
    fi
    log "  Qwen3-4B blind: $n/1000 — waiting 60s..."
    sleep 60
done

run_infer() {
    local GPUS="$1" MODEL="$2" MT="$3" COND="$4" SDIR="$5" LABEL="$6"
    shift 6
    local EXTRA="$*"
    local OUT="$SDIR/pretrained/$(basename $(echo $MODEL | tr '/' '\n' | tail -1))/${DATASET}${COND}.jsonl"
    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 1000 ]; then
        log "SKIP (complete): $LABEL | $COND"; return
    fi
    log "▶ [GPU $GPUS] $LABEL | $COND"
    CUDA_VISIBLE_DEVICES="$GPUS" HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model "$MODEL" --model_type "$MT" \
        --dataset "$DATASET" --condition "$COND" \
        --savedir "$SDIR" \
        $EXTRA \
        >> "$LOG" 2>&1 \
        && log "✓ $LABEL | $COND" \
        || { log "✗ FAILED $LABEL | $COND"; exit 1; }
}

# ── 1. vlm Qwen3-VL-32B blind ────────────────────────────────────────────────
log "=== [1/4] vlm Qwen3-VL-32B blind (fill) ==="
run_infer 0,1 \
    "Qwen/Qwen3-VL-32B-Instruct" vlm _control_blind \
    "evaluation/logits/vlm" "vlm/Qwen3-VL-32B" \
    --template_type qwen3_nothinking --fill_missing --quantization_bit 4

# ── 2. vlm Qwen3-VL-32B inst_blind ──────────────────────────────────────────
log "=== [2/4] vlm Qwen3-VL-32B inst_blind (fill) ==="
run_infer 0,1 \
    "Qwen/Qwen3-VL-32B-Instruct" vlm _control_inst_blind \
    "evaluation/logits/vlm" "vlm/Qwen3-VL-32B" \
    --template_type qwen3_nothinking --fill_missing --quantization_bit 4

# ── 3. vlm Qwen3-VL-32B control (with image) ─────────────────────────────────
log "=== [3/4] vlm Qwen3-VL-32B control (fill) ==="
run_infer 0,1 \
    "Qwen/Qwen3-VL-32B-Instruct" vlm _control \
    "evaluation/logits/vlm" "vlm/Qwen3-VL-32B" \
    --template_type qwen3_nothinking --fill_missing --quantization_bit 4

# ── 4. lm_decoder Qwen3-VL-32B inst_blind ────────────────────────────────────
log "=== [4/4] lm_decoder Qwen3-VL-32B inst_blind (fill) ==="
run_infer 0,1 \
    "Qwen/Qwen3-VL-32B-Instruct" lm _control_inst_blind \
    "evaluation/logits/lm_decoder" "lm_decoder/Qwen3-VL-32B" \
    --fill_missing --quantization_bit 4

log "=== run_after_4b.sh complete ==="
