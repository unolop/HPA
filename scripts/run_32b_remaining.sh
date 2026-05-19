#!/bin/bash
# Queue remaining Qwen3-VL-32B conditions after vlm blind finishes.
#
# Conditions (in order):
#   vlm/Qwen3-VL-32B        _control_blind      888→1000  (already running — wait)
#   vlm/Qwen3-VL-32B        _control_inst_blind 974→1000  fill
#   vlm/Qwen3-VL-32B        _control            137→1000  fill
#   lm_decoder/Qwen3-VL-32B _control_inst_blind  59→1000  fill
#
# Usage:
#   nohup bash scripts/run_32b_remaining.sh >> logs/run_32b_manual.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/run_32b_manual.log"
PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
MODEL="Qwen/Qwen3-VL-32B-Instruct"
DATASET="vqa_1k"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run_infer() {
    local GPUS="$1" MT="$2" COND="$3" SDIR="$4" LABEL="$5"
    shift 5
    local EXTRA="$*"
    local MDIR=$(basename $(echo $MODEL | tr '/' '\n' | tail -1))
    local OUT="$SDIR/pretrained/$MDIR/${DATASET}${COND}.jsonl"
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

# ── Wait for vlm blind to finish ─────────────────────────────────────────────
TARGET="evaluation/logits/vlm/pretrained/Qwen3-VL-32B-Instruct/${DATASET}_control_blind.jsonl"
log "=== Waiting for vlm blind to reach 1000 rows ==="
while true; do
    n=$(wc -l < "$TARGET" 2>/dev/null || echo 0)
    [ "$n" -ge 1000 ] && { log "vlm blind complete. Proceeding."; break; }
    log "  vlm blind: $n/1000 — waiting 30s..."
    sleep 30
done

log "=== run_32b_remaining.sh started ==="

# ── 1. vlm inst_blind ─────────────────────────────────────────────────────────
log "=== [1/3] vlm Qwen3-VL-32B inst_blind (fill) ==="
run_infer 0,1 vlm _control_inst_blind \
    "evaluation/logits/vlm" "vlm/Qwen3-VL-32B" \
    --template_type qwen3_nothinking --fill_missing --quantization_bit 4

# ── 2. vlm control (real image) ───────────────────────────────────────────────
log "=== [2/3] vlm Qwen3-VL-32B control (fill) ==="
run_infer 0,1 vlm _control \
    "evaluation/logits/vlm" "vlm/Qwen3-VL-32B" \
    --template_type qwen3_nothinking --fill_missing --quantization_bit 4

# ── 3. lm_decoder inst_blind ─────────────────────────────────────────────────
log "=== [3/3] lm_decoder Qwen3-VL-32B inst_blind (fill) ==="
run_infer 0,1 lm _control_inst_blind \
    "evaluation/logits/lm_decoder" "lm_decoder/Qwen3-VL-32B" \
    --fill_missing --quantization_bit 4

log "=== run_32b_remaining.sh complete ==="
