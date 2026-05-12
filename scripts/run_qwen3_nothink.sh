#!/bin/bash
# Run all Qwen3 backbone models with nothinking into backbone_nothink/
# GPU 0: 0.6B, 1.7B, 4B, 8B (sequential)
# GPU 0,1: 32B (both GPUs)
#
#   nohup bash scripts/run_qwen3_nothink.sh > logs/qwen3_nothink.log 2>&1 &

cd "$(dirname "$0")/.."

LOG="logs/qwen3_nothink.log"; mkdir -p logs
PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
SAVEDIR="evaluation/logits/backbone_nothink"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run() {
    local GPUS="$1" MODEL="$2" SAVE="$3" COND="$4"
    local OUT="$SAVEDIR/pretrained/$SAVE/vqa_1k${COND}.jsonl"
    if [ -f "$OUT" ] && [ "$(wc -l < $OUT)" -ge 1000 ]; then
        log "SKIP (done): $SAVE $COND"; return
    fi
    log "▶ [GPU $GPUS] $SAVE $COND"
    mkdir -p "$SAVEDIR/pretrained/$SAVE"
    CUDA_VISIBLE_DEVICES="$GPUS" HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model      "$MODEL" \
        --model_type lm \
        --dataset    vqa_1k \
        --condition  "$COND" \
        --savedir    "$SAVEDIR" \
        --template_type qwen3_nothinking \
        >> "$LOG" 2>&1 \
        && log "✓ $SAVE $COND" \
        || { log "✗ FAILED $SAVE $COND"; exit 1; }
}

log "=== run_qwen3_nothink.sh started ==="

# ── Small models — GPU 0 only ─────────────────────────────────────────────────
for SIZE in 0.6B 1.7B 4B 8B; do
    for COND in _control_blind _control_inst_blind; do
        run 0 "Qwen/Qwen3-${SIZE}" "Qwen3-${SIZE}" "$COND"
    done
done

# ── 32B — both GPUs ───────────────────────────────────────────────────────────
for COND in _control_blind _control_inst_blind; do
    run 0,1 "Qwen/Qwen3-32B" "Qwen3-32B" "$COND"
done

log "=== run_qwen3_nothink.sh complete ==="
