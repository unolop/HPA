#!/bin/bash
# Run Qwen3-32B nothink (both conditions) then 32B think inst_blind fill_missing
# Waits for current GPU jobs to finish before starting
cd "$(dirname "$0")/.."

LOG="logs/run_32b_queue.log"; mkdir -p logs
PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== run_32b_queue.sh started — waiting for GPUs to free ==="

# Wait until no inference jobs remain (besides potential 4B, which finishes fast)
while pgrep -f "inference.py" > /dev/null; do
    log "Waiting for running inference jobs to finish..."
    sleep 60
done

log "=== All GPUs free — starting 32B runs ==="

run() {
    local SAVEDIR="$1" COND="$2" EXTRA="${3:-}"
    local OUT="$SAVEDIR/pretrained/Qwen3-32B/vqa_1k${COND}.jsonl"
    if [ -f "$OUT" ] && [ "$(wc -l < $OUT)" -ge 1000 ]; then
        log "SKIP (done): Qwen3-32B $COND in $SAVEDIR"; return
    fi
    log "▶ [GPU 0,1] Qwen3-32B $COND → $SAVEDIR"
    CUDA_VISIBLE_DEVICES=0,1 HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model Qwen/Qwen3-32B --model_type lm --dataset vqa_1k \
        --condition "$COND" --savedir "$SAVEDIR" \
        --template_type qwen3_nothinking \
        --quantization_bit 4 \
        $EXTRA \
        >> "$LOG" 2>&1 \
        && log "✓ Qwen3-32B $COND nothink" \
        || { log "✗ FAILED"; exit 1; }
}

# 32B nothink — both conditions (blind now done, will SKIP; inst_blind from scratch)
run "evaluation/logits/backbone_nothink" "_control_blind" "--fill_missing"
run "evaluation/logits/backbone_nothink" "_control_inst_blind" "--fill_missing"

# 32B think inst_blind fill_missing (135 lines exist, need 865 more)
THINK_OUT="evaluation/logits/backbone_think/pretrained/Qwen3-32B/vqa_1k_control_inst_blind.jsonl"
if [ -f "$THINK_OUT" ] && [ "$(wc -l < $THINK_OUT)" -ge 1000 ]; then
    log "SKIP: 32B think inst_blind already done"
else
    log "▶ [GPU 0,1] Qwen3-32B inst_blind think (fill_missing)"
    CUDA_VISIBLE_DEVICES=0,1 HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model Qwen/Qwen3-32B --model_type lm --dataset vqa_1k \
        --condition _control_inst_blind \
        --savedir evaluation/logits/backbone_think \
        --fill_missing \
        --quantization_bit 4 \
        >> "$LOG" 2>&1 \
        && log "✓ Qwen3-32B think inst_blind fill_missing" \
        || { log "✗ FAILED think fill_missing"; exit 1; }
fi

# Qwen3-4B think blind fill_missing (888/1000 done)
THINK4B_OUT="evaluation/logits/backbone_think/pretrained/Qwen3-4B/vqa_1k_control_blind.jsonl"
if [ -f "$THINK4B_OUT" ] && [ "$(wc -l < $THINK4B_OUT)" -ge 1000 ]; then
    log "SKIP: Qwen3-4B think blind already done"
else
    log "▶ [GPU 0,1] Qwen3-4B blind think (fill_missing)"
    CUDA_VISIBLE_DEVICES=0,1 HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model Qwen/Qwen3-4B --model_type lm --dataset vqa_1k \
        --condition _control_blind \
        --savedir evaluation/logits/backbone_think \
        --fill_missing \
        --quantization_bit 4 \
        >> "$LOG" 2>&1 \
        && log "✓ Qwen3-4B think blind fill_missing" \
        || { log "✗ FAILED Qwen3-4B think blind"; exit 1; }
fi

log "=== run_32b_queue.sh complete ==="
