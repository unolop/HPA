#!/bin/bash
# run_lm_decoder_chained.sh
# GPU1 starts immediately (backbone GPU1 already finished).
# GPU0 waits for backbone --gpu0 (Qwen3-1.7B) to finish, then runs LLaVA family.
# Serial backbone (vicuna-13b, Qwen3-32B) runs after both lm_decoder workers finish.
# Then lm_decoder serial (llava-vicuna-13b, Qwen3-VL-32B).
# After each model finishes, runs the monitor script.

set -e
cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG_DIR="logs/lm_decoder"
mkdir -p "$LOG_DIR"

MONITOR="python scripts/monitor_lm_decoder.py"
MONITOR_LOG="$LOG_DIR/monitor.log"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/chain.log"; }

run_lm() {
    local GPU=$1; local MODEL=$2; local CONDITION=$3; local EXTRA="${4:-}"
    local SHORT=$(echo "$MODEL" | sed 's|.*/||')
    local LOGFILE="$LOG_DIR/${SHORT}${CONDITION}.log"
    log "▶ [GPU$GPU] $SHORT | $CONDITION"
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" \
        --model_type lm \
        --condition "$CONDITION" \
        --dataset "vqa_1k" \
        --savedir "evaluation/logits/lm_decoder" \
        --resume \
        $EXTRA \
        > "$LOGFILE" 2>&1 \
        && log "✓ [GPU$GPU] $SHORT | $CONDITION" \
        || { log "✗ FAILED [GPU$GPU] $SHORT | $CONDITION — see $LOGFILE"; return 1; }
}

monitor_model() {
    local SHORT=$1
    log "--- Monitoring $SHORT ---"
    {
        echo ""
        echo "=============================="
        echo "[$(timestamp)] $SHORT"
        echo "=============================="
        $MONITOR "$SHORT" "_control_blind"    2>&1 || true
        $MONITOR "$SHORT" "_control_inst_blind" 2>&1 || true
    } | tee -a "$MONITOR_LOG"
}

# ── GPU 1 worker (starts immediately) ────────────────────────────────────────
worker_gpu1() {
    log "=== [GPU1] Starting: Qwen3-VL + InternVL3.5 ==="
    for MODEL in \
        "Qwen/Qwen3-VL-2B-Instruct" \
        "Qwen/Qwen3-VL-4B-Instruct" \
        "Qwen/Qwen3-VL-8B-Instruct" \
        "OpenGVLab/InternVL3_5-1B" \
        "OpenGVLab/InternVL3_5-2B" \
        "OpenGVLab/InternVL3_5-8B"; do
        SHORT=$(echo "$MODEL" | sed 's|.*/||')
        EXTRA=""
        [[ "$MODEL" == *"InternVL"* ]] && EXTRA="--attn_impl eager"
        for COND in _control_blind _control_inst_blind; do
            run_lm 1 "$MODEL" "$COND" "$EXTRA"
        done
        monitor_model "$SHORT"
    done
    log "=== GPU1 done ==="
}

# ── GPU 0 worker (waits for backbone --gpu0 to finish) ───────────────────────
worker_gpu0() {
    log "=== [GPU0] Waiting for backbone Qwen3-1.7B to finish... ==="
    # Wait until backbone gpu0 process exits
    while pgrep -f "run_backbone.sh --gpu0" > /dev/null 2>&1 || \
          pgrep -f "inference.py.*Qwen3-1.7B" > /dev/null 2>&1 || \
          pgrep -f "inference.py.*Qwen3-0.6B" > /dev/null 2>&1; do
        sleep 30
    done
    log "=== [GPU0] Backbone done. Starting LLaVA family ==="
    for MODEL in \
        "llava-hf/llava-1.5-7b-hf" \
        "llava-hf/llava-v1.6-mistral-7b-hf" \
        "llava-hf/llava-v1.6-vicuna-7b-hf"; do
        SHORT=$(echo "$MODEL" | sed 's|.*/||')
        for COND in _control_blind _control_inst_blind; do
            run_lm 0 "$MODEL" "$COND"
        done
        monitor_model "$SHORT"
    done
    log "=== GPU0 done ==="
}

# ── Main ─────────────────────────────────────────────────────────────────────
log "=== run_lm_decoder_chained.sh started ==="

worker_gpu1 &
PID1=$!
log "GPU1 PID=$PID1"

worker_gpu0 &
PID0=$!
log "GPU0 PID=$PID0"

wait $PID1 && log "✓ GPU1 done" || log "✗ GPU1 FAILED"
wait $PID0 && log "✓ GPU0 done" || log "✗ GPU0 FAILED"

# ── Serial: backbone large models (both GPUs) ─────────────────────────────────
log "=== Serial: backbone vicuna-13b + Qwen3-32B ==="
bash scripts/run_backbone.sh --serial

# ── Serial: lm_decoder large models (both GPUs) ───────────────────────────────
log "=== Serial: lm_decoder llava-vicuna-13b + Qwen3-VL-32B ==="
for MODEL in \
    "llava-hf/llava-v1.6-vicuna-13b-hf" \
    "Qwen/Qwen3-VL-32B-Instruct"; do
    SHORT=$(echo "$MODEL" | sed 's|.*/||')
    for COND in _control_blind _control_inst_blind; do
        log "▶ [GPU0,1] $SHORT | $COND"
        CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
            --model "$MODEL" \
            --model_type lm \
            --condition "$COND" \
            --dataset "vqa_1k" \
            --savedir "evaluation/logits/lm_decoder" \
            --resume \
            > "$LOG_DIR/${SHORT}${COND}.log" 2>&1 \
            && log "✓ [GPU0,1] $SHORT | $COND" \
            || log "✗ FAILED [GPU0,1] $SHORT | $COND"
    done
    monitor_model "$SHORT"
done

log "=== All done ==="
