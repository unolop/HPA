#!/bin/bash
# run_control_parallel.sh
# Parallel GPU inference for vqa_1k_control.
#
# Strategy:
#   GPU 0 (worker A): Qwen3-VL-2B, 4B, 8B + MiniCPM-V-2_6
#   GPU 1 (worker B): Molmo-7B-D + Idefics3-8B + LLaVA fill-pron (already have all conditions)
#   Serial GPU 0,1:   LLaVA-1.6-Vicuna-13B (control_blind only, 26GB)
#                     Llama-3.2-11B-Vision (all 3 conditions, 22GB)
#   Fill pronominalized runs on the same GPU as inference for each model.
#
# Usage:
#   bash scripts/run_control_parallel.sh          # full run
#   bash scripts/run_control_parallel.sh --phase A  # only worker A (GPU 0)
#   bash scripts/run_control_parallel.sh --phase B  # only worker B (GPU 1)
#   bash scripts/run_control_parallel.sh --phase S  # only serial (large models)
#   bash scripts/run_control_parallel.sh --phase F  # only fill-pron pass

set -e
cd "$(dirname "$0")/.."

LOG_DIR="logs/control_inference"
mkdir -p "$LOG_DIR"

PHASE_ONLY=""
for arg in "$@"; do
    case $arg in
        --phase=*) PHASE_ONLY="${arg#*=}" ;;
        --phase) shift; PHASE_ONLY="$1" ;;
    esac
done

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/run_parallel.log"; }
logA() { echo "[$(timestamp)] [GPU0] $*" | tee -a "$LOG_DIR/workerA.log"; }
logB() { echo "[$(timestamp)] [GPU1] $*" | tee -a "$LOG_DIR/workerB.log"; }

# ── helpers ───────────────────────────────────────────────────────────────────
run_infer() {
    local GPU=$1; local MODEL=$2; local CONDITION=$3; local EXTRA="${4:-}"
    local SHORT=$(echo "$MODEL" | sed 's|.*/||')
    local LOGFILE="$LOG_DIR/${SHORT}${CONDITION//\\//_}.log"
    echo "[$(timestamp)] ▶ ${SHORT} | ${CONDITION} ${EXTRA}" | tee -a "$LOG_DIR/workerA.log" "$LOG_DIR/workerB.log" > /dev/null
    # write to the right worker log
    if [ "$GPU" = "0" ]; then
        logA "▶ $SHORT | $CONDITION $EXTRA"
    else
        logB "▶ $SHORT | $CONDITION $EXTRA"
    fi
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" \
        --condition "$CONDITION" \
        --dataset "vqa_1k" \
        $EXTRA \
        > "$LOGFILE" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        [ "$GPU" = "0" ] && logA "✓ $SHORT | $CONDITION" || logB "✓ $SHORT | $CONDITION"
    else
        [ "$GPU" = "0" ] && logA "✗ FAILED $SHORT | $CONDITION — $LOGFILE" \
                         || logB "✗ FAILED $SHORT | $CONDITION — $LOGFILE"
    fi
}

run_fill() {
    local GPU=$1; local MODEL=$2; local CONDITION=$3
    local SHORT=$(echo "$MODEL" | sed 's|.*/||')
    local LOGFILE="$LOG_DIR/${SHORT}${CONDITION//\\//_}_fill_pron.log"
    [ "$GPU" = "0" ] && logA "▶ fill pron: $SHORT | $CONDITION" \
                     || logB "▶ fill pron: $SHORT | $CONDITION"
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/inference.py \
        --model "$MODEL" \
        --condition "$CONDITION" \
        --dataset "vqa_1k" \
        --fill_missing \
        --control_types "pronominalized" \
        > "$LOGFILE" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        [ "$GPU" = "0" ] && logA "✓ fill pron: $SHORT | $CONDITION" \
                         || logB "✓ fill pron: $SHORT | $CONDITION"
    else
        [ "$GPU" = "0" ] && logA "✗ FAILED fill pron: $SHORT | $CONDITION — $LOGFILE" \
                         || logB "✗ FAILED fill pron: $SHORT | $CONDITION — $LOGFILE"
    fi
}

# ── Worker A: GPU 0 ───────────────────────────────────────────────────────────
worker_a() {
    logA "=== Worker A started (GPU 0) ==="

    # Qwen3-VL-2B — all three missing
    run_infer 0 "Qwen/Qwen3-VL-2B-Instruct" "_control"
    run_infer 0 "Qwen/Qwen3-VL-2B-Instruct" "_control_blind"
    run_infer 0 "Qwen/Qwen3-VL-2B-Instruct" "_control_inst_blind"

    # Qwen3-VL-4B — control + control_blind missing
    run_infer 0 "Qwen/Qwen3-VL-4B-Instruct" "_control"
    run_infer 0 "Qwen/Qwen3-VL-4B-Instruct" "_control_blind"

    # Qwen3-VL-8B — inst_blind incomplete, resume
    run_infer 0 "Qwen/Qwen3-VL-8B-Instruct" "_control_inst_blind" "--resume"

    # MiniCPM-V-2_6 — all three missing
    run_infer 0 "openbmb/MiniCPM-V-2_6" "_control"
    run_infer 0 "openbmb/MiniCPM-V-2_6" "_control_blind"
    run_infer 0 "openbmb/MiniCPM-V-2_6" "_control_inst_blind"

    # Fill pronominalized for GPU-0 models
    for MODEL in \
        "Qwen/Qwen3-VL-2B-Instruct" \
        "Qwen/Qwen3-VL-4B-Instruct" \
        "Qwen/Qwen3-VL-8B-Instruct" \
        "openbmb/MiniCPM-V-2_6"; do
        run_fill 0 "$MODEL" "_control"
        run_fill 0 "$MODEL" "_control_blind"
        run_fill 0 "$MODEL" "_control_inst_blind"
    done

    logA "=== Worker A done ==="
}

# ── Worker B: GPU 1 ───────────────────────────────────────────────────────────
worker_b() {
    logB "=== Worker B started (GPU 1) ==="

    # Molmo-7B-D — all three missing
    run_infer 1 "allenai/Molmo-7B-D-0924" "_control"
    run_infer 1 "allenai/Molmo-7B-D-0924" "_control_blind"
    run_infer 1 "allenai/Molmo-7B-D-0924" "_control_inst_blind"

    # Idefics3-8B — all three missing
    run_infer 1 "HuggingFaceM4/Idefics3-8B-Llama3" "_control"
    run_infer 1 "HuggingFaceM4/Idefics3-8B-Llama3" "_control_blind"
    run_infer 1 "HuggingFaceM4/Idefics3-8B-Llama3" "_control_inst_blind"

    # Fill pronominalized: LLaVA family (all conditions already have inference, just add pron)
    for MODEL in \
        "llava-hf/llava-1.5-7b-hf" \
        "llava-hf/llava-v1.6-mistral-7b-hf" \
        "llava-hf/llava-v1.6-vicuna-7b-hf"; do
        run_fill 1 "$MODEL" "_control"
        run_fill 1 "$MODEL" "_control_blind"
        run_fill 1 "$MODEL" "_control_inst_blind"
    done

    # Fill pronominalized for GPU-1 inference models
    for MODEL in \
        "allenai/Molmo-7B-D-0924" \
        "HuggingFaceM4/Idefics3-8B-Llama3"; do
        run_fill 1 "$MODEL" "_control"
        run_fill 1 "$MODEL" "_control_blind"
        run_fill 1 "$MODEL" "_control_inst_blind"
    done

    logB "=== Worker B done ==="
}

# ── Serial: large models needing both GPUs ────────────────────────────────────
serial_large() {
    log "=== Serial large models (GPU 0,1) ==="

    # LLaVA-1.6-Vicuna-13B — only control_blind is missing
    CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
        --model "llava-hf/llava-v1.6-vicuna-13b-hf" \
        --condition "_control_blind" \
        --dataset "vqa_1k" \
        > "$LOG_DIR/llava-v1.6-vicuna-13b-hf_control_blind.log" 2>&1 \
        && log "✓ vicuna-13b | _control_blind" \
        || log "✗ FAILED vicuna-13b | _control_blind"

    # Llama-3.2-11B-Vision — all three missing
    for COND in "_control" "_control_blind" "_control_inst_blind"; do
        CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
            --model "meta-llama/Llama-3.2-11B-Vision-Instruct" \
            --condition "$COND" \
            --dataset "vqa_1k" \
            > "$LOG_DIR/Llama-3.2-11B-Vision-Instruct${COND//\\//_}.log" 2>&1 \
            && log "✓ Llama-3.2-11B | $COND" \
            || log "✗ FAILED Llama-3.2-11B | $COND"
    done

    # Fill pronominalized for serial models
    for MODEL in \
        "llava-hf/llava-v1.6-vicuna-13b-hf" \
        "meta-llama/Llama-3.2-11B-Vision-Instruct"; do
        for COND in "_control" "_control_blind" "_control_inst_blind"; do
            CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
                --model "$MODEL" \
                --condition "$COND" \
                --dataset "vqa_1k" \
                --fill_missing \
                --control_types "pronominalized" \
                > "$LOG_DIR/$(echo $MODEL | sed 's|.*/||')${COND//\\//_}_fill_pron.log" 2>&1 \
                && log "✓ fill pron: $(echo $MODEL | sed 's|.*/||') | $COND" \
                || log "✗ FAILED fill pron: $(echo $MODEL | sed 's|.*/||') | $COND"
        done
    done

    log "=== Serial large models done ==="
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "=== run_control_parallel.sh started === phase: ${PHASE_ONLY:-all}"

case "$PHASE_ONLY" in
    A) worker_a ;;
    B) worker_b ;;
    S) serial_large ;;
    *)
        # Run A and B in parallel, then serial
        worker_a &
        PID_A=$!
        worker_b &
        PID_B=$!

        log "Workers A (PID $PID_A) and B (PID $PID_B) running in parallel"
        wait $PID_A; log "Worker A finished"
        wait $PID_B; log "Worker B finished"

        serial_large
        ;;
esac

log "=== run_control_parallel.sh finished ==="
