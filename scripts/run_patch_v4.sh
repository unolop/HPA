#!/bin/bash
# Patch inference for 10 new V4 questions (16631021, 56400000, 63617002, 82021002,
# 102872002, 156555004, 404479011, 436162001, 516902002, 533805004).
#
# Uses --json_path to load only these 10 questions and --fill_missing to merge
# results into existing logit files without touching already-completed question IDs.
#
# Run with:
#   nohup bash scripts/run_patch_v4.sh > logs/patch_v4.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/patch_v4.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()       { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== run_patch_v4.sh started ==="

PATCH_JSONL="dataset/vqa/vqa1k_v4_patch.jsonl"
SAVEDIR="evaluation/logits"
DATASET="vqa_1k"

# ── Helper: run one model × condition ────────────────────────────────────────
run_cond() {
    local MODEL="$1"
    local MODEL_TYPE="$2"
    local TEMPLATE="$3"
    local SWIFT_TYPE="$4"
    local CONDITION="$5"
    local LABEL="$6"

    local EXTRA=""
    [[ -n "$TEMPLATE"   ]] && EXTRA="$EXTRA --template_type $TEMPLATE"
    [[ -n "$SWIFT_TYPE" ]] && EXTRA="$EXTRA --swift_model_type $SWIFT_TYPE"

    log "--- $LABEL | condition='$CONDITION' ---"
    CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
        --model "$MODEL" \
        --model_type "$MODEL_TYPE" \
        --dataset "$DATASET" \
        --savedir "$SAVEDIR" \
        --condition "$CONDITION" \
        --json_path "$PATCH_JSONL" \
        --fill_missing \
        $EXTRA \
        >> "$LOG" 2>&1 \
        && log "✓ $LABEL | $CONDITION" \
        || { log "✗ FAILED $LABEL | $CONDITION"; return 1; }
}

# ── LLaVA-1.5-7B ─────────────────────────────────────────────────────────────
log "=== LLaVA-1.5-7B ==="
M="llava-hf/llava-1.5-7b-hf"; MT="vlm"; T=""; ST=""
for COND in "_control" "" "_control_blind" "_control_inst_blind"; do
    run_cond "$M" "$MT" "$T" "$ST" "$COND" "llava-1.5-7b" || exit 1
done

# ── LLaVA-v1.6-Mistral-7B ────────────────────────────────────────────────────
log "=== LLaVA-v1.6-Mistral-7B ==="
M="llava-hf/llava-v1.6-mistral-7b-hf"; MT="vlm"; T=""; ST=""
for COND in "_control" "" "_control_blind" "_control_inst_blind"; do
    run_cond "$M" "$MT" "$T" "$ST" "$COND" "llava-mistral-7b" || exit 1
done

# ── LLaVA-v1.6-Vicuna-7B ─────────────────────────────────────────────────────
log "=== LLaVA-v1.6-Vicuna-7B ==="
M="llava-hf/llava-v1.6-vicuna-7b-hf"; MT="vlm"; T=""; ST=""
for COND in "_control" "" "_control_blind" "_control_inst_blind"; do
    run_cond "$M" "$MT" "$T" "$ST" "$COND" "llava-vicuna-7b" || exit 1
done

# ── Qwen3-VL-8B-Instruct ─────────────────────────────────────────────────────
log "=== Qwen3-VL-8B-Instruct ==="
M="Qwen/Qwen3-VL-8B-Instruct"; MT="vlm"; T="qwen3_nothinking"; ST=""
for COND in "_control" "" "_control_blind" "_control_inst_blind"; do
    run_cond "$M" "$MT" "$T" "$ST" "$COND" "Qwen3-VL-8B" || exit 1
done

log "=== run_patch_v4.sh done ==="
