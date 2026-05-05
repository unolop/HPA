#!/bin/bash
# Run all missing / incomplete inference conditions as of 2026-05-02.
#
# Missing:
#   [pretrained VLM]
#     InternVL3_5-{1B,2B,8B}  + Qwen3-VL-2B  : vqa_1k (orig with image)
#     Qwen3-VL-32B             : all 4 conditions (partial, --fill_missing)
#
#   [lm_decoder]
#     Qwen3-VL-32B             : _control_blind (997/1000) + _control_inst_blind (0/1000)
#
#   [backbone]
#     Qwen3-32B                : _control_inst_blind
#     vicuna-13b-v1.5          : _control
#
# NOTE: lm_decoder models do NOT need vqa_1k_control — they are text-only
#       (no image is ever passed), so _control_blind IS their baseline.
#
# Run with:
#   nohup bash scripts/run_missing.sh > logs/missing.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/missing.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()       { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== run_missing.sh started ==="

DATASET="vqa_1k"
SAVEDIR="evaluation/logits"

# ── Helper ────────────────────────────────────────────────────────────────────
run_infer() {
    local GPUS="$1"; local MODEL="$2"; local MT="$3"
    local COND="$4"; local SDIR="$5"; local LABEL="$6"
    shift 6
    local EXTRA="$*"

    log "▶ [GPU $GPUS] $LABEL | $COND"
    CUDA_VISIBLE_DEVICES="$GPUS" python evaluation/inference.py \
        --model "$MODEL" --model_type "$MT" \
        --dataset "$DATASET" --condition "$COND" \
        --savedir "$SDIR" \
        $EXTRA \
        >> "$LOG" 2>&1 \
        && log "✓ $LABEL | $COND" \
        || { log "✗ FAILED $LABEL | $COND"; exit 1; }
}

# ═════════════════════════════════════════════════════════════════════════════
# 1. Pretrained VLM — vqa_1k (orig with image), single GPU each
# ═════════════════════════════════════════════════════════════════════════════
log "=== [1/5] Pretrained VLM: vqa_1k orig (InternVL + Qwen3-VL-2B) ==="

run_if_missing() {
    local OUT="$1"; shift
    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 1000 ]; then
        log "  SKIP (complete): $OUT"
    else
        run_infer "$@"
    fi
}

run_if_missing "$SAVEDIR/pretrained/InternVL3_5-1B/vqa_1k.jsonl" \
    "0" "OpenGVLab/InternVL3_5-1B" "vlm" "" "$SAVEDIR" "InternVL3_5-1B" --attn_impl eager

run_if_missing "$SAVEDIR/pretrained/InternVL3_5-2B/vqa_1k.jsonl" \
    "0" "OpenGVLab/InternVL3_5-2B" "vlm" "" "$SAVEDIR" "InternVL3_5-2B" --attn_impl eager

run_if_missing "$SAVEDIR/pretrained/InternVL3_5-8B/vqa_1k.jsonl" \
    "0" "OpenGVLab/InternVL3_5-8B" "vlm" "" "$SAVEDIR" "InternVL3_5-8B" --attn_impl eager

run_if_missing "$SAVEDIR/pretrained/Qwen3-VL-2B-Instruct/vqa_1k.jsonl" \
    "0" "Qwen/Qwen3-VL-2B-Instruct" "vlm" "" "$SAVEDIR" "Qwen3-VL-2B" --template_type qwen3_nothinking

# ═════════════════════════════════════════════════════════════════════════════
# 2. Pretrained VLM — Qwen3-VL-32B fill (all 4 conditions, both GPUs)
# ═════════════════════════════════════════════════════════════════════════════
log "=== [2/5] Pretrained VLM: Qwen3-VL-32B fill (both GPUs) ==="

COMMON_32B="--model Qwen/Qwen3-VL-32B-Instruct --model_type vlm
            --dataset $DATASET --savedir $SAVEDIR
            --template_type qwen3_nothinking --fill_missing"

log "▶ [GPU 0,1] Qwen3-VL-32B | _control (466/1000)"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    $COMMON_32B --condition _control >> "$LOG" 2>&1 \
    && log "✓ Qwen3-VL-32B | _control" || { log "✗ FAILED Qwen3-VL-32B | _control"; exit 1; }

log "▶ [GPU 0,1] Qwen3-VL-32B | vqa_1k orig (607/1000)"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    $COMMON_32B --condition "" >> "$LOG" 2>&1 \
    && log "✓ Qwen3-VL-32B | orig" || { log "✗ FAILED Qwen3-VL-32B | orig"; exit 1; }

log "▶ [GPU 0,1] Qwen3-VL-32B | _control_blind (888/1000)"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    $COMMON_32B --condition _control_blind >> "$LOG" 2>&1 \
    && log "✓ Qwen3-VL-32B | _control_blind" || { log "✗ FAILED Qwen3-VL-32B | _control_blind"; exit 1; }

log "▶ [GPU 0,1] Qwen3-VL-32B | _control_inst_blind (974/1000)"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    $COMMON_32B --condition _control_inst_blind >> "$LOG" 2>&1 \
    && log "✓ Qwen3-VL-32B | _control_inst_blind" || { log "✗ FAILED Qwen3-VL-32B | _control_inst_blind"; exit 1; }

# ═════════════════════════════════════════════════════════════════════════════
# 3. LM-decoder — Qwen3-VL-32B (both GPUs)
# ═════════════════════════════════════════════════════════════════════════════
log "=== [3/5] LM-decoder: Qwen3-VL-32B blind + inst_blind ==="

log "▶ [GPU 0,1] Qwen3-VL-32B lm | _control_blind (997/1000 — fill)"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "Qwen/Qwen3-VL-32B-Instruct" --model_type lm \
    --dataset "$DATASET" --condition _control_blind \
    --savedir "$SAVEDIR/lm_decoder" --fill_missing \
    >> "$LOG" 2>&1 \
    && log "✓ lm Qwen3-VL-32B | _control_blind" \
    || { log "✗ FAILED lm Qwen3-VL-32B | _control_blind"; exit 1; }

log "▶ [GPU 0,1] Qwen3-VL-32B lm | _control_inst_blind (0/1000)"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "Qwen/Qwen3-VL-32B-Instruct" --model_type lm \
    --dataset "$DATASET" --condition _control_inst_blind \
    --savedir "$SAVEDIR/lm_decoder" \
    >> "$LOG" 2>&1 \
    && log "✓ lm Qwen3-VL-32B | _control_inst_blind" \
    || { log "✗ FAILED lm Qwen3-VL-32B | _control_inst_blind"; exit 1; }

# ═════════════════════════════════════════════════════════════════════════════
# 4. Backbone — Qwen3-32B inst_blind (both GPUs)
# ═════════════════════════════════════════════════════════════════════════════
log "=== [4/5] Backbone: Qwen3-32B _control_inst_blind ==="

CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "Qwen/Qwen3-32B" --model_type lm \
    --dataset "$DATASET" --condition _control_inst_blind \
    --savedir "$SAVEDIR/backbone" \
    --template_type qwen3_nothinking \
    >> "$LOG" 2>&1 \
    && log "✓ backbone Qwen3-32B | _control_inst_blind" \
    || { log "✗ FAILED backbone Qwen3-32B | _control_inst_blind"; exit 1; }

# ═════════════════════════════════════════════════════════════════════════════
# 5. Backbone — vicuna-13b-v1.5 control (single GPU)
# ═════════════════════════════════════════════════════════════════════════════
log "=== [5/5] Backbone: vicuna-13b-v1.5 _control ==="

CUDA_VISIBLE_DEVICES=0,1 python evaluation/inference.py \
    --model "lmsys/vicuna-13b-v1.5" --model_type lm \
    --dataset "$DATASET" --condition _control \
    --savedir "$SAVEDIR/backbone" \
    --swift_model_type llama --fill_missing \
    >> "$LOG" 2>&1 \
    && log "✓ backbone vicuna-13b | _control" \
    || { log "✗ FAILED backbone vicuna-13b | _control"; exit 1; }

log "=== run_missing.sh done ==="
