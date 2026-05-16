#!/bin/bash
# Fill all incomplete 32B inference conditions, then queue Qwen3-4B think blind.
#
# Missing/incomplete as of 2026-05-16:
#   backbone/Qwen3-32B         inst_blind        135/1000  (fill)
#   lm_decoder/Qwen3-VL-32B   blind             997/1000  (fill)
#   lm_decoder/Qwen3-VL-32B   inst_blind          0/1000  (new)
#   vlm/Qwen3-VL-32B          blind             888/1000  (fill)
#   vlm/Qwen3-VL-32B          inst_blind        974/1000  (fill)
#   vlm/Qwen3-VL-32B          control           137/1000  (fill)
#   backbone_think/Qwen3-4B   blind             172/1000  (fill — queued after 32B)
#
# Usage:
#   nohup bash scripts/run_fill_32b.sh > logs/fill_32b_new.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/fill_32b_new.log"
mkdir -p logs

PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
DATASET="vqa_1k"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== run_fill_32b.sh started ==="

run_infer() {
    local GPUS="$1" MODEL="$2" MT="$3" COND="$4" SDIR="$5" LABEL="$6"
    shift 6
    local EXTRA="$*"
    local OUT="$SDIR/pretrained/$(basename $(echo $MODEL | tr '/' '\n' | tail -1))/vqa_1k${COND}.jsonl"
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

# ── 1. backbone Qwen3-32B inst_blind (135 → 1000) ─────────────────────────────
log "=== [1/6] backbone Qwen3-32B inst_blind (fill) ==="
run_infer 0,1 \
    "Qwen/Qwen3-32B" lm _control_inst_blind \
    "evaluation/logits/backbone" "backbone/Qwen3-32B" \
    --template_type qwen3_nothinking --fill_missing --quantization_bit 4

# ── 2. lm_decoder Qwen3-VL-32B blind (997 → 1000) ────────────────────────────
log "=== [2/6] lm_decoder Qwen3-VL-32B blind (fill) ==="
run_infer 0,1 \
    "Qwen/Qwen3-VL-32B-Instruct" lm _control_blind \
    "evaluation/logits/lm_decoder" "lm_decoder/Qwen3-VL-32B" \
    --fill_missing --quantization_bit 4

# ── 3. lm_decoder Qwen3-VL-32B inst_blind (0 → 1000) ─────────────────────────
log "=== [3/6] lm_decoder Qwen3-VL-32B inst_blind (new) ==="
run_infer 0,1 \
    "Qwen/Qwen3-VL-32B-Instruct" lm _control_inst_blind \
    "evaluation/logits/lm_decoder" "lm_decoder/Qwen3-VL-32B" \
    --quantization_bit 4

# ── 4. vlm Qwen3-VL-32B blind (888 → 1000) ───────────────────────────────────
log "=== [4/6] vlm Qwen3-VL-32B blind (fill) ==="
run_infer 0,1 \
    "Qwen/Qwen3-VL-32B-Instruct" vlm _control_blind \
    "evaluation/logits/vlm" "vlm/Qwen3-VL-32B" \
    --template_type qwen3_nothinking --fill_missing --quantization_bit 4

# ── 5. vlm Qwen3-VL-32B inst_blind (974 → 1000) ──────────────────────────────
log "=== [5/6] vlm Qwen3-VL-32B inst_blind (fill) ==="
run_infer 0,1 \
    "Qwen/Qwen3-VL-32B-Instruct" vlm _control_inst_blind \
    "evaluation/logits/vlm" "vlm/Qwen3-VL-32B" \
    --template_type qwen3_nothinking --fill_missing --quantization_bit 4

# ── 6. vlm Qwen3-VL-32B control (137 → 1000) ─────────────────────────────────
log "=== [6/6] vlm Qwen3-VL-32B control with image (fill) ==="
run_infer 0,1 \
    "Qwen/Qwen3-VL-32B-Instruct" vlm _control \
    "evaluation/logits/vlm" "vlm/Qwen3-VL-32B" \
    --template_type qwen3_nothinking --fill_missing --quantization_bit 4

# ── 7. backbone_think Qwen3-4B blind (172 → 1000) ────────────────────────────
log "=== [7/7] backbone_think Qwen3-4B blind (fill — after 32B) ==="
run_infer 0,1 \
    "Qwen/Qwen3-4B" lm _control_blind \
    "evaluation/logits/backbone_think" "backbone_think/Qwen3-4B" \
    --fill_missing --quantization_bit 4

log "=== run_fill_32b.sh complete ==="
