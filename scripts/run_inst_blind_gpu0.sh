#!/bin/bash
# GPU 0: inst_blind for Qwen3 backbone (nothinking re-runs + 32B new)
# Then GPU 0+1: vicuna-13b, Qwen3-32B (after gpu1 script finishes)
#
#   nohup bash scripts/run_inst_blind_gpu0.sh > logs/inst_blind_gpu0.log 2>&1 &
#   nohup bash scripts/run_inst_blind_gpu1.sh > logs/inst_blind_gpu1.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/inst_blind_gpu0.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

BB_DIR="evaluation/logits/backbone"
HF_CACHE="/home/david/Desktop/yuna/.cache/hf"
COND="_control_inst_blind"

run_bb() {
    local GPUS="$1" MODEL_ID="$2" SAVE_NAME="$3"
    shift 3
    local OUT="$BB_DIR/pretrained/$SAVE_NAME/vqa_1k${COND}.jsonl"

    # skip if already complete (1000 lines) AND clean (no think tags)
    if [ -f "$OUT" ]; then
        N=$(wc -l < "$OUT")
        HAS_THINK=$(head -1 "$OUT" | python3 -c "import sys,json; r=json.loads(sys.stdin.read()); v=list(r.get('generated_answers',{}).values()); print('1' if v and '<think>' in str(v[0]) else '0')" 2>/dev/null)
        if [ "$N" -ge 1000 ] && [ "$HAS_THINK" = "0" ]; then
            log "SKIP (done, clean): $SAVE_NAME"
            return
        fi
        log "RE-RUN (think tags present or incomplete): $SAVE_NAME  [${N}/1000]"
    else
        log "NEW: $SAVE_NAME"
    fi

    log "▶ [GPU $GPUS] $SAVE_NAME | $COND"
    CUDA_VISIBLE_DEVICES="$GPUS" \
    TRANSFORMERS_CACHE="$HF_CACHE" HF_HOME="$HF_CACHE" \
    python evaluation/inference.py \
        --model      "$MODEL_ID" \
        --model_type lm \
        --dataset    vqa_1k \
        --condition  "$COND" \
        --savedir    "$BB_DIR" \
        "$@" \
        >> "$LOG" 2>&1 \
        && log "✓ $SAVE_NAME" \
        || { log "✗ FAILED $SAVE_NAME"; exit 1; }
}

log "=== run_inst_blind_gpu0.sh started ==="

# ── §1  Qwen3 small nothinking (GPU 0) ───────────────────────────────────────
# 1.7B already clean → skipped automatically by the check above
log "=== [1/3] Qwen3 small — nothinking inst_blind (GPU 0) ==="
run_bb 0 "Qwen/Qwen3-0.6B" "Qwen3-0.6B" --template_type qwen3_nothinking
run_bb 0 "Qwen/Qwen3-1.7B" "Qwen3-1.7B" --template_type qwen3_nothinking
run_bb 0 "Qwen/Qwen3-4B"   "Qwen3-4B"   --template_type qwen3_nothinking
run_bb 0 "Qwen/Qwen3-8B"   "Qwen3-8B"   --template_type qwen3_nothinking

# ── §2  vicuna-13b (both GPUs — after gpu1 finishes vicuna-7b) ───────────────
log "=== [2/3] vicuna-13b inst_blind (GPU 0,1) ==="
run_bb 0,1 "lmsys/vicuna-13b-v1.5" "vicuna-13b-v1.5" --swift_model_type llama

# ── §3  Qwen3-32B nothinking (both GPUs) ─────────────────────────────────────
log "=== [3/3] Qwen3-32B inst_blind nothinking (GPU 0,1) ==="
run_bb 0,1 "Qwen/Qwen3-32B" "Qwen3-32B" --template_type qwen3_nothinking

log "=== run_inst_blind_gpu0.sh complete ==="
