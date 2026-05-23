#!/bin/bash
# Re-run Vicuna backbone models with the correct longchat template.
#
# Root cause: previous runs used the llama (Llama-2 [INST]...[/INST]) template,
# which Vicuna doesn't understand — the model immediately outputs </s> (EOS).
# Fix: --template_type longchat → produces "USER: {prompt} ASSISTANT:" format.
#
# Models:
#   lmsys/vicuna-7b-v1.5   (~14 GB fp16, fits on 1 GPU)
#   lmsys/vicuna-13b-v1.5  (~26 GB fp16, needs 2 GPUs)
#
# backbone/ is identical to backbone/ for Vicuna (no thinking mode),
# so results are copied over after each run.
#
# Run AFTER the 32B job finishes (both GPUs must be free):
#   nohup bash scripts/run_vicuna_backbone.sh > logs/run_vicuna_backbone.log 2>&1 &

cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
DATASET="vqa_1k"
LOG="logs/run_vicuna_backbone.log"
mkdir -p logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run_infer() {
    local GPUS="$1" MODEL="$2" COND="$3" SDIR="$4" LABEL="$5"
    shift 5
    local EXTRA="$*"
    local SAVENAME
    SAVENAME=$(basename "$MODEL")
    local OUT="$SDIR/pretrained/$SAVENAME/${DATASET}${COND}.jsonl"

    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 1000 ]; then
        log "SKIP (complete): $LABEL | $COND"; return
    fi
    log "▶ [GPU $GPUS] $LABEL | $COND → $OUT"
    CUDA_VISIBLE_DEVICES="$GPUS" HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
    $PYTHON evaluation/inference.py \
        --model "$MODEL" --model_type lm \
        --swift_model_type llama \
        --template_type longchat \
        --dataset "$DATASET" --condition "$COND" \
        --savedir "$SDIR" \
        $EXTRA \
        >> "$LOG" 2>&1 \
        && log "✓ $LABEL | $COND" \
        || { log "✗ FAILED $LABEL | $COND"; exit 1; }
}

# ── vicuna-7b-v1.5 (GPU 0 only, ~14 GB) ──────────────────────────────────────
log "=== [1/4] vicuna-7b-v1.5 blind ==="
run_infer 0 "lmsys/vicuna-7b-v1.5" "_control_blind" "evaluation/logits/backbone" "backbone/vicuna-7b"

log "=== [2/4] vicuna-7b-v1.5 inst_blind ==="
run_infer 0 "lmsys/vicuna-7b-v1.5" "_control_inst_blind" "evaluation/logits/backbone" "backbone/vicuna-7b"

# ── vicuna-13b-v1.5 (both GPUs, ~26 GB) ──────────────────────────────────────
log "=== [3/4] vicuna-13b-v1.5 blind ==="
run_infer 0,1 "lmsys/vicuna-13b-v1.5" "_control_blind" "evaluation/logits/backbone" "backbone/vicuna-13b"

log "=== [4/4] vicuna-13b-v1.5 inst_blind ==="
run_infer 0,1 "lmsys/vicuna-13b-v1.5" "_control_inst_blind" "evaluation/logits/backbone" "backbone/vicuna-13b"

log "=== run_vicuna_backbone.sh complete ==="
