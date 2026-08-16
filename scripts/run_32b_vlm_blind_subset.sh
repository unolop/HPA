#!/bin/bash
# Run Qwen3-VL-32B VLM blind inference on the 113 human-study questions only.
# Builds a 113-question subset JSONL, then fills in the missing entries
# (36 already done; 77 remaining) using 4-bit quantization.
#
# Usage:
#   nohup bash scripts/run_32b_vlm_blind_subset.sh >> logs/run_32b_vlm_blind_subset.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

source /home/david/miniconda3/etc/profile.d/conda.sh
conda activate zero

LOG="logs/run_32b_vlm_blind_subset.log"
PYTHON="/home/david/miniconda3/envs/zero/bin/python"
CACHE="/home/david/Desktop/yuna/.cache/hf"
SUBSET_JSON="dataset/vqa/vqa1k_control_study_subset.jsonl"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()       { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

log "=== Step 1: build 113-question study subset JSONL ==="
$PYTHON - <<'PYEOF'
import json
import pandas as pd
from pathlib import Path

ROOT = Path(".")
pc   = pd.read_parquet(ROOT / "analysis/session2/exports/pair_cache_cleaned.parquet")
study_qids = set(pc["question_id"].astype(int).unique())

with open(ROOT / "dataset/vqa/vqa1k_control.jsonl") as f:
    all_entries = [json.loads(l) for l in f]

subset = [e for e in all_entries if int(e["question_id"]) in study_qids]
print(f"Subset size: {len(subset)} (expected 113)")

out = ROOT / "dataset/vqa/vqa1k_control_study_subset.jsonl"
with open(out, "w") as f:
    for e in subset:
        f.write(json.dumps(e) + "\n")
print(f"Saved: {out}")
PYEOF

log "=== Step 2: VLM blind inference on study subset (4-bit, fill missing) ==="
CUDA_VISIBLE_DEVICES=0,1 HF_HOME="$CACHE" TRANSFORMERS_CACHE="$CACHE" \
$PYTHON evaluation/inference.py \
    --model "Qwen/Qwen3-VL-32B-Instruct" \
    --model_type vlm \
    --dataset vqa_1k \
    --condition _control_blind \
    --json_path "$SUBSET_JSON" \
    --savedir "evaluation/logits/vlm" \
    --template_type qwen3_nothinking \
    --fill_missing \
    --quantization_bit 4 \
    2>&1 | tee -a "$LOG"

log "=== Done. Check evaluation/logits/vlm/pretrained/Qwen3-VL-32B-Instruct/vqa_1k_control_blind.jsonl ==="
