#!/bin/bash
# GPU0 remediation — fills missing control types and completes interrupted runs
# Run with: bash logs/control_inference/remediate_gpu0.sh 2>&1 | tee logs/control_inference/remediate_gpu0.log
cd /home/david/Desktop/yuna/HPA
FILL4="--fill_missing --control_types question,deictic_removed,object_removed,weaker_object"
FILLP="--fill_missing --control_types pronominalized"

# ── LLaVA-1.5: _control has only pronominalized → fill base4 ──────────────────
echo "[$(date '+%H:%M:%S')] START llava-1.5 _control fill-base4"
CUDA_VISIBLE_DEVICES=0 python evaluation/inference.py \
  --model llava-hf/llava-1.5-7b-hf --condition _control --dataset vqa_1k $FILL4
echo "[$(date '+%H:%M:%S')] DONE  llava-1.5 _control fill-base4"

# ── LLaVA-Mistral: _control has only pronominalized → fill base4 ──────────────
echo "[$(date '+%H:%M:%S')] START llava-mistral _control fill-base4"
CUDA_VISIBLE_DEVICES=0 python evaluation/inference.py \
  --model llava-hf/llava-v1.6-mistral-7b-hf --condition _control --dataset vqa_1k $FILL4
echo "[$(date '+%H:%M:%S')] DONE  llava-mistral _control fill-base4"

# ── Vicuna-13b: _control and _inst_blind have base4 only → fill pron ──────────
for COND in _control _control_inst_blind; do
  echo "[$(date '+%H:%M:%S')] START vicuna-13b $COND fill-pron"
  CUDA_VISIBLE_DEVICES=0 python evaluation/inference.py \
    --model llava-hf/llava-v1.6-vicuna-13b-hf --condition $COND --dataset vqa_1k $FILLP
  echo "[$(date '+%H:%M:%S')] DONE  vicuna-13b $COND fill-pron"
done

# ── Vicuna-13b: _blind entirely missing → full run ────────────────────────────
echo "[$(date '+%H:%M:%S')] START vicuna-13b _control_blind full run"
CUDA_VISIBLE_DEVICES=0 python evaluation/inference.py \
  --model llava-hf/llava-v1.6-vicuna-13b-hf --condition _control_blind --dataset vqa_1k
echo "[$(date '+%H:%M:%S')] DONE  vicuna-13b _control_blind full run"

echo "[$(date '+%H:%M:%S')] GPU0 remediation complete."
