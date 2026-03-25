#!/bin/bash
# GPU1 remediation — fills missing control types and completes interrupted runs
# Run with: bash logs/control_inference/remediate_gpu1.sh 2>&1 | tee logs/control_inference/remediate_gpu1.log
cd /home/david/Desktop/yuna/HPA
FILL4="--fill_missing --control_types question,deictic_removed,object_removed,weaker_object"
FILLP="--fill_missing --control_types pronominalized"

# ── Qwen4B: _blind has only pronominalized → fill base4 ──────────────────────
echo "[$(date '+%H:%M:%S')] START Qwen4B _control_blind fill-base4"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-4B-Instruct --condition _control_blind --dataset vqa_1k $FILL4
echo "[$(date '+%H:%M:%S')] DONE  Qwen4B _control_blind fill-base4"

# ── Qwen8B: _control and _inst_blind have only pronominalized → fill base4 ────
for COND in _control _control_inst_blind; do
  echo "[$(date '+%H:%M:%S')] START Qwen8B $COND fill-base4"
  CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
    --model Qwen/Qwen3-VL-8B-Instruct --condition $COND --dataset vqa_1k $FILL4
  echo "[$(date '+%H:%M:%S')] DONE  Qwen8B $COND fill-base4"
done

# ── Qwen2B: _blind interrupted at 15 records → resume ────────────────────────
echo "[$(date '+%H:%M:%S')] START Qwen2B _control_blind resume (from q16)"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-2B-Instruct --condition _control_blind --dataset vqa_1k --resume
echo "[$(date '+%H:%M:%S')] DONE  Qwen2B _control_blind resume"

# ── Qwen32B: _control interrupted at 43 records → resume, then fill pron ─────
# NOTE: 32B is large — ensure enough VRAM or use tensor parallelism if needed
echo "[$(date '+%H:%M:%S')] START Qwen32B _control resume"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-32B-Instruct --condition _control --dataset vqa_1k --resume
echo "[$(date '+%H:%M:%S')] DONE  Qwen32B _control resume"
echo "[$(date '+%H:%M:%S')] START Qwen32B _control fill-pron"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-32B-Instruct --condition _control --dataset vqa_1k $FILLP
echo "[$(date '+%H:%M:%S')] DONE  Qwen32B _control fill-pron"

# ── Qwen32B: _blind entirely missing → full run ───────────────────────────────
echo "[$(date '+%H:%M:%S')] START Qwen32B _control_blind full run"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-32B-Instruct --condition _control_blind --dataset vqa_1k
echo "[$(date '+%H:%M:%S')] DONE  Qwen32B _control_blind full run"

# ── Qwen32B: _inst_blind at 974/1000 base4 → resume, then fill pron ──────────
echo "[$(date '+%H:%M:%S')] START Qwen32B _control_inst_blind resume"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-32B-Instruct --condition _control_inst_blind --dataset vqa_1k --resume
echo "[$(date '+%H:%M:%S')] DONE  Qwen32B _control_inst_blind resume"
echo "[$(date '+%H:%M:%S')] START Qwen32B _control_inst_blind fill-pron"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-32B-Instruct --condition _control_inst_blind --dataset vqa_1k $FILLP
echo "[$(date '+%H:%M:%S')] DONE  Qwen32B _control_inst_blind fill-pron"

echo "[$(date '+%H:%M:%S')] GPU1 remediation complete."
