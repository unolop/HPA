#!/bin/bash
cd /home/david/Desktop/yuna/HPA
FILL4="--fill_missing --control_types question,deictic_removed,object_removed,weaker_object"
FILLP="--fill_missing --control_types pronominalized"

# Qwen4B _control_blind: only pronominalized → fill in 4 keys
echo "[$(date '+%H:%M:%S')] START Qwen4B _control_blind"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-4B-Instruct --condition _control_blind --dataset vqa_1k $FILL4
echo "[$(date '+%H:%M:%S')] DONE  Qwen4B _control_blind"

# Qwen8B _control: will be 1000 pron-only after current fill-pron → fill in 4 keys
echo "[$(date '+%H:%M:%S')] START Qwen8B _control fill-4keys"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-8B-Instruct --condition _control --dataset vqa_1k $FILL4
echo "[$(date '+%H:%M:%S')] DONE  Qwen8B _control fill-4keys"

# Qwen8B _control_blind: has 5 keys, only missing pronominalized
echo "[$(date '+%H:%M:%S')] START Qwen8B _control_blind fill-pron"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-8B-Instruct --condition _control_blind --dataset vqa_1k $FILLP
echo "[$(date '+%H:%M:%S')] DONE  Qwen8B _control_blind fill-pron"

# Qwen8B _control_inst_blind: 994 records missing pron → fill pron, then resume last 6
echo "[$(date '+%H:%M:%S')] START Qwen8B _control_inst_blind fill-pron (994 records)"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-8B-Instruct --condition _control_inst_blind --dataset vqa_1k $FILLP
echo "[$(date '+%H:%M:%S')] DONE  Qwen8B _control_inst_blind fill-pron"
echo "[$(date '+%H:%M:%S')] START Qwen8B _control_inst_blind resume last 6"
CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
  --model Qwen/Qwen3-VL-8B-Instruct --condition _control_inst_blind --dataset vqa_1k --resume
echo "[$(date '+%H:%M:%S')] DONE  Qwen8B _control_inst_blind resume"
