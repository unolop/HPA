#!/bin/bash
cd /home/david/Desktop/yuna/HPA
# Fill pronominalized for all 3 Qwen models (8B just finished, 2B/4B already done on GPU 0)
for MODEL in Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-VL-8B-Instruct; do
  SHORT=$(echo $MODEL | sed 's|.*/||')
  for COND in _control _control_blind _control_inst_blind; do
    echo "[$(date '+%H:%M:%S')] START fill-pron $SHORT $COND"
    CUDA_VISIBLE_DEVICES=1 python evaluation/inference.py \
      --model $MODEL --condition $COND --dataset vqa_1k \
      --fill_missing --control_types pronominalized
    echo "[$(date '+%H:%M:%S')] DONE fill-pron $SHORT $COND"
  done
done
