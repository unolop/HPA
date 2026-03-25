#!/bin/bash
cd /home/david/Desktop/yuna/HPA
for MODEL in llava-hf/llava-1.5-7b-hf llava-hf/llava-v1.6-mistral-7b-hf llava-hf/llava-v1.6-vicuna-7b-hf; do
  SHORT=$(echo $MODEL | sed 's|.*/||')
  for COND in _control _control_blind _control_inst_blind; do
    echo "[$(date '+%H:%M:%S')] START $SHORT $COND"
    CUDA_VISIBLE_DEVICES=0 python evaluation/inference.py \
      --model $MODEL --condition $COND --dataset vqa_1k \
      --fill_missing --control_types pronominalized
    echo "[$(date '+%H:%M:%S')] DONE $SHORT $COND"
  done
done
