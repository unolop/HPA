#!/bin/bash
cd /home/david/Desktop/yuna/HPA
FILL="--fill_missing --control_types question,deictic_removed,object_removed,weaker_object"

# Qwen2B: blind and inst_blind have only pronominalized → fill in the other 4 keys
for COND in _control_blind _control_inst_blind; do
  echo "[$(date '+%H:%M:%S')] START Qwen2B $COND"
  CUDA_VISIBLE_DEVICES=0 python evaluation/inference.py \
    --model Qwen/Qwen3-VL-2B-Instruct --condition $COND --dataset vqa_1k $FILL
  echo "[$(date '+%H:%M:%S')] DONE  Qwen2B $COND"
done

# LLaVA-1.5 and LLaVA-Mistral: _control has only pronominalized → fill in 4 keys
for MODEL in llava-hf/llava-1.5-7b-hf llava-hf/llava-v1.6-mistral-7b-hf; do
  SHORT=$(echo $MODEL | sed 's|.*/||')
  echo "[$(date '+%H:%M:%S')] START $SHORT _control"
  CUDA_VISIBLE_DEVICES=0 python evaluation/inference.py \
    --model $MODEL --condition _control --dataset vqa_1k $FILL
  echo "[$(date '+%H:%M:%S')] DONE  $SHORT _control"
done
