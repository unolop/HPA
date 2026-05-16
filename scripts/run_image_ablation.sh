#!/bin/bash
set -e
cd /home/david/Desktop/yuna/HPA

LOG_DIR="logs/image_ablation"
mkdir -p "$LOG_DIR"

MODEL="Qwen/Qwen3-VL-8B-Instruct"

echo "=== Image ablation: gray ===" | tee "$LOG_DIR/gray.log"
CUDA_VISIBLE_DEVICES=0 python evaluation/inference.py \
    --model "$MODEL" \
    --model_type vlm \
    --dataset vqa_1k \
    --condition _control_blind \
    --image_override gray \
    --savedir evaluation/logits/vlm \
    --quantization_bit 4 \
    2>&1 | tee -a "$LOG_DIR/gray.log"

echo "=== Image ablation: noise ===" | tee "$LOG_DIR/noise.log"
CUDA_VISIBLE_DEVICES=0 python evaluation/inference.py \
    --model "$MODEL" \
    --model_type vlm \
    --dataset vqa_1k \
    --condition _control_blind \
    --image_override noise \
    --savedir evaluation/logits/vlm \
    --quantization_bit 4 \
    2>&1 | tee -a "$LOG_DIR/noise.log"

echo "=== Done ===" | tee -a "$LOG_DIR/noise.log"
