#!/bin/bash
# Quick test script to check model configurations

echo "=========================================="
echo "Testing Model Configurations"
echo "=========================================="
echo ""

MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "OpenGVLab/InternVL3_5-8B"
    "llava-hf/llava-v1.6-mistral-7b-hf"
)

for MODEL in "${MODELS[@]}"; do
    echo "Testing: $MODEL"
    python model_configs.py --model_path "$MODEL" 2>&1 | grep -A 15 "Training Configuration"
    echo ""
    echo "---"
    echo ""
done

echo "=========================================="
echo "Configuration test complete"
echo "=========================================="
