#!/bin/bash
# Quick test to verify all three models load and train correctly
# Runs only 10 steps per model

MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "OpenGVLab/InternVL3_5-8B"
    "llava-hf/llava-v1.6-mistral-7b-hf"
)

data_path="/home/work/yuna/HPA/data/training/s1_choice/train_agg_15_blind_inst.jsonl"

for MODEL in "${MODELS[@]}"; do
    model_name=$(basename ${MODEL})

    echo ""
    echo "=========================================="
    echo "Testing: ${model_name}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=1 python train_universal.py \
        --model_path ${MODEL} \
        --data_path ${data_path} \
        --output_dir ./output/test_${model_name} \
        --run_name test_${model_name} \
        --max_steps 10 \
        --mode JS \
        --lambda_dist 1.0 \
        --lambda_l2 0.1 \
        --use_l2_penalty \
        --learning_rate 2e-5 \
        --lora_rank 8 \
        --lora_alpha 16 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --save_steps 100 \
        --logging_steps 5

    if [ $? -eq 0 ]; then
        echo "✓ ${model_name} training successful"
    else
        echo "✗ ${model_name} training failed"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All models tested successfully!"
echo "=========================================="
