#!/bin/bash

# Test all three model families with universal training script
MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "OpenGVLab/InternVL3_5-8B"
    "llava-hf/llava-v1.6-mistral-7b-hf"
)

data_path="/home/work/yuna/HPA/data/training/s1_choice/train_agg_15_blind_inst.jsonl"
val_data=""
RUNNAME="A1_JS_universal_test"

for MODEL in "${MODELS[@]}"; do
    model_name=$(basename ${MODEL})

    # Build validation argument conditionally
    val_arg=""
    if [ -n "$val_data" ]; then
        val_arg="--val_data_path ${val_data}"
    fi

    echo "=========================================="
    echo "Training: ${model_name}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=1 python train_universal.py \
        --model_path ${MODEL} \
        --data_path ${data_path} \
        --output_dir ./output/${model_name}/${RUNNAME} \
        ${val_arg} \
        --run_name ${model_name}/${RUNNAME} \
        --max_steps -1 \
        --num_epochs 10 \
        --mode JS \
        --lambda_dist 1.0 \
        --lambda_l2 0.1 \
        --use_l2_penalty \
        --learning_rate 2e-5 \
        --lora_rank 8 \
        --lora_alpha 16 \
        --batch_size 1 \
        --gradient_accumulation_steps 8
done
