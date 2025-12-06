#!/bin/bash

MODELS=(
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    # "OpenGVLab/InternVL3_5-8B"
)
data_path="/home/work/yuna/HPA/data/training/vqa1k_374.jsonl"
val_data="/home/work/yuna/HPA/data/training/vqa5k_agg.jsonl"

for MODEL_8B in "${MODELS[@]}"; do 

    CUDA_VISIBLE_DEVICES=0 python train_human_alignment.py \
        --model_path ${MODEL_8B} \
        --data_path ${data_path} \
        --val_data_path ${val_data} \
        --output_dir ./output/${MODEL_8B}/vqa1k_374_alignment_js \
        --run_name vqa1k_374_alignment_js \
        --max_steps 5000 \
        --num_epochs 50 \
        --mode JS \
        --lambda_dist 1.0 \
        --lambda_l2 0.1 \
        --use_l2_penalty \
        --learning_rate 2e-5 \
        --lora_rank 8 \
        --lora_alpha 16 \
        --batch_size 1 \
        --gradient_accumulation_steps 8

    CUDA_VISIBLE_DEVICES=0 python train_sft_standard.py \
        --model_path ${MODEL_8B} \
        --data_path ${data_path} \
        --val_data_path ${val_data} \
        --output_dir ./output/${MODEL_8B}/vqa1k_374_standard \
        --run_name sft_standard \
        --learning_rate 2e-5 \
        --max_steps 5000 \
        --num_epochs 50 \
        --lora_rank 8 \
        --lora_alpha 16 \
        --batch_size 1 \
        --gradient_accumulation_steps 8

done 