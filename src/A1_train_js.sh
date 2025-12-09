#!/bin/bash

MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "llava-hf/llava-v1.6-mistral-7b-hf"
    # "OpenGVLab/InternVL3_5-8B"
)

data_path="/home/work/yuna/HPA/data/training/s1_text/train_agg_15_blind_inst.jsonl"
data_path="/home/work/yuna/HPA/data/training/s1_text/train_agg_vqa_gt.jsonl" 
data_path="/home/work/yuna/HPA/data/training/s1_choice/train_agg_15_blind_inst.jsonl"

val_data=""
# val_data="/home/work/yuna/HPA/data/training/vqa5k_agg.jsonl"
# RUNNAME="D1_JS_mmstar_n15_blind"
RUNNAME="A2_JS_vqa_gt" 

for MODEL_8B in "${MODELS[@]}"; do
    # Build validation argument conditionally
    val_arg=""
    if [ -n "$val_data" ]; then
        val_arg="--val_data_path ${val_data}"
    fi

    CUDA_VISIBLE_DEVICES=1 python train_human_alignment.py \
        --model_path ${MODEL_8B} \
        --data_path ${data_path} \
        --output_dir ./output/${MODEL_8B}/${RUNNAME} \
        ${val_arg} \
        --run_name ${MODEL_8B}/${RUNNAME} \
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
    done 