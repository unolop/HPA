#!/bin/bash

MODELS=(
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    # "OpenGVLab/InternVL3_5-8B"
)

data_path="/home/work/yuna/HPA/data/training/vqa1k_374.jsonl"
data_path="/home/work/yuna/HPA/data/training/s1_text_10_blind_inst_mixed.jsonl"
data_path="/home/work/yuna/HPA/data/training/s1_choice/train_agg_14_blind_inst.jsonl"

val_data=""
# val_data="/home/work/yuna/HPA/data/training/vqa5k_agg.jsonl"
RUNNAME="mmstar_alignment_js_blind"

for MODEL_8B in "${MODELS[@]}"; do
    # Build validation argument conditionally
    val_arg=""
    if [ -n "$val_data" ]; then
        val_arg="--val_data_path ${val_data}"
    fi

    CUDA_VISIBLE_DEVICES=0 python train_human_alignment.py \
        --model_path ${MODEL_8B} \
        --data_path ${data_path} \
        --output_dir ./output/${MODEL_8B}/${RUNNAME} \
        ${val_arg} \
        --run_name ${RUNNAME} \
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