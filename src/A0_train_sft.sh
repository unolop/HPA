#!/bin/bash

MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "llava-hf/llava-v1.6-mistral-7b-hf"
    # "OpenGVLab/InternVL3_5-8B"
)

data_path="/home/work/yuna/HPA/data/training/s1_text/train_agg_vqa_gt.jsonl" 
data_path="/home/work/yuna/HPA/data/training/s1_choice/train_agg_15_blind_inst.jsonl"
val_data=""
RUNNAME="D0_SFT_mmstar_gt" 

for MODEL in "${MODELS[@]}"; do

    CUDA_VISIBLE_DEVICES=1 python train_sft_standard.py \
        --model_path ${MODEL} \
        --data_path ${data_path} \
        --output_dir ./output/${MODEL}/${RUNNAME} \
        ${val_arg} \
        --run_name ${RUNNAME} \
        --max_steps -1 \
        --num_epochs 10 \
        --learning_rate 2e-5 \
        --lora_rank 8 \
        --lora_alpha 16 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --save_steps 100 \
        --eval_steps 100 \
        --logging_steps 20
    done

