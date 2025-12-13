#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=("spubench" "mmstar" "vqa_1k"  "vqa_5k"  )  # 
CONDITION_LIST=("" "_inst_blind") #  "_sys_inst_blind" "_blind"
BASE_MODEL=(
    # "llava-hf/llava-1.5-7b-hf"
    "OpenGVLab/InternVL3_5-8B"
    # "llava-hf/llava-v1.6-mistral-7b-hf"
    # "Qwen/Qwen3-VL-8B-Instruct"
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "llava-hf/llava-v1.6-vicuna-7b-hf"
    )
    
ADAPTERS_PATHS=(
    "/home/work/yuna/HPA/src/output/JS/llava-hf/llava-v1.6-mistral-7b-hf_A1_vqa_gt/fold_0/v0-20251213-133418/checkpoint-120"
    # "llava-v1.5-7b-Mixed-lora_VISPR_LoRA_r32_vlguard/v1-20251031-193648/checkpoint-2500"
)

for BASE_MODEL in "${BASE_MODEL[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        for CONDITION in "${CONDITION_LIST[@]}"; do 

            CUDA_VISIBLE_DEVICES=${GPU} python /home/work/yuna/HPA/evaluation/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                --
                --resume
        done 
    done
done 