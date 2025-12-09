#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=1
DATASET_PATHS=("mmstar")  #  "vqa_1k"  "vqa_5k" "spubench"  
CONDITION_LIST=("_blind" "" ) #  "_sys_inst_blind" "_inst_blind" 
BASE_MODEL=(
    # "llava-hf/llava-1.5-7b-hf"
    "OpenGVLab/InternVL3_5-4B"
    # "OpenGVLab/InternVL3_5-2B"
    # "OpenGVLab/InternVL3_5-1B"
    # "OpenGVLab/InternVL3_5-8B"
    
    ## 15gb 
    # "llava-hf/llava-v1.6-mistral-7b-hf"
    # "OpenGVLab/InternVL3_5-8B"
    # "Qwen/Qwen3-VL-8B-Instruct"
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "Qwen/Qwen3-VL-2B-Instruct"
    # "llava-hf/llava-v1.6-vicuna-7b-hf"
            )
# ADAPTERS_PATHS=(
    # "llava-v1.5-7b-Mixed-lora_VISPR_LoRA_r32_vlguard/v1-20251031-193648/checkpoint-2500"
# )
        # --adapters "${ROOT_DIR}${CHECKPOINT}" \
for BASE_MODEL in "${BASE_MODEL[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        for CONDITION in "${CONDITION_LIST[@]}"; do 

            CUDA_VISIBLE_DEVICES=${GPU} python /home/work/yuna/HPA/evaluation/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                --resume
        done 
    done
done 