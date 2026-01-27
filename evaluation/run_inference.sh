#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=1
DATASET_PATHS=( "okvqa"  )  # "textvqa" "mmstar" "vqa_1k"  "vqa_5k" spubench 
CONDITION_LIST=("") #  "_sys_inst_blind" "_blind" "_inst_blind" 
BASE_MODEL=(
    # "OpenGVLab/InternVL3_5-1B"
    # "OpenGVLab/InternVL3_5-2B"
    "Qwen/Qwen3-VL-4B-Instruct"
    # "Qwen3-4B-Base"  
    # "Qwen3-0.6B-Base"  
    "llava-hf/llava-v1.6-vicuna-7b-hf"
    # "Qwen3-8B-Base"  
    # "llava-hf/llava-1.5-7b-hf"
    # "llava-hf/llava-v1.6-mistral-7b-hf"
    # "OpenGVLab/InternVL3_5-8B"
    # "Qwen/Qwen3-VL-8B-Instruct" 
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "OpenGVLab/InternVL3_5-4B"
    )

for BASE_MODEL in "${BASE_MODEL[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        for CONDITION in "${CONDITION_LIST[@]}"; do 

            CUDA_VISIBLE_DEVICES=${GPU} python /home/work/yuna/HPA/evaluation/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                # --resume
        done 
    done
done 