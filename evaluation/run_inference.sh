#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=( "okvqa" "textvqa"  )  # "mmstar" "vqa_1k"  "vqa_5k" spubench 
CONDITION_LIST=("_inst_blind" ) #  "_sys_inst_blind" "_blind" ""
BASE_MODEL=(
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "OpenGVLab/InternVL3_5-4B"
    
    # "llava-hf/llava-1.5-7b-hf"
    # "llava-hf/llava-v1.6-mistral-7b-hf"
    # "llava-hf/llava-v1.6-vicuna-7b-hf"
    "OpenGVLab/InternVL3_5-1B"
    "OpenGVLab/InternVL3_5-2B"
    "Qwen/Qwen3-VL-2B-Instruct"
    "OpenGVLab/InternVL3_5-8B"
    "Qwen/Qwen3-VL-8B-Instruct" 
    )

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