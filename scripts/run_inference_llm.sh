#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

CONDITION_LIST=("_inst_blind" "_blind") 
DATASET_PATHS=("vqa_1k" "mmstar" ) 
BASE_MODEL=(
            "Qwen/Qwen3-4B-Base"
            "Qwen/Qwen3-1.7B-Base" 
            "Qwen/Qwen3-0.6B-Base" # doesnt work 
            "Qwen/Qwen3-8B-Base"
            )

for CONDITON in "${CONDITION_LIST[@]}"; do 
    for BASE_MODEL in "${BASE_MODEL[@]}"; do 
        for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
            CUDA_VISIBLE_DEVICES=1 python /home/work/yuna/HPA/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --model_type "llm"
        done
    done 
done 