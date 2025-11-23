#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU="1" 
CONDITION_LIST=("_inst_blind" ) # "_blind" 
DATASET_PATHS=("vqa_1k" "mmstar" ) 
BASE_MODEL=(
            # "internlm/internlm2_5-1_8b"
            "internlm/internlm3-8b-instruct" 
            )

for CONDITON in "${CONDITION_LIST[@]}"; do 
    for BASE_MODEL in "${BASE_MODEL[@]}"; do 
        for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
            python /home/work/yuna/HPA/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --model_type "llm" \
                --gpu "${GPU}"
        done
    done 
done 