#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=("mmstar" "vqa_1k"  "vqa_5k"  )  # "spubench" 
CONDITION_LIST=("" ) #  "_sys_inst_blind" "_blind" "_inst_blind" 
BASE_MODEL=(
    "gemini-3-pro-preview"
    # "gpt-5.2" 
    )

for BASE_MODEL in "${BASE_MODEL[@]}"; do 
    for CONDITION in "${CONDITION_LIST[@]}"; do 
        for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
            python /home/work/yuna/HPA/evaluation/inference_api.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                --resume
            done 
    done 
done 