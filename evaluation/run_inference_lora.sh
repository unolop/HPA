#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=1
DATASET_PATHS=( "spubench"   "mmstar"  "vqa_1k" "vqa_5k"  ) 
CONDITION_LIST=("" "_inst_blind") #  "_sys_inst_blind"  "_blind"  
BASE_MODEL=(
    "Qwen/Qwen3-VL-4B-Instruct"
    )

ADAPTERS_PATHS=(
    "/home/work/yuna/HPA/src/output/SFT/Qwen/Qwen3-VL-4B-Instruct_SFT_vqa_15_blind_inst/fold_0/v0-20251212-070155/checkpoint-80"
    "/home/work/yuna/HPA/src/output/SFT/Qwen/Qwen3-VL-4B-Instruct_SFT_vqa_gt/fold_0/v0-20251212-020158/checkpoint-80"
    )

for ADAPTER in "${ADAPTERS_PATHS[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        for CONDITION in "${CONDITION_LIST[@]}"; do 

            CUDA_VISIBLE_DEVICES=${GPU} python /home/work/yuna/HPA/evaluation/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                --lora_path "${ADAPTER}" \
                --savedir "/home/work/yuna/HPA/evaluation/results" \
                --resume
        done 
    done
done 