#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=( "mmstar" "vqa_5k" "vqa_1k"  "spubench") 
CONDITION_LIST=("" "_inst_blind") #  "_sys_inst_blind"   "_blind" 
BASE_MODEL=(
    "Qwen/Qwen3-VL-8B-Instruct")

ADAPTERS_PATHS=(
    "/home/work/yuna/HPA/src/output/SFT/Qwen/Qwen3-VL-8B-Instruct_SFT_vqa_gt/fold_0/v3-20251212-132105/checkpoint-80"

    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A4_mmstar_15_blind_inst_K3/fold_3/v0-20251211-021230/checkpoint-200" 
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A4_mmstar_15_blind_inst_K3/fold_4/v0-20251211-024406/checkpoint-160" 

)
for ADAPTER in "${ADAPTERS_PATHS[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        for CONDITION in "${CONDITION_LIST[@]}"; do 

            CUDA_VISIBLE_DEVICES=${GPU} python /home/work/yuna/HPA/evaluation/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                --lora_path "${ADAPTER}" \
                --resume
        done 
    done
done 

