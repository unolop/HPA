#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=1
DATASET_PATHS=( "spubench" "mmstar" "vqa_5k" "vqa_1k"  )  
CONDITION_LIST=("" "_inst_blind") #  "_sys_inst_blind"   "_blind" 
BASE_MODEL=(
    "llava-hf/llava-v1.6-mistral-7b-hf"
)

ADAPTERS_PATHS=(
    "/home/work/yuna/HPA/src/output/JS/llava-hf/llava-v1.6-mistral-7b-hf_A1_vqa_gt/fold_0/v0-20251213-133418/checkpoint-120"
    "/home/work/yuna/HPA/src/output/JS/llava-hf/llava-v1.6-mistral-7b-hf_A2_vqa_10_blind_inst/fold_0/v0-20251213-150521/checkpoint-120"
    "/home/work/yuna/HPA/src/output/JS/llava-hf/llava-v1.6-mistral-7b-hf_A4_mmstar_15_blind_inst/fold_0/v0-20251213-165045/checkpoint-80"
    "/home/work/yuna/HPA/src/output/SFT/llava-hf/llava-v1.6-mistral-7b-hf_SFT_mmstar_15_blind_inst/fold_0/v0-20251213-103318/checkpoint-40"
    "/home/work/yuna/HPA/src/output/SFT/llava-hf/llava-v1.6-mistral-7b-hf_SFT_vqa_15_blind_inst/fold_0/v0-20251213-062443/checkpoint-190"
    "/home/work/yuna/HPA/src/output/SFT/llava-hf/llava-v1.6-mistral-7b-hf_SFT_vqa_gt/fold_0/v2-20251212-225307/checkpoint-40" 
    "/home/work/yuna/HPA/src/output/JS/llava-hf/llava-v1.6-mistral-7b-hf_A3_vqa_15_blind_inst/fold_0/v0-20251213-155750/checkpoint-190"
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


