#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=("spubench"  ) # "mmstar"  "vqa_1k"  "vqa_5k" 
CONDITION_LIST=("" ) #  "_sys_inst_blind" "_blind"  "_inst_blind" 
BASE_MODEL=(
    "Qwen/Qwen3-VL-8B-Instruct")

ADAPTERS_PATHS=(
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-8B-Instruct/vqa1k_374_alignment_js/v0-20251206-025413/checkpoint-5000"
    "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-8B-Instruct/vqa1k_374_standard/v1-20251206-143222/checkpoint-5000"
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-8B-Instruct/vqa1k_374_alignment_js_blind_mixed_8b/v0-20251207-102847/checkpoint-5000"
)
for ADAPTER in "${ADAPTERS_PATHS[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        for CONDITION in "${CONDITION_LIST[@]}"; do 

            CUDA_VISIBLE_DEVICES=${GPU} python /home/work/yuna/HPA/evaluation/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                --lora_path "${ADAPTER}" \
                --savedir "/home/work/yuna/HPA/data/finetuned" \
                --resume
        done 
    done
done 