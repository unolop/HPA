#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=("spubench" ) # "mmstar"  "vqa_1k"  "vqa_5k" 
CONDITION_LIST=("" "_inst_blind" ) #  "_sys_inst_blind" "_blind"  
BASE_MODEL=(
    # "Qwen/Qwen3-VL-2B-Instruct"
    # "llava-hf/llava-v1.6-vicuna-7b-hf"
    # "llava-hf/llava-1.5-7b-hf"
    # "OpenGVLab/InternVL3_5-4B"
    # "OpenGVLab/InternVL3_5-2B"
    # "OpenGVLab/InternVL3_5-1B"
    # "OpenGVLab/InternVL3_5-8B"
    
    ## 15gb 
    # "llava-hf/llava-v1.6-mistral-7b-hf"
    # "OpenGVLab/InternVL3_5-8B"
    # "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-4B-Instruct"
    )

ADAPTERS_PATHS=(
    "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/vqa1k_374_alignment_js/v0-20251206-141535/checkpoint-4200"
    "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/vqa1k_374_alignment_js_blind_mixed/v5-20251207-011638/checkpoint-4320"
    "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/vqa1k_374_standard/v0-20251206-021720/checkpoint-4840"
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