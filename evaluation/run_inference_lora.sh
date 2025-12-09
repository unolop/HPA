#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=( "mmstar" ) #  "vqa_1k" "spubench"   ) #   # "vqa_5k" 
CONDITION_LIST=("" "_inst_blind" ) #  "_sys_inst_blind" "_blind"  
BASE_MODEL=(
    "Qwen/Qwen3-VL-4B-Instruct"
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
    )

ADAPTERS_PATHS=(
    "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/A1_JS_vqa1k_n10_blind_inst/v0-20251209-052435/checkpoint-5000"
    "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/A1_JS_vqa1k_n15_blind_inst/v0-20251208-202445/checkpoint-5000"
    "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/D0_SFT_mmstar_gt/v0-20251209-083334/checkpoint-3300"
    "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/D1_JS_mmstar_n15_blind/v0-20251208-204719/checkpoint-5000"
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