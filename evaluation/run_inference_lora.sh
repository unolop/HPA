#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=1
DATASET_PATHS=( "spubench"   "mmstar"  "vqa_1k" "vqa_5k"  ) #   # ) #
CONDITION_LIST=("" "_inst_blind" "_blind"  ) #  "_sys_inst_blind" 
BASE_MODEL=(
    "Qwen/Qwen3-VL-4B-Instruct"
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
    
    "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/A1_JS_vqa_n15_blind_inst/v1-20251209-183814/checkpoint-470"
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/A1_JS_vqa_blind_n10/v1-20251209-204249/checkpoint-470"
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/D1_JS_mmstar_blind/v1-20251209-202250/checkpoint-340" 

    ### DONE 
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/A0_SFT_vqa_gt/v0-20251209-210552/checkpoint-470"
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/A1_JS_vqa_n15_blind_inst/v1-20251209-183814/checkpoint-470"
    
    ### OLD 
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/A1_JS_vqa1k_n10_blind_inst/v0-20251209-052435/checkpoint-5000"
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/A1_JS_vqa1k_n15_blind_inst/v0-20251208-202445/checkpoint-5000"
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/D0_SFT_mmstar_gt/v0-20251209-083334/checkpoint-3300"
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-4B-Instruct/D1_JS_mmstar_n15_blind/v0-20251208-204719/checkpoint-5000"
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