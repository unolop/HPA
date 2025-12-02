#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=("mmstar" ) # "spubench"  ) #     "vqa_1k"  "vqa_5k" ) # 
CONDITION_LIST=("_inst_blind" ""  ) #  "_sys_inst_blind" "_blind"

BASE_MODEL=(
    # "Qwen/Qwen3-VL-2B-Instruct"
    "OpenGVLab/InternVL3_5-8B"
    "OpenGVLab/InternVL3_5-4B"
    "OpenGVLab/InternVL3_5-2B"
    "OpenGVLab/InternVL3_5-1B"
    
    ## 15gb 
    # "llava-hf/llava-v1.6-mistral-7b-hf"
    # "OpenGVLab/InternVL3_5-8B"
    # "Qwen/Qwen3-VL-8B-Instruct"
    # "llava-hf/llava-v1.6-vicuna-7b-hf"
    # "Qwen/Qwen3-VL-4B-Instruct"

    # "llava-hf/llava-1.5-7b-hf"
            )
# ADAPTERS_PATHS=(
    # "llava-v1.5-7b-Mixed-lora_VISPR_LoRA_r32_vlguard/v1-20251031-193648/checkpoint-2500"
# )
        # --adapters "${ROOT_DIR}${CHECKPOINT}" \
for BASE_MODEL in "${BASE_MODEL[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        for CONDITION in "${CONDITION_LIST[@]}"; do 

            python /home/work/yuna/HPA/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                --gpu "${GPU}" 
        done 
    done
done 