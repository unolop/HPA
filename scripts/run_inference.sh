#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=1
DATASET_PATHS=( "mmstar" ) #"vqa_1k") #
CONDITION_LIST=("_inst_blind" "" ) #  "_sys_inst_blind" "_blind" 
BASE_MODEL=(
            # "llava-hf/llava-v1.6-mistral-7b-hf"
            # "llava-hf/llava-v1.6-vicuna-7b-hf" 
            # "llava-hf/llava-v1.5-7b-hf"

            # "Qwen/Qwen3-VL-8B-Instruct"
            # "OpenGVLab/InternVL3_5-8B"

            # "OpenGVLab/InternVL3_5-2B"
            # "Qwen/Qwen3-VL-2B-Instruct"
            # "OpenGVLab/InternVL3_5-1B"

            "OpenGVLab/InternVL3_5-4B"
            "Qwen/Qwen3-VL-4B-Instruct"

            )
# ADAPTERS_PATHS=(
    # "llava-v1.5-7b-Mixed-lora_VISPR_LoRA_r32_vlguard/v1-20251031-193648/checkpoint-2500"
# )
        # --adapters "${ROOT_DIR}${CHECKPOINT}" \

for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
    for BASE_MODEL in "${BASE_MODEL[@]}"; do 
        for CONDITION in "${CONDITION_LIST[@]}"; do 

            python /home/work/yuna/HPA/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                --gpu "${GPU}"
        done 
    done
done 