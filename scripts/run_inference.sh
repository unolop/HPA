#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0,1 
CONDITION_LIST=( "_control_inst_blind"  ) #  "_sys_inst_blind" "_blind" ""  "_control" 
BASE_MODEL=(
    # "Qwen3-4B-Base"  
    # "Qwen3-0.6B-Base"  
    # "Qwen3-8B-Base"  
    
    # "OpenGVLab/InternVL3_5-1B"
    # "OpenGVLab/InternVL3_5-2B" 
    # "OpenGVLab/InternVL3_5-4B"
    # "OpenGVLab/InternVL3_5-8B"

    # "llava-hf/llava-v1.6-vicuna-13b-hf"
    # "llava-hf/llava-v1.6-vicuna-7b-hf"
    # "llava-hf/llava-1.5-7b-hf"
    # "llava-hf/llava-v1.6-mistral-7b-hf"
    "Qwen/Qwen3-VL-32B-Instruct"

    # "Qwen/Qwen3-VL-8B-Instruct" 
    # "Qwen/Qwen3-VL-4B-Instruct"
    )

for BASE_MODEL in "${BASE_MODEL[@]}"; do 
    for CONDITION in "${CONDITION_LIST[@]}"; do 

        CUDA_VISIBLE_DEVICES=${GPU} python /home/david/Desktop/yuna/HPA/evaluation/inference.py \
            --model "${BASE_MODEL}" \
            --condition "${CONDITION}" \
            --dataset "vqa_1k"
            # --resume
    done 
done 