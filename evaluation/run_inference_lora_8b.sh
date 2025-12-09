#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=1
DATASET_PATHS=("mmstar" "vqa_5k" ) #  "vqa_1k" "spubench"   ) #   # 
CONDITION_LIST=("" "_inst_blind" ) #  "_sys_inst_blind" "_blind"  
BASE_MODEL=(
    "Qwen/Qwen3-VL-8B-Instruct")

ADAPTERS_PATHS=(
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-8B-Instruct/A1_JS_vqa1k_n10_blind_inst/v0-20251208-203338/checkpoint-5000"
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-8B-Instruct/A1_JS_vqa1k_n15_blind_inst/v0-20251208-201254/checkpoint-5000"
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-8B-Instruct/A2_JS_vqa_gt/v0-20251209-081458/checkpoint-3520" 
    "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-8B-Instruct/D1_JS_mmstar_n15_blind/v0-20251209-052322/checkpoint-5000" 

    ### INCOMPLETE 
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-8B-Instruct/A0_SFT_vqa_gt/v0-20251209-082856/checkpoint-3300"
    # "/home/work/yuna/HPA/src/output/Qwen/Qwen3-VL-8B-Instruct/D0_SFT_mmstar_gt/v0-20251209-140015/checkpoint-200" 

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