#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=( "spubench" )  #   "mmstar"  "vqa_1k" "vqa_5k"   
CONDITION_LIST=("") #  "_inst_blind" "_sys_inst_blind"  "_blind"  
BASE_MODEL=(
    "Qwen/Qwen3-VL-4B-Instruct"
    )
/home/work/yuna/HPA/evaluation/results/finetuned/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_A1_JS_vqa_n15_blind_inst/spubench.jsonl 
ADAPTERS_PATHS=(
    "/home/work/yuna/HPA/src/output/JS/Qwen/Qwen3-VL-4B-Instruct_A1_JS_vqa_n15_blind_inst

    # "/home/work/yuna/HPA/src/output/JS/Qwen/Qwen3-VL-4B-Instruct_A2_vqa_10_blind_inst/fold_0/v0-20251212-154601/checkpoint-160"
    "/home/work/yuna/HPA/src/output/JS/Qwen/Qwen3-VL-4B-Instruct_A4_mmstar_15_blind_inst/fold_0/v0-20251212-163818/checkpoint-80"
    "/home/work/yuna/HPA/src/output/SFT/Qwen/Qwen3-VL-4B-Instruct_SFT_vqa_15_blind_inst/fold_0/v0-20251212-070155/checkpoint-80"
    "/home/work/yuna/HPA/src/output/SFT/Qwen/Qwen3-VL-4B-Instruct_SFT_vqa_gt/fold_0/v0-20251212-020158/checkpoint-80"
    "/home/work/yuna/HPA/src/output/JS/Qwen/Qwen3-VL-4B-Instruct_A3_vqa_15_blind_inst/fold_0/v0-20251212-161156/checkpoint-190"
    "/home/work/yuna/HPA/src/output/JS/Qwen/Qwen3-VL-4B-Instruct_A1_vqa_gt/fold_0/v0-20251212-152247/checkpoint-190"
    )

for ADAPTER in "${ADAPTERS_PATHS[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        for CONDITION in "${CONDITION_LIST[@]}"; do 

            CUDA_VISIBLE_DEVICES=${GPU} python /home/work/yuna/HPA/evaluation/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                --lora_path "${ADAPTER}" \
                --savedir "/home/work/yuna/HPA/evaluation/results"
        done 
    done
done 