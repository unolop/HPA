#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=( "spubench"   "mmstar"  "vqa_1k" "vqa_5k"  ) 
CONDITION_LIST=("" "_inst_blind") #  "_sys_inst_blind"  "_blind"  
BASE_MODEL=(
    "Qwen/Qwen3-VL-4B-Instruct"
    )

ADAPTERS_PATHS=(
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A1_vqa_gt_K0/fold_0/v0-20251210-134350/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A1_vqa_gt_K0/fold_1/v0-20251210-143135/checkpoint-160"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A1_vqa_gt_K0/fold_2/v0-20251210-151444/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A1_vqa_gt_K0/fold_3/v0-20251210-155838/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A1_vqa_gt_K0/fold_4/v0-20251210-164127/checkpoint-380"

    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A2_vqa_10_blind_inst_K1/fold_0/v0-20251210-172349/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A2_vqa_10_blind_inst_K1/fold_1/v0-20251210-181054/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A2_vqa_10_blind_inst_K1/fold_2/v0-20251210-185353/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A2_vqa_10_blind_inst_K1/fold_3/v0-20251210-193550/checkpoint-160"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A2_vqa_10_blind_inst_K1/fold_4/v0-20251210-201731/checkpoint-380"

    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A3_vqa_15_blind_inst_K2/fold_0/v0-20251210-210058/checkpoint-160"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A3_vqa_15_blind_inst_K2/fold_1/v0-20251210-214738/checkpoint-160"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A3_vqa_15_blind_inst_K2/fold_2/v0-20251210-222954/checkpoint-200"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A3_vqa_15_blind_inst_K2/fold_3/v0-20251210-231408/checkpoint-160"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A3_vqa_15_blind_inst_K2/fold_4/v0-20251210-235803/checkpoint-160"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A4_mmstar_15_blind_inst_K3/fold_0/v0-20251211-004134/checkpoint-80"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A4_mmstar_15_blind_inst_K3/fold_1/v0-20251211-011323/checkpoint-160"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A4_mmstar_15_blind_inst_K3/fold_2/v0-20251211-014541/checkpoint-120"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A4_mmstar_15_blind_inst_K3/fold_3/v0-20251211-021722/checkpoint-120"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-4B-Instruct_A4_mmstar_15_blind_inst_K3/fold_4/v0-20251211-024822/checkpoint-80"
    
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