#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=1
DATASET_PATHS=( "mmstar" "vqa_5k" "vqa_1k"  "spubench") 
CONDITION_LIST=("" "_inst_blind") #  "_sys_inst_blind"   "_blind" 
BASE_MODEL=(
    "Qwen/Qwen3-VL-8B-Instruct")

ADAPTERS_PATHS=(
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A1_vqa_gt_K0/fold_0/v1-20251210-133942/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A1_vqa_gt_K0/fold_1/v0-20251210-142515/checkpoint-120"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A1_vqa_gt_K0/fold_2/v0-20251210-150941/checkpoint-120"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A1_vqa_gt_K0/fold_3/v0-20251210-155349/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A1_vqa_gt_K0/fold_4/v0-20251210-163739/checkpoint-120"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A2_vqa_10_blind_inst_K1/fold_0/v0-20251210-172126/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A2_vqa_10_blind_inst_K1/fold_1/v0-20251210-180500/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A2_vqa_10_blind_inst_K1/fold_2/v0-20251210-184820/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A2_vqa_10_blind_inst_K1/fold_3/v0-20251210-193153/checkpoint-380"
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A2_vqa_10_blind_inst_K1/fold_4/v0-20251210-201459/checkpoint-380" 
    # "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A3_vqa_15_blind_inst_K2/fold_0/v0-20251210-205810/checkpoint-380"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A3_vqa_15_blind_inst_K2/fold_1/v0-20251210-214144/checkpoint-120"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A3_vqa_15_blind_inst_K2/fold_2/v0-20251210-222519/checkpoint-240"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A3_vqa_15_blind_inst_K2/fold_3/v0-20251210-230915/checkpoint-160"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A3_vqa_15_blind_inst_K2/fold_4/v0-20251210-235332/checkpoint-120" 
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A4_mmstar_15_blind_inst_K3/fold_0/v0-20251211-003713/checkpoint-80"
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A4_mmstar_15_blind_inst_K3/fold_1/v0-20251211-010858/checkpoint-160" 
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A4_mmstar_15_blind_inst_K3/fold_2/v0-20251211-014043/checkpoint-160" 
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A4_mmstar_15_blind_inst_K3/fold_3/v0-20251211-021230/checkpoint-200" 
    "/home/work/yuna/HPA/src/output/kfold/JS_Qwen/Qwen3-VL-8B-Instruct_A4_mmstar_15_blind_inst_K3/fold_4/v0-20251211-024406/checkpoint-160" 

)
for ADAPTER in "${ADAPTERS_PATHS[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        for CONDITION in "${CONDITION_LIST[@]}"; do 

            CUDA_VISIBLE_DEVICES=${GPU} python /home/work/yuna/HPA/evaluation/inference.py \
                --model "${BASE_MODEL}" \
                --dataset "${VAL_DATASET}" \
                --condition "${CONDITION}" \
                --lora_path "${ADAPTER}" \
                --resume
        done 
    done
done 