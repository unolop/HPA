#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 
BASE_MODEL=(
    # "OpenGVLab/InternVL3_5-1B"
    # "OpenGVLab/InternVL3_5-2B"
    # "OpenGVLab/InternVL3_5-4B"
    "OpenGVLab/InternVL3_5-8B"
            )
# ADAPTERS_PATHS=(
    # "llava-v1.5-7b-Mixed-lora_VISPR_LoRA_r32_vlguard/v1-20251031-193648/checkpoint-2500"
# )
        # --adapters "${ROOT_DIR}${CHECKPOINT}" \

DATASET_PATHS=(
    # "vlm_vqav2_val_1k.jsonl" # LLM VQA 1K 
    # "vlm_vqav2_val_1k_blind.jsonl" # LLM VQA 1K 
    # blind conditions
    "vlm_mmstar_blind.jsonl"
)

for BASE_MODEL in "${BASE_MODEL[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        CUDA_VISIBLE_DEVICES=0 \
        swift infer \
            --model "${BASE_MODEL}" \
            --infer_backend pt \
            --model_type internvl3_5 \
            --stream true \
            --temperature 0 \
            --result_path "/home/work/yuna/HPA/swift-results/${BASE_MODEL}_${VAL_DATASET##*/}" \
            --val_dataset "/home/work/yuna/HPA/data/swift/${VAL_DATASET}" \
            --max_new_tokens 128
    done
done 