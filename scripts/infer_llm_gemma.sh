#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

BASE_MODEL=(
            "google/gemma-3-270m"
            "google/gemma-3-270m-it"
            # "AI-ModelScope/gemma-2b"
            # "AI-ModelScope/gemma-7b"
            )
# ADAPTERS_PATHS=(
    # "llava-v1.5-7b-Mixed-lora_VISPR_LoRA_r32_vlguard/v1-20251031-193648/checkpoint-2500"
# )
        # --adapters "${ROOT_DIR}${CHECKPOINT}" \

DATASET_PATHS=(
    # blind conditions
    "llm_vqav2_val_1k.jsonl" # LLM VQA 1K 
    "llm_mmstar_blind.jsonl"
    # "/home/work/yuna/data/VLGuard/test_safe_unsafes.jsonl" ## MMStar 
    # "/home/work/yuna/data/VLGuard/test_safe_safes.jsonl" ## VQAv2 5k 
)

for BASE_MODEL in "${BASE_MODEL[@]}"; do 
    for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
        CUDA_VISIBLE_DEVICES=1 \
        swift infer \
            --model "${BASE_MODEL}" \
            --infer_backend pt \
            --model_type gemma3_text \
            --stream true \
            --temperature 0 \
            --result_path "./swift-results/${VAL_DATASET##*/}_${BASE_MODEL}" \
            --val_dataset "/home/work/yuna/HPA/data/swift/${VAL_DATASET}" \
            --max_new_tokens 128
    done
done 