#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

DATASET_PATHS=(
    "vlm_vqav2_val_1k.jsonl" # LLM VQA 1K 
    "vlm_vqav2_val_1k_blind.jsonl" # LLM VQA 1K 
    # blind conditions
    "vlm_mmstar_blind.jsonl"
)

for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
    CUDA_VISIBLE_DEVICES=1 \
    swift infer \
        --model "llava-hf/llava-v1.6-mistral-7b-hf" \
        --infer_backend pt \
        --model_type llava1_6_mistral_hf \
        --stream true \
        --temperature 0 \
        --result_path "/home/work/yuna/HPA/swift-results/llava-hf/llava-v1.6-mistral-7b-hf_${VAL_DATASET##*/}" \
        --val_dataset "/home/work/yuna/HPA/data/swift/${VAL_DATASET}" \
        --max_new_tokens 128
done

for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
    CUDA_VISIBLE_DEVICES=1 \
    swift infer \
        --model "llava-hf/llava-v1.6-vicuna-7b-hf" \
        --infer_backend pt \
        --model_type llava1_6_vicuna_hf \
        --stream true \
        --temperature 0 \
        --result_path "/home/work/yuna/HPA/swift-results/llava-hf/llava-v1.6-mistral-7b-hf_${VAL_DATASET##*/}" \
        --val_dataset "/home/work/yuna/HPA/data/swift/${VAL_DATASET}" \
        --max_new_tokens 128
done