#!/bin/bash
# https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html 

GPU=0
DATASET_PATHS=(
    # "vlm_vqav2_val_1k.jsonl" # LLM VQA 1K 
    # "vlm_vqav2_val_1k_blind.jsonl" # LLM VQA 1K 
    # # blind conditions
    # "vlm_mmstar_blind.jsonl"
    
    ## INSTRUCTIONS 
    # "vlm_vqav2_val_1k_inst_blind.jsonl"
    "vlm_mmstar_inst_blind.jsonl"
)
for VAL_DATASET in "${DATASET_PATHS[@]}"; do 
    CUDA_VISIBLE_DEVICES=$GPU \
    swift infer \
        --model "llava-hf/llava-v1.6-mistral-7b-hf" \
        --infer_backend pt \
        --model_type llava1_6_mistral_hf \
        --stream true \
        --torch_dtype bfloat16 \
        --temperature 0 \
        --result_path "/home/work/yuna/HPA/swift-results/llava-hf/llava-v1.6-mistral-7b-hf_${VAL_DATASET##*/}" \
        --val_dataset "/home/work/yuna/HPA/data/swift/${VAL_DATASET}" \
        --max_new_tokens 128
        
    CUDA_VISIBLE_DEVICES=$GPU \
    swift infer \
        --model "llava-hf/llava-1.5-7b-hf" \
        --infer_backend pt \
        --model_type llava1_5_hf \
        --stream true \
        --torch_dtype bfloat16 \
        --temperature 0 \
        --result_path "/home/work/yuna/HPA/swift-results/llava-hf/llava-1.5-7b-hf_${VAL_DATASET##*/}" \
        --val_dataset "/home/work/yuna/HPA/data/swift/${VAL_DATASET}" \
        --max_new_tokens 128
done