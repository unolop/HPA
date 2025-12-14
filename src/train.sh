#!/bin/bash
# Run k-fold cross-validation on all datasets and models (Standard SFT)

GPU_ID=1
MODELS=(
    # "Qwen/Qwen3-VL-8B-Instruct"
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "OpenGVLab/InternVL3_5-8B"
    # "Qwen/Qwen3-VL-4B-Instruct"
)
KFOLD_DIRS=(
    "/home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt"
    "/home/work/yuna/HPA/data/training/s1_text/kfold_15_blind_inst"
    "/home/work/yuna/HPA/data/training/s1_choice/kfold_15_blind_inst"
)
DATASET_NAMES=(
    "SFT_vqa_gt"
    "SFT_vqa_15_blind_inst"
    "SFT_mmstar_15_blind_inst"
)
for dataset_idx in "${!KFOLD_DIRS[@]}"; do
    kfold_dir="${KFOLD_DIRS[$dataset_idx]}"
    dataset_name="${DATASET_NAMES[$dataset_idx]}"

    for model_idx in "${!MODELS[@]}"; do
        model_path="${MODELS[$model_idx]}"

        output_dir="./output/SFT/${model_path}_${dataset_name}"
        run_name="${model_path}_${dataset_name}" 
        python train_kfold_sft.py \
            --kfold_dir "${kfold_dir}" \
            --model_path "${model_path}" \
            --output_base_dir "${output_dir}" \
            --run_name "${run_name}" \
            --gpu_id ${GPU_ID} \
            --num_epochs 5 \
            --max_steps -1 \
            --folds 0 \
            --learning_rate 2e-5 \
            --lora_rank 32 \
            --lora_alpha 64 \
            --batch_size 1 \
            --gradient_accumulation_steps 8 \
            --save_steps 40 \
            --eval_steps 40 \
            --logging_steps 20 \
            --max_pixels 448

    done
done
