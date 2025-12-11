#!/bin/bash
# Run k-fold cross-validation on all datasets and models (Standard SFT)

GPU_ID=${1:-0}

echo "=========================================="
echo "Running All K-Fold CV Experiments (SFT)"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "=========================================="

# Define datasets (k-fold directories)
KFOLD_DIRS=(
    "/home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt"
    "/home/work/yuna/HPA/data/training/s1_text/kfold_10_blind_inst"
    "/home/work/yuna/HPA/data/training/s1_text/kfold_15_blind_inst"
    "/home/work/yuna/HPA/data/training/s1_choice/kfold_15_blind_inst"
)

# Dataset names for output
DATASET_NAMES=(
    "A1_vqa_gt"
    "A2_vqa_10_blind_inst"
    "A3_vqa_15_blind_inst"
    "A4_mmstar_15_blind_inst"
)

# Models to train
MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "OpenGVLab/InternVL3_5-8B"
    # "llava-hf/llava-v1.6-mistral-7b-hf"
)

# Model short names
MODEL_NAMES=(
    "Qwen3-VL-8B-Instruct"
    # "internvl"
    # "llava"
)

# Run all combinations
for dataset_idx in "${!KFOLD_DIRS[@]}"; do
    kfold_dir="${KFOLD_DIRS[$dataset_idx]}"
    dataset_name="${DATASET_NAMES[$dataset_idx]}"

    for model_idx in "${!MODELS[@]}"; do
        model_path="${MODELS[$model_idx]}"

        output_dir="./output/kfold/SFT_${model_path}_${dataset_name}_K${dataset_idx}"
        run_name="SFT_kfold_${model_path}_${dataset_name}_K${dataset_idx}"

        echo ""
        echo "=========================================="
        echo "Experiment: ${model_path} on ${dataset_name} (SFT)"
        echo "=========================================="

        python train_kfold_sft.py \
            --kfold_dir "${kfold_dir}" \
            --model_path "${model_path}" \
            --output_base_dir "${output_dir}" \
            --run_name "${run_name}" \
            --gpu_id ${GPU_ID} \
            --num_epochs 10 \
            --max_steps -1 \
            --learning_rate 2e-5 \
            --lora_rank 32 \
            --lora_alpha 64 \
            --batch_size 1 \
            --gradient_accumulation_steps 8 \
            --save_steps 40 \
            --eval_steps 40 \
            --logging_steps 20 \
            --max_pixels 448

        if [ $? -ne 0 ]; then
            echo "❌ Experiment failed: ${model_path} on ${dataset_name}"
            echo "Stopping all experiments."
            exit 1
        fi

        echo "✅ Completed: ${model_path} on ${dataset_name}"
    done
done

echo ""
echo "=========================================="
echo "All K-Fold SFT CV Experiments Completed!"
echo "=========================================="
echo ""
echo "Results saved in: ./output/kfold/"
echo ""
echo "Directory structure:"
echo "  ./output/kfold/"
echo "    ├── SFT_Qwen3-VL-8B-Instruct_A1_vqa_gt_K0/"
echo "    │   ├── fold_0/ (checkpoint, logs)"
echo "    │   ├── fold_1/"
echo "    │   ├── ..."
echo "    │   └── kfold_training_summary.json"
echo "    ├── SFT_Qwen3-VL-8B-Instruct_A2_vqa_10_blind_inst_K1/"
echo "    ├── ..."
echo ""
