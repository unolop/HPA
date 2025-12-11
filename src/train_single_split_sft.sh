#!/bin/bash
# Train on single train/val split with standard SFT (CE loss)

# Usage: ./train_single_split_sft.sh <dataset_name> <model_path> <gpu_id>

DATASET=$1
MODEL=$2
GPU_ID=${3:-0}

if [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
    echo "Usage: $0 <dataset_name> <model_path> [gpu_id]"
    echo ""
    echo "Dataset names:"
    echo "  vqa_gt"
    echo "  10_blind_inst"
    echo "  15_blind_inst"
    echo "  15_blind_inst_choice"
    echo ""
    echo "Example:"
    echo "  $0 vqa_gt Qwen/Qwen3-VL-8B-Instruct 0"
    exit 1
fi

# Map dataset name to path
case $DATASET in
    vqa_gt)
        DATA_DIR="/home/work/yuna/HPA/data/training/s1_text/single_vqa_gt"
        ;;
    10_blind_inst)
        DATA_DIR="/home/work/yuna/HPA/data/training/s1_text/single_10_blind_inst"
        ;;
    15_blind_inst)
        DATA_DIR="/home/work/yuna/HPA/data/training/s1_text/single_15_blind_inst"
        ;;
    15_blind_inst_choice)
        DATA_DIR="/home/work/yuna/HPA/data/training/s1_choice/single_15_blind_inst"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

TRAIN_PATH="${DATA_DIR}/train.jsonl"
VAL_PATH="${DATA_DIR}/val.jsonl"

if [ ! -f "$TRAIN_PATH" ]; then
    echo "Error: Training data not found at ${TRAIN_PATH}"
    echo "Run ./generate_single_splits_from_kfold.sh first"
    exit 1
fi

if [ ! -f "$VAL_PATH" ]; then
    echo "Error: Validation data not found at ${VAL_PATH}"
    echo "Run ./generate_single_splits_from_kfold.sh first"
    exit 1
fi

OUTPUT_DIR="./output/single_split/SFT_${MODEL##*/}_${DATASET}"
RUN_NAME="SFT_${MODEL##*/}_${DATASET}"

echo "=========================================="
echo "Single Split Training (Standard SFT)"
echo "=========================================="
echo "Dataset:     ${DATASET}"
echo "Model:       ${MODEL}"
echo "Train data:  ${TRAIN_PATH}"
echo "Val data:    ${VAL_PATH}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "GPU:         ${GPU_ID}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_sft_standard.py \
    --model_path "${MODEL}" \
    --data_path "${TRAIN_PATH}" \
    --val_data_path "${VAL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
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

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "Model saved to: ${OUTPUT_DIR}"
else
    echo ""
    echo "❌ Training failed!"
    exit 1
fi
