#!/bin/bash
# Run k-fold cross-validation training on a single dataset

# Usage: ./run_kfold_training.sh <kfold_dir> <model_path> <output_dir> <run_name> [gpu_id]

KFOLD_DIR=$1
MODEL_PATH=$2
OUTPUT_DIR=$3
RUN_NAME=$4
GPU_ID=${5:-0}

if [ -z "$KFOLD_DIR" ] || [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: $0 <kfold_dir> <model_path> <output_dir> <run_name> [gpu_id]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/kfold_vqa_gt \\"
    echo "     Qwen/Qwen3-VL-8B-Instruct \\"
    echo "     ./output/qwen_vqa_gt_kfold \\"
    echo "     qwen_vqa_gt_cv \\"
    echo "     0"
    exit 1
fi

echo "=========================================="
echo "K-Fold Cross-Validation Training"
echo "=========================================="
echo "K-fold dir:  ${KFOLD_DIR}"
echo "Model:       ${MODEL_PATH}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Run name:    ${RUN_NAME}"
echo "GPU:         ${GPU_ID}"
echo "=========================================="

python train_kfold.py \
    --kfold_dir "${KFOLD_DIR}" \
    --model_path "${MODEL_PATH}" \
    --output_base_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --gpu_id ${GPU_ID} \
    --num_epochs 10 \
    --max_steps -1 \
    --mode JS \
    --lambda_dist 1.0 \
    --lambda_l2 0.1 \
    --use_l2_penalty \
    --learning_rate 2e-5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_steps 40 \
    --eval_steps 40 \
    --logging_steps 20

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ K-fold training completed successfully!"
else
    echo ""
    echo "❌ K-fold training failed!"
    exit 1
fi
