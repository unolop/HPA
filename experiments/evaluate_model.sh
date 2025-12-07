#!/bin/bash
# Evaluate a trained model on test set

set -e

if [ $# -eq 0 ]; then
    echo "Usage: bash evaluate_model.sh <ablation_name> [test_data]"
    echo ""
    echo "Example:"
    echo "  bash evaluate_model.sh A4"
    echo "  bash evaluate_model.sh A4 data/test/test_blind.jsonl"
    exit 1
fi

ABLATION=$1
TEST_DATA=${2:-"data/train/train_aggregated_val.jsonl"}  # Default to val set

MODEL_PATH="outputs/checkpoints/${ABLATION}_blind/final_model"
OUTPUT_DIR="outputs/results/eval_${ABLATION}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "   Train first: bash experiments/train_${ABLATION,,}_blind.sh"
    exit 1
fi

echo "========================================"
echo "Evaluating $ABLATION"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Test data: $TEST_DATA"
echo "Output: $OUTPUT_DIR"
echo "========================================"

python src/evaluation/inference.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --blind \
    --batch_size 4

echo ""
echo "✅ Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Check results:"
echo "  cat ${OUTPUT_DIR}/metrics.json"
