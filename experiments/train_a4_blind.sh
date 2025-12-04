#!/bin/bash
# A4: Human Blind + Confidence Weighting (Main Method without KL)
#
# Uses human responses with blind images.
# Confidence-weighted loss: higher confidence = higher weight.
# This is the recommended method when KL loss is not feasible.

set -e

# Configuration
MODEL_PATH="OpenGVLab/InternVL2-2B"
TRAIN_DATA="data/train/train_aggregated_train.jsonl"
VAL_DATA="data/train/train_aggregated_val.jsonl"
OUTPUT_DIR="outputs/checkpoints/A4_blind"
RUN_NAME="A4_human_blind_confidence"

# Training hyperparameters
LEARNING_RATE=2e-5
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=16
LORA_RANK=32
LORA_ALPHA=64

# Confidence weighting
WEIGHT_STRATEGY="linear"  # or "quadratic"
CONF_MIN_WEIGHT=0.2
CONF_MAX_WEIGHT=1.0

# Logging
SAVE_STEPS=50
EVAL_STEPS=50
LOGGING_STEPS=10

echo "========================================"
echo "A4: Human Blind + Confidence Weighting"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Train data: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
echo "Confidence weighting: $WEIGHT_STRATEGY ($CONF_MIN_WEIGHT - $CONF_MAX_WEIGHT)"
echo "========================================"

python src/training/train_supervised.py \
    --ablation A4 \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS

echo "✅ Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Evaluate: bash experiments/evaluate_model.sh A4"
echo "  2. Compare with A3: bash experiments/compare_ablations.sh"
