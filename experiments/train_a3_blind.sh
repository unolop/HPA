#!/bin/bash
# A3: Human Blind + Standard Cross-Entropy (No Confidence Weighting)
#
# Uses human responses with blind images but treats all examples equally.
# No confidence weighting - standard supervised learning.

set -e

# Configuration
MODEL_PATH="OpenGVLab/InternVL2-2B"
TRAIN_DATA="data/train/train_aggregated_train.jsonl"
VAL_DATA="data/train/train_aggregated_val.jsonl"
OUTPUT_DIR="outputs/checkpoints/A3_blind"
RUN_NAME="A3_human_blind_standard_ce"

# Training hyperparameters
LEARNING_RATE=2e-5
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=16
LORA_RANK=32
LORA_ALPHA=64

# Logging
SAVE_STEPS=50
EVAL_STEPS=50
LOGGING_STEPS=10

echo "========================================"
echo "A3: Human Blind + Standard CE"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Train data: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
echo "========================================"

python src/training/train_supervised.py \
    --ablation A3 \
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
