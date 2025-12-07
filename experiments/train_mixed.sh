#!/bin/bash
# Mixed Training: Human Blind + Original Dataset with Images
#
# Combines:
# - 80% Human responses (blind, confidence-weighted)
# - 20% Original dataset (with real images)
#
# This prevents catastrophic forgetting of visual capabilities
# while learning from human responses.

set -e

# Configuration
MODEL_PATH="OpenGVLab/InternVL2-2B"
HUMAN_DATA="data/train/train_aggregated_train.jsonl"
ORIGINAL_DATA="data/vqa5k_val.jsonl"  # Replace with your original dataset
OUTPUT_DIR="outputs/checkpoints/A4_mixed"
RUN_NAME="A4_human_blind_plus_visual"

# Mixing ratio
HUMAN_RATIO=0.8
ORIGINAL_RATIO=0.2

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
echo "Mixed Training: Blind + Visual"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Human data (blind): $HUMAN_DATA (${HUMAN_RATIO})"
echo "Original data (visual): $ORIGINAL_DATA (${ORIGINAL_RATIO})"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Step 1: Create mixed training file
MIXED_DATA="data/train/train_mixed.jsonl"

echo "Creating mixed dataset..."
python experiments/scripts/mix_datasets.py \
    --human_data "$HUMAN_DATA" \
    --original_data "$ORIGINAL_DATA" \
    --output "$MIXED_DATA" \
    --human_ratio $HUMAN_RATIO \
    --original_ratio $ORIGINAL_RATIO

# Step 2: Train on mixed data
echo "Training on mixed data..."
python src/training/train_supervised.py \
    --ablation A4 \
    --model_path "$MODEL_PATH" \
    --train_data "$MIXED_DATA" \
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

echo "✅ Mixed training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "This model should:"
echo "  ✓ Learn human response patterns from blind VQA"
echo "  ✓ Maintain visual grounding from original dataset"
