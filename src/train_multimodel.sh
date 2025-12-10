#!/bin/bash
# Multi-Model Training Script
# Works with QwenVL, InternVL, and Llava

# Select models to train (uncomment desired models)
MODELS=(
    # "Qwen/Qwen3-VL-8B-Instruct"
    # "Qwen/Qwen3-VL-4B-Instruct"
    "OpenGVLab/InternVL3_5-8B"
    # "OpenGVLab/InternVL3_5-4B"
    # "OpenGVLab/InternVL3_5-2B"
    "llava-hf/llava-v1.6-mistral-7b-hf"
    # "llava-hf/llava-v1.6-vicuna-7b-hf"
)

# Data configuration
data_path="/home/work/yuna/HPA/data/training/s1_choice/train_agg_15_blind_inst.jsonl"
val_data=""
RUNNAME="D0_SFT_mmstar_gt"

# Training hyperparameters (optional - script uses model-specific defaults if not set)
# Uncomment to override defaults
# LEARNING_RATE="2e-5"
# LORA_RANK="8"
# LORA_ALPHA="16"
# MAX_PIXELS="448"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Training Model: ${MODEL}"
    echo "Run Name: ${RUNNAME}"
    echo "========================================================================"

    # Build validation argument
    val_arg=""
    if [ -n "$val_data" ]; then
        val_arg="--val_data_path ${val_data}"
    fi

    # Build optional override arguments
    override_args=""
    if [ -n "$LEARNING_RATE" ]; then
        override_args="$override_args --learning_rate $LEARNING_RATE"
    fi
    if [ -n "$LORA_RANK" ]; then
        override_args="$override_args --lora_rank $LORA_RANK"
    fi
    if [ -n "$LORA_ALPHA" ]; then
        override_args="$override_args --lora_alpha $LORA_ALPHA"
    fi
    if [ -n "$MAX_PIXELS" ]; then
        override_args="$override_args --max_pixels $MAX_PIXELS"
    fi

    # Check model info before training
    echo ""
    echo "Checking model-specific configuration..."
    python model_configs.py --model_path "${MODEL}"

    echo ""
    echo "Starting training..."
    echo ""

    # Run training
    CUDA_VISIBLE_DEVICES=0 python train_sft_multimodel.py \
        --model_path "${MODEL}" \
        --data_path "${data_path}" \
        --output_dir ./output/${MODEL}/${RUNNAME} \
        ${val_arg} \
        --run_name "${MODEL}/${RUNNAME}" \
        --max_steps -1 \
        --num_epochs 10 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --save_steps 100 \
        --eval_steps 100 \
        --logging_steps 20 \
        ${override_args}

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ Successfully trained: ${MODEL}"
        echo ""
    else
        echo ""
        echo "✗ Training failed for: ${MODEL} (exit code: $exit_code)"
        echo "Check logs for details"
        echo ""
        # Continue to next model instead of exiting
    fi
done

echo ""
echo "========================================================================"
echo "Training complete for all models"
echo "========================================================================"
