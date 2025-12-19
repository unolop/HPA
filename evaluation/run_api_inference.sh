#!/bin/bash

# API Model Inference Runner
# Usage: ./run_api_inference.sh

# Set your API keys (or export them in your environment)
# export OPENAI_API_KEY="your-openai-api-key"
# export GOOGLE_API_KEY="your-google-api-key"

SAVEDIR="/home/work/yuna/HPA/evaluation/results"
DATASETS=("mmstar" "spubench" "vqa_1k")
CONDITIONS=("" "_blind" "_inst_blind")

# OpenAI GPT Models
GPT_MODELS=(
    "gpt-4o"
    "gpt-4o-mini"
    "gpt-4-turbo"
)

# Google Gemini Models
GEMINI_MODELS=(
    "gemini-1.5-pro"
    "gemini-1.5-flash"
)

echo "=================================================="
echo "API Model Inference Runner"
echo "=================================================="

# Function to run inference for a single configuration
run_inference() {
    local provider=$1
    local model=$2
    local dataset=$3
    local condition=$4
    local model_type=$5

    echo ""
    echo "Running: $provider / $model / $dataset$condition"
    echo "--------------------------------------------------"

    python evaluation/inference_api.py \
        --provider "$provider" \
        --model "$model" \
        --model_type "$model_type" \
        --dataset "$dataset" \
        --condition "$condition" \
        --savedir "$SAVEDIR" \
        --resume \
        --max_token_length 4096

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $provider / $model / $dataset$condition"
    else
        echo "✗ Failed: $provider / $model / $dataset$condition"
    fi
}

# Run GPT models
echo ""
echo "=========================================="
echo "Running OpenAI GPT Models"
echo "=========================================="

for model in "${GPT_MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for condition in "${CONDITIONS[@]}"; do
            # VLM mode (with images)
            run_inference "openai" "$model" "$dataset" "$condition" "vlm"
        done
    done
done

# Run Gemini models
echo ""
echo "=========================================="
echo "Running Google Gemini Models"
echo "=========================================="

for model in "${GEMINI_MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for condition in "${CONDITIONS[@]}"; do
            # VLM mode (with images)
            run_inference "gemini" "$model" "$dataset" "$condition" "vlm"
        done
    done
done

echo ""
echo "=================================================="
echo "All inference tasks completed!"
echo "=================================================="
echo "Results saved in: $SAVEDIR/api/"
