#!/bin/bash

dataset="mmstar" # "vqa_1k" #
MODELS=(
    "InternVL3_5-8B"
)

# Loop through models
for MODEL_NAME in "${MODELS[@]}"; do

    python src/analysis/analyze_human_model.py \
        --human_responses outputs/results/processed/individual_responses.json \
        --model_results outputs/results/swift/${MODEL_NAME}_${dataset}_inst_blind.jsonl \
        --output_dir ./outputs/analysis/human_model_${MODEL_NAME}_${dataset}

    python src/analysis/analyze_calibration.py \
        --human_responses outputs/results/processed/individual_responses.json \
        --model_results outputs/results/swift/${MODEL_NAME}_${dataset}_inst_blind.jsonl \
        --output_dir ./outputs/analysis/calibration_${MODEL_NAME}_${dataset}

    # 4. Generate figures
    python src/analysis/visualize_results.py \
        --human_model_analysis ./outputs/analysis/human_model_${MODEL_NAME}_${dataset}/human_model_analysis.json \
        --calibration_analysis ./outputs/analysis/calibration_${MODEL_NAME}_${dataset}/calibration_analysis.json \
        --scored_results "outputs/results/swift/scored/scored_results.json" \
        --output_dir ./outputs/figures/${MODEL_NAME}_${dataset}
done

