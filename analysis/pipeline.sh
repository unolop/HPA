#!/bin/bash

dataset="mmstar" # "vqa_1k" #  
MODELS=(
    "InternVL3_5-8B"
)

# Loop through models
for MODEL_NAME in "${MODELS[@]}"; do

    python analysis/5_analyze_human_model.py \
        --human_responses /home/work/yuna/HPA/results/processed/individual_responses.json \
        --model_results /home/work/yuna/HPA/results/swift/${MODEL_NAME}_${dataset}_inst_blind.jsonl \
        --output_dir ./analysis/human_model_${MODEL_NAME}_${dataset}

    python analysis/6_analyze_calibration.py \
        --human_responses /home/work/yuna/HPA/results/processed/individual_responses.json \
        --model_results /home/work/yuna/HPA/results/swift/${MODEL_NAME}_${dataset}_inst_blind.jsonl \
        --output_dir ./analysis/calibration_${MODEL_NAME}_${dataset}

    # 4. Generate figures
    python analysis/7_visualize_results.py \
        --human_model_analysis ./analysis/human_model_${MODEL_NAME}_${dataset}/human_model_analysis.json \
        --calibration_analysis ./analysis/calibration_${MODEL_NAME}_${dataset}/calibration_analysis.json \
        --scored_results "/home/work/yuna/HPA/results/swift/scored/scored_results.json" \
        --output_dir ./figures/_${MODEL_NAME}_${dataset}
done

