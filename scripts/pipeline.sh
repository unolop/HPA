#!/bin/bash
# Main pipeline for HPA project
# Updated paths for reorganized directory structure

### PILOT
python src/preprocessing/preprocess.py \
    --input_csvs outputs/results/humans/pilot_cleaned/_all_pilot_cleaned.jsonl \
    --questions_csv data/questions/s1.csv \
    --output_dir outputs/results/humans/pilot_cleaned/pilot

python src/preprocessing/prepare_training_data.py \
    --processed_dir outputs/results/humans/pilot_cleaned/pilot \
    --questions_csv data/questions/s1.csv \
    --output_dir data/pilot \
    --with_instruction # --create_split

### ACTUAL DATA
# # 1. Preprocess (with Korean translation caching)
python src/preprocessing/preprocess.py \
    --input_csvs outputs/results/humans/all_results_20251202_112501/*/*.csv \
    --questions_csv data/questions/s1.csv \
    --output_dir outputs/results/processed \
    --translate --cache_file data/translation_cache.json

# 2. Create training data
python src/preprocessing/prepare_training_data.py \
    --responses_path outputs/results/processed/s1_choice.json \
    --questions_csv data/questions/s1.csv \
    --output_dir data/s1_mmstar \
    --with_instruction # --create_split

python src/preprocessing/prepare_training_data.py \
    --responses_path outputs/results/processed/s1_text.json \
    --questions_csv data/questions/s1.csv \
    --output_dir data/s1_vqa \
    --with_instruction # --also_create_gt --images_dir /path/to/val2014  # --create_split

# # 3. Train (main method A5)
# CUDA_VISIBLE_DEVICES=1 python src/training/train_supervised.py --ablation A5 \
#     --model_path OpenGVLab/InternVL3_5-2B \
#     --train_data data/s1_mmstar/train_aggregated.jsonl \
#     --output_dir outputs/checkpoints/A5 --run_name A5_InternVL3_5-2B_s1_mmstar

# python src/training/train_supervised.py --ablation A5 \
#     --model_path OpenGVLab/InternVL3_5-2B \
#     --train_data data/s1_vqa/train_aggregated.jsonl \
#     --output_dir outputs/checkpoints/A5 --run_name A5_InternVL3_5-2B_s1_vqa

# # 4. Evaluate
# python src/evaluation/evaluate.py --model_path ./outputs/checkpoints/A5/checkpoint-* \
#     --base_model_path OpenGVLab/InternVL3_5-2B \
#     --test_data ./data/test.csv --output_dir ./outputs/evaluation/A5 --eval_blind

# # 5-6. Analyze
# python src/analysis/analyze_human_model.py \
#     --human_responses outputs/results/processed/individual_responses.json \
#     --model_predictions outputs/evaluation/A5/eval_full.json \
#     --output_dir outputs/analysis

# python src/analysis/analyze_calibration.py \
#     --human_responses outputs/results/processed/individual_responses.json \
#     --gt_path data/questions/s1.csv \
#     --output_dir outputs/analysis

# # 7. Figures
# python src/analysis/visualize_results.py \
#     --calibration_results outputs/analysis/*.json \
#     --comparison_results outputs/analysis/*.json \
#     --output_dir outputs/figures