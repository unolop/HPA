# 1. Prepare data (convert CSVs)
python prepare_blind_vqa_data.py \
    --csv_files human_data/vqav2/*.csv \
    --questions_file data/vqav2/questions.json \
    --output_path data/vqav2_blind_train.jsonl \
    --aggregate --create_split

# 2. Run single ablation
python train_ablations.py \
    --ablation A5 \
    --model_path OpenGVLab/InternVL3_5-2B \
    --train_benchmark vqav2 \
    --human_csv_files human_data/vqav2/*.csv \
    --questions_path data/vqav2/questions.json \
    --output_dir ./output/ablations

# 3. Run all ablations
python train_ablations.py \
    --run_all \
    --model_path OpenGVLab/InternVL3_5-2B \
    ...

# 4. Evaluate
python evaluate_ablations.py \
    --model_path ./output/ablations/A5_Soft_Human_Blind_KL/vqav2/checkpoint-xxx \
    --base_model_path OpenGVLab/InternVL3_5-2B \
    --benchmark vqav2 \
    --eval_mode both