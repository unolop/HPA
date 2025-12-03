### PILOT 
python 1_preprocess_answers.py \
    --input_csvs /home/work/yuna/HPA/results/humans/pilot_cleaned/_all_pilot_cleaned.jsonl \
    --questions_csv /home/work/yuna/HPA/eda/questions/s1.csv \
    --output_dir /home/work/yuna/HPA/data/pilot
python data/2_prepare_training_data.py --responses_path /home/work/yuna/HPA/results/humans/pilot_cleaned/_all_pilot_cleaned.jsonl --questions_csv /home/work/yuna/HPA/eda/questions/s1.csv --output_dir /home/work/yuna/HPA/data/pilot --with_instruction # --create_split  

### ACTUAL DATA 
# # 1. Preprocess (with Korean translation caching)
python data/1_preprocess_answers.py --input_csvs /home/work/yuna/HPA/results/humans/all_results_20251202_112501/*/*.csv \
    --questions_csv /home/work/yuna/HPA/eda/questions/s1.csv --output_dir /home/work/yuna/HPA/results/processed \
    --translate --cache_file data/translation_cache.json

# 2. Create training data
python data/2_prepare_training_data.py --responses_path /home/work/yuna/HPA/results/processed/s1_choice.json \
    --questions_csv /home/work/yuna/HPA/eda/questions/s1.csv --output_dir /home/work/yuna/HPA/data/s1_mmstar \
    --with_instruction # --create_split 

python data/2_prepare_training_data.py --responses_path /home/work/yuna/HPA/results/processed/s1_text.json \
    --questions_csv /home/work/yuna/HPA/eda/questions/s1.csv --output_dir /home/work/yuna/HPA/data/s1_vqa \
    --with_instruction # --also_create_gt --images_dir /home/work/yuna/VLMEval/data/val2014  # --create_split  
    
# # # 3. Train (main method A5)
# CUDA_VISIBLE_DEVICES=1 python 3_train_soft_supervised.py --ablation A4 \
#     --model_path OpenGVLab/InternVL3_5-2B \
#     --train_data /home/work/yuna/HPA/data/s1_mmstar/train_aggregated.jsonl \
#     --output_dir yuna/HPA/checkpoint/A5 --run_name A5_InternVL3_5-2B_s1_mmstar

# python 3_train_soft_supervised.py --ablation A4 \
#     --model_path OpenGVLab/InternVL3_5-2B \
#     --train_data /home/work/yuna/HPA/data/s1_vqa/train_aggregated.jsonl \
#     --output_dir yuna/HPA/checkpoint/A5 --run_name A5_InternVL3_5-2B_s1_vqa

# # 4. Evaluate
# python 4_evaluate_models.py --model_path ./output/A5/checkpoint-* \
#     --base_model_path OpenGVLab/InternVL3_5-2B \
#     --test_data ./data/test.csv --output_dir ./eval/A5 --eval_blind

# # 5-6. Analyze
# python 5_analyze_human_model.py --human_responses ./processed/individual_responses.json \
#     --model_predictions ./eval/A5/eval_full.json --output_dir ./analysis

# python 6_analyze_calibration.py --human_responses ./processed/individual_responses.json \
#     --gt_path ./data/questions.csv --output_dir ./calibration
 
# # 7. Figures
# python 7_visualize_results.py --calibration_results ./calibration/*.json \
#     --comparison_results ./analysis/*.json --output_dir ./figures