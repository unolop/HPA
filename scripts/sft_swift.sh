
MODEL="llava-hf/llava-v1.6-mistral-7b-hf" # "Qwen/Qwen3-VL-2B-Instruct" # "llava-hf/llava-v1.6-vicuna-7b-hf" 
LR=1e-5              # recommended for tiny dataset
EPOCH=3              # 2–3 is ideal

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model $MODEL \
    --model_name $MODEL \
    --train_type lora \
    --dataset /home/work/yuna/HPA/data/swift/s1_vqa.jsonl \
    --val_dataset /home/work/yuna/HPA/data/swift/vqa5k_val.jsonl \
    --torch_dtype bfloat16 \
    --max_step 300 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate $LR \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 40 \
    --save_steps 40 \
    --save_total_limit 2 \
    --logging_steps 20 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --output_dir ${MODEL}_vqa_lr_${LR} \
    --truncation_strategy right \
    --max_pixels 448 \
    --load_best_model_at_end true \
    --eval_strategy steps \
    --use_hf true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --save_only_model false
