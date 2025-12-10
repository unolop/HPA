set -x

MODEL=${MODEL}

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model ${MODEL} \
    --model_name ${MODEL} \
    --train_type lora \
    --dataset annotations/VISPR/vispr_train_swift.jsonl \
    --val_dataset annotations/VISPR/vispr_val_swift.jsonl \
    --torch_dtype bfloat16 \
    --max_step 2000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 5 \
    --logging_steps 200 \
    --max_length 3072 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --output_dir checkpoint/${MODEL}_VISPR_LoRA_t3K \
    --truncation_strategy right \
    --max_pixels 448 \
    --load_best_model_at_end true \
    --eval_strategy steps \
    --use_hf true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --save_only_model false \
    --report_to wandb \
    --run_name ${MODEL}_VISPR_t3K \
    --resume_from_checkpoint /home/work/main/Privacy/checkpoint/Qwen/Qwen3-VL-4B-Instruct_VISPR_LoRA_t3K/v0-20251112-145715/checkpoint-1200