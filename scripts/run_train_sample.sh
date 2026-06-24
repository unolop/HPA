#!/bin/bash
# Run VLM inference WITH images on the 1K training sample.
# This tests whether models perform differently on training vs validation images.
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/run_train_sample.sh
#
# Output: evaluation/logits/vlm/pretrained/{model}/vqa_1k_train.jsonl

set -e

TRAIN_JSON="/home/david/Desktop/yuna/HPA/dataset/vqa/vqa1k_train.json"
TRAIN_IMG="/home/david/Desktop/yuna/data/train2014"

cd /home/david/Desktop/yuna/HPA

run_model() {
    local model="$1"
    local extra_args="${2:-}"
    local save_name=$(echo "$model" | rev | cut -d'/' -f1 | rev)
    local out="evaluation/logits/vlm/pretrained/${save_name}/vqa_1k_train.jsonl"

    if [ -f "$out" ]; then
        lines=$(wc -l < "$out")
        if [ "$lines" -ge 1000 ]; then
            echo "SKIP $save_name — already has $lines lines"
            return
        fi
        echo "RESUME $save_name — has $lines/1000 lines"
    fi

    echo "=== Running $save_name ==="
    conda run -n zero python evaluation/inference.py \
        --model "$model" \
        --model_type vlm \
        --dataset vqa_1k_train \
        --condition "" \
        --json_path "$TRAIN_JSON" \
        --image_dir "$TRAIN_IMG" \
        --savedir "evaluation/logits/vlm" \
        --control_types "question" \
        --resume \
        $extra_args
}

# 7B VLMs (fit in fp16 on single 24GB GPU)
run_model "llava-hf/llava-1.5-7b-hf"
run_model "llava-hf/llava-v1.6-mistral-7b-hf"
run_model "llava-hf/llava-v1.6-vicuna-7b-hf"
run_model "Qwen/Qwen3-VL-8B-Instruct" "--template_type qwen3_nothinking"
run_model "OpenGVLab/InternVL3_5-8B" "--attn_impl eager"

echo "=== All done ==="
