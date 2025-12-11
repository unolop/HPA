#!/bin/bash
# Train all datasets on single splits with JS divergence

GPU_ID=${1:-0}
MODEL=${2:-"Qwen/Qwen3-VL-8B-Instruct"}

DATASETS=(
    "vqa_gt"
    "10_blind_inst"
    "15_blind_inst"
    "15_blind_inst_choice"
)

echo "=========================================="
echo "Training All Datasets (Single Split, JS)"
echo "=========================================="
echo "Model: ${MODEL}"
echo "GPU:   ${GPU_ID}"
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training: ${dataset}"
    echo "=========================================="

    ./train_single_split_js.sh "${dataset}" "${MODEL}" ${GPU_ID}

    if [ $? -ne 0 ]; then
        echo "❌ Training failed for ${dataset}"
        echo "Stopping all training."
        exit 1
    fi

    echo "✅ Completed: ${dataset}"
done

echo ""
echo "=========================================="
echo "All Single Split JS Training Completed!"
echo "=========================================="
echo ""
echo "Results saved in: ./output/single_split/"
echo ""
