#!/bin/bash
# Generate k-fold splits for all training datasets

K=5  # Number of folds
SEED=42

# Define datasets
DATASETS=(
    "/home/work/yuna/HPA/data/training/s1_text/train_agg_vqa_gt.jsonl"
    "/home/work/yuna/HPA/data/training/s1_text/train_agg_10_blind_inst.jsonl"
    "/home/work/yuna/HPA/data/training/s1_text/train_agg_15_blind_inst.jsonl"
    "/home/work/yuna/HPA/data/training/s1_choice/train_agg_15_blind_inst.jsonl"
)

# Output directories
OUTPUT_DIRS=(
    "/home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt"
    "/home/work/yuna/HPA/data/training/s1_text/kfold_10_blind_inst"
    "/home/work/yuna/HPA/data/training/s1_text/kfold_15_blind_inst"
    "/home/work/yuna/HPA/data/training/s1_choice/kfold_15_blind_inst"
)

echo "=========================================="
echo "Creating ${K}-fold splits for all datasets"
echo "=========================================="

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    output_dir="${OUTPUT_DIRS[$i]}"

    echo ""
    echo "Processing: $(basename ${dataset})"
    echo "Output: ${output_dir}"

    python create_kfold_splits.py \
        --input_path "${dataset}" \
        --output_dir "${output_dir}" \
        --k ${K} \
        --seed ${SEED}

    if [ $? -eq 0 ]; then
        echo "✓ Successfully created splits"
    else
        echo "✗ Failed to create splits"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All k-fold splits created successfully!"
echo "=========================================="
echo ""
echo "Fold structure:"
echo "  Each dataset has ${K} folds with:"
echo "  - fold_0_train.jsonl, fold_0_val.jsonl"
echo "  - fold_1_train.jsonl, fold_1_val.jsonl"
echo "  - ..."
echo "  - fold_$((K-1))_train.jsonl, fold_$((K-1))_val.jsonl"
