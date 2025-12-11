#!/bin/bash
# Generate single train/val splits from existing k-fold data
# Uses fold_0 as validation, other folds combined as training

VAL_FOLD=0  # Which fold to use as validation

# Define k-fold directories
KFOLD_DIRS=(
    "/home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt"
    "/home/work/yuna/HPA/data/training/s1_text/kfold_10_blind_inst"
    "/home/work/yuna/HPA/data/training/s1_text/kfold_15_blind_inst"
    "/home/work/yuna/HPA/data/training/s1_choice/kfold_15_blind_inst"
)

# Output directories
OUTPUT_DIRS=(
    "/home/work/yuna/HPA/data/training/s1_text/single_vqa_gt"
    "/home/work/yuna/HPA/data/training/s1_text/single_10_blind_inst"
    "/home/work/yuna/HPA/data/training/s1_text/single_15_blind_inst"
    "/home/work/yuna/HPA/data/training/s1_choice/single_15_blind_inst"
)

echo "=========================================="
echo "Creating Single Train/Val Splits from K-Fold Data"
echo "=========================================="
echo "Validation fold: ${VAL_FOLD}"
echo "=========================================="

for i in "${!KFOLD_DIRS[@]}"; do
    kfold_dir="${KFOLD_DIRS[$i]}"
    output_dir="${OUTPUT_DIRS[$i]}"

    echo ""
    echo "Processing: $(basename $(dirname $kfold_dir))/$(basename $kfold_dir)"
    echo "Output: ${output_dir}"

    python create_single_split_from_kfold.py \
        --kfold_dir "${kfold_dir}" \
        --val_fold_idx ${VAL_FOLD} \
        --output_train_path "${output_dir}/train.jsonl" \
        --output_val_path "${output_dir}/val.jsonl"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully created single split"
    else
        echo "✗ Failed to create single split"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All single splits created successfully!"
echo "=========================================="
echo ""
echo "Single split structure:"
echo "  Each dataset has:"
echo "  - train.jsonl (k-1 folds combined)"
echo "  - val.jsonl (1 fold)"
echo ""
