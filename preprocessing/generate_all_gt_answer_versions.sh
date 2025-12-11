#!/bin/bash
# Generate GT answer versions of all blind instruction training data

echo "=========================================="
echo "Creating GT Answer Versions"
echo "=========================================="

# Text (VQA open-ended) datasets
TEXT_DATASETS=(
    "/home/user/HPA/data/training/s1_text/train_agg_10_blind_inst.jsonl"
    "/home/user/HPA/data/training/s1_text/train_agg_15_blind_inst.jsonl"
    "/home/user/HPA/data/training/s1_text/train_agg_20_blind_inst.jsonl"
    "/home/user/HPA/data/training/s1_text/train_agg_30_blind_inst.jsonl"
)

# Multiple choice datasets
CHOICE_DATASETS=(
    "/home/user/HPA/data/training/s1_choice/train_agg_10_blind_inst.jsonl"
    "/home/user/HPA/data/training/s1_choice/train_agg_14_blind.jsonl"
    "/home/user/HPA/data/training/s1_choice/train_agg_15_blind_inst.jsonl"
    "/home/user/HPA/data/training/s1_choice/train_agg_20_blind_inst.jsonl"
    "/home/user/HPA/data/training/s1_choice/train_agg_30_blind_inst.jsonl"
)

# Process text datasets
echo ""
echo "Processing text (VQA) datasets..."
echo "=========================================="
for input_file in "${TEXT_DATASETS[@]}"; do
    if [ ! -f "$input_file" ]; then
        echo "  ⚠️  File not found: $input_file, skipping"
        continue
    fi

    # Create output filename by inserting "_gt_answer" before .jsonl
    output_file="${input_file%.jsonl}_gt_answer.jsonl"

    echo ""
    echo "Processing: $(basename $input_file)"

    python create_gt_answer_from_blind.py \
        --input_path "$input_file" \
        --output_path "$output_file" \
        --answer_type text

    if [ $? -eq 0 ]; then
        echo "  ✅ Success: $(basename $output_file)"
    else
        echo "  ❌ Failed: $(basename $input_file)"
        exit 1
    fi
done

# Process choice datasets
echo ""
echo "Processing multiple choice datasets..."
echo "=========================================="
for input_file in "${CHOICE_DATASETS[@]}"; do
    if [ ! -f "$input_file" ]; then
        echo "  ⚠️  File not found: $input_file, skipping"
        continue
    fi

    # Create output filename by inserting "_gt_answer" before .jsonl
    output_file="${input_file%.jsonl}_gt_answer.jsonl"

    echo ""
    echo "Processing: $(basename $input_file)"

    python create_gt_answer_from_blind.py \
        --input_path "$input_file" \
        --output_path "$output_file" \
        --answer_type choice

    if [ $? -eq 0 ]; then
        echo "  ✅ Success: $(basename $output_file)"
    else
        echo "  ❌ Failed: $(basename $input_file)"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All GT Answer Versions Created!"
echo "=========================================="
echo ""
echo "Output files have '_gt_answer' suffix:"
echo "  Example: train_agg_15_blind_inst_gt_answer.jsonl"
echo ""
echo "These files have:"
echo "  - GT answers in assistant responses"
echo "  - Human confidence distributions in labels"
echo "  - Real images (for VQA text datasets)"
echo ""
