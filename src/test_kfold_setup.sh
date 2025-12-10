#!/bin/bash
# Quick test script to verify k-fold training setup

echo "=========================================="
echo "Testing K-Fold Training Setup"
echo "=========================================="

# Check if k-fold splits exist
KFOLD_DIR="/home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt"

if [ ! -d "$KFOLD_DIR" ]; then
    echo "❌ K-fold directory not found: $KFOLD_DIR"
    echo "Run ./generate_all_kfold_splits.sh first"
    exit 1
fi

# Check for fold files
if [ ! -f "$KFOLD_DIR/fold_0_train.jsonl" ]; then
    echo "❌ Fold files not found in $KFOLD_DIR"
    echo "Run ./generate_all_kfold_splits.sh first"
    exit 1
fi

echo "✅ K-fold splits found"

# Check if train_human_alignment.py exists
if [ ! -f "train_human_alignment.py" ]; then
    echo "❌ train_human_alignment.py not found in current directory"
    exit 1
fi

echo "✅ train_human_alignment.py found"

# Check if train_kfold.py exists
if [ ! -f "train_kfold.py" ]; then
    echo "❌ train_kfold.py not found"
    exit 1
fi

echo "✅ train_kfold.py found"

# Display k-fold info
echo ""
echo "K-fold directory contents:"
ls -lh "$KFOLD_DIR"/fold_*.jsonl | head -5

echo ""
echo "=========================================="
echo "Setup verified! Ready to run k-fold training."
echo "=========================================="
echo ""
echo "To run a single experiment:"
echo "  ./run_kfold_training.sh \\"
echo "    $KFOLD_DIR \\"
echo "    Qwen/Qwen3-VL-8B-Instruct \\"
echo "    ./output/test_kfold \\"
echo "    test_experiment \\"
echo "    0"
echo ""
echo "To run all experiments:"
echo "  ./run_all_kfold_experiments.sh 0"
echo ""
