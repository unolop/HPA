#!/bin/bash
# Compare multiple ablations side-by-side

set -e

echo "========================================"
echo "Ablation Comparison"
echo "========================================"
echo ""

# Find all completed ablation checkpoints
CHECKPOINTS_DIR="outputs/checkpoints"

if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "❌ No checkpoints found in $CHECKPOINTS_DIR"
    exit 1
fi

echo "Completed ablations:"
echo ""
printf "%-15s %-50s %-15s\n" "Ablation" "Path" "Status"
echo "-------------------------------------------------------------------------------------"

for dir in "$CHECKPOINTS_DIR"/*/; do
    ablation=$(basename "$dir")

    if [ -f "$dir/final_model/config.json" ]; then
        status="✅ Complete"
    elif [ -f "$dir/ablation_config.json" ]; then
        status="🔄 Training"
    else
        status="❌ Incomplete"
    fi

    printf "%-15s %-50s %-15s\n" "$ablation" "$dir" "$status"
done

echo ""
echo "Evaluation results:"
echo ""

RESULTS_DIR="outputs/results"

if [ -d "$RESULTS_DIR" ]; then
    printf "%-15s %-15s %-15s %-15s\n" "Ablation" "Accuracy" "F1" "Calibration ECE"
    echo "--------------------------------------------------------------------"

    for result_dir in "$RESULTS_DIR"/eval_*/; do
        if [ -f "$result_dir/metrics.json" ]; then
            ablation=$(basename "$result_dir" | sed 's/eval_//')

            # Extract metrics using jq if available
            if command -v jq &> /dev/null; then
                accuracy=$(jq -r '.accuracy // "N/A"' "$result_dir/metrics.json")
                f1=$(jq -r '.f1 // "N/A"' "$result_dir/metrics.json")
                ece=$(jq -r '.calibration.ece // "N/A"' "$result_dir/metrics.json")

                printf "%-15s %-15s %-15s %-15s\n" "$ablation" "$accuracy" "$f1" "$ece"
            else
                echo "$ablation  (install jq to see metrics)"
            fi
        fi
    done
fi

echo ""
echo "To evaluate a specific ablation:"
echo "  bash experiments/evaluate_model.sh A4"
