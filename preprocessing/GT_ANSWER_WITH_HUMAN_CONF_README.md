# Ground Truth Answers with Human Confidence Distributions

Create training data that combines **ground truth answers** with **human confidence distributions** from blind annotations.

## Overview

### Problem
You have two types of training data:
1. **Blind instruction data** - Has human confidence distributions from blind annotations
2. **GT data** - Has ground truth answers but uniform confidence (1.0)

### Solution
Create a **hybrid training set** that combines the best of both:
- **Assistant answers**: Ground truth (correct answers)
- **Labels**: Human confidence distributions (from blind annotations)
- **Images**: Real images (for VQA, not blank)

### Why This Matters
This allows human alignment training (JS divergence) to:
- ✅ Train the model to generate **correct** answers (GT)
- ✅ Match **human confidence distributions** (uncertainty/ambiguity)
- ✅ Learn when humans are uncertain vs confident
- ✅ Better model calibration and alignment

## Quick Start

### 1. Generate GT Answer Versions

```bash
cd preprocessing
chmod +x generate_all_gt_answer_versions.sh
./generate_all_gt_answer_versions.sh
```

This creates files like:
- `train_agg_15_blind_inst.jsonl` → `train_agg_15_blind_inst_gt_answer.jsonl`
- `train_agg_10_blind_inst.jsonl` → `train_agg_10_blind_inst_gt_answer.jsonl`

### 2. Create K-Fold Splits with GT Answers

```bash
cd ../src

# Example: Create k-fold splits from GT answer version
python create_kfold_splits.py \
    --input_path /home/user/HPA/data/training/s1_text/train_agg_15_blind_inst_gt_answer.jsonl \
    --output_dir /home/user/HPA/data/training/s1_text/kfold_15_blind_inst_gt_answer \
    --k 5 \
    --seed 42
```

### 3. Train with GT Answers + Human Confidences

```bash
# K-fold training
./run_kfold_training.sh \
    /home/user/HPA/data/training/s1_text/kfold_15_blind_inst_gt_answer \
    "Qwen/Qwen3-VL-8B-Instruct" \
    ./output/kfold/JS_qwen_15_blind_inst_gt \
    js_qwen_15_blind_inst_gt \
    0

# Or single split training
./train_single_split_js.sh 15_blind_inst_gt "Qwen/Qwen3-VL-8B-Instruct" 0
```

## Files

### Scripts

1. **`create_gt_answer_from_blind.py`** - Main conversion script
   ```bash
   python create_gt_answer_from_blind.py \
       --input_path train_agg_15_blind_inst.jsonl \
       --output_path train_agg_15_blind_inst_gt_answer.jsonl \
       --answer_type text  # or "choice" for multiple choice
   ```

2. **`generate_all_gt_answer_versions.sh`** - Process all datasets at once

3. **`GT_ANSWER_WITH_HUMAN_CONF_README.md`** - This guide

## Data Format

### Input: Blind Instruction Data
```json
{
    "qid": "664",
    "images": ["/home/work/yuna/HPA/data/blank_224.png"],
    "conversations": [
        {
            "role": "user",
            "content": "<image>Question: Where is the bag?\nAnswer:"
        },
        {
            "role": "assistant",
            "content": "A"  // From blind annotations (may be wrong)
        }
    ],
    "labels": {
        "confidences": [0.46, 0.43, 0.11],  // Human confidence distribution
        "answers": ["A", "B", "C"]
    }
}
```

### Output: GT Answer with Human Confidence
```json
{
    "qid": "664",
    "images": ["/path/to/real/image.jpg"],  // Real image (for VQA)
    "conversations": [
        {
            "role": "user",
            "content": "<image>Question: Where is the bag?\nAnswer:"
        },
        {
            "role": "assistant",
            "content": "B"  // Ground truth answer (correct)
        }
    ],
    "labels": {
        "confidences": [0.46, 0.43, 0.11],  // Same human confidence distribution!
        "answers": ["A", "B", "C"]
    }
}
```

### Key Changes
- ✅ **Assistant answer**: Changed from blind answer ("A") to GT answer ("B")
- ✅ **Images**: Changed from blank to real image (for VQA)
- ✅ **Labels**: **Kept the same** - human confidence distribution preserved!

## Answer Types

### Text (VQA Open-Ended)
```bash
--answer_type text
```
- Gets GT answers from VQADataset
- Uses `multiple_choice_answer` field
- Replaces blank images with real images
- For questions like "What color is the sky?" → "blue"

### Multiple Choice
```bash
--answer_type choice
```
- Gets GT answer from most confident label (first answer in sorted list)
- For questions like "Where is the bag? A: hand B: shoulder..." → "B"
- Keeps blank images (for blind condition)

## Training Comparison

### Option 1: Blind Answers + Human Confidence
```bash
# Original blind instruction data
input: train_agg_15_blind_inst.jsonl

# Model learns:
- Generate blind answers (may be wrong)
- Match human confidence distributions
```
**Result**: Model generates plausible but potentially incorrect answers

### Option 2: GT Answers + Uniform Confidence
```bash
# Original GT data
input: train_agg_vqa_gt.jsonl

# Model learns:
- Generate correct answers
- Always be 100% confident
```
**Result**: Model generates correct answers but overconfident

### Option 3: GT Answers + Human Confidence ⭐
```bash
# New GT answer with human confidence
input: train_agg_15_blind_inst_gt_answer.jsonl

# Model learns:
- Generate correct answers (from GT)
- Match human confidence distributions (from blind annotations)
```
**Result**: Model generates correct answers AND has appropriate uncertainty!

## Use Cases

### Research Question: Does Human Confidence Help?
Compare training with:
1. GT answers + uniform confidence (baseline)
2. GT answers + human confidence (proposed)

**Hypothesis**: Human confidence distributions improve model calibration

### Training Pipeline
```bash
# 1. Generate GT answer versions
cd preprocessing
./generate_all_gt_answer_versions.sh

# 2. Create k-fold splits
cd ../src
for dataset in train_agg_*_gt_answer.jsonl; do
    python create_kfold_splits.py \
        --input_path "/home/user/HPA/data/training/s1_text/$dataset" \
        --output_dir "/home/user/HPA/data/training/s1_text/kfold_${dataset%.jsonl}" \
        --k 5
done

# 3. Train with k-fold CV
./run_all_kfold_experiments.sh 0

# 4. Compare with original (blind answer) training
# Results: GT answer version should have higher accuracy but similar confidence distribution
```

## Expected Results

### Metrics to Track

1. **Accuracy** (should improve)
   - GT answer version: Higher accuracy than blind answer version
   - GT answer version: Similar to pure GT training

2. **JS Divergence** (should stay similar)
   - GT answer version: Similar JS divergence to blind answer version
   - Both match human confidence distributions

3. **Model Calibration** (should improve)
   - GT answer version: Better calibrated than pure GT training
   - Confidence matches correctness

### Example
```
Training Data              | Accuracy | JS Divergence | Calibration
---------------------------|----------|---------------|-------------
Blind answers + Human conf | 75%      | 0.12          | Good
GT answers + Uniform conf  | 90%      | 0.35          | Poor (overconfident)
GT answers + Human conf    | 90%      | 0.13          | Excellent ⭐
```

## Advanced: Custom Processing

### Process Single File
```bash
python create_gt_answer_from_blind.py \
    --input_path /path/to/blind_inst.jsonl \
    --output_path /path/to/blind_inst_gt_answer.jsonl \
    --answer_type text
```

### Process with Custom VQA Data
Edit `create_gt_answer_from_blind.py` to point to different VQA annotations:
```python
def load_vqa_annotations(prompt: str = ""):
    vqa_dataset = VQADataset(
        question_path="/path/to/custom/questions.json",
        annotations_path="/path/to/custom/annotations.json",
        prompt=prompt
    )
    return vqa_dataset.get_by_qid()
```

## Troubleshooting

### Issue: "No GT annotation found for qid=XXX"
**Cause**: QID from blind data doesn't match VQA dataset

**Solutions**:
1. Check if QID format matches (int vs string)
2. Verify VQA annotations contain this QID
3. For choice questions, uses first label answer as fallback

### Issue: Different number of samples in output
**Cause**: Some samples skipped due to missing GT

**Expected**: Some QIDs may not have GT annotations
**Check**: Look for "Skipped (no GT)" in summary

### Issue: Blank images in VQA output
**Cause**: GT image path not found

**Solution**:
1. Check VQA dataset image paths
2. Verify image directory exists
3. Check `dataset/vqav2.py` image_dir_path

## Integration with Existing Workflows

### K-Fold Cross-Validation
```bash
# 1. Generate GT versions
cd preprocessing
./generate_all_gt_answer_versions.sh

# 2. Generate k-fold splits with GT versions
cd ../src
# Update generate_all_kfold_splits.sh to include GT versions:
DATASETS=(
    "/home/user/HPA/data/training/s1_text/train_agg_15_blind_inst_gt_answer.jsonl"
    # ... other datasets
)

# 3. Run k-fold training
./run_all_kfold_experiments.sh 0
```

### Single Split Training
```bash
# 1. Create single splits from GT version k-folds
# First create k-fold splits of GT version
python create_kfold_splits.py \
    --input_path train_agg_15_blind_inst_gt_answer.jsonl \
    --output_dir kfold_15_blind_inst_gt_answer \
    --k 5

# 2. Create single split
python create_single_split_from_kfold.py \
    --kfold_dir kfold_15_blind_inst_gt_answer \
    --val_fold_idx 0 \
    --output_train_path single_15_blind_inst_gt_answer/train.jsonl \
    --output_val_path single_15_blind_inst_gt_answer/val.jsonl

# 3. Train
./train_single_split_js.sh 15_blind_inst_gt_answer "Qwen/Qwen3-VL-8B-Instruct" 0
```

## Summary

**What This Does:**
- ✅ Creates training data with GT answers AND human confidence distributions
- ✅ Enables human alignment training on correct answers
- ✅ Improves model accuracy while maintaining calibration
- ✅ Uses real images for VQA (not blank)

**When to Use:**
- Research on human-AI alignment
- Improving model calibration
- Training on correct answers with uncertainty information
- Comparing different confidence distributions

**Key Insight:**
You can have the best of both worlds - **correct answers** (from GT) with **realistic confidence** (from human annotations)!
