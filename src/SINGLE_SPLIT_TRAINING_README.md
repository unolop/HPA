# Single Train/Val Split Training from K-Fold Data

Training with a single train/validation split by reusing existing k-fold splits.

## Overview

Instead of performing full k-fold cross-validation (training k models), this approach:
1. **Uses existing k-fold splits** (no new data splitting needed)
2. **Combines k-1 folds** as training data
3. **Uses 1 fold** as validation data
4. **Trains a single model** per dataset

**Benefits:**
- ✅ Faster than k-fold CV (train once instead of k times)
- ✅ Reuses existing k-fold splits (no wasted data)
- ✅ Still has proper train/val separation
- ✅ Good for initial experiments and hyperparameter tuning

## Quick Start

### 1. Generate Single Splits from K-Fold Data

```bash
cd src
chmod +x generate_single_splits_from_kfold.sh
./generate_single_splits_from_kfold.sh
```

This creates `train.jsonl` and `val.jsonl` for each dataset by:
- **Validation**: fold_0 data (1/5 of data)
- **Training**: fold_1 + fold_2 + fold_3 + fold_4 combined (4/5 of data)

### 2. Train Single Model (JS Divergence)

```bash
chmod +x train_single_split_js.sh
./train_single_split_js.sh vqa_gt "Qwen/Qwen3-VL-8B-Instruct" 0
```

### 3. Train Single Model (Standard SFT)

```bash
chmod +x train_single_split_sft.sh
./train_single_split_sft.sh vqa_gt "Qwen/Qwen3-VL-8B-Instruct" 0
```

### 4. Train All Datasets

```bash
# JS divergence on all datasets
chmod +x train_all_single_split_js.sh
./train_all_single_split_js.sh 0 "Qwen/Qwen3-VL-8B-Instruct"

# Standard SFT on all datasets
chmod +x train_all_single_split_sft.sh
./train_all_single_split_sft.sh 0 "Qwen/Qwen3-VL-8B-Instruct"
```

## Files Created

### Data Splitting

1. **`create_single_split_from_kfold.py`** - Combine k-fold splits
   ```bash
   python create_single_split_from_kfold.py \
       --kfold_dir /path/to/kfold_vqa_gt \
       --val_fold_idx 0 \
       --output_train_path ./single_vqa_gt/train.jsonl \
       --output_val_path ./single_vqa_gt/val.jsonl
   ```

2. **`generate_single_splits_from_kfold.sh`** - Generate splits for all datasets
   - Uses fold_0 as validation
   - Combines other folds as training
   - Creates `single_*/` directories alongside `kfold_*/`

### Training Scripts

3. **`train_single_split_js.sh`** - Train one dataset with JS divergence
   ```bash
   ./train_single_split_js.sh <dataset_name> <model_path> [gpu_id]
   ```

4. **`train_single_split_sft.sh`** - Train one dataset with standard SFT
   ```bash
   ./train_single_split_sft.sh <dataset_name> <model_path> [gpu_id]
   ```

5. **`train_all_single_split_js.sh`** - Train all datasets with JS
   ```bash
   ./train_all_single_split_js.sh [gpu_id] [model_path]
   ```

6. **`train_all_single_split_sft.sh`** - Train all datasets with SFT
   ```bash
   ./train_all_single_split_sft.sh [gpu_id] [model_path]
   ```

## Dataset Names

Available datasets (use with training scripts):
- `vqa_gt` - VQA ground truth data
- `10_blind_inst` - 10 annotator blind instruction
- `15_blind_inst` - 15 annotator blind instruction (text)
- `15_blind_inst_choice` - 15 annotator blind instruction (multiple choice)

## Directory Structure

### Input (K-Fold Data)
```
data/training/
├── s1_text/
│   ├── kfold_vqa_gt/
│   │   ├── fold_0_train.jsonl
│   │   ├── fold_1_train.jsonl
│   │   ├── ...
│   │   └── fold_4_train.jsonl
│   ├── kfold_10_blind_inst/
│   └── kfold_15_blind_inst/
└── s1_choice/
    └── kfold_15_blind_inst/
```

### Output (Single Splits)
```
data/training/
├── s1_text/
│   ├── single_vqa_gt/
│   │   ├── train.jsonl  (fold_1 + fold_2 + fold_3 + fold_4)
│   │   └── val.jsonl    (fold_0)
│   ├── single_10_blind_inst/
│   └── single_15_blind_inst/
└── s1_choice/
    └── single_15_blind_inst/
```

### Training Output
```
output/single_split/
├── JS_Qwen3-VL-8B-Instruct_vqa_gt/
│   ├── checkpoint-xxx/
│   ├── trainer_state.json
│   └── ...
├── SFT_Qwen3-VL-8B-Instruct_vqa_gt/
├── JS_Qwen3-VL-8B-Instruct_10_blind_inst/
└── ...
```

## Training Configuration

### JS Divergence (Human Alignment)
```bash
--mode JS
--lambda_dist 1.0
--lambda_l2 0.1
--use_l2_penalty
--lora_rank 8
--lora_alpha 16
--num_epochs 10
--learning_rate 2e-5
```

### Standard SFT
```bash
--lora_rank 32
--lora_alpha 64
--num_epochs 10
--learning_rate 2e-5
```

## Comparison: Single Split vs K-Fold CV

| Aspect | Single Split | K-Fold CV |
|--------|--------------|-----------|
| **Training Time** | 1× (one model) | k× (k models) |
| **Models Trained** | 1 | k (typically 5) |
| **Data Usage** | 80% train, 20% val | 100% (all data used) |
| **Robustness** | Single estimate | Averaged across k folds |
| **Variance Estimate** | No | Yes (std across folds) |
| **Use Case** | Fast experimentation | Final evaluation, paper results |
| **Output** | One checkpoint | k checkpoints + aggregated stats |

## When to Use Each Approach

### Use Single Split When:
- ✅ Initial experimentation and debugging
- ✅ Hyperparameter tuning
- ✅ Quick model comparisons
- ✅ Limited compute budget
- ✅ Fast iteration needed

### Use K-Fold CV When:
- ✅ Final model evaluation
- ✅ Publishing results (more rigorous)
- ✅ Quantifying model variance
- ✅ Small dataset (maximize data usage)
- ✅ Model selection with confidence intervals

## Example Workflows

### Experiment: Compare JS vs SFT on Single Dataset

```bash
cd src

# 1. Generate splits (if not done)
./generate_single_splits_from_kfold.sh

# 2. Train JS model
./train_single_split_js.sh vqa_gt "Qwen/Qwen3-VL-8B-Instruct" 0

# 3. Train SFT model
./train_single_split_sft.sh vqa_gt "Qwen/Qwen3-VL-8B-Instruct" 0

# 4. Compare results
ls ./output/single_split/JS_*/trainer_state.json
ls ./output/single_split/SFT_*/trainer_state.json
```

### Experiment: Train All Datasets Quickly

```bash
# Train all with JS divergence
./train_all_single_split_js.sh 0 "Qwen/Qwen3-VL-8B-Instruct"

# Or train all with SFT
./train_all_single_split_sft.sh 0 "Qwen/Qwen3-VL-8B-Instruct"
```

### Experiment: After Single Split, Run Full K-Fold

```bash
# 1. Quick experiment with single split
./train_single_split_js.sh vqa_gt "Qwen/Qwen3-VL-8B-Instruct" 0

# 2. If results look good, run full k-fold CV
./run_kfold_training.sh \
    /home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt \
    "Qwen/Qwen3-VL-8B-Instruct" \
    ./output/kfold/JS_qwen_vqa_gt \
    js_qwen_vqa_gt \
    0

# 3. Analyze k-fold results
python analyze_kfold_results.py \
    --kfold_output_dir ./output/kfold/JS_qwen_vqa_gt
```

## Customization

### Change Validation Fold

By default, fold_0 is used for validation. To use a different fold:

Edit `generate_single_splits_from_kfold.sh`:
```bash
VAL_FOLD=2  # Use fold_2 as validation
```

Then regenerate:
```bash
./generate_single_splits_from_kfold.sh
```

### Train Specific Model

```bash
# InternVL
./train_single_split_js.sh vqa_gt "OpenGVLab/InternVL3_5-8B" 0

# Llava
./train_single_split_sft.sh vqa_gt "llava-hf/llava-v1.6-mistral-7b-hf" 1
```

### Custom Hyperparameters

Edit the training scripts directly or copy and modify:

```bash
cp train_single_split_js.sh train_single_split_js_custom.sh
# Edit hyperparameters in the copy
./train_single_split_js_custom.sh vqa_gt "Qwen/Qwen3-VL-8B-Instruct" 0
```

## Troubleshooting

### Issue: "Training data not found"

**Cause**: Single splits haven't been generated yet

**Solution**:
```bash
./generate_single_splits_from_kfold.sh
```

### Issue: K-fold data doesn't exist

**Cause**: K-fold splits haven't been created

**Solution**:
```bash
./generate_all_kfold_splits.sh
```

### Issue: OOM errors

**Solutions**:
1. Reduce batch size (already at 1)
2. Increase gradient accumulation:
   ```bash
   --gradient_accumulation_steps 16  # Change from 8 to 16
   ```
3. Reduce max_pixels:
   ```bash
   --max_pixels 336  # Change from 448 to 336
   ```

### Issue: Want to change validation fold

Edit `generate_single_splits_from_kfold.sh`:
```bash
VAL_FOLD=1  # Change from 0 to 1
```

Then regenerate splits:
```bash
./generate_single_splits_from_kfold.sh
```

## Advanced: Manual Split Creation

If you want more control:

```bash
python create_single_split_from_kfold.py \
    --kfold_dir /home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt \
    --val_fold_idx 2 \
    --output_train_path ./custom_split/train.jsonl \
    --output_val_path ./custom_split/val.jsonl
```

Then train with custom paths:

```bash
CUDA_VISIBLE_DEVICES=0 python train_human_alignment.py \
    --model_path "Qwen/Qwen3-VL-8B-Instruct" \
    --data_path ./custom_split/train.jsonl \
    --val_data_path ./custom_split/val.jsonl \
    --output_dir ./output/custom \
    --run_name custom_experiment \
    --num_epochs 10 \
    --mode JS \
    --lambda_dist 1.0 \
    --lambda_l2 0.1 \
    --use_l2_penalty \
    --learning_rate 2e-5 \
    --lora_rank 8 \
    --lora_alpha 16
```

## Summary

**Single Split Training provides**:
- ✅ Fast experimentation (1 model vs k models)
- ✅ Reuses existing k-fold splits (no data waste)
- ✅ Proper train/val separation
- ✅ Good for hyperparameter tuning
- ✅ Easy to scale to full k-fold later

**Recommended Workflow**:
1. Start with single split for quick iteration
2. Once hyperparameters are tuned, run full k-fold CV
3. Use k-fold results for final evaluation and papers

## See Also

- **K-Fold CV Training**: `KFOLD_CV_README.md` (JS divergence)
- **K-Fold SFT Training**: `KFOLD_SFT_README.md` (standard SFT)
- **Human Alignment Training**: `train_human_alignment.py`
- **Standard SFT Training**: `train_sft_standard.py`
