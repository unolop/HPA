# K-Fold Cross-Validation Training

Complete workflow for k-fold cross-validation training on VLM models with human alignment loss.

## ⚠️ Important: Fixed Training Script Call

**The k-fold training now correctly calls `train_human_alignment.py`** instead of `train_universal.py`.

Training flow:
```
run_all_kfold_experiments.sh
  → train_kfold.py
    → train_human_alignment.py (for each fold)
```

## Overview

K-fold cross-validation provides robust model evaluation by:
1. Splitting data into k equal parts (folds)
2. Training k models, each using a different fold for validation
3. Aggregating results across all folds

This gives better estimates of model performance and reduces variance from a single train/val split.

## Quick Start

### 1. Generate K-Fold Splits

First, create k-fold splits for your datasets:

```bash
cd src
chmod +x generate_all_kfold_splits.sh
./generate_all_kfold_splits.sh
```

This creates 5-fold splits for all three datasets:
- `s1_text/kfold_vqa_gt/` - VQA ground truth data
- `s1_text/kfold_10_blind_inst/` - 10 annotator blind instruction data
- `s1_choice/kfold_15_blind_inst/` - 15 annotator blind instruction data

Each directory contains:
```
kfold_vqa_gt/
├── fold_0_train.jsonl
├── fold_0_val.jsonl
├── fold_1_train.jsonl
├── fold_1_val.jsonl
├── ...
├── fold_4_train.jsonl
├── fold_4_val.jsonl
└── kfold_metadata.json
```

### 2. Run K-Fold Training

#### Option A: Single Dataset + Model

```bash
chmod +x run_kfold_training.sh
./run_kfold_training.sh \
    /home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt \
    "Qwen/Qwen3-VL-8B-Instruct" \
    ./output/qwen_vqa_gt_kfold \
    qwen_vqa_gt_cv \
    0  # GPU ID
```

#### Option B: All Datasets + All Models

```bash
chmod +x run_all_kfold_experiments.sh
./run_all_kfold_experiments.sh 0  # GPU ID
```

This runs all combinations of:
- **Models**: Qwen, InternVL, Llava
- **Datasets**: vqa_gt, 10_blind_inst, 15_blind_inst
- **Total**: 9 experiments (3 models × 3 datasets)

### 3. Analyze Results

After training completes, analyze each experiment:

```bash
python analyze_kfold_results.py \
    --kfold_output_dir ./output/kfold/qwen_vqa_gt
```

This generates `kfold_analysis.json` with:
- Mean, std, min, max for all metrics across folds
- Individual fold results
- Aggregated statistics

## Files and Scripts

### Data Preparation

1. **`create_kfold_splits.py`** - Create k-fold splits from JSONL
   ```bash
   python create_kfold_splits.py \
       --input_path data.jsonl \
       --output_dir ./kfold_splits \
       --k 5 \
       --seed 42
   ```

2. **`generate_all_kfold_splits.sh`** - Generate splits for all datasets
   - Processes all three training datasets
   - Uses k=5, seed=42
   - Creates organized directory structure

### Training

3. **`train_kfold.py`** - K-fold CV training wrapper
   ```bash
   python train_kfold.py \
       --kfold_dir /path/to/kfold_splits \
       --model_path "OpenGVLab/InternVL3_5-8B" \
       --output_base_dir ./output/internvl_kfold \
       --run_name internvl_cv \
       --num_epochs 10 \
       --mode JS \
       --lambda_dist 1.0 \
       --lambda_l2 0.1 \
       --use_l2_penalty
   ```

   **Key Features**:
   - Automatically loads all folds from kfold_dir
   - Trains each fold sequentially
   - Creates fold_0/, fold_1/, etc. subdirectories
   - Saves training summary JSON

4. **`run_kfold_training.sh`** - Shell wrapper for single experiment
   - Easier command-line interface
   - Fixed hyperparameters
   - Customizable GPU selection

5. **`run_all_kfold_experiments.sh`** - Run all experiments
   - 3 models × 3 datasets = 9 experiments
   - Sequential execution (one GPU)
   - Stops on first failure

### Analysis

6. **`analyze_kfold_results.py`** - Aggregate and analyze results
   ```bash
   python analyze_kfold_results.py \
       --kfold_output_dir ./output/kfold/qwen_vqa_gt
   ```

   **Output**:
   - Console table with mean/std/min/max for all metrics
   - JSON file with detailed statistics
   - Per-fold breakdown

## Output Structure

```
output/kfold/
├── qwen_vqa_gt/
│   ├── fold_0/
│   │   ├── checkpoint-xxx/
│   │   ├── trainer_state.json
│   │   └── ...
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   ├── kfold_training_summary.json
│   └── kfold_analysis.json
├── qwen_10_blind_inst/
├── qwen_15_blind_inst/
├── internvl_vqa_gt/
├── internvl_10_blind_inst/
├── internvl_15_blind_inst/
├── llava_vqa_gt/
├── llava_10_blind_inst/
└── llava_15_blind_inst/
```

## Hyperparameters

Default settings optimized for human alignment training:

### Loss Configuration
- `--mode JS` - Jensen-Shannon divergence
- `--lambda_dist 1.0` - Weight for distribution matching
- `--lambda_l2 0.1` - Weight for L2 penalty
- `--use_l2_penalty` - Enable L2 penalty (flag)
- `--use_sft_loss` - Include SFT loss (default: False)

### Training Control
- `--num_epochs 10` - Number of epochs
- `--max_steps -1` - Use num_epochs instead of max_steps
- `--learning_rate 2e-5` - Learning rate

### LoRA Settings
- `--lora_rank 8` - LoRA rank
- `--lora_alpha 16` - LoRA alpha

### Batch Settings
- `--batch_size 1` - Per-device batch size
- `--gradient_accumulation_steps 8` - Effective batch size = 8

### Logging & Checkpointing
- `--save_steps 40` - Save checkpoint every 40 steps
- `--eval_steps 40` - Evaluate every 40 steps
- `--logging_steps 20` - Log every 20 steps

## Customization

### Change Number of Folds

Edit `generate_all_kfold_splits.sh`:
```bash
K=10  # Change from 5 to 10
```

Then re-generate splits:
```bash
./generate_all_kfold_splits.sh
```

### Train Specific Folds Only

```bash
python train_kfold.py \
    --kfold_dir /path/to/kfold_splits \
    --model_path "model" \
    --output_base_dir ./output \
    --run_name experiment \
    --folds 0 1 2  # Only train folds 0, 1, 2
    # ... other args
```

### Custom Hyperparameters

Edit the shell scripts or call `train_kfold.py` directly with custom values:

```bash
python train_kfold.py \
    --kfold_dir /path/to/kfold_splits \
    --model_path "model" \
    --output_base_dir ./output \
    --run_name experiment \
    --num_epochs 15 \
    --learning_rate 1e-5 \
    --lora_rank 16 \
    --lambda_dist 2.0 \
    # ... other args
```

## Workflow Example

Complete workflow for a new experiment:

```bash
# 1. Generate k-fold splits (do this once)
cd src
./generate_all_kfold_splits.sh

# 2. Train a single model on one dataset (test first)
./run_kfold_training.sh \
    /home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt \
    "Qwen/Qwen3-VL-8B-Instruct" \
    ./output/test_qwen_vqa_gt \
    test_experiment \
    0

# 3. Analyze results
python analyze_kfold_results.py \
    --kfold_output_dir ./output/test_qwen_vqa_gt

# 4. If successful, run all experiments
./run_all_kfold_experiments.sh 0

# 5. Analyze all results
for exp in ./output/kfold/*/; do
    echo "Analyzing $exp"
    python analyze_kfold_results.py --kfold_output_dir "$exp"
done
```

## Interpreting Results

### Key Metrics

From `kfold_analysis.json`:

```json
{
  "aggregated_stats": {
    "eval_loss": {
      "mean": 1.234,
      "std": 0.056,
      "min": 1.180,
      "max": 1.290
    }
  }
}
```

- **mean**: Average performance across folds
- **std**: Variability between folds (lower is better)
- **min/max**: Best and worst fold performance

### What to Look For

1. **Low mean eval_loss** - Good model performance
2. **Low std** - Consistent across folds (robust)
3. **Small range (max - min)** - Stable training

### Comparing Models

```bash
# Compare eval_loss across models for same dataset
grep '"eval_loss"' ./output/kfold/*/kfold_analysis.json
```

## Troubleshooting

### Issue: "No fold directories found"

**Cause**: K-fold splits not generated yet

**Solution**:
```bash
./generate_all_kfold_splits.sh
```

### Issue: Training fails on specific fold

**Cause**: Could be data issue or OOM

**Solution**:
1. Check fold data: `head -5 /path/to/fold_X_train.jsonl`
2. Reduce batch size or increase gradient accumulation
3. Train other folds: `--folds 1 2 3 4` (skip fold 0)

### Issue: "trainer_state.json not found"

**Cause**: Training didn't complete successfully

**Solution**:
1. Check logs in fold directory
2. Re-run training for that fold
3. Verify GPU memory

### Issue: OOM (Out of Memory)

**Solution**: Adjust memory usage in training scripts:
```bash
# Edit run_kfold_training.sh or call train_kfold.py with:
--batch_size 1 \
--gradient_accumulation_steps 16  # Increase from 8
--max_pixels 336  # Decrease from 448
```

## Advanced Usage

### Parallel Training on Multiple GPUs

Train different experiments in parallel:

```bash
# Terminal 1 (GPU 0)
./run_kfold_training.sh /path/to/kfold1 model1 output1 exp1 0 &

# Terminal 2 (GPU 1)
./run_kfold_training.sh /path/to/kfold2 model2 output2 exp2 1 &

# Terminal 3 (GPU 2)
./run_kfold_training.sh /path/to/kfold3 model3 output3 exp3 2 &
```

### Resume from Checkpoint

Modify `train_kfold.py` to add `--resume_from_checkpoint` support, or manually resume:

```bash
python train_universal.py \
    --model_path "model" \
    --data_path /path/to/fold_0_train.jsonl \
    --val_data_path /path/to/fold_0_val.jsonl \
    --output_dir ./output/fold_0 \
    --resume_from_checkpoint ./output/fold_0/checkpoint-xxx \
    # ... other args
```

## References

- **Universal Training**: See `UNIVERSAL_TRAINING_README.md` for model-agnostic training
- **Human Alignment**: See `train_human_alignment.py` for JS divergence loss details
- **Data Preparation**: See data processing scripts in `evaluation/`

## Summary

**K-fold CV provides**:
- ✅ Robust performance estimates
- ✅ Better model selection
- ✅ Variance quantification
- ✅ Efficient data usage (all data used for both training and validation)

**Use k-fold CV when**:
- Dataset is relatively small
- Need robust performance estimates
- Comparing multiple models or hyperparameters
- Publishing results (more rigorous evaluation)
