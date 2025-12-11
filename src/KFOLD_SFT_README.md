# K-Fold Cross-Validation Training (Standard SFT)

K-fold cross-validation training using **standard supervised fine-tuning (SFT)** with cross-entropy loss.

## Overview

This is the baseline SFT training variant of k-fold CV, compared to the JS divergence human alignment training.

**Training Types Available:**
1. **Standard SFT** (this guide) - Uses `train_sft_standard.py` with CE loss
2. **JS Divergence** - Uses `train_human_alignment.py` with custom human alignment loss (see `KFOLD_CV_README.md`)

## Training Flow

```
run_all_kfold_sft_experiments.sh
  → train_kfold_sft.py
    → train_sft_standard.py (for each fold)
```

## Quick Start

### 1. Generate K-Fold Splits (if not done yet)

```bash
cd src
./generate_all_kfold_splits.sh
```

### 2. Run SFT K-Fold Training

#### Option A: Single Dataset + Model

```bash
./run_kfold_sft_training.sh \
    /home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt \
    "Qwen/Qwen3-VL-8B-Instruct" \
    ./output/qwen_vqa_gt_kfold_sft \
    qwen_vqa_gt_sft_cv \
    0  # GPU ID
```

#### Option B: All Datasets + All Models

```bash
./run_all_kfold_sft_experiments.sh 0  # GPU ID
```

### 3. Analyze Results

```bash
python analyze_kfold_results.py \
    --kfold_output_dir ./output/kfold/SFT_Qwen3-VL-8B-Instruct_A1_vqa_gt_K0
```

## Files

### SFT-Specific Scripts

1. **`train_kfold_sft.py`** - K-fold wrapper for SFT training
   - Calls `train_sft_standard.py` for each fold
   - Uses standard CE loss (no custom alignment loss)

2. **`run_kfold_sft_training.sh`** - Shell wrapper for single SFT experiment
   ```bash
   ./run_kfold_sft_training.sh <kfold_dir> <model_path> <output_dir> <run_name> [gpu_id]
   ```

3. **`run_all_kfold_sft_experiments.sh`** - Run all SFT k-fold experiments
   - 1 model × 4 datasets = 4 experiments (by default)
   - Can uncomment additional models in the script

### Shared Scripts

These work for both SFT and JS training:

- **`create_kfold_splits.py`** - Create k-fold splits
- **`generate_all_kfold_splits.sh`** - Generate splits for all datasets
- **`analyze_kfold_results.py`** - Analyze results from any k-fold training

## Hyperparameters

### SFT-Specific Defaults

```bash
--learning_rate 2e-5
--num_epochs 10
--max_steps -1           # Use num_epochs instead
--lora_rank 32           # Higher than JS (32 vs 8)
--lora_alpha 64          # Higher than JS (64 vs 16)
--batch_size 1
--gradient_accumulation_steps 8
--save_steps 40
--eval_steps 40
--logging_steps 20
--max_pixels 448
```

**Key Differences from JS Training:**
- **No custom loss parameters** (mode, lambda_dist, lambda_l2, use_l2_penalty, use_sft_loss)
- **Higher LoRA rank/alpha** (32/64 vs 8/16) - standard SFT typically uses higher capacity
- Uses `target_modules=["all-linear"]` in `train_sft_standard.py`

## Output Structure

```
output/kfold/
├── SFT_Qwen3-VL-8B-Instruct_A1_vqa_gt_K0/
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
├── SFT_Qwen3-VL-8B-Instruct_A2_vqa_10_blind_inst_K1/
├── ...
```

Note the `SFT_` prefix to distinguish from `JS_` (JS divergence) experiments.

## Comparison: SFT vs JS Divergence

| Feature | Standard SFT | JS Divergence |
|---------|--------------|---------------|
| **Loss Function** | Cross-Entropy (CE) | Jensen-Shannon Divergence + L2 |
| **Training Script** | `train_sft_standard.py` | `train_human_alignment.py` |
| **K-fold Wrapper** | `train_kfold_sft.py` | `train_kfold.py` |
| **Shell Scripts** | `run_*_sft_*.sh` | `run_kfold_*.sh` |
| **LoRA Rank** | 32 | 8 |
| **LoRA Alpha** | 64 | 16 |
| **Custom Loss Params** | None | mode, λ_dist, λ_l2 |
| **Use Case** | Baseline, standard fine-tuning | Human alignment, distribution matching |
| **Output Prefix** | `SFT_` | `JS_` |

## When to Use Each

### Use Standard SFT When:
- Establishing baseline performance
- Standard supervised fine-tuning on labeled data
- No human confidence/distribution data available
- Comparing against human alignment methods

### Use JS Divergence When:
- Training with human confidence distributions
- Matching model output to human uncertainty
- Research on human-model alignment
- Have multiple annotator responses per question

## Example Workflows

### Baseline SFT Training

```bash
cd src

# 1. Generate splits (if not done)
./generate_all_kfold_splits.sh

# 2. Run SFT k-fold on one dataset (test)
./run_kfold_sft_training.sh \
    /home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt \
    "Qwen/Qwen3-VL-8B-Instruct" \
    ./output/test_sft_kfold \
    test_sft \
    0

# 3. Check results
python analyze_kfold_results.py \
    --kfold_output_dir ./output/test_sft_kfold

# 4. If successful, run all experiments
./run_all_kfold_sft_experiments.sh 0
```

### Comparing SFT vs JS Divergence

```bash
# Train both methods on same dataset
# 1. SFT baseline
./run_kfold_sft_training.sh \
    /home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt \
    "Qwen/Qwen3-VL-8B-Instruct" \
    ./output/kfold/SFT_qwen_vqa_gt \
    sft_qwen_vqa_gt \
    0

# 2. JS divergence
./run_kfold_training.sh \
    /home/work/yuna/HPA/data/training/s1_text/kfold_vqa_gt \
    "Qwen/Qwen3-VL-8B-Instruct" \
    ./output/kfold/JS_qwen_vqa_gt \
    js_qwen_vqa_gt \
    0

# 3. Compare results
python analyze_kfold_results.py --kfold_output_dir ./output/kfold/SFT_qwen_vqa_gt
python analyze_kfold_results.py --kfold_output_dir ./output/kfold/JS_qwen_vqa_gt

# Compare eval_loss between methods
```

## Customization

### Change LoRA Settings

Edit the shell scripts or call `train_kfold_sft.py` directly:

```bash
python train_kfold_sft.py \
    --kfold_dir /path/to/kfold_splits \
    --model_path "Qwen/Qwen3-VL-8B-Instruct" \
    --output_base_dir ./output/custom_sft \
    --run_name custom_experiment \
    --lora_rank 64 \
    --lora_alpha 128 \
    --learning_rate 1e-5 \
    --num_epochs 15
```

### Train Specific Folds

```bash
python train_kfold_sft.py \
    --kfold_dir /path/to/kfold_splits \
    --model_path "model" \
    --output_base_dir ./output \
    --run_name experiment \
    --folds 0 1 2  # Only train folds 0, 1, 2
```

### Add More Models

Edit `run_all_kfold_sft_experiments.sh`:

```bash
MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-4B-Instruct"      # Uncomment
    "OpenGVLab/InternVL3_5-8B"       # Uncomment
    "llava-hf/llava-v1.6-mistral-7b-hf"  # Uncomment
)
```

## Troubleshooting

### Issue: Different results from JS training

**Expected** - SFT and JS use different loss functions and LoRA settings.

**To compare fairly:**
1. Use same LoRA rank/alpha in both
2. For JS: set `--use_sft_loss` to include CE loss
3. Compare on held-out test set

### Issue: OOM errors

**Solutions:**
1. Reduce batch size: `--batch_size 1`
2. Increase gradient accumulation: `--gradient_accumulation_steps 16`
3. Reduce max_pixels: `--max_pixels 336`
4. Reduce LoRA rank: `--lora_rank 16`

### Issue: Training too slow

**Solutions:**
1. Reduce num_epochs: `--num_epochs 5`
2. Use max_steps instead: `--max_steps 1000 --num_epochs -1`
3. Increase batch size if memory allows
4. Train fewer folds: `--folds 0 1 2` (3 folds instead of 5)

## Summary

**Key Takeaways:**
- ✅ Standard SFT k-fold training for baseline comparisons
- ✅ Uses `train_sft_standard.py` with CE loss
- ✅ Higher LoRA capacity (32/64) than JS training (8/16)
- ✅ Output prefixed with `SFT_` for easy identification
- ✅ Compatible with same k-fold splits and analysis tools
- ✅ Parallel structure to JS divergence training

## See Also

- **JS Divergence Training**: `KFOLD_CV_README.md`
- **Data Splitting**: `generate_all_kfold_splits.sh`
- **Results Analysis**: `analyze_kfold_results.py`
- **Standard SFT Training**: `train_sft_standard.py`
- **Human Alignment Training**: `train_human_alignment.py`
