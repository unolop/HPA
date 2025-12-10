# Universal Multi-Model Training with Human Alignment

## Overview

`train_universal.py` is a simplified training script that works with **QwenVL, InternVL, and Llava** models using Swift's built-in model handling.

## Key Design Principle

**Let Swift auto-determine model-specific settings** instead of hardcoding them.

## What Makes It Universal?

### ❌ Old Approach (train_human_alignment.py)
```python
# Hardcoded target modules - only works for InternVL/Llava style models
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### ✅ New Approach (train_universal.py)
```python
# NO target_modules specified - Swift auto-determines based on model:
# - QwenVL: Uses "all-linear"
# - InternVL: Uses specific attention/MLP layers
# - Llava: Uses Mistral-style layers
```

## Usage

### Quick Test (10 steps per model)
```bash
cd src
./test_universal_training.sh
```

### Full Training
```bash
cd src
./A1_train_js_universal.sh
```

Or run directly:
```bash
python train_universal.py \
    --model_path "OpenGVLab/InternVL3_5-8B" \
    --data_path "data.jsonl" \
    --output_dir "./output/internvl" \
    --run_name "experiment" \
    --num_epochs 10 \
    --mode JS \
    --lambda_dist 1.0 \
    --lambda_l2 0.1 \
    --use_l2_penalty \
    --learning_rate 2e-5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --batch_size 1 \
    --gradient_accumulation_steps 8
```

## Fixed Hyperparameters

The script keeps only essential hyperparameters as arguments:

**Loss Settings (Fixed)**:
- `--mode`: "JS" (Jensen-Shannon) or "CE" (Cross-Entropy)
- `--lambda_dist`: Weight for distribution matching (default: 1.0)
- `--lambda_l2`: Weight for L2 penalty (default: 0.1)
- `--use_l2_penalty`: Enable L2 penalty (flag)
- `--use_sft_loss`: Include SFT loss (default: False, JS loss only)

**Training Control**:
- `--learning_rate`: Learning rate (default: 2e-5)
- `--num_epochs`: Number of epochs (default: 10)
- `--max_steps`: Max steps (-1 to use num_epochs)

**LoRA Settings**:
- `--lora_rank`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha (default: 16)

**Batch Settings**:
- `--batch_size`: Per-device batch size (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 8)

## What Swift Auto-Handles

Swift automatically configures:
- ✅ `target_modules` (model-specific LoRA targets)
- ✅ `max_pixels` (image resolution per model)
- ✅ Vision encoder settings
- ✅ Tokenizer configuration
- ✅ Model architecture differences

## Human Alignment Loss

The custom trainer applies **JS divergence loss** to match human confidence distributions:

```python
# For each question with human annotations
human_dist = build_distribution(answers, confidences)  # Human confidence distribution
model_probs = softmax(model_logits)                     # Model predictions

# JS Divergence
m = 0.5 * (human_dist + model_probs)
js_loss = 0.5 * KL(human_dist || m) + 0.5 * KL(model_probs || m)

# Optional L2 penalty
l2_loss = ||human_dist - model_probs||^2

# Total loss
total = lambda_dist * js_loss + lambda_l2 * l2_loss
```

## Tested Models

- ✅ **Qwen/Qwen3-VL-8B-Instruct**: Custom ViT, Qwen2 base
- ✅ **OpenGVLab/InternVL3_5-8B**: InternViT, InternLM2 base
- ✅ **llava-hf/llava-v1.6-mistral-7b-hf**: CLIP ViT, Mistral base

## Comparison with train_human_alignment.py

| Feature | train_human_alignment.py | train_universal.py |
|---------|--------------------------|-------------------|
| Target modules | Hardcoded InternVL style | Auto-determined by Swift |
| Models supported | InternVL, Llava only | QwenVL, InternVL, Llava |
| Complexity | Model-specific args | Simplified, minimal args |
| Custom loss | ✅ JS divergence | ✅ JS divergence (same) |
| Use case | Single model family | All models |

## Files Created

1. **train_universal.py** - Universal training script
2. **A1_train_js_universal.sh** - Shell wrapper for full training
3. **test_universal_training.sh** - Quick test (10 steps per model)
4. **UNIVERSAL_TRAINING_README.md** - This file

## Obsolete Files

These were created during debugging but are no longer needed:
- ❌ `model_configs.py` - No longer needed (Swift handles it)
- ❌ `train_sft_multimodel.py` - Replaced by train_universal.py
- ❌ `train_multimodel.sh` - Replaced by A1_train_js_universal.sh

## Troubleshooting

If you encounter issues:

1. **Verify data path exists**:
   ```bash
   ls -lh /home/work/yuna/HPA/data/training/s1_choice/train_agg_15_blind_inst.jsonl
   ```

2. **Check GPU availability**:
   ```bash
   nvidia-smi
   ```

3. **Test with minimal steps first**:
   ```bash
   ./test_universal_training.sh
   ```

4. **Check logs for model loading**:
   - Should see "Model loaded successfully"
   - Swift should auto-detect model type
   - No errors about target_modules

## Next Steps

To start training:
```bash
cd src
./test_universal_training.sh  # Quick sanity check
./A1_train_js_universal.sh    # Full training on all models
```
