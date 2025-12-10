# Multi-Model Training Debug Guide

> **⚠️ NOTE**: This debug guide describes the old model-specific approach. For the simplified universal training approach, see **[UNIVERSAL_TRAINING_README.md](UNIVERSAL_TRAINING_README.md)** and use `train_universal.py` instead.

## Common Issues with InternVL and Llava (vs QwenVL)

### Problem Summary
QwenVL training works fine, but InternVL and Llava models fail with the same code.

## Key Differences Between Models

| Aspect | QwenVL | InternVL | Llava |
|--------|--------|----------|-------|
| **Vision Encoder** | Custom ViT | InternViT | CLIP ViT-L |
| **LLM Base** | Qwen2 | InternLM2 | Mistral/Vicuna |
| **Image Resolution** | 448x448 | 448-672 | 336-672 |
| **Projector** | MLP | MLP | MLP (mm_projector) |
| **Default LR** | 2e-5 | 1e-5 | 2e-5 |
| **LoRA Targets** | all-linear | Attention + MLP only | Attention + MLP only |

## Common Errors and Fixes

### Error 1: "`target_modules` not found in model"

**Symptom**:
```
ValueError: Target modules ['all-linear'] not found in model
```

**Cause**: InternVL and Llava have different layer names than QwenVL.

**Fix**: Use model-specific target modules:

```python
# For InternVL and Llava
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",     # MLP
]

# For QwenVL
target_modules = ["all-linear"]  # Works because Swift handles it
```

**Solution in new script**: `model_configs.py` automatically selects correct targets.

---

### Error 2: "Out of Memory (OOM)"

**Symptom**:
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**Cause**: InternVL and Llava have larger vision encoders.

**Fix Options**:

1. **Reduce max_pixels**:
   ```bash
   --max_pixels 336  # Instead of 448
   ```

2. **Increase gradient accumulation**:
   ```bash
   --gradient_accumulation_steps 16  # Instead of 8
   ```

3. **Enable gradient checkpointing** (already default in new script):
   ```python
   gradient_checkpointing=True
   ```

4. **Freeze more components**:
   ```python
   freeze_vit=True         # Always
   freeze_aligner=True     # Projector
   freeze_llm=False        # Only train LLM
   ```

---

### Error 3: "Learning rate too high - loss not decreasing"

**Symptom**:
- Loss starts high and doesn't decrease
- Or loss becomes NaN

**Cause**: InternVL needs lower learning rate due to architecture sensitivity.

**Fix**:
```bash
# For InternVL
--learning_rate 1e-5  # Instead of 2e-5

# For Llava (can use higher)
--learning_rate 2e-5  # Same as QwenVL
```

**Solution in new script**: Automatically uses `1e-5` for InternVL.

---

### Error 4: "Image preprocessing error"

**Symptom**:
```
ValueError: Image size mismatch
AssertionError: Expected image shape ...
```

**Cause**: Different models expect different image preprocessing.

**Fix**: Ensure data format matches model expectations:

```python
# InternVL expects:
max_pixels = 448 * 448  # or 672 * 672

# Llava expects:
max_pixels = 336 * 336  # or 672 * 672
image_aspect_ratio = "pad"  # or "square"
```

---

### Error 5: "Flash Attention not available"

**Symptom**:
```
ImportError: flash_attn is not installed
```

**Cause**: InternVL can use Flash Attention but it's optional.

**Fix Option 1** - Install Flash Attention:
```bash
pip install flash-attn --no-build-isolation
```

**Fix Option 2** - Disable it:
```bash
python train_sft_multimodel.py ... # Don't use --use_flash_attn flag
```

**Solution in new script**: Only enables for InternVL if flag is set.

---

### Error 6: "Tokenizer mismatch"

**Symptom**:
```
ValueError: Token ID out of bounds
KeyError: special token not found
```

**Cause**: Different tokenizer formats.

**Fix**: Ensure data uses correct format:

```json
{
  "conversations": [
    {"role": "user", "content": "<image>Question text"},
    {"role": "assistant", "content": "Answer"}
  ]
}
```

All three models use `<image>` token, but check spacing and format.

---

## Debugging Workflow

### Step 1: Check Model Configuration

```bash
# Check what config will be used
python model_configs.py --model_path OpenGVLab/InternVL3_5-8B
```

This shows:
- Vision encoder type
- Recommended learning rate
- LoRA targets
- Max pixels
- Freezing strategy

### Step 2: Test with Minimal Config

```bash
# Start with conservative settings
python train_sft_multimodel.py \
    --model_path OpenGVLab/InternVL3_5-8B \
    --data_path data.jsonl \
    --output_dir output/test \
    --run_name test \
    --num_epochs 1 \
    --max_steps 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 16
```

If this works, gradually adjust parameters.

### Step 3: Monitor Memory Usage

```bash
# Run with memory monitoring
watch -n 1 nvidia-smi

# Look for:
# - GPU memory usage (should stay under 80%)
# - Memory allocation errors
```

### Step 4: Check Logs

Common warning signs:
```
# Bad:
Loss: nan
Loss: 10.5, 10.8, 11.2  # Increasing

# Good:
Loss: 2.5, 2.3, 2.1, 1.9  # Decreasing
```

## Model-Specific Recommendations

### InternVL Training

```bash
python train_sft_multimodel.py \
    --model_path OpenGVLab/InternVL3_5-8B \
    --data_path data.jsonl \
    --output_dir output/internvl \
    --run_name experiment \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    # Script automatically uses:
    # --learning_rate 1e-5
    # --lora_rank 16
    # --lora_alpha 32
    # --max_pixels 200704  # 448*448
```

**Key points**:
- Lower LR than QwenVL
- Higher LoRA rank (more capacity)
- Can use higher resolution if GPU allows

### Llava Training

```bash
python train_sft_multimodel.py \
    --model_path llava-hf/llava-v1.6-mistral-7b-hf \
    --data_path data.jsonl \
    --output_dir output/llava \
    --run_name experiment \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    # Script automatically uses:
    # --learning_rate 2e-5
    # --lora_rank 8
    # --lora_alpha 16
    # --max_pixels 112896  # 336*336
```

**Key points**:
- Same LR as QwenVL
- Smaller default image size
- Mistral-based variants are more stable

## Quick Comparison Test

Test all three models with identical data:

```bash
# Create test script
cat > test_all_models.sh << 'EOF'
#!/bin/bash

MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "OpenGVLab/InternVL3_5-8B"
    "llava-hf/llava-v1.6-mistral-7b-hf"
)

for MODEL in "${MODELS[@]}"; do
    echo "Testing: $MODEL"
    python train_sft_multimodel.py \
        --model_path "$MODEL" \
        --data_path data.jsonl \
        --output_dir "output/test_$(basename $MODEL)" \
        --run_name test \
        --max_steps 10 \
        --batch_size 1

    echo "---"
done
EOF

chmod +x test_all_models.sh
./test_all_models.sh
```

## Troubleshooting Checklist

- [ ] Model is correctly specified (check HuggingFace path)
- [ ] Swift framework is installed and up to date
- [ ] CUDA is available (`torch.cuda.is_available()`)
- [ ] Enough GPU memory (check `nvidia-smi`)
- [ ] Data format is correct (check with `head -1 data.jsonl`)
- [ ] Image files exist (if using file paths)
- [ ] Using model-specific configuration (run `model_configs.py`)
- [ ] Learning rate is appropriate for model
- [ ] LoRA targets match model architecture
- [ ] Gradient checkpointing is enabled
- [ ] Vision encoder is frozen

## Still Having Issues?

### Debug Mode

Add debug flags:

```python
# In train_sft_multimodel.py, add:
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages
os.environ['TORCH_USE_CUDA_DSA'] = '1'     # Device-side assertions
```

### Minimal Reproduction

Create minimal test:

```python
from swift.llm import get_model_tokenizer

model, tokenizer = get_model_tokenizer(
    "OpenGVLab/InternVL3_5-8B",
    model_kwargs={"device_map": "auto"}
)

print(f"✓ Model loaded successfully")
print(f"  Type: {type(model)}")
print(f"  Device: {next(model.parameters()).device}")
```

If this fails, issue is with model loading, not training.

## Using the New Scripts

### Quick Start

```bash
# 1. Check model config
python model_configs.py --model_path OpenGVLab/InternVL3_5-8B

# 2. Edit train_multimodel.sh
# Uncomment your desired model(s)

# 3. Run training
bash train_multimodel.sh
```

### Advanced: Override Defaults

```bash
# Use specific hyperparameters instead of model defaults
python train_sft_multimodel.py \
    --model_path OpenGVLab/InternVL3_5-8B \
    --data_path data.jsonl \
    --output_dir output/ \
    --run_name experiment \
    --learning_rate 5e-6 \      # Override
    --lora_rank 32 \             # Override
    --max_pixels 672             # Override
```

## Summary of Changes

**Old approach** (QwenVL-only):
- Hard-coded `target_modules = ["all-linear"]`
- Fixed learning rate `2e-5`
- Same config for all models ❌

**New approach** (Multi-model):
- Model-specific `target_modules` ✓
- Model-specific learning rates ✓
- Automatic configuration ✓
- Clearer error messages ✓

## Files Created

1. **`model_configs.py`** - Model-specific configurations
2. **`train_sft_multimodel.py`** - Multi-model training script
3. **`train_multimodel.sh`** - Convenient shell wrapper
4. **`MULTIMODEL_TRAINING_DEBUG.md`** - This file

## Example: Full Training Run

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Training
python train_sft_multimodel.py \
    --model_path OpenGVLab/InternVL3_5-8B \
    --data_path /home/work/yuna/HPA/data/training/s1_choice/train_agg_15_blind_inst.jsonl \
    --output_dir ./output/InternVL3_5-8B/D0_SFT_mmstar_gt \
    --run_name InternVL3_5-8B/D0_SFT_mmstar_gt \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --eval_steps 100

# Script automatically:
# - Uses LR 1e-5 (not 2e-5)
# - Sets LoRA rank 16 (not 8)
# - Targets specific layers
# - Uses max_pixels 200704
```

## Contact

If you encounter issues not covered here, check:
1. Swift documentation: https://github.com/modelscope/swift
2. Model-specific docs: HuggingFace model cards
3. GPU memory: Reduce batch size or max_pixels
