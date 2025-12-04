# Training Experiments

This directory contains training scripts for different ablation studies.

## Quick Start

### 1. Prepare Training Data

```bash
# Select 10 participants and create training data
python data/prepare_human_training_data.py \
    --human_results_dir outputs/results/humans/all_results_20251202_112501 \
    --questions_csv data/questions/s1.csv \
    --output_dir data/train \
    --num_participants 10 \
    --translate

# Output: data/train/train_aggregated_train.jsonl
#         data/train/train_aggregated_val.jsonl
```

### 2. Train Models

```bash
# A3: Standard cross-entropy (no confidence weighting)
bash experiments/train_a3_blind.sh

# A4: Confidence-weighted (recommended)
bash experiments/train_a4_blind.sh

# Mixed: Blind + Visual (prevents forgetting)
bash experiments/train_mixed.sh
```

### 3. Evaluate

```bash
# Evaluate a single model
bash experiments/evaluate_model.sh A4

# Compare all ablations
bash experiments/compare_ablations.sh
```

---

## Ablation Studies

### A3: Human Blind + Standard CE
- **Method**: Standard supervised learning
- **Confidence**: ❌ Not used
- **Loss**: Cross-entropy (all examples weighted equally)
- **Use case**: Baseline without confidence weighting

**Train:**
```bash
bash experiments/train_a3_blind.sh
```

### A4: Human Blind + Confidence Weighting ⭐ **RECOMMENDED**
- **Method**: Soft supervised learning
- **Confidence**: ✅ Used to weight loss
- **Loss**: Weighted cross-entropy (conf 1→weight 0.2, conf 5→weight 1.0)
- **Use case**: Main method when KL loss is not feasible
- **Why better**: Learns when humans are uncertain

**Train:**
```bash
bash experiments/train_a4_blind.sh
```

**How it works:**
```python
# Confidence (1-5) → Weight (0.2-1.0)
weight = 0.2 + (confidence - 1) / 4 * 0.8

# Loss per example
loss = weight * cross_entropy(prediction, answer)
```

### Mixed: Blind + Visual (Prevents Catastrophic Forgetting)
- **Method**: Combined training
- **Data split**: 80% human blind + 20% original visual
- **Why needed**: Prevents model from forgetting visual capabilities
- **Use case**: When you need both human patterns AND visual grounding

**Train:**
```bash
bash experiments/train_mixed.sh
```

---

## Aggregation vs Individual Responses

**Current approach: AGGREGATED** (default)

### Aggregated (Recommended)
- Groups all responses per question
- Computes average confidence
- Uses consensus answer
- **Pros**: Reduces noise, captures inter-rater agreement
- **Output**: `train_aggregated.jsonl`

### Individual
- Each participant response = separate training example
- Preserves individual confidence scores
- **Pros**: More training examples, data augmentation
- **Cons**: More noise, slower training
- **Output**: `train_individual.jsonl`

To use individual responses, modify training scripts to use `train_individual_train.jsonl`.

---

## Clustering vs Aggregation

### Aggregation (Default)
- Groups **identical** answers
- Simple, fast, good for MCQ

### Clustering (Optional)
- Groups **semantically similar** answers
- Useful for free-text answers
- Example: "red car" ≈ "a red vehicle"

**Enable clustering:**
```bash
python data/prepare_human_training_data.py \
    --human_results_dir outputs/results/humans/all_results_20251202_112501 \
    --questions_csv data/questions/s1.csv \
    --output_dir data/train \
    --use_clustering \
    --cluster_threshold 0.4  # 0.3=tight, 0.7=loose
```

---

## Preventing Catastrophic Forgetting

When training on blind VQA, the model may forget how to process images.

### Solution 1: Mixed Training (Recommended)
Combine blind VQA with original visual dataset:

```bash
bash experiments/train_mixed.sh
```

### Solution 2: Freeze Vision Encoder
The training scripts already freeze the vision encoder (`freeze_vit=True`), which helps maintain visual capabilities.

### Solution 3: Curriculum Training
1. Train on original visual dataset first
2. Fine-tune on blind VQA data
3. Use lower learning rate for second stage

---

## File Structure

```
experiments/
├── README.md                    # This file
├── train_a3_blind.sh           # A3: Standard CE
├── train_a4_blind.sh           # A4: Confidence weighting (recommended)
├── train_mixed.sh              # Mixed training
├── evaluate_model.sh           # Evaluate single model
├── compare_ablations.sh        # Compare all ablations
└── scripts/
    └── mix_datasets.py         # Mix blind + visual data
```

---

## Training Hyperparameters

Current settings (optimized for 2B models):

```bash
LEARNING_RATE=2e-5
NUM_EPOCHS=3
BATCH_SIZE=1
GRADIENT_ACCUMULATION=16  # Effective batch size: 16
LORA_RANK=32
LORA_ALPHA=64
```

**Adjust for larger models:**
- 7B models: increase `GRAD_ACCUM` to 32
- 13B models: decrease `LORA_RANK` to 16

---

## Expected Results

Based on similar studies:

| Ablation | Expected Accuracy | Notes |
|----------|-------------------|-------|
| A0 (Zero-shot) | 40-50% | Baseline, no training |
| A3 (No confidence) | 55-65% | Standard SFT |
| **A4 (Confidence)** | **60-70%** | **Best without KL** |
| A5 (+ KL) | 65-72% | Requires 2x GPU memory |
| Mixed (Blind+Visual) | 58-68% | Maintains visual capability |

---

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` to 1
- Increase `GRADIENT_ACCUMULATION`
- Reduce `LORA_RANK` to 16
- Use smaller model (2B instead of 7B)

### Low Accuracy
- Check training data quality: `cat data/train/train_aggregated_train.jsonl | head -1 | jq`
- Verify confidence distribution: most should be 3-5
- Try longer training: increase `NUM_EPOCHS` to 5
- Use A4 instead of A3 (confidence weighting helps)

### Model Forgot Visual Capabilities
- Use mixed training: `bash experiments/train_mixed.sh`
- Verify vision encoder is frozen: check `ablation_config.json`

---

## Next Steps

1. **Prepare data** (select N participants)
2. **Train A4** (recommended method)
3. **Evaluate** on validation set
4. **Compare** with A3 to see benefit of confidence weighting
5. **(Optional)** Try mixed training if visual capability is important

Questions? Check the main README in the repository root.
