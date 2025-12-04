# Complete Training Workflow

## Step-by-Step Guide

### 1️⃣ Prepare Training Data

Select participants and preprocess their responses:

```bash
cd /home/user/HPA

# Select 10 participants (adjust --num_participants as needed)
python data/prepare_human_training_data.py \
    --human_results_dir outputs/results/humans/all_results_20251202_112501 \
    --questions_csv data/questions/s1.csv \
    --output_dir data/train \
    --num_participants 10 \
    --translate
```

**What this does:**
1. Selects N participants (default: all, or specify with `--num_participants`)
2. Filters by completion rate (default: >90%)
3. Translates Korean answers to English (if `--translate` flag)
4. Normalizes answers
5. Aggregates responses per question (computes average confidence)
6. Creates train/val split (90/10)

**Output files:**
```
data/train/
├── preprocessed_responses.json      # Raw preprocessed responses
├── train_aggregated.jsonl           # All aggregated data
├── train_aggregated_train.jsonl     # Training set (90%)
├── train_aggregated_val.jsonl       # Validation set (10%)
├── train_individual.jsonl           # All individual responses
├── train_individual_train.jsonl     # Individual training set
└── train_individual_val.jsonl       # Individual validation set
```

**Check the data:**
```bash
# View first example
cat data/train/train_aggregated_train.jsonl | head -1 | jq

# Count examples
wc -l data/train/train_aggregated_train.jsonl
wc -l data/train/train_aggregated_val.jsonl

# Check confidence distribution
cat data/train/train_aggregated_train.jsonl | jq -r '.confidence' | sort | uniq -c
```

---

### 2️⃣ Train Model (Choose One)

#### Option A: A4 Confidence Weighting ⭐ **RECOMMENDED**

```bash
bash experiments/train_a4_blind.sh
```

- Uses confidence to weight training loss
- High confidence examples have more influence
- Best method without KL loss
- Training time: ~2-4 hours on single GPU

#### Option B: A3 Standard CE (Baseline)

```bash
bash experiments/train_a3_blind.sh
```

- Standard supervised learning
- All examples weighted equally
- Good baseline for comparison

#### Option C: Mixed Training (Blind + Visual)

```bash
# Edit train_mixed.sh to set your original dataset path
nano experiments/train_mixed.sh
# Change: ORIGINAL_DATA="data/vqa5k_val.jsonl"

bash experiments/train_mixed.sh
```

- Combines blind VQA with original visual dataset
- Prevents catastrophic forgetting
- Maintains visual capabilities

---

### 3️⃣ Monitor Training

Training logs will show:
```
Epoch 1/3
Step 50/300: loss=2.134, lr=1.9e-5
Step 100/300: loss=1.876, lr=1.8e-5
...
Validation: accuracy=0.65, loss=1.543
✅ Saved checkpoint: outputs/checkpoints/A4_blind/checkpoint-100
```

**Check progress:**
```bash
# View training logs
tail -f outputs/checkpoints/A4_blind/training.log

# Check saved checkpoints
ls -lh outputs/checkpoints/A4_blind/

# View config
cat outputs/checkpoints/A4_blind/ablation_config.json
```

---

### 4️⃣ Evaluate Model

```bash
# Evaluate on validation set
bash experiments/evaluate_model.sh A4

# Or specify custom test set
bash experiments/evaluate_model.sh A4 data/test/custom_test.jsonl
```

**Output:**
```
outputs/results/eval_A4/
├── metrics.json              # Overall metrics
├── predictions.jsonl         # Per-question predictions
├── confusion_matrix.png      # Confusion matrix
└── calibration_plot.png      # Confidence vs accuracy
```

**View results:**
```bash
cat outputs/results/eval_A4/metrics.json | jq
```

---

### 5️⃣ Compare Ablations

```bash
bash experiments/compare_ablations.sh
```

Shows side-by-side comparison:
```
Ablation        Accuracy        F1              Calibration ECE
--------------------------------------------------------------------
A3_blind        0.58            0.56            0.12
A4_blind        0.65            0.63            0.08  ← Better!
A4_mixed        0.62            0.60            0.09
```

---

## Advanced Usage

### Use Clustering Instead of Aggregation

For free-text answers, use semantic clustering:

```bash
python data/prepare_human_training_data.py \
    --human_results_dir outputs/results/humans/all_results_20251202_112501 \
    --questions_csv data/questions/s1.csv \
    --output_dir data/train_clustered \
    --use_clustering \
    --cluster_threshold 0.4  # 0.3=tight, 0.7=loose
```

**Effect:** Groups similar answers like "red car" ≈ "a red vehicle"

### Use Individual Responses

```bash
# In train_a4_blind.sh, change:
TRAIN_DATA="data/train/train_individual_train.jsonl"
VAL_DATA="data/train/train_individual_val.jsonl"
```

**Effect:** More training examples but noisier (not recommended)

### Adjust Participant Selection

```bash
# Select specific number
--num_participants 15

# Include lower completion rates
--min_completion 0.7

# Use specific random seed for reproducibility
--seed 123
```

### Skip Translation

If answers are already in English:

```bash
python data/prepare_human_training_data.py \
    --human_results_dir outputs/results/humans/all_results_20251202_112501 \
    --questions_csv data/questions/s1.csv \
    --output_dir data/train \
    --no_translate  # Skip translation
```

---

## Troubleshooting

### Issue: "No participants found"

**Check:**
```bash
ls outputs/results/humans/all_results_20251202_112501/
```

**Solution:** Verify the path is correct and contains participant folders

### Issue: "Translation failed"

**Solutions:**
1. Set OpenAI API key: `export OPENAI_API_KEY="sk-..."`
2. Or skip translation: add `--no_translate` flag
3. Check cache: `ls -lh data/translation_cache.json`

### Issue: "Out of memory during training"

**Solutions:**
```bash
# In train_a4_blind.sh, reduce batch size:
BATCH_SIZE=1
GRAD_ACCUM=32  # Increase to maintain effective batch size

# Or use smaller LoRA rank:
LORA_RANK=16
LORA_ALPHA=32
```

### Issue: "Model forgot visual capabilities"

**Solution:** Use mixed training
```bash
bash experiments/train_mixed.sh
```

---

## FAQ

**Q: Should I use aggregated or individual responses?**
A: **Aggregated** (default). It reduces noise and captures inter-rater agreement.

**Q: Should I use clustering?**
A: Only if you have free-text answers. For MCQ (your case), aggregation is sufficient.

**Q: Which ablation should I use?**
A: **A4** (confidence weighting). It's the best method without KL loss.

**Q: Should I mix with original visual dataset?**
A: **Yes**, if you need the model to maintain visual capabilities. Otherwise, optional.

**Q: How many participants should I select?**
A: Start with 10-15. More participants = better consensus but slower preprocessing.

**Q: How do I know if training is working?**
A: Loss should decrease over time. Final validation accuracy should be >60%.

---

## Summary Commands

```bash
# 1. Prepare data
python data/prepare_human_training_data.py \
    --human_results_dir outputs/results/humans/all_results_20251202_112501 \
    --questions_csv data/questions/s1.csv \
    --output_dir data/train \
    --num_participants 10 \
    --translate

# 2. Train (recommended)
bash experiments/train_a4_blind.sh

# 3. Evaluate
bash experiments/evaluate_model.sh A4

# 4. Compare
bash experiments/compare_ablations.sh
```

That's it! 🚀
