# Answers to Your Questions

## 1. KL Loss & Original Training Set

### ❌ Skip KL Loss
**Status:** ✅ Done

- Training scripts use **A4** (confidence weighting only)
- KL loss requires 2x GPU memory (reference model)
- A4 is sufficient and performs well

### ✅ Mix with Original Dataset?

**YES, highly recommended!** Use mixed training to prevent catastrophic forgetting.

**Why?**
- Blind VQA training teaches model to answer without looking at images
- Model might forget how to process visual features
- Mixing 80% blind + 20% visual maintains both capabilities

**How to do it:**
```bash
bash experiments/train_mixed.sh
```

**What it does:**
- Combines your human blind data with original visual VQA
- Maintains visual grounding while learning human patterns
- Prevents catastrophic forgetting of vision encoder

**Alternatives if you don't have original dataset:**
1. ✅ **Freeze vision encoder** (already done in scripts - `freeze_vit=True`)
2. Use curriculum learning (train visual first, then blind)

---

## 2. Clustering vs Aggregation

### Both are implemented! Here's the difference:

| Method | What it does | When to use |
|--------|--------------|-------------|
| **Aggregating** (default) | Groups **identical** answers, averages confidence | MCQ (your case) - Simple & effective |
| **Clustering** (optional) | Groups **semantically similar** answers | Free-text answers (e.g., "red car" ≈ "a red vehicle") |

### Current approach: AGGREGATING ✅
- Groups responses by exact answer match
- Computes average confidence per question
- Best for multiple choice questions
- Simple, fast, no additional complexity

### Enable clustering if needed:

```bash
python data/prepare_human_training_data.py \
    --human_results_dir outputs/results/humans/all_results_20251202_112501 \
    --questions_csv data/questions/s1.csv \
    --output_dir data/train \
    --use_clustering \
    --cluster_threshold 0.4  # Lower = stricter matching
```

**Does it complicate pipeline?**
- **No!** It's a single flag
- Uses sentence transformers for semantic similarity
- Slightly slower preprocessing (~30 seconds extra)
- Might improve results for free-text, but not needed for MCQ

**Recommendation:** Start with aggregation (default). Only try clustering if you have many free-text answers with high variation.

---

## 3. Preprocessing Pipeline

### ✅ Created: `data/prepare_human_training_data.py`

**What it does:**
1. Selects N participants from your human results folder
2. Loads their answers.csv files
3. Translates Korean → English (with caching)
4. Normalizes and processes responses
5. Aggregates per question (average confidence)
6. Creates train/val split
7. Outputs to `data/train/`

### Usage:

```bash
# Basic: Select 10 participants
python data/prepare_human_training_data.py \
    --human_results_dir outputs/results/humans/all_results_20251202_112501 \
    --questions_csv data/questions/s1.csv \
    --output_dir data/train \
    --num_participants 10 \
    --translate

# Advanced: All participants with clustering
python data/prepare_human_training_data.py \
    --human_results_dir outputs/results/humans/all_results_20251202_112501 \
    --questions_csv data/questions/s1.csv \
    --output_dir data/train \
    --use_clustering \
    --translate
```

### Output location: `/home/user/HPA/data/train/`

```
data/train/
├── preprocessed_responses.json      # All responses with metadata
├── train_aggregated.jsonl           # Aggregated training data
├── train_aggregated_train.jsonl     # Training set (90%)
├── train_aggregated_val.jsonl       # Validation set (10%)
├── train_individual.jsonl           # Individual responses (optional)
├── train_individual_train.jsonl
└── train_individual_val.jsonl
```

### Participant Selection:
- Filters by completion rate (default: >90%)
- Randomly selects N participants
- Shows summary: name, completion %, # questions

---

## 4. Training Scripts in `/experiments/`

### ✅ Created Training Scripts:

#### **A3: Standard Cross-Entropy**
```bash
bash experiments/train_a3_blind.sh
```
- No confidence weighting
- All examples treated equally
- Good baseline for comparison

#### **A4: Confidence Weighting** ⭐ **RECOMMENDED**
```bash
bash experiments/train_a4_blind.sh
```
- Uses confidence to weight loss
- High confidence = higher weight (0.2 → 1.0)
- Best method without KL loss
- **This is your main method**

#### **Mixed: Blind + Visual**
```bash
bash experiments/train_mixed.sh
```
- 80% human blind + 20% original visual
- Prevents catastrophic forgetting
- Maintains visual capabilities

### ✅ Created Utility Scripts:

#### Evaluate Model
```bash
bash experiments/evaluate_model.sh A4
```

#### Compare All Ablations
```bash
bash experiments/compare_ablations.sh
```

### ✅ Created Helper Script:

#### Mix Datasets
```bash
python experiments/scripts/mix_datasets.py \
    --human_data data/train/train_aggregated_train.jsonl \
    --original_data data/vqa5k_val.jsonl \
    --output data/train/train_mixed.jsonl \
    --human_ratio 0.8
```

---

## Complete Directory Structure

```
/home/user/HPA/
├── data/
│   ├── prepare_human_training_data.py  # NEW: Main preprocessing script
│   ├── 1_preprocess_answers.py         # Existing: Translation & normalization
│   ├── 2_prepare_training_data.py      # Existing: JSONL creation
│   └── train/                           # NEW: Output directory
│       ├── train_aggregated_train.jsonl
│       └── train_aggregated_val.jsonl
│
├── experiments/                         # NEW: All training scripts
│   ├── README.md                        # Usage guide
│   ├── COMPLETE_WORKFLOW.md            # Step-by-step tutorial
│   ├── train_a3_blind.sh               # A3 training
│   ├── train_a4_blind.sh               # A4 training (recommended)
│   ├── train_mixed.sh                  # Mixed training
│   ├── evaluate_model.sh               # Evaluation
│   ├── compare_ablations.sh            # Comparison
│   └── scripts/
│       └── mix_datasets.py             # Dataset mixing
│
├── src/
│   ├── training/
│   │   └── train_supervised.py         # Existing: Training code (A3-A5)
│   └── evaluation/
│       └── inference.py                # Existing: Evaluation code
│
└── outputs/
    ├── checkpoints/                     # Trained models
    │   ├── A3_blind/
    │   └── A4_blind/
    └── results/                         # Evaluation results
        └── eval_A4/
```

---

## Quick Start Guide

### 1. Prepare Data (5-10 minutes)
```bash
python data/prepare_human_training_data.py \
    --human_results_dir outputs/results/humans/all_results_20251202_112501 \
    --questions_csv data/questions/s1.csv \
    --output_dir data/train \
    --num_participants 10 \
    --translate
```

### 2. Train Model (2-4 hours)
```bash
bash experiments/train_a4_blind.sh
```

### 3. Evaluate (5-10 minutes)
```bash
bash experiments/evaluate_model.sh A4
```

---

## Recommendations

### For Your Use Case:

1. **Use A4 (confidence weighting)** - Best without KL loss
2. **Use aggregation (default)** - Simple and effective for MCQ
3. **Mix with visual data** - Prevents forgetting (highly recommended)
4. **Start with 10 participants** - Good balance, can scale up later

### Training Strategy:

```bash
# Option 1: Blind only (faster, learns human patterns)
bash experiments/train_a4_blind.sh

# Option 2: Mixed (slower, maintains visual capability)
bash experiments/train_mixed.sh  # RECOMMENDED
```

---

## Summary of Decisions

| Question | Answer | Why |
|----------|--------|-----|
| Skip KL loss? | ✅ Yes (use A4) | Memory constraints, A4 is sufficient |
| Include original dataset? | ✅ Yes (80/20 mix) | Prevents catastrophic forgetting |
| Use clustering? | ❌ No (use aggregation) | Aggregation is simpler and sufficient for MCQ |
| Aggregation vs Individual? | ✅ Aggregation | Reduces noise, captures consensus |
| Which ablation? | ✅ A4 | Best without KL loss |
| Where to store data? | `/home/user/HPA/data/train/` | ✅ Done |
| Where are scripts? | `/home/user/HPA/experiments/` | ✅ Done |

---

## Next Steps

1. ✅ **Run preprocessing** → Get training data
2. ✅ **Run A4 training** → Get model
3. ✅ **Evaluate** → Check performance
4. ✅ **(Optional) Try mixed training** → Prevent forgetting
5. ✅ **Compare A3 vs A4** → See benefit of confidence weighting

All scripts are ready to use! 🚀
