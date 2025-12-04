# HPA: Human-Annotated VQA for Vision Language Model Training

Training vision-language models with human blind VQA responses and confidence-weighted soft supervision.

---

## 1. Data Processing

### Pipeline Overview

```
Raw Human Responses → Preprocessing → Training Data → Model Training → Evaluation → Analysis
```

### Step 1: Preprocess Human Responses

Convert raw CSV responses to normalized JSON with translation support.

```bash
python src/preprocessing/preprocess.py \
    --input_csvs outputs/results/humans/all_results_*/*/*.csv \
    --questions_csv data/questions/s1.csv \
    --output_dir outputs/results/processed \
    --translate --cache_file data/translation_cache.json
```

**Input**:
- `outputs/results/humans/all_results_*/*/answers.csv` - Raw participant responses
- `data/questions/s1.csv` - Question metadata

**Output**:
- `outputs/results/processed/individual_responses.json` - All responses (2.3MB)
- `outputs/results/processed/s1_choice.json` - Multiple choice only (1.3MB)
- `outputs/results/processed/s1_text.json` - Free-text only (1.4MB)
- `outputs/results/processed/*_stats.json` - Processing statistics

**Features**:
- Korean → English translation with caching
- Answer normalization (lowercase, article removal, whitespace)
- Confidence score preservation

### Step 2: Prepare Training Data

Aggregate responses per question with confidence distributions.

```bash
# For MMStar (multiple choice)
python src/preprocessing/prepare_training_data.py \
    --responses_path outputs/results/processed/s1_choice.json \
    --questions_csv data/questions/s1.csv \
    --output_dir data/s1_mmstar \
    --with_instruction

# For VQA (free-text)
python src/preprocessing/prepare_training_data.py \
    --responses_path outputs/results/processed/s1_text.json \
    --questions_csv data/questions/s1.csv \
    --output_dir data/s1_vqa \
    --with_instruction
```

**Input**: Processed responses JSON from Step 1

**Output**:
- `data/s1_mmstar/train_aggregated.jsonl` - MCQ training data
- `data/s1_vqa/train_aggregated.jsonl` - VQA training data

**Format**: Each line contains aggregated human responses with confidence distribution per question.

### Quick Start

Run the full preprocessing pipeline:

```bash
bash scripts/pipeline.sh
```

---

## 2. Fine-tuning

### Training Methods

We implement 5 ablation methods (A0-A5) comparing different supervision strategies:

#### **A0: Zero-Shot Baseline**
- No training
- Direct evaluation on base model
- Baseline for comparison

#### **A1: Ground Truth + Real Images**
- Standard supervised learning
- Uses correct answers with real images
- Traditional VQA training approach

#### **A2: Ground Truth + Blind (Black Images)**
- Standard supervised learning with black images
- Tests language-only capabilities
- Ablation control for blind VQA

#### **A3: Human Blind + Standard Cross-Entropy**
- Uses human responses from blind VQA
- Standard hard-label training (majority vote)
- No confidence weighting

#### **A4: Human Blind + Confidence Weighting**
- Uses human responses from blind VQA
- Weighted loss by confidence scores
- Answer distribution considered

#### **A5: Human Blind + Confidence + KL Regularization** ⭐ **MAIN METHOD**
- Uses human responses from blind VQA
- Confidence-weighted soft supervision
- KL divergence regularization to match human answer distribution
- Preserves uncertainty in ambiguous cases

### Training Command

```bash
# Main method (A5) - Recommended
python src/training/train_supervised.py \
    --ablation A5 \
    --model_path OpenGVLab/InternVL3_5-2B \
    --train_data data/s1_mmstar/train_aggregated.jsonl \
    --output_dir outputs/checkpoints/A5 \
    --run_name A5_InternVL3_5-2B_s1_mmstar

# For other ablations, change --ablation to A0, A1, A2, A3, or A4
```

**Key Parameters**:
- `--ablation`: Training method (A0-A5)
- `--model_path`: HuggingFace model identifier
- `--train_data`: Path to aggregated training data
- `--output_dir`: Checkpoint save directory
- `--run_name`: Experiment identifier for logging

**Supported Models**:
- InternVL3.5 series (1B, 2B, 4B, 8B)
- Qwen2-VL, Qwen3-VL series
- LLaVA series

**Output**:
- `outputs/checkpoints/{ablation}/checkpoint-*` - Model checkpoints
- `outputs/checkpoints/{ablation}/trainer_state.json` - Training metrics

---

## 3. Evaluation

### Inference on Benchmark Datasets

Run model inference on test sets:

```bash
# Using SWIFT for batch inference
bash scripts/run_inference.sh

# Or manually
python src/evaluation/inference.py \
    --model_path outputs/checkpoints/A5/checkpoint-best \
    --dataset mmstar \
    --output_dir outputs/results/swift \
    --blind  # For blind VQA evaluation
```

**Supported Datasets**:
- `mmstar` - MMStar benchmark
- `vqa_1k` - VQA v2 1k subset
- `vqa_5k` - VQA v2 5k subset
- `spubench` - SPUBench

**Output**:
- `outputs/results/swift/{model}_{dataset}_inst_blind.jsonl` - Predictions with correctness

### Analysis Pipeline

Compare human and model performance:

```bash
bash scripts/analysis_pipeline.sh
```

**This runs**:
1. **Human-Model Comparison** (`src/analysis/analyze_human_model.py`)
   - Agreement matrix (both correct, both wrong, disagreements)
   - Per-category accuracy breakdown
   - Interesting cases analysis

2. **Calibration Analysis** (`src/analysis/analyze_calibration.py`)
   - Confidence-accuracy correlation
   - Calibration curves (ECE, MCE)
   - Over/under-confidence patterns

3. **Visualization** (`src/analysis/visualize_results.py`)
   - Scatter plots (human vs model accuracy)
   - Category comparisons
   - Agreement heatmaps

**Input**:
- `outputs/results/processed/individual_responses.json` - Human responses
- `outputs/results/swift/*_inst_blind.jsonl` - Model predictions

**Output**:
- `outputs/analysis/` - JSON analysis results
- `outputs/figures/` - PNG visualizations

---

## Directory Structure

```
HPA/
├── src/                        # Source code
│   ├── preprocessing/          # Data processing scripts
│   ├── training/               # Training scripts
│   ├── evaluation/             # Inference scripts
│   └── analysis/               # Analysis scripts
├── scripts/                    # Pipeline automation
│   ├── pipeline.sh             # Full data processing pipeline
│   └── analysis_pipeline.sh   # Analysis workflow
├── data/                       # Input data
│   ├── questions/              # Question datasets
│   ├── s1_mmstar/              # MCQ training data
│   └── s1_vqa/                 # VQA training data
├── outputs/                    # All outputs
│   ├── results/                # Human & model results
│   ├── analysis/               # Analysis results
│   ├── figures/                # Visualizations
│   └── checkpoints/            # Model checkpoints (not tracked)
└── notebooks/                  # Jupyter notebooks for EDA
```

---

## Quick Reference

```bash
# 1. Process data
bash scripts/pipeline.sh

# 2. Train model (A5 method)
python src/training/train_supervised.py \
    --ablation A5 \
    --model_path OpenGVLab/InternVL3_5-2B \
    --train_data data/s1_mmstar/train_aggregated.jsonl \
    --output_dir outputs/checkpoints/A5

# 3. Run inference
bash scripts/run_inference.sh

# 4. Analyze results
bash scripts/analysis_pipeline.sh
```

---

## Requirements

```bash
# Core
pip install torch transformers datasets

# Training
pip install swift-ms  # For InternVL models

# Preprocessing
pip install openai pandas numpy tqdm

# Analysis
pip install scipy scikit-learn matplotlib seaborn
```

---

## Citation

If you use this code or data, please cite our work:

```bibtex
@article{hpa2024,
  title={Human-Annotated VQA for Vision Language Model Training},
  author={Your Name},
  year={2024}
}
```

---

## License

MIT License - See LICENSE file for details
