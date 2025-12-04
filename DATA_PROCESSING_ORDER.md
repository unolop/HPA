# Data Processing Pipeline

This document outlines the proper order for data processing in the HPA project.

## Directory Structure

```
data/                           # Input data (raw and training)
├── questions/                  # Question datasets
│   ├── s1.csv                 # Study 1 questions
│   └── s2-mmspubench_q400.csv # Study 2 questions
├── vqa_pilot.csv              # VQA pilot dataset
├── pilot/                      # Pilot training data
├── s1_mmstar/                  # Study 1 MMStar training data
└── s1_vqa_1k/                  # Study 1 VQA training data

outputs/results/               # All experiment results
├── humans/                    # Human study results
│   ├── pilot_cleaned/        # Cleaned pilot human responses (FINALIZED)
│   │   ├── _all_pilot_cleaned.jsonl  # Aggregated pilot data
│   │   └── pilot/            # Processed pilot outputs
│   └── all_results_*/        # Study participants' responses
├── processed/                # Processed human responses
│   ├── individual_responses.json
│   ├── s1_choice.json
│   └── s1_text.json
├── swift/                    # Model inference results (main)
│   ├── *_mmstar.jsonl
│   ├── *_vqa_*.jsonl
│   └── scored/
└── check/                    # Experimental/validation results
    ├── pilot/                # Pilot model results
    └── *.jsonl               # VQA validation experiments
```

## Data Processing Flow

### 1. Pilot Data (FINALIZED - DO NOT REPROCESS)

**Status**: ✅ Complete and frozen

**Location**: `outputs/results/humans/pilot_cleaned/`

**Description**: Pilot study data has been fully processed, cleaned, and finalized. Raw files have been removed to prevent accidental reprocessing.

**Files**:
- Individual cleaned responses: `outputs/results/humans/pilot_cleaned/*.jsonl`
- Aggregated data: `outputs/results/humans/pilot_cleaned/_all_pilot_cleaned.jsonl`
- Final processed outputs: `outputs/results/humans/pilot_cleaned/pilot/`

**Processing script** (archived): `src/preprocessing/preprocess_pilot.py`

### 2. Human Study Data Processing

**Pipeline**: Raw CSV → Preprocessed JSON → Training Data

#### Step 1: Preprocess Human Responses

```bash
# Process human study responses with translation
python src/preprocessing/preprocess.py \
    --input_csvs outputs/results/humans/all_results_*/*/*.csv \
    --questions_csv data/questions/s1.csv \
    --output_dir outputs/results/processed \
    --translate --cache_file data/translation_cache.json
```

**Inputs**:
- Raw participant CSVs: `outputs/results/humans/all_results_*/*/answers.csv`
- Questions: `data/questions/s1.csv`

**Outputs**:
- `outputs/results/processed/individual_responses.json` - All individual responses
- `outputs/results/processed/s1_choice.json` - Multiple choice responses
- `outputs/results/processed/s1_text.json` - Free-text responses
- `outputs/results/processed/preprocessing_stats.json` - Statistics

**Features**:
- Korean → English translation with caching
- Answer normalization
- Optional semantic clustering for free-text

#### Step 2: Prepare Training Data

```bash
# Create training data for MMStar (multiple choice)
python src/preprocessing/prepare_training_data.py \
    --responses_path outputs/results/processed/s1_choice.json \
    --questions_csv data/questions/s1.csv \
    --output_dir data/s1_mmstar \
    --with_instruction

# Create training data for VQA (free-text)
python src/preprocessing/prepare_training_data.py \
    --responses_path outputs/results/processed/s1_text.json \
    --questions_csv data/questions/s1.csv \
    --output_dir data/s1_vqa \
    --with_instruction
```

**Inputs**:
- Processed responses JSON
- Questions CSV

**Outputs**:
- `data/s1_mmstar/train_aggregated.jsonl` - Aggregated MCQ training data
- `data/s1_vqa/train_aggregated.jsonl` - Aggregated VQA training data

### 3. Model Training

```bash
# Train with soft supervision (Method A5)
python src/training/train_supervised.py \
    --ablation A5 \
    --model_path OpenGVLab/InternVL3_5-2B \
    --train_data data/s1_mmstar/train_aggregated.jsonl \
    --output_dir outputs/checkpoints/A5 \
    --run_name A5_InternVL3_5-2B_s1_mmstar
```

**Inputs**:
- Training data from Step 2
- Base model path

**Outputs**:
- Model checkpoints: `outputs/checkpoints/A5/`

### 4. Model Evaluation

```bash
# Evaluate trained model
python src/evaluation/evaluate.py \
    --model_path outputs/checkpoints/A5/checkpoint-* \
    --base_model_path OpenGVLab/InternVL3_5-2B \
    --test_data data/test.csv \
    --output_dir outputs/evaluation/A5 \
    --eval_blind
```

**Inputs**:
- Trained model checkpoint
- Test dataset

**Outputs**:
- Evaluation results: `outputs/evaluation/A5/`
- Model predictions: `outputs/results/swift/*.jsonl`

### 5. Analysis

```bash
# Run analysis pipeline
bash scripts/analysis_pipeline.sh
```

**This executes**:
1. Human-model comparison analysis
2. Calibration analysis
3. Visualization generation

**Inputs**:
- Human responses: `outputs/results/processed/individual_responses.json`
- Model results: `outputs/results/swift/*_inst_blind.jsonl`

**Outputs**:
- Analysis results: `outputs/analysis/`
- Figures: `outputs/figures/`

## Important Notes

### ✅ Do's

1. **Always use relative paths** in scripts (relative to project root)
2. **Use the pipeline scripts** in `scripts/` directory for reproducibility
3. **Cache translations** using `--cache_file data/translation_cache.json`
4. **Track preprocessing stats** - review `preprocessing_stats.json` after each run
5. **Keep outputs organized** - all results go to `outputs/`

### ❌ Don'ts

1. **DO NOT reprocess pilot data** - it is finalized and frozen
2. **DO NOT commit large model files** - use `.gitignore`
3. **DO NOT mix raw and processed data** - keep them in separate directories
4. **DO NOT edit processed files manually** - always rerun preprocessing
5. **DO NOT hardcode absolute paths** - use relative paths or config files

## Quick Reference

### Common Commands

```bash
# Full pipeline (from human responses to figures)
bash scripts/pipeline.sh

# Analysis only (after model inference)
bash scripts/analysis_pipeline.sh

# Reprocess human responses
python src/preprocessing/preprocess.py \
    --input_csvs outputs/results/humans/all_results_*/*/*.csv \
    --questions_csv data/questions/s1.csv \
    --output_dir outputs/results/processed \
    --translate --cache_file data/translation_cache.json
```

### Directory Purpose

| Directory | Purpose | Git Tracked |
|-----------|---------|-------------|
| `data/` | Input data and training datasets | Yes (except large files) |
| `src/` | Source code | Yes |
| `scripts/` | Pipeline scripts | Yes |
| `notebooks/` | Jupyter notebooks for EDA | Yes |
| `outputs/results/` | Experiment results | Partial (important results only) |
| `outputs/figures/` | Visualizations | Partial |
| `outputs/temp/` | Temporary files | No |
| `outputs/analysis/` | Analysis results | Yes |

## Troubleshooting

### "File not found" errors
- Check you're running from project root (`/home/user/HPA`)
- Verify input files exist with `ls` before running
- Check file paths in script output

### Translation failures
- Verify `OPENAI_API_KEY` is set in environment
- Check translation cache: `data/translation_cache.json`
- Review failed translations in console output

### Missing dependencies
- Install preprocessing requirements: `pip install openai pandas numpy tqdm`
- Install training requirements: `pip install swift transformers torch`
- Install analysis requirements: `pip install scipy sklearn sentence-transformers`

## Contact

For questions about the data processing pipeline, refer to the individual script documentation or check the notebooks in `notebooks/eda/` for examples.
