# Human-Model Alignment Analysis Pipeline

Complete pipeline for analyzing human responses and comparing with vision-language model outputs.

## Overview

This pipeline processes human crowdsourced responses, computes evaluation metrics, and generates comparative visualizations to assess human-model alignment on VQA and multiple-choice tasks.

## Files

### Core Scripts

1. **`score_human_results.py`** - Process and score human responses
   - Computes VQA accuracy and embedding similarity
   - Extracts MC choices and compares with ground truth
   - Saves QID mappings for later comparison
   - Generates statistics and distributions

2. **`visualize_human_analysis.py`** - Generate visualizations
   - Confidence distributions
   - Accuracy comparisons (human vs models)
   - Answer similarity distributions
   - Question type breakdowns
   - Confidence-accuracy calibration plots

3. **`score_results.py`** - Score model outputs (existing)
   - Processes model inference results
   - Computes same metrics as human scoring
   - Now includes `--with_similarity` flag

### Documentation

- **`ANALYSIS_PIPELINE.tex`** - Comprehensive LaTeX documentation for paper
- **`HUMAN_ANALYSIS_README.md`** - This file

## Quick Start

### 1. Score Human Responses

```bash
# VQA (text answers)
python evaluation/score_human_results.py \
    --text_data data/training/s1_text/train_agg_10_blind_inst.jsonl \
    --output_dir evaluation/human_scored/ \
    --with_similarity

# Multiple Choice
python evaluation/score_human_results.py \
    --choice_data data/training/s1_choice/train_agg_10_blind_inst.jsonl \
    --output_dir evaluation/human_scored/

# Both together
python evaluation/score_human_results.py \
    --text_data data/training/s1_text/train_agg_10_blind_inst.jsonl \
    --choice_data data/training/s1_choice/train_agg_10_blind_inst.jsonl \
    --output_dir evaluation/human_scored/ \
    --with_similarity
```

### 2. Score Model Outputs (if not done)

```bash
python evaluation/score_results.py \
    --input_dir data/models/ \
    --output_dir evaluation/data/scored/ \
    --with_similarity
```

### 3. Generate Visualizations

```bash
# VQA visualizations (human only)
python evaluation/visualize_human_analysis.py \
    --human_dir evaluation/human_scored/ \
    --output_dir evaluation/figures/ \
    --answer_type vqa

# VQA with model comparison
python evaluation/visualize_human_analysis.py \
    --human_dir evaluation/human_scored/ \
    --model_dir evaluation/data/scored/ \
    --models InternVL3_5-2B Qwen3-VL-4B-Instruct llava-v1.6-mistral-7b-hf \
    --dataset vqa_1k_inst_blind \
    --output_dir evaluation/figures/ \
    --answer_type vqa

# Multiple Choice visualizations
python evaluation/visualize_human_analysis.py \
    --human_dir evaluation/human_scored/ \
    --output_dir evaluation/figures/ \
    --answer_type mc
```

## Output Structure

### Scored Results

```
evaluation/human_scored/
├── human_vqa_scored.jsonl      # Individual scored VQA responses
├── human_vqa_stats.json        # Aggregate statistics
├── human_vqa_qids.json         # Question ID mappings
├── human_mc_scored.jsonl       # Individual scored MC responses
├── human_mc_stats.json         # Aggregate statistics
└── human_mc_qids.json          # Question ID mappings
```

### Visualizations

```
evaluation/figures/
├── human_vqa_confidence_dist.png
├── human_model_accuracy_comparison.png
├── human_model_similarity_comparison.png
├── human_vqa_question_types.png
├── human_confidence_vs_accuracy.png
├── human_vqa_answer_dist.png
└── human_mc_*.png
```

## Data Format

### Human VQA Result

```json
{
  "qid": "418297001",
  "answer": "yes",
  "confidence": 0.75,
  "accuracy": 1.0,
  "correct": true,
  "gt_answers": ["yes", "yes", "yes", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
  "all_human_answers": ["yes", "no"],
  "all_confidences": [0.75, 0.25],
  "answer_similarity": 0.98,
  "question_type": "is"
}
```

### Statistics File

```json
{
  "total_questions": 374,
  "total_responses": 3740,
  "mean_accuracy": 0.6234,
  "std_accuracy": 0.3421,
  "correct_count": 267,
  "mean_similarity": 0.7845,
  "std_similarity": 0.1923,
  "confidence_dist": {
    "0.05": 45,
    "0.25": 123,
    "0.5": 892,
    "0.75": 1456,
    "1.0": 1224
  },
  "answer_dist": {
    "yes": 456,
    "no": 234,
    "2": 123,
    ...
  },
  "question_type_dist": {
    "what": 156,
    "is": 89,
    "how": 67,
    ...
  }
}
```

### QID Mapping

```json
{
  "qids": [
    "418297001",
    "444302001",
    "262245002",
    ...
  ],
  "count": 374
}
```

## Metrics

### VQA Accuracy

Standard VQA v2 metric:

```
accuracy = min(1, #matches / 3)
```

An answer is correct if ≥3 out of 10 ground truth annotators gave the same answer.

### Multiple Choice Accuracy

Binary correctness:

```
accuracy = 1 if extracted_choice == ground_truth else 0
```

### Answer Similarity

Cosine similarity using SentenceTransformer embeddings:

```
similarity = cos(embedding_pred, embedding_gt)
```

### Confidence Mapping

```
Rating    | Confidence
----------|------------
1         | 0.05
2         | 0.25
3         | 0.50
4         | 0.75
5         | 1.00
yes       | 1.00
maybe     | 0.50
no        | 0.01
```

## Analysis Use Cases

### 1. Human Performance Baseline

Establish human accuracy and confidence patterns:

```bash
python evaluation/score_human_results.py \
    --text_data data/training/s1_text/train_agg_10_blind_inst.jsonl \
    --output_dir evaluation/human_scored/ \
    --with_similarity

# Check statistics
cat evaluation/human_scored/human_vqa_stats.json | jq '.mean_accuracy'
```

### 2. Model Comparison

Compare specific models with human performance:

```bash
python evaluation/visualize_human_analysis.py \
    --human_dir evaluation/human_scored/ \
    --model_dir evaluation/data/scored/ \
    --models InternVL3_5-2B Qwen3-VL-4B-Instruct \
    --dataset vqa_1k_inst_blind \
    --output_dir evaluation/figures/ \
    --answer_type vqa
```

### 3. Confidence Calibration

Analyze whether confidence predicts accuracy:

The `human_confidence_vs_accuracy.png` plot shows:
- Scatter plot with trend line
- Binned averages by confidence level
- Correlation coefficient

### 4. Question Type Analysis

Identify which question types are easiest/hardest:

```python
import json
with open('evaluation/human_scored/human_vqa_stats.json') as f:
    stats = json.load(f)

qt_dist = stats['question_type_dist']
# Analyze distribution
```

### 5. Distribution Matching

Compare human and model distributions:

```python
import json
import scipy.stats as stats

# Load data
with open('evaluation/human_scored/human_vqa_scored.jsonl') as f:
    human = [json.loads(line)['accuracy'] for line in f]

with open('evaluation/data/scored/InternVL3_5-2B_vqa_1k_inst_blind.jsonl') as f:
    model = [json.loads(line)['correct'] for line in f]

# KS test
ks_stat, p_value = stats.ks_2samp(human, model)
print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")
```

## Using QID Mappings

The saved QIDs enable fair comparison on identical question sets:

```python
import json

# Load human QIDs
with open('evaluation/human_scored/human_vqa_qids.json') as f:
    human_qids = set(json.load(f)['qids'])

# Filter model results to matching QIDs
with open('evaluation/data/scored/model_results.jsonl') as f:
    filtered = [
        json.loads(line) for line in f
        if str(json.loads(line)['qid']) in human_qids
    ]

# Now compute metrics on filtered set
```

## Troubleshooting

### Issue: "VQA annotations not found"

**Solution**: Check that VQA annotations file exists at:
```
/home/work/yuna/VLMEval/data/v2_mscoco_val2014_annotations.json
```

Update path in scripts if needed.

### Issue: "CUDA out of memory"

**Solution**: For CPU-only execution, modify `get_encoder()` in scripts:

```python
def get_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")  # Remove .to('cuda')
    except:
        return None
```

### Issue: "No similarity data available"

**Solution**: Run with `--with_similarity` flag:

```bash
python evaluation/score_human_results.py \
    --text_data ... \
    --output_dir ... \
    --with_similarity  # Add this flag
```

### Issue: "Model file not found"

**Solution**: Check model naming convention. Expected format:
```
{ModelName}_{dataset}_{condition}.jsonl

Examples:
- InternVL3_5-2B_vqa_1k_inst_blind.jsonl
- Qwen3-VL-4B-Instruct_mmstar.jsonl
```

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn tqdm sentence-transformers torch
```

## Paper Integration

The LaTeX documentation (`ANALYSIS_PIPELINE.tex`) provides:
- Complete methodology description
- Algorithm pseudocode
- Metric definitions
- Expected results template
- Statistical analysis procedures

Use it as a reference when writing the methods and results sections.

## Future Enhancements

- [ ] Category-wise analysis for VQA question types
- [ ] Difficulty stratification (easy/medium/hard questions)
- [ ] Multi-turn conversational analysis
- [ ] Uncertainty quantification comparison
- [ ] Fine-grained error analysis by question category
- [ ] Temporal analysis of confidence evolution

## Questions?

For issues or questions, refer to:
1. `ANALYSIS_PIPELINE.tex` - Detailed methodology
2. Script docstrings - Usage examples
3. This README - Quick reference

---

**Created**: 2025-12-09
**Last Updated**: 2025-12-09
**Version**: 1.0
