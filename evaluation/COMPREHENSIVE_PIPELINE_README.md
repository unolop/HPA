# Comprehensive Human-Model Analysis Pipeline

Complete end-to-end pipeline for analyzing human responses against model outputs, with focus on multimodal gains, instruction effects, and finetuning benefits.

## 📋 Overview

This pipeline provides:
1. **Per-question human metrics** from raw crowdsourced data
2. **Model-human correlation analysis** (GT vs visual similarity)
3. **Multimodal gain** computation (baseline - blind conditions)
4. **Instruction effect** analysis (blind_inst vs blind)
5. **Pretrained vs finetuned** distribution comparison
6. **Paper-ready** figures, tables, and LaTeX outputs

## 🎯 Key Research Questions

### 1. **Should we use GT or Visual Similarity?**

**Ground Truth (GT) Similarity**:
- Compares against all 10 VQA annotators
- Measures objective correctness
- **Use when**: Evaluating model accuracy objectively

**Visual Ground Truth Similarity**:
- Compares against "oracle" humans who saw images (multiple_choice_answer)
- Measures alignment with informed humans
- **Use when**: Understanding if models align with human reasoning when full context is available

**Recommendation**: Report **both** and discuss the difference:
- High correlation with GT → Model is objectively correct
- High correlation with Visual GT → Model aligns with human visual reasoning
- Gap between them → Reveals where model reasoning diverges from human perception

### 2. **Distribution Similarity Loss**

Similar to training loss, we compute distribution differences using:
- **KS Test**: Tests if distributions differ
- **Earth Mover's Distance**: Quantifies distribution shift
- **Cohen's d**: Effect size of improvement

This is implemented in the pretrained vs finetuned analysis.

## 🚀 Quick Start

### Step 1: Process Raw Human Responses

```bash
python evaluation/process_raw_human_responses.py \
    --human_data_dir data/humans/all_results_20251206_154732 \
    --session s1 \
    --output_dir evaluation/human_analysis/ \
    --with_similarity
```

**Output**:
- `human_vqa_per_question.jsonl` - Per-question VQA metrics
- `human_mc_per_question.jsonl` - Per-question MC metrics
- `human_vqa_stats.json` - Aggregate VQA statistics
- `human_mc_stats.json` - Aggregate MC statistics
- QID mappings for fair comparison

**What it does**:
- Loads raw CSV responses from crowdsourcing
- Groups by QID using `get_responses_by_qid` logic
- Computes **per-question averages**:
  - Mean accuracy across all human responses
  - Mean confidence
  - Mean GT similarity (vs 10 annotators)
  - Mean visual similarity (vs oracle humans)
- Correlations between confidence and accuracy

### Step 2: Run Comprehensive Analysis

```bash
python evaluation/comprehensive_analysis.py \
    --human_dir evaluation/human_analysis/ \
    --model_dir evaluation/data/scored/ \
    --finetuned_dir evaluation/data/finetuned_scored/ \
    --output_dir evaluation/comprehensive_results/ \
    --dataset vqa_1k
```

**Output**:
- `comprehensive_analysis_summary.json` - All analysis results
- `multimodal_gains_vqa_1k.csv` - MG table
- `instruction_effects_vqa_1k.csv` - Instruction effects table
- Correlation plots for each model
- Distribution comparison plots

**What it does**:
1. **Correlation Analysis**: Merges human and model data on QID, computes Pearson/Spearman correlations
2. **Multimodal Gains**: Computes `baseline_acc - blind_acc` for each model
3. **Instruction Effects**: Computes `inst_blind_acc - blind_acc`
4. **Pretrained vs Finetuned**: Statistical tests (KS, Mann-Whitney, Cohen's d)

### Step 3: Generate Paper Outputs (Jupyter Notebook)

```bash
jupyter notebook evaluation/paper_analysis_notebook.ipynb
```

Run all cells to generate:
- LaTeX tables for paper
- High-resolution figures
- Statistical summaries
- Formatted results

## 📊 Analysis Details

### Analysis 1: Model-Human Correlation

**Question**: How well do model predictions align with human responses?

**Method**:
```python
# Merge on QID
merged = merge_human_model(human_df, model_df)

# Compute correlations
r_gt = pearsonr(merged['correct'], merged['mean_gt_similarity'])
r_visual = pearsonr(merged['correct'], merged['mean_visual_similarity'])
```

**Interpretation**:
- `r > 0.7`: Strong alignment
- `r = 0.4-0.7`: Moderate alignment
- `r < 0.4`: Weak alignment

**Visual GT vs GT**:
- If `r_visual > r_gt`: Model aligns better with informed humans
- If `r_gt > r_visual`: Model is more objectively correct

### Analysis 2: Multimodal Gains

**Question**: How much does visual information help?

**Formula**:
```
MG_absolute = Acc(baseline) - Acc(blind)
MG_relative = (MG_absolute / Acc(baseline)) × 100%
```

**Interpretation**:
- `MG > 30%`: High visual dependence
- `MG = 10-30%`: Moderate visual dependence
- `MG < 10%`: Low visual dependence (good blind reasoning)

**Conditions**:
- **baseline**: Normal evaluation with images
- **blind**: No image provided
- **inst_blind**: No image + instruction about missing image

### Analysis 3: Instruction Effects

**Question**: Do instructions about missing images help?

**Formula**:
```
IE_absolute = Acc(inst_blind) - Acc(blind)
IE_relative = (IE_absolute / Acc(blind)) × 100%
```

**Interpretation**:
- `IE > 0`: Instructions help (model adapts to missing context)
- `IE < 0`: Instructions hurt (model confused by meta-information)
- `IE ≈ 0`: No effect

### Analysis 4: Pretrained vs Finetuned

**Question**: Does finetuning on human blind responses improve alignment?

**Statistical Tests**:
1. **Kolmogorov-Smirnov (KS) Test**:
   - H0: Distributions are the same
   - p < 0.05 → Significant difference

2. **Mann-Whitney U Test**:
   - Tests median differences
   - More robust to outliers

3. **Cohen's d**:
   - Effect size: `d = (μ_ft - μ_pt) / σ_pooled`
   - `|d| > 0.8`: Large effect
   - `|d| = 0.5-0.8`: Medium effect
   - `|d| = 0.2-0.5`: Small effect

**Distribution Similarity Loss**:
```python
# Similar to training, but on accuracy distributions
from scipy.stats import wasserstein_distance

loss = wasserstein_distance(pretrained_scores, finetuned_scores)
```

Lower loss → More similar distributions

## 📁 Output Structure

```
evaluation/
├── human_analysis/                    # Step 1 outputs
│   ├── human_vqa_per_question.jsonl   # Per-question VQA metrics
│   ├── human_mc_per_question.jsonl    # Per-question MC metrics
│   ├── human_vqa_stats.json           # VQA statistics
│   ├── human_mc_stats.json            # MC statistics
│   ├── human_vqa_qids.json            # VQA question IDs
│   └── human_mc_qids.json             # MC question IDs
│
├── comprehensive_results/             # Step 2 outputs
│   ├── comprehensive_analysis_summary.json
│   ├── multimodal_gains_vqa_1k.csv
│   ├── multimodal_gains_vqa_1k.png
│   ├── instruction_effects_vqa_1k.csv
│   ├── instruction_effects_vqa_1k.png
│   ├── pretrained_vs_finetuned_vqa_1k_inst_blind.png
│   └── correlations/
│       ├── InternVL3_5-2B/
│       │   ├── correlation_mean_gt_similarity.png
│       │   └── correlation_mean_visual_similarity.png
│       └── ...
│
└── paper_results/                     # Step 3 outputs (notebook)
    ├── figures/
    │   ├── correlation_analysis.png
    │   └── ...
    ├── table1_human_baseline.tex
    ├── table2_multimodal_gains.tex
    └── ...
```

## 📝 Per-Question Data Format

### VQA (human_vqa_per_question.jsonl)

```json
{
  "qid": "418297001",
  "answer_type": "text",
  "num_responses": 10,
  "answers": ["yes", "yes", "no", "yes", ...],
  "confidences": [0.75, 1.0, 0.5, 0.75, ...],
  "gt_answers": ["yes", "yes", "yes", "no", ...],
  "visual_gt": "yes",
  "mean_accuracy": 0.8333,
  "std_accuracy": 0.2357,
  "mean_confidence": 0.75,
  "std_confidence": 0.1893,
  "accuracies": [1.0, 1.0, 0.33, 1.0, ...],
  "mean_gt_similarity": 0.8234,
  "std_gt_similarity": 0.1234,
  "gt_similarities": [0.95, 0.87, 0.42, ...],
  "mean_visual_similarity": 0.9123,
  "std_visual_similarity": 0.0876,
  "visual_similarities": [0.98, 0.95, 0.67, ...]
}
```

**Key Fields**:
- `mean_accuracy`: Average VQA accuracy across all human responses
- `mean_gt_similarity`: Average embedding similarity to GT annotators
- `mean_visual_similarity`: Average embedding similarity to visual oracle
- Individual lists preserved for further analysis

### Multiple Choice (human_mc_per_question.jsonl)

```json
{
  "qid": "123",
  "answer_type": "choice",
  "num_responses": 10,
  "answers": ["A", "A", "B", "A", ...],
  "extracted_choices": ["A", "A", "B", "A", ...],
  "confidences": [0.75, 1.0, 0.5, 0.75, ...],
  "gt_answer": "A",
  "category": "coarse perception",
  "l2_category": "image scene",
  "mean_accuracy": 0.8,
  "std_accuracy": 0.4,
  "mean_confidence": 0.7,
  "std_confidence": 0.18,
  "accuracies": [1, 1, 0, 1, ...]
}
```

## 🎓 Paper Integration

### Methods Section

Use this structure:

```latex
\subsection{Human Response Analysis}

We collected $n=10$ responses per question from crowdworkers under blind
conditions (no images provided). For each question, we computed:

\begin{itemize}
    \item \textbf{Mean Accuracy}: Average VQA accuracy using
          $\text{acc} = \min(1, \#\text{matches}/3)$ against ground truth
          annotators
    \item \textbf{GT Similarity}: Average cosine similarity between human
          responses and ground truth annotators
    \item \textbf{Visual Similarity}: Average similarity to visual ground
          truth (humans who saw images)
\end{itemize}

\subsection{Model-Human Alignment}

We measured alignment using Pearson correlation between model accuracy and
human similarity metrics on matched question sets. We report both GT
similarity (objective correctness) and visual similarity (alignment with
informed humans).

\subsection{Multimodal Gains}

We define multimodal gain (MG) as:
\begin{equation}
    MG = \text{Acc}_{\text{baseline}} - \text{Acc}_{\text{blind}}
\end{equation}

This quantifies performance loss when visual information is removed.

\subsection{Distribution Analysis}

We compared pretrained and finetuned models using:
\begin{itemize}
    \item Kolmogorov-Smirnov test for distributional differences
    \item Cohen's $d$ for effect size:
          $d = (\mu_{\text{ft}} - \mu_{\text{pt}}) / \sigma_{\text{pooled}}$
\end{itemize}
```

### Results Section Template

```latex
\subsection{Human Performance Baseline}

Human participants achieved mean accuracy of $X \pm Y$ on VQA tasks and
$Z \pm W$ on multiple choice tasks. Confidence-accuracy correlation was
$r = R$ ($p < 0.001$), indicating [well/poorly]-calibrated responses.

\subsection{Model-Human Correlation}

Model predictions showed [strong/moderate/weak] correlation with human
GT similarity ($r = R_{\text{GT}}$, $p < 0.001$) and [higher/lower]
correlation with visual similarity ($r = R_{\text{visual}}$), suggesting...

\subsection{Multimodal Gains}

Models exhibited mean MG of $M\%$ (range: $[M_{\min}, M_{\max}]$).
[Model X] showed highest MG ($MG_{\max}\%$), indicating strong visual
dependence, while [Model Y] showed lowest MG ($MG_{\min}\%$), suggesting
better blind reasoning ability.

\subsection{Finetuning Effects}

Finetuning on human blind responses improved performance by $I \pm J$
with medium effect size (Cohen's $d = D$). Distribution comparison showed
significant shift (KS test: $p < 0.001$).
```

## ⚙️ Configuration Options

### Process Raw Human Responses

```bash
python evaluation/process_raw_human_responses.py \
    --human_data_dir data/humans/all_results_20251206_154732 \
    --session s1 \
    --output_dir evaluation/human_analysis/ \
    --with_similarity  # Optional: compute embedding similarity
```

### Comprehensive Analysis

```bash
python evaluation/comprehensive_analysis.py \
    --human_dir evaluation/human_analysis/ \
    --model_dir evaluation/data/scored/ \
    --finetuned_dir evaluation/data/finetuned_scored/ \
    --output_dir evaluation/comprehensive_results/ \
    --dataset vqa_1k  # or vqa_5k, mmstar, spubench
```

## 🔍 Interpreting Results

### GT vs Visual Similarity

| Scenario | r_GT | r_visual | Interpretation |
|----------|------|----------|----------------|
| Scenario A | 0.7 | 0.8 | Model aligns well with human visual reasoning |
| Scenario B | 0.8 | 0.6 | Model is objectively correct but diverges from human perception |
| Scenario C | 0.5 | 0.5 | Moderate alignment on both dimensions |

### Multimodal Gains

| MG | Interpretation | Action |
|----|----------------|--------|
| >30% | High visual dependence | Improve reasoning without images |
| 10-30% | Moderate dependence | Balanced multimodal learning |
| <10% | Low dependence | Strong blind reasoning (may hallucinate less) |

### Instruction Effects

| IE | Interpretation |
|----|----------------|
| Positive | Model benefits from explicit context |
| Negative | Model confused by meta-information |
| Near zero | Model ignores instruction |

### Finetuning Benefits

| Cohen's d | Magnitude | Interpretation |
|-----------|-----------|----------------|
| <0.2 | Negligible | Finetuning didn't help |
| 0.2-0.5 | Small | Minor improvement |
| 0.5-0.8 | Medium | Meaningful improvement |
| >0.8 | Large | Substantial improvement |

## 📚 Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy jupyter tqdm sentence-transformers torch
```

## 🐛 Troubleshooting

### Issue: "No matching QIDs"

**Cause**: QID format mismatch (string vs int)

**Fix**: Check QID types in both datasets:
```python
human_df['qid'] = human_df['qid'].astype(str)
model_df['qid'] = model_df['qid'].astype(str)
```

### Issue: "VQA annotations not found"

**Cause**: Missing ground truth file

**Fix**: Update path in scripts:
```python
VQA_ANNOTATIONS_PATH = "/path/to/v2_mscoco_val2014_annotations.json"
```

### Issue: "CUDA out of memory"

**Cause**: Sentence transformer on GPU

**Fix**: Use CPU:
```python
def get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")  # Remove .to('cuda')
```

## 📖 Citation

If you use this pipeline, please cite:

```bibtex
@article{yourpaper2025,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Venue},
  year={2025}
}
```

## 🤝 Contributing

This pipeline is designed to be modular and extensible. To add new analyses:

1. Add analysis function to `comprehensive_analysis.py`
2. Add visualization function
3. Update notebook with new section
4. Document in this README

---

**Version**: 1.0
**Last Updated**: 2025-12-09
**Contact**: [Your contact info]
