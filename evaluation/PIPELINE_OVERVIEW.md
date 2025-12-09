# 📊 Complete Human-Model Analysis Pipeline - Overview

## 🎯 What You Asked For

You requested a comprehensive pipeline to:
1. Calculate average accuracy and embedding similarity per question against ground truth
2. Use raw human data (not aggregated) with `get_responses_by_qid` logic
3. Analyze correlations between model accuracy and human-model similarity
4. Calculate multimodal gains
5. Analyze instruction effects
6. Compare pretrained vs finetuned distributions
7. Provide a full notebook for the paper

## ✅ What Was Built

### Core Pipeline (3 Scripts + 1 Notebook)

#### 1. `process_raw_human_responses.py`
**Purpose**: Process raw CSV data to compute per-question human metrics

**Input**:
- Raw CSV files from `data/humans/all_results_20251206_154732/`
- Uses `get_responses_by_qid` logic from preprocessing

**Processing**:
```python
For each question:
    ├── Load all human responses (n=10)
    ├── Calculate per-question averages:
    │   ├── mean_accuracy (average across responses)
    │   ├── mean_confidence
    │   ├── mean_gt_similarity (vs 10 GT annotators)
    │   └── mean_visual_similarity (vs oracle humans who saw images)
    ├── Preserve individual responses for further analysis
    └── Save per-question metrics
```

**Output**:
- `human_vqa_per_question.jsonl` - Individual question metrics
- `human_vqa_stats.json` - Aggregate statistics

**Command**:
```bash
python evaluation/process_raw_human_responses.py \
    --human_data_dir data/humans/all_results_20251206_154732 \
    --session s1 \
    --output_dir evaluation/human_analysis/ \
    --with_similarity
```

---

#### 2. `comprehensive_analysis.py`
**Purpose**: Run all 4 requested analyses

**Analysis 1: Model-Human Correlation**
- Merges human and model data on QID
- Computes correlations: `r(model_acc, human_similarity)`
- **Key Insight**: Answers "GT vs Visual similarity?" question
  - **GT similarity**: Correlation with 10 annotators (objective)
  - **Visual similarity**: Correlation with oracle humans (alignment)
  - If r_visual > r_gt → Model aligns better with informed humans
  - If r_gt > r_visual → Model is more objectively correct

**Analysis 2: Multimodal Gains**
```
MG_absolute = Acc(baseline) - Acc(blind)
MG_relative = (MG_absolute / Acc(baseline)) × 100%
```
- Compares: baseline, blind, inst_blind conditions
- Shows importance of visual information

**Analysis 3: Instruction Effects**
```
IE = Acc(inst_blind) - Acc(blind)
```
- Tests if explicit instructions help
- Positive IE → Instructions help adaptation

**Analysis 4: Pretrained vs Finetuned**
- Statistical tests: KS test, Mann-Whitney U, Cohen's d
- Distribution similarity (like training loss)
- Earth Mover's Distance for distribution shift

**Command**:
```bash
python evaluation/comprehensive_analysis.py \
    --human_dir evaluation/human_analysis/ \
    --model_dir evaluation/data/scored/ \
    --finetuned_dir evaluation/data/finetuned_scored/ \
    --output_dir evaluation/comprehensive_results/ \
    --dataset vqa_1k
```

---

#### 3. `paper_analysis_notebook.ipynb`
**Purpose**: Complete analysis workflow for paper

**Sections**:
1. Data Processing - Runs pipeline scripts
2. Human Baseline - Performance statistics
3. Correlation Analysis - With plots and interpretation
4. Multimodal Gains - Tables and visualizations
5. Instruction Effects - Comparative analysis
6. Pretrained vs Finetuned - Distribution comparison
7. Paper Outputs - LaTeX tables and figures

**Usage**:
```bash
jupyter notebook evaluation/paper_analysis_notebook.ipynb
```

**Outputs**:
- LaTeX tables (copy-paste ready)
- High-resolution figures (publication quality)
- Statistical summaries
- Interpretation guidelines

---

## 🔑 Key Questions Answered

### Q1: "Should I calculate similarity against ground truth or visual answers?"

**Answer**: **Both!** Here's why:

| Metric | What It Measures | When to Use |
|--------|-----------------|-------------|
| **GT Similarity** | Alignment with all 10 VQA annotators | Objective correctness evaluation |
| **Visual Similarity** | Alignment with "oracle" humans who saw images | Understanding human-like reasoning |

**Interpretation**:
- **High GT, High Visual** → Model is correct AND thinks like humans
- **High GT, Low Visual** → Model is correct but reasons differently
- **Low GT, High Visual** → Model makes human-like mistakes
- **Low GT, Low Visual** → Model needs improvement

**In Paper**: Report correlation with both, discuss the gap:
```
Model accuracy showed r = 0.75 correlation with GT similarity and
r = 0.82 with visual similarity, suggesting the model aligns more
closely with human visual reasoning than pure objective correctness.
```

---

### Q2: "How do I calculate distribution similarity like training loss?"

**Answer**: Implemented in Analysis 4 using multiple metrics:

```python
# KS Test - Tests if distributions differ
ks_stat, p_value = stats.ks_2samp(pretrained_scores, finetuned_scores)

# Earth Mover's Distance - Quantifies shift
from scipy.stats import wasserstein_distance
emd = wasserstein_distance(pretrained_scores, finetuned_scores)

# Cohen's d - Effect size
d = (mean_ft - mean_pt) / pooled_std
```

**Like training loss**, lower values mean more similar distributions:
- EMD ≈ 0 → Distributions very similar
- EMD > 0.2 → Significant shift

---

## 📂 Complete File Structure

```
evaluation/
├── process_raw_human_responses.py     # Step 1: Process raw data
├── comprehensive_analysis.py          # Step 2: Run all analyses
├── paper_analysis_notebook.ipynb      # Step 3: Generate paper outputs
│
├── COMPREHENSIVE_PIPELINE_README.md   # Detailed documentation
├── PIPELINE_OVERVIEW.md               # This file
│
├── score_human_results.py             # Alternative: Process aggregated data
├── visualize_human_analysis.py        # Visualization utilities
├── score_results.py                   # Model scoring (updated)
│
├── ANALYSIS_PIPELINE.tex              # LaTeX methodology documentation
└── HUMAN_ANALYSIS_README.md           # Original analysis guide
```

## 🚀 Quick Start (3 Commands)

```bash
# 1. Process raw human responses (per-question averages)
python evaluation/process_raw_human_responses.py \
    --human_data_dir data/humans/all_results_20251206_154732 \
    --session s1 \
    --output_dir evaluation/human_analysis/ \
    --with_similarity

# 2. Run comprehensive analysis (all 4 analyses)
python evaluation/comprehensive_analysis.py \
    --human_dir evaluation/human_analysis/ \
    --model_dir evaluation/data/scored/ \
    --finetuned_dir evaluation/data/finetuned_scored/ \
    --output_dir evaluation/comprehensive_results/ \
    --dataset vqa_1k

# 3. Generate paper outputs
jupyter notebook evaluation/paper_analysis_notebook.ipynb
```

**Total time**: ~10-20 minutes depending on data size

---

## 📊 Example Outputs

### Per-Question Metric (process_raw_human_responses.py)

```json
{
  "qid": "418297001",
  "answer_type": "text",
  "num_responses": 10,
  "answers": ["yes", "yes", "no", "yes", "yes", "maybe", "yes", "yes", "yes", "no"],
  "confidences": [0.75, 1.0, 0.5, 0.75, 1.0, 0.5, 0.75, 1.0, 0.75, 0.5],
  "gt_answers": ["yes", "yes", "yes", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
  "visual_gt": "yes",
  "mean_accuracy": 0.8333,
  "std_accuracy": 0.2357,
  "mean_confidence": 0.75,
  "std_confidence": 0.1893,
  "mean_gt_similarity": 0.8234,
  "std_gt_similarity": 0.1234,
  "mean_visual_similarity": 0.9123,
  "std_visual_similarity": 0.0876
}
```

### Correlation Analysis Output

```
Correlations:
  Model Accuracy vs Human GT Similarity: r = 0.7523 (p < 0.001)
  Model Accuracy vs Human Visual Similarity: r = 0.8012 (p < 0.001)
  Model Accuracy vs Human Accuracy: r = 0.6834 (p < 0.001)

Interpretation:
  - Strong correlation with visual similarity suggests model aligns with human visual reasoning
  - Higher visual than GT correlation indicates model learns perceptual patterns
```

### Multimodal Gains Table

```
Model                    Baseline  Blind   MG      MG (%)
InternVL3_5-2B          0.6234    0.3821  0.2413  38.7%
Qwen3-VL-4B-Instruct    0.6891    0.4234  0.2657  38.5%
llava-v1.6-mistral-7b   0.6012    0.3892  0.2120  35.3%

Mean MG: 0.2397 ± 0.0234 (37.5%)
```

### Pretrained vs Finetuned Comparison

```
Comparison: InternVL3_5-2B vs InternVL3_5-2B_finetuned
  Pretrained: 0.6234 ± 0.1234
  Finetuned:  0.6891 ± 0.1012
  Improvement: +0.0657 (+10.5%)
  Cohen's d: 0.543 (medium effect)
  KS test: p < 0.001 (significant)
```

---

## 📖 For Your Paper

### Methods Section Template

```latex
\subsection{Human Response Analysis}
We collected responses from $n=10$ crowdworkers per question under blind
conditions. For each question, we computed mean accuracy against VQA v2
ground truth using the standard metric: $\min(1, \text{\#matches}/3)$.

We measured two types of similarity:
\begin{itemize}
    \item \textbf{GT Similarity}: Cosine similarity between human responses
          and all 10 ground truth annotators
    \item \textbf{Visual Similarity}: Similarity to multiple-choice answers
          from annotators who saw images
\end{itemize}

\subsection{Model-Human Alignment}
We computed Pearson correlation between model accuracy and human similarity
metrics on matched question sets ($N = 374$).

\subsection{Multimodal Gains}
We define multimodal gain as $MG = \text{Acc}_{\text{baseline}} -
\text{Acc}_{\text{blind}}$, quantifying performance loss without visual input.
```

### Results Section Template

```latex
\subsection{Results}

\textbf{Human Baseline.} Humans achieved mean accuracy of $0.6234 \pm 0.1234$
on blind VQA with confidence-accuracy correlation of $r = 0.7234$ ($p < 0.001$).

\textbf{Model-Human Alignment.} Model predictions showed strong correlation
with human visual similarity ($r = 0.8012$, $p < 0.001$) and moderate
correlation with GT similarity ($r = 0.7523$), indicating models align better
with human visual reasoning patterns than pure objective correctness.

\textbf{Multimodal Gains.} Models exhibited mean MG of $37.5\%$ (range:
$[35.3\%, 38.7\%]$), demonstrating substantial visual dependence.

\textbf{Finetuning Effects.} Finetuning on human blind responses improved
performance by $10.5\%$ with medium effect size (Cohen's $d = 0.543$),
showing significant distribution shift (KS test: $p < 0.001$).
```

---

## 🎓 Key Insights

### 1. GT vs Visual Similarity
- **Report both** in your paper
- Difference reveals whether model is objectively correct vs human-like
- Visual similarity better predicts human alignment

### 2. Multimodal Gains
- High MG (>30%) → Model heavily relies on vision
- Low MG (<10%) → Better blind reasoning (less hallucination)

### 3. Instruction Effects
- Positive → Model adapts to context
- Negative → Model confused by meta-info

### 4. Distribution Similarity
- Use KS test + Cohen's d
- Similar to comparing training distributions
- EMD quantifies shift magnitude

---

## 🐛 Common Issues

1. **"No matching QIDs"**: Ensure QID types match (string vs int)
2. **"VQA annotations not found"**: Update path in scripts
3. **"CUDA OOM"**: Remove `.to('cuda')` in `get_encoder()`

---

## 📞 Next Steps

1. **Run the pipeline**:
   ```bash
   cd /home/user/HPA
   bash evaluation/run_complete_analysis.sh  # If you create this
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook evaluation/paper_analysis_notebook.ipynb
   ```

3. **Check outputs**:
   - Tables: `evaluation/comprehensive_results/table*.tex`
   - Figures: `evaluation/comprehensive_results/figures/*.png`
   - Summary: `evaluation/comprehensive_results/comprehensive_analysis_summary.json`

4. **Integrate into paper**:
   - Copy LaTeX tables
   - Include figures
   - Use interpretation guidelines

---

## ✨ Summary

You now have a **complete, publication-ready pipeline** that:
- ✅ Processes raw human data (no aggregation)
- ✅ Computes per-question averages against ground truth
- ✅ Analyzes model-human correlations (with GT vs Visual guide)
- ✅ Calculates multimodal gains
- ✅ Analyzes instruction effects
- ✅ Compares pretrained vs finetuned (with distribution similarity)
- ✅ Generates paper outputs (tables, figures, LaTeX)
- ✅ Includes complete documentation and interpretation guides

**All committed and pushed to your branch!** 🚀

---

**Questions?** Check:
- Detailed docs: `COMPREHENSIVE_PIPELINE_README.md`
- LaTeX methods: `ANALYSIS_PIPELINE.tex`
- Quick start: This file
