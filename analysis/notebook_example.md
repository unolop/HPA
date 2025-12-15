# Example Code for progress.ipynb

## Setup

```python
import sys
sys.path.append('/home/user/HPA/analysis')
from interrater_analysis import *
import pandas as pd
import numpy as np
```

## 1. Subject-to-Subject Agreement Matrix

### For VQA (semantic similarity)
```python
# Load human VQA data
human_vqa = pd.read_csv('/home/user/HPA/evaluation/scored/humans/human_vqa_per_question.csv')

# If you have a similarity function (e.g., using sentence transformers)
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def similarity_func(text1, text2):
    emb = encoder.encode([text1, text2])
    return encoder.similarity(emb, emb)[1, 0]

# Compute subject-to-subject matrices for all questions
vqa_matrices, vqa_avg_matrix = aggregate_subject_matrices(
    human_vqa,
    qid_col='qid',
    similarity_func=similarity_func,
    metric_type='similarity'
)

# Plot average agreement across all questions
plot_subject_matrix_heatmap(
    vqa_avg_matrix,
    title='VQA: Average Subject-to-Subject Similarity',
    output_path='/home/user/HPA/analysis/figures/vqa_subject_matrix.png'
)
```

### For MMStar (exact agreement)
```python
# Load human MC data
human_mc = pd.read_csv('/home/user/HPA/evaluation/scored/humans/human_mc_per_question.csv')

# Compute subject-to-subject agreement matrices
mc_matrices, mc_avg_matrix = aggregate_subject_matrices(
    human_mc,
    qid_col='qid',
    similarity_func=None,  # Use exact match
    metric_type='agreement'
)

# Plot average agreement
plot_subject_matrix_heatmap(
    mc_avg_matrix,
    title='MMStar: Average Subject-to-Subject Agreement',
    output_path='/home/user/HPA/analysis/figures/mc_subject_matrix.png'
)
```

### View specific question's matrix
```python
# Pick a question
qid = '418297001'
question_matrix = vqa_matrices[qid]

# Plot it
plot_subject_matrix_heatmap(
    question_matrix,
    title=f'Subject Agreement for Question {qid}',
    output_path=f'/home/user/HPA/analysis/figures/subject_matrix_{qid}.png'
)
```

## 2. VQA Analysis by Question Type and Answer Type

```python
# First, prepare aggregated model data
# Assuming you have model results per question already aggregated
# If not, use the plot_human_model_comparison_fixed.py script output

# Load aggregated model data (example structure)
model_vqa_agg = pd.DataFrame({
    'qid': [...],
    'human_accuracy': [...],
    'model_accuracy': [...],
})

# Run complete VQA analysis
vqa_results = analyze_vqa_by_types(
    human_data=human_vqa,
    model_data=model_vqa_agg,
    output_dir='/home/user/HPA/analysis/figures'
)

# View correlation results
print("By Question Type:")
print(vqa_results['correlation_by_question_type'])

print("\nBy Answer Type:")
print(vqa_results['correlation_by_answer_type'])
```

### Manual analysis for custom metrics
```python
# Load VQA metadata
vqa_metadata = load_vqa_metadata()

# Aggregate by question_type
agg_by_qtype = aggregate_by_metadata(
    model_vqa_agg,
    vqa_metadata,
    groupby_field='question_type',
    qid_col='qid',
    metric_cols=['human_accuracy', 'model_accuracy']
)
print(agg_by_qtype)

# Compute correlation by question_type
corr_by_qtype = correlation_by_metadata(
    model_vqa_agg,
    vqa_metadata,
    groupby_field='question_type',
    x_col='human_accuracy',
    y_col='model_accuracy',
    method='pearson'
)
print(corr_by_qtype)

# Plot scatter plots
fig, axes = plot_scatter_by_metadata(
    model_vqa_agg,
    vqa_metadata,
    groupby_field='question_type',
    x_col='human_accuracy',
    y_col='model_accuracy',
    title='Human vs Model by Question Type',
    output_path='/home/user/HPA/analysis/figures/vqa_by_qtype.png'
)

# Plot correlation bar chart
fig, ax = plot_correlation_heatmap(
    corr_by_qtype,
    groupby_field='question_type',
    title='Correlation by Question Type',
    output_path='/home/user/HPA/analysis/figures/vqa_corr_qtype.png'
)
```

## 3. MMStar Analysis by Category and L2 Category

```python
# Load aggregated model data for MMStar
model_mmstar_agg = pd.DataFrame({
    'qid': [...],
    'human_accuracy': [...],
    'model_accuracy': [...],
})

# Run complete MMStar analysis
mmstar_results = analyze_mmstar_by_categories(
    human_data=human_mc,
    model_data=model_mmstar_agg,
    output_dir='/home/user/HPA/analysis/figures'
)

# View correlation results
print("By Category:")
print(mmstar_results['correlation_by_category'])

print("\nBy L2 Category:")
print(mmstar_results['correlation_by_l2_category'])
```

### Manual analysis for custom metrics
```python
# Load MMStar metadata
mmstar_metadata = load_mmstar_metadata()

# Aggregate by category
agg_by_cat = aggregate_by_metadata(
    model_mmstar_agg,
    mmstar_metadata,
    groupby_field='category',
    qid_col='qid',
    metric_cols=['human_accuracy', 'model_accuracy']
)
print(agg_by_cat)

# Compute correlation by l2_category
corr_by_l2 = correlation_by_metadata(
    model_mmstar_agg,
    mmstar_metadata,
    groupby_field='l2_category',
    x_col='human_accuracy',
    y_col='model_accuracy',
    method='pearson'
)
print(corr_by_l2)

# Plot scatter plots by l2_category
fig, axes = plot_scatter_by_metadata(
    model_mmstar_agg,
    mmstar_metadata,
    groupby_field='l2_category',
    x_col='human_accuracy',
    y_col='model_accuracy',
    title='Human vs Model by L2 Category',
    figsize=(20, 12),
    output_path='/home/user/HPA/analysis/figures/mmstar_by_l2.png'
)
```

## 4. Computing Correlations Only (for custom analysis)

```python
# Simple correlation between two arrays
x = model_data['human_accuracy'].values
y = model_data['model_accuracy'].values

# Pearson correlation
r_pearson, p_pearson = compute_correlation(x, y, method='pearson')
print(f"Pearson r={r_pearson:.3f}, p={p_pearson:.4f}")

# Spearman correlation
r_spearman, p_spearman = compute_correlation(x, y, method='spearman')
print(f"Spearman ρ={r_spearman:.3f}, p={p_spearman:.4f}")
```

## 5. Loading Metadata for Custom Analysis

```python
# Load VQA metadata
vqa_meta = load_vqa_metadata()
print("Sample VQA metadata:", vqa_meta['189936002'])
# Output: {'question_type': 'what is the woman', 'answer_type': 'other', 'question': '...'}

# Load MMStar metadata
mmstar_meta = load_mmstar_metadata()
print("Sample MMStar metadata:", list(mmstar_meta.values())[0])
# Output: {'category': 'coarse perception', 'l2_category': 'image topic', 'question': '...'}

# Add metadata to your dataframe
df['question_type'] = df['qid'].map(lambda q: vqa_meta.get(str(q), {}).get('question_type', 'unknown'))
df['answer_type'] = df['qid'].map(lambda q: vqa_meta.get(str(q), {}).get('answer_type', 'unknown'))
```

## 6. All Available Functions

### Data Loading
- `load_vqa_metadata()` - Load VQA question_type and answer_type
- `load_mmstar_metadata()` - Load MMStar category and l2_category

### Subject-to-Subject Analysis
- `compute_subject_to_subject_matrix()` - Compute pairwise agreement matrix for one question
- `aggregate_subject_matrices()` - Compute matrices for all questions and average

### Aggregation
- `aggregate_by_metadata()` - Group by metadata field and compute mean/std
- `correlation_by_metadata()` - Compute correlation for each metadata group

### Correlation
- `compute_correlation()` - Pearson or Spearman correlation with p-value

### Plotting
- `plot_scatter_by_metadata()` - Grid of scatter plots, one per metadata group
- `plot_correlation_heatmap()` - Bar chart of correlations with significance
- `plot_subject_matrix_heatmap()` - Heatmap of subject-to-subject matrix

### Complete Analysis
- `analyze_vqa_by_types()` - Full VQA analysis pipeline
- `analyze_mmstar_by_categories()` - Full MMStar analysis pipeline
