#!/usr/bin/env python3
"""
Plot human-model comparison figures in paper-ready style.

Creates:
1. Scatter plots: human accuracy/similarity/alignment (x) vs model scores (y)
2. Histogram distributions: average human and model accuracies by dataset

Based on /home/work/yuna/HPA/evaluation/analysis/progress.ipynb
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from collections import defaultdict
from pathlib import Path
from scipy import stats

# Set paper-ready style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Color palette
COLORS = {
    'mmstar': '#2E86AB',
    'vqa_1k': '#A23B72',
    'vqa_5k': '#F18F01',
    'spubench': '#C73E1D',
}


# =============================================================================
# Data Loading
# =============================================================================

def load_human_mc_data(path='/home/user/HPA/evaluation/scored/humans/human_mc_per_question.csv'):
    """Load human MC (multiple choice) data."""
    print(f"Loading human MC data from {path}")
    df = pd.read_csv(path)

    # Parse string representations of lists/dicts
    for col in ['extracted_choices', 'choice_distribution']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Create dict keyed by qid
    human_data = {}
    for _, row in df.iterrows():
        qid = str(row['qid'])
        human_data[qid] = row.to_dict()

    print(f"  Loaded {len(human_data)} MC questions")
    return human_data


def load_human_vqa_data(path='/home/user/HPA/evaluation/scored/humans/human_vqa_per_question.csv'):
    """Load human VQA (open-ended) data."""
    print(f"Loading human VQA data from {path}")
    df = pd.read_csv(path)

    # Parse string representations
    for col in ['answers', 'normalized_answers', 'answer_distribution', 'gt_answers']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x)

    # Create dict keyed by qid
    human_data = {}
    for _, row in df.iterrows():
        qid = str(row['qid'])
        human_data[qid] = row.to_dict()

    print(f"  Loaded {len(human_data)} VQA questions")
    return human_data


def load_model_results(file_path):
    """Load model results from JSONL file."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def identify_dataset(filename):
    """Identify dataset from filename."""
    if 'mmstar' in filename:
        return 'mmstar'
    elif 'spubench' in filename:
        return 'spubench'
    elif 'vqa_5k' in filename:
        return 'vqa_5k'
    elif 'vqa_1k' in filename:
        return 'vqa_1k'
    return 'unknown'


def load_all_model_results(results_dir='/home/user/HPA/evaluation/scored/pretrained'):
    """Load all model results organized by dataset."""
    print(f"\nLoading model results from {results_dir}")

    by_dataset = defaultdict(list)
    files = glob(f"{results_dir}/*.jsonl")

    # Filter to only non-blind conditions (standard evaluation)
    files = [f for f in files if '_blind' not in f]

    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        dataset = identify_dataset(filename)

        if dataset == 'unknown':
            continue

        model_name = filename.replace(f'_{dataset}', '').replace('.jsonl', '')
        results = load_model_results(file_path)

        by_dataset[dataset].append({
            'model': model_name,
            'file': file_path,
            'results': results
        })

    for dataset, models in by_dataset.items():
        print(f"  {dataset}: {len(models)} models")

    return by_dataset


# =============================================================================
# Data Mapping by QID
# =============================================================================

def map_human_model_by_qid(human_data, model_results, dataset_type='mc'):
    """
    Map human and model results by question ID.

    Args:
        human_data: Dict of {qid: human_metrics}
        model_results: List of model result dicts
        dataset_type: 'mc' or 'vqa'

    Returns:
        List of matched records with both human and model data
    """
    matched = []

    for model_item in model_results:
        # Get QID from model result
        # Try different QID field names
        qid = str(model_item.get('qid', model_item.get('question_id', model_item.get('pid', model_item.get('index', '')))))

        # Skip if no human data for this QID
        if qid not in human_data:
            continue

        human_metrics = human_data[qid]

        # Combine human and model data
        record = {
            'qid': qid,
            # Human metrics
            'human_accuracy': human_metrics.get('mean_accuracy', 0),
            'human_agreement': human_metrics.get('percent_agreement', human_metrics.get('agreement', 0)),
            'human_num_raters': human_metrics.get('num_raters', human_metrics.get('num_responses', 0)),
            # Model metrics
            'model_correct': model_item.get('correct', False),
            'model_score': int(model_item.get('correct', False)),  # 1 or 0
            'model_output': model_item.get('output', ''),
        }

        # Add VQA-specific metrics if available
        if 'mean_gt_similarity' in human_metrics:
            record['human_gt_similarity'] = human_metrics['mean_gt_similarity']
        if 'mean_visual_similarity' in human_metrics:
            record['human_visual_similarity'] = human_metrics['mean_visual_similarity']

        matched.append(record)

    return matched


# =============================================================================
# Distribution Calculations
# =============================================================================

def calculate_distribution_stats(matched_data):
    """
    Calculate distribution statistics for human and model performance.

    Returns:
        Dict with mean, std, median, etc.
    """
    df = pd.DataFrame(matched_data)

    stats = {
        'human': {
            'mean': df['human_accuracy'].mean(),
            'std': df['human_accuracy'].std(),
            'median': df['human_accuracy'].median(),
            'min': df['human_accuracy'].min(),
            'max': df['human_accuracy'].max(),
        },
        'model': {
            'mean': df['model_score'].mean(),
            'std': df['model_score'].std(),
            'median': df['model_score'].median(),
            'min': df['model_score'].min(),
            'max': df['model_score'].max(),
        },
        'n_questions': len(df),
    }

    return stats


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_scatter_correlation(matched_data, dataset_name, output_path,
                             x_metric='human_accuracy',
                             x_label='Human Accuracy',
                             title_suffix=''):
    """
    Create scatter plot showing correlation between human and model performance.

    Args:
        matched_data: List of dicts with human and model metrics
        dataset_name: Name of dataset for title
        output_path: Path to save figure
        x_metric: Which human metric to use on x-axis
        x_label: Label for x-axis
        title_suffix: Additional text for title
    """
    df = pd.DataFrame(matched_data)

    if len(df) == 0:
        print(f"  ⚠️  No data for {dataset_name}")
        return

    # Calculate correlation
    x = df[x_metric]
    y = df['model_score']

    # Filter out NaN values
    valid = ~(x.isna() | y.isna())
    x_clean = x[valid]
    y_clean = y[valid]

    if len(x_clean) < 3:
        print(f"  ⚠️  Insufficient data for correlation ({len(x_clean)} points)")
        return

    r, p = stats.pearsonr(x_clean, y_clean)
    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Scatter plot with jitter for binary model scores
    y_jittered = y + np.random.normal(0, 0.02, len(y))

    ax.scatter(x, y_jittered, alpha=0.4, s=30,
              color=COLORS.get(dataset_name, '#666666'))

    # Add regression line
    z = np.polyfit(x_clean, y_clean, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p_fit(x_line), 'r--', alpha=0.7, linewidth=1.5,
            label=f'Linear fit')

    # Formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel('Model Score (1=Correct, 0=Wrong)')
    ax.set_title(f'{dataset_name.upper()}{title_suffix}', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add correlation text
    textstr = f'Pearson r = {r:.3f} (p={p:.4f})\n' \
              f'Spearman ρ = {spearman_r:.3f} (p={spearman_p:.4f})\n' \
              f'n = {len(x_clean)}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
           fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved scatter plot: {output_path}")
    print(f"    Pearson r={r:.3f}, p={p:.4f}")


def plot_distribution_histogram(all_stats_by_dataset, output_path):
    """
    Create histogram showing distribution of average accuracies by dataset.

    Args:
        all_stats_by_dataset: Dict of {dataset: {models: [{stats}]}}
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    datasets = sorted(all_stats_by_dataset.keys())

    for idx, dataset in enumerate(datasets):
        if idx >= 4:
            break

        ax = axes[idx]
        models_data = all_stats_by_dataset[dataset]

        # Collect human and model means across all models
        human_means = [m['stats']['human']['mean'] for m in models_data if 'stats' in m]
        model_means = [m['stats']['model']['mean'] for m in models_data if 'stats' in m]

        if not human_means or not model_means:
            continue

        # Create histogram
        bins = np.linspace(0, 1, 21)

        ax.hist(human_means, bins=bins, alpha=0.6, label='Human Avg Accuracy',
               color='#2E86AB', edgecolor='black', linewidth=0.5)
        ax.hist(model_means, bins=bins, alpha=0.6, label='Model Avg Accuracy',
               color='#F18F01', edgecolor='black', linewidth=0.5)

        # Add vertical lines for means
        ax.axvline(np.mean(human_means), color='#2E86AB', linestyle='--',
                  linewidth=2, label=f'Human μ={np.mean(human_means):.3f}')
        ax.axvline(np.mean(model_means), color='#F18F01', linestyle='--',
                  linewidth=2, label=f'Model μ={np.mean(model_means):.3f}')

        # Formatting
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Count')
        ax.set_title(f'{dataset.upper()}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved distribution histogram: {output_path}")


def plot_combined_scatter_grid(all_matched_by_dataset, output_path):
    """Create a grid of scatter plots for all datasets."""
    datasets = sorted(all_matched_by_dataset.keys())
    n_datasets = len(datasets)

    # Determine grid size
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        matched_data = all_matched_by_dataset[dataset]

        if not matched_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(dataset.upper(), fontweight='bold')
            continue

        df = pd.DataFrame(matched_data)
        x = df['human_accuracy']
        y = df['model_score']

        # Filter out NaN
        valid = ~(x.isna() | y.isna())
        x_clean = x[valid]
        y_clean = y[valid]

        if len(x_clean) < 3:
            ax.text(0.5, 0.5, f'Insufficient data\n({len(x_clean)} points)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dataset.upper(), fontweight='bold')
            continue

        # Calculate correlation
        r, p = stats.pearsonr(x_clean, y_clean)

        # Scatter with jitter
        y_jittered = y + np.random.normal(0, 0.02, len(y))
        ax.scatter(x, y_jittered, alpha=0.4, s=20,
                  color=COLORS.get(dataset, '#666666'))

        # Regression line
        z = np.polyfit(x_clean, y_clean, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p_fit(x_line), 'r--', alpha=0.7, linewidth=1.5)

        # Formatting
        ax.set_xlabel('Human Accuracy')
        ax.set_ylabel('Model Score')
        ax.set_title(f'{dataset.upper()}\nr={r:.3f}, p={p:.4f}', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.1)

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved combined scatter grid: {output_path}")


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("="*80)
    print("HUMAN-MODEL COMPARISON: PAPER-READY FIGURES")
    print("="*80)

    # Output directory
    output_dir = '/home/user/HPA/analysis/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Load human data
    print("\n📊 Loading human data...")
    human_mc = load_human_mc_data()
    human_vqa = load_human_vqa_data()

    # Load model results
    model_results_by_dataset = load_all_model_results()

    # Match and analyze by dataset
    print("\n🔗 Matching human and model data by QID...")

    all_matched_by_dataset = {}
    all_stats_by_dataset = defaultdict(list)

    for dataset, models_list in model_results_by_dataset.items():
        print(f"\n  Dataset: {dataset.upper()}")

        # Determine which human data to use
        if dataset == 'mmstar':
            human_data = human_mc
            dataset_type = 'mc'
        elif dataset in ['vqa_1k', 'vqa_5k']:
            human_data = human_vqa
            dataset_type = 'vqa'
        elif dataset == 'spubench':
            # Spubench doesn't have human annotations
            print(f"    ⚠️  No human data for spubench, skipping")
            continue
        else:
            print(f"    ⚠️  Unknown dataset type, skipping")
            continue

        print(f"    Human data: {len(human_data)} questions available")

        # Aggregate matched data across all models for this dataset
        all_matched = []

        for model_info in models_list:
            model_name = model_info['model']
            model_results = model_info['results']

            # Match by QID
            matched = map_human_model_by_qid(human_data, model_results, dataset_type)

            if matched:
                print(f"    {model_name}: {len(matched)} matched questions")
                all_matched.extend(matched)

                # Calculate stats for this model
                stats_dict = calculate_distribution_stats(matched)
                all_stats_by_dataset[dataset].append({
                    'model': model_name,
                    'stats': stats_dict,
                    'matched_count': len(matched)
                })
            else:
                print(f"    {model_name}: 0 matched questions (check QID format)")

        if all_matched:
            all_matched_by_dataset[dataset] = all_matched
            print(f"    Total matched: {len(all_matched)}")

    # Create figures
    print("\n📈 Creating figures...")

    # 1. Individual scatter plots for each dataset
    for dataset, matched_data in all_matched_by_dataset.items():
        output_path = os.path.join(output_dir, f'scatter_{dataset}_human_model.png')
        plot_scatter_correlation(
            matched_data,
            dataset,
            output_path,
            x_metric='human_accuracy',
            x_label='Human Accuracy'
        )

        # If VQA, also plot similarity metrics
        if dataset in ['vqa_1k', 'vqa_5k']:
            # Check if similarity data exists
            df = pd.DataFrame(matched_data)
            if 'human_gt_similarity' in df.columns and df['human_gt_similarity'].notna().any():
                output_path = os.path.join(output_dir, f'scatter_{dataset}_similarity.png')
                plot_scatter_correlation(
                    matched_data,
                    dataset,
                    output_path,
                    x_metric='human_gt_similarity',
                    x_label='Human GT Similarity',
                    title_suffix=' (GT Similarity)'
                )

    # 2. Combined scatter grid
    if all_matched_by_dataset:
        output_path = os.path.join(output_dir, 'scatter_all_datasets.png')
        plot_combined_scatter_grid(all_matched_by_dataset, output_path)

    # 3. Distribution histogram
    if all_stats_by_dataset:
        output_path = os.path.join(output_dir, 'distribution_histogram.png')
        plot_distribution_histogram(all_stats_by_dataset, output_path)

    # Save summary statistics
    summary = {}
    for dataset in all_matched_by_dataset.keys():
        df = pd.DataFrame(all_matched_by_dataset[dataset])
        summary[dataset] = {
            'n_questions': len(df),
            'human_mean_accuracy': df['human_accuracy'].mean(),
            'model_mean_accuracy': df['model_score'].mean(),
            'correlation_pearson': stats.pearsonr(
                df['human_accuracy'].dropna(),
                df['model_score'].dropna()
            )[0] if len(df) > 2 else None,
        }

    summary_path = os.path.join(output_dir, 'human_model_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Saved summary statistics: {summary_path}")

    print("\n" + "="*80)
    print("✅ All figures generated successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
