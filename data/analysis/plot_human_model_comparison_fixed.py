#!/usr/bin/env python3
"""
Fixed version: Plot human-model comparison with proper aggregation by question.

The key fix: Instead of plotting individual model responses (binary 0/1),
we aggregate by question to get average model accuracy per question,
then correlate with average human accuracy per question.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Paper-ready styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'mmstar': '#E74C3C',
    'vqa_1k': '#3498DB',
    'vqa_5k': '#2ECC71',
}


def load_human_mc_data(base_path='/home/user/HPA/evaluation/scored/humans'):
    """Load human MC response data grouped by question."""
    mc_file = Path(base_path) / 'human_mc_per_question.jsonl'

    human_by_qid = {}
    with open(mc_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = str(item.get('qid', ''))
            if qid:
                human_by_qid[qid] = item

    return human_by_qid


def load_human_vqa_data(base_path='/home/user/HPA/evaluation/scored/humans'):
    """Load human VQA response data grouped by question."""
    vqa_file = Path(base_path) / 'human_vqa_per_question.jsonl'

    human_by_qid = {}
    with open(vqa_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = str(item.get('qid', ''))
            if qid:
                human_by_qid[qid] = item

    return human_by_qid


def load_model_results(scored_dir='/home/user/HPA/evaluation/scored'):
    """Load all model results grouped by dataset."""
    scored_path = Path(scored_dir)

    results_by_dataset = defaultdict(list)

    for file in scored_path.glob('*.jsonl'):
        filename = file.stem

        # Parse filename to extract dataset name
        if 'mmstar' in filename:
            dataset = 'mmstar'
        elif 'vqa_1k' in filename:
            dataset = 'vqa_1k'
        elif 'vqa_5k' in filename:
            dataset = 'vqa_5k'
        elif 'spubench' in filename:
            continue  # Skip spubench (no human data)
        else:
            continue

        # Load results
        with open(file, 'r') as f:
            results = [json.loads(line) for line in f]

        results_by_dataset[dataset].append({
            'model': filename,
            'results': results
        })

    return results_by_dataset


def aggregate_by_question(human_data, model_results_list):
    """
    Aggregate model results by question to get per-question accuracy.

    Returns:
        DataFrame with columns: qid, human_accuracy, model_accuracy, n_models, n_humans
    """
    # First, collect all model responses by QID
    model_by_qid = defaultdict(list)

    for model_info in model_results_list:
        for item in model_info['results']:
            qid = str(item.get('qid', item.get('question_id', item.get('pid', item.get('index', '')))))
            if qid:
                model_by_qid[qid].append(int(item.get('correct', False)))

    # Now create aggregated records
    records = []
    for qid, human_metrics in human_data.items():
        if qid not in model_by_qid:
            continue  # Skip questions without model data

        model_scores = model_by_qid[qid]

        record = {
            'qid': qid,
            'human_accuracy': human_metrics.get('mean_accuracy', 0),
            'human_agreement': human_metrics.get('percent_agreement', human_metrics.get('agreement', 0)),
            'model_accuracy': np.mean(model_scores),  # Average across models
            'n_models': len(model_scores),
            'n_humans': human_metrics.get('num_raters', human_metrics.get('num_responses', 0)),
        }

        # Add VQA-specific metrics if available
        if 'mean_gt_similarity' in human_metrics:
            record['human_gt_similarity'] = human_metrics['mean_gt_similarity']
        if 'mean_visual_similarity' in human_metrics:
            record['human_visual_similarity'] = human_metrics['mean_visual_similarity']

        records.append(record)

    return pd.DataFrame(records)


def plot_scatter_correlation(df, dataset_name, output_path,
                             x_col='human_accuracy',
                             x_label='Human Accuracy',
                             title_suffix=''):
    """
    Create scatter plot showing correlation between human and model performance.

    Args:
        df: DataFrame with aggregated data
        dataset_name: Name of dataset for title
        output_path: Path to save figure
        x_col: Column name for x-axis
        x_label: Label for x-axis
        title_suffix: Additional text for title
    """
    if len(df) == 0:
        print(f"  ⚠️  No data for {dataset_name}")
        return

    # Calculate correlation
    x = df[x_col]
    y = df['model_accuracy']

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

    # Scatter plot
    ax.scatter(x, y, alpha=0.5, s=40,
              color=COLORS.get(dataset_name, '#666666'),
              edgecolors='white', linewidths=0.5)

    # Add regression line
    z = np.polyfit(x_clean, y_clean, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p_fit(x_line), '--', color='#2C3E50', alpha=0.8, linewidth=1.5)

    # Add diagonal reference line (perfect correlation)
    ax.plot([0, 1], [0, 1], ':', color='gray', alpha=0.5, linewidth=1, label='y=x')

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel('Model Accuracy')
    title = f'{dataset_name.upper()}: Human vs Model Performance'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title)

    # Add correlation statistics
    stats_text = f'Pearson r = {r:.3f} (p = {p:.4f})\nSpearman ρ = {spearman_r:.3f} (p = {spearman_p:.4f})\nn = {len(x_clean)} questions'
    ax.text(0.05, 0.95, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
           fontsize=8)

    # Set limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

    # Grid
    ax.grid(True, alpha=0.2, linestyle='--')

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    print(f"    Pearson r={r:.3f} (p={p:.4f}), n={len(x_clean)}")


def plot_distribution_histogram(stats_by_dataset, output_path):
    """
    Create histogram comparing human and model accuracy distributions.

    Args:
        stats_by_dataset: Dict of {dataset: DataFrame with aggregated data}
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, len(stats_by_dataset), figsize=(15, 4))

    if len(stats_by_dataset) == 1:
        axes = [axes]

    for idx, (dataset, df) in enumerate(stats_by_dataset.items()):
        ax = axes[idx]

        # Plot histograms
        ax.hist(df['human_accuracy'], bins=20, alpha=0.6, label='Human',
               color='#3498DB', edgecolor='white', linewidth=0.5)
        ax.hist(df['model_accuracy'], bins=20, alpha=0.6, label='Model',
               color='#E74C3C', edgecolor='white', linewidth=0.5)

        # Add mean lines
        human_mean = df['human_accuracy'].mean()
        model_mean = df['model_accuracy'].mean()
        ax.axvline(human_mean, color='#3498DB', linestyle='--', linewidth=2,
                  label=f'Human μ={human_mean:.3f}')
        ax.axvline(model_mean, color='#E74C3C', linestyle='--', linewidth=2,
                  label=f'Model μ={model_mean:.3f}')

        # Labels
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{dataset.upper()}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def main():
    print("="*70)
    print("FIXED: Human-Model Comparison Analysis (Aggregated by Question)")
    print("="*70)

    # Create output directory
    output_dir = Path('/home/user/HPA/analysis/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load human data
    print("\n📊 Loading human data...")
    human_mc = load_human_mc_data()
    human_vqa = load_human_vqa_data()
    print(f"  MC questions: {len(human_mc)}")
    print(f"  VQA questions: {len(human_vqa)}")

    # Load model results
    print("\n📊 Loading model results...")
    model_results_by_dataset = load_model_results()
    for dataset, models_list in model_results_by_dataset.items():
        print(f"  {dataset}: {len(models_list)} models")

    # Aggregate and analyze each dataset
    print("\n📈 Aggregating data by question...")
    aggregated_by_dataset = {}

    for dataset, models_list in model_results_by_dataset.items():
        print(f"\n  {dataset}:")

        # Get appropriate human data
        if dataset == 'mmstar':
            human_data = human_mc
        elif dataset in ['vqa_1k', 'vqa_5k']:
            human_data = human_vqa
        else:
            print(f"    ⚠️  Unknown dataset, skipping")
            continue

        # Aggregate by question
        df = aggregate_by_question(human_data, models_list)
        aggregated_by_dataset[dataset] = df

        print(f"    {len(df)} questions with both human and model data")
        print(f"    Human accuracy: {df['human_accuracy'].mean():.3f} ± {df['human_accuracy'].std():.3f}")
        print(f"    Model accuracy: {df['model_accuracy'].mean():.3f} ± {df['model_accuracy'].std():.3f}")

    # Create scatter plots
    print("\n📈 Creating scatter plots...")

    for dataset, df in aggregated_by_dataset.items():
        output_path = output_dir / f'scatter_{dataset}_human_model_fixed.png'
        plot_scatter_correlation(df, dataset, output_path)

    # Create combined scatter grid
    if len(aggregated_by_dataset) > 0:
        print("\n📈 Creating combined scatter grid...")
        fig, axes = plt.subplots(1, len(aggregated_by_dataset), figsize=(15, 5))

        if len(aggregated_by_dataset) == 1:
            axes = [axes]

        for idx, (dataset, df) in enumerate(aggregated_by_dataset.items()):
            ax = axes[idx]

            x = df['human_accuracy']
            y = df['model_accuracy']

            valid = ~(x.isna() | y.isna())
            x_clean = x[valid]
            y_clean = y[valid]

            if len(x_clean) >= 3:
                r, p = stats.pearsonr(x_clean, y_clean)

                ax.scatter(x, y, alpha=0.5, s=30,
                          color=COLORS.get(dataset, '#666666'),
                          edgecolors='white', linewidths=0.5)

                # Regression line
                z = np.polyfit(x_clean, y_clean, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p_fit(x_line), '--', color='#2C3E50', alpha=0.8, linewidth=1.5)

                # Diagonal reference
                ax.plot([0, 1], [0, 1], ':', color='gray', alpha=0.5, linewidth=1)

                ax.set_xlabel('Human Accuracy')
                ax.set_ylabel('Model Accuracy')
                ax.set_title(f'{dataset.upper()}\nr={r:.3f}, p={p:.4f}')
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.2, linestyle='--')

        plt.tight_layout()
        output_path = output_dir / 'scatter_all_datasets_fixed.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")

    # Create distribution histogram
    if len(aggregated_by_dataset) > 0:
        print("\n📈 Creating distribution histogram...")
        output_path = output_dir / 'distribution_histogram_fixed.png'
        plot_distribution_histogram(aggregated_by_dataset, output_path)

    print("\n✓ Analysis complete!")
    print(f"✓ Figures saved to {output_dir}")


if __name__ == '__main__':
    main()
