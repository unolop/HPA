#!/usr/bin/env python3
"""
Visualize subject-level inter-rater metrics.

Creates heatmaps, barcharts, and histograms from subject-level similarity/agreement data.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict


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


def load_subject_metrics(base_dir: str, dataset_type: str):
    """Load subject-level metrics from CSV."""
    csv_path = Path(base_dir) / f'{dataset_type}_subject_level_metrics.csv'
    if not csv_path.exists():
        print(f"⚠️  File not found: {csv_path}")
        return None

    df = pd.DataFrame(pd.read_csv(csv_path))
    print(f"✓ Loaded {len(df)} subject-question pairs from {csv_path}")
    return df


def create_participant_heatmap(df: pd.DataFrame, dataset_type: str, output_path: str):
    """
    Create heatmap showing average similarity/agreement per participant.
    """
    metric_col = 'avg_similarity_to_others' if dataset_type == 'vqa' else 'avg_agreement_with_others'
    metric_label = 'Average Similarity' if dataset_type == 'vqa' else 'Average Agreement'

    # Calculate per-participant average
    participant_metrics = df.groupby('participant_id')[metric_col].agg(['mean', 'std', 'count'])
    participant_metrics = participant_metrics.sort_values('mean', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(participant_metrics) * 0.3)))

    # Create horizontal barplot with error bars
    y_pos = np.arange(len(participant_metrics))
    ax.barh(y_pos, participant_metrics['mean'],
            xerr=participant_metrics['std'],
            color='#3498DB', alpha=0.7, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(participant_metrics.index, fontsize=8)
    ax.set_xlabel(metric_label)
    ax.set_ylabel('Participant ID')
    ax.set_title(f'{dataset_type.upper()}: {metric_label} by Participant')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2, linestyle='--', axis='x')

    # Add count annotations
    for i, (idx, row) in enumerate(participant_metrics.iterrows()):
        ax.text(row['mean'] + 0.02, i, f"n={int(row['count'])}",
               va='center', fontsize=7, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved participant heatmap: {output_path}")


def create_distribution_histogram(df: pd.DataFrame, dataset_type: str, output_path: str):
    """
    Create histogram showing distribution of similarity/agreement scores.
    """
    metric_col = 'avg_similarity_to_others' if dataset_type == 'vqa' else 'avg_agreement_with_others'
    metric_label = 'Average Similarity to Others' if dataset_type == 'vqa' else 'Average Agreement with Others'

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of all subject scores
    axes[0].hist(df[metric_col], bins=30, alpha=0.7, color='#3498DB', edgecolor='white', linewidth=0.5)
    axes[0].axvline(df[metric_col].mean(), color='#E74C3C', linestyle='--', linewidth=2,
                   label=f'Mean = {df[metric_col].mean():.3f}')
    axes[0].axvline(df[metric_col].median(), color='#2ECC71', linestyle='--', linewidth=2,
                   label=f'Median = {df[metric_col].median():.3f}')
    axes[0].set_xlabel(metric_label)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{dataset_type.upper()}: Distribution of Subject Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2, linestyle='--', axis='y')
    axes[0].set_xlim(0, 1)

    # Per-question mean distribution
    question_means = df.groupby('qid')[metric_col].mean()
    axes[1].hist(question_means, bins=30, alpha=0.7, color='#E74C3C', edgecolor='white', linewidth=0.5)
    axes[1].axvline(question_means.mean(), color='#3498DB', linestyle='--', linewidth=2,
                   label=f'Mean = {question_means.mean():.3f}')
    axes[1].axvline(question_means.median(), color='#2ECC71', linestyle='--', linewidth=2,
                   label=f'Median = {question_means.median():.3f}')
    axes[1].set_xlabel(f'Question-level Mean {metric_label}')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{dataset_type.upper()}: Distribution of Question Difficulty')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2, linestyle='--', axis='y')
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved distribution histogram: {output_path}")


def create_participant_vs_question_heatmap(df: pd.DataFrame, dataset_type: str, output_path: str,
                                          max_questions: int = 50):
    """
    Create heatmap showing similarity/agreement matrix: participants × questions.
    """
    metric_col = 'avg_similarity_to_others' if dataset_type == 'vqa' else 'avg_agreement_with_others'
    metric_label = 'Similarity' if dataset_type == 'vqa' else 'Agreement'

    # Pivot to create matrix
    # Limit to first N questions for readability
    unique_qids = df['qid'].unique()[:max_questions]
    df_subset = df[df['qid'].isin(unique_qids)]

    pivot_df = df_subset.pivot_table(
        index='participant_id',
        columns='qid',
        values=metric_col,
        aggfunc='mean'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(unique_qids) * 0.3), max(6, len(pivot_df) * 0.4)))

    # Heatmap
    sns.heatmap(pivot_df, annot=False, fmt='.2f', cmap='RdYlGn',
               vmin=0, vmax=1, cbar_kws={'label': metric_label},
               linewidths=0.5, linecolor='white', ax=ax)

    ax.set_xlabel('Question ID')
    ax.set_ylabel('Participant ID')
    ax.set_title(f'{dataset_type.upper()}: {metric_label} Heatmap (Participants × Questions)')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved participant×question heatmap: {output_path} (showing first {max_questions} questions)")


def create_accuracy_vs_agreement_scatter(df: pd.DataFrame, dataset_type: str, output_path: str):
    """
    Create scatter plot showing relationship between accuracy and agreement/similarity.
    """
    if 'accuracy' not in df.columns and 'correct' not in df.columns:
        print(f"⚠️  No accuracy data available for {dataset_type}")
        return

    metric_col = 'avg_similarity_to_others' if dataset_type == 'vqa' else 'avg_agreement_with_others'
    metric_label = 'Average Similarity to Others' if dataset_type == 'vqa' else 'Average Agreement with Others'

    accuracy_col = 'accuracy' if 'accuracy' in df.columns else 'correct'

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(df[metric_col], df[accuracy_col], alpha=0.3, s=20, color='#3498DB')

    # Add trend line
    z = np.polyfit(df[metric_col].dropna(), df[accuracy_col].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[metric_col].min(), df[metric_col].max(), 100)
    ax.plot(x_line, p(x_line), '--', color='#E74C3C', linewidth=2, alpha=0.8)

    # Correlation
    corr = np.corrcoef(df[metric_col].dropna(), df[accuracy_col].dropna())[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(metric_label)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{dataset_type.upper()}: Accuracy vs {metric_label}')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved accuracy vs agreement scatter: {output_path}")


def main():
    print("="*70)
    print("Subject-Level Metrics Visualization")
    print("="*70)

    # Paths
    base_dir = Path('/home/user/HPA/evaluation/scored/humans')
    output_dir = Path('/home/user/HPA/analysis/figures/subject_metrics')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process VQA
    print("\n📊 VQA Dataset:")
    vqa_df = load_subject_metrics(base_dir, 'vqa')
    if vqa_df is not None:
        create_participant_heatmap(vqa_df, 'vqa', output_dir / 'vqa_participant_similarity.png')
        create_distribution_histogram(vqa_df, 'vqa', output_dir / 'vqa_distribution.png')
        create_participant_vs_question_heatmap(vqa_df, 'vqa', output_dir / 'vqa_heatmap.png')
        create_accuracy_vs_agreement_scatter(vqa_df, 'vqa', output_dir / 'vqa_accuracy_vs_similarity.png')

    # Process MC
    print("\n📊 MC Dataset:")
    mc_df = load_subject_metrics(base_dir, 'mc')
    if mc_df is not None:
        create_participant_heatmap(mc_df, 'mc', output_dir / 'mc_participant_agreement.png')
        create_distribution_histogram(mc_df, 'mc', output_dir / 'mc_distribution.png')
        create_participant_vs_question_heatmap(mc_df, 'mc', output_dir / 'mc_heatmap.png')
        create_accuracy_vs_agreement_scatter(mc_df, 'mc', output_dir / 'mc_accuracy_vs_agreement.png')

    print("\n✓ All visualizations complete!")
    print(f"✓ Saved to: {output_dir}")


if __name__ == '__main__':
    main()
