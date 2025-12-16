#!/usr/bin/env python3
"""
Inter-rater analysis functions for notebook use.

Provides reusable functions for:
1. Computing subject-to-subject agreement/similarity matrices
2. Aggregating by question/answer types and categories
3. Plotting scatter plots with correlations
4. Analyzing human-model agreement by metadata
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from itertools import combinations


# =============================================================================
# Subject-to-Subject Agreement/Similarity Matrix
# =============================================================================

def compute_subject_to_subject_matrix(responses: List[str],
                                     participant_ids: List[str],
                                     similarity_func=None,
                                     metric_type='agreement') -> pd.DataFrame:
    """
    Compute pairwise similarity/agreement matrix between all subjects.

    Args:
        responses: List of subject responses for a single question
        participant_ids: List of participant IDs corresponding to responses
        similarity_func: Function to compute similarity (for VQA), if None uses exact match
        metric_type: 'agreement' for MC (exact match) or 'similarity' for VQA (semantic)

    Returns:
        DataFrame with subjects as both rows and columns, values are similarity/agreement scores
    """
    n = len(responses)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0  # Perfect agreement with self
            elif similarity_func and metric_type == 'similarity':
                # VQA: semantic similarity
                matrix[i, j] = similarity_func(responses[i], responses[j])
            else:
                # MC: exact match
                matrix[i, j] = 1.0 if responses[i] == responses[j] else 0.0

    # Create DataFrame
    df = pd.DataFrame(matrix, index=participant_ids, columns=participant_ids)
    return df


def aggregate_subject_matrices(human_data: pd.DataFrame,
                               qid_col='question_id',
                               similarity_func=None,
                               metric_type='agreement') -> Tuple[Dict, pd.DataFrame]:
    """
    Compute subject-to-subject matrices for all questions and aggregate.

    Returns:
        matrices: Dict of {qid: subject_matrix}
        avg_matrix: Average matrix across all questions (subjects must be consistent)
    """
    from ast import literal_eval

    matrices = {}
    all_participant_ids = None

    for _, row in human_data.iterrows():
        qid = str(row[qid_col])

        # Parse responses (assuming stored as string representation of list)
        if isinstance(row['answers'], str):
            responses = literal_eval(row['answers'])
        else:
            responses = row['answers']

        # Get participant IDs if available, otherwise use indices
        if 'participant_ids' in row and row['participant_ids']:
            if isinstance(row['participant_ids'], str):
                participant_ids = literal_eval(row['participant_ids'])
            else:
                participant_ids = row['participant_ids']
        else:
            participant_ids = [f'subject_{i}' for i in range(len(responses))]

        # Compute matrix for this question
        matrix = compute_subject_to_subject_matrix(
            responses, participant_ids, similarity_func, metric_type
        )
        matrices[qid] = matrix

        # Track participant IDs
        if all_participant_ids is None:
            all_participant_ids = participant_ids

    # Compute average matrix across all questions
    if matrices:
        avg_matrix = sum(matrices.values()) / len(matrices)
    else:
        avg_matrix = None

    return matrices, avg_matrix


# =============================================================================
# Data Aggregation by Metadata
# =============================================================================

def load_vqa_metadata(vqa_json_path='/home/work/yuna/HPA/dataset/vqav2_1k_val.json') -> Dict:
    """Load VQA metadata with question_type and answer_type."""
    with open(vqa_json_path, 'r') as f:
        vqa_data = json.load(f)

    # Create mapping from question_id to metadata
    metadata = {}
    for item in vqa_data:
        qid = str(item['question_id'])
        metadata[qid] = {
            'question_type': item.get('question_type', 'unknown'),
            'answer_type': item.get('answer_type', 'unknown'),
            'question': item.get('question', ''),
        }

    return metadata


def load_mmstar_metadata(annotation_json_path='/home/work/yuna/HPA/dataset/annotation.json') -> Dict:
    """Load MMStar metadata with category and l2_category."""
    with open(annotation_json_path, 'r') as f:
        mmstar_data = json.load(f)

    # Create mapping from qid to metadata
    metadata = {}
    for item in mmstar_data:
        qid = str(item.get('index', item.get('pid', '')))
        metadata[qid] = {
            'category': item.get('category', 'unknown'),
            'l2_category': item.get('l2_category', 'unknown'),
            'question': item.get('question', ''),
        }

    return metadata


def aggregate_by_metadata(data: pd.DataFrame,
                         metadata: Dict,
                         groupby_field: str,
                         qid_col='qid',
                         metric_cols=['human_accuracy', 'model_accuracy']) -> pd.DataFrame:
    """
    Aggregate human and model performance by metadata field.

    Args:
        data: DataFrame with qid, human metrics, model metrics
        metadata: Dict mapping qid to metadata
        groupby_field: Field to group by (e.g., 'question_type', 'category')
        qid_col: Column name for question ID
        metric_cols: List of columns to aggregate

    Returns:
        Aggregated DataFrame grouped by metadata field
    """
    # Add metadata to data
    data_copy = data.copy()
    data_copy['metadata_field'] = data_copy[qid_col].map(
        lambda qid: metadata.get(str(qid), {}).get(groupby_field, 'unknown')
    )

    # Group and aggregate
    grouped = data_copy.groupby('metadata_field')[metric_cols].agg(['mean', 'std', 'count'])
    grouped = grouped.sort_values((metric_cols[0], 'mean'), ascending=False)

    return grouped


# =============================================================================
# Correlation Analysis
# =============================================================================

def compute_correlation(x: np.ndarray, y: np.ndarray,
                       method='pearson') -> Tuple[float, float]:
    """
    Compute correlation and p-value.

    Args:
        x, y: Arrays to correlate
        method: 'pearson' or 'spearman'

    Returns:
        (correlation, p_value)
    """
    # Remove NaN values
    valid = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid]
    y_clean = y[valid]

    if len(x_clean) < 3:
        return np.nan, np.nan

    if method == 'pearson':
        r, p = stats.pearsonr(x_clean, y_clean)
    elif method == 'spearman':
        r, p = stats.spearmanr(x_clean, y_clean)
    else:
        raise ValueError(f"Unknown method: {method}")

    return r, p


def correlation_by_metadata(data: pd.DataFrame,
                           metadata: Dict,
                           groupby_field: str,
                           x_col='human_accuracy',
                           y_col='model_accuracy',
                           qid_col='qid',
                           method='pearson') -> pd.DataFrame:
    """
    Compute correlation between x and y for each metadata group.

    Returns:
        DataFrame with columns: metadata_field, correlation, p_value, n_samples
    """
    # Add metadata
    data_copy = data.copy()
    data_copy['metadata_field'] = data_copy[qid_col].map(
        lambda qid: metadata.get(str(qid), {}).get(groupby_field, 'unknown')
    )

    # Compute correlation per group
    results = []
    for group, group_data in data_copy.groupby('metadata_field'):
        x = group_data[x_col].values
        y = group_data[y_col].values

        r, p = compute_correlation(x, y, method)

        results.append({
            groupby_field: group,
            'correlation': r,
            'p_value': p,
            'n_samples': len(group_data),
            f'mean_{x_col}': np.mean(x),
            f'mean_{y_col}': np.mean(y),
        })

    return pd.DataFrame(results).sort_values('correlation', ascending=False)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_scatter_by_metadata(data: pd.DataFrame,
                            metadata: Dict,
                            groupby_field: str,
                            x_col='human_accuracy',
                            y_col='model_accuracy',
                            qid_col='qid',
                            figsize=(15, 10),
                            title=None,
                            output_path=None):
    """
    Create scatter plots for each metadata group.

    Args:
        data: DataFrame with qid, x_col, y_col
        metadata: Dict mapping qid to metadata
        groupby_field: Field to group by
        x_col, y_col: Column names for x and y axes
        figsize: Figure size
        title: Overall title
        output_path: Path to save figure
    """
    # Add metadata
    data_copy = data.copy()
    data_copy['metadata_field'] = data_copy[qid_col].map(
        lambda qid: metadata.get(str(qid), {}).get(groupby_field, 'unknown')
    )

    # Get unique groups
    groups = sorted(data_copy['metadata_field'].unique())
    n_groups = len(groups)

    # Calculate grid dimensions
    n_cols = min(4, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    # Plot each group
    for idx, group in enumerate(groups):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Filter data
        group_data = data_copy[data_copy['metadata_field'] == group]
        x = group_data[x_col].values
        y = group_data[y_col].values

        # Scatter plot
        ax.scatter(x, y, alpha=0.5, s=30, edgecolors='white', linewidths=0.5)

        # Regression line
        if len(x) >= 2:
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 2:
                z = np.polyfit(x[valid], y[valid], 1)
                p = np.poly1d(z)
                x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
                ax.plot(x_line, p(x_line), '--', color='red', linewidth=2, alpha=0.8)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], ':', color='gray', alpha=0.5, linewidth=1)

        # Compute correlation
        r, p = compute_correlation(x, y, 'pearson')

        # Labels
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=9)
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=9)
        ax.set_title(f'{group}\nr={r:.3f}, p={p:.4f}, n={len(group_data)}', fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle='--')

    # Hide unused subplots
    for idx in range(n_groups, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig, axes


def plot_correlation_heatmap(correlation_df: pd.DataFrame,
                            groupby_field: str,
                            figsize=(10, 6),
                            title=None,
                            output_path=None):
    """
    Create bar chart showing correlation by metadata group.

    Args:
        correlation_df: DataFrame from correlation_by_metadata()
        groupby_field: Name of groupby field
        figsize: Figure size
        title: Plot title
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by correlation
    df = correlation_df.sort_values('correlation', ascending=True)

    # Color bars by significance
    colors = ['#E74C3C' if p < 0.05 else '#95A5A6' for p in df['p_value']]

    # Horizontal bar chart
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['correlation'], color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)

    # Add vertical line at r=0
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df[groupby_field], fontsize=9)
    ax.set_xlabel('Correlation (r)', fontsize=11)
    ax.set_ylabel(groupby_field.replace('_', ' ').title(), fontsize=11)

    if title:
        ax.set_title(title, fontsize=12)

    # Add n annotations
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row['correlation'] + 0.02 if row['correlation'] > 0 else row['correlation'] - 0.02,
               i,
               f"n={int(row['n_samples'])}",
               va='center', fontsize=8, alpha=0.7,
               ha='left' if row['correlation'] > 0 else 'right')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', alpha=0.7, label='p < 0.05 (significant)'),
        Patch(facecolor='#95A5A6', alpha=0.7, label='p ≥ 0.05 (not significant)')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)

    ax.grid(True, alpha=0.2, linestyle='--', axis='x')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig, ax


def plot_subject_matrix_heatmap(matrix: pd.DataFrame,
                                title='Subject-to-Subject Agreement',
                                figsize=(10, 8),
                                output_path=None):
    """
    Plot heatmap of subject-to-subject similarity/agreement matrix.

    Args:
        matrix: DataFrame with subjects as rows and columns
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
               vmin=0, vmax=1, square=True,
               linewidths=0.5, linecolor='white',
               cbar_kws={'label': 'Agreement/Similarity'},
               ax=ax)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Subject ID', fontsize=11)
    ax.set_ylabel('Subject ID', fontsize=11)
    ax.set_xticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.index, rotation=45, ha='right')
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, rotation=0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig, ax


# =============================================================================
# Example Usage Functions
# =============================================================================

def analyze_vqa_by_types(human_data: pd.DataFrame,
                        model_data: pd.DataFrame,
                        output_dir: str = '/home/work/yuna/HPA/analysis/figures'):
    """
    Complete analysis of VQA data by question_type and answer_type.

    Args:
        human_data: DataFrame with human results per question
        model_data: DataFrame with model results aggregated per question
        output_dir: Directory to save plots

    Returns:
        Dict with correlation results and aggregated data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Analyze by question_type
    print("\n📊 Analyzing by question_type...")
    corr_qtype = correlation_by_metadata(
        model_data, vqa_metadata, 'question_type',
        x_col='human_accuracy', y_col='model_accuracy'
    )
    results['correlation_by_question_type'] = corr_qtype
    print(corr_qtype)

    # Plot scatter by question_type
    plot_scatter_by_metadata(
        model_data, vqa_metadata, 'question_type',
        x_col='human_accuracy', y_col='model_accuracy',
        title='VQA: Human vs Model Accuracy by Question Type',
        output_path=output_dir / 'vqa_scatter_by_question_type.png'
    )

    # Plot correlation heatmap
    plot_correlation_heatmap(
        corr_qtype, 'question_type',
        title='VQA: Correlation by Question Type',
        output_path=output_dir / 'vqa_correlation_by_question_type.png'
    )

    # Analyze by answer_type
    print("\n📊 Analyzing by answer_type...")
    corr_atype = correlation_by_metadata(
        model_data, vqa_metadata, 'answer_type',
        x_col='human_accuracy', y_col='model_accuracy'
    )
    results['correlation_by_answer_type'] = corr_atype
    print(corr_atype)

    # Plot scatter by answer_type
    plot_scatter_by_metadata(
        model_data, vqa_metadata, 'answer_type',
        x_col='human_accuracy', y_col='model_accuracy',
        title='VQA: Human vs Model Accuracy by Answer Type',
        output_path=output_dir / 'vqa_scatter_by_answer_type.png'
    )

    # Plot correlation heatmap
    plot_correlation_heatmap(
        corr_atype, 'answer_type',
        title='VQA: Correlation by Answer Type',
        output_path=output_dir / 'vqa_correlation_by_answer_type.png'
    )

    return results


def analyze_mmstar_by_categories(human_data: pd.DataFrame,
                                 model_data: pd.DataFrame,
                                 output_dir: str = '/home/work/yuna/HPA/analysis/figures'):
    """
    Complete analysis of MMStar data by category and l2_category.

    Args:
        human_data: DataFrame with human results per question
        model_data: DataFrame with model results aggregated per question
        output_dir: Directory to save plots

    Returns:
        Dict with correlation results and aggregated data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    mmstar_metadata = load_mmstar_metadata()

    results = {}

    # Analyze by category
    print("\n📊 Analyzing by category...")
    corr_cat = correlation_by_metadata(
        model_data, mmstar_metadata, 'category',
        x_col='human_accuracy', y_col='model_accuracy'
    )
    results['correlation_by_category'] = corr_cat
    print(corr_cat)

    # Plot scatter by category
    plot_scatter_by_metadata(
        model_data, mmstar_metadata, 'category',
        x_col='human_accuracy', y_col='model_accuracy',
        title='MMStar: Human vs Model Accuracy by Category',
        output_path=output_dir / 'mmstar_scatter_by_category.png'
    )

    # Plot correlation heatmap
    plot_correlation_heatmap(
        corr_cat, 'category',
        title='MMStar: Correlation by Category',
        output_path=output_dir / 'mmstar_correlation_by_category.png'
    )

    # Analyze by l2_category
    print("\n📊 Analyzing by l2_category...")
    corr_l2 = correlation_by_metadata(
        model_data, mmstar_metadata, 'l2_category',
        x_col='human_accuracy', y_col='model_accuracy'
    )
    results['correlation_by_l2_category'] = corr_l2
    print(corr_l2)

    # Plot scatter by l2_category
    plot_scatter_by_metadata(
        model_data, mmstar_metadata, 'l2_category',
        x_col='human_accuracy', y_col='model_accuracy',
        title='MMStar: Human vs Model Accuracy by L2 Category',
        output_path=output_dir / 'mmstar_scatter_by_l2_category.png',
        figsize=(20, 12)
    )

    # Plot correlation heatmap
    plot_correlation_heatmap(
        corr_l2, 'l2_category',
        title='MMStar: Correlation by L2 Category',
        output_path=output_dir / 'mmstar_correlation_by_l2_category.png',
        figsize=(12, 8)
    )

    return results


if __name__ == '__main__':
    # Example: Load data and run analysis
    print("This module provides reusable functions for notebook analysis.")
    print("\nExample usage in notebook:")
    print("""
import sys
sys.path.append('/home/user/HPA/analysis')
from interrater_analysis import *

# Load your data
human_vqa = pd.read_csv('/home/user/HPA/evaluation/scored/humans/human_vqa_per_question.csv')
model_vqa = pd.read_csv('/home/user/HPA/analysis/model_results_aggregated.csv')

# Run VQA analysis
vqa_results = analyze_vqa_by_types(human_vqa, model_vqa)

# Compute subject-to-subject matrix
matrices, avg_matrix = aggregate_subject_matrices(human_vqa, metric_type='similarity')
plot_subject_matrix_heatmap(avg_matrix, title='Average VQA Subject Agreement')
""")
