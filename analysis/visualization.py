#!/usr/bin/env python3
"""
Visualization Script for Human-Model Comparison Analysis

Generates publication-ready plots:
1. Correlation scatter plots (human vs model accuracy per question)
2. Category-wise comparison bar charts
3. Confidence calibration curves
4. Heatmaps of agreement across models/categories
5. Embedding similarity distributions

Usage:
    python visualize_analysis.py \
        --results_dir ./analysis_results \
        --output_dir ./figures \
        --benchmark vqav2
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Publication-ready settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'human': '#2E86AB',
    'model': '#A23B72',
    'correct': '#28A745',
    'incorrect': '#DC3545',
    'neutral': '#6C757D',
}


def load_analysis_results(results_dir: str) -> Dict[str, Any]:
    """Load analysis results from directory."""
    results = {}
    
    analysis_path = os.path.join(results_dir, 'analysis_results.json')
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            results['analysis'] = json.load(f)
    
    per_question_path = os.path.join(results_dir, 'per_question_data.json')
    if os.path.exists(per_question_path):
        with open(per_question_path, 'r') as f:
            results['per_question'] = json.load(f)
    
    return results


# =============================================================================
# Plot 1: Human vs Model Accuracy Scatter
# =============================================================================

def plot_accuracy_correlation(
    per_question_data: List[Dict],
    model_name: str,
    output_path: str,
    title: str = None,
):
    """
    Scatter plot of human accuracy vs model correctness per question.
    
    Shows:
    - X-axis: Human accuracy (0-1)
    - Y-axis: Model correct (0/1) with jitter
    - Color by category
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract data
    human_acc = [d['human_accuracy'] for d in per_question_data]
    model_correct = [1 if d.get(f'{model_name}_correct', False) else 0 for d in per_question_data]
    categories = [d.get('category', 'Unknown') for d in per_question_data]
    
    # Add jitter to model_correct for visibility
    model_jittered = np.array(model_correct) + np.random.normal(0, 0.03, len(model_correct))
    
    # Get unique categories and assign colors
    unique_cats = list(set(categories))
    color_map = plt.cm.get_cmap('tab10')
    cat_colors = {cat: color_map(i / len(unique_cats)) for i, cat in enumerate(unique_cats)}
    
    colors = [cat_colors[cat] for cat in categories]
    
    # Scatter plot
    scatter = ax.scatter(human_acc, model_jittered, c=colors, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(human_acc, model_correct, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), '--', color='red', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    # Add reference line (perfect correlation)
    ax.plot([0, 1], [0, 1], ':', color='gray', alpha=0.5, label='Perfect correlation')
    
    # Labels
    ax.set_xlabel('Human Accuracy (per question)')
    ax.set_ylabel(f'{model_name} Correct (with jitter)')
    ax.set_title(title or f'Human vs {model_name} Performance Correlation')
    
    # Legend for categories
    legend_patches = [mpatches.Patch(color=cat_colors[cat], label=cat) for cat in unique_cats[:10]]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.15)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Plot 2: Category-wise Comparison
# =============================================================================

def plot_category_comparison(
    category_results: Dict[str, Dict],
    model_names: List[str],
    output_path: str,
    title: str = None,
    max_categories: int = 15,
):
    """
    Grouped bar chart comparing human and model accuracy by category.
    """
    # Sort by number of questions
    sorted_cats = sorted(
        category_results.items(),
        key=lambda x: -x[1]['num_questions']
    )[:max_categories]
    
    categories = [cat for cat, _ in sorted_cats]
    human_accs = [data['human_mean_accuracy'] for _, data in sorted_cats]
    
    # Setup figure
    n_groups = len(categories)
    n_bars = 1 + len(model_names)
    bar_width = 0.8 / n_bars
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_groups)
    
    # Human bars
    bars_human = ax.bar(x - bar_width * len(model_names) / 2, human_accs, bar_width,
                        label='Human', color=COLORS['human'], alpha=0.8)
    
    # Model bars
    for i, model_name in enumerate(model_names):
        model_accs = [
            sorted_cats[j][1]['models'].get(model_name, {}).get('accuracy', 0)
            for j in range(n_groups)
        ]
        offset = bar_width * (i + 1 - len(model_names) / 2)
        ax.bar(x + offset, model_accs, bar_width, label=model_name, alpha=0.8)
    
    # Labels
    ax.set_xlabel('Category')
    ax.set_ylabel('Accuracy')
    ax.set_title(title or 'Human vs Model Accuracy by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    
    # Add count labels
    for i, (cat, data) in enumerate(sorted_cats):
        ax.annotate(f'n={data["num_questions"]}', 
                    xy=(i, -0.05), ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Plot 3: Confidence Calibration Curve
# =============================================================================

def plot_calibration_curve(
    calibration_results: Dict,
    output_path: str,
    title: str = None,
):
    """
    Plot calibration curve: confidence level vs actual accuracy.
    
    Perfect calibration = diagonal line
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    
    by_conf = calibration_results.get('by_confidence_level', {})
    
    conf_levels = sorted(by_conf.keys())
    accuracies = [by_conf[c]['accuracy'] for c in conf_levels]
    counts = [by_conf[c]['num_responses'] for c in conf_levels]
    
    # Normalize confidence to 0-1
    conf_normalized = [c / 5.0 for c in conf_levels]
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration', linewidth=2)
    
    # Actual calibration
    ax.plot(conf_normalized, accuracies, 'o-', color=COLORS['human'], 
            markersize=10, linewidth=2, label='Human calibration')
    
    # Shade the gap (calibration error)
    ax.fill_between(conf_normalized, conf_normalized, accuracies, 
                    alpha=0.3, color=COLORS['human'])
    
    # Add count labels
    for i, (conf, acc, count) in enumerate(zip(conf_normalized, accuracies, counts)):
        ax.annotate(f'n={count}', xy=(conf, acc), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Confidence Level (normalized)')
    ax.set_ylabel('Actual Accuracy')
    ax.set_title(title or f'Human Confidence Calibration\n(Error = {calibration_results.get("calibration_error", 0):.3f})')
    ax.legend(loc='upper left')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Plot 4: Correlation Heatmap
# =============================================================================

def plot_correlation_heatmap(
    category_results: Dict[str, Dict],
    model_names: List[str],
    output_path: str,
    title: str = None,
):
    """
    Heatmap of human-model correlation by category.
    """
    # Build correlation matrix
    categories = list(category_results.keys())
    
    # Filter to categories with enough data
    categories = [c for c in categories if category_results[c]['num_questions'] >= 5]
    
    data = np.zeros((len(categories), len(model_names)))
    
    for i, cat in enumerate(categories):
        for j, model in enumerate(model_names):
            corr = category_results[cat]['models'].get(model, {}).get('correlation_with_human', 0)
            data[i, j] = corr
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(categories) * 0.4)))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(model_names)
    ax.set_yticklabels(categories)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add correlation values
    for i in range(len(categories)):
        for j in range(len(model_names)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title(title or 'Human-Model Correlation by Category')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Spearman Correlation', rotation=-90, va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Plot 5: Embedding Similarity Distribution
# =============================================================================

def plot_embedding_similarity(
    embedding_results: Dict,
    model_names: List[str],
    output_path: str,
    title: str = None,
):
    """
    Distribution plots for embedding similarity scores.
    """
    detailed = embedding_results.get('detailed', {})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Similarity to Ground Truth
    ax1 = axes[0]
    
    data_gt = []
    labels_gt = []
    
    human_gt = detailed.get('human_gt_similarity', [])
    if human_gt:
        data_gt.append(human_gt)
        labels_gt.append('Human')
    
    for model_name in model_names:
        model_gt = detailed.get('model_gt_similarity', {}).get(model_name, [])
        if model_gt:
            data_gt.append(model_gt)
            labels_gt.append(model_name)
    
    if data_gt:
        parts = ax1.violinplot(data_gt, positions=range(len(data_gt)), showmeans=True)
        ax1.set_xticks(range(len(labels_gt)))
        ax1.set_xticklabels(labels_gt, rotation=45, ha='right')
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_title('Similarity to Ground Truth')
        ax1.set_ylim(0, 1)
    
    # Plot 2: Human-Model Similarity
    ax2 = axes[1]
    
    data_hm = []
    labels_hm = []
    
    for model_name in model_names:
        hm_sim = detailed.get('human_model_similarity', {}).get(model_name, [])
        if hm_sim:
            data_hm.append(hm_sim)
            labels_hm.append(model_name)
    
    if data_hm:
        parts = ax2.violinplot(data_hm, positions=range(len(data_hm)), showmeans=True)
        ax2.set_xticks(range(len(labels_hm)))
        ax2.set_xticklabels(labels_hm, rotation=45, ha='right')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Human-Model Answer Similarity')
        ax2.set_ylim(0, 1)
    
    plt.suptitle(title or 'Embedding-based Answer Similarity')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Plot 6: Agreement Matrix (Human vs Multiple Models)
# =============================================================================

def plot_agreement_matrix(
    per_question_data: List[Dict],
    model_names: List[str],
    output_path: str,
    title: str = None,
):
    """
    Show agreement between human majority and each model.
    
    Matrix shows: Both correct, Both wrong, Human only, Model only
    """
    fig, axes = plt.subplots(1, len(model_names), figsize=(4 * len(model_names), 4))
    
    if len(model_names) == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        
        # Compute agreement categories
        both_correct = 0
        both_wrong = 0
        human_only = 0
        model_only = 0
        
        for d in per_question_data:
            human_correct = d['human_accuracy'] > 0.5  # Majority correct
            model_correct = d.get(f'{model_name}_correct', False)
            
            if human_correct and model_correct:
                both_correct += 1
            elif not human_correct and not model_correct:
                both_wrong += 1
            elif human_correct and not model_correct:
                human_only += 1
            else:
                model_only += 1
        
        # Create confusion matrix
        matrix = np.array([
            [both_correct, model_only],
            [human_only, both_wrong]
        ])
        
        total = matrix.sum()
        matrix_pct = matrix / total * 100
        
        # Plot
        im = ax.imshow(matrix_pct, cmap='Blues', vmin=0, vmax=60)
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Model ✓', 'Model ✗'])
        ax.set_yticklabels(['Human ✓', 'Human ✗'])
        
        # Add values
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{matrix_pct[i, j]:.1f}%\n({matrix[i, j]})',
                              ha='center', va='center', fontsize=10)
        
        ax.set_title(f'{model_name}\n(n={total})')
    
    plt.suptitle(title or 'Human-Model Agreement Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Plot 7: Difficulty Analysis
# =============================================================================

def plot_difficulty_analysis(
    per_question_data: List[Dict],
    model_names: List[str],
    output_path: str,
):
    """
    Analyze which questions are hardest for humans vs models.
    
    Quadrant plot:
    - Hard for both (high correlation signal)
    - Easy for both (high correlation signal)
    - Hard for human only (model has different priors)
    - Hard for model only (human has better priors)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    model_name = model_names[0] if model_names else None
    if not model_name:
        return
    
    human_acc = np.array([d['human_accuracy'] for d in per_question_data])
    model_acc = np.array([1.0 if d.get(f'{model_name}_correct', False) else 0.0 
                          for d in per_question_data])
    
    # Color by quadrant
    colors = []
    for h, m in zip(human_acc, model_acc):
        if h > 0.5 and m > 0.5:
            colors.append('#28A745')  # Easy for both (green)
        elif h <= 0.5 and m <= 0.5:
            colors.append('#DC3545')  # Hard for both (red)
        elif h > 0.5 and m <= 0.5:
            colors.append('#FFC107')  # Hard for model only (yellow)
        else:
            colors.append('#17A2B8')  # Hard for human only (cyan)
    
    ax.scatter(human_acc, model_acc, c=colors, alpha=0.6, s=60, edgecolors='white')
    
    # Quadrant lines
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant labels
    ax.text(0.75, 0.85, 'Easy for Both', ha='center', fontsize=10, color='#28A745', weight='bold')
    ax.text(0.25, 0.15, 'Hard for Both', ha='center', fontsize=10, color='#DC3545', weight='bold')
    ax.text(0.75, 0.15, 'Hard for Model', ha='center', fontsize=10, color='#FFC107', weight='bold')
    ax.text(0.25, 0.85, 'Hard for Human', ha='center', fontsize=10, color='#17A2B8', weight='bold')
    
    # Count per quadrant
    n_easy_both = sum(1 for h, m in zip(human_acc, model_acc) if h > 0.5 and m > 0.5)
    n_hard_both = sum(1 for h, m in zip(human_acc, model_acc) if h <= 0.5 and m <= 0.5)
    n_hard_model = sum(1 for h, m in zip(human_acc, model_acc) if h > 0.5 and m <= 0.5)
    n_hard_human = sum(1 for h, m in zip(human_acc, model_acc) if h <= 0.5 and m > 0.5)
    
    ax.text(0.75, 0.75, f'n={n_easy_both}', ha='center', fontsize=9)
    ax.text(0.25, 0.25, f'n={n_hard_both}', ha='center', fontsize=9)
    ax.text(0.75, 0.25, f'n={n_hard_model}', ha='center', fontsize=9)
    ax.text(0.25, 0.75, f'n={n_hard_human}', ha='center', fontsize=9)
    
    ax.set_xlabel('Human Accuracy')
    ax.set_ylabel(f'{model_name} Accuracy')
    ax.set_title('Question Difficulty Analysis')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def generate_all_plots(
    results_dir: str,
    output_dir: str,
    benchmark: str = "vqav2",
):
    """Generate all visualization plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_analysis_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        return
    
    analysis = results.get('analysis', {})
    per_question = results.get('per_question', [])
    
    model_names = analysis.get('summary', {}).get('model_names', [])
    
    print(f"Generating plots for {benchmark}...")
    print(f"Models: {model_names}")
    
    # Generate each plot
    if per_question and model_names:
        plot_accuracy_correlation(
            per_question,
            model_names[0],
            os.path.join(output_dir, f'{benchmark}_correlation_scatter.png'),
            title=f'{benchmark.upper()}: Human vs Model Accuracy'
        )
    
    if analysis.get('by_category') and model_names:
        plot_category_comparison(
            analysis['by_category'],
            model_names,
            os.path.join(output_dir, f'{benchmark}_category_comparison.png'),
            title=f'{benchmark.upper()}: Accuracy by Category'
        )
    
    if analysis.get('calibration'):
        plot_calibration_curve(
            analysis['calibration'],
            os.path.join(output_dir, f'{benchmark}_calibration.png'),
            title=f'{benchmark.upper()}: Human Confidence Calibration'
        )
    
    if analysis.get('by_category') and model_names:
        plot_correlation_heatmap(
            analysis['by_category'],
            model_names,
            os.path.join(output_dir, f'{benchmark}_correlation_heatmap.png'),
            title=f'{benchmark.upper()}: Correlation by Category'
        )
    
    if analysis.get('embedding_similarity') and model_names:
        plot_embedding_similarity(
            analysis['embedding_similarity'],
            model_names,
            os.path.join(output_dir, f'{benchmark}_embedding_similarity.png'),
            title=f'{benchmark.upper()}: Embedding Similarity'
        )
    
    if per_question and model_names:
        plot_agreement_matrix(
            per_question,
            model_names,
            os.path.join(output_dir, f'{benchmark}_agreement_matrix.png'),
            title=f'{benchmark.upper()}: Agreement Analysis'
        )
    
    if per_question and model_names:
        plot_difficulty_analysis(
            per_question,
            model_names,
            os.path.join(output_dir, f'{benchmark}_difficulty_quadrant.png'),
        )
    
    print(f"\n✅ All plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization plots")
    
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with analysis results")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for figures")
    parser.add_argument("--benchmark", type=str, default="vqav2",
                        help="Benchmark name for titles")
    
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir, args.output_dir, args.benchmark)


if __name__ == "__main__":
    main()