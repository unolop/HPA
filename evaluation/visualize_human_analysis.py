#!/usr/bin/env python3
"""
visualize_human_analysis.py - Create visualizations for human response analysis

Generates plots for:
- Confidence distributions (human vs model)
- Accuracy distributions (human vs model)
- Answer similarity distributions (human vs model)
- Question type distributions
- Category distributions for MC

Usage:
    python visualize_human_analysis.py --human_dir evaluation/human_scored/ \
                                        --model_dir evaluation/data/scored/ \
                                        --output_dir evaluation/figures/
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


# =============================================================================
# Data Loading
# =============================================================================

def load_human_results(human_dir: str, answer_type: str = 'vqa') -> Dict:
    """Load human scored results and statistics."""
    if answer_type == 'vqa':
        results_path = os.path.join(human_dir, 'human_vqa_scored.jsonl')
        stats_path = os.path.join(human_dir, 'human_vqa_stats.json')
    else:
        results_path = os.path.join(human_dir, 'human_mc_scored.jsonl')
        stats_path = os.path.join(human_dir, 'human_mc_stats.json')

    # Load results
    results = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Load stats
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    return {'results': results, 'statistics': stats}


def load_model_results(model_dir: str, model_name: str, dataset: str) -> Dict:
    """Load model scored results."""
    # Find matching file
    filename = f"{model_name}_{dataset}.jsonl"
    filepath = os.path.join(model_dir, filename)

    if not os.path.exists(filepath):
        print(f"⚠️  Model file not found: {filepath}")
        return {'results': []}

    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    return {'results': results}


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_confidence_distribution(human_data: Dict, output_path: str):
    """Plot confidence distribution for human responses."""
    stats = human_data['statistics']
    conf_dist = stats['confidence_dist']

    # Sort by confidence value
    confidences = sorted(conf_dist.keys())
    counts = [conf_dist[c] for c in confidences]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(confidences)), counts, color='steelblue', alpha=0.7)

    ax.set_xlabel('Confidence Level', fontsize=12)
    ax.set_ylabel('Number of Responses', fontsize=12)
    ax.set_title('Human Response Confidence Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(confidences)))
    ax.set_xticklabels([f'{c:.1f}' for c in confidences], rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_accuracy_comparison(human_data: Dict, model_data: List[Dict], output_path: str):
    """Plot accuracy comparison between human and models."""
    # Extract accuracies
    human_acc = [r['accuracy'] for r in human_data['results'] if 'accuracy' in r]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Distribution histogram
    ax = axes[0]
    ax.hist(human_acc, bins=20, alpha=0.6, label='Human', color='steelblue', density=True)

    for model in model_data:
        model_acc = [r.get('correct', 0) for r in model['results']]
        if model_acc:
            ax.hist(model_acc, bins=20, alpha=0.4, label=model.get('name', 'Model'), density=True)

    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Box plot
    ax = axes[1]
    data_to_plot = [human_acc]
    labels = ['Human']

    for model in model_data:
        model_acc = [r.get('correct', 0) for r in model['results']]
        if model_acc:
            data_to_plot.append(model_acc)
            labels.append(model.get('name', 'Model'))

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue'] + ['coral'] * len(model_data)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_similarity_comparison(human_data: Dict, model_data: List[Dict], output_path: str):
    """Plot answer similarity comparison between human and models."""
    human_sim = [r['answer_similarity'] for r in human_data['results'] if 'answer_similarity' in r]

    if not human_sim:
        print("   ⚠️  No similarity data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Distribution histogram
    ax = axes[0]
    ax.hist(human_sim, bins=20, alpha=0.6, label='Human', color='steelblue', density=True)

    for model in model_data:
        model_sim = [r.get('answer_similarity', 0) for r in model['results'] if 'answer_similarity' in r]
        if model_sim:
            ax.hist(model_sim, bins=20, alpha=0.4, label=model.get('name', 'Model'), density=True)

    ax.set_xlabel('Answer Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Answer Similarity Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: CDF plot
    ax = axes[1]

    # Human CDF
    human_sorted = np.sort(human_sim)
    human_cdf = np.arange(1, len(human_sorted) + 1) / len(human_sorted)
    ax.plot(human_sorted, human_cdf, label='Human', linewidth=2, color='steelblue')

    # Model CDFs
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_data)))
    for model, color in zip(model_data, colors):
        model_sim = [r.get('answer_similarity', 0) for r in model['results'] if 'answer_similarity' in r]
        if model_sim:
            model_sorted = np.sort(model_sim)
            model_cdf = np.arange(1, len(model_sorted) + 1) / len(model_sorted)
            ax.plot(model_sorted, model_cdf, label=model.get('name', 'Model'), linewidth=2, color=color)

    ax.set_xlabel('Answer Similarity', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_question_type_distribution(human_data: Dict, output_path: str):
    """Plot question type distribution for VQA."""
    stats = human_data['statistics']
    qt_dist = stats.get('question_type_dist', {})

    if not qt_dist:
        print("   ⚠️  No question type data available")
        return

    # Sort by count
    items = sorted(qt_dist.items(), key=lambda x: -x[1])
    types = [item[0] for item in items]
    counts = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(types)), counts, color='steelblue', alpha=0.7)

    ax.set_xlabel('Question Type', fontsize=12)
    ax.set_ylabel('Number of Questions', fontsize=12)
    ax.set_title('Question Type Distribution (VQA)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_answer_distribution(human_data: Dict, output_path: str, top_n: int = 20):
    """Plot top N answer distribution."""
    stats = human_data['statistics']
    ans_dist = stats.get('answer_dist', {})

    if not ans_dist:
        print("   ⚠️  No answer distribution data available")
        return

    # Get top N
    items = sorted(ans_dist.items(), key=lambda x: -x[1])[:top_n]
    answers = [item[0] for item in items]
    counts = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(range(len(answers)), counts, color='coral', alpha=0.7)

    ax.set_ylabel('Answer', fontsize=12)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Most Frequent Answers', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(answers)))
    ax.set_yticklabels(answers)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}',
                ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


def plot_confidence_vs_accuracy(human_data: Dict, output_path: str):
    """Plot relationship between confidence and accuracy."""
    results = human_data['results']

    # Extract confidence and accuracy pairs
    conf_acc_pairs = [(r['confidence'], r['accuracy']) for r in results if 'confidence' in r and 'accuracy' in r]

    if not conf_acc_pairs:
        print("   ⚠️  No confidence-accuracy pairs available")
        return

    confidences, accuracies = zip(*conf_acc_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter plot
    ax = axes[0]
    ax.scatter(confidences, accuracies, alpha=0.3, s=30, color='steelblue')
    ax.set_xlabel('Human Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Confidence vs Accuracy', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(confidences, accuracies, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(confidences), max(confidences), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax.legend()

    # Plot 2: Binned averages
    ax = axes[1]

    # Bin confidences
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    bin_labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0']
    binned_data = defaultdict(list)

    for conf, acc in conf_acc_pairs:
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            if low <= conf < high or (i == len(bins)-2 and conf == high):
                binned_data[bin_labels[i]].append(acc)
                break

    bin_means = [np.mean(binned_data[label]) if label in binned_data else 0 for label in bin_labels]
    bin_stds = [np.std(binned_data[label]) if label in binned_data else 0 for label in bin_labels]

    bars = ax.bar(range(len(bin_labels)), bin_means, yerr=bin_stds, capsize=5,
                   color='steelblue', alpha=0.7, error_kw={'linewidth': 2})

    ax.set_xlabel('Confidence Bin', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title('Mean Accuracy by Confidence Bin', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create visualizations for human response analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate VQA visualizations
  python visualize_human_analysis.py --human_dir evaluation/human_scored/ \
                                      --output_dir evaluation/figures/ \
                                      --answer_type vqa

  # Compare with models
  python visualize_human_analysis.py --human_dir evaluation/human_scored/ \
                                      --model_dir evaluation/data/scored/ \
                                      --models InternVL3_5-2B Qwen3-VL-4B-Instruct \
                                      --dataset vqa_1k_inst_blind \
                                      --output_dir evaluation/figures/
        """
    )

    parser.add_argument("--human_dir", type=str, required=True,
                        help="Directory with human scored results")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory with model scored results (optional)")
    parser.add_argument("--models", type=str, nargs='*', default=[],
                        help="Model names to compare (optional)")
    parser.add_argument("--dataset", type=str, default='vqa_1k_inst_blind',
                        help="Dataset name for model comparison")
    parser.add_argument("--answer_type", type=str, default='vqa', choices=['vqa', 'mc'],
                        help="Answer type: vqa or mc")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for figures")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📊 Generating Visualizations")
    print(f"{'='*60}")

    # Load human data
    human_data = load_human_results(args.human_dir, args.answer_type)
    print(f"✓ Loaded human {args.answer_type.upper()} results: {len(human_data['results'])} items")

    # Load model data if provided
    model_data = []
    if args.model_dir and args.models:
        for model_name in args.models:
            model_results = load_model_results(args.model_dir, model_name, args.dataset)
            if model_results['results']:
                model_data.append({
                    'name': model_name.split('/')[-1],  # Use short name
                    'results': model_results['results']
                })
                print(f"✓ Loaded model {model_name}: {len(model_results['results'])} items")

    # Generate visualizations
    print(f"\n📈 Generating plots...")

    # 1. Confidence distribution
    plot_confidence_distribution(
        human_data,
        os.path.join(args.output_dir, f'human_{args.answer_type}_confidence_dist.png')
    )

    # 2. Accuracy comparison (VQA only)
    if args.answer_type == 'vqa':
        plot_accuracy_comparison(
            human_data,
            model_data,
            os.path.join(args.output_dir, f'human_model_accuracy_comparison.png')
        )

        # 3. Similarity comparison (if available)
        if any('answer_similarity' in r for r in human_data['results']):
            plot_similarity_comparison(
                human_data,
                model_data,
                os.path.join(args.output_dir, f'human_model_similarity_comparison.png')
            )

        # 4. Question type distribution
        plot_question_type_distribution(
            human_data,
            os.path.join(args.output_dir, f'human_vqa_question_types.png')
        )

        # 5. Confidence vs accuracy
        plot_confidence_vs_accuracy(
            human_data,
            os.path.join(args.output_dir, f'human_confidence_vs_accuracy.png')
        )

    # 6. Answer distribution
    plot_answer_distribution(
        human_data,
        os.path.join(args.output_dir, f'human_{args.answer_type}_answer_dist.png')
    )

    print(f"\n{'='*60}")
    print(f"✅ Visualization complete!")
    print(f"   Figures saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
