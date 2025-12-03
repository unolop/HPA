#!/usr/bin/env python3
"""
7_visualize_results.py - Generate Paper Figures

Creates publication-ready plots from analysis results.

Figures:
1. Calibration curve
2. Human-model agreement matrix (heatmap)
3. Category comparison (grouped bars)
4. Confidence-accuracy scatter
5. Ablation comparison

Usage:
    python 7_visualize_results.py \
        --human_model_analysis ./analysis/human_model/human_model_analysis.json \
        --calibration_analysis ./analysis/calibration/calibration_analysis.json \
        --scored_results ./results/scored/*.json \
        --output_dir ./figures
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'human': '#2E86AB',
    'model': '#A23B72',
    'both_correct': '#28A745',
    'both_wrong': '#DC3545',
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
}


# =============================================================================
# Plot 1: Calibration Curve
# =============================================================================

def plot_calibration_curve(calibration_data: Dict, output_path: str):
    """Plot human confidence calibration curve."""
    curve = calibration_data.get('calibration_curve', {})
    metrics = calibration_data.get('calibration_metrics', {})
    
    if not curve:
        print("⚠️ No calibration curve data")
        return
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Data
    confs = sorted([int(k) for k in curve.keys()])
    conf_norm = [c / 5.0 for c in confs]
    accs = [curve[str(c)]['accuracy'] for c in confs]
    errors = [curve[str(c)].get('std_error', 0) for c in confs]
    counts = [curve[str(c)]['count'] for c in confs]
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=2, label='Perfect calibration')
    
    # Actual calibration
    ax.errorbar(conf_norm, accs, yerr=errors, fmt='o-', color=COLORS['human'],
                markersize=12, lw=2.5, capsize=5, label='Human responses')
    
    # Fill gap
    ax.fill_between(conf_norm, conf_norm, accs, alpha=0.2, color=COLORS['human'])
    
    # Annotations
    for c, acc, count in zip(conf_norm, accs, counts):
        ax.annotate(f'n={count}', xy=(c, acc), xytext=(5, 10),
                    textcoords='offset points', fontsize=9, color='gray')
    
    # Metrics box
    ece = metrics.get('ece', 0)
    brier = metrics.get('brier', 0)
    ax.text(0.05, 0.95, f'ECE = {ece:.3f}\nBrier = {brier:.3f}',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Confidence (normalized)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Human Confidence Calibration')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved: {output_path}")


# =============================================================================
# Plot 2: Agreement Matrix Heatmap
# =============================================================================

def plot_agreement_matrix(analysis_data: Dict, output_path: str):
    """Plot 2x2 agreement matrix as heatmap."""
    matrix = analysis_data.get('agreement_matrix', {})
    
    if not matrix:
        print("⚠️ No agreement matrix data")
        return
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Build matrix
    data = np.array([
        [matrix.get('both_correct', 0), matrix.get('model_only', 0)],
        [matrix.get('human_only', 0), matrix.get('both_wrong', 0)]
    ])
    
    total = data.sum()
    data_pct = data / total * 100
    
    # Heatmap
    im = ax.imshow(data_pct, cmap='Blues', vmin=0, vmax=60)
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Model ✓', 'Model ✗'])
    ax.set_yticklabels(['Human ✓', 'Human ✗'])
    
    # Values
    labels = [['Both Correct', 'Human Only'], ['Model Only', 'Both Wrong']]
    for i in range(2):
        for j in range(2):
            color = 'white' if data_pct[i, j] > 30 else 'black'
            ax.text(j, i, f'{labels[i][j]}\n{data_pct[i, j]:.1f}%\n({int(data[i, j])})',
                    ha='center', va='center', fontsize=10, color=color)
    
    ax.set_title('Human-Model Agreement')
    plt.colorbar(im, ax=ax, label='Percentage')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved: {output_path}")


# =============================================================================
# Plot 3: Category Comparison
# =============================================================================

def plot_category_comparison(analysis_data: Dict, output_path: str, max_cats: int = 12):
    """Plot human vs model accuracy by category."""
    categories = analysis_data.get('by_category', {})
    
    if not categories:
        print("⚠️ No category data")
        return
    
    # Sort by total samples
    sorted_cats = sorted(categories.items(), key=lambda x: -x[1].get('total', 0))[:max_cats]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [c[0][:20] for c in sorted_cats]
    human_acc = [c[1].get('human_accuracy', 0) for c in sorted_cats]
    model_acc = [c[1].get('model_accuracy', 0) for c in sorted_cats]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, human_acc, width, label='Human', color=COLORS['human'], alpha=0.8)
    ax.bar(x + width/2, model_acc, width, label='Model', color=COLORS['model'], alpha=0.8)
    
    ax.set_xlabel('Category')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Sample sizes
    for i, (_, data) in enumerate(sorted_cats):
        ax.annotate(f'n={data.get("total", 0)}',
                    xy=(i, -0.08), ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved: {output_path}")


# =============================================================================
# Plot 4: Confidence vs Accuracy Scatter
# =============================================================================

def plot_confidence_accuracy(calibration_data: Dict, output_path: str):
    """Scatter plot of confidence vs accuracy per question."""
    questions = calibration_data.get('questions', {})
    
    if not questions:
        print("⚠️ No question data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    confs = [q['mean_confidence'] for q in questions.values()]
    accs = [q['accuracy'] for q in questions.values()]
    sizes = [q['num_responses'] * 10 for q in questions.values()]
    
    ax.scatter(confs, accs, s=sizes, alpha=0.5, c=COLORS['primary'])
    
    # Trend line
    z = np.polyfit(confs, accs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(confs), max(confs), 100)
    ax.plot(x_line, p(x_line), '--', color='red', lw=2, label=f'Trend')
    
    # Perfect calibration
    ax.plot([1, 5], [0.2, 1.0], ':', color='gray', lw=2, label='Perfect')
    
    ax.set_xlabel('Mean Confidence (1-5)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Question Confidence vs Accuracy')
    ax.legend()
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved: {output_path}")


# =============================================================================
# Plot 5: Ablation Comparison
# =============================================================================

def plot_ablation_comparison(scored_results: List[Dict], output_path: str):
    """Bar chart comparing ablation conditions."""
    if not scored_results:
        print("⚠️ No scored results")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by condition
    names = []
    blind_accs = []
    real_accs = []
    
    for result in scored_results:
        name = result.get('file', result.get('model', 'Unknown'))
        name = name.replace('.jsonl', '').replace('.json', '')
        
        # Get accuracy
        acc = result.get('accuracy', result.get('metrics', {}).get('accuracy', 0))
        condition = result.get('condition', '')
        
        if 'blind_inst' in condition or 'blind' in name.lower():
            names.append(name[:30])
            blind_accs.append(acc)
            real_accs.append(0)  # No real data
        else:
            names.append(name[:30])
            real_accs.append(acc)
            blind_accs.append(0)
    
    x = np.arange(len(names))
    width = 0.6
    
    # Only plot blind if we have data
    if any(blind_accs):
        ax.bar(x, blind_accs, width, label='Blind', color=COLORS['primary'], alpha=0.8)
    if any(real_accs):
        ax.bar(x, real_accs, width, label='Real', color=COLORS['secondary'], alpha=0.8)
    
    ax.set_xlabel('Model / Condition')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    
    # Value labels
    for i, (b, r) in enumerate(zip(blind_accs, real_accs)):
        val = b if b > 0 else r
        if val > 0:
            ax.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=9)
    
    if any(blind_accs) and any(real_accs):
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved: {output_path}")


# =============================================================================
# Plot 6: Human vs Model Accuracy Scatter by Category
# =============================================================================

def plot_human_model_scatter(analysis_data: Dict, output_path: str):
    """Scatter plot of human vs model accuracy per category."""
    categories = analysis_data.get('by_category', {})
    
    if not categories:
        print("⚠️ No category data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    human_acc = [c['human_accuracy'] for c in categories.values()]
    model_acc = [c['model_accuracy'] for c in categories.values()]
    sizes = [c['total'] * 3 for c in categories.values()]
    
    ax.scatter(human_acc, model_acc, s=sizes, alpha=0.6, c=COLORS['primary'])
    
    # y = x line
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=2, label='y = x')
    
    # Label large categories
    for cat, data in categories.items():
        if data['total'] > np.percentile([c['total'] for c in categories.values()], 75):
            ax.annotate(cat[:15], (data['human_accuracy'], data['model_accuracy']),
                        fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Human Accuracy')
    ax.set_ylabel('Model Accuracy')
    ax.set_title('Human vs Model Accuracy by Category')
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def generate_all_figures(
    human_model_path: str = None,
    calibration_path: str = None,
    scored_paths: List[str] = None,
    output_dir: str = "./figures",
):
    """Generate all figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📊 GENERATING FIGURES")
    print("=" * 60)
    
    # Load data
    human_model_data = None
    if human_model_path and os.path.exists(human_model_path):
        with open(human_model_path) as f:
            human_model_data = json.load(f)
        print(f"✓ Loaded human-model analysis")
    
    calibration_data = None
    if calibration_path and os.path.exists(calibration_path):
        with open(calibration_path) as f:
            calibration_data = json.load(f)
        print(f"✓ Loaded calibration analysis")
    
    scored_results = []
    if scored_paths:
        for p in scored_paths:
            for f in glob(p):
                with open(f) as file:
                    data = json.load(file)
                    if isinstance(data, list):
                        scored_results.extend(data)
                    else:
                        scored_results.append(data)
        print(f"✓ Loaded {len(scored_results)} scored results")
    
    # Generate figures
    print("\n📈 Generating plots...")
    
    if calibration_data:
        plot_calibration_curve(calibration_data, 
                               os.path.join(output_dir, 'calibration_curve.png'))
        plot_confidence_accuracy(calibration_data,
                                 os.path.join(output_dir, 'confidence_accuracy.png'))
    
    if human_model_data:
        plot_agreement_matrix(human_model_data,
                              os.path.join(output_dir, 'agreement_matrix.png'))
        plot_category_comparison(human_model_data,
                                 os.path.join(output_dir, 'category_comparison.png'))
        plot_human_model_scatter(human_model_data,
                                 os.path.join(output_dir, 'human_model_scatter.png'))
    
    if scored_results:
        plot_ablation_comparison(scored_results,
                                 os.path.join(output_dir, 'ablation_comparison.png'))
    
    print(f"\n✅ All figures saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--human_model_analysis", type=str, default=None)
    parser.add_argument("--calibration_analysis", type=str, default=None)
    parser.add_argument("--scored_results", type=str, nargs='*', default=None)
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()
    
    generate_all_figures(
        human_model_path=args.human_model_analysis,
        calibration_path=args.calibration_analysis,
        scored_paths=args.scored_results,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()