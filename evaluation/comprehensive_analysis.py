#!/usr/bin/env python3
"""
comprehensive_analysis.py - Complete analysis pipeline for human-model alignment

Analyzes:
1. Correlations between model accuracy and human-model similarity
2. Multimodal gains (baseline - blind conditions)
3. Instruction effects (blind_inst vs blind)
4. Distribution differences (pretrained vs finetuned)

Usage:
    python comprehensive_analysis.py \
        --human_dir evaluation/human_analysis/ \
        --model_dir evaluation/data/scored/ \
        --finetuned_dir evaluation/data/finetuned_scored/ \
        --output_dir evaluation/comprehensive_results/
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
from typing import Dict, List, Tuple
from scipy import stats
from tqdm import tqdm

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


# =============================================================================
# Data Loading
# =============================================================================

def load_human_results(human_dir: str, answer_type: str = 'vqa') -> pd.DataFrame:
    """Load per-question human results."""
    if answer_type == 'vqa':
        path = os.path.join(human_dir, 'human_vqa_per_question.jsonl')
    else:
        path = os.path.join(human_dir, 'human_mc_per_question.jsonl')

    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return pd.DataFrame(data)


def load_model_results(file_path: str) -> pd.DataFrame:
    """Load model scored results."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)


def get_model_files(directory: str, pattern: str = "*.jsonl") -> Dict[str, List[str]]:
    """
    Get model files organized by model name and condition.

    Returns:
        {model_name: {condition: filepath}}
    """
    import glob

    files = glob.glob(os.path.join(directory, pattern))

    organized = defaultdict(dict)

    for file in files:
        basename = os.path.basename(file).replace('.jsonl', '')

        # Parse: ModelName_dataset_condition
        parts = basename.split('_')

        # Extract model name (before dataset)
        if 'vqa' in basename:
            if 'vqa_5k' in basename:
                dataset_idx = basename.index('vqa_5k')
                model = basename[:dataset_idx].rstrip('_')
                dataset = 'vqa_5k'
                condition = basename[dataset_idx + 6:].lstrip('_')
            elif 'vqa_1k' in basename:
                dataset_idx = basename.index('vqa_1k')
                model = basename[:dataset_idx].rstrip('_')
                dataset = 'vqa_1k'
                condition = basename[dataset_idx + 6:].lstrip('_')
        elif 'mmstar' in basename:
            dataset_idx = basename.index('mmstar')
            model = basename[:dataset_idx].rstrip('_')
            dataset = 'mmstar'
            condition = basename[dataset_idx + 7:].lstrip('_')
        elif 'spubench' in basename:
            dataset_idx = basename.index('spubench')
            model = basename[:dataset_idx].rstrip('_')
            dataset = 'spubench'
            condition = basename[dataset_idx + 8:].lstrip('_')
        else:
            continue

        if not condition:
            condition = 'baseline'

        organized[model][condition] = file

    return organized


# =============================================================================
# Analysis 1: Model-Human Correlation
# =============================================================================

def analyze_model_human_correlation(
    human_df: pd.DataFrame,
    model_df: pd.DataFrame,
    human_metric: str = 'mean_gt_similarity',  # or 'mean_visual_similarity'
    output_dir: str = None
) -> Dict:
    """
    Analyze correlation between model accuracy and human metrics.

    Args:
        human_df: Human per-question results
        model_df: Model per-question results
        human_metric: 'mean_gt_similarity' or 'mean_visual_similarity'
        output_dir: Optional directory to save plots

    Returns:
        Dict with correlation statistics
    """
    # Merge on QID
    human_df['qid'] = human_df['qid'].astype(str)
    model_df['qid'] = model_df['qid'].astype(str) if 'qid' in model_df else \
                      model_df['question_id'].astype(str)

    # Handle different column names
    if 'qid' not in model_df.columns and 'question_id' in model_df.columns:
        model_df['qid'] = model_df['question_id']

    merged = pd.merge(
        human_df[['qid', 'mean_accuracy', human_metric, 'mean_confidence']],
        model_df[['qid', 'correct']],
        on='qid',
        how='inner'
    )

    if len(merged) == 0:
        print("⚠️  No matching QIDs found between human and model data")
        return {}

    # Correlations
    results = {
        'num_questions': len(merged),
        'correlation_model_acc_human_sim': float(stats.pearsonr(
            merged['correct'], merged[human_metric]
        )[0]),
        'correlation_model_acc_human_sim_pvalue': float(stats.pearsonr(
            merged['correct'], merged[human_metric]
        )[1]),
        'correlation_model_acc_human_acc': float(stats.pearsonr(
            merged['correct'], merged['mean_accuracy']
        )[0]),
        'correlation_model_acc_human_acc_pvalue': float(stats.pearsonr(
            merged['correct'], merged['mean_accuracy']
        )[1]),
        'spearman_model_acc_human_sim': float(stats.spearmanr(
            merged['correct'], merged[human_metric]
        )[0]),
    }

    # Plot if output dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Model accuracy vs Human similarity
        ax = axes[0]
        scatter = ax.scatter(
            merged[human_metric],
            merged['correct'],
            c=merged['mean_confidence'],
            alpha=0.5,
            cmap='viridis',
            s=50
        )
        ax.set_xlabel(f'Human {human_metric.replace("_", " ").title()}', fontsize=12)
        ax.set_ylabel('Model Accuracy', fontsize=12)
        ax.set_title(
            f'Model Accuracy vs Human {human_metric.split("_")[1].title()} Similarity\n'
            f'r = {results["correlation_model_acc_human_sim"]:.3f} '
            f'(p = {results["correlation_model_acc_human_sim_pvalue"]:.4f})',
            fontsize=12, fontweight='bold'
        )

        # Add trend line
        z = np.polyfit(merged[human_metric], merged['correct'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(merged[human_metric].min(), merged[human_metric].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Human Confidence', fontsize=10)
        ax.grid(alpha=0.3)

        # Plot 2: Model accuracy vs Human accuracy
        ax = axes[1]
        scatter = ax.scatter(
            merged['mean_accuracy'],
            merged['correct'],
            c=merged['mean_confidence'],
            alpha=0.5,
            cmap='viridis',
            s=50
        )
        ax.set_xlabel('Human Mean Accuracy', fontsize=12)
        ax.set_ylabel('Model Accuracy', fontsize=12)
        ax.set_title(
            f'Model Accuracy vs Human Accuracy\n'
            f'r = {results["correlation_model_acc_human_acc"]:.3f} '
            f'(p = {results["correlation_model_acc_human_acc_pvalue"]:.4f})',
            fontsize=12, fontweight='bold'
        )

        # Add trend line
        z = np.polyfit(merged['mean_accuracy'], merged['correct'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(merged['mean_accuracy'].min(), merged['mean_accuracy'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Human Confidence', fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'correlation_{human_metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {output_path}")

    return results


# =============================================================================
# Analysis 2: Multimodal Gains
# =============================================================================

def compute_multimodal_gains(
    model_files: Dict[str, Dict[str, str]],
    dataset: str = 'vqa_1k'
) -> pd.DataFrame:
    """
    Compute multimodal gains: baseline - blind conditions.

    Args:
        model_files: {model: {condition: filepath}}
        dataset: Dataset name

    Returns:
        DataFrame with gains analysis
    """
    results = []

    for model_name, conditions in model_files.items():
        if f'{dataset}' not in conditions:
            continue

        baseline_path = conditions.get(f'{dataset}', None) or conditions.get('baseline', None)
        blind_path = conditions.get(f'{dataset}_blind', None)
        inst_blind_path = conditions.get(f'{dataset}_inst_blind', None)

        if not baseline_path:
            continue

        baseline_df = load_model_results(baseline_path)
        baseline_acc = baseline_df['correct'].mean()

        model_result = {
            'model': model_name,
            'dataset': dataset,
            'baseline_accuracy': baseline_acc,
        }

        # Blind gain
        if blind_path:
            blind_df = load_model_results(blind_path)
            blind_acc = blind_df['correct'].mean()
            model_result['blind_accuracy'] = blind_acc
            model_result['mg_blind'] = baseline_acc - blind_acc
            model_result['mg_blind_relative'] = ((baseline_acc - blind_acc) / baseline_acc) * 100

        # Inst blind gain
        if inst_blind_path:
            inst_blind_df = load_model_results(inst_blind_path)
            inst_blind_acc = inst_blind_df['correct'].mean()
            model_result['inst_blind_accuracy'] = inst_blind_acc
            model_result['mg_inst_blind'] = baseline_acc - inst_blind_acc
            model_result['mg_inst_blind_relative'] = ((baseline_acc - inst_blind_acc) / baseline_acc) * 100

        results.append(model_result)

    return pd.DataFrame(results)


def plot_multimodal_gains(gains_df: pd.DataFrame, output_path: str):
    """Plot multimodal gains."""
    if len(gains_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Absolute gains
    ax = axes[0]
    x = np.arange(len(gains_df))
    width = 0.35

    if 'mg_blind' in gains_df.columns:
        ax.bar(x - width/2, gains_df['mg_blind'], width, label='MG (Blind)', alpha=0.7)
    if 'mg_inst_blind' in gains_df.columns:
        ax.bar(x + width/2, gains_df['mg_inst_blind'], width, label='MG (Inst+Blind)', alpha=0.7)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Multimodal Gain (Absolute)', fontsize=12)
    ax.set_title('Multimodal Gains: Baseline - Blind Conditions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gains_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Plot 2: Relative gains
    ax = axes[1]
    if 'mg_blind_relative' in gains_df.columns:
        ax.bar(x - width/2, gains_df['mg_blind_relative'], width, label='MG (Blind)', alpha=0.7)
    if 'mg_inst_blind_relative' in gains_df.columns:
        ax.bar(x + width/2, gains_df['mg_inst_blind_relative'], width, label='MG (Inst+Blind)', alpha=0.7)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Multimodal Gain (%)', fontsize=12)
    ax.set_title('Multimodal Gains (Relative)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gains_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# =============================================================================
# Analysis 3: Instruction Effects
# =============================================================================

def analyze_instruction_effects(
    model_files: Dict[str, Dict[str, str]],
    dataset: str = 'vqa_1k'
) -> pd.DataFrame:
    """
    Analyze instruction effects: blind_inst vs blind.

    Returns:
        DataFrame with instruction effect analysis
    """
    results = []

    for model_name, conditions in model_files.items():
        blind_path = conditions.get(f'{dataset}_blind', None)
        inst_blind_path = conditions.get(f'{dataset}_inst_blind', None)

        if not blind_path or not inst_blind_path:
            continue

        blind_df = load_model_results(blind_path)
        inst_blind_df = load_model_results(inst_blind_path)

        blind_acc = blind_df['correct'].mean()
        inst_blind_acc = inst_blind_df['correct'].mean()

        instruction_effect = inst_blind_acc - blind_acc
        instruction_effect_relative = (instruction_effect / blind_acc) * 100

        results.append({
            'model': model_name,
            'dataset': dataset,
            'blind_accuracy': blind_acc,
            'inst_blind_accuracy': inst_blind_acc,
            'instruction_effect': instruction_effect,
            'instruction_effect_relative': instruction_effect_relative,
        })

    return pd.DataFrame(results)


def plot_instruction_effects(effects_df: pd.DataFrame, output_path: str):
    """Plot instruction effects."""
    if len(effects_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Accuracies comparison
    ax = axes[0]
    x = np.arange(len(effects_df))
    width = 0.35

    ax.bar(x - width/2, effects_df['blind_accuracy'], width, label='Blind', alpha=0.7, color='coral')
    ax.bar(x + width/2, effects_df['inst_blind_accuracy'], width, label='Inst+Blind', alpha=0.7, color='steelblue')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy: Blind vs Inst+Blind', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(effects_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Instruction effects
    ax = axes[1]
    colors = ['green' if x > 0 else 'red' for x in effects_df['instruction_effect']]
    ax.bar(x, effects_df['instruction_effect'], color=colors, alpha=0.7)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Instruction Effect (Inst+Blind - Blind)', fontsize=12)
    ax.set_title('Instruction Effects', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(effects_df['model'], rotation=45, ha='right')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# =============================================================================
# Analysis 4: Pretrained vs Finetuned Distribution
# =============================================================================

def analyze_pretrained_finetuned_distribution(
    pretrained_files: Dict[str, Dict[str, str]],
    finetuned_files: Dict[str, Dict[str, str]],
    dataset: str = 'vqa_1k',
    condition: str = 'inst_blind'
) -> Dict:
    """
    Compare distribution of accuracies between pretrained and finetuned models.

    Args:
        pretrained_files: Pretrained model files
        finetuned_files: Finetuned model files
        dataset: Dataset name
        condition: Condition to compare

    Returns:
        Dict with distribution comparison statistics
    """
    results = {}

    # Match pretrained and finetuned models
    for base_model in pretrained_files.keys():
        # Find corresponding finetuned versions
        finetuned_variants = [m for m in finetuned_files.keys() if base_model in m]

        if not finetuned_variants:
            continue

        pretrained_path = pretrained_files[base_model].get(f'{dataset}_{condition}', None)
        if not pretrained_path:
            continue

        pretrained_df = load_model_results(pretrained_path)
        pretrained_scores = pretrained_df['correct'].values

        for finetuned_model in finetuned_variants:
            finetuned_path = finetuned_files[finetuned_model].get(f'{dataset}_{condition}', None)
            if not finetuned_path:
                continue

            finetuned_df = load_model_results(finetuned_path)
            finetuned_scores = finetuned_df['correct'].values

            # Statistical tests
            ks_stat, ks_pval = stats.ks_2samp(pretrained_scores, finetuned_scores)
            mw_stat, mw_pval = stats.mannwhitneyu(pretrained_scores, finetuned_scores)

            # Effect size
            pretrained_mean = pretrained_scores.mean()
            finetuned_mean = finetuned_scores.mean()
            pooled_std = np.sqrt((pretrained_scores.std()**2 + finetuned_scores.std()**2) / 2)
            cohens_d = (finetuned_mean - pretrained_mean) / pooled_std if pooled_std > 0 else 0

            results[f'{base_model}_vs_{finetuned_model}'] = {
                'pretrained_model': base_model,
                'finetuned_model': finetuned_model,
                'dataset': dataset,
                'condition': condition,
                'pretrained_mean': float(pretrained_mean),
                'pretrained_std': float(pretrained_scores.std()),
                'finetuned_mean': float(finetuned_mean),
                'finetuned_std': float(finetuned_scores.std()),
                'improvement': float(finetuned_mean - pretrained_mean),
                'improvement_relative': float((finetuned_mean - pretrained_mean) / pretrained_mean * 100),
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'mann_whitney_statistic': float(mw_stat),
                'mann_whitney_pvalue': float(mw_pval),
                'cohens_d': float(cohens_d),
            }

    return results


def plot_pretrained_finetuned_comparison(
    pretrained_files: Dict[str, Dict[str, str]],
    finetuned_files: Dict[str, Dict[str, str]],
    dataset: str,
    condition: str,
    output_dir: str
):
    """Plot distribution comparison between pretrained and finetuned."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_idx = 0

    for base_model in list(pretrained_files.keys())[:4]:  # Limit to 4 for visualization
        finetuned_variants = [m for m in finetuned_files.keys() if base_model in m]
        if not finetuned_variants:
            continue

        pretrained_path = pretrained_files[base_model].get(f'{dataset}_{condition}', None)
        if not pretrained_path:
            continue

        pretrained_df = load_model_results(pretrained_path)

        ax = axes[plot_idx]

        # Plot pretrained
        ax.hist(pretrained_df['correct'], bins=20, alpha=0.6, label='Pretrained',
                color='coral', density=True)

        # Plot finetuned variants
        colors = plt.cm.Set2(np.linspace(0, 1, len(finetuned_variants)))
        for finetuned_model, color in zip(finetuned_variants, colors):
            finetuned_path = finetuned_files[finetuned_model].get(f'{dataset}_{condition}', None)
            if not finetuned_path:
                continue

            finetuned_df = load_model_results(finetuned_path)
            ax.hist(finetuned_df['correct'], bins=20, alpha=0.4,
                   label=finetuned_model.split('/')[-1][:20], color=color, density=True)

        ax.set_xlabel('Accuracy', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{base_model.split("/")[-1]}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plot_idx += 1
        if plot_idx >= 4:
            break

    # Hide unused subplots
    for idx in range(plot_idx, 4):
        axes[idx].set_visible(False)

    plt.suptitle(f'Accuracy Distributions: Pretrained vs Finetuned\n{dataset} - {condition}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'pretrained_vs_finetuned_{dataset}_{condition}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_comprehensive_analysis(
    human_dir: str,
    model_dir: str,
    finetuned_dir: str,
    output_dir: str,
    dataset: str = 'vqa_1k'
):
    """
    Run complete analysis pipeline.
    """
    print(f"\n{'='*80}")
    print("🔬 COMPREHENSIVE ANALYSIS PIPELINE")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n📚 Loading data...")
    human_vqa_df = load_human_results(human_dir, 'vqa')
    print(f"   ✓ Loaded human VQA: {len(human_vqa_df)} questions")

    pretrained_files = get_model_files(model_dir)
    finetuned_files = get_model_files(finetuned_dir)
    print(f"   ✓ Found {len(pretrained_files)} pretrained models")
    print(f"   ✓ Found {len(finetuned_files)} finetuned models")

    all_results = {}

    # =========================================================================
    # Analysis 1: Model-Human Correlation
    # =========================================================================
    print(f"\n{'='*80}")
    print("📊 Analysis 1: Model-Human Correlation")
    print(f"{'='*80}")

    correlation_results = {}

    # Try both GT and visual similarity
    for metric in ['mean_gt_similarity', 'mean_visual_similarity']:
        if metric not in human_vqa_df.columns:
            continue

        print(f"\n  Analyzing with {metric}...")

        for model_name, conditions in list(pretrained_files.items())[:5]:  # Sample
            for condition, filepath in conditions.items():
                if dataset not in condition:
                    continue

                try:
                    model_df = load_model_results(filepath)
                    result = analyze_model_human_correlation(
                        human_vqa_df,
                        model_df,
                        human_metric=metric,
                        output_dir=os.path.join(output_dir, 'correlations', model_name.replace('/', '_'))
                    )

                    if result:
                        key = f"{model_name}_{condition}_{metric}"
                        correlation_results[key] = result
                        print(f"    ✓ {model_name} ({condition}): r = {result.get('correlation_model_acc_human_sim', 0):.3f}")
                except Exception as e:
                    print(f"    ⚠️  Error processing {model_name}: {e}")

    all_results['correlations'] = correlation_results

    # =========================================================================
    # Analysis 2: Multimodal Gains
    # =========================================================================
    print(f"\n{'='*80}")
    print("📊 Analysis 2: Multimodal Gains")
    print(f"{'='*80}")

    mg_df = compute_multimodal_gains(pretrained_files, dataset=dataset)
    print(f"\n  Computed gains for {len(mg_df)} models")
    print(mg_df[['model', 'baseline_accuracy', 'mg_blind', 'mg_inst_blind']].to_string(index=False))

    plot_multimodal_gains(mg_df, os.path.join(output_dir, f'multimodal_gains_{dataset}.png'))

    mg_df.to_csv(os.path.join(output_dir, f'multimodal_gains_{dataset}.csv'), index=False)
    all_results['multimodal_gains'] = mg_df.to_dict('records')

    # =========================================================================
    # Analysis 3: Instruction Effects
    # =========================================================================
    print(f"\n{'='*80}")
    print("📊 Analysis 3: Instruction Effects")
    print(f"{'='*80}")

    inst_effects_df = analyze_instruction_effects(pretrained_files, dataset=dataset)
    print(f"\n  Computed instruction effects for {len(inst_effects_df)} models")
    print(inst_effects_df[['model', 'blind_accuracy', 'inst_blind_accuracy', 'instruction_effect']].to_string(index=False))

    plot_instruction_effects(inst_effects_df, os.path.join(output_dir, f'instruction_effects_{dataset}.png'))

    inst_effects_df.to_csv(os.path.join(output_dir, f'instruction_effects_{dataset}.csv'), index=False)
    all_results['instruction_effects'] = inst_effects_df.to_dict('records')

    # =========================================================================
    # Analysis 4: Pretrained vs Finetuned
    # =========================================================================
    print(f"\n{'='*80}")
    print("📊 Analysis 4: Pretrained vs Finetuned Distribution")
    print(f"{'='*80}")

    dist_comparison = analyze_pretrained_finetuned_distribution(
        pretrained_files,
        finetuned_files,
        dataset=dataset,
        condition='inst_blind'
    )

    print(f"\n  Compared {len(dist_comparison)} model pairs")
    for comparison_name, stats_dict in dist_comparison.items():
        print(f"\n  {comparison_name}:")
        print(f"    Pretrained: {stats_dict['pretrained_mean']:.4f} ± {stats_dict['pretrained_std']:.4f}")
        print(f"    Finetuned:  {stats_dict['finetuned_mean']:.4f} ± {stats_dict['finetuned_std']:.4f}")
        print(f"    Improvement: {stats_dict['improvement']:.4f} ({stats_dict['improvement_relative']:.2f}%)")
        print(f"    Cohen's d: {stats_dict['cohens_d']:.3f}")
        print(f"    KS test: p = {stats_dict['ks_pvalue']:.4f}")

    plot_pretrained_finetuned_comparison(
        pretrained_files,
        finetuned_files,
        dataset=dataset,
        condition='inst_blind',
        output_dir=output_dir
    )

    all_results['pretrained_vs_finetuned'] = dist_comparison

    # =========================================================================
    # Save Summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("💾 Saving results...")
    print(f"{'='*80}")

    summary_path = os.path.join(output_dir, 'comprehensive_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"   ✓ Saved: {summary_path}")

    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"   Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive human-model alignment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--human_dir", type=str, required=True,
                        help="Directory with processed human results")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with pretrained model results")
    parser.add_argument("--finetuned_dir", type=str, required=True,
                        help="Directory with finetuned model results")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for analysis results")
    parser.add_argument("--dataset", type=str, default='vqa_1k',
                        help="Dataset to analyze (default: vqa_1k)")

    args = parser.parse_args()

    run_comprehensive_analysis(
        args.human_dir,
        args.model_dir,
        args.finetuned_dir,
        args.output_dir,
        args.dataset
    )


if __name__ == '__main__':
    main()
