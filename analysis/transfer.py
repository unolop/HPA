#!/usr/bin/env python3
"""
Cross-Benchmark Transfer Analysis

Analyzes how well linguistic priors learned from one benchmark
transfer to other benchmarks.

Key Questions:
1. Are linguistic priors benchmark-specific or general?
2. Which benchmark's priors transfer best?
3. What types of questions show best/worst transfer?

Usage:
    python analyze_cross_benchmark_transfer.py \
        --train_benchmark vqav2 \
        --test_benchmarks mmstar mmspubench vqav2 \
        --human_data_dir ./human_data \
        --model_predictions_dir ./predictions \
        --output_dir ./transfer_analysis
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def load_benchmark_data(
    benchmark: str,
    human_data_dir: str,
    predictions_dir: str,
) -> Dict[str, Any]:
    """Load human data and model predictions for a benchmark."""
    import csv
    
    # Load human data
    human_responses = defaultdict(list)
    human_dir = os.path.join(human_data_dir, benchmark)
    
    if os.path.exists(human_dir):
        for csv_file in Path(human_dir).glob("*.csv"):
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    qid = str(row['qid'])
                    human_responses[qid].append({
                        'answer': row['answer'],
                        'confidence': int(row['confidence']),
                    })
    
    # Load model predictions
    model_predictions = {}
    pred_file = os.path.join(predictions_dir, f"{benchmark}_predictions.json")
    
    if os.path.exists(pred_file):
        with open(pred_file, 'r') as f:
            model_predictions = json.load(f)
    
    return {
        'human_responses': dict(human_responses),
        'model_predictions': model_predictions,
    }


def compute_human_prior_vector(
    human_responses: Dict[str, List[Dict]],
) -> Dict[str, Dict]:
    """
    Compute human prior "fingerprint" for each question.
    
    Returns:
        qid -> {
            'majority_answer': str,
            'accuracy': float (if GT available),
            'confidence': float,
            'agreement': float,
            'answer_distribution': Dict[str, float]
        }
    """
    priors = {}
    
    for qid, responses in human_responses.items():
        if not responses:
            continue
        
        # Answer distribution
        answer_counts = defaultdict(int)
        for r in responses:
            answer_counts[r['answer']] += 1
        
        total = sum(answer_counts.values())
        answer_dist = {a: c / total for a, c in answer_counts.items()}
        
        # Majority answer
        majority = max(answer_counts.keys(), key=lambda x: answer_counts[x])
        
        # Mean confidence
        mean_conf = np.mean([r['confidence'] for r in responses])
        
        # Agreement (entropy-based)
        probs = list(answer_dist.values())
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(probs) + 1e-10)
        agreement = 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        priors[qid] = {
            'majority_answer': majority,
            'confidence': mean_conf,
            'agreement': agreement,
            'answer_distribution': answer_dist,
            'num_responses': total,
        }
    
    return priors


def compute_transfer_metrics(
    source_priors: Dict[str, Dict],
    target_priors: Dict[str, Dict],
    shared_qids: List[str] = None,
) -> Dict[str, float]:
    """
    Compute how well priors from source transfer to target.
    
    If questions are shared, compare directly.
    If not shared, compare aggregate statistics.
    """
    metrics = {}
    
    if shared_qids:
        # Direct comparison for shared questions
        source_conf = [source_priors[qid]['confidence'] for qid in shared_qids]
        target_conf = [target_priors[qid]['confidence'] for qid in shared_qids]
        
        source_agree = [source_priors[qid]['agreement'] for qid in shared_qids]
        target_agree = [target_priors[qid]['agreement'] for qid in shared_qids]
        
        # Correlation
        if len(source_conf) >= 3:
            conf_corr, _ = spearmanr(source_conf, target_conf)
            agree_corr, _ = spearmanr(source_agree, target_agree)
            metrics['confidence_correlation'] = conf_corr
            metrics['agreement_correlation'] = agree_corr
        
        # Answer match rate
        matches = sum(
            1 for qid in shared_qids
            if source_priors[qid]['majority_answer'] == target_priors[qid]['majority_answer']
        )
        metrics['answer_match_rate'] = matches / len(shared_qids)
    
    else:
        # Aggregate comparison (different questions)
        source_confs = [p['confidence'] for p in source_priors.values()]
        target_confs = [p['confidence'] for p in target_priors.values()]
        
        # Distribution similarity (KL divergence of confidence distributions)
        source_hist, _ = np.histogram(source_confs, bins=5, range=(1, 5), density=True)
        target_hist, _ = np.histogram(target_confs, bins=5, range=(1, 5), density=True)
        
        # Add small epsilon to avoid log(0)
        source_hist = source_hist + 1e-10
        target_hist = target_hist + 1e-10
        
        kl_div = np.sum(source_hist * np.log(source_hist / target_hist))
        metrics['confidence_kl_divergence'] = kl_div
        
        # Mean difference
        metrics['confidence_mean_diff'] = abs(np.mean(source_confs) - np.mean(target_confs))
        
        source_agrees = [p['agreement'] for p in source_priors.values()]
        target_agrees = [p['agreement'] for p in target_priors.values()]
        metrics['agreement_mean_diff'] = abs(np.mean(source_agrees) - np.mean(target_agrees))
    
    return metrics


def analyze_model_transfer(
    train_benchmark: str,
    test_benchmarks: List[str],
    human_data_dir: str,
    predictions_dir: str,
) -> Dict[str, Any]:
    """
    Analyze how model trained on one benchmark performs on others.
    
    Compares:
    1. Trained model accuracy
    2. Zero-shot model accuracy
    3. Human accuracy
    """
    results = {
        'train_benchmark': train_benchmark,
        'test_benchmarks': {},
    }
    
    for test_benchmark in test_benchmarks:
        # Load data
        test_data = load_benchmark_data(test_benchmark, human_data_dir, predictions_dir)
        
        # Compute human priors
        human_priors = compute_human_prior_vector(test_data['human_responses'])
        
        # Get model predictions (trained and zero-shot)
        trained_preds = test_data['model_predictions'].get(f'trained_on_{train_benchmark}', {})
        zeroshot_preds = test_data['model_predictions'].get('zeroshot', {})
        
        benchmark_results = {
            'num_questions': len(human_priors),
            'human_mean_confidence': np.mean([p['confidence'] for p in human_priors.values()]),
            'human_mean_agreement': np.mean([p['agreement'] for p in human_priors.values()]),
        }
        
        # TODO: Add accuracy computation if ground truth available
        
        results['test_benchmarks'][test_benchmark] = benchmark_results
    
    return results


def plot_transfer_matrix(
    transfer_results: Dict[str, Dict],
    output_path: str,
    metric: str = 'accuracy',
):
    """
    Plot transfer matrix showing performance across train/test benchmarks.
    """
    benchmarks = list(transfer_results.keys())
    n = len(benchmarks)
    
    matrix = np.zeros((n, n))
    
    for i, train_bench in enumerate(benchmarks):
        for j, test_bench in enumerate(benchmarks):
            if test_bench in transfer_results[train_bench].get('test_benchmarks', {}):
                value = transfer_results[train_bench]['test_benchmarks'][test_bench].get(metric, 0)
                matrix[i, j] = value
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.set_yticklabels(benchmarks)
    
    ax.set_xlabel('Test Benchmark')
    ax.set_ylabel('Train Benchmark')
    ax.set_title(f'Transfer Performance ({metric})')
    
    # Add values
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=10)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def compare_prior_distributions(
    benchmarks: List[str],
    human_data_dir: str,
    output_path: str,
):
    """
    Compare distribution of human priors across benchmarks.
    
    Shows:
    - Confidence distributions
    - Agreement distributions
    - Answer entropy distributions
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    benchmark_data = {}
    
    for benchmark in benchmarks:
        data = load_benchmark_data(benchmark, human_data_dir, "")
        priors = compute_human_prior_vector(data['human_responses'])
        benchmark_data[benchmark] = priors
    
    # Plot 1: Confidence distribution
    ax1 = axes[0]
    for benchmark, priors in benchmark_data.items():
        confs = [p['confidence'] for p in priors.values()]
        ax1.hist(confs, bins=5, range=(1, 5), alpha=0.5, label=benchmark, density=True)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Density')
    ax1.set_title('Confidence Distribution')
    ax1.legend()
    
    # Plot 2: Agreement distribution
    ax2 = axes[1]
    for benchmark, priors in benchmark_data.items():
        agrees = [p['agreement'] for p in priors.values()]
        ax2.hist(agrees, bins=20, alpha=0.5, label=benchmark, density=True)
    ax2.set_xlabel('Agreement')
    ax2.set_ylabel('Density')
    ax2.set_title('Inter-Annotator Agreement')
    ax2.legend()
    
    # Plot 3: Box plot comparison
    ax3 = axes[2]
    conf_data = [
        [p['confidence'] for p in priors.values()]
        for priors in benchmark_data.values()
    ]
    ax3.boxplot(conf_data, labels=list(benchmark_data.keys()))
    ax3.set_ylabel('Confidence')
    ax3.set_title('Confidence by Benchmark')
    
    plt.suptitle('Human Prior Distributions Across Benchmarks')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-benchmark transfer analysis"
    )
    
    parser.add_argument("--train_benchmark", type=str, default="vqav2",
                        help="Benchmark used for training")
    parser.add_argument("--test_benchmarks", type=str, nargs='+',
                        default=["vqav2", "mmstar", "mmspubench"],
                        help="Benchmarks to test on")
    parser.add_argument("--human_data_dir", type=str, required=True,
                        help="Directory with human data")
    parser.add_argument("--predictions_dir", type=str, default="",
                        help="Directory with model predictions")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compare prior distributions
    compare_prior_distributions(
        args.test_benchmarks,
        args.human_data_dir,
        os.path.join(args.output_dir, 'prior_distributions.png'),
    )
    
    # Analyze transfer
    if args.predictions_dir:
        results = analyze_model_transfer(
            args.train_benchmark,
            args.test_benchmarks,
            args.human_data_dir,
            args.predictions_dir,
        )
        
        with open(os.path.join(args.output_dir, 'transfer_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()