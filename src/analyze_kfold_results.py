#!/usr/bin/env python3
"""
Analyze k-fold cross-validation results.

Aggregates metrics across all folds and computes statistics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_trainer_state(fold_dir: Path) -> Dict:
    """Load trainer state from a fold directory."""
    trainer_state_path = fold_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return None

    with open(trainer_state_path, 'r') as f:
        return json.load(f)


def extract_best_metrics(trainer_state: Dict) -> Dict:
    """Extract best validation metrics from trainer state."""
    if trainer_state is None:
        return {}

    log_history = trainer_state.get('log_history', [])

    # Find best eval metrics
    best_eval_loss = float('inf')
    best_metrics = {}

    for entry in log_history:
        if 'eval_loss' in entry:
            eval_loss = entry.get('eval_loss', float('inf'))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_metrics = {k: v for k, v in entry.items() if k.startswith('eval_')}

    return best_metrics


def analyze_kfold_results(kfold_output_dir: str) -> Dict:
    """
    Analyze k-fold cross-validation results.

    Args:
        kfold_output_dir: Directory containing fold_0/, fold_1/, etc.

    Returns:
        Dictionary with aggregated statistics
    """
    kfold_path = Path(kfold_output_dir)

    # Find all fold directories
    fold_dirs = sorted([d for d in kfold_path.iterdir() if d.is_dir() and d.name.startswith('fold_')])

    if not fold_dirs:
        raise ValueError(f"No fold directories found in {kfold_output_dir}")

    print(f"\n{'=' * 80}")
    print(f"K-Fold Cross-Validation Results Analysis")
    print(f"{'=' * 80}")
    print(f"Directory: {kfold_output_dir}")
    print(f"Folds found: {len(fold_dirs)}")
    print(f"{'=' * 80}\n")

    # Collect metrics from all folds
    all_fold_metrics = []

    for fold_dir in fold_dirs:
        fold_idx = fold_dir.name
        print(f"Loading metrics from {fold_idx}...")

        trainer_state = load_trainer_state(fold_dir)
        if trainer_state is None:
            print(f"  ⚠️  No trainer_state.json found in {fold_idx}")
            continue

        best_metrics = extract_best_metrics(trainer_state)
        if not best_metrics:
            print(f"  ⚠️  No evaluation metrics found in {fold_idx}")
            continue

        all_fold_metrics.append({
            'fold': fold_idx,
            'metrics': best_metrics
        })

        print(f"  ✅ Loaded: eval_loss = {best_metrics.get('eval_loss', 'N/A')}")

    if not all_fold_metrics:
        raise ValueError("No valid fold metrics found!")

    # Aggregate metrics
    print(f"\n{'=' * 80}")
    print(f"Aggregating Metrics Across {len(all_fold_metrics)} Folds")
    print(f"{'=' * 80}\n")

    # Collect all metric names
    metric_names = set()
    for fold_data in all_fold_metrics:
        metric_names.update(fold_data['metrics'].keys())

    # Compute statistics for each metric
    aggregated_stats = {}

    for metric_name in sorted(metric_names):
        values = []
        for fold_data in all_fold_metrics:
            if metric_name in fold_data['metrics']:
                values.append(fold_data['metrics'][metric_name])

        if values:
            aggregated_stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'values': values,
            }

    # Print results
    print(f"{'Metric':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-' * 80}")

    for metric_name, stats in sorted(aggregated_stats.items()):
        print(f"{metric_name:<30} "
              f"{stats['mean']:<12.6f} "
              f"{stats['std']:<12.6f} "
              f"{stats['min']:<12.6f} "
              f"{stats['max']:<12.6f}")

    # Save results
    results = {
        'kfold_output_dir': str(kfold_output_dir),
        'n_folds': len(all_fold_metrics),
        'fold_metrics': all_fold_metrics,
        'aggregated_stats': aggregated_stats,
    }

    output_path = kfold_path / "kfold_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Analysis saved: {output_path}")
    print(f"{'=' * 80}\n")

    # Print summary
    if 'eval_loss' in aggregated_stats:
        eval_loss = aggregated_stats['eval_loss']
        print(f"📊 Summary:")
        print(f"   Eval Loss: {eval_loss['mean']:.6f} ± {eval_loss['std']:.6f}")
        print(f"   Range: [{eval_loss['min']:.6f}, {eval_loss['max']:.6f}]")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze k-fold cross-validation results")
    parser.add_argument("--kfold_output_dir", type=str, required=True,
                        help="Directory containing fold_0/, fold_1/, etc.")

    args = parser.parse_args()

    analyze_kfold_results(args.kfold_output_dir)


if __name__ == "__main__":
    main()
