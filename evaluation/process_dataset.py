#!/usr/bin/env python3
"""
Process and combine results for a specific dataset and condition.
Combines all sources (models, humans, finetuned) into one JSONL file.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import numpy as np

from eval_utils import (
    load_jsonl, save_jsonl, parse_filename, compute_accuracy,
    get_category, DATASET_TYPES, answer_similarity
)


def process_files(
    file_paths: List[str],
    dataset_name: str,
    condition: str,
    encoder=None,
) -> tuple[List[Dict], Dict]:
    """
    Process and combine all result files.

    Returns:
        (combined_results, summary_stats)
    """
    dataset_type = DATASET_TYPES.get(dataset_name, 'multi-choice')
    all_results = []
    stats_by_source = defaultdict(lambda: {
        'count': 0,
        'correct': 0,
        'similarities': [],
    })

    print(f"\n{'='*80}")
    print(f"Processing {dataset_name} - {condition or '(no condition)'}")
    print(f"Dataset type: {dataset_type}")
    print(f"{'='*80}\n")

    for filepath in file_paths:
        filepath = str(filepath)
        # print(f"Loading: {filepath}")

        # Parse metadata
        metadata = parse_filename(filepath)

        # Skip if doesn't match dataset/condition
        if metadata['dataset'] != dataset_name:
            print(f"  ⚠️  Skipping - dataset mismatch: {metadata['dataset']} != {dataset_name}")
            continue

        if condition and metadata['condition'] != condition:
            print(f"  ⚠️  Skipping - condition mismatch: {metadata['condition']} != {condition}")
            continue

        # Load results
        results = load_jsonl(filepath)
        if not results:
            print(f"  ⚠️  Empty file")
            continue

        # Process each item
        for item in results:
            # Compute accuracy
            is_correct = compute_accuracy(item, dataset_type)
            item['correct'] = is_correct

            # Compute embedding similarity for VQA
            if dataset_type == 'open-ended' and encoder is not None:
                gt = item.get('answer', item.get('ground_truth', ''))
                pred = item.get('output', item.get('response', item.get('prediction', '')))
                all_answers = item.get('all_answers', item.get('answers', []))
                if not all_answers:
                    all_answers = [gt] if gt else []
                if all_answers and isinstance(all_answers[0], dict):
                    all_answers = [a.get('answer', '') for a in all_answers if 'answer' in a]

                sim = answer_similarity(all_answers if all_answers else [gt], pred, encoder)
                item['embedding_similarity'] = sim
            else:
                item['embedding_similarity'] = 0.0

            # Add metadata
            item['source'] = metadata['source']
            item['model'] = metadata['model']
            if metadata['training_method']:
                item['training_method'] = metadata['training_method']

            all_results.append(item)

            # Update stats
            source_key = metadata['source']
            stats_by_source[source_key]['count'] += 1
            stats_by_source[source_key]['correct'] += int(is_correct)
            if item['embedding_similarity'] > 0:
                stats_by_source[source_key]['similarities'].append(item['embedding_similarity'])

        acc = sum(1 for r in results if r.get('correct', False)) / len(results) if results else 0
        print(f"  ✓ {metadata['model']} {len(results)} items, accuracy: {acc:.4f}")

    # Generate summary stats
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")

    summary = {
        'dataset': dataset_name,
        'condition': condition or 'none',
        'total_items': len(all_results),
        'by_source': {},
        'by_category': {},
        'overall': {},
    }

    # Overall stats
    total_correct = sum(1 for r in all_results if r.get('correct', False))
    all_sims = [r['embedding_similarity'] for r in all_results if r['embedding_similarity'] > 0]

    summary['overall'] = {
        'num_items': len(all_results),
        'num_correct': total_correct,
        'accuracy': total_correct / len(all_results) if all_results else 0.0,
        'embedding_similarity': float(np.mean(all_sims)) if all_sims else 0.0,
    }

    # By source
    for source, stats in stats_by_source.items():
        summary['by_source'][source] = {
            'num_items': stats['count'],
            'num_correct': stats['correct'],
            'accuracy': stats['correct'] / stats['count'] if stats['count'] > 0 else 0.0,
            'embedding_similarity': float(np.mean(stats['similarities'])) if stats['similarities'] else 0.0,
        }

    # By category
    cat_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'similarities': []})
    for item in all_results:
        cat = get_category(item, dataset_name)
        cat_stats[cat]['total'] += 1
        cat_stats[cat]['correct'] += int(item.get('correct', False))
        if item['embedding_similarity'] > 0:
            cat_stats[cat]['similarities'].append(item['embedding_similarity'])

    for cat, stats in cat_stats.items():
        summary['by_category'][cat] = {
            'num_items': stats['total'],
            'num_correct': stats['correct'],
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
            'embedding_similarity': float(np.mean(stats['similarities'])) if stats['similarities'] else 0.0,
        }

    # Print summary
    print(f"\nOverall:")
    print(f"  Items: {summary['overall']['num_items']}")
    print(f"  Accuracy: {summary['overall']['accuracy']:.4f}")
    if summary['overall']['embedding_similarity'] > 0:
        print(f"  Embedding Similarity: {summary['overall']['embedding_similarity']:.4f}")

    print(f"\nBy Source:")
    for source, stats in summary['by_source'].items():
        print(f"  {source}: {stats['num_items']} items, {stats['accuracy']:.4f} accuracy")

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Process and combine results for a specific dataset and condition"
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mmstar', 'spubench', 'vqa1k', 'vqa5k'],
                        help='Dataset name')
    parser.add_argument('--condition', type=str, default='',
                        help='Condition (e.g., inst_blind, blind, sys_inst_blind, or empty for no condition)')
    parser.add_argument('--models', nargs='*', default=[],
                        help='Paths to model result files')
    parser.add_argument('--humans', nargs='*', default=[],
                        help='Paths to human result files')
    parser.add_argument('--finetuned', nargs='*', default=[],
                        help='Paths to finetuned model result files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for combined results')
    parser.add_argument('--use_encoder', action='store_true',
                        help='Use sentence encoder for VQA embedding similarity')

    args = parser.parse_args()

    # Collect all file paths
    all_files = []
    all_files.extend(args.models)
    all_files.extend(args.humans)
    all_files.extend(args.finetuned)

    if not all_files:
        print("⚠️  No input files specified!")
        return

    # Load encoder if needed
    encoder = None
    if args.use_encoder and args.dataset in ['vqa1k', 'vqa5k']:
        print("Loading sentence transformer...")
        from evaluate import get_encoder
        encoder = get_encoder()

    # Process files
    combined_results, summary = process_files(
        all_files,
        args.dataset,
        args.condition,
        encoder=encoder,
    )

    # Save combined results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filename: dataset_condition_combined.jsonl
    condition_str = f"_{args.condition}" if args.condition else ""
    combined_file = output_dir / f"{args.dataset}{condition_str}.jsonl"
    save_jsonl(combined_results, str(combined_file))
    print(f"\n✓ Combined results saved: {combined_file}")

    # Save summary
    summary_file = output_dir / f"{args.dataset}{condition_str}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {summary_file}")

    print(f"\n{'='*80}")
    print("✅ Processing complete")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
