#!/usr/bin/env python3
"""
Analyze spubench model performance by spurious correlation type.

This script loads scored spubench results and breaks down accuracy by
spurious correlation type (e.g., Background, Shape, Orientation, etc.)
instead of by categories like the mmstar dataset.
"""

import os
import json
import argparse
import pandas as pd
from collections import defaultdict
from glob import glob
from typing import Dict, List


def load_annotations(annotation_path='/home/user/HPA/dataset/annotation.json'):
    """Load spubench annotations with spurious correlation types."""
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    return annotations


def analyze_by_spurious_type(scored_file_path: str, annotations: List[Dict]) -> Dict:
    """
    Analyze a scored spubench file by spurious correlation type.

    Args:
        scored_file_path: Path to scored JSONL file
        annotations: List of annotation dicts with spurious_correlation_type

    Returns:
        Dict with overall accuracy and per-type breakdown
    """
    # Load scored results
    results = []
    with open(scored_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        return {}

    # Build mapping from pid to spurious types
    pid_to_types = {}
    for ann in annotations:
        pid = ann.get('pid')
        if pid is None:
            # Use index if pid not present
            pid = annotations.index(ann)
        pid_to_types[pid] = ann.get('spurious_correlation_type', [])

    # Analyze by spurious correlation type
    by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct = 0
    overall_total = 0

    for item in results:
        pid = item.get('pid', item.get('index', item.get('idx')))
        correct = item.get('correct', False)

        # Update overall stats
        overall_correct += int(correct)
        overall_total += 1

        # Update per-type stats
        types = pid_to_types.get(pid, [])
        for spur_type in types:
            by_type[spur_type]['correct'] += int(correct)
            by_type[spur_type]['total'] += 1

    # Compute accuracies
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0

    type_accuracies = {}
    for spur_type, stats in by_type.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        type_accuracies[spur_type] = {
            'accuracy': acc,
            'correct': stats['correct'],
            'total': stats['total']
        }

    return {
        'file': os.path.basename(scored_file_path),
        'overall_accuracy': overall_acc,
        'overall_correct': overall_correct,
        'overall_total': overall_total,
        'by_spurious_type': type_accuracies
    }


def extract_model_name(path: str) -> str:
    """Extract model name from file path."""
    filename = os.path.basename(path)
    # Remove dataset name and condition
    filename = filename.replace('_spubench', '').replace('.jsonl', '')
    filename = filename.replace('_inst_blind', '').replace('_blind', '')

    # Handle finetuned models (path contains model name in directory)
    if 'finetuned' in path:
        parts = path.split('/')
        # Find the model directory
        for i, part in enumerate(parts):
            if part == 'finetuned' and i+1 < len(parts):
                model_name = parts[i+1]
                # Add training config if available
                if i+2 < len(parts):
                    training = parts[i+2]
                    return f"{model_name}/{training}"
                return model_name

    return filename


def analyze_directory(scored_dir: str, annotation_path: str = '/home/user/HPA/dataset/annotation.json'):
    """
    Analyze all spubench scored files in a directory.

    Args:
        scored_dir: Directory containing scored spubench JSONL files
        annotation_path: Path to annotation.json with spurious types
    """
    # Load annotations
    print(f"Loading annotations from {annotation_path}...")
    annotations = load_annotations(annotation_path)
    print(f"Loaded {len(annotations)} annotations\n")

    # Find all scored spubench files
    patterns = [
        f"{scored_dir}/*spubench*.jsonl",
        f"{scored_dir}/*/*spubench*.jsonl",
        f"{scored_dir}/*/*/*spubench*.jsonl"
    ]

    files = []
    for pattern in patterns:
        files.extend(glob(pattern))

    files = sorted(set(files))
    print(f"Found {len(files)} spubench files to analyze\n")

    if not files:
        print(f"No spubench files found in {scored_dir}")
        return

    # Analyze each file
    all_results = []
    for file_path in files:
        print(f"Analyzing: {os.path.basename(file_path)}")
        result = analyze_by_spurious_type(file_path, annotations)
        if result:
            result['model'] = extract_model_name(file_path)
            result['path'] = file_path
            all_results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Performance by Spurious Correlation Type")
    print("="*80)

    # Collect all spurious types
    all_types = set()
    for res in all_results:
        all_types.update(res['by_spurious_type'].keys())

    # Create summary table
    summary_data = []
    for res in all_results:
        row = {
            'model': res['model'],
            'overall': f"{res['overall_accuracy']:.3f} ({res['overall_correct']}/{res['overall_total']})"
        }

        # Add per-type accuracies
        for spur_type in sorted(all_types):
            if spur_type in res['by_spurious_type']:
                stats = res['by_spurious_type'][spur_type]
                row[spur_type] = f"{stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})"
            else:
                row[spur_type] = "N/A"

        summary_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(summary_data)

    # Reorder columns: model, overall, then types by frequency
    type_counts = defaultdict(int)
    for res in all_results:
        for spur_type, stats in res['by_spurious_type'].items():
            type_counts[spur_type] += stats['total']

    sorted_types = sorted(type_counts.keys(), key=lambda x: -type_counts[x])
    columns = ['model', 'overall'] + sorted_types
    df = df[columns]

    print(df.to_string(index=False))

    # Save to CSV
    output_path = os.path.join(scored_dir, 'spubench_by_spurious_type.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved summary to: {output_path}")

    # Also save detailed JSON
    json_output = os.path.join(scored_dir, 'spubench_by_spurious_type.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved detailed results to: {json_output}")

    # Print per-type statistics across all models
    print("\n" + "="*80)
    print("Per-Type Statistics (across all models)")
    print("="*80)

    type_stats = defaultdict(lambda: {'total_correct': 0, 'total_items': 0})
    for res in all_results:
        for spur_type, stats in res['by_spurious_type'].items():
            type_stats[spur_type]['total_correct'] += stats['correct']
            type_stats[spur_type]['total_items'] += stats['total']

    type_summary = []
    for spur_type in sorted_types:
        stats = type_stats[spur_type]
        avg_acc = stats['total_correct'] / stats['total_items'] if stats['total_items'] > 0 else 0
        type_summary.append({
            'Type': spur_type,
            'Avg Accuracy': f"{avg_acc:.3f}",
            'Total Evaluations': stats['total_items'] // len(all_results) if all_results else 0
        })

    type_df = pd.DataFrame(type_summary)
    print(type_df.to_string(index=False))


def analyze_single_file(file_path: str, annotation_path: str = '/home/user/HPA/dataset/annotation.json'):
    """Analyze a single scored spubench file."""
    print(f"Loading annotations from {annotation_path}...")
    annotations = load_annotations(annotation_path)

    print(f"\nAnalyzing: {file_path}")
    result = analyze_by_spurious_type(file_path, annotations)

    if not result:
        print("No results found!")
        return

    print("\n" + "="*60)
    print(f"Overall Accuracy: {result['overall_accuracy']:.4f} ({result['overall_correct']}/{result['overall_total']})")
    print("="*60)

    print("\nPer Spurious Correlation Type:")
    print("-"*60)

    # Sort by accuracy
    sorted_types = sorted(
        result['by_spurious_type'].items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )

    for spur_type, stats in sorted_types:
        print(f"{spur_type:25s}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze spubench results by spurious correlation type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single scored file
  python analyze_spubench_by_correlation.py --file evaluation/scored/pretrained/InternVL3_5-8B_spubench.jsonl

  # Analyze entire directory
  python analyze_spubench_by_correlation.py --dir evaluation/scored/pretrained

  # Use custom annotation file
  python analyze_spubench_by_correlation.py --dir evaluation/scored/pretrained --annotations custom_annotation.json
        """
    )

    parser.add_argument("--file", type=str, help="Single scored spubench JSONL file to analyze")
    parser.add_argument("--dir", type=str, help="Directory containing scored spubench files")
    parser.add_argument("--annotations", type=str,
                       default="/home/user/HPA/dataset/annotation.json",
                       help="Path to annotation.json file")

    args = parser.parse_args()

    if args.file:
        analyze_single_file(args.file, args.annotations)
    elif args.dir:
        analyze_directory(args.dir, args.annotations)
    else:
        print("Please specify --file or --dir")
        parser.print_help()


if __name__ == "__main__":
    main()
