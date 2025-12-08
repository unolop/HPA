#!/usr/bin/env python3
"""
Process all result files in data/(models/humans/finetuned) directories.
Calculate VQA accuracy and embedding scores, save to processed directory.
Generate summary statistics for all results.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
from tqdm import tqdm
import pandas as pd
import numpy as np

# Import scoring functions from evaluate.py
from evaluate import (
    get_encoder,
    normalize_answer,
    extract_mc_choice,
    vqa_accuracy,
    exact_match,
    mc_accuracy,
    answer_similarity,
)


# Dataset type mappings
DATASET_TYPE = {
    'mmstar': 'multi-choice',
    'spubench': 'multi-choice',
    'vqa_1k': 'open-ended',
    'vqa_5k': 'open-ended',
    'vqa1k': 'open-ended',
    'vqa5k': 'open-ended',
}


def parse_filename(filepath: str) -> Dict[str, str]:
    """
    Parse filename to extract metadata.

    Returns:
        {
            'source': 'models'|'humans'|'finetuned',
            'model': model name,
            'dataset': dataset name,
            'condition': condition string,
            'training_method': training method (for finetuned),
        }
    """
    filepath = str(filepath)
    parts = filepath.split('/')

    # Determine source
    if '/humans/' in filepath:
        source = 'humans'
    elif '/finetuned/' in filepath:
        source = 'finetuned'
    elif '/models/' in filepath:
        source = 'models'
    else:
        source = 'unknown'

    filename = Path(filepath).stem

    # Extract condition
    conditions = ['sys_inst_blind', 'inst_blind', 'blind', '']
    condition = ''
    for cond in conditions:
        if cond and cond in filename:
            condition = cond
            filename = filename.replace(f'_{cond}', '').replace(cond, '')
            break

    # Special handling for humans (only have blind_inst)
    if source == 'humans':
        condition = 'blind_inst' if 'blind' in filepath else condition

    # Extract model name
    model_names = [
        "InternVL3_5-8B", "InternVL3_5-4B", "InternVL3_5-2B", "InternVL3_5-1B",
        "Qwen3-VL-8B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-2B-Instruct",
        "llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-1.5-7b-hf",
    ]

    model = None
    for m in model_names:
        if m in filename:
            model = m
            filename = filename.replace(f'{m}_', '').replace(m, '')
            break

    # Extract training method (for finetuned)
    training_method = None
    if source == 'finetuned':
        methods = ['alignment_js', 'standard', 'sft']
        for method in methods:
            if method in filename:
                training_method = method
                filename = filename.replace(f'{method}_', '').replace(method, '')
                break

    # Extract dataset
    dataset = filename.strip('_')

    # Clean up dataset name
    for ds in ['mmstar', 'spubench', 'vqa_1k', 'vqa_5k', 'vqa1k', 'vqa5k']:
        if ds in dataset:
            dataset = ds.replace('_', '')  # Normalize to vqa1k, vqa5k
            break

    return {
        'source': source,
        'model': model or 'unknown',
        'dataset': dataset,
        'condition': condition,
        'training_method': training_method,
    }


def load_results_file(filepath: str) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    results.append(item)
                except json.JSONDecodeError:
                    continue
    return results


def compute_metrics_for_file(
    results: List[Dict],
    dataset_name: str,
    encoder=None,
) -> Dict[str, Any]:
    """
    Compute metrics for a result file.

    Args:
        results: List of result items
        dataset_name: Name of dataset
        encoder: Sentence transformer encoder

    Returns:
        Dictionary of metrics
    """
    dataset_type = DATASET_TYPE.get(dataset_name, 'multi-choice')

    metrics = {
        'num_samples': len(results),
        'num_correct': 0,
        'accuracy': 0.0,
        'embedding_similarity': 0.0,
        'by_category': {},
    }

    if not results:
        return metrics

    correct_count = 0
    similarities = []
    by_category = defaultdict(lambda: {
        'correct': 0,
        'total': 0,
        'similarities': [],
    })

    for item in results:
        # Get ground truth and prediction
        gt = item.get('answer', item.get('ground_truth', ''))
        pred = item.get('output', item.get('response', item.get('prediction', '')))

        # Handle different answer formats
        all_answers = item.get('all_answers', item.get('answers', []))
        if not all_answers:
            all_answers = [gt] if gt else []

        # Extract answer strings from dict format if needed
        if all_answers and isinstance(all_answers[0], dict):
            all_answers = [a.get('answer', '') for a in all_answers if 'answer' in a]

        category = item.get('category', item.get('question_type', 'Unknown'))

        # Compute accuracy
        is_correct = False
        if dataset_type == 'multi-choice':
            is_correct = mc_accuracy(gt, pred)
        else:
            # VQA accuracy
            if len(all_answers) > 1:
                vqa_acc = vqa_accuracy(all_answers, pred)
                is_correct = vqa_acc >= 0.5
            else:
                is_correct = exact_match(gt, pred)

        item['correct'] = is_correct
        correct_count += int(is_correct)
        by_category[category]['correct'] += int(is_correct)
        by_category[category]['total'] += 1

        # Compute embedding similarity
        if encoder is not None and dataset_type == 'open-ended':
            answers_for_sim = all_answers if all_answers else [gt]
            sim = answer_similarity(answers_for_sim, pred, encoder)
            similarities.append(sim)
            by_category[category]['similarities'].append(sim)
            item['embedding_similarity'] = sim

    metrics['num_correct'] = correct_count
    metrics['accuracy'] = correct_count / len(results) if results else 0.0

    if similarities:
        metrics['embedding_similarity'] = float(np.mean(similarities))

    # Per-category metrics
    for cat, cat_data in by_category.items():
        cat_acc = cat_data['correct'] / cat_data['total'] if cat_data['total'] > 0 else 0
        cat_sim = np.mean(cat_data['similarities']) if cat_data['similarities'] else 0
        metrics['by_category'][cat] = {
            'accuracy': float(cat_acc),
            'embedding_similarity': float(cat_sim),
            'num_samples': cat_data['total'],
            'num_correct': cat_data['correct'],
        }

    return metrics


def process_result_file(
    filepath: str,
    output_dir: str,
    encoder=None,
) -> Dict[str, Any]:
    """
    Process a single result file.

    Args:
        filepath: Path to result file
        output_dir: Output directory for processed files
        encoder: Sentence transformer encoder

    Returns:
        Dictionary with metadata and metrics
    """
    # Parse filename
    metadata = parse_filename(filepath)

    # Load results
    results = load_results_file(filepath)

    if not results:
        print(f"⚠ Empty file: {filepath}")
        return None

    # Compute metrics
    metrics = compute_metrics_for_file(results, metadata['dataset'], encoder)

    # Save processed results
    relative_path = Path(filepath).relative_to('/home/work/yuna/HPA/data')
    output_path = Path(output_dir) / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Return summary
    return {
        'filepath': str(filepath),
        'output_path': str(output_path),
        **metadata,
        'metrics': metrics,
    }


def generate_summary_stats(all_results: List[Dict], output_dir: str):
    """
    Generate summary statistics file.

    Args:
        all_results: List of processed result summaries
        output_dir: Output directory
    """
    # Create summary DataFrame
    rows = []
    for result in all_results:
        if result is None:
            continue

        row = {
            'source': result['source'],
            'model': result['model'],
            'dataset': result['dataset'],
            'condition': result['condition'],
            'training_method': result.get('training_method', ''),
            'num_samples': result['metrics']['num_samples'],
            'num_correct': result['metrics']['num_correct'],
            'accuracy': result['metrics']['accuracy'],
            'embedding_similarity': result['metrics']['embedding_similarity'],
            'filepath': result['filepath'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save main summary
    summary_path = Path(output_dir) / 'summary_stats.csv'
    df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved: {summary_path}")

    # Create pivot tables
    print("\n" + "=" * 80)
    print("📊 SUMMARY BY SOURCE, MODEL, DATASET, CONDITION")
    print("=" * 80)

    # Group by source
    print("\n## By Source:")
    source_summary = df.groupby('source').agg({
        'num_samples': 'sum',
        'num_correct': 'sum',
        'accuracy': 'mean',
        'embedding_similarity': 'mean',
    })
    print(source_summary)

    # Group by model and dataset
    print("\n## By Model and Dataset:")
    model_dataset_summary = df.groupby(['source', 'model', 'dataset', 'condition']).agg({
        'num_samples': 'sum',
        'accuracy': 'mean',
        'embedding_similarity': 'mean',
    }).round(4)
    print(model_dataset_summary)

    # Save detailed summary
    detailed_summary = {
        'overall': {
            'total_files': len(all_results),
            'total_samples': int(df['num_samples'].sum()),
            'overall_accuracy': float(df['num_correct'].sum() / df['num_samples'].sum()),
        },
        'by_source': source_summary.to_dict(),
        'by_model_dataset': model_dataset_summary.to_dict(),
        'all_results': all_results,
    }

    detailed_path = Path(output_dir) / 'detailed_stats.json'
    with open(detailed_path, 'w') as f:
        json.dump(detailed_summary, f, indent=2)
    print(f"\n✓ Detailed stats saved: {detailed_path}")

    # Create progress tracking CSV (like progress.ipynb)
    pivot = df.pivot_table(
        index=['source', 'model'],
        columns=['dataset', 'condition'],
        values='num_samples',
        aggfunc='sum'
    )

    progress_path = Path(output_dir) / 'progress_summary.csv'
    pivot.to_csv(progress_path)
    print(f"✓ Progress summary saved: {progress_path}")

    return detailed_summary


def main():
    """Main processing pipeline."""
    print("=" * 80)
    print("📊 PROCESSING ALL RESULT FILES")
    print("=" * 80)

    # Setup paths
    data_dir = Path('/home/work/yuna/HPA/data')
    output_dir = Path('/home/work/yuna/HPA/data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all result files
    result_files = []
    for source_dir in ['models', 'finetuned', 'humans']:
        source_path = data_dir / source_dir
        if source_path.exists():
            # Find all .jsonl files recursively
            files = list(source_path.rglob('*.jsonl'))
            result_files.extend(files)
            print(f"Found {len(files)} files in {source_dir}/")

    print(f"\nTotal files to process: {len(result_files)}")

    if not result_files:
        print("⚠ No result files found!")
        return

    # Load encoder for embedding similarity
    print("\n📦 Loading sentence transformer...")
    encoder = get_encoder()

    # Process each file
    print("\n" + "=" * 80)
    print("Processing files...")
    print("=" * 80)

    all_results = []
    for filepath in tqdm(result_files, desc="Processing"):
        try:
            result = process_result_file(str(filepath), str(output_dir), encoder)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n⚠ Error processing {filepath}: {e}")
            continue

    print(f"\n✓ Successfully processed {len(all_results)} files")

    # Generate summary statistics
    print("\n" + "=" * 80)
    print("Generating summary statistics...")
    print("=" * 80)

    summary = generate_summary_stats(all_results, str(output_dir))

    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE")
    print("=" * 80)
    print(f"   Processed files: {len(all_results)}")
    print(f"   Total samples: {summary['overall']['total_samples']}")
    print(f"   Overall accuracy: {summary['overall']['overall_accuracy']:.4f}")
    print(f"   Output directory: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
