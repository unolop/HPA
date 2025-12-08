#!/usr/bin/env python3
"""
Map VQA categories (answer_type, question_type) to result files using qid.

For VQA datasets, category information may not be in the result files.
This script maps categories from the original VQA dataset to processed results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def load_vqa_categories(dataset_name: str) -> Dict[str, Dict]:
    """
    Load VQA category mappings from dataset.

    Args:
        dataset_name: 'vqa_1k' or 'vqa_5k' or 'vqa1k' or 'vqa5k'

    Returns:
        Dict mapping qid -> {'answer_type': ..., 'question_type': ...}
    """
    categories = {}

    # Normalize dataset name
    dataset_name = dataset_name.replace('_', '').lower()

    if dataset_name == 'vqa1k':
        # Load from vqav2_1k_val.json
        vqa_path = '/home/user/HPA/dataset/vqav2_1k_val.json'
        if Path(vqa_path).exists():
            with open(vqa_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    qid = str(item.get('question_id', ''))
                    categories[qid] = {
                        'answer_type': item.get('answer_type', ''),
                        'question_type': item.get('question_type', ''),
                    }
            print(f"✓ Loaded {len(categories)} categories from vqa1k")
        else:
            print(f"⚠ VQA 1k dataset not found at {vqa_path}")

    elif dataset_name == 'vqa5k':
        # Load from VQA dataset (would need to be generated if not exists)
        # For now, try to load from s1_vqa.jsonl if exists
        vqa_path = '/home/user/HPA/dataset/s1_vqa.jsonl'
        if Path(vqa_path).exists():
            with open(vqa_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    qid = str(item.get('question_id', ''))
                    categories[qid] = {
                        'answer_type': item.get('answer_type', ''),
                        'question_type': item.get('question_type', ''),
                    }
            print(f"✓ Loaded {len(categories)} categories from vqa5k")
        else:
            print(f"⚠ VQA 5k dataset not found at {vqa_path}")
            print("   Categories will need to be generated from VQA dataset")

    return categories


def map_categories_to_file(
    result_file: str,
    category_map: Dict[str, Dict],
    output_file: str = None,
) -> int:
    """
    Map categories to a result file.

    Args:
        result_file: Path to result JSONL file
        category_map: Dict mapping qid -> category info
        output_file: Output path (overwrites input if None)

    Returns:
        Number of items updated
    """
    if output_file is None:
        output_file = result_file

    # Load results
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Map categories
    updated_count = 0
    for item in results:
        qid = str(item.get('qid', item.get('question_id', '')))

        if qid in category_map:
            # Add category info if not already present
            if 'answer_type' not in item or not item.get('answer_type'):
                item['answer_type'] = category_map[qid]['answer_type']
                updated_count += 1

            if 'question_type' not in item or not item.get('question_type'):
                item['question_type'] = category_map[qid]['question_type']

    # Save updated results
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return updated_count


def process_all_vqa_files(processed_dir: str = '/home/user/HPA/data/processed'):
    """
    Process all VQA result files in the processed directory.
    """
    processed_path = Path(processed_dir)

    # Load category mappings
    print("📂 Loading VQA category mappings...")
    vqa1k_categories = load_vqa_categories('vqa1k')
    vqa5k_categories = load_vqa_categories('vqa5k')

    # Find all VQA result files
    vqa_files = []
    for pattern in ['*vqa*1k*.jsonl', '*vqa*5k*.jsonl', '*vqa_1k*.jsonl', '*vqa_5k*.jsonl']:
        vqa_files.extend(processed_path.rglob(pattern))

    print(f"\nFound {len(vqa_files)} VQA result files")

    if not vqa_files:
        print("⚠ No VQA files found")
        return

    # Process each file
    print("\n" + "=" * 80)
    print("Mapping categories to VQA files...")
    print("=" * 80)

    total_updated = 0
    for filepath in tqdm(vqa_files, desc="Processing"):
        # Determine which category map to use
        filename = filepath.name.lower()

        if 'vqa1k' in filename or 'vqa_1k' in filename:
            category_map = vqa1k_categories
        elif 'vqa5k' in filename or 'vqa_5k' in filename:
            category_map = vqa5k_categories
        else:
            continue

        if not category_map:
            print(f"⚠ No category map available for {filepath.name}")
            continue

        try:
            updated = map_categories_to_file(str(filepath), category_map)
            total_updated += updated
        except Exception as e:
            print(f"\n⚠ Error processing {filepath}: {e}")

    print(f"\n✓ Updated {total_updated} items across {len(vqa_files)} files")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Map VQA categories to result files by qid"
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Single file to process (optional)'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default='/home/user/HPA/data/processed',
        help='Processed results directory'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("📊 MAPPING VQA CATEGORIES")
    print("=" * 80)

    if args.file:
        # Process single file
        dataset = 'vqa1k' if 'vqa1k' in args.file or 'vqa_1k' in args.file else 'vqa5k'
        categories = load_vqa_categories(dataset)

        if categories:
            updated = map_categories_to_file(args.file, categories)
            print(f"✓ Updated {updated} items in {args.file}")
        else:
            print(f"⚠ No categories loaded for {dataset}")
    else:
        # Process all files
        process_all_vqa_files(args.processed_dir)

    print("\n" + "=" * 80)
    print("✅ CATEGORY MAPPING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
