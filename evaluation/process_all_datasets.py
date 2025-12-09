#!/usr/bin/env python3
"""
Automatically find and process all datasets and conditions.
"""
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict


def find_all_files(data_dir: str) -> dict:
    """
    Find all result files organized by dataset and condition.

    Returns:
        {
            'mmstar': {
                '': [list of files with no condition],
                'inst_blind': [list of files],
                ...
            },
            'spubench': {...},
            ...
        }
    """
    data_path = Path(data_dir)
    files_by_dataset = defaultdict(lambda: defaultdict(list))

    # Search in models, humans, finetuned directories
    for source_dir in ['models', 'humans', 'finetuned']:
        source_path = data_path / source_dir
        if not source_path.exists():
            continue

        # Find all .jsonl files
        for filepath in source_path.rglob('*.jsonl'):
            filename = filepath.stem

            # Determine dataset
            dataset = None
            for ds in ['mmstar', 'spubench', 'vqa_5k', 'vqa_1k', 'vqa5k', 'vqa1k']:
                if ds in filename:
                    dataset = ds.replace('_', '')  # Normalize
                    break

            if not dataset:
                continue

            # Determine condition
            condition = ''
            for cond in ['sys_inst_blind', 'inst_blind', 'blind']:
                if cond in filename:
                    condition = cond
                    break

            # Special handling for humans
            if source_dir == 'humans' and 'blind' in str(filepath):
                condition = 'blind_inst'

            files_by_dataset[dataset][condition].append(str(filepath))

    return files_by_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Automatically process all datasets and conditions"
    )
    parser.add_argument('--data_dir', type=str, default='/home/work/yuna/HPA/data',
                        help='Base data directory')
    parser.add_argument('--output_dir', type=str, default='/home/work/yuna/HPA/data/combined',
                        help='Output directory for combined results')
    parser.add_argument('--use_encoder', action='store_true',
                        help='Use sentence encoder for VQA datasets')

    args = parser.parse_args()

    print("=" * 80)
    print("🔍 Finding all result files...")
    print("=" * 80)

    files_by_dataset = find_all_files(args.data_dir)

    # Print summary
    for dataset, conditions in sorted(files_by_dataset.items()):
        print(f"\n{dataset}:")
        for condition, files in sorted(conditions.items()):
            cond_label = condition or '(no condition)'
            print(f"  {cond_label}: {len(files)} files")

    print(f"\n{'='*80}")
    print("🚀 Processing all datasets...")
    print(f"{'='*80}\n")

    # Process each dataset/condition combination
    for dataset, conditions in sorted(files_by_dataset.items()):
        for condition, files in sorted(conditions.items()):
            print(f"\n{'='*80}")
            print(f"Processing: {dataset} - {condition or '(no condition)'}")
            print(f"{'='*80}")

            # Build command
            cmd = [
                'python', 'process_dataset.py',
                '--dataset', dataset,
                '--output_dir', args.output_dir,
            ]

            if condition:
                cmd.extend(['--condition', condition])

            # Add files by source
            models = [f for f in files if '/models/' in f]
            humans = [f for f in files if '/humans/' in f]
            finetuned = [f for f in files if '/finetuned/' in f]

            if models:
                cmd.extend(['--models'] + models)
            if humans:
                cmd.extend(['--humans'] + humans)
            if finetuned:
                cmd.extend(['--finetuned'] + finetuned)

            if args.use_encoder and dataset in ['vqa1k', 'vqa5k']:
                cmd.append('--use_encoder')

            # Run processing
            result = subprocess.run(cmd, cwd=Path(__file__).parent)
            if result.returncode != 0:
                print(f"⚠️  Error processing {dataset} - {condition}")

    print(f"\n{'='*80}")
    print("✅ All datasets processed!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
