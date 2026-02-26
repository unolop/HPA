#!/usr/bin/env python3
"""
Create k-fold cross-validation splits from JSONL datasets.

For each dataset:
- Splits data into k folds
- Each fold gets a train and val file
- Fold i uses fold i as validation, remaining as training
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import random


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save data to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def create_kfold_splits(
    input_path: str,
    output_dir: str,
    k: int = 5,
    seed: int = 42,
    shuffle: bool = True
):
    """
    Create k-fold cross-validation splits.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to save fold splits
        k: Number of folds
        seed: Random seed for shuffling
        shuffle: Whether to shuffle data before splitting
    """
    print(f"Loading data from: {input_path}")
    data = load_jsonl(input_path)
    n_samples = len(data)
    print(f"Total samples: {n_samples}")

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
        print(f"Data shuffled with seed={seed}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate fold sizes
    fold_size = n_samples // k
    remainder = n_samples % k

    print(f"\nCreating {k} folds:")
    print(f"  Base fold size: {fold_size}")
    print(f"  Remainder: {remainder}")

    # Create folds
    folds = []
    start_idx = 0
    for i in range(k):
        # First 'remainder' folds get one extra sample
        end_idx = start_idx + fold_size + (1 if i < remainder else 0)
        fold_data = data[start_idx:end_idx]
        folds.append(fold_data)
        start_idx = end_idx

    # Verify all data is accounted for
    assert sum(len(f) for f in folds) == n_samples, "Data split verification failed!"

    # Save train/val splits for each fold
    for fold_idx in range(k):
        # Validation data: current fold
        val_data = folds[fold_idx]

        # Training data: all other folds
        train_data = []
        for i in range(k):
            if i != fold_idx:
                train_data.extend(folds[i])

        # Save files
        fold_name = f"fold_{fold_idx}"
        train_path = output_dir / f"{fold_name}_train.jsonl"
        val_path = output_dir / f"{fold_name}_val.jsonl"

        save_jsonl(train_data, str(train_path))
        save_jsonl(val_data, str(val_path))

        print(f"\n  Fold {fold_idx}:")
        print(f"    Train: {len(train_data)} samples -> {train_path}")
        print(f"    Val:   {len(val_data)} samples -> {val_path}")

    # Save metadata
    metadata = {
        'input_path': str(input_path),
        'total_samples': n_samples,
        'k_folds': k,
        'seed': seed,
        'shuffle': shuffle,
        'fold_sizes': [len(f) for f in folds],
    }

    metadata_path = output_dir / "kfold_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ K-fold splits created successfully!")
    print(f"   Output directory: {output_dir}")
    print(f"   Metadata saved: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Create k-fold cross-validation splits")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save fold splits")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="Do not shuffle data before splitting")

    args = parser.parse_args()

    create_kfold_splits(
        input_path=args.input_path,
        output_dir=args.output_dir,
        k=args.k,
        seed=args.seed,
        shuffle=not args.no_shuffle
    )


if __name__ == "__main__":
    main()
