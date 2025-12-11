#!/usr/bin/env python3
"""
Single-split training using existing k-fold splits.

Instead of training on all k folds, use:
- K-1 folds combined as training data
- 1 fold as validation data

This provides a single train/val split per dataset without creating new splits.
"""

import argparse
import json
from pathlib import Path
from typing import List


def combine_folds_for_single_split(
    kfold_dir: str,
    val_fold_idx: int,
    output_train_path: str,
    output_val_path: str
):
    """
    Combine k-fold splits into a single train/val split.

    Args:
        kfold_dir: Directory containing k-fold splits
        val_fold_idx: Which fold to use as validation (others become training)
        output_train_path: Where to save combined training data
        output_val_path: Where to save validation data
    """
    kfold_path = Path(kfold_dir)

    # Load metadata to get number of folds
    metadata_path = kfold_path / "kfold_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    k_folds = metadata['k_folds']

    if not (0 <= val_fold_idx < k_folds):
        raise ValueError(f"val_fold_idx must be in range [0, {k_folds}), got {val_fold_idx}")

    print(f"\n{'=' * 80}")
    print(f"Creating Single Train/Val Split from K-Fold Data")
    print(f"{'=' * 80}")
    print(f"  K-fold dir: {kfold_dir}")
    print(f"  Total folds: {k_folds}")
    print(f"  Validation fold: {val_fold_idx}")
    print(f"  Training folds: {[i for i in range(k_folds) if i != val_fold_idx]}")
    print(f"{'=' * 80}\n")

    # Read validation fold
    val_fold_path = kfold_path / f"fold_{val_fold_idx}_train.jsonl"
    if not val_fold_path.exists():
        raise FileNotFoundError(f"Validation fold not found: {val_fold_path}")

    with open(val_fold_path, 'r') as f:
        val_data = [json.loads(line) for line in f if line.strip()]

    print(f"✅ Validation data: {len(val_data)} samples from fold_{val_fold_idx}_train.jsonl")

    # Combine all other folds for training
    train_data = []
    for fold_idx in range(k_folds):
        if fold_idx == val_fold_idx:
            continue  # Skip validation fold

        train_fold_path = kfold_path / f"fold_{fold_idx}_train.jsonl"
        if not train_fold_path.exists():
            print(f"⚠️  Warning: {train_fold_path} not found, skipping")
            continue

        with open(train_fold_path, 'r') as f:
            fold_data = [json.loads(line) for line in f if line.strip()]

        train_data.extend(fold_data)
        print(f"✅ Added fold_{fold_idx}_train.jsonl: {len(fold_data)} samples")

    print(f"\n📊 Total training samples: {len(train_data)}")
    print(f"📊 Total validation samples: {len(val_data)}")
    print(f"📊 Train/Val ratio: {len(train_data)/len(val_data):.2f}\n")

    # Save combined data
    Path(output_train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_val_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(output_val_path, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Training data saved: {output_train_path}")
    print(f"✅ Validation data saved: {output_val_path}")
    print(f"{'=' * 80}\n")

    return {
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'train_folds': [i for i in range(k_folds) if i != val_fold_idx],
        'val_fold': val_fold_idx,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create single train/val split from k-fold data"
    )
    parser.add_argument("--kfold_dir", type=str, required=True,
                        help="Directory containing k-fold splits")
    parser.add_argument("--val_fold_idx", type=int, default=0,
                        help="Which fold to use as validation (default: 0)")
    parser.add_argument("--output_train_path", type=str, required=True,
                        help="Where to save combined training data")
    parser.add_argument("--output_val_path", type=str, required=True,
                        help="Where to save validation data")

    args = parser.parse_args()

    combine_folds_for_single_split(
        kfold_dir=args.kfold_dir,
        val_fold_idx=args.val_fold_idx,
        output_train_path=args.output_train_path,
        output_val_path=args.output_val_path
    )


if __name__ == "__main__":
    main()
