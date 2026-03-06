#!/usr/bin/env python3
"""
K-fold cross-validation training wrapper for standard SFT training.

Runs standard SFT training on each fold and aggregates results.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List
import sys
import os


def load_kfold_metadata(kfold_dir: str) -> Dict:
    """Load k-fold metadata."""
    metadata_path = Path(kfold_dir) / "kfold_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        return json.load(f)

def train_single_fold(
    fold_idx: int,
    kfold_dir: str,
    model_path: str,
    output_base_dir: str,
    run_name: str,
    learning_rate: float = 2e-5,
    num_epochs: int = 10,
    max_steps: int = -1,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 40,
    eval_steps: int = 40,
    logging_steps: int = 20,
    max_pixels: int = 448,
    gpu_id: int = 0,
):
    """Train a single fold with standard SFT."""
    kfold_path = Path(kfold_dir)

    # Get train and val paths
    train_path = kfold_path / f"fold_{fold_idx}_train.jsonl"
    val_path = kfold_path / f"fold_{fold_idx}_val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")

    # Output directory for this fold
    output_dir = Path(output_base_dir) / f"fold_{fold_idx}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Training Fold {fold_idx} (Standard SFT)")
    print(f"{'=' * 80}")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 80}\n")

    # Build command - call train_sft_standard.py
    cmd = [
        "python", "train_sft_standard.py",
        "--model_path", model_path,
        "--data_path", str(train_path),
        "--val_data_path", str(val_path),
        "--output_dir", str(output_dir),
        "--run_name", f"{run_name}/fold_{fold_idx}",
        "--learning_rate", str(learning_rate),
        "--num_epochs", str(num_epochs),
        "--max_steps", str(max_steps),
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--save_steps", str(save_steps),
        "--eval_steps", str(eval_steps),
        "--logging_steps", str(logging_steps),
        "--max_pixels", str(max_pixels),
    ]

    # Set GPU
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    # Run training
    print(f"Command: {' '.join(cmd)}")
    print(f"GPU: {gpu_id}\n")

    result = subprocess.run(cmd, env={**subprocess.os.environ, **env})

    if result.returncode != 0:
        print(f"\n❌ Fold {fold_idx} training failed!")
        return False
    else:
        print(f"\n✅ Fold {fold_idx} training completed successfully!")
        return True


def run_kfold_training(
    kfold_dir: str,
    model_path: str,
    output_base_dir: str,
    run_name: str,
    folds: List[int] = None,
    **train_kwargs
):
    """
    Run k-fold cross-validation training with standard SFT.

    Args:
        kfold_dir: Directory containing k-fold splits
        model_path: Model to train
        output_base_dir: Base output directory (will create fold_i subdirs)
        run_name: Experiment name
        folds: List of fold indices to train (None = all folds)
        **train_kwargs: Training hyperparameters
    """
    # Load metadata
    metadata = load_kfold_metadata(kfold_dir)
    k_folds = metadata['k_folds']

    print(f"\n{'=' * 80}")
    print(f"K-Fold Cross-Validation Training (Standard SFT)")
    print(f"{'=' * 80}")
    print(f"  Dataset: {metadata['input_path']}")
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  K-folds: {k_folds}")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_base_dir}")
    print(f"{'=' * 80}\n")

    # Determine which folds to train
    if folds is None:
        folds = list(range(k_folds))
    print(f"Training folds: {folds}\n")

    # Train each fold
    results = {}
    for fold_idx in folds:
        success = train_single_fold(
            fold_idx=fold_idx,
            kfold_dir=kfold_dir,
            model_path=model_path,
            output_base_dir=output_base_dir,
            run_name=run_name,
            **train_kwargs
        )

        results[f"fold_{fold_idx}"] = "success" if success else "failed"

        if not success:
            print(f"\n⚠️  Stopping k-fold training due to failure in fold {fold_idx}")
            break

    # Save results summary
    summary_path = Path(output_base_dir) / "kfold_training_summary.json"
    summary = {
        'model': model_path,
        'kfold_dir': kfold_dir,
        'k_folds': k_folds,
        'trained_folds': folds,
        'results': results,
        'training_params': train_kwargs,
        'training_type': 'standard_sft',
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"K-Fold Training Summary (SFT)")
    print(f"{'=' * 80}")
    for fold, status in results.items():
        icon = "✅" if status == "success" else "❌"
        print(f"  {icon} {fold}: {status}")
    print(f"\nSummary saved: {summary_path}")
    print(f"{'=' * 80}\n")

    # Return success if all folds succeeded
    return all(status == "success" for status in results.values())


def main():
    parser = argparse.ArgumentParser(description="K-fold cross-validation training (Standard SFT)")

    # K-fold settings
    parser.add_argument("--kfold_dir", type=str, required=True,
                        help="Directory containing k-fold splits")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Model to train")
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="Base output directory (will create fold_i subdirs)")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--folds", type=int, nargs='+', default=None,
                        help="Specific folds to train (default: all)")

    # Training hyperparameters (SFT specific)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=40)
    parser.add_argument("--eval_steps", type=int, default=40)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--max_pixels", type=int, default=448)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    # Extract k-fold settings
    kfold_args = {
        'kfold_dir': args.kfold_dir,
        'model_path': args.model_path,
        'output_base_dir': args.output_base_dir,
        'run_name': args.run_name,
        'folds': args.folds,
    }

    # Define keys to exclude (use explicit key list instead of dictionary)
    kfold_keys = {'kfold_dir', 'model_path', 'output_base_dir', 'run_name', 'folds'}

    # Extract training hyperparameters
    train_kwargs = {k: v for k, v in vars(args).items() if k not in kfold_keys}

    # Run k-fold training
    success = run_kfold_training(**kfold_args, **train_kwargs)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
