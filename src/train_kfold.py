#!/usr/bin/env python3
"""
K-fold cross-validation training wrapper.

Runs training on each fold and aggregates results.
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
    mode: str = "JS",
    lambda_dist: float = 1.0,
    lambda_l2: float = 0.1,
    use_l2_penalty: bool = True,
    use_sft_loss: bool = False,
    learning_rate: float = 2e-5,
    num_epochs: int = 10,
    max_steps: int = -1,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 40,
    eval_steps: int = 40,
    logging_steps: int = 20,
    max_pixels: int = 448,
    gpu_id: int = 0,
):
    """Train a single fold."""
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
    print(f"Training Fold {fold_idx}")
    print(f"{'=' * 80}")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 80}\n")

    # Build command - call train_js.py (has argparse main)
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    train_js_path = script_dir / "train_js.py"

    cmd = [
        "python", str(train_js_path),
        "--model_path", model_path,
        "--data_path", str(train_path),
        "--val_data_path", str(val_path),
        "--output_dir", str(output_dir),
        "--run_name", f"{run_name}/fold_{fold_idx}",
        "--mode", mode,
        "--lambda_dist", str(lambda_dist),
        "--lambda_l2", str(lambda_l2),
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

    if use_l2_penalty:
        cmd.append("--use_l2_penalty")

    if use_sft_loss:
        cmd.append("--use_sft_loss")

    # Set GPU
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    # Run training
    print(f"\n{'=' * 80}")
    print(f"EXECUTING TRAINING COMMAND:")
    print(f"{'=' * 80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu_id}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script path: {train_js_path}")
    print(f"Script exists: {train_js_path.exists()}")
    print(f"{'=' * 80}\n")

    # Run subprocess with output streaming to console
    try:
        result = subprocess.run(
            cmd,
            env={**os.environ, **env},
            check=False  # Don't raise exception, we'll handle return code
        )

        print(f"\n{'=' * 80}")
        print(f"Training subprocess exited with code: {result.returncode}")
        print(f"{'=' * 80}\n")

        if result.returncode != 0:
            print(f"\n❌ Fold {fold_idx} training failed with return code {result.returncode}!")
            return False
        else:
            print(f"\n✅ Fold {fold_idx} training completed successfully!")
            return True

    except Exception as e:
        print(f"\n❌ Exception while running training subprocess: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_kfold_training(
    kfold_dir: str,
    model_path: str,
    output_base_dir: str,
    run_name: str,
    folds: List[int] = None,
    num_folds: int = None,
    **train_kwargs
):
    """
    Run k-fold cross-validation training.

    Args:
        kfold_dir: Directory containing k-fold splits
        model_path: Model to train
        output_base_dir: Base output directory (will create fold_i subdirs)
        run_name: Experiment name
        folds: List of fold indices to train (None = all folds)
        num_folds: Number of folds to train (None = all folds, 1 = first fold only)
        **train_kwargs: Training hyperparameters
    """
    # Load metadata
    metadata = load_kfold_metadata(kfold_dir)
    k_folds = metadata['k_folds']

    print(f"\n{'=' * 80}")
    print(f"K-Fold Cross-Validation Training")
    print(f"{'=' * 80}")
    print(f"  Dataset: {metadata['input_path']}")
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  K-folds available: {k_folds}")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_base_dir}")
    print(f"{'=' * 80}\n")

    # Determine which folds to train
    if folds is None:
        if num_folds is not None and num_folds > 0:
            # Train only the first num_folds
            folds = list(range(min(num_folds, k_folds)))
            print(f"Training first {num_folds} fold(s) out of {k_folds} available")
        else:
            # Train all folds
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
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"K-Fold Training Summary")
    print(f"{'=' * 80}")
    for fold, status in results.items():
        icon = "✅" if status == "success" else "❌"
        print(f"  {icon} {fold}: {status}")
    print(f"\nSummary saved: {summary_path}")
    print(f"{'=' * 80}\n")

    # Return success if all folds succeeded
    return all(status == "success" for status in results.values())


def main():
    parser = argparse.ArgumentParser(description="K-fold cross-validation training")

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
    parser.add_argument("--num_folds", type=int, default=None,
                        help="Number of folds to train (1=first fold only, None=all folds)")

    # Training hyperparameters
    parser.add_argument("--mode", type=str, default="JS", choices=["CE", "JS"])
    parser.add_argument("--lambda_dist", type=float, default=1.0)
    parser.add_argument("--lambda_l2", type=float, default=0.1)
    parser.add_argument("--use_l2_penalty", action="store_true", default=True)
    parser.add_argument("--use_sft_loss", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
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
        'num_folds': args.num_folds,
    }

    # Define keys to exclude (use explicit key list instead of dictionary)
    kfold_keys = {'kfold_dir', 'model_path', 'output_base_dir', 'run_name', 'folds', 'num_folds'}

    # Extract training hyperparameters
    train_kwargs = {k: v for k, v in vars(args).items() if k not in kfold_keys}

    # Debug: Print argument distributions
    print(f"\n{'=' * 80}")
    print("Argument Distribution:")
    print(f"{'=' * 80}")
    print(f"K-fold args: {list(kfold_args.keys())}")
    print(f"  model_path: {kfold_args['model_path']}")
    print(f"  output_base_dir: {kfold_args['output_base_dir']}")
    print(f"\nTraining kwargs: {list(train_kwargs.keys())}")
    if 'gpu_id' in train_kwargs:
        print(f"  gpu_id: {train_kwargs['gpu_id']}")
    if 'model_path' in train_kwargs:
        print(f"  WARNING: model_path in train_kwargs: {train_kwargs['model_path']}")
    print(f"{'=' * 80}\n")

    # Run k-fold training
    success = run_kfold_training(**kfold_args, **train_kwargs)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
