#!/usr/bin/env python3
"""
K-fold (or single-fold) wrapper for HUMAN-ALIGNMENT TRAINING (JS divergence).

This script calls:
  - train_human_alignment.py

It is JS-only (no standard SFT).
"""

import argparse
import json
import subprocess
from pathlib import Path
import sys
import os
from typing import Dict, List


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_kfold_metadata(kfold_dir: str) -> Dict:
    metadata_path = Path(kfold_dir) / "kfold_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    with open(metadata_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------

def train_single_fold(
    fold_idx: int,
    kfold_dir: str,
    model_path: str,
    output_base_dir: str,
    run_name: str,
    gpu_id: int = 0,
    **train_kwargs,
) -> bool:
    kfold_path = Path(kfold_dir)

    train_path = kfold_path / f"fold_{fold_idx}_train.jsonl"
    val_path = kfold_path / f"fold_{fold_idx}_val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")

    output_dir = Path(output_base_dir) / f"fold_{fold_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"JS Alignment Training — Fold {fold_idx}")
    print("=" * 80)
    print(f"Train:  {train_path}")
    print(f"Val:    {val_path}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    cmd = [
        "python", "train_human_alignment.py",
        "--model_path", model_path,
        "--data_path", str(train_path),
        "--val_data_path", str(val_path),
        "--output_dir", str(output_dir),
        "--run_name", f"{run_name}/fold_{fold_idx}",
        "--mode", "JS",
    ]

    # Append remaining JS trainer args
    for k, v in train_kwargs.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print("Command:")
    print(" ".join(cmd))
    print(f"GPU: {gpu_id}")
    print()

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"❌ Fold {fold_idx} FAILED")
        return False

    print(f"✅ Fold {fold_idx} completed successfully")
    return True


# ---------------------------------------------------------------------
# K-fold runner
# ---------------------------------------------------------------------

def run_kfold_training(
    kfold_dir: str,
    model_path: str,
    output_base_dir: str,
    run_name: str,
    folds: List[int] = None,
    gpu_id: int = 0,
    **train_kwargs,
):
    metadata = load_kfold_metadata(kfold_dir)
    k_folds = metadata["k_folds"]

    if folds is None:
        folds = list(range(k_folds))

    print("=" * 80)
    print("Human Alignment Training (JS Divergence)")
    print("=" * 80)
    print(f"Dataset: {metadata['input_path']}")
    print(f"Model:   {model_path}")
    print(f"Folds:   {folds}")
    print("=" * 80)

    results = {}

    for fold_idx in folds:
        success = train_single_fold(
            fold_idx=fold_idx,
            kfold_dir=kfold_dir,
            model_path=model_path,
            output_base_dir=output_base_dir,
            run_name=run_name,
            gpu_id=gpu_id,
            **train_kwargs,
        )
        results[f"fold_{fold_idx}"] = "success" if success else "failed"
        if not success:
            break

    summary = {
        "trainer": "human_alignment_js",
        "model": model_path,
        "kfold_dir": kfold_dir,
        "folds": folds,
        "results": results,
        "train_kwargs": train_kwargs,
    }

    summary_path = Path(output_base_dir) / "kfold_js_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("K-Fold Summary")
    print("=" * 80)
    for k, v in results.items():
        print(f"{k}: {v}")
    print(f"Saved summary → {summary_path}")
    print("=" * 80)

    return all(v == "success" for v in results.values())


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("K-fold JS alignment training")

    parser.add_argument("--kfold_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_base_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--gpu_id", type=int, default=0)

    # JS trainer arguments (must match train_human_alignment.py)
    parser.add_argument("--lambda_dist", type=float, default=1.0)
    parser.add_argument("--lambda_l2", type=float, default=0.1)
    parser.add_argument("--use_l2_penalty", action="store_true")
    parser.add_argument("--use_sft_loss", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=40)
    parser.add_argument("--eval_steps", type=int, default=40)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--max_pixels", type=int, default=448)

    args = parser.parse_args()

    kfold_keys = {
        "kfold_dir", "model_path", "output_base_dir",
        "run_name", "folds", "gpu_id"
    }

    train_kwargs = {
        k: v for k, v in vars(args).items()
        if k not in kfold_keys
    }

    success = run_kfold_training(
        kfold_dir=args.kfold_dir,
        model_path=args.model_path,
        output_base_dir=args.output_base_dir,
        run_name=args.run_name,
        folds=args.folds,
        gpu_id=args.gpu_id,
        **train_kwargs,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
