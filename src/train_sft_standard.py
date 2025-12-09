#!/usr/bin/env python3
"""
Standard SFT Training for Visual Grounding
Uses CE loss on conversation data (normal SFT training step)
"""

import torch
import argparse
from dataclasses import dataclass, field

from swift.llm import sft_main, TrainArguments
from swift.utils import get_logger

logger = get_logger()


@dataclass
class SFTArguments(TrainArguments):
    """Standard SFT training arguments."""
    pass


def train_sft(
    model_path: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    val_data_path: str = None,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_steps: int = -1,  # Set to -1 to use num_epochs instead
    lora_rank: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 40,
    eval_steps: int = 40,
    logging_steps: int = 20,
    max_pixels: int = 448,
):
    """Standard SFT training with CE loss on conversations."""

    logger.info("=" * 80)
    logger.info("🚀 Standard SFT Training")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Data: {data_path}")
    logger.info("=" * 80)

    # Warn if both max_steps and num_epochs are set
    if max_steps > 0:
        logger.warning(f"⚠️  Both max_steps ({max_steps}) and num_epochs ({num_epochs}) are set.")
        logger.warning(f"⚠️  Training will run for {max_steps} steps and IGNORE num_epochs.")
        logger.warning(f"⚠️  To use num_epochs, set max_steps=-1 instead.")
    else:
        logger.info(f"📊 Training for {num_epochs} epochs (max_steps={max_steps})")

    # Handle None/empty string for val_data_path
    if val_data_path is not None and (val_data_path.lower() == 'none' or val_data_path.strip() == ''):
        val_data_path = None

    # Swift framework expects a list (can be empty) for val_dataset, not None
    if val_data_path is not None:
        val_dataset = [val_data_path]
    else:
        val_dataset = []  # Empty list instead of None to avoid len() error

    sft_args = SFTArguments(
        model=model_path,
        dataset=[data_path],
        val_dataset=val_dataset,
        output_dir=output_dir,
        
        train_type="lora",
        torch_dtype="bfloat16",
        
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.05,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        lora_bias="none",
        target_modules=["all-linear"],
        
        freeze_llm=False,
        freeze_vit=True,
        freeze_aligner=True,
        
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        
        max_length=1024,
        max_pixels=max_pixels,
        
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        seed=42,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
    )
    
    return sft_main(sft_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=40)
    parser.add_argument("--eval_steps", type=int, default=40)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--max_pixels", type=int, default=448)
    
    args = parser.parse_args()
    train_sft(**vars(args))


if __name__ == "__main__":
    main()