#!/usr/bin/env python3
"""
Multi-Model SFT Training - Works with QwenVL, InternVL, and Llava

This script automatically adapts training configuration based on the model architecture.
"""

import torch
import argparse
from dataclasses import dataclass, field
from model_configs import get_model_config, get_data_format_config

from swift.llm import sft_main, TrainArguments
from swift.utils import get_logger

logger = get_logger()


@dataclass
class MultiModelSFTArguments(TrainArguments):
    """SFT training arguments that adapt to different model architectures."""
    pass


def train_sft(
    model_path: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    val_data_path: str = None,
    learning_rate: float = None,  # Will use model-specific default if None
    num_epochs: int = 3,
    max_steps: int = -1,
    lora_rank: int = None,  # Will use model-specific default if None
    lora_alpha: int = None,  # Will use model-specific default if None
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 20,
    max_pixels: int = None,  # Will use model-specific default if None
    resume_from_checkpoint: str = None,
    use_flash_attn: bool = True,
):
    """
    Multi-model SFT training that adapts to model architecture.

    Args:
        model_path: HuggingFace model path (Qwen/InternVL/Llava)
        All other args: Standard training parameters
    """
    logger.info("=" * 80)
    logger.info("🚀 Multi-Model SFT Training")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Data: {data_path}")
    logger.info("=" * 80)

    # Get model-specific configuration
    model_config = get_model_config(model_path)
    data_config = get_data_format_config(model_path)

    logger.info(f"\n📋 Detected Model Type:")
    model_name_lower = model_path.lower()
    if "internvl" in model_name_lower:
        model_type = "InternVL"
    elif "llava" in model_name_lower:
        model_type = "Llava"
    elif "qwen" in model_name_lower:
        model_type = "QwenVL"
    else:
        model_type = "Unknown (using defaults)"

    logger.info(f"   Type: {model_type}")
    logger.info(f"   Vision Encoder: {model_config.get('vision_encoder_name', 'Default')}")

    # Use model-specific defaults if not provided
    if learning_rate is None:
        learning_rate = model_config["learning_rate"]
        logger.info(f"   Using model-specific learning_rate: {learning_rate}")

    if lora_rank is None:
        lora_rank = model_config["lora_rank"]
        logger.info(f"   Using model-specific lora_rank: {lora_rank}")

    if lora_alpha is None:
        lora_alpha = model_config["lora_alpha"]
        logger.info(f"   Using model-specific lora_alpha: {lora_alpha}")

    if max_pixels is None:
        max_pixels = model_config["max_pixels"]
        logger.info(f"   Using model-specific max_pixels: {max_pixels}")

    # Warn about training duration
    if max_steps > 0:
        logger.warning(f"⚠️  Training for {max_steps} steps (ignoring num_epochs={num_epochs})")
    else:
        logger.info(f"📊 Training for {num_epochs} epochs")

    # Log checkpoint resumption
    if resume_from_checkpoint:
        if resume_from_checkpoint.lower() == 'auto':
            logger.info(f"🔄 Resuming from latest checkpoint in {output_dir}")
        else:
            logger.info(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")

    # Handle validation data
    if val_data_path is not None and (val_data_path.lower() == 'none' or val_data_path.strip() == ''):
        val_data_path = None

    val_dataset = [val_data_path] if val_data_path else []

    # Build training arguments with model-specific config
    sft_args = MultiModelSFTArguments(
        model=model_path,
        dataset=[data_path],
        val_dataset=val_dataset,
        output_dir=output_dir,

        # Training type
        train_type="lora",
        torch_dtype="bfloat16",

        # Training schedule
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.05,
        weight_decay=0.1,
        lr_scheduler_type="cosine",

        # LoRA configuration (model-specific)
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=model_config["lora_dropout"],
        lora_bias=model_config["lora_bias"],
        target_modules=model_config["target_modules"],

        # Freezing strategy (model-specific)
        freeze_llm=model_config["freeze_llm"],
        freeze_vit=model_config["freeze_vit"],
        freeze_aligner=model_config["freeze_aligner"],

        # Checkpointing
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        resume_from_checkpoint=resume_from_checkpoint if resume_from_checkpoint and resume_from_checkpoint.lower() != 'none' else None,

        # Model-specific settings
        max_length=1024,
        max_pixels=max_pixels,

        # Performance
        dataloader_num_workers=2,
        gradient_checkpointing=model_config["gradient_checkpointing"],
        use_flash_attn=use_flash_attn if use_flash_attn and "internvl" in model_name_lower else False,

        # Other
        seed=42,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
    )

    logger.info("\n📊 Final Training Configuration:")
    logger.info(f"   Learning Rate: {learning_rate}")
    logger.info(f"   LoRA Rank: {lora_rank} | Alpha: {lora_alpha}")
    logger.info(f"   Max Pixels: {max_pixels}")
    logger.info(f"   Freeze ViT: {model_config['freeze_vit']}")
    logger.info(f"   Freeze Aligner: {model_config['freeze_aligner']}")
    logger.info(f"   Target Modules: {', '.join(model_config['target_modules'][:3])}...")
    logger.info("=" * 80 + "\n")

    return sft_main(sft_args)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model SFT Training (QwenVL / InternVL / Llava)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train QwenVL (uses model-specific defaults)
  python train_sft_multimodel.py --model_path Qwen/Qwen3-VL-8B-Instruct --data_path data.jsonl --output_dir output/ --run_name experiment

  # Train InternVL (automatically uses InternVL-specific config)
  python train_sft_multimodel.py --model_path OpenGVLab/InternVL3_5-8B --data_path data.jsonl --output_dir output/ --run_name experiment

  # Train Llava (automatically uses Llava-specific config)
  python train_sft_multimodel.py --model_path llava-hf/llava-v1.6-mistral-7b-hf --data_path data.jsonl --output_dir output/ --run_name experiment

  # Override defaults
  python train_sft_multimodel.py --model_path OpenGVLab/InternVL3_5-8B --data_path data.jsonl --output_dir output/ --run_name experiment --lora_rank 32 --learning_rate 5e-6
        """
    )

    parser.add_argument("--model_path", type=str, required=True,
                       help="HuggingFace model path")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--run_name", type=str, required=True,
                       help="Wandb run name")

    parser.add_argument("--val_data_path", type=str, default=None,
                       help="Path to validation JSONL file")

    # Optional: override model-specific defaults
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate (uses model-specific default if not set)")
    parser.add_argument("--lora_rank", type=int, default=None,
                       help="LoRA rank (uses model-specific default if not set)")
    parser.add_argument("--lora_alpha", type=int, default=None,
                       help="LoRA alpha (uses model-specific default if not set)")
    parser.add_argument("--max_pixels", type=int, default=None,
                       help="Max pixels for image (uses model-specific default if not set)")

    # Standard training parameters
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                       help="Max steps (overrides num_epochs if > 0)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")

    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=20,
                       help="Log every N steps")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint dir or 'auto' for latest")
    parser.add_argument("--use_flash_attn", action="store_true",
                       help="Use Flash Attention (for InternVL)")

    args = parser.parse_args()
    train_sft(**vars(args))


if __name__ == "__main__":
    main()
