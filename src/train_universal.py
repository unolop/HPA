#!/usr/bin/env python3
"""
Universal Multi-Model Training with Human Alignment Loss

Works with QwenVL, InternVL, Llava using Swift's standard patterns.
Simply applies custom JS divergence loss on top of working configurations.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import argparse

from swift.llm import sft_main, TrainArguments, get_model_tokenizer
from swift.utils import get_logger
from swift.trainers import Trainer

logger = get_logger()


class HumanAlignmentTrainer(Trainer):
    """Custom Trainer with Human Alignment Loss using JS Divergence."""

    def __init__(
        self,
        *args,
        tokenizer=None,
        mode: str = "JS",
        lambda_dist: float = 1.0,
        lambda_l2: float = 0.1,
        use_l2_penalty: bool = True,
        use_sft_loss: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.mode = mode
        self.lambda_dist = lambda_dist
        self.lambda_l2 = lambda_l2
        self.use_l2_penalty = use_l2_penalty
        self.use_sft_loss = use_sft_loss
        self._answer_token_cache = {}

        logger.info(f"[Human Alignment Trainer] Mode: {mode}, λ_dist: {lambda_dist}, λ_L2: {lambda_l2}")

    def _get_answer_token_ids(self, answer: str) -> List[int]:
        if answer not in self._answer_token_cache:
            tokens = self.tokenizer.encode(answer, add_special_tokens=False)
            self._answer_token_cache[answer] = tokens
        return self._answer_token_cache[answer]

    def _build_human_distribution(
        self,
        answers: List[str],
        confidences: List[float],
        vocab_size: int,
        device: torch.device
    ) -> torch.Tensor:
        dist = torch.zeros(vocab_size, device=device, dtype=torch.float32)

        total_conf = sum(confidences)
        if total_conf > 0:
            confidences = [c / total_conf for c in confidences]

        for answer, conf in zip(answers, confidences):
            token_ids = self._get_answer_token_ids(answer)
            if token_ids:
                first_token = token_ids[0]
                if first_token < vocab_size:
                    dist[first_token] += conf

        if dist.sum() > 0:
            dist = dist / dist.sum()

        return dist

    def _compute_distributional_loss(
        self,
        logits: torch.Tensor,
        human_dist: torch.Tensor
    ):
        model_probs = F.softmax(logits, dim=-1)

        if self.mode == "JS":
            m = 0.5 * (model_probs + human_dist)
            kl_pm = F.kl_div(torch.log(model_probs + 1e-10), m, reduction='sum')
            kl_qm = F.kl_div(torch.log(human_dist + 1e-10), m, reduction='sum')
            js_div = 0.5 * (kl_pm + kl_qm)
            return js_div, model_probs
        else:  # CE
            ce_loss = -torch.sum(human_dist * torch.log(model_probs + 1e-10))
            return ce_loss, model_probs

    def _compute_l2_penalty(self, model_probs: torch.Tensor, human_dist: torch.Tensor):
        return torch.sum((model_probs - human_dist) ** 2)

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Extract labels info if present
        labels_info = inputs.pop('labels_info', None)

        # Forward pass
        with self.compute_loss_context_manager():
            outputs = model(**inputs)

        # Get SFT loss
        sft_loss = outputs.loss if hasattr(outputs, 'loss') else outputs.get('loss', torch.tensor(0.0))
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs.get('logits')
        labels = inputs.get('labels')

        # Initialize distributional losses
        dist_loss = torch.tensor(0.0, device=sft_loss.device)
        l2_loss = torch.tensor(0.0, device=sft_loss.device)

        # Compute custom loss if labels_info exists
        if labels_info is not None and labels is not None:
            answers = labels_info.get('answers', [])
            confidences = labels_info.get('confidences', [])

            if len(answers) > 0 and len(confidences) > 0:
                # Find answer position (last valid token)
                valid_mask = (labels != -100)
                valid_indices = valid_mask.nonzero(as_tuple=True)

                if len(valid_indices[0]) > 0:
                    batch_idx = valid_indices[0][0].item()
                    pos = valid_indices[1][0].item()

                    if pos < logits.shape[1]:
                        answer_logits = logits[batch_idx, pos, :]

                        human_dist = self._build_human_distribution(
                            answers, confidences,
                            logits.shape[-1], logits.device
                        )

                        dist_loss, model_probs = self._compute_distributional_loss(
                            answer_logits, human_dist
                        )

                        if self.use_l2_penalty:
                            l2_loss = self._compute_l2_penalty(model_probs, human_dist)

        # Combine losses
        if self.use_sft_loss:
            total_loss = sft_loss + self.lambda_dist * dist_loss
        else:
            total_loss = self.lambda_dist * dist_loss

        if self.use_l2_penalty:
            total_loss = total_loss + self.lambda_l2 * l2_loss

        # Logging
        if self.state.is_world_process_zero and self.state.global_step % 10 == 0:
            self.log({
                'sft_loss': sft_loss.detach().item(),
                'dist_loss': dist_loss.detach().item(),
                'l2_loss': l2_loss.detach().item(),
                'total_loss': total_loss.detach().item(),
            })

        return total_loss


@dataclass
class UniversalTrainArguments(TrainArguments):
    """Universal training arguments."""
    mode: str = field(default="JS")
    lambda_dist: float = field(default=1.0)
    lambda_l2: float = field(default=0.1)
    use_l2_penalty: bool = field(default=True)
    use_sft_loss: bool = field(default=False)


def train_universal(
    model_path: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    val_data_path: str = None,
    # Fixed hyperparameters
    mode: str = "JS",
    lambda_dist: float = 1.0,
    lambda_l2: float = 0.1,
    use_l2_penalty: bool = True,
    use_sft_loss: bool = False,
    # Training control
    learning_rate: float = 2e-5,
    num_epochs: int = 10,
    max_steps: int = -1,
    # LoRA settings
    lora_rank: int = 8,
    lora_alpha: int = 16,
    # Batch settings
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    # Checkpointing
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 20,
):
    """
    Universal training that works with all models.
    Uses Swift's default model handling - no custom config needed.
    """
    logger.info("=" * 80)
    logger.info("🚀 Universal Multi-Model Training")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Data: {data_path}")
    logger.info(f"   Mode: {mode} | λ_dist: {lambda_dist} | λ_L2: {lambda_l2}")
    logger.info("=" * 80)

    # Handle validation data
    val_dataset = [val_data_path] if val_data_path and val_data_path.strip() else []

    # Create training arguments - let Swift handle model-specific details
    train_args = UniversalTrainArguments(
        # Model and data
        model=model_path,
        dataset=[data_path],
        val_dataset=val_dataset,
        output_dir=output_dir,

        # LoRA training
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

        # LoRA configuration
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        lora_bias="none",
        # Let Swift auto-determine target_modules based on model
        # No need to specify - Swift knows InternVL vs QwenVL vs Llava

        # Freezing strategy - Swift handles this per model
        freeze_llm=False,
        freeze_vit=True,

        # Checkpointing
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=eval_steps if val_dataset else None,

        # Model-specific settings
        max_length=1024,
        # Swift auto-handles max_pixels per model

        # Performance
        dataloader_num_workers=2,
        gradient_checkpointing=True,

        # Other
        seed=42,
        bf16=True,
        report_to="wandb",
        run_name=run_name,

        # Custom loss parameters
        mode=mode,
        lambda_dist=lambda_dist,
        lambda_l2=lambda_l2,
        use_l2_penalty=use_l2_penalty,
        use_sft_loss=use_sft_loss,
    )

    logger.info(f"\n📊 Training Configuration:")
    logger.info(f"   Epochs: {num_epochs} | Max Steps: {max_steps}")
    logger.info(f"   Learning Rate: {learning_rate}")
    logger.info(f"   LoRA: rank={lora_rank}, alpha={lora_alpha}")
    logger.info(f"   Batch: {batch_size} × {gradient_accumulation_steps} (effective={batch_size * gradient_accumulation_steps})")
    logger.info("=" * 80 + "\n")

    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(
        model_path,
        torch_dtype=torch.bfloat16,
        model_kwargs={'device_map': 'auto'}
    )

    # Create custom trainer
    trainer = HumanAlignmentTrainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        mode=mode,
        lambda_dist=lambda_dist,
        lambda_l2=lambda_l2,
        use_l2_penalty=use_l2_penalty,
        use_sft_loss=use_sft_loss,
    )

    # Train
    result = trainer.train()

    return result


def main():
    parser = argparse.ArgumentParser(description="Universal Multi-Model Training")

    # Required
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)

    # Optional
    parser.add_argument("--val_data_path", type=str, default=None)

    # Fixed hyperparameters (can adjust these)
    parser.add_argument("--mode", type=str, default="JS", choices=["JS", "CE"])
    parser.add_argument("--lambda_dist", type=float, default=1.0)
    parser.add_argument("--lambda_l2", type=float, default=0.1)
    parser.add_argument("--use_l2_penalty", action="store_true", default=True)
    parser.add_argument("--use_sft_loss", action="store_true", default=False)

    # Training control
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # Batch
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=20)

    args = parser.parse_args()
    train_universal(**vars(args))


if __name__ == "__main__":
    main()
