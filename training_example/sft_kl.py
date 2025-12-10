import torch
import torch.nn.functional as F
import os
from dataclasses import dataclass, field
from typing import Optional
import argparse

from swift.llm import sft_main, TrainArguments, get_model_tokenizer
from swift.utils import get_logger
from swift.trainers import Trainer

logger = get_logger()


class KLRegularizedTrainer(Trainer):
    """
    Custom Trainer that adds KL Divergence Regularization to SFT Loss.
    
    Total Loss = SFT Loss + beta * KL Loss
    
    The KL loss measures the divergence between the policy model's output distribution
    and a reference model's output distribution, helping to prevent the model from
    deviating too far from the reference during fine-tuning.
    """    
    def __init__(self, *args, ref_model=None, beta=0.1, **kwargs):
        """
        Initialize the KL Regularized Trainer.
        
        Args:
            ref_model: Reference model for KL divergence calculation
            beta: Weight for KL divergence term (default: 0.1)
        """
        super().__init__(*args, **kwargs) 
        
        if ref_model is None:
            raise ValueError("KLRegularizedTrainer requires a reference model.")
        
        # Setup reference model
        self.ref_model = ref_model.to(self.args.device)
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        
        self.beta = beta
        self._first_batch_logged = False
        
        logger.info(f"[KL Trainer] Initialized with beta={self.beta}")

    def training_step(self, model, inputs):
        """
        Override training step to add KL divergence regularization.
        
        Args:
            model: The model being trained
            inputs: Batch inputs including input_ids, attention_mask, labels, etc.
            
        Returns:
            total_loss: Combined SFT loss and KL loss
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Debug logging for first batch
        if not self._first_batch_logged:
            self._log_first_batch(inputs)
            self._first_batch_logged = True
        
        # Forward pass - Policy model
        with self.compute_loss_context_manager():
            outputs = model(**inputs)
        
        sft_loss = outputs.loss
        if sft_loss is None:
            raise ValueError("SFT loss is None! Check if labels are properly generated.")
        
        policy_logits = outputs.logits
        
        # Forward pass - Reference model (no gradient)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits.detach()
        
        # Calculate KL divergence loss
        labels = inputs.get("labels")
        
        if labels is not None and policy_logits.shape[:2] == ref_logits.shape[:2]:
            kl_loss, num_active_tokens = self._compute_kl_loss(
                policy_logits, ref_logits, labels
            )
            total_loss = sft_loss + self.beta * kl_loss
            
            # Log metrics
            if self.state.is_world_process_zero:
                self.log({
                    'sft_loss': sft_loss.detach().item(),
                    'kl_loss': kl_loss.detach().item(),
                    'total_loss': total_loss.detach().item(),
                    'num_active_tokens': num_active_tokens.item(),
                })
        else:
            # Fallback to SFT loss only if KL cannot be computed
            total_loss = sft_loss
            if self.state.is_world_process_zero:
                self.log({'sft_loss': sft_loss.detach().item()})
        
        # Backward pass
        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()
        
        self.accelerator.backward(total_loss)
        
        return total_loss.detach()
    
    def _compute_kl_loss(self, policy_logits, ref_logits, labels):
        """
        Compute KL divergence loss between policy and reference model outputs.
        
        Args:
            policy_logits: Logits from the policy (training) model
            ref_logits: Logits from the reference model
            labels: Ground truth labels with -100 for padding tokens
            
        Returns:
            kl_loss: Mean KL divergence over non-padding tokens
            num_active_tokens: Number of tokens used in calculation
        """
        # Apply causal LM shift: predict token t+1 from logits at position t
        shift_policy_logits = policy_logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Convert logits to probabilities
        policy_log_probs = F.log_softmax(shift_policy_logits, dim=-1)
        ref_probs = F.softmax(shift_ref_logits, dim=-1)
        
        # Calculate KL divergence per token
        kl_div_per_token = F.kl_div(
            policy_log_probs, ref_probs, reduction='none'
        ).sum(dim=-1)
        
        # Mask padding tokens (where label == -100)
        mask = (shift_labels != -100).to(kl_div_per_token.dtype)
        
        # Average over non-padding tokens
        masked_kl_sum = (kl_div_per_token * mask).sum()
        num_active_tokens = mask.sum().clamp(min=1.0)
        kl_loss = masked_kl_sum / num_active_tokens
        
        return kl_loss, num_active_tokens
    
    def _log_first_batch(self, inputs):
        """Log information about the first batch for debugging."""
        logger.info("=" * 80)
        logger.info("[First Batch Debug]")
        logger.info(f"Input keys: {list(inputs.keys())}")
        
        if 'labels' in inputs:
            labels = inputs['labels']
            num_total = labels.numel()
            num_masked = (labels == -100).sum().item()
            num_active = num_total - num_masked
            logger.info(f"✅ Labels found!")
            logger.info(f"   Total tokens: {num_total}")
            logger.info(f"   Masked tokens: {num_masked}")
            logger.info(f"   Active tokens: {num_active}")
        else:
            logger.warning("❌ No labels found in inputs!")
        
        logger.info("=" * 80)


@dataclass
class KLSftArguments(TrainArguments):
    """Extended TrainArguments with KL divergence parameter."""
    kl_beta: float = field(
        default=0.1, 
        metadata={'help': 'Weight for KL divergence regularization term'}
    )


def train_with_kl(
    model_path: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    val_data_path: str = None,
    beta: float = 0.1,
    learning_rate: float = 1e-5,
    num_epochs: int = 2,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 100,
    logging_steps: int = 100,
):
    """
    Train a model with KL Divergence Regularization using Swift's SFT pipeline.
    
    This function:
    1. Loads a reference model for KL divergence calculation
    2. Sets up training arguments matching the original SFT configuration
    3. Injects a custom trainer that adds KL regularization
    4. Runs the Swift SFT training pipeline
    
    Args:
        model_path: Path to the base model
        data_path: Path to training data (.jsonl format)
        output_dir: Directory to save checkpoints
        beta: KL divergence weight (default: 0.1)
        learning_rate: Learning rate (default: 1e-5)
        num_epochs: Number of training epochs (default: 2)
        lora_rank: LoRA rank parameter (default: 32)
        lora_alpha: LoRA alpha parameter (default: 64)
        batch_size: Per-device batch size (default: 1)
        gradient_accumulation_steps: Gradient accumulation steps (default: 8)
        save_steps: Save checkpoint every N steps (default: 100)
        logging_steps: Log metrics every N steps (default: 100)
        
    Returns:
        Training results from sft_main
    """
    logger.info("=" * 80)
    logger.info("🚀 KL-Regularized SFT Training")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Data: {data_path}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Beta (KL weight): {beta}")
    logger.info(f"   Learning Rate: {learning_rate}")
    logger.info(f"   Epochs: {num_epochs}")
    logger.info("=" * 80)
    
    # Load reference model (kept frozen for KL divergence)
    logger.info("📥 Loading reference model...")
    ref_model, _ = get_model_tokenizer(model_path, torch_dtype=torch.bfloat16)
    ref_model.eval()
    ref_model.requires_grad_(False)
    logger.info("✅ Reference model loaded and frozen")
    
    # Configure training arguments
    sft_args = KLSftArguments(
        # Model & Data
        model=model_path,
        dataset=[data_path],
        val_dataset=val_data_path,
        output_dir=output_dir,
        
        # Training type
        train_type="lora",
        torch_dtype="bfloat16",
        
        # Training hyperparameters
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.05,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="cosine",
        
        # LoRA configuration
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        lora_bias="none",
        target_modules=["all-linear"],
        
        # Freeze configuration
        freeze_llm=False,
        freeze_vit=True,
        freeze_aligner=True,
        
        # Saving & Logging
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=logging_steps,
        logging_first_step=True,
        eval_steps=save_steps,
        
        # Template configuration
        max_length=4096,
        truncation_strategy="right",
        max_pixels=448,
        
        # Other settings
        dataloader_num_workers=8,
        gradient_checkpointing=False,
        seed=42,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
        attn_impl="flash_attn",        
        # KL divergence parameter
        kl_beta=beta,
    )
    
    # Create custom trainer factory
    def create_trainer(*args, **kwargs):
        """Factory function to inject reference model into trainer."""
        kwargs['ref_model'] = ref_model
        kwargs['beta'] = sft_args.kl_beta
        return KLRegularizedTrainer(*args, **kwargs)
    
    # Monkey patch Swift's Trainer class
    import swift.trainers
    original_trainer = swift.trainers.Trainer
    swift.trainers.Trainer = create_trainer
    
    try:
        logger.info("🚀 Starting training with KL regularization...")
        result = sft_main(sft_args)
        logger.info("✅ Training completed successfully!")
        return result
    finally:
        # Restore original Trainer class
        logger.info("🔄 Restoring original Trainer class...")
        swift.trainers.Trainer = original_trainer


def main():
    """Parse command line arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Train a model with KL Divergence Regularization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the base model directory"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to training data file (.jsonl format)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--run_name", type=str, required=True,
        help="Directory to save model checkpoints"
    )        

    # Optional arguments
    parser.add_argument(
        "--val_data_path", type=str, default=None,
        help="Path to training data file (.jsonl format)"
    )        
    parser.add_argument(
        "--beta", type=float, default=0.1,
        help="KL divergence weight (higher = stronger regularization)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=32,
        help="LoRA rank parameter"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=64,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=100,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100,
        help="Log metrics every N steps"
    )
    
    args = parser.parse_args()
    
    # Start training
    train_with_kl(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        val_data_path=args.val_data_path,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()