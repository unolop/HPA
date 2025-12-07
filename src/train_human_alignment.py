#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import gc
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import argparse
import json

from swift.llm import sft_main, TrainArguments, get_model_tokenizer
from swift.utils import get_logger
from swift.trainers import Trainer

logger = get_logger()


class HumanAlignmentTrainer(Trainer):
    """
    Custom Trainer with Human Alignment Loss using JS Divergence.
    
    Combines:
    1. Standard SFT loss (CE) on conversations
    2. JS Divergence loss to match human confidence distributions
    3. Optional L2 penalty (Brier-style)
    """
    
    def __init__(
        self,
        *args,
        tokenizer=None,
        mode: str = "JS",  # "CE" or "JS"
        lambda_dist: float = 1.0,  # Weight for distributional matching
        lambda_l2: float = 0.1,  # Weight for L2 penalty
        use_l2_penalty: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.tokenizer = tokenizer
        self.mode = mode
        self.lambda_dist = lambda_dist
        self.lambda_l2 = lambda_l2
        self.use_l2_penalty = use_l2_penalty
        self._first_batch_logged = False
        
        # Pre-tokenize common answers for efficiency
        self._answer_token_cache = {}
        
        self.stats = {
            'total_samples': 0,
            'samples_with_labels': 0,
            'total_sft_loss': 0.0,
            'total_dist_loss': 0.0,
            'total_l2_loss': 0.0,
        }
        
        logger.info(f"[Human Alignment Trainer] Initialized")
        logger.info(f"   Mode: {self.mode}")
        logger.info(f"   Lambda (dist): {self.lambda_dist}")
        logger.info(f"   Lambda (L2): {self.lambda_l2}")
        logger.info(f"   Use L2 penalty: {self.use_l2_penalty}")

    def _get_answer_token_ids(self, answer: str) -> List[int]:
        """Get token IDs for an answer, with caching."""
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
        """
        Build human confidence distribution over vocabulary.
        
        Args:
            answers: List of possible answers
            confidences: Corresponding confidence scores (should sum to 1)
            vocab_size: Model vocabulary size
            device: Target device
            
        Returns:
            Tensor of shape [vocab_size] with probability mass on answer tokens
        """
        dist = torch.zeros(vocab_size, device=device, dtype=torch.float32)
        
        # Normalize confidences if they don't sum to 1
        total_conf = sum(confidences)
        if total_conf > 0:
            confidences = [c / total_conf for c in confidences]
        
        for answer, conf in zip(answers, confidences):
            token_ids = self._get_answer_token_ids(answer)
            if token_ids:
                # Put confidence on first token of answer
                # (for multi-token answers, could distribute or use first token)
                first_token = token_ids[0]
                dist[first_token] += conf
        
        # Ensure it's a valid distribution
        if dist.sum() > 0:
            dist = dist / dist.sum()
        else:
            # Fallback to uniform if no valid tokens
            dist = torch.ones(vocab_size, device=device) / vocab_size
            
        return dist

    def _compute_distributional_loss(
        self,
        model_logits: torch.Tensor,
        human_dist: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distributional matching loss.
        
        Args:
            model_logits: [vocab_size] logits from model
            human_dist: [vocab_size] target distribution
        """
        model_probs = F.softmax(model_logits, dim=-1)
        
        if self.mode == "CE":
            # Cross-Entropy: -sum(H(a) * log M(a))
            log_probs = F.log_softmax(model_logits, dim=-1)
            dist_loss = -torch.sum(human_dist * log_probs)
            
        elif self.mode == "JS":
            # Jensen-Shannon Divergence
            m_dist = (human_dist + model_probs) / 2.0
            m_dist = m_dist.clamp(min=1e-12)
            
            # JS = 0.5 * KL(H || M) + 0.5 * KL(P || M)
            kl_human_m = F.kl_div(
                m_dist.log(), human_dist, reduction='sum', log_target=False
            )
            kl_model_m = F.kl_div(
                m_dist.log(), model_probs, reduction='sum', log_target=False
            )
            dist_loss = 0.5 * kl_human_m + 0.5 * kl_model_m
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return dist_loss, model_probs

    def _compute_l2_penalty(
        self,
        model_probs: torch.Tensor,
        human_dist: torch.Tensor
    ) -> torch.Tensor:
        """Brier-style L2 penalty between distributions."""
        return torch.norm(human_dist - model_probs, p=2).pow(2)

    def training_step(self, model, inputs):
        """Override training step with human alignment loss."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Extract labels info (human confidences)
        labels_info = inputs.pop('labels_info', None)
        
        self.stats['total_samples'] += 1
        
        if not self._first_batch_logged:
            self._log_first_batch(inputs, labels_info)
            self._first_batch_logged = True
        
        # Standard forward pass
        with self.compute_loss_context_manager():
            outputs = model(**inputs)
        
        sft_loss = outputs.loss
        if sft_loss is None:
            raise ValueError("SFT loss is None!")
        
        total_loss = sft_loss
        dist_loss = torch.tensor(0.0, device=sft_loss.device)
        l2_loss = torch.tensor(0.0, device=sft_loss.device)
        
        # Compute alignment loss if we have label info
        if labels_info is not None and self._has_valid_labels(labels_info):
            self.stats['samples_with_labels'] += 1
            
            # Get logits at the answer position
            # For VQA, the answer is typically the last assistant turn
            logits = outputs.logits  # [batch, seq, vocab]
            labels = inputs.get("labels")
            
            # Find position of the answer token (last non-padding label)
            if labels is not None:
                answer_positions = self._find_answer_positions(labels)
                
                for batch_idx, pos in enumerate(answer_positions):
                    if pos >= 0 and pos < logits.shape[1]:
                        # Get logits at answer position
                        answer_logits = logits[batch_idx, pos, :]
                        
                        # Build human distribution
                        answers = labels_info['answers']
                        confidences = labels_info['confidences']
                        
                        human_dist = self._build_human_distribution(
                            answers, confidences, 
                            logits.shape[-1], logits.device
                        )
                        
                        # Compute distributional loss
                        batch_dist_loss, model_probs = self._compute_distributional_loss(
                            answer_logits, human_dist
                        )
                        dist_loss = dist_loss + batch_dist_loss
                        
                        # Compute L2 penalty
                        if self.use_l2_penalty:
                            batch_l2_loss = self._compute_l2_penalty(model_probs, human_dist)
                            l2_loss = l2_loss + batch_l2_loss
            
            # Average over batch
            batch_size = logits.shape[0]
            dist_loss = dist_loss / max(batch_size, 1)
            l2_loss = l2_loss / max(batch_size, 1)
            
            # Combine losses
            total_loss = sft_loss + self.lambda_dist * dist_loss
            if self.use_l2_penalty:
                total_loss = total_loss + self.lambda_l2 * l2_loss
        
        # Track losses
        self.stats['total_sft_loss'] += sft_loss.item()
        self.stats['total_dist_loss'] += dist_loss.item()
        self.stats['total_l2_loss'] += l2_loss.item()
        
        # Logging
        if self.state.is_world_process_zero:
            log_dict = {
                'sft_loss': sft_loss.detach().item(),
                'dist_loss': dist_loss.detach().item(),
                'l2_loss': l2_loss.detach().item(),
                'total_loss': total_loss.detach().item(),
            }
            self.log(log_dict)
            
            if self.state.global_step % 50 == 0 and self.state.global_step > 0:
                self._log_statistics()
        
        # Backward
        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()
        
        self.accelerator.backward(total_loss)
        
        del outputs
        torch.cuda.empty_cache()
        
        return total_loss.detach()

    def _has_valid_labels(self, labels_info: Dict) -> bool:
        """Check if labels_info has valid data."""
        if labels_info is None:
            return False
        answers = labels_info.get('answers', [])
        confidences = labels_info.get('confidences', [])
        return len(answers) > 0 and len(confidences) > 0

    def _find_answer_positions(self, labels: torch.Tensor) -> List[int]:
        """Find positions where answers start (first non -100 after -100 sequence)."""
        batch_size = labels.shape[0]
        positions = []
        
        for b in range(batch_size):
            label_seq = labels[b]
            # Find last valid (non -100) position
            valid_mask = (label_seq != -100)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            
            if len(valid_indices) > 0:
                # Use the first valid position as answer start
                positions.append(valid_indices[0].item())
            else:
                positions.append(-1)
        
        return positions

    def _log_statistics(self):
        """Log cumulative statistics."""
        n = self.stats['total_samples']
        if n == 0:
            return
        
        logger.info("=" * 60)
        logger.info(f"[Statistics] Step {self.state.global_step}")
        logger.info(f"   Total samples: {n}")
        logger.info(f"   Samples with labels: {self.stats['samples_with_labels']} ({self.stats['samples_with_labels']/n*100:.1f}%)")
        logger.info(f"   Avg SFT loss:  {self.stats['total_sft_loss']/n:.4f}")
        logger.info(f"   Avg Dist loss: {self.stats['total_dist_loss']/n:.4f}")
        logger.info(f"   Avg L2 loss:   {self.stats['total_l2_loss']/n:.4f}")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"   GPU Memory: {allocated:.2f}GB allocated")
        logger.info("=" * 60)

    def _log_first_batch(self, inputs, labels_info):
        """Log first batch debug info."""
        logger.info("=" * 80)
        logger.info("[First Batch Debug]")
        logger.info(f"Input keys: {list(inputs.keys())}")
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                size_mb = v.numel() * v.element_size() / 1024**2
                logger.info(f"   {k}: shape={v.shape}, dtype={v.dtype}, size={size_mb:.2f}MB")
        
        if labels_info is not None:
            logger.info(f"✅ Labels info: answers={labels_info.get('answers')}, confidences={labels_info.get('confidences')}")
        else:
            logger.warning("❌ No labels info found!")
        logger.info("=" * 80)


@dataclass
class HumanAlignmentArguments(TrainArguments):
    """Extended TrainArguments with human alignment parameters."""
    mode: str = field(default="JS", metadata={'help': 'CE or JS for distributional matching'})
    lambda_dist: float = field(default=1.0, metadata={'help': 'Weight for distributional loss'})
    lambda_l2: float = field(default=0.1, metadata={'help': 'Weight for L2 penalty'})
    use_l2_penalty: bool = field(default=True)


def train_human_alignment(
    model_path: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    val_data_path,
    mode: str = "JS",
    lambda_dist: float = 1.0,
    lambda_l2: float = 0.1,
    use_l2_penalty: bool = True,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_steps: int = 2000,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 40,
    eval_steps: int = 40,
    logging_steps: int = 20,
    max_pixels: int = 448,
):
    """Train with Human Alignment Loss (JS Divergence)."""
    
    logger.info("=" * 80)
    logger.info("🚀 Human Alignment Training")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Lambda (dist): {lambda_dist}")
    logger.info(f"   Lambda (L2): {lambda_l2}")
    logger.info("=" * 80)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Get tokenizer for answer token lookup
    _, tokenizer = get_model_tokenizer(model_path, torch_dtype=torch.bfloat16, device_map="cpu")
    if val_data_path is not None: 
        val_data_path = [val_data_path] 
    else: 
        val_data_path = None 

    sft_args = HumanAlignmentArguments(
        model=model_path,
        dataset=[data_path],
        val_dataset=val_data_path,
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
        lora_dropout=0.1,  # Higher dropout for small dataset regularization
        lora_bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
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
        
        mode=mode,
        lambda_dist=lambda_dist,
        lambda_l2=lambda_l2,
        use_l2_penalty=use_l2_penalty,
    )
    
    def create_trainer(*args, **kwargs):
        kwargs['tokenizer'] = tokenizer
        kwargs['mode'] = sft_args.mode
        kwargs['lambda_dist'] = sft_args.lambda_dist
        kwargs['lambda_l2'] = sft_args.lambda_l2
        kwargs['use_l2_penalty'] = sft_args.use_l2_penalty
        return HumanAlignmentTrainer(*args, **kwargs)
    
    import swift.trainers
    original_trainer = swift.trainers.Trainer
    swift.trainers.Trainer = create_trainer
    
    try:
        result = sft_main(sft_args)
        return result
    finally:
        swift.trainers.Trainer = original_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="JS", choices=["CE", "JS"])
    parser.add_argument("--lambda_dist", type=float, default=1.0)
    parser.add_argument("--lambda_l2", type=float, default=0.1)
    parser.add_argument("--use_l2_penalty", action="store_true", default=True)
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
    train_human_alignment(**vars(args))


if __name__ == "__main__":
    main()