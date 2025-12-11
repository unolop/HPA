import torch
import torch.nn.functional as F
import gc
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
import argparse

from swift.llm import sft_main, TrainArguments, get_model_tokenizer
from swift.utils import get_logger
from swift.trainers import Trainer

logger = get_logger()


class HumanAlignmentTrainer(Trainer):
    """
    Trainer with human-alignment loss using JS Divergence (or CE) over the vocabulary.

    Loss components:
      1. Optional SFT loss (outputs.loss) on the conversation tokens
      2. Distributional loss (CE or JS) between human distribution and model distribution
         at the "answer position" (last non -100 label token).
      3. Optional L2 / Brier-style penalty between distributions.
    """

    def __init__(
        self,
        *args,
        tokenizer=None,
        mode: str = "JS",        # "CE" or "JS"
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
        self._first_batch_logged = False

        # Cache tokenizations for efficiency
        self._answer_token_cache: Dict[str, List[int]] = {}

        self.stats = {
            "total_samples": 0,
            "samples_with_labels": 0,
            "total_sft_loss": 0.0,
            "total_dist_loss": 0.0,
            "total_l2_loss": 0.0,
        }

        logger.info("[Human Alignment Trainer] Initialized")
        logger.info(f"   Mode: {self.mode}")
        logger.info(f"   Use SFT loss: {self.use_sft_loss}")
        logger.info(f"   Lambda (dist): {self.lambda_dist}")
        logger.info(f"   Lambda (L2): {self.lambda_l2}")
        logger.info(f"   Use L2 penalty: {self.use_l2_penalty}")

    # -------------------------------------------------------------------------
    #  Helpers for distributions
    # -------------------------------------------------------------------------

    def _get_answer_token_ids(self, answer: str) -> List[int]:
        """Get token IDs for an answer string, with caching."""
        if answer not in self._answer_token_cache:
            tokens = self.tokenizer.encode(answer, add_special_tokens=False)
            self._answer_token_cache[answer] = tokens
        return self._answer_token_cache[answer]

    def _build_human_distribution(
        self,
        answers: List[str],
        confidences: List[float],
        vocab_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build human confidence distribution over the vocabulary.

        For each human answer:
          * tokenize it,
          * spread its confidence evenly across its tokens,
          * sum across all answers.

        This is a vocab-level approximation of the human belief.

        Args:
            answers: List of possible answers (strings)
            confidences: Corresponding confidence scores (need not sum to 1)
            vocab_size: Model vocabulary size
            device: Target device

        Returns:
            Tensor of shape [vocab_size] with probability mass on answer tokens.
        """
        dist = torch.zeros(vocab_size, device=device, dtype=torch.float32)

        if not answers or not confidences:
            # Fallback: uniform if nothing is provided
            return torch.ones(vocab_size, device=device, dtype=torch.float32) / vocab_size

        # Normalize confidences to sum to 1
        total_conf = float(sum(confidences))
        if total_conf > 0.0:
            confidences = [c / total_conf for c in confidences]

        for ans, conf in zip(answers, confidences):
            token_ids = self._get_answer_token_ids(ans)
            if not token_ids:
                continue
            share = conf / len(token_ids)
            for tid in token_ids:
                if 0 <= tid < vocab_size:
                    dist[tid] += share

        mass = dist.sum()
        if mass > 0:
            dist = dist / mass
        else:
            # If nothing got mass (e.g. all tokenizations empty), fallback to uniform
            dist = torch.ones(vocab_size, device=device, dtype=torch.float32)
            dist = dist / dist.sum()

        return dist

    def _compute_distributional_loss(
        self,
        model_logits: torch.Tensor,   # [vocab_size]
        human_dist: torch.Tensor,     # [vocab_size]
    ):
        """
        Compute distributional matching loss between model and human distributions.

        Args:
            model_logits: [vocab_size] logits from model at the answer position
            human_dist:   [vocab_size] human target distribution

        Returns:
            dist_loss: scalar loss
            model_probs: [vocab_size] model probability distribution
        """
        model_probs = F.softmax(model_logits, dim=-1)

        if self.mode == "CE":
            # Cross-Entropy: -sum(H(a) * log M(a))
            log_probs = F.log_softmax(model_logits, dim=-1)
            dist_loss = -(human_dist * log_probs).sum()

        elif self.mode == "JS":
            # JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),
            # where M = 0.5 * (P + Q)
            eps = 1e-12
            p = human_dist.clamp(min=eps)
            q = model_probs.clamp(min=eps)
            m = 0.5 * (p + q).clamp(min=eps)

            kl_pm = (p * (p.log() - m.log())).sum()
            kl_qm = (q * (q.log() - m.log())).sum()
            dist_loss = 0.5 * (kl_pm + kl_qm)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return dist_loss, model_probs

    def _compute_l2_penalty(
        self,
        model_probs: torch.Tensor,
        human_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Brier-style L2 penalty between distributions."""
        return torch.norm(human_dist - model_probs, p=2).pow(2)

    # -------------------------------------------------------------------------
    #  Training loop override
    # -------------------------------------------------------------------------

    def training_step(self, model, inputs):
        """
        Override training step to add human-alignment loss.

        Assumes:
          - inputs['labels_info'] exists and is either:
              * a dict for the whole batch (same labels for all items), or
              * a list of dicts, one per batch element.
          - Each dict has keys 'answers' (List[str]) and 'confidences' (List[float]).
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Extract labels info (human confidences for this batch)
        labels_info = inputs.pop("labels_info", None)

        self.stats["total_samples"] += 1

        if not self._first_batch_logged:
            self._log_first_batch(inputs, labels_info)
            self._first_batch_logged = True

        # Standard forward pass (provides SFT loss and logits)
        with self.compute_loss_context_manager():
            outputs = model(**inputs)

        sft_loss = outputs.loss
        if sft_loss is None:
            raise ValueError("SFT loss is None!")

        logits = outputs.logits  # [batch, seq, vocab]
        labels = inputs.get("labels")  # [batch, seq] or None

        # Start from zero; we add components explicitly
        total_loss = sft_loss.new_tensor(0.0)
        dist_loss = sft_loss.new_tensor(0.0)
        l2_loss = sft_loss.new_tensor(0.0)

        # 1) Optionally include SFT loss
        #    If labels_info is None (no human dist), we fall back to pure SFT.
        if self.use_sft_loss or labels_info is None:
            total_loss = total_loss + sft_loss

        # 2) Distributional loss (JS / CE) if we have human annotations
        if labels_info is not None and labels is not None and self._has_valid_labels(labels_info):
            self.stats["samples_with_labels"] += 1

            batch_size, seq_len, vocab_size = logits.shape
            answer_positions = self._find_answer_positions(labels)

            # labels_info may be a dict (shared) or list[dict] (per sample)
            per_example_labels: List[Dict] = []
            if isinstance(labels_info, dict):
                # Same labels for all items
                per_example_labels = [labels_info] * batch_size
            elif isinstance(labels_info, (list, tuple)):
                # One dict per item (assume len matches batch_size)
                if len(labels_info) != batch_size:
                    raise ValueError(
                        f"labels_info has length {len(labels_info)} but batch size is {batch_size}"
                    )
                per_example_labels = list(labels_info)
            else:
                raise ValueError(
                    f"Unsupported labels_info type: {type(labels_info)}. "
                    "Expected dict or list of dicts."
                )

            valid_count = 0

            for b_idx, pos in enumerate(answer_positions):
                if pos < 0 or pos >= seq_len:
                    continue

                label_info_b = per_example_labels[b_idx]
                answers = label_info_b.get("answers", [])
                confidences = label_info_b.get("confidences", [])

                if not answers or not confidences:
                    continue

                # Build human distribution for this sample
                human_dist = self._build_human_distribution(
                    answers=answers,
                    confidences=confidences,
                    vocab_size=vocab_size,
                    device=logits.device,
                )

                answer_logits = logits[b_idx, pos, :]  # [vocab]
                sample_dist_loss, sample_model_probs = self._compute_distributional_loss(
                    answer_logits, human_dist
                )

                dist_loss = dist_loss + sample_dist_loss

                if self.use_l2_penalty:
                    sample_l2 = self._compute_l2_penalty(sample_model_probs, human_dist)
                    l2_loss = l2_loss + sample_l2

                valid_count += 1

            if valid_count > 0:
                dist_loss = dist_loss / valid_count
                l2_loss = l2_loss / valid_count

                # Scale and add distributional components
                total_loss = total_loss + self.lambda_dist * dist_loss
                if self.use_l2_penalty and self.lambda_l2 != 0.0:
                    total_loss = total_loss + self.lambda_l2 * l2_loss

        # Safety: avoid NaN / Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(
                f"⚠️ NaN/Inf detected! "
                f"sft={sft_loss.item():.4f}, dist={dist_loss.item():.4f}, l2={l2_loss.item():.4f}"
            )
            # Fall back to pure SFT (if available)
            total_loss = sft_loss

        # Track running sums (for global averages)
        self.stats["total_sft_loss"] += float(sft_loss.item())
        self.stats["total_dist_loss"] += float(dist_loss.item()) if not torch.isnan(dist_loss) else 0.0
        self.stats["total_l2_loss"] += float(l2_loss.item()) if not torch.isnan(l2_loss) else 0.0

        # Logging
        if self.state.is_world_process_zero:
            log_dict = {
                "sft_loss": float(sft_loss.detach().item()),
                "dist_loss": float(dist_loss.detach().item()),
                "l2_loss": float(l2_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
            }
            self.log(log_dict)

            if self.state.global_step % 50 == 0 and self.state.global_step > 0:
                self._log_statistics()

        # Multi-GPU
        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()

        # Backprop
        self.accelerator.backward(total_loss)

        del outputs
        torch.cuda.empty_cache()

        return total_loss.detach()

    # -------------------------------------------------------------------------
    #  Utilities
    # -------------------------------------------------------------------------

    def _has_valid_labels(self, labels_info: Union[Dict, List[Dict]]) -> bool:
        """
        Check if labels_info has valid human annotation data.

        Works if labels_info is:
          - dict with 'answers' & 'confidences'
          - list of such dicts (we just check the first)
        """
        if labels_info is None:
            return False

        if isinstance(labels_info, dict):
            answers = labels_info.get("answers", [])
            confidences = labels_info.get("confidences", [])
            return len(answers) > 0 and len(confidences) > 0

        if isinstance(labels_info, (list, tuple)) and len(labels_info) > 0:
            first = labels_info[0]
            if not isinstance(first, dict):
                return False
            answers = first.get("answers", [])
            confidences = first.get("confidences", [])
            return len(answers) > 0 and len(confidences) > 0

        return False

    def _find_answer_positions(self, labels: torch.Tensor) -> List[int]:
        """
        Find positions where answers end.

        We treat the LAST non -100 label as the "answer position" for
        distributional supervision. This matches typical SFT setups where
        the assistant's answer is at the end of the sequence.
        """
        batch_size = labels.shape[0]
        positions: List[int] = []

        for b in range(batch_size):
            label_seq = labels[b]  # [seq_len]
            valid_indices = (label_seq != -100).nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                positions.append(int(valid_indices[-1].item()))
            else:
                positions.append(-1)

        return positions

    def _log_statistics(self):
        """Log cumulative statistics (averages over all processed samples)."""
        n = self.stats["total_samples"]
        if n == 0:
            return

        logger.info("=" * 60)
        logger.info(f"[Statistics] Step {self.state.global_step}")
        logger.info(f"   Total samples: {n}")
        logger.info(
            f"   Samples with labels: {self.stats['samples_with_labels']} "
            f"({self.stats['samples_with_labels']/n*100:.1f}%)"
        )
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
                logger.info(
                    f"   {k}: shape={tuple(v.shape)}, dtype={v.dtype}, size={size_mb:.2f}MB"
                )

        if labels_info is not None:
            logger.info(
                f"✅ Labels info example: "
                f"{labels_info if isinstance(labels_info, dict) else labels_info[0]}"
            )
        else:
            logger.warning("❌ No labels info found!")
        logger.info("=" * 80)


# -------------------------------------------------------------------------
#  Extended arguments
# -------------------------------------------------------------------------

@dataclass
class HumanAlignmentArguments(TrainArguments):
    """Extended TrainArguments with human alignment parameters."""
    mode: str = field(default="JS", metadata={"help": "CE or JS for distributional matching"})
    lambda_dist: float = field(default=1.0, metadata={"help": "Weight for distributional loss"})
    lambda_l2: float = field(default=0.1, metadata={"help": "Weight for L2 penalty"})
    use_l2_penalty: bool = field(default=True)
    use_sft_loss: bool = field(
        default=False,
        metadata={"help": "Include SFT loss on conversation"},
    )


# -------------------------------------------------------------------------
#  Training entrypoint
# -------------------------------------------------------------------------

def train_human_alignment(
    model_path: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    val_data_path: Optional[str],
    mode: str = "JS",
    lambda_dist: float = 1.0,
    lambda_l2: float = 0.1,
    use_l2_penalty: bool = True,
    use_sft_loss: bool = False,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_steps: int = -1,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 40,
    eval_steps: int = 40,
    logging_steps: int = 20,
    max_pixels: int = 448,
    resume_from_checkpoint: Optional[str] = None,
):
    """Train with Human Alignment Loss (JS Divergence or CE)."""

    logger.info("=" * 80)
    logger.info("🚀 Human Alignment Training")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Lambda (dist): {lambda_dist}")
    logger.info(f"   Lambda (L2): {lambda_l2}")
    logger.info("=" * 80)

    if max_steps > 0:
        logger.warning(
            f"⚠️ Both max_steps ({max_steps}) and num_epochs ({num_epochs}) are set."
        )
        logger.warning("⚠️ Training will run for max_steps and IGNORE num_epochs.")
        logger.warning("⚠️ To use num_epochs, set max_steps = -1.")
    else:
        logger.info(f"📊 Training for {num_epochs} epochs (max_steps={max_steps})")

    # Checkpoint info
    if resume_from_checkpoint:
        if resume_from_checkpoint.lower() == "auto":
            logger.info(f"🔄 Will resume from latest checkpoint in {output_dir}")
        else:
            logger.info(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Get tokenizer for answer token lookup
    _, tokenizer = get_model_tokenizer(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Validation data path normalization
    if val_data_path is not None and (
        val_data_path.lower() == "none" or val_data_path.strip() == ""
    ):
        val_data_path = None

    if val_data_path is not None:
        val_dataset_list = [val_data_path]
    else:
        val_dataset_list = []

    sft_args = HumanAlignmentArguments(
        model=model_path,
        dataset=[data_path],
        val_dataset=val_dataset_list,
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
        lora_dropout=0.1,
        lora_bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        freeze_llm=False,
        freeze_vit=True,
        freeze_aligner=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        eval_steps=eval_steps if len(val_dataset_list) > 0 else None,
        eval_strategy="steps" if len(val_dataset_list) > 0 else "no",
        resume_from_checkpoint=(
            resume_from_checkpoint
            if resume_from_checkpoint and resume_from_checkpoint.lower() != "none"
            else None
        ),
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
        use_sft_loss=use_sft_loss,
    )

    def create_trainer(*args, **kwargs):
        kwargs["tokenizer"] = tokenizer
        kwargs["mode"] = sft_args.mode
        kwargs["lambda_dist"] = sft_args.lambda_dist
        kwargs["lambda_l2"] = sft_args.lambda_l2
        kwargs["use_l2_penalty"] = sft_args.use_l2_penalty
        kwargs["use_sft_loss"] = sft_args.use_sft_loss
        return HumanAlignmentTrainer(*args, **kwargs)

    import swift.trainers
    original_trainer_cls = swift.trainers.Trainer
    swift.trainers.Trainer = create_trainer

    try:
        result = sft_main(sft_args)
        return result
    finally:
        swift.trainers.Trainer = original_trainer_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="Validation data JSONL file (optional, omit for no validation)",
    )
    parser.add_argument("--mode", type=str, default="JS", choices=["CE", "JS"])
    parser.add_argument("--lambda_dist", type=float, default=1.0)
    parser.add_argument("--lambda_l2", type=float, default=0.1)
    parser.add_argument(
        "--use_l2_penalty",
        action="store_true",
        default=False,
        help="Enable L2 (Brier-style) penalty between distributions",
    )
    parser.add_argument(
        "--use_sft_loss",
        action="store_true",
        default=False,
        help="Include SFT loss on conversation tokens",
    )
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
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume from, or 'auto' for latest checkpoint",
    )

    args = parser.parse_args()
    train_human_alignment(**vars(args))


if __name__ == "__main__":
    main()
