#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Human Alignment Training with Answer-Level JS Divergence.

- JS / CE is computed over ANSWERS (not vocabulary tokens)
- Supports free-form VQA + multiple-choice (MMStar)
- Batch size MUST be 1
"""

import torch
import torch.nn.functional as F
import gc
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from swift.llm import sft_main, TrainArguments, get_model_tokenizer
from swift.utils import get_logger
from swift.trainers import Trainer

logger = get_logger()

# ============================================================
# Trainer
# ============================================================

class HumanAlignmentTrainer(Trainer):
    """
    Answer-level human alignment trainer using JS / CE divergence.
    """

    def __init__(
        self,
        *args,
        tokenizer=None,
        mode: str = "JS",        # "JS" or "CE"
        lambda_dist: float = 1.0,
        use_sft_loss: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer
        self.mode = mode
        self.lambda_dist = lambda_dist
        self.use_sft_loss = use_sft_loss

        self._answer_token_cache: Dict[str, List[int]] = {}
        self._first_batch_logged = False

        logger.info("[HumanAlignmentTrainer]")
        logger.info(f"  mode={mode}")
        logger.info(f"  lambda_dist={lambda_dist}")
        logger.info(f"  use_sft_loss={use_sft_loss}")

    # ------------------------------------------------------------
    # Token utilities
    # ------------------------------------------------------------

    def _encode_answer(self, answer: str) -> List[int]:
        if answer not in self._answer_token_cache:
            self._answer_token_cache[answer] = self.tokenizer.encode(
                answer, add_special_tokens=False
            )
        return self._answer_token_cache[answer]

    # ------------------------------------------------------------
    # Answer log-probability
    # ------------------------------------------------------------

    def _logprob_of_answer(
        self,
        logits: torch.Tensor,        # [T, V]
        answer_token_ids: List[int],
        start_pos: int,
    ) -> torch.Tensor:
        """
        log P(answer | context) = sum_t log P(y_t | x, y_<t)
        """
        log_probs = F.log_softmax(logits, dim=-1)

        lp = logits.new_tensor(0.0)
        for i, tok_id in enumerate(answer_token_ids):
            pos = start_pos + i
            if pos >= log_probs.shape[0]:
                break
            lp = lp + log_probs[pos, tok_id]

        return lp

    # ------------------------------------------------------------
    # JS / CE over answers
    # ------------------------------------------------------------

    def _answer_level_loss(
        self,
        answer_logprobs: torch.Tensor,   # [K]
        human_conf: torch.Tensor         # [K]
    ) -> torch.Tensor:

        model_probs = F.softmax(answer_logprobs, dim=-1)

        if self.mode == "CE":
            return -(human_conf * torch.log(model_probs + 1e-12)).sum()

        # JS divergence
        p = human_conf.clamp(min=1e-12)
        q = model_probs.clamp(min=1e-12)
        m = 0.5 * (p + q)

        js = 0.5 * (p * (p.log() - m.log())).sum() + \
             0.5 * (q * (q.log() - m.log())).sum()
        return js

    # ------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        labels_info = inputs.pop("labels_info", None)

        with self.compute_loss_context_manager():
            outputs = model(**inputs)

        sft_loss = outputs.loss
        logits = outputs.logits        # [B, T, V]
        labels = inputs.get("labels")  # [B, T]

        # --------------------------------------------------------
        # Enforce batch_size = 1
        # --------------------------------------------------------
        assert logits.shape[0] == 1, "Answer-level JS requires batch_size = 1"

        total_loss = logits.new_tensor(0.0)

        if self.use_sft_loss:
            total_loss = total_loss + sft_loss

        # --------------------------------------------------------
        # Human alignment
        # --------------------------------------------------------
        if labels_info is not None:
            answers = labels_info["answers"]
            confidences = torch.tensor(
                labels_info["confidences"],
                device=logits.device,
                dtype=torch.float32,
            )
            confidences = confidences / confidences.sum()

            # find answer start position
            label_seq = labels[0]
            valid = (label_seq != -100).nonzero(as_tuple=True)[0]
            answer_start = valid[0].item()

            answer_logprobs = []
            for ans in answers:
                tok_ids = self._encode_answer(ans)
                lp = self._logprob_of_answer(logits[0], tok_ids, answer_start)
                answer_logprobs.append(lp)

            answer_logprobs = torch.stack(answer_logprobs)
            dist_loss = self._answer_level_loss(answer_logprobs, confidences)

            total_loss = total_loss + self.lambda_dist * dist_loss

            # ----------------------------------------------------
            # Sanity logging (first batch only)
            # ----------------------------------------------------
            if not self._first_batch_logged and self.state.is_world_process_zero:
                self._first_batch_logged = True
                model_probs = F.softmax(answer_logprobs, dim=-1)

                logger.info("=" * 60)
                logger.info("First-batch answer probabilities")
                for a, hp, mp in zip(
                    answers,
                    confidences.tolist(),
                    model_probs.tolist(),
                ):
                    logger.info(f"  {a!r:20s}  human={hp:.3f}  model={mp:.3f}")
                logger.info("=" * 60)

        # --------------------------------------------------------
        # Safety
        # --------------------------------------------------------
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning("NaN/Inf detected — falling back to SFT loss")
            total_loss = sft_loss

        self.accelerator.backward(total_loss)
        return total_loss.detach()

# ============================================================
# Arguments
# ============================================================

@dataclass
class HumanAlignmentArguments(TrainArguments):
    mode: str = field(default="JS")
    lambda_dist: float = field(default=1.0)
    use_sft_loss: bool = field(default=False)

# ============================================================
# Training entrypoint
# ============================================================

def train_human_alignment(
    model_path: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    val_data_path: Optional[str] = None,
    mode: str = "JS",
    lambda_dist: float = 1.0,
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
    logger.info("=" * 80)
    logger.info("🚀 Human Alignment Training (Answer-level JS)")
    logger.info(f"Model: {model_path}")
    logger.info(f"Mode: {mode}")
    logger.info(f"lambda_dist: {lambda_dist}")
    logger.info("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    _, tokenizer = get_model_tokenizer(
        model_path,
        use_hf=True, 
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    val_dataset = [val_data_path] if val_data_path else []

    args = HumanAlignmentArguments(
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
        use_hf=True,
        warmup_ratio=0.05,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        lora_bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        freeze_llm=False,
        freeze_vit=True,
        freeze_aligner=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        eval_steps=eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        resume_from_checkpoint=resume_from_checkpoint,
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
        use_sft_loss=use_sft_loss,
    )

    # --------------------------------------------------------
    # Inject custom trainer
    # --------------------------------------------------------

    import swift.trainers
    original_trainer = swift.trainers.Trainer

    def create_trainer(*a, **kw):
        kw["tokenizer"] = tokenizer
        kw["mode"] = args.mode
        kw["lambda_dist"] = args.lambda_dist
        kw["use_sft_loss"] = args.use_sft_loss
        return HumanAlignmentTrainer(*a, **kw)

    swift.trainers.Trainer = create_trainer
    try:
        return sft_main(args)
    finally:
        swift.trainers.Trainer = original_trainer

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser("Human Alignment Training (JS)")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, default=None)

    parser.add_argument("--mode", type=str, default="JS", choices=["JS", "CE"])
    parser.add_argument("--lambda_dist", type=float, default=1.0)
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

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    train_human_alignment(**vars(args))

if __name__ == "__main__":
    main()
