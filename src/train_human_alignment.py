import torch
import torch.nn.functional as F
import gc
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import argparse

from swift.llm import sft_main, TrainArguments, get_model_tokenizer
from swift.utils import get_logger
from swift.trainers import Trainer

logger = get_logger()


class HumanAlignmentTrainer(Trainer):
    """
    Semantically aligned Human Alignment Trainer.

    JS / CE is computed over ANSWERS, not vocabulary tokens.
    Works for:
      - Blind VQA (free-form answers)
      - MMStar (A/B/C/D)
    """

    def __init__(
        self,
        *args,
        tokenizer=None,
        mode: str = "JS",          # "JS" or "CE"
        lambda_dist: float = 1.0,
        use_sft_loss: bool = False,
        use_hf: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.mode = mode
        self.lambda_dist = lambda_dist
        self.use_sft_loss = use_sft_loss
        self.use_hf = use_hf

        self._answer_token_cache: Dict[str, List[int]] = {}
        self._first_batch_logged = False

        logger.info("[HumanAlignmentTrainer] Semantically aligned JS")
        logger.info(f"  mode={mode}, lambda_dist={lambda_dist}, use_sft_loss={use_sft_loss}, use_hf={use_hf}")

    # ------------------------------------------------------------------
    # Token utilities
    # ------------------------------------------------------------------

    def _encode_answer(self, answer: str) -> List[int]:
        if answer not in self._answer_token_cache:
            self._answer_token_cache[answer] = self.tokenizer.encode(
                answer, add_special_tokens=False
            )
        return self._answer_token_cache[answer]

    # ------------------------------------------------------------------
    # Core: answer-level probability
    # ------------------------------------------------------------------

    def _logprob_of_answer(
        self,
        logits: torch.Tensor,      # [seq, vocab]
        answer_token_ids: List[int],
        answer_start_pos: int
    ) -> torch.Tensor:
        """
        Compute log P(answer | context) by summing token log-probs.
        """
        log_probs = F.log_softmax(logits, dim=-1)

        lp = 0.0
        for i, tok_id in enumerate(answer_token_ids):
            pos = answer_start_pos + i
            if pos >= log_probs.shape[0]:
                break
            lp = lp + log_probs[pos, tok_id]

        return lp

    # ------------------------------------------------------------------
    # Distributional loss (answer-level)
    # ------------------------------------------------------------------

    def _compute_answer_js(
        self,
        answer_logprobs: torch.Tensor,   # [num_answers]
        human_conf: torch.Tensor         # [num_answers]
    ) -> torch.Tensor:
        """
        JS or CE over answers.
        """
        model_probs = F.softmax(answer_logprobs, dim=-1)

        if self.mode == "CE":
            return -(human_conf * torch.log(model_probs + 1e-12)).sum()

        # JS
        p = human_conf.clamp(min=1e-12)
        q = model_probs.clamp(min=1e-12)
        m = 0.5 * (p + q)

        js = 0.5 * (p * (p.log() - m.log())).sum() + \
             0.5 * (q * (q.log() - m.log())).sum()
        return js

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        labels_info = inputs.pop("labels_info", None)

        with self.compute_loss_context_manager():
            outputs = model(**inputs)

        sft_loss = outputs.loss
        logits = outputs.logits      # [B, T, V]
        labels = inputs.get("labels")

        total_loss = torch.tensor(0.0, device=logits.device)

        if self.use_sft_loss:
            total_loss = total_loss + sft_loss

        if labels_info is not None:
            assert logits.shape[0] == 1, "Answer-level JS currently assumes batch_size=1"

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
                lp = self._logprob_of_answer(
                    logits[0], tok_ids, answer_start
                )
                answer_logprobs.append(lp)

            answer_logprobs = torch.stack(answer_logprobs)
            dist_loss = self._compute_answer_js(answer_logprobs, confidences)

            total_loss = total_loss + self.lambda_dist * dist_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning("NaN/Inf detected, falling back to SFT loss")
            total_loss = sft_loss

        self.accelerator.backward(total_loss)
        return total_loss.detach()

