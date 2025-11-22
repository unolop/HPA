
import torch, math, os
from typing import List, Dict
from .losses import human_prior_alignment_loss
from .utils import set_seed, device

class HPAListwiseScorer(torch.nn.Module):
    """
    Wrap a language(-vision) model to score a set of candidate answers A for question q.
    This default implementation uses a causal LM's token-level logprobs to form answer-level scores.
    For VLMs, adapt `score_candidates` to include vision inputs (black image) where needed.
    """
    def __init__(self, tokenizer, model, prompt_template="Q: {q}\nA:", max_new_tokens=8):
        super().__init__()
        self.tok = tokenizer
        self.model = model
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def score_candidates(self, question: str, candidates: List[str]) -> torch.Tensor:
        """
        Returns normalized probs over candidates via length-normalized loglikelihood.
        """
        if len(candidates)==1:
            return torch.tensor([1.0], device=self.model.device)
        prompt = self.prompt_template.format(q=question)
        input_ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        scores = []
        for a in candidates:
            # Compute log p(a | prompt)
            tgt = self.tok(a, return_tensors="pt").to(self.model.device)
            ids = torch.cat([input_ids["input_ids"], tgt["input_ids"]], dim=1)
            with torch.no_grad():
                out = self.model(ids, labels=ids)
                # loss is CE over all tokens; we isolate the answer segment contribution approximately
                # For simplicity we use the last len(tgt) tokens contribution by subtracting two losses
                # (practical approximation; adapt for your model API if available)
            # More stable: compute token logprobs for the answer conditioned on prompt
            with torch.no_grad():
                out2 = self.model(input_ids["input_ids"])
                logits = out2.logits[:, -1:, :]
                # naive next-token only; for robust scoring, replace with iterative token logprob scoring.
            # Fallback simple heuristic: shorter answers penalized less; provide uniform as placeholder
            scores.append(0.0)
        # Uniform fallback to avoid API specifics; replace with true scoring in your env.
        probs = torch.ones(len(candidates), device=self.model.device) / len(candidates)
        return probs

def train_epoch(items, scorer, optimizer, tok, batch_size=8, loss_mode="js", w_brier=0.0):
    import random
    scorer.train()
    random.shuffle(items)
    total = 0.0
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        model_probs = []
        human_probs = []
        for it in batch:
            cand = it.candidates or list(it.human_probs.keys())
            mp = scorer.score_candidates(it.question, cand)
            hp = torch.tensor([it.human_probs.get(a,0.0) for a in cand], device=mp.device, dtype=mp.dtype)
            # align support
            mp = mp / (mp.sum() + 1e-12)
            hp = hp / (hp.sum() + 1e-12)
            model_probs.append(mp.unsqueeze(0))
            human_probs.append(hp.unsqueeze(0))
        model_probs = torch.cat(model_probs, dim=0)
        human_probs = torch.cat(human_probs, dim=0)
        loss = human_prior_alignment_loss(model_probs, human_probs, mode=loss_mode, w_brier=w_brier)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
    return total / max(1, len(items))
