
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

def vqa_score(answer: str, human_answers: List[str]) -> float:
    a = answer.strip().lower()
    c = Counter([h.strip().lower() for h in human_answers])
    return min(c.get(a, 0) / 3.0, 1.0)

def pairwise_cosine_sims(emb: np.ndarray) -> np.ndarray:
    return emb @ emb.T

def human_semantic_agreement(answer_groups: List[List[str]], embed_fn) -> Dict[str, float]:
    pair_means = []
    for answers in answer_groups:
        if len(answers) < 2: 
            continue
        vecs = embed_fn(answers)
        S = pairwise_cosine_sims(vecs)
        n = len(answers); triu = S[np.triu_indices(n,1)]
        if triu.size:
            pair_means.append(float(triu.mean()))
    return {"pairwise_cosine_mean": float(np.mean(pair_means)) if pair_means else 0.0}
