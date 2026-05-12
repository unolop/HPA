"""
Inter-participant free-text agreement metrics for VQA answers.

Three methods, all return a scalar in [0, 1] per question:
  - exact_match_agreement : fraction of pairs with identical normalized text
  - jaccard_agreement     : mean token Jaccard over all pairs
  - sbert_agreement       : mean cosine similarity from sentence embeddings

Usage
-----
from utils.agreement import compute_question_agreement

agree_df = compute_question_agreement(df, common_qids, methods=['exact', 'jaccard', 'sbert'])
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from analysis.utils.vqa import preprocess_answer


def _normalize(text: str) -> str:
    return preprocess_answer(text)


def clip_cosine_score(score: float, clip_min: Optional[float] = None) -> float:
    """Optionally clamp cosine similarity to a lower bound."""
    if clip_min is None:
        return float(score)
    return float(max(score, clip_min))


# ── Per-pair metrics ──────────────────────────────────────────────────────────

def _pair_exact(a: str, b: str) -> float:
    return float(_normalize(a) == _normalize(b))


def _pair_jaccard(a: str, b: str) -> float:
    wa = set(_normalize(a).split())
    wb = set(_normalize(b).split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _pair_rouge1(a: str, b: str) -> float:
    """ROUGE-1 F1: unigram precision/recall F-score between two strings."""
    wa = Counter(a.split())
    wb = Counter(b.split())
    total_a = sum(wa.values())
    total_b = sum(wb.values())
    if total_a == 0 and total_b == 0:
        return 1.0
    if total_a == 0 or total_b == 0:
        return 0.0
    matches = sum((wa & wb).values())
    prec = matches / total_a
    rec  = matches / total_b
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _char_ngrams(text: str, n: int) -> Counter:
    return Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))


def _pair_chrf(a: str, b: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF: mean character n-gram F-score over n=1..max_n.

    beta=2 weights recall more than precision (sacrebleu default).
    Inputs should already be normalized/lowercased.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prec_sum = rec_sum = n_counted = 0
    for n in range(1, max_n + 1):
        ng_a    = _char_ngrams(a, n)
        ng_b    = _char_ngrams(b, n)
        total_a = sum(ng_a.values())
        total_b = sum(ng_b.values())
        if total_a == 0 and total_b == 0:
            continue
        matches  = sum((ng_a & ng_b).values())
        prec_sum += matches / total_a if total_a else 0.0
        rec_sum  += matches / total_b if total_b else 0.0
        n_counted += 1
    if n_counted == 0:
        return 0.0
    avg_prec = prec_sum / n_counted
    avg_rec  = rec_sum  / n_counted
    denom = beta ** 2 * avg_prec + avg_rec
    if denom == 0:
        return 0.0
    return (1 + beta ** 2) * avg_prec * avg_rec / denom


# ── Aggregate over a list of answers ─────────────────────────────────────────

def exact_match_agreement(answers: List[str]) -> float:
    """Fraction of participant pairs with identical normalized answers."""
    answers = [a for a in answers if a and str(a).strip()]
    if len(answers) < 2:
        return np.nan
    pairs = list(combinations(answers, 2))
    return float(np.mean([_pair_exact(a, b) for a, b in pairs]))


def jaccard_agreement(answers: List[str]) -> float:
    """Mean token Jaccard similarity over all participant pairs."""
    answers = [a for a in answers if a and str(a).strip()]
    if len(answers) < 2:
        return np.nan
    pairs = list(combinations(answers, 2))
    return float(np.mean([_pair_jaccard(a, b) for a, b in pairs]))


def rouge1_f1_agreement(answers: List[str]) -> float:
    """Mean ROUGE-1 F1 over all participant pairs."""
    answers = [_normalize(a) for a in answers if a and str(a).strip()]
    if len(answers) < 2:
        return np.nan
    pairs = list(combinations(answers, 2))
    return float(np.mean([_pair_rouge1(a, b) for a, b in pairs]))


def chrf_agreement(answers: List[str], max_n: int = 6, beta: float = 2.0) -> float:
    """Mean chrF (character n-gram F-score) over all participant pairs."""
    answers = [_normalize(a) for a in answers if a and str(a).strip()]
    if len(answers) < 2:
        return np.nan
    pairs = list(combinations(answers, 2))
    return float(np.mean([_pair_chrf(a, b, max_n=max_n, beta=beta) for a, b in pairs]))


def sbert_agreement(answers: List[str], model=None, clip_min: Optional[float] = None) -> float:
    """Mean pairwise cosine similarity from SBERT embeddings."""
    answers = [_normalize(a) for a in answers if a and str(a).strip()]
    if len(answers) < 2:
        return np.nan
    if model is None:
        raise ValueError('Pass a loaded SentenceTransformer model as `model`.')
    import torch
    embs = model.encode(answers, convert_to_tensor=True, show_progress_bar=False)
    embs = embs / embs.norm(dim=1, keepdim=True)
    sim_matrix = (embs @ embs.T).cpu().numpy()
    n = len(answers)
    upper = [clip_cosine_score(sim_matrix[i, j], clip_min=clip_min) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(upper))


# ── Main coordinator ──────────────────────────────────────────────────────────

def compute_question_agreement(
    df: pd.DataFrame,
    qids: set,
    methods: List[str] = ('exact', 'jaccard', 'sbert'),
    answer_col: str = 'answer_en',
    variant: str = 'C',
    sbert_model_name: str = 'all-mpnet-base-v2',
    sbert_clip_min: Optional[float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute inter-participant agreement for each question.

    Parameters
    ----------
    df            : human responses DataFrame (one row per participant × question × variant)
    qids          : question_ids to compute agreement for
    methods       : subset of {'exact', 'jaccard', 'sbert'}
    answer_col    : column containing the answer text
    variant       : which variant to analyse (default 'C')
    sbert_model_name : HuggingFace model id for sentence-transformers
    sbert_clip_min: optional lower bound for SBERT cosine before averaging
    verbose       : print progress

    Returns
    -------
    DataFrame indexed by question_id with one column per method.
    """
    sub = df[(df['variant'] == variant) & df['question_id'].isin(qids)].copy()

    sbert_model = None
    if 'sbert' in methods:
        from sentence_transformers import SentenceTransformer
        if verbose:
            print(f'Loading SBERT model: {sbert_model_name} …')
        sbert_model = SentenceTransformer(sbert_model_name)

    rows = []
    qid_list = sorted(sub['question_id'].unique())
    for qid in qid_list:
        answers = [_normalize(a) for a in sub[sub['question_id'] == qid][answer_col].dropna().tolist()]
        row = {'question_id': qid}
        if 'exact'   in methods: row['exact']   = exact_match_agreement(answers)
        if 'jaccard' in methods: row['jaccard'] = jaccard_agreement(answers)
        if 'rouge1'  in methods: row['rouge1']  = rouge1_f1_agreement(answers)
        if 'chrf'    in methods: row['chrf']    = chrf_agreement(answers)
        if 'sbert'   in methods: row['sbert']   = sbert_agreement(answers, sbert_model, clip_min=sbert_clip_min)
        rows.append(row)

    result = pd.DataFrame(rows).set_index('question_id')
    if verbose:
        print(f'Agreement computed for {len(result)} questions (variant={variant})')
        for m in [c for c in result.columns]:
            print(f'  {m}: mean={result[m].mean():.3f}  std={result[m].std():.3f}')
    return result
