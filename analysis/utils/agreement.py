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

import re
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd

_PUNCT = re.compile(r'[^\w\s]')
_SPACE = re.compile(r'\s+')


def _normalize(text: str) -> str:
    t = str(text).lower().strip()
    t = _PUNCT.sub(' ', t)
    t = _SPACE.sub(' ', t).strip()
    return t


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


def sbert_agreement(answers: List[str], model=None) -> float:
    """Mean pairwise cosine similarity from SBERT embeddings."""
    answers = [a for a in answers if a and str(a).strip()]
    if len(answers) < 2:
        return np.nan
    if model is None:
        raise ValueError('Pass a loaded SentenceTransformer model as `model`.')
    import torch
    embs = model.encode(answers, convert_to_tensor=True, show_progress_bar=False)
    embs = embs / embs.norm(dim=1, keepdim=True)
    sim_matrix = (embs @ embs.T).cpu().numpy()
    n = len(answers)
    upper = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(upper))


# ── Main coordinator ──────────────────────────────────────────────────────────

def compute_question_agreement(
    df: pd.DataFrame,
    qids: set,
    methods: List[str] = ('exact', 'jaccard', 'sbert'),
    answer_col: str = 'answer_en',
    variant: str = 'C',
    sbert_model_name: str = 'all-MiniLM-L6-v2',
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
        answers = sub[sub['question_id'] == qid][answer_col].dropna().tolist()
        row = {'question_id': qid}
        if 'exact'   in methods: row['exact']   = exact_match_agreement(answers)
        if 'jaccard' in methods: row['jaccard'] = jaccard_agreement(answers)
        if 'sbert'   in methods: row['sbert']   = sbert_agreement(answers, sbert_model)
        rows.append(row)

    result = pd.DataFrame(rows).set_index('question_id')
    if verbose:
        print(f'Agreement computed for {len(result)} questions (variant={variant})')
        for m in [c for c in result.columns]:
            print(f'  {m}: mean={result[m].mean():.3f}  std={result[m].std():.3f}')
    return result
