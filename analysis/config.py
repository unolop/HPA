"""
Centralized model-list configuration shared across all export scripts.

Import from here to keep model selection consistent:
    from config import MODELS_ALL, MODELS_7B, MODEL_GROUP, MIN_ANSWERS_DEFAULT,
                       extend_pair_cache_with_yesno

All display names match the 'model' column in exports CSVs and the keys in
utils/constants.py (MODEL_FAMILY, MODEL_SIZE_B, GROUP_MARKER, etc.).
"""

# ── Default participant-qualification threshold ───────────────────────────────
MIN_ANSWERS_DEFAULT = 348

# ── All models present in the main analysis (inst_blind / blind exports) ──────
MODELS_ALL = [
    # VLM
    'Qwen3-VL-2B', 'Qwen3-VL-4B', 'Qwen3-VL-8B', 'Qwen3-VL-32B',
    'InternVL-1B', 'InternVL-2B', 'InternVL-8B',
    'LLaVA-1.5-7B', 'LLaVA-Mistral', 'LLaVA-Vicuna', 'LLaVA-Vicuna-13B',
    # VLM backbone decoder
    'Qwen3-VL-2B (LM)', 'Qwen3-VL-4B (LM)', 'Qwen3-VL-8B (LM)', 'Qwen3-VL-32B (LM)',
    'LLaVA-1.5 (LM)', 'LLaVA-Mistral (LM)', 'LLaVA-Vicuna (LM)', 'LLaVA-Vicuna-13B (LM)',
    'InternVL-1B (LM)', 'InternVL-2B (LM)', 'InternVL-8B (LM)',
    # Standalone LLM (no-think)
    'Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-32B',
    'Qwen2.5-7B', 'Qwen2.5-7B-Instruct', 'Phi-3.5-mini', 'Mistral-7B', 'Vicuna-7B', 'Vicuna-13B',
    # Standalone LLM (think)
    'Qwen3-0.6B (think)', 'Qwen3-1.7B (think)', 'Qwen3-4B (think)',
    'Qwen3-8B (think)', 'Qwen3-32B (think)',
]

# ── Subset for matched-size comparison (7–8 B parameter models only) ──────────
MODELS_7B = [
    'Qwen3-VL-8B',
    'InternVL-8B',
    'LLaVA-1.5-7B', 'LLaVA-Mistral', 'LLaVA-Vicuna',
    'Qwen3-VL-8B (LM)', 'InternVL-8B (LM)',
    'LLaVA-1.5 (LM)', 'LLaVA-Mistral (LM)', 'LLaVA-Vicuna (LM)',
    'Qwen3-8B', 'Qwen3-8B (think)',
    'Qwen2.5-7B', 'Qwen2.5-7B-Instruct', 'Mistral-7B', 'Vicuna-7B',
]

# ── Group assignment for every model in MODELS_ALL ────────────────────────────
# Must match the 'model_group' / 'subject_group_2' column in exports CSVs.
MODEL_GROUP = {
    # VLM
    'Qwen3-VL-2B':          'VLM',
    'Qwen3-VL-4B':          'VLM',
    'Qwen3-VL-8B':          'VLM',
    'Qwen3-VL-32B':         'VLM',
    'InternVL-1B':          'VLM',
    'InternVL-2B':          'VLM',
    'InternVL-8B':          'VLM',
    'LLaVA-1.5-7B':         'VLM',
    'LLaVA-Mistral':        'VLM',
    'LLaVA-Vicuna':         'VLM',
    'LLaVA-Vicuna-13B':     'VLM',
    # VLM backbone decoder
    'Qwen3-VL-2B (LM)':     'VLM backbone decoder',
    'Qwen3-VL-4B (LM)':     'VLM backbone decoder',
    'Qwen3-VL-8B (LM)':     'VLM backbone decoder',
    'Qwen3-VL-32B (LM)':    'VLM backbone decoder',
    'LLaVA-1.5 (LM)':       'VLM backbone decoder',
    'LLaVA-Mistral (LM)':   'VLM backbone decoder',
    'LLaVA-Vicuna (LM)':    'VLM backbone decoder',
    'LLaVA-Vicuna-13B (LM)':'VLM backbone decoder',
    'InternVL-1B (LM)':     'VLM backbone decoder',
    'InternVL-2B (LM)':     'VLM backbone decoder',
    'InternVL-8B (LM)':     'VLM backbone decoder',
    # Standalone LLM (no-think)
    'Qwen3-0.6B':           'standalone LLM',
    'Qwen3-1.7B':           'standalone LLM',
    'Qwen3-4B':             'standalone LLM',
    'Qwen3-8B':             'standalone LLM',
    'Qwen3-32B':            'standalone LLM',
    'Qwen2.5-7B':           'standalone LLM',
    'Qwen2.5-7B-Instruct':  'standalone LLM',
    'Phi-3.5-mini':         'standalone LLM',
    'Mistral-7B':           'standalone LLM',
    'Vicuna-7B':            'standalone LLM',
    'Vicuna-13B':           'standalone LLM',
    # Standalone LLM (think)
    'Qwen3-0.6B (think)':   'standalone LLM (think)',
    'Qwen3-1.7B (think)':   'standalone LLM (think)',
    'Qwen3-4B (think)':     'standalone LLM (think)',
    'Qwen3-8B (think)':     'standalone LLM (think)',
    'Qwen3-32B (think)':    'standalone LLM (think)',
}

# ── Short display labels for figures ─────────────────────────────────────────
MODEL_LABEL_SHORT = {
    'LLaVA-1.5 (LM)':       'LLaVA-1.5 (bb)',
    'LLaVA-Mistral (LM)':   'LLaVA-M (bb)',
    'LLaVA-Vicuna (LM)':    'LLaVA-V (bb)',
    'LLaVA-Mistral':        'LLaVA-M',
    'LLaVA-Vicuna':         'LLaVA-V',
    'LLaVA-Vicuna-13B':     'LLaVA-V-13B',
    'LLaVA-1.5-7B':         'LLaVA-1.5',
    'LLaVA-Vicuna-13B (LM)':'LLaVA-V-13B (bb)',
    'Qwen3-VL-2B (LM)':     'Qwen3-VL-2B (bb)',
    'Qwen3-VL-4B (LM)':     'Qwen3-VL-4B (bb)',
    'Qwen3-VL-8B (LM)':     'Qwen3-VL-8B (bb)',
    'Qwen3-VL-32B (LM)':    'Qwen3-VL-32B (bb)',
    'Qwen3-VL-8B':          'Qwen3-VL-8B',
    'InternVL-1B':          'IVL-1B',
    'InternVL-2B':          'IVL-2B',
    'InternVL-8B':          'IVL-8B',
    'InternVL-1B (LM)':     'IVL-1B (bb)',
    'InternVL-2B (LM)':     'IVL-2B (bb)',
    'InternVL-8B (LM)':     'IVL-8B (bb)',
    'Qwen3-8B':             'Qwen3-8B',
    'Qwen3-8B (think)':     'Qwen3-8B-T',
    'Qwen3-32B':            'Qwen3-32B',
    'Qwen3-32B (think)':    'Qwen3-32B-T',
    'Qwen3-4B':             'Qwen3-4B',
    'Qwen3-4B (think)':     'Qwen3-4B-T',
    'Qwen3-1.7B':           'Qwen3-1.7B',
    'Qwen3-1.7B (think)':   'Qwen3-1.7B-T',
    'Qwen3-0.6B':           'Qwen3-0.6B',
    'Qwen3-0.6B (think)':   'Qwen3-0.6B-T',
    'Qwen2.5-7B':           'Qwen2.5-7B',
    'Qwen2.5-7B-Instruct':  'Qwen2.5-7B-IT',
    'Phi-3.5-mini':         'Phi-3.5',
    'Mistral-7B':           'Mistral-7B',
    'Vicuna-7B':            'Vicuna-7B',
    'Vicuna-13B':           'Vicuna-13B',
}


# ─────────────────────────────────────────────────────────────────────────────
# Pair-cache extension: merge yes/no question pairs with all available metrics
# ─────────────────────────────────────────────────────────────────────────────

def extend_pair_cache_with_yesno(pair_df, exports):
    """
    Merge answer_pairs_yesno.csv into pair_df.

    Computes:
      exact_score   — from preprocess_answer normalization
      jaccard_score — token Jaccard
      rouge1_score  — unigram F1
      chrf_score    — character n-gram F1 (sacrebleu, beta=2)
      sbert_score / sbert_score_clip — from embeddings_all-mpnet-base-v2.npz cache
      simcse_score / simcse_score_clip — from embeddings_sup-simcse-roberta-large.npz cache
      bertscore_f1  — NaN (BERTScore not computed for yes/no)

    Parameters
    ----------
    pair_df : pd.DataFrame  existing pair_cache (text questions only)
    exports : Path          path to analysis/session2/exports/

    Returns
    -------
    pd.DataFrame  concatenated (text + yesno) pair data
    """
    import numpy as np
    import pandas as pd
    from collections import Counter
    from pathlib import Path

    exports = Path(exports)

    # ── Imports that may not be top-level in every calling script ─────────────
    import sys as _sys
    _root = str(exports.parent.parent.parent)  # HPA root
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    if str(exports.parent.parent.parent / 'analysis') not in _sys.path:
        _sys.path.insert(0, str(exports.parent.parent.parent / 'analysis'))

    from utils.vqa import preprocess_answer
    from utils.agreement import load_embedding_cache

    # ── Load raw yes/no pairs ─────────────────────────────────────────────────
    yesno = pd.read_csv(exports / 'answer_pairs_yesno.csv')

    # ── Normalize answers ─────────────────────────────────────────────────────
    a1 = yesno['answer_1'].fillna('').apply(preprocess_answer)
    a2 = yesno['answer_2'].fillna('').apply(preprocess_answer)

    # ── Exact ─────────────────────────────────────────────────────────────────
    yesno['exact_score'] = (a1 == a2).astype(float)

    # ── Jaccard ───────────────────────────────────────────────────────────────
    def _jaccard(s1, s2):
        t1, t2 = set(s1.split()), set(s2.split())
        union = t1 | t2
        return len(t1 & t2) / len(union) if union else 1.0

    yesno['jaccard_score'] = [_jaccard(s1, s2) for s1, s2 in zip(a1, a2)]

    # ── ROUGE-1 (unigram F1) ──────────────────────────────────────────────────
    def _rouge1(s1, s2):
        t1, t2 = s1.split(), s2.split()
        if not t1 or not t2:
            return 0.0
        c1, c2 = Counter(t1), Counter(t2)
        m = sum((c1 & c2).values())
        p, r = m / len(t1), m / len(t2)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    yesno['rouge1_score'] = [_rouge1(s1, s2) for s1, s2 in zip(a1, a2)]

    # ── chrF (sacrebleu, beta=2) ──────────────────────────────────────────────
    try:
        from sacrebleu.metrics import CHRF
        _chrf = CHRF(beta=2)
        yesno['chrf_score'] = [
            _chrf.sentence_score(s1, [s2]).score / 100.0
            for s1, s2 in zip(a1, a2)
        ]
    except Exception:
        yesno['chrf_score'] = np.nan

    # ── Cosine similarity from embedding cache ────────────────────────────────
    def _cosine_from_cache(cache, s1_col, s2_col):
        scores = []
        for s1, s2 in zip(s1_col, s2_col):
            v1, v2 = cache.get(s1), cache.get(s2)
            if v1 is None or v2 is None:
                scores.append(np.nan)
            else:
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                scores.append(float(np.dot(v1, v2) / (n1 * n2)) if n1 and n2 else 0.0)
        return np.array(scores)

    for metric, fname, raw_col, clip_col in [
        ('sbert',  'embeddings_all-mpnet-base-v2.npz',
         'sbert_score',  'sbert_score_clip'),
        ('simcse', 'embeddings_sup-simcse-roberta-large.npz',
         'simcse_score', 'simcse_score_clip'),
    ]:
        cache_path = exports / fname
        if cache_path.exists():
            cache = load_embedding_cache(cache_path)
            scores = _cosine_from_cache(cache, a1, a2)
            yesno[raw_col]  = scores
            yesno[clip_col] = np.where(np.isnan(scores), np.nan, np.clip(scores, 0, None))
        else:
            yesno[raw_col]  = np.nan
            yesno[clip_col] = np.nan

    # BERTScore not computed for yes/no (too expensive + not meaningful)
    yesno['bertscore_f1'] = np.nan

    # ── Align columns with pair_df then concatenate ───────────────────────────
    for col in pair_df.columns:
        if col not in yesno.columns:
            yesno[col] = np.nan

    return pd.concat([pair_df, yesno[pair_df.columns]], ignore_index=True)
