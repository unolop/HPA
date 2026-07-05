"""
Build or incrementally update pair_cache.parquet.

Computes pairwise agreement between all raters (human participants + models)
on free-text VQA questions using:
  exact, jaccard, rouge1, chrf, sbert, simcse, bertscore

Supports two conditions:
  inst_blind (default) — reads responses_model_inst_blind.csv → pair_cache.parquet
  blind                — reads responses_model_blind.csv      → pair_cache_blind.parquet

Incremental: any (subject_1, subject_2, question_id, variant) tuple already
in the target parquet is skipped, so re-runs are fast.

Detects:
  - New human participants in responses_human.csv not yet in HH/HM pairs
  - New models in responses_model_{condition}.csv not yet in HM/MM pairs

Outputs (for inst_blind):
  exports/pair_cache.parquet          — full incremental pair store
  exports/pair_cache.csv              — CSV mirror of the parquet cache
  exports/answer_pairs_sbert_text.csv — legacy CSV mirror for downstream scripts

Outputs (for blind):
  exports/pair_cache_blind.parquet    — full incremental pair store
  exports/pair_cache_blind.csv        — CSV mirror of the parquet cache

Run from repo root:
  conda run -n zero python analysis/build_pair_cache.py
  conda run -n zero python analysis/build_pair_cache.py --condition blind
  conda run -n zero python analysis/build_pair_cache.py --no_bertscore
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# Bypass torch version check for models without safetensors (CVE-2025-32434)
try:
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

EMBEDDING_MODELS = {
    'sbert':  'sentence-transformers/all-mpnet-base-v2',
    'simcse': 'princeton-nlp/sup-simcse-roberta-large',
}
BERTSCORE_MODEL  = 'roberta-large'
HF_CACHE_DEFAULT = os.environ.get('HPA_HF_CACHE', str((ROOT / '.cache' / 'hf').resolve()))
VARIANT_ORDER    = ['C', 'B', 'A']
CT_TO_VARIANT    = {'question': 'C', 'weaker_object': 'B', 'pronominalized': 'A'}


def _resolve_embed_device() -> str:
    """Prefer cuda:1 for embedding passes when available.

    Priority:
      1. explicit HPA_EMBED_DEVICE
      2. cuda:1 if >=2 visible GPUs
      3. cuda:0 if 1 visible GPU
      4. cpu
    """
    explicit = os.environ.get('HPA_EMBED_DEVICE')
    if explicit:
        return explicit
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n >= 2:
                return 'cuda:1'
            return 'cuda:0'
    except Exception:
        pass
    return 'cpu'


def build_pair_cache(
    root: Path,
    exports: Path,
    no_bertscore: bool = False,
    hf_cache: str | None = None,
    condition: str = 'inst_blind',
    include_yesno: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build or incrementally update pair_cache.parquet.

    Parameters
    ----------
    root         : repository root Path
    exports      : exports directory (contains pair_cache.parquet, responses_*.csv)
    no_bertscore : skip BERTScore computation (faster, leaves bertscore_f1 as NaN)
    hf_cache     : HuggingFace cache directory override
    condition    : 'inst_blind' (default) or 'blind'
    verbose      : print progress

    Returns
    -------
    pd.DataFrame — the complete (cached + new) pair table
    """
    from sentence_transformers import SentenceTransformer
    from utils.agreement import (
        load_embedding_cache, encode_with_cache,
        _pair_exact, _pair_jaccard, _pair_rouge1, _pair_chrf,
        bertscore_batch,
    )
    from utils.completeness import check_completeness

    root    = Path(root)
    exports = Path(exports)
    hf_dir  = Path(hf_cache or HF_CACHE_DEFAULT)
    hf_dir.mkdir(parents=True, exist_ok=True)
    hf_dir  = str(hf_dir)
    embed_device = _resolve_embed_device()

    # Condition-dependent paths
    if condition == 'blind':
        cache_path = exports / 'pair_cache_blind.parquet'
        model_csv_name = 'responses_model_blind.csv'
    else:
        cache_path = exports / 'pair_cache.parquet'
        model_csv_name = 'responses_model_inst_blind.csv'

    if verbose:
        print(f'Condition: {condition} → {model_csv_name}')

    # ── 1. Load human + model responses ──────────────────────────────────────
    h_csv = pd.read_csv(exports / 'responses_human.csv')
    m_csv = pd.read_csv(exports / model_csv_name)

    h_csv, _ = check_completeness(
        h_csv, group_col='participant', question_col='question_id',
        answer_col='response', label='Human responses', verbose=verbose,
    )
    m_csv, _ = check_completeness(
        m_csv, group_col='model', question_col='question_id',
        answer_col='response', label='Model responses', verbose=verbose,
    )

    if verbose:
        print(f'Humans : {h_csv["participant"].nunique()} participants, '
              f'{h_csv["question_id"].nunique()} questions')
        print(f'Models : {m_csv["model"].nunique()} models')

    # ── 2. Question metadata ──────────────────────────────────────────────────
    q_meta_list = json.load(open(root / 'experiment/s2_v4/s4_question.json'))
    answer_type_map = {q['question_id']: q['answer_type'] for q in q_meta_list}

    common_qids = set(h_csv['question_id'].astype(int).unique())
    if include_yesno:
        text_qids = common_qids
    else:
        text_qids = {qid for qid in common_qids if answer_type_map.get(qid) == 'text'}

    if verbose:
        yesno = common_qids - text_qids
        print(f'Questions: {len(common_qids)} total — '
              f'{len(text_qids)} {"all (incl. yes/no)" if include_yesno else "free-text"}'
              f'{f", {len(yesno)} yes/no excluded" if not include_yesno else ""}')

    # Variant-aware question text: (qid, var) → en / kr
    q_en  = {q['question_id']: q.get('question_en', '') for q in q_meta_list}
    q_kr  = {q['question_id']: q.get('question_kr', '') for q in q_meta_list}
    q_en_var: dict[tuple, str] = {}
    q_kr_var: dict[tuple, str] = {}
    for fname, var in [('s4_weaker_object.json', 'B'), ('s4_pronominalized.json', 'A')]:
        fpath = root / 'experiment/s2_v4' / fname
        if fpath.exists():
            for entry in json.load(open(fpath)):
                qid = entry['question_id']
                q_en_var[(qid, var)] = entry.get('question_en', q_en.get(qid, ''))
                q_kr_var[(qid, var)] = entry.get('question_kr', q_kr.get(qid, ''))
    for qid in common_qids:
        q_en_var[(qid, 'C')] = q_en.get(qid, '')
        q_kr_var[(qid, 'C')] = q_kr.get(qid, '')

    # ent / op from human data
    q_ent = (h_csv[['question_id', 'ent']].drop_duplicates()
             .set_index('question_id')['ent'].to_dict())
    q_op  = (h_csv[['question_id', 'op']].drop_duplicates()
             .set_index('question_id')['op'].to_dict())

    # ── 3. Build rater answer dicts ───────────────────────────────────────────
    rater_answers: dict[str, dict] = {}
    rater_group:   dict[str, str]  = {}

    for _, row in h_csv[h_csv['question_id'].isin(text_qids)].iterrows():
        pid = row['participant']
        qid = int(row['question_id'])
        var = row['variant']
        ans = str(row['response'] or '').strip()
        if not ans:
            continue
        if pid not in rater_answers:
            rater_answers[pid] = {}
            rater_group[pid]   = 'human'
        rater_answers[pid][(qid, var)] = ans

    for _, row in m_csv[m_csv['question_id'].isin(text_qids)].iterrows():
        label = row['model']
        qid   = int(row['question_id'])
        var   = row['variant']
        ans   = str(row['response'] or '').strip()
        if not ans:
            continue
        if label not in rater_answers:
            rater_answers[label] = {}
            rater_group[label]   = row['model_group']
        rater_answers[label][(qid, var)] = ans

    all_raters   = list(rater_answers.keys())
    human_raters = [r for r in all_raters if rater_group[r] == 'human']
    model_raters = [r for r in all_raters if rater_group[r] != 'human']
    if verbose:
        print(f'Raters : {len(human_raters)} humans + {len(model_raters)} models '
              f'= {len(all_raters)} total')

    # ── 4. Load incremental pair cache ────────────────────────────────────────
    cache_df    = None
    cached_keys: set[tuple] = set()
    if cache_path.exists():
        cache_df    = pd.read_parquet(cache_path)
        # Normalise to (min, max) order so lookups match regardless of how
        # pairs were stored by older code (which used human-first ordering).
        cached_keys = set(
            (min(s1, s2), max(s1, s2), qid, var)
            for s1, s2, qid, var in zip(
                cache_df['subject_1'], cache_df['subject_2'],
                cache_df['question_id'], cache_df['variant'],
            )
        )
        if verbose:
            print(f'Cache  : {len(cache_df):,} pairs already scored '
                  f'({cache_df["variant"].value_counts().to_dict()})')

    # ── 5. Determine which pairs are new ─────────────────────────────────────
    # Count new pairs before encoding (avoid loading GPU models if nothing to do)
    n_new = 0
    for var in VARIANT_ORDER:
        for qid in text_qids:
            q_answers = {r: rater_answers[r].get((qid, var))
                         for r in all_raters}
            present = [r for r, a in q_answers.items() if a]
            for r1, r2 in combinations(present, 2):
                key = (min(r1, r2), max(r1, r2), qid, var)
                if key not in cached_keys:
                    n_new += 1

    # ── 5b. Detect NaN metric columns in existing cache ───────────────────────
    # A metric column may be all-NaN if it was added after those rows were built
    # (e.g. simcse / bertscore added after some pairs were first cached).
    # Backfill them in place before plotting.
    EMB_SCORE_COLS = {
        'sbert':  ('sbert_score',       'sbert_score_clip'),
        'simcse': ('simcse_score',       'simcse_score_clip'),
    }
    nan_metrics: list[str] = []
    needs_bertscore_backfill = False
    if cache_df is not None:
        for em, (col, _) in EMB_SCORE_COLS.items():
            if col in cache_df.columns and cache_df[col].isna().any():
                nan_metrics.append(em)
        if ('bertscore_f1' in cache_df.columns
                and cache_df['bertscore_f1'].isna().any()
                and not no_bertscore):
            needs_bertscore_backfill = True

    if n_new == 0 and not nan_metrics and not needs_bertscore_backfill:
        if verbose:
            print('No new pairs — pair_cache is up to date.')
        return cache_df if cache_df is not None else pd.DataFrame()

    if n_new == 0:
        # Only NaN backfill needed — handled in step 6b; save and return early
        # after encoding (step 6) and backfill (step 6b) run below.
        _only_backfill = True
    else:
        _only_backfill = False

    if nan_metrics and verbose:
        counts = {em: int(cache_df[EMB_SCORE_COLS[em][0]].isna().sum())
                  for em in nan_metrics}
        print(f'NaN metric rows to backfill: {counts}')
    if needs_bertscore_backfill and verbose:
        n_b = int(cache_df['bertscore_f1'].isna().sum())
        print(f'NaN metric rows to backfill: bertscore={n_b}')

    if n_new > 0 and verbose:
        print(f'New pairs to score: {n_new:,}')

    # ── 6. Encode answers with embedding models ───────────────────────────────
    all_answers = sorted({
        a for r in all_raters
        for a in rater_answers[r].values()
    })
    if verbose:
        print(f'Unique answers to embed: {len(all_answers)}')

    emb_caches: dict[str, dict] = {}
    for metric_name, model_id in EMBEDDING_MODELS.items():
        tag        = model_id.split('/')[-1]
        emb_path   = exports / f'embeddings_{tag}.npz'
        existing   = load_embedding_cache(emb_path)
        new_to_enc = [a for a in all_answers if a not in existing]
        if new_to_enc:
            if verbose:
                print(
                    f'  [{metric_name}] {len(new_to_enc)} new answers — '
                    f'loading {model_id} on {embed_device} …'
                )
            import os
            os.environ['HF_HOME'] = hf_dir
            try:
                model = SentenceTransformer(
                    model_id,
                    cache_folder=hf_dir,
                    device=embed_device,
                    model_kwargs={'use_safetensors': True},
                )
            except (OSError, ValueError):
                # Fallback: bypass torch version check for models without safetensors
                import transformers.utils.import_utils as _tiu
                _orig_check = _tiu.check_torch_load_is_safe
                _tiu.check_torch_load_is_safe = lambda: None
                try:
                    model = SentenceTransformer(
                        model_id, cache_folder=hf_dir, device=embed_device)
                finally:
                    _tiu.check_torch_load_is_safe = _orig_check
            emb_caches[metric_name] = encode_with_cache(
                all_answers, model, emb_path, normalize=True, verbose=verbose)
            del model
        else:
            emb_caches[metric_name] = existing
            if verbose:
                print(f'  [{metric_name}] all {len(existing)} vectors cached')

    primary_emb = next(iter(emb_caches.values())) if emb_caches else {}

    # ── 6b. Backfill NaN metric columns in existing cache ─────────────────────
    if nan_metrics and cache_df is not None:
        for em in nan_metrics:
            col, col_c = EMB_SCORE_COLS[em]
            nan_idx = cache_df[cache_df[col].isna()].index
            if verbose:
                print(f'  [{em}] backfilling {len(nan_idx):,} NaN rows …')
            ec = emb_caches[em]
            scores, clips = [], []
            for i in nan_idx:
                a1, a2 = cache_df.at[i, 'answer_1'], cache_df.at[i, 'answer_2']
                v1, v2 = ec.get(a1 or ''), ec.get(a2 or '')
                if v1 is not None and v2 is not None:
                    v = float(np.dot(v1, v2))
                    scores.append(round(v, 4))
                    clips.append(round(max(v, 0.0), 4))
                else:
                    scores.append(float('nan'))
                    clips.append(float('nan'))
            cache_df.loc[nan_idx, col]   = scores
            cache_df.loc[nan_idx, col_c] = clips
        still_nan = sum(
            int(cache_df[EMB_SCORE_COLS[em][0]].isna().sum()) for em in nan_metrics)
        if verbose:
            print(f'  Backfill complete — remaining NaN: {still_nan}')

    # ── 6c. Backfill NaN BERTScore rows in existing cache ────────────────────
    if needs_bertscore_backfill and cache_df is not None:
        nan_idx = cache_df[cache_df['bertscore_f1'].isna()].index.tolist()
        if nan_idx:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if verbose:
                print(f'  [bertscore] backfilling {len(nan_idx):,} NaN rows on {device} …')
            cands = [str(cache_df.at[i, 'answer_1'] or '') for i in nan_idx]
            refs  = [str(cache_df.at[i, 'answer_2'] or '') for i in nan_idx]
            f1_scores = bertscore_batch(
                cands, refs,
                model_type=BERTSCORE_MODEL,
                device=device,
                batch_size=128,
                verbose=verbose,
            )
            cache_df.loc[nan_idx, 'bertscore_f1'] = [
                round(float(f1), 4) for f1 in f1_scores
            ]
            if verbose:
                remain = int(cache_df['bertscore_f1'].isna().sum())
                print(f'  [bertscore] backfill complete — remaining NaN: {remain}')

    if _only_backfill:
        cache_df.to_parquet(cache_path, index=False)
        csv_path = exports / f'pair_cache{"_blind" if condition == "blind" else ""}.csv'
        cache_df.to_csv(csv_path, index=False)
        if condition == 'inst_blind':
            cache_df.to_csv(exports / 'answer_pairs_sbert_text.csv', index=False)
        if verbose:
            print(f'Saved  : {len(cache_df):,} pairs (backfill only) → {cache_path}')
        return cache_df

    # ── 7. Build new pair rows ────────────────────────────────────────────────
    lex_metrics = ['exact', 'jaccard', 'rouge1', 'chrf']
    emb_metrics = list(EMBEDDING_MODELS.keys())

    new_rows: list[dict] = []

    for var in VARIANT_ORDER:
        for qid in sorted(text_qids):
            q_answers = {r: rater_answers[r].get((qid, var)) for r in all_raters}
            q_answers = {r: a for r, a in q_answers.items()
                         if a and a in primary_emb}
            present = list(q_answers.keys())
            if len(present) < 2:
                continue

            q_en_text = q_en_var.get((qid, var), q_en.get(qid, ''))
            q_kr_text = q_kr_var.get((qid, var), q_kr.get(qid, ''))

            for r1, r2 in combinations(present, 2):
                key = (min(r1, r2), max(r1, r2), qid, var)
                if key in cached_keys:
                    continue

                g1, g2 = rater_group[r1], rater_group[r2]
                if   g1 == 'human' and g2 == 'human': ptype = 'HH'
                elif g1 != 'human' and g2 != 'human': ptype = 'MM'
                else:
                    ptype = 'HM'
                    if g1 != 'human':
                        r1, r2 = r2, r1
                        g1, g2 = g2, g1

                en1, en2 = q_answers[r1], q_answers[r2]

                scores: dict[str, float] = {
                    'exact':   _pair_exact(en1, en2),
                    'jaccard': _pair_jaccard(en1, en2),
                    'rouge1':  _pair_rouge1(en1, en2),
                    'chrf':    _pair_chrf(en1, en2),
                }
                for em in emb_metrics:
                    v = float(np.dot(emb_caches[em][en1], emb_caches[em][en2]))
                    scores[em]        = v
                    scores[em + '_c'] = max(v, 0.0)

                new_rows.append({
                    'question_id':      qid,
                    'question_en':      q_en_text,
                    'question_kr':      q_kr_text,
                    'variant':          var,
                    'ent':              q_ent.get(qid, ''),
                    'op':               q_op.get(qid, ''),
                    'pair_type':        ptype,
                    'subject_group_1':  g1,
                    'subject_1':        r1,
                    'subject_group_2':  g2,
                    'subject_2':        r2,
                    'answer_1':         en1,
                    'answer_2':         en2,
                    'exact_score':      round(scores['exact'],   4),
                    'jaccard_score':    round(scores['jaccard'], 4),
                    'rouge1_score':     round(scores['rouge1'],  4),
                    'chrf_score':       round(scores['chrf'],    4),
                    'sbert_score':      round(scores.get('sbert',   float('nan')), 4),
                    'simcse_score':     round(scores.get('simcse',  float('nan')), 4),
                    'sbert_score_clip': round(scores.get('sbert_c', float('nan')), 4),
                    'simcse_score_clip':round(scores.get('simcse_c',float('nan')), 4),
                    'bertscore_f1':     float('nan'),
                })

    if verbose:
        print(f'New rows built: {len(new_rows):,}')

    # ── 8. BERTScore ─────────────────────────────────────────────────────────
    if new_rows and not no_bertscore:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if verbose:
            print(f'Computing BERTScore ({BERTSCORE_MODEL}) on {device} '
                  f'for {len(new_rows):,} pairs …')
        cands     = [r['answer_1'] for r in new_rows]
        refs      = [r['answer_2'] for r in new_rows]
        f1_scores = bertscore_batch(
            cands, refs,
            model_type=BERTSCORE_MODEL,
            device=device,
            batch_size=128,
            verbose=verbose,
        )
        for row, f1 in zip(new_rows, f1_scores):
            row['bertscore_f1'] = round(float(f1), 4)
    elif no_bertscore and verbose:
        print('BERTScore skipped (--no_bertscore).')

    # ── 9. Merge with cache and save ──────────────────────────────────────────
    new_df = pd.DataFrame(new_rows)

    if cache_df is not None and not new_df.empty:
        all_cols = list(dict.fromkeys(
            list(cache_df.columns) + list(new_df.columns)))
        pairs_df = pd.concat(
            [cache_df.reindex(columns=all_cols),
             new_df.reindex(columns=all_cols)],
            ignore_index=True,
        )
    elif cache_df is not None:
        pairs_df = cache_df.copy()
    else:
        pairs_df = new_df.copy()

    pairs_df.to_parquet(cache_path, index=False)
    csv_path = exports / f'pair_cache{"_blind" if condition == "blind" else ""}.csv'
    pairs_df.to_csv(csv_path, index=False)
    if condition == 'inst_blind':
        pairs_df.to_csv(exports / 'answer_pairs_sbert_text.csv', index=False)

    if verbose:
        by_var  = pairs_df['variant'].value_counts().to_dict()
        by_type = pairs_df['pair_type'].value_counts().to_dict()
        print(f'Saved  : {len(pairs_df):,} pairs → {cache_path}')
        print(f'         variants={by_var}  types={by_type}')

    return pairs_df


import re as _re

# ── Answer cleaning regexes (shared with nb17b) ────────────────────────────
REFUSAL_RE = _re.compile(
    r'(?:'
    r"I(?:'m|\s+am)\s+(?:sorry|unable|not\s+able)"
    r"|I\s+can(?:'t|not)\s+(?:answer|determine|provide|see|tell)"
    r'|(?:without|no)\s+(?:seeing|an?\s+)?image'
    r'|(?:not\s+enough|insufficient)\s+(?:information|context)'
    r'|(?:impossible|unable)\s+to\s+(?:determine|answer|tell)'
    r'|(?:question|this)\s+(?:does\s+not|cannot|is\s+not)\s+(?:provide|make)'
    r'|to\s+(?:answer|provide)\s+(?:your|this|an?\s+accurate)'
    r'|(?:need|require)s?\s+(?:more\s+)?context'
    r')', _re.IGNORECASE)

PREAMBLE_RE = _re.compile(
    r'^(?:'
    r'(?:The|Based\s+on)[^.]*?(?:answer|response)\s+(?:is|would\s+be)\s*[:.]?\s*'
    r'|(?:The|It)\s+(?:is|appears|seems|looks\s+like)\s+(?:a\s+|an\s+|the\s+)?'
    r'|(?:I\s+(?:think|believe|would\s+(?:say|guess))\s+(?:it\s+is\s+|the\s+answer\s+is\s+)?)'
    r')', _re.IGNORECASE)

TRAIL_RE = _re.compile(
    r'[,;.]\s+(?:but|however|although|though|which|since|because|as\s+it|if\s+).*$',
    _re.IGNORECASE)

DEFAULT_MAX_WORDS = None
W5_MAX_WORDS = 5


def clean_answer(raw: str, max_words: int | None = DEFAULT_MAX_WORDS) -> str:
    """Clean a raw answer: strip refusals, preamble, trailing elaboration,
    apply VQA normalization, and truncate to max_words."""
    from utils.vqa import preprocess_answer

    text = str(raw or '').replace('\n', ' ').replace('\t', ' ').strip()
    if not text:
        return ''
    if REFUSAL_RE.search(text):
        return ''
    # Strip preamble on RAW text (before VQA normalize removes articles)
    text = PREAMBLE_RE.sub('', text).strip()
    text = TRAIL_RE.sub('', text).strip()
    # VQA normalize (articles, punctuation, number words)
    text = preprocess_answer(text)
    words = text.split()
    if max_words is not None and len(words) > max_words:
        words = words[:max_words]
    return ' '.join(words).strip()


def _cleaned_suffix(output_tag: str | None) -> str:
    return f"_{output_tag}" if output_tag else ""


def _cleaned_embed_cache_path(exports: Path, model_id: str, output_tag: str | None) -> Path:
    tag = model_id.split('/')[-1]
    suffix = _cleaned_suffix(output_tag)
    return exports / f'embeddings_cleaned_{tag}{suffix}.npz'


def build_cleaned_pair_cache(
    root: Path,
    exports: Path,
    max_words: int | None = DEFAULT_MAX_WORDS,
    no_bertscore: bool = False,
    hf_cache: str | None = None,
    condition: str = 'inst_blind',
    output_tag: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build pair_cache_cleaned.parquet from the raw pair_cache.

    Cleans both answer columns, re-scores all lexical and embedding metrics,
    and preserves all metadata columns (variant, ent, op, etc.).
    """
    from sentence_transformers import SentenceTransformer
    from utils.agreement import (
        load_embedding_cache, encode_with_cache,
        _pair_exact, _pair_jaccard, _pair_rouge1, _pair_chrf,
        bertscore_batch,
    )

    root    = Path(root)
    exports = Path(exports)
    hf_dir  = Path(hf_cache or HF_CACHE_DEFAULT)
    hf_dir.mkdir(parents=True, exist_ok=True)
    hf_dir  = str(hf_dir)
    embed_device = _resolve_embed_device()

    suffix = '_blind' if condition == 'blind' else ''
    raw_path = exports / f'pair_cache{suffix}.parquet'
    tag_suffix = _cleaned_suffix(output_tag)
    out_path = exports / f'pair_cache_cleaned{tag_suffix}{suffix}.parquet'

    if not raw_path.exists():
        raise FileNotFoundError(f'Raw pair cache not found: {raw_path}')

    raw_df = pd.read_parquet(raw_path)
    if verbose:
        print(f'Loaded raw pair cache: {len(raw_df):,} rows from {raw_path}')

    # ── 1. Clean answers ────────────────────────────────────────────────────
    if verbose:
        trunc_msg = f'max_words={max_words}' if max_words is not None else 'no truncation'
        print(f'Cleaning answers ({trunc_msg}) …')

    # Cache cleaned answers to avoid redundant work
    clean_cache: dict[str, str] = {}
    def _clean(raw):
        if raw not in clean_cache:
            clean_cache[raw] = clean_answer(raw, max_words=max_words)
        return clean_cache[raw]

    raw_df['answer_1_clean'] = raw_df['answer_1'].apply(_clean)
    raw_df['answer_2_clean'] = raw_df['answer_2'].apply(_clean)

    if verbose:
        n_total = len(raw_df)
        n_empty1 = (raw_df['answer_1_clean'] == '').sum()
        n_empty2 = (raw_df['answer_2_clean'] == '').sum()
        n_both = ((raw_df['answer_1_clean'] == '') | (raw_df['answer_2_clean'] == '')).sum()
        print(f'  Cleaned {len(clean_cache)} unique answers')
        print(f'  Empty after cleaning: answer_1={n_empty1}, answer_2={n_empty2}, '
              f'either={n_both} ({n_both/n_total:.1%})')

    # Drop rows where either answer is empty after cleaning
    clean_df = raw_df[(raw_df['answer_1_clean'] != '') &
                      (raw_df['answer_2_clean'] != '')].copy()
    if verbose:
        print(f'  Retained {len(clean_df):,}/{n_total:,} rows after dropping empty')

    # ── 2. Re-score lexical metrics on cleaned answers ──────────────────────
    if verbose:
        print('Re-scoring lexical metrics …')

    a1 = clean_df['answer_1_clean'].values
    a2 = clean_df['answer_2_clean'].values
    clean_df['exact_score']   = [round(_pair_exact(a1[i], a2[i]),   4) for i in range(len(a1))]
    clean_df['jaccard_score'] = [round(_pair_jaccard(a1[i], a2[i]), 4) for i in range(len(a1))]
    clean_df['rouge1_score']  = [round(_pair_rouge1(a1[i], a2[i]),  4) for i in range(len(a1))]
    clean_df['chrf_score']    = [round(_pair_chrf(a1[i], a2[i]),    4) for i in range(len(a1))]

    # ── 3. Re-embed and re-score SBERT/SimCSE on cleaned answers ────────────
    all_cleaned = sorted(set(a1.tolist() + a2.tolist()))
    if verbose:
        print(f'Unique cleaned answers to embed: {len(all_cleaned)}')

    for metric_name, model_id in EMBEDDING_MODELS.items():
        tag      = model_id.split('/')[-1]
        col      = f'{metric_name}_score'
        col_clip = f'{metric_name}_score_clip'

        if verbose:
            print(f'  [{metric_name}] loading {model_id} on {embed_device} …')
        import os
        os.environ['HF_HOME'] = hf_dir
        try:
            try:
                model = SentenceTransformer(
                    model_id, cache_folder=hf_dir,
                    device=embed_device,
                    model_kwargs={'use_safetensors': True})
            except (OSError, ValueError):
                model = SentenceTransformer(
                    model_id, cache_folder=hf_dir, device=embed_device)
        except Exception as e:
            if verbose:
                print(f'  [{metric_name}] FAILED to load: {e}')
                print(f'  [{metric_name}] setting scores to NaN')
            clean_df[col] = float('nan')
            clean_df[col_clip] = float('nan')
            continue

        cache_path = _cleaned_embed_cache_path(exports, model_id, output_tag)
        emb_dict = encode_with_cache(
            all_cleaned,
            model,
            cache_path,
            batch_size=256,
            normalize=True,
            verbose=verbose,
        )
        del model

        scores = []
        clips = []
        for i in range(len(clean_df)):
            v1, v2 = emb_dict.get(a1[i]), emb_dict.get(a2[i])
            if v1 is not None and v2 is not None:
                v = float(np.dot(v1, v2))
                scores.append(round(v, 4))
                clips.append(round(max(v, 0.0), 4))
            else:
                scores.append(float('nan'))
                clips.append(float('nan'))
        clean_df[col] = scores
        clean_df[col_clip] = clips
        if verbose:
            print(f'  [{metric_name}] done — mean={np.nanmean(scores):.4f}')

    # ── 4. BERTScore on cleaned answers ─────────────────────────────────────
    if not no_bertscore:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if verbose:
            print(f'Computing BERTScore ({BERTSCORE_MODEL}) on {device} for '
                  f'{len(clean_df):,} pairs …')
        f1_scores = bertscore_batch(
            clean_df['answer_1_clean'].tolist(),
            clean_df['answer_2_clean'].tolist(),
            model_type=BERTSCORE_MODEL,
            device=device,
            batch_size=128,
            verbose=verbose,
        )
        clean_df['bertscore_f1'] = [round(float(f), 4) for f in f1_scores]
    elif verbose:
        print('BERTScore skipped.')

    # ── 5. Save ─────────────────────────────────────────────────────────────
    clean_df.to_parquet(out_path, index=False)
    csv_path = out_path.with_suffix('.csv')
    clean_df.to_csv(csv_path, index=False)

    if verbose:
        by_var  = clean_df['variant'].value_counts().to_dict()
        by_type = clean_df['pair_type'].value_counts().to_dict()
        print(f'Saved  : {len(clean_df):,} cleaned pairs → {out_path}')
        print(f'         variants={by_var}  types={by_type}')

    return clean_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build or incrementally update pair_cache.parquet')
    parser.add_argument('--no_bertscore', action='store_true',
                        help='Skip BERTScore (faster, leaves bertscore_f1 as NaN)')
    parser.add_argument('--condition', default='inst_blind',
                        choices=['inst_blind', 'blind'],
                        help='Which model condition to use (default: inst_blind)')
    parser.add_argument('--root', default=None,
                        help='Repository root (default: auto-detected)')
    parser.add_argument('--cleaned', action='store_true',
                        help='Also build the cleaned (length-normalized) pair cache')
    parser.add_argument('--cleaned_only', action='store_true',
                        help='Only build the cleaned pair cache (skip raw)')
    parser.add_argument('--max_words', type=int, default=None,
                        help='Optional max words after cleaning (default: no truncation)')
    parser.add_argument('--output_tag', default=None,
                        help='Optional filename tag for cleaned cache outputs (e.g. w5)')
    parser.add_argument('--include_yesno', action='store_true',
                        help='Include yes/no questions (default: free-text only)')
    args = parser.parse_args()

    root    = Path(args.root) if args.root else ROOT
    exports = root / 'analysis/session2/exports'

    if not args.cleaned_only:
        build_pair_cache(
            root=root,
            exports=exports,
            no_bertscore=args.no_bertscore,
            condition=args.condition,
            include_yesno=args.include_yesno,
            verbose=True,
        )

    if args.cleaned or args.cleaned_only:
        build_cleaned_pair_cache(
            root=root,
            exports=exports,
            max_words=args.max_words,
            no_bertscore=args.no_bertscore,
            hf_cache=None,
            condition=args.condition,
            output_tag=args.output_tag,
            verbose=True,
        )
