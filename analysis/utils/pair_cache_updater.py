"""
Auto-update pair_cache_raw.parquet with HM pairs for newly available models.

Called at the start of export scripts that consume pair_cache so plots
are always generated from the freshest data without running nb10 manually.

Only computes HM (Human × Model) pairs for free-text questions.
Metrics computed: exact, jaccard, rouge1, chrf, sbert.
SimCSE and BERTScore are left as NaN (run nb10 to fill them in).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CT_TO_VARIANT = {'question': 'C', 'weaker_object': 'B', 'pronominalized': 'A'}
SBERT_CACHE_NAME = 'embeddings_all-mpnet-base-v2.npz'
SBERT_MODEL_ID   = 'sentence-transformers/all-mpnet-base-v2'
HF_CACHE_DEFAULT = '/home/david/Desktop/yuna/.cache/hf'


def _load_model_answers(jsonl_path: Path, text_qids: set) -> dict:
    """Read a logit JSONL and return {(question_id, variant): answer_str}."""
    from utils.load_session import clean_answer

    answers: dict = {}
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = int(ex['question_id'])
            if qid not in text_qids:
                continue
            ga = ex.get('generated_answers', {})
            for ct, var in CT_TO_VARIANT.items():
                answers[(qid, var)] = clean_answer(str(ga.get(ct, '')))
    return answers


def ensure_pair_cache(
    root: Path,
    exports: Path,
    hf_cache: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return an up-to-date pair_cache DataFrame.

    For every model in default_all_models() whose inst_blind JSONL file is
    complete (1000 lines) but is not yet represented in pair_cache, this
    function computes HM pairs and appends them before returning.

    The updated parquet is written back to exports/pair_cache_raw.parquet so
    subsequent calls are instant (all models already cached).

    Parameters
    ----------
    root     : repository root (Path)
    exports  : exports directory containing pair_cache_raw.parquet, responses_human.csv
    hf_cache : optional HuggingFace cache directory for the SBERT model
    verbose  : print progress messages
    """
    from utils.model_registry import default_all_models
    from utils.model_registry import MODEL_TYPE as _MODEL_TYPE
    from utils.agreement import (
        load_embedding_cache, encode_with_cache,
        _pair_exact, _pair_jaccard, _pair_rouge1, _pair_chrf,
    )

    root    = Path(root)
    exports = Path(exports)
    cache_path = exports / 'pair_cache_raw.parquet'

    pair_cache = pd.read_parquet(cache_path)
    cached_models = set(
        pair_cache[pair_cache['pair_type'] == 'HM']['subject_2'].unique()
    )

    # ── Discover models with complete data not yet in cache ───────────────────
    all_models = default_all_models(root)
    missing: list[tuple[str, Path]] = []
    for label, (rdir, mdir) in all_models.items():
        if label in cached_models:
            continue
        jsonl = rdir / mdir / 'vqa_1k_control_inst_blind.jsonl'
        if not jsonl.exists():
            continue
        n_lines = sum(1 for _ in open(jsonl))
        if n_lines < 1000:
            if verbose:
                print(f'  [pair_cache] skip {label}: {n_lines}/1000 lines')
            continue
        missing.append((label, jsonl))

    if not missing:
        if verbose:
            print('  [pair_cache] up to date — no new models')
        return pair_cache

    if verbose:
        print(f'  [pair_cache] adding {len(missing)} model(s): '
              f'{[m for m, _ in missing]}')

    # ── Load supporting data ──────────────────────────────────────────────────
    h_csv = pd.read_csv(exports / 'responses_human.csv')
    # common question IDs answered by humans
    common_qids = set(h_csv['question_id'].astype(int).unique())

    # question metadata for ent / op / question_en / question_kr
    q_meta_raw = json.load(open(root / 'experiment/s2_v4/s4_question.json'))
    q_meta = {int(q['question_id']): q for q in q_meta_raw}

    # free-text questions only (same scope as nb10)
    text_qids = {
        qid for qid, q in q_meta.items()
        if q.get('answer_type') == 'text' and qid in common_qids
    }

    # ── Collect all new model answers for encoding ────────────────────────────
    model_ans_maps: dict[str, dict] = {}
    all_new_answers: set[str] = set()
    for label, jsonl in missing:
        ans_map = _load_model_answers(jsonl, text_qids)
        model_ans_maps[label] = ans_map
        all_new_answers.update(ans_map.values())

    # also collect human answers that might not be encoded yet
    human_answers = set(h_csv['response'].dropna().astype(str).unique())
    all_needed = all_new_answers | human_answers

    # ── SBERT: use cache, encode only what's missing ──────────────────────────
    sbert_cache_path = exports / SBERT_CACHE_NAME
    ans2emb = load_embedding_cache(sbert_cache_path)
    new_to_encode = [a for a in all_needed if a not in ans2emb]

    if new_to_encode:
        import os
        os.environ['HF_HOME'] = hf_cache or HF_CACHE_DEFAULT
        if verbose:
            print(f'  [pair_cache] encoding {len(new_to_encode)} new answers with SBERT…')
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(SBERT_MODEL_ID,
                                    cache_folder=hf_cache or HF_CACHE_DEFAULT)
        ans2emb = encode_with_cache(
            list(all_needed), sbert, sbert_cache_path,
            normalize=True, verbose=verbose)
        del sbert
    elif verbose:
        print('  [pair_cache] all SBERT embeddings found in cache')

    # ── Build new pair rows ───────────────────────────────────────────────────
    new_rows: list[dict] = []
    for label, _ in missing:
        grp     = _MODEL_TYPE.get(label, 'standalone LLM')
        ans_map = model_ans_maps[label]

        for var in ['C', 'B', 'A']:
            h_var = h_csv[h_csv['variant'] == var].copy()

            for _, h_row in h_var.iterrows():
                qid = int(h_row['question_id'])
                if qid not in text_qids:
                    continue
                human_ans = str(h_row['response']) if pd.notna(h_row['response']) else ''
                model_ans = ans_map.get((qid, var), '')
                qm = q_meta.get(qid, {})

                exact = _pair_exact(human_ans, model_ans)
                jac   = _pair_jaccard(human_ans, model_ans)
                r1    = _pair_rouge1(human_ans, model_ans)
                chrf  = _pair_chrf(human_ans, model_ans)

                emb_h = ans2emb.get(human_ans)
                emb_m = ans2emb.get(model_ans)
                if emb_h is not None and emb_m is not None:
                    sbert_s = float(np.dot(emb_h, emb_m))
                    sbert_c = max(sbert_s, 0.0)
                else:
                    sbert_s = sbert_c = np.nan

                new_rows.append({
                    'question_id':       qid,
                    'question_en':       qm.get('question_en',
                                                 h_row.get('question_en', '')),
                    'question_kr':       qm.get('question_kr',
                                                 h_row.get('question_kr', '')),
                    'variant':           var,
                    'ent':               qm.get('ent', ''),
                    'op':                qm.get('op', ''),
                    'pair_type':         'HM',
                    'subject_group_1':   'human',
                    'subject_1':         str(h_row['participant']),
                    'subject_group_2':   grp,
                    'subject_2':         label,
                    'answer_1':          human_ans,
                    'answer_2':          model_ans,
                    'exact_score':       exact,
                    'jaccard_score':     jac,
                    'rouge1_score':      r1,
                    'chrf_score':        chrf,
                    'sbert_score':       sbert_s,
                    'simcse_score':      np.nan,
                    'sbert_score_clip':  sbert_c,
                    'simcse_score_clip': np.nan,
                    'bertscore_f1':      np.nan,
                })

    if not new_rows:
        return pair_cache

    new_df    = pd.DataFrame(new_rows)
    all_cols  = list(dict.fromkeys(
        list(pair_cache.columns) + list(new_df.columns)))
    updated   = pd.concat(
        [pair_cache.reindex(columns=all_cols),
         new_df.reindex(columns=all_cols)],
        ignore_index=True)

    updated.to_parquet(cache_path, index=False)
    if verbose:
        print(f'  [pair_cache] saved: {len(pair_cache):,} → {len(updated):,} rows '
              f'(+{len(new_rows):,} HM pairs for {len(missing)} model(s))')

    return updated
