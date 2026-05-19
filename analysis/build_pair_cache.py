"""
Build or incrementally update pair_cache.parquet.

Computes pairwise agreement between all raters (human participants + models)
on free-text VQA questions (inst_blind condition) using:
  exact, jaccard, rouge1, chrf, sbert, simcse, bertscore

Incremental: any (subject_1, subject_2, question_id, variant) tuple already
in pair_cache.parquet is skipped, so re-runs are fast.

Detects:
  - New human participants in responses_human.csv not yet in HH/HM pairs
  - New models in responses_model_inst_blind.csv not yet in HM/MM pairs

Outputs:
  exports/pair_cache.parquet          — full incremental pair store
  exports/answer_pairs_sbert_text.csv — CSV mirror for downstream scripts

Run from repo root:
  conda run -n zero python analysis/build_pair_cache.py
  conda run -n zero python analysis/build_pair_cache.py --no_bertscore
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

EMBEDDING_MODELS = {
    'sbert':  'sentence-transformers/all-mpnet-base-v2',
    'simcse': 'princeton-nlp/sup-simcse-roberta-large',
}
BERTSCORE_MODEL  = 'roberta-large'
HF_CACHE_DEFAULT = '/home/david/Desktop/yuna/.cache/hf'
VARIANT_ORDER    = ['C', 'B', 'A']
CT_TO_VARIANT    = {'question': 'C', 'weaker_object': 'B', 'pronominalized': 'A'}


def build_pair_cache(
    root: Path,
    exports: Path,
    no_bertscore: bool = False,
    hf_cache: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build or incrementally update pair_cache.parquet.

    Parameters
    ----------
    root         : repository root Path
    exports      : exports directory (contains pair_cache.parquet, responses_*.csv)
    no_bertscore : skip BERTScore computation (faster, leaves bertscore_f1 as NaN)
    hf_cache     : HuggingFace cache directory override
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

    root    = Path(root)
    exports = Path(exports)
    hf_dir  = hf_cache or HF_CACHE_DEFAULT
    cache_path = exports / 'pair_cache.parquet'

    # ── 1. Load human + model responses ──────────────────────────────────────
    h_csv = pd.read_csv(exports / 'responses_human.csv')
    m_csv = pd.read_csv(exports / 'responses_model_inst_blind.csv')

    if verbose:
        print(f'Humans : {h_csv["participant"].nunique()} participants, '
              f'{h_csv["question_id"].nunique()} questions')
        print(f'Models : {m_csv["model"].nunique()} models')

    # ── 2. Question metadata ──────────────────────────────────────────────────
    q_meta_list = json.load(open(root / 'experiment/s2_v4/s4_question.json'))
    answer_type_map = {q['question_id']: q['answer_type'] for q in q_meta_list}

    common_qids = set(h_csv['question_id'].astype(int).unique())
    text_qids   = {qid for qid in common_qids if answer_type_map.get(qid) == 'text'}

    if verbose:
        yesno = common_qids - text_qids
        print(f'Questions: {len(common_qids)} total — '
              f'{len(text_qids)} free-text, {len(yesno)} yes/no')

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

    if n_new == 0:
        if verbose:
            print('No new pairs — pair_cache is up to date.')
        return cache_df if cache_df is not None else pd.DataFrame()

    if verbose:
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
                print(f'  [{metric_name}] {len(new_to_enc)} new answers — loading {model_id} …')
            import os
            os.environ['HF_HOME'] = hf_dir
            model = SentenceTransformer(model_id, cache_folder=hf_dir)
            emb_caches[metric_name] = encode_with_cache(
                all_answers, model, emb_path, normalize=True, verbose=verbose)
            del model
        else:
            emb_caches[metric_name] = existing
            if verbose:
                print(f'  [{metric_name}] all {len(existing)} vectors cached')

    primary_emb = next(iter(emb_caches.values())) if emb_caches else {}

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
    pairs_df.to_csv(exports / 'answer_pairs_sbert_text.csv', index=False)

    if verbose:
        by_var  = pairs_df['variant'].value_counts().to_dict()
        by_type = pairs_df['pair_type'].value_counts().to_dict()
        print(f'Saved  : {len(pairs_df):,} pairs → {cache_path}')
        print(f'         variants={by_var}  types={by_type}')

    return pairs_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build or incrementally update pair_cache.parquet')
    parser.add_argument('--no_bertscore', action='store_true',
                        help='Skip BERTScore (faster, leaves bertscore_f1 as NaN)')
    parser.add_argument('--root', default=None,
                        help='Repository root (default: auto-detected)')
    args = parser.parse_args()

    root    = Path(args.root) if args.root else ROOT
    exports = root / 'analysis/session2/exports'

    build_pair_cache(
        root=root,
        exports=exports,
        no_bertscore=args.no_bertscore,
        verbose=True,
    )
