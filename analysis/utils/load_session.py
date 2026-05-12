"""
Centralised data-loading helpers for S2 human study notebooks.

Usage
-----
from utils.load_session import load_human_data, load_model_results

participants, common_qids, df, mapper = load_human_data(BASE)
model_q_acc, MODEL_TYPE, MODELS, ALL_MODELS = load_model_results(BASE, q_ids)
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── make sure repo root is on sys.path ───────────────────────────────────────
_REPO = Path(__file__).parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.utils.normalize_korean import normalize_korean_answer
from analysis.utils.model_registry import MODEL_TYPE, default_all_models
from analysis.utils.vqa import preprocess_answer, VQAAnswerMapper, vqa_accuracy

_KR_CHAR = re.compile(r'[가-힣]')


# ── Question text backfill ────────────────────────────────────────────────────

def backfill_question_text(
    base: Path,
    s2_version: str = 's2_v4',
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """Fill null question_en / question_kr in participant JSON files from s4 experiment JSONs.

    Parameters
    ----------
    base       : repo root Path
    s2_version : experiment version folder (default 's2_v4')
    dry_run    : if True, report what would change without writing
    verbose    : print summary

    Returns
    -------
    dict with keys: files_checked, files_updated, answers_filled
    """
    base = Path(base)
    exp_dir = base / 'experiment' / s2_version

    # Build (question_id, variant) → {question_en, question_kr} lookup
    stem = 's' + exp_dir.name.split('_v')[-1]  # s2_v4 → s4
    lookup: dict[tuple, dict] = {}
    for variant, fname in [('C', f'{stem}_question.json'),
                            ('B', f'{stem}_weaker_object.json'),
                            ('A', f'{stem}_pronominalized.json')]:
        fpath = exp_dir / fname
        if not fpath.exists():
            continue
        for entry in json.load(open(fpath)):
            key = (int(entry['question_id']), variant)
            lookup[key] = {
                'question_en': entry.get('question_en', ''),
                'question_kr': entry.get('question_kr', ''),
            }

    if verbose:
        print(f'Loaded question text lookup: {len(lookup)} entries from {exp_dir.name}')

    part_dir = base / 'evaluation/humans/by_participant'
    stats = {'files_checked': 0, 'files_updated': 0, 'answers_filled': 0}

    for fpath in sorted(part_dir.glob('*.json')):
        stats['files_checked'] += 1
        p = json.load(open(fpath))
        filled = 0
        for a in p.get('answers', []):
            qid = int(a['question_id'])
            var = a.get('variant', 'C')
            key = (qid, var)
            if key not in lookup:
                continue
            changed = False
            if not a.get('question_en'):
                a['question_en'] = lookup[key]['question_en']
                changed = True
            if not a.get('question_kr'):
                a['question_kr'] = lookup[key]['question_kr']
                changed = True
            if changed:
                filled += 1
        if filled:
            stats['files_updated'] += 1
            stats['answers_filled'] += filled
            if verbose:
                print(f'  {fpath.name}: backfilled question_en/question_kr for {filled} rows')
            if not dry_run:
                with open(fpath, 'w') as f:
                    json.dump(p, f, ensure_ascii=False, indent=2)

    if verbose:
        action = 'Would backfill' if dry_run else 'Backfilled'
        print(f'\n{action} question_en/question_kr on {stats["answers_filled"]} rows '
              f'across {stats["files_updated"]}/{stats["files_checked"]} participant files')
# Re-exported from constants for backwards compatibility
from analysis.utils.constants import CT_TO_VARIANT, VARIANT_ORDER  # noqa: E402


def load_jsonl(path) -> list:
    """Load a .jsonl file into a list of dicts. Returns [] if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def clean_answer(ans: str) -> str:
    """Backwards-compatible model answer cleaner used by older notebooks."""
    return preprocess_answer(ans, strip_think=True)


def load_participants(base: Path, min_answers: int = 348) -> list:
    """Load participant JSON files and keep entries with at least min_answers."""
    all_entries = []
    for fpath in sorted((base / 'experiment/s2_humans').glob('*.json')):
        raw = json.load(open(fpath))
        entries = raw if isinstance(raw, list) else [raw]
        all_entries.extend(entries)

    seen = {}
    for participant in all_entries:
        code = participant['code']
        if code not in seen or len(participant.get('answers', [])) > len(seen[code].get('answers', [])):
            seen[code] = participant

    return [p for p in seen.values() if len(p.get('answers', [])) >= min_answers]


def flatten_to_df(participants: list, mapper, s2_df: pd.DataFrame, normalize_fn=None) -> pd.DataFrame:
    """Flatten participant answers into a scored DataFrame using shared VQA preprocessing."""
    rows = []
    for participant in participants:
        for answer in participant['answers']:
            raw_ans = str(answer.get('answer_text', '')).strip()
            answer_en = normalize_fn(raw_ans) if normalize_fn else raw_ans
            answer_en = preprocess_answer(answer_en)
            gt = mapper.get_answers(answer['question_id'])
            acc = vqa_accuracy(answer_en, gt) if gt else np.nan
            rows.append({
                'participant': participant['code'],
                'tag': participant.get('tag', ''),
                'session': answer.get('session_number'),
                'question_id': answer['question_id'],
                'variant': answer.get('variant', 'C'),
                'answer_kr': raw_ans,
                'answer_en': answer_en,
                'gt': gt,
                'confidence': answer.get('confidence', np.nan),
                'time_ms': answer.get('time_spent_ms', np.nan),
                'time_s': answer.get('time_spent_ms', 0) / 1000,
                'seq': answer.get('sequence_number', np.nan),
                'accuracy': acc,
            })

    df = pd.DataFrame(rows)
    return df.merge(s2_df, on='question_id', how='left')


# ── Human data ────────────────────────────────────────────────────────────────

def load_human_data(
    base: Path,
    min_answers: int = 348,
    s2_csv: Optional[Path] = None,
    translate: bool = True,
    verbose: bool = True,
) -> tuple:
    """Load, translate, flatten and intersect human participant data.

    Parameters
    ----------
    base        : repo root Path
    min_answers : minimum answers to include a participant
    s2_csv      : path to s4.csv metadata (defaults to experiment/s2_v4/s4.csv)
    translate   : whether to run the GPT auto-translate step
    verbose     : print progress

    Returns
    -------
    participants : list of participant dicts
    common_qids  : set[int] — question_ids answered by every qualifying participant
    df           : flattened DataFrame with accuracy and metadata
    mapper       : VQAAnswerMapper instance
    """
    base = Path(base)
    mapper = VQAAnswerMapper()

    # ── 1. Load participants ──────────────────────────────────────────────────
    all_files = sorted((base / 'evaluation/humans/by_participant').glob('*.json'))
    participants = []
    for fpath in all_files:
        p = json.load(open(fpath))
        if len(p.get('answers', [])) >= min_answers:
            participants.append(p)

    if verbose:
        print(f'Qualifying participants (>= {min_answers} answers): {len(participants)}')
        for p in sorted(participants, key=lambda x: x['code']):
            qids = {a['question_id'] for a in p['answers']}
            print(f"  {p['code']}  answers={len(p['answers'])}  qids={len(qids)}")

    # ── 2. Auto-translate Korean answers ──────────────────────────────────────
    if translate:
        from analysis.utils.translate_kr_answers import translate_unmapped_korean_answers
        translate_unmapped_korean_answers(
            participants,
            base / 'analysis/utils/kr_answer_map.json',
            model='gpt-4o-mini',
        )

    # ── 3. Flatten to DataFrame ───────────────────────────────────────────────
    s2_path = s2_csv or (base / 'experiment/s2_v4/s4.csv')
    s2 = pd.read_csv(s2_path)[['question_id', 'ent', 'op', 'op_grp', 'signal_score', 'acc_original']]
    df = flatten_to_df(
        participants,
        mapper,
        s2,
        normalize_fn=normalize_korean_answer,
    )

    # ── 4. Intersect to common question_ids ───────────────────────────────────
    qid_sets = [{a['question_id'] for a in p['answers']} for p in participants]
    common_qids = set.intersection(*qid_sets) if qid_sets else set()
    df = df[df['question_id'].isin(common_qids)].copy()

    # ── 5. Coverage check ─────────────────────────────────────────────────────
    df['normalized'] = ~df['answer_en'].apply(lambda x: bool(_KR_CHAR.search(str(x))))
    n_unnorm = (~df['normalized']).sum()
    if verbose:
        print(f'\nCommon question_ids: {len(common_qids)}')
        print(f'Rows: {len(df)} | questions: {df.question_id.nunique()} '
              f'| participants: {df.participant.nunique()}')
        print(f'Answers still in Korean (scored 0): {n_unnorm} / {len(df)} '
              f'({100 * n_unnorm / len(df):.1f}%)')

    return participants, common_qids, df, mapper


def check_translation_coverage(df: pd.DataFrame, map_path: Path, verbose: bool = True) -> dict:
    """Print translation coverage and auto-append unmapped Korean strings to the map.

    Returns a dict with keys: total, translated, untranslated, added.
    """
    unmapped = df[~df['normalized']].copy()
    total = len(df)
    stats = {
        'total': total,
        'translated': total - len(unmapped),
        'untranslated': len(unmapped),
        'added': 0,
    }
    if verbose:
        print(f'Total     : {total}')
        print(f'Translated: {stats["translated"]}  ({100 * stats["translated"] / total:.1f}%)')
        print(f'Untranslated: {len(unmapped)}  ({100 * len(unmapped) / total:.1f}%)')

    if len(unmapped):
        kr_map = json.load(open(map_path))
        existing = set(kr_map['map'].keys())
        new_strings = sorted(unmapped['answer_kr'].unique(), key=lambda x: -unmapped['answer_kr'].eq(x).sum())
        added = [s for s in new_strings if s not in existing]
        if added:
            for s in added:
                kr_map['map'][s] = ''
            with open(map_path, 'w') as f:
                json.dump(kr_map, f, ensure_ascii=False, indent=2)
            stats['added'] = len(added)
            if verbose:
                print(f'Appended {len(added)} new entries to {map_path}')
    else:
        if verbose:
            print('✓ All answers translated.')
    return stats


# ── Model data ────────────────────────────────────────────────────────────────

def _default_all_models(base: Path) -> dict:
    """Backward-compatible wrapper for the canonical ALL_MODELS registry."""
    return default_all_models(base)


def load_model_results(
    base: Path,
    q_ids: set,
    condition: str = 'vqa_1k_control_inst_blind',
    variant_order: list = VARIANT_ORDER,
    all_models: Optional[dict] = None,
    verbose: bool = True,
) -> tuple:
    """Load per-question per-variant accuracy for all models.

    Parameters
    ----------
    base          : repo root Path
    q_ids         : set of question_ids to include (typically common_qids)
    condition     : JSONL filename stem to load
    variant_order : list of variant labels e.g. ['C','B','A']
    all_models    : override the default ALL_MODELS registry
    verbose       : print load summary

    Returns
    -------
    model_q_acc : dict[label → DataFrame(index=question_id, cols=variant_order)]
    MODEL_TYPE  : dict[label → group string]
    MODELS      : dict[label → Path to model logit dir]
    ALL_MODELS  : dict[label → (results_dir, mdir)]
    """
    base = Path(base)
    mapper = VQAAnswerMapper()
    if all_models is None:
        all_models = _default_all_models(base)

    models_flat = {label: rdir / mdir for label, (rdir, mdir) in all_models.items()}
    acc_raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for label, (results_dir, mdir) in all_models.items():
        path = results_dir / mdir / f'{condition}.jsonl'
        if not path.exists():
            if verbose:
                print(f'  MISSING: {label} — {path}')
            continue
        for line in open(path):
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                print(f'  MALFORMED JSON skipped in {path}')
                continue
            qid = int(ex['question_id'])
            if qid not in q_ids:
                continue
            gt = mapper.get_answers(qid)
            ga = ex.get('generated_answers', {})
            for ct, var in CT_TO_VARIANT.items():
                ans = clean_answer(ga.get(ct, ''))
                if ans and gt:
                    acc_raw[label][qid][var].append(vqa_accuracy(ans, gt))

    model_q_acc = {}
    for label in all_models:
        if label not in acc_raw:
            continue
        rows = []
        for qid in q_ids:
            row = {'question_id': qid}
            for var in variant_order:
                vals = acc_raw[label][qid][var]
                row[var] = float(np.mean(vals)) if vals else np.nan
            rows.append(row)
        model_q_acc[label] = pd.DataFrame(rows).set_index('question_id')

    if verbose:
        print(f'Loaded {condition} for: {list(model_q_acc.keys())}')
        for label, mdf in model_q_acc.items():
            n = mdf['C'].notna().sum()
            g = MODEL_TYPE.get(label, '?')
            print(f'  [{g:12}] {label}: {n} questions, mean C = {mdf["C"].mean():.3f}')

    return model_q_acc, MODEL_TYPE, models_flat, all_models
