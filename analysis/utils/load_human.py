"""Utilities for loading and flattening s2 human study data."""
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_participants(base: Path, min_answers: int = 300) -> list:
    """Load all *.json files from experiment/s2_humans/, deduplicate by code,
    and filter to participants with >= min_answers answers."""
    all_entries = []
    for fpath in sorted((base / 'experiment/s2_humans').glob('*.json')):
        raw = json.load(open(fpath))
        entries = raw if isinstance(raw, list) else [raw]
        all_entries.extend(entries)

    # Deduplicate: keep the entry with the most answers for each code
    seen = {}
    for p in all_entries:
        code = p['code']
        if code not in seen or len(p.get('answers', [])) > len(seen[code].get('answers', [])):
            seen[code] = p

    valid = [p for p in seen.values() if len(p.get('answers', [])) >= min_answers]
    return valid


def flatten_to_df(participants: list, mapper, s2_df: pd.DataFrame,
                  normalize_fn=None) -> pd.DataFrame:
    """Flatten participant answer lists into a DataFrame scored with vqa_accuracy.

    Args:
        participants: list of participant dicts (from load_participants)
        mapper: VQAAnswerMapper instance
        s2_df: s2 metadata DataFrame with question_id + ent/op/etc columns
        normalize_fn: optional callable(str) -> str for answer normalisation
    """
    from utils.vqa import vqa_accuracy

    rows = []
    for p in participants:
        for a in p['answers']:
            raw_ans = str(a.get('answer_text', '')).strip()
            answer_en = normalize_fn(raw_ans) if normalize_fn else raw_ans
            gt = mapper.get_answers(a['question_id'])
            acc = vqa_accuracy(answer_en, gt) if gt else np.nan
            rows.append({
                'participant':    p['code'],
                'tag':            p.get('tag', ''),
                'session':        a.get('session_number'),
                'question_id':    a['question_id'],
                'variant':        a.get('variant', 'C'),
                'answer_kr':      raw_ans,
                'answer_en':      answer_en,
                'gt':             gt,
                'confidence':     a.get('confidence', np.nan),
                'time_ms':        a.get('time_spent_ms', np.nan),
                'time_s':         a.get('time_spent_ms', 0) / 1000,
                'seq':            a.get('sequence_number', np.nan),
                'accuracy':       acc,
            })

    df = pd.DataFrame(rows)
    df = df.merge(s2_df, on='question_id', how='left')
    return df
