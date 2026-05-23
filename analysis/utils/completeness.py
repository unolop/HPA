"""
Completeness validation for analysis data loading.

Usage
-----
from analysis.utils.completeness import check_completeness

# After reading a CSV:
df, report = check_completeness(df, group_col='model', question_col='question_id')

# With an explicit expected count:
df, report = check_completeness(df, group_col='model', expected_n=113)
"""
from __future__ import annotations

import pandas as pd


def check_completeness(
    df: pd.DataFrame,
    group_col: str,
    question_col: str = 'question_id',
    expected_n: int | None = None,
    min_frac: float = 1.0,
    answer_col: str | None = None,
    empty_thresh: float = 0.10,
    label: str = '',
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Validate that every group has the expected number of unique questions.

    Parameters
    ----------
    df           : DataFrame to validate (one row per group × question)
    group_col    : column identifying the rater/model/participant
    question_col : column identifying the question
    expected_n   : expected unique questions per group; if None uses the max
                   across groups
    min_frac     : minimum fraction of expected_n to be considered complete
                   (default 1.0 = require exact match)
    answer_col   : if provided, also check for empty/NaN answers per group
    empty_thresh : flag groups where fraction of empty answers > this threshold
    label        : prefix string for printed messages
    verbose      : print the report

    Returns
    -------
    filtered_df  : df with flagged (incomplete) groups removed
    report       : dict with keys:
                     expected_n, groups_ok, groups_flagged,
                     flagged (list of dicts with group name and counts)
    """
    prefix = f'[{label}] ' if label else ''

    counts = (
        df.groupby(group_col)[question_col]
        .nunique()
        .rename('n_questions')
        .reset_index()
    )

    if expected_n is None:
        expected_n = int(counts['n_questions'].max())

    min_n = int(expected_n * min_frac)

    flagged_groups = []
    ok_groups = []

    for _, row in counts.iterrows():
        grp  = row[group_col]
        n    = int(row['n_questions'])
        info = {group_col: grp, 'n_questions': n, 'expected': expected_n}

        if n < min_n:
            missing = expected_n - n
            info['missing'] = missing
            flagged_groups.append(info)
        else:
            ok_groups.append(grp)

    # Optional: flag high empty-answer rates
    empty_flagged = []
    if answer_col is not None and answer_col in df.columns:
        for grp in ok_groups + [g[group_col] for g in flagged_groups]:
            sub = df[df[group_col] == grp][answer_col]
            empty_rate = (
                sub.isna() | (sub.astype(str).str.strip() == '')
            ).mean()
            if empty_rate > empty_thresh:
                empty_flagged.append({
                    group_col: grp,
                    'empty_rate': round(float(empty_rate), 3),
                })

    if verbose:
        tag = f'{prefix}Completeness check — {group_col}, expected {expected_n} questions'
        print(tag)
        print('─' * len(tag))
        if not flagged_groups and not empty_flagged:
            print(f'  ✓ All {len(ok_groups)} {group_col}s complete.')
        else:
            if flagged_groups:
                print(f'  ✗ {len(flagged_groups)} incomplete {group_col}(s) — excluded from results:')
                for f in sorted(flagged_groups, key=lambda x: x['n_questions']):
                    print(f'      {f[group_col]}: {f["n_questions"]}/{expected_n} questions '
                          f'(missing {f["expected"] - f["n_questions"]})')
            if empty_flagged:
                print(f'  ⚠  {len(empty_flagged)} {group_col}(s) with high empty-answer rate:')
                for f in sorted(empty_flagged, key=lambda x: -x['empty_rate']):
                    print(f'      {f[group_col]}: {f["empty_rate"]*100:.1f}% empty')
            if ok_groups:
                print(f'  ✓ {len(ok_groups)} {group_col}(s) complete and included.')
        print()

    all_flagged_names = (
        {f[group_col] for f in flagged_groups}
        | {f[group_col] for f in empty_flagged}
    )
    filtered_df = df[~df[group_col].isin(all_flagged_names)].copy()

    report = {
        'expected_n':     expected_n,
        'groups_ok':      ok_groups,
        'groups_flagged': [f[group_col] for f in flagged_groups],
        'empty_flagged':  [f[group_col] for f in empty_flagged],
        'flagged':        flagged_groups,
    }
    return filtered_df, report
