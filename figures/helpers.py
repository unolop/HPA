"""
Shared helpers for export scripts.

Centralises canonical reads from analysis/session2/exports and pair_cache.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Ensure repo root is on sys.path so utils.completeness is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def get_exports_dir(root: Path) -> Path:
    return Path(root) / "analysis/session2/exports"


def read_export(
    root: Path,
    filename: str,
    *,
    subset_qids: Optional[Iterable[int]] = None,
    variant: Optional[str] = None,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read a canonical session-2 export with consistent optional filtering."""
    df = pd.read_csv(get_exports_dir(root) / filename, usecols=columns)
    if subset_qids is not None and "question_id" in df.columns:
        qids = set(int(q) for q in subset_qids)
        df = df[df["question_id"].isin(qids)].copy()
    if variant is not None and "variant" in df.columns:
        df = df[df["variant"] == variant].copy()
    return df


def read_response_exports(
    root: Path,
    *,
    subset_qids: Optional[Iterable[int]] = None,
    variant: Optional[str] = None,
    check_completeness: bool = True,
) -> dict[str, pd.DataFrame]:
    from utils.completeness import check_completeness as _check

    exports: dict[str, pd.DataFrame] = {}

    human_df = read_export(root, "responses_human.csv", subset_qids=subset_qids, variant=variant)
    if check_completeness:
        human_df, _ = _check(human_df, group_col='participant', question_col='question_id',
                             answer_col='response', label='Human responses')
    exports["human"] = human_df

    for key, fname in [
        ("model_blind",      "responses_model_blind.csv"),
        ("model_inst_blind", "responses_model_inst_blind.csv"),
        ("model_control",    "responses_model_control.csv"),
    ]:
        df = read_export(root, fname, subset_qids=subset_qids, variant=variant)
        if check_completeness and not df.empty:
            df, _ = _check(df, group_col='model', question_col='question_id',
                           answer_col='response', label=key)
        exports[key] = df

    return exports


def load_pair_cache(
    root: Path,
    *,
    include_yesno: bool = False,
    subset_qids: Optional[Iterable[int]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    from build_pair_cache import build_pair_cache
    from config import extend_pair_cache_with_yesno

    exports = get_exports_dir(root)
    pair_df = build_pair_cache(root, exports, verbose=verbose)
    if include_yesno:
        if verbose:
            print("Extending pair_cache with yes/no question pairs…")
        pair_df = extend_pair_cache_with_yesno(pair_df, exports)
    if subset_qids is not None:
        qids = set(int(q) for q in subset_qids)
        pair_df = pair_df[pair_df["question_id"].isin(qids)].copy()
    return pair_df


def load_human_subset(
    root: Path,
    *,
    min_answers: int,
    translate: bool = False,
    verbose: bool = True,
):
    from utils.load_session import load_human_data

    return load_human_data(root, min_answers=min_answers, translate=translate, verbose=verbose)
