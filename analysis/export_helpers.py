"""
Shared helpers for export scripts.

Centralises canonical reads from analysis/session2/exports and pair_cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


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
) -> dict[str, pd.DataFrame]:
    return {
        "human": read_export(root, "responses_human.csv", subset_qids=subset_qids, variant=variant),
        "model_blind": read_export(root, "responses_model_blind.csv", subset_qids=subset_qids, variant=variant),
        "model_inst_blind": read_export(root, "responses_model_inst_blind.csv", subset_qids=subset_qids, variant=variant),
        "model_control": read_export(root, "responses_model_control.csv", subset_qids=subset_qids, variant=variant),
    }


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
