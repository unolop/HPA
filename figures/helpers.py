"""
Shared helpers for export scripts.

Centralises canonical reads from analysis/session2/exports and pair_cache,
plus common figure-output utilities.
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
    condition: str = 'inst_blind',
    include_yesno: bool = False,
    subset_qids: Optional[Iterable[int]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    from build_pair_cache import build_pair_cache
    from config import extend_pair_cache_with_yesno

    exports = get_exports_dir(root)
    pair_df = build_pair_cache(root, exports, condition=condition, verbose=verbose)
    if include_yesno:
        if verbose:
            print("Extending pair_cache with yes/no question pairs…")
        pair_df = extend_pair_cache_with_yesno(pair_df, exports)
    if subset_qids is not None:
        qids = set(int(q) for q in subset_qids)
        pair_df = pair_df[pair_df["question_id"].isin(qids)].copy()
    return pair_df


def load_cleaned_pair_cache(
    root: Path,
    *,
    condition: str = 'inst_blind',
    max_words: int | None = None,
    output_tag: Optional[str] = None,
    include_yesno: bool = False,
    subset_qids: Optional[Iterable[int]] = None,
    no_bertscore: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the cleaned (length-normalized) pair cache, building if needed."""
    from build_pair_cache import build_cleaned_pair_cache
    from config import extend_pair_cache_with_yesno

    exports = get_exports_dir(root)
    suffix = '_blind' if condition == 'blind' else ''
    tag_suffix = f'_{output_tag}' if output_tag else ''
    out_path = exports / f'pair_cache_cleaned{tag_suffix}{suffix}.parquet'

    if out_path.exists():
        pair_df = pd.read_parquet(out_path)
        # Validate it has the expected columns (not a stale version)
        if 'variant' in pair_df.columns and 'sbert_score' in pair_df.columns:
            if verbose:
                print(f'Loaded cleaned pair cache: {len(pair_df):,} rows from {out_path}')
        else:
            if verbose:
                print(f'Stale cleaned pair cache detected, rebuilding …')
            pair_df = build_cleaned_pair_cache(
                root, exports, max_words=max_words,
                no_bertscore=no_bertscore, condition=condition,
                output_tag=output_tag, verbose=verbose)
    else:
        pair_df = build_cleaned_pair_cache(
            root, exports, max_words=max_words,
            no_bertscore=no_bertscore, condition=condition,
            output_tag=output_tag, verbose=verbose)

    if include_yesno:
        if verbose:
            print("Extending cleaned pair_cache with yes/no question pairs…")
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


def filter_abstained_pairs(pc: pd.DataFrame, *,
                           answer_col_1: str = 'answer_1_clean',
                           answer_col_2: str = 'answer_2_clean') -> pd.DataFrame:
    """Remove pair_cache rows where either answer is a hard or soft abstention.

    Applied uniformly to HH and HM pairs using the same classify() standard.
    Human abstention is ~0.25% so HH pairs are barely affected; this primarily
    removes model-abstaining HM pairs.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from analysis.utils.abstention import classify, is_abstained

    mask1 = pc[answer_col_1].fillna('').astype(str).apply(
        lambda x: is_abstained(classify(x, None)))
    mask2 = pc[answer_col_2].fillna('').astype(str).apply(
        lambda x: is_abstained(classify(x, None)))
    keep = ~(mask1 | mask2)
    n_removed = (~keep).sum()
    print(f'  [filter_abstained] removed {n_removed:,}/{len(pc):,} pairs '
          f'({n_removed/len(pc):.1%})')
    return pc[keep].copy()


PLOT_FILE_PATTERNS = ("*.png", "*.pdf", "*.svg", "*.jpg", "*.jpeg")


def clear_output_plots(
    out_dir: Path | str,
    *,
    overwrite: bool,
    patterns: tuple[str, ...] = PLOT_FILE_PATTERNS,
) -> int:
    """Delete plot files from an output directory when overwrite is requested."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return 0

    removed = 0
    for pattern in patterns:
        for path in out_path.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    print(f"Cleared {removed} existing plot file(s) from {out_path}")
    return removed


# ── Figure save utility ───────────────────────────────────────────────────────

def save_fig(fig, out_folder, name: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure and print the saved path.

    Parameters
    ----------
    fig : matplotlib Figure
    out_folder : directory (str or Path) to save into
    name : filename, may include subdirectory (e.g. 'subdir/fig.png')
    dpi : resolution (default 150)

    Returns
    -------
    Path of the saved file
    """
    path = Path(out_folder) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f'Saved: {path}')
    return path
