"""
Correlation and inter-rater agreement utilities for S2 analysis.

Used by: 10_model_correlations.ipynb
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import matthews_corrcoef

Label = Union[str, int, float]


def gwet_ac1(
    rater_a: Sequence[Optional[Label]],
    rater_b: Sequence[Optional[Label]],
    *,
    categories: Optional[Sequence[Label]] = None,
    drop_missing: bool = True,
) -> float:
    """Gwet's AC1 (two raters, nominal categories)."""
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length.")

    def is_missing(x) -> bool:
        if x is None:
            return True
        try:
            return isinstance(x, float) and x != x
        except Exception:
            return False

    paired: List[Tuple[Label, Label]] = [
        (a, b) for a, b in zip(rater_a, rater_b)
        if not (is_missing(a) or is_missing(b))
    ]
    if not paired:
        raise ValueError("No valid (non-missing) paired ratings to score.")

    if categories is None:
        cats = sorted({x for ab in paired for x in ab}, key=lambda x: (str(type(x)), str(x)))
    else:
        cats = list(categories)

    q = len(cats)
    if q < 2:
        raise ValueError("AC1 requires at least 2 categories.")

    n = len(paired)
    po = sum(1 for a, b in paired if a == b) / n

    from collections import Counter
    counts = Counter(x for ab in paired for x in ab)
    pi = {k: counts[k] / (2 * n) for k in cats}
    pe = (1 / (q - 1)) * sum(pi[k] * (1 - pi[k]) for k in cats)

    denom = 1 - pe
    if denom == 0:
        return 1.0 if po == 1.0 else 0.0
    return float(np.clip((po - pe) / denom, -1.0, 1.0))


def pairwise_metric(a, b, metric: str = "spearman") -> float:
    """Compute a pairwise correlation/agreement metric between two aligned Series."""
    # Convert to numpy to avoid scipy returning a 2×2 matrix for pandas Series inputs
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if metric == "spearman":
        stat = spearmanr(a, b).statistic
        return float(stat) if np.ndim(stat) == 0 else float(stat[0, 1])
    elif metric == "kendall":
        stat = kendalltau(a, b).statistic
        return float(stat) if np.ndim(stat) == 0 else float(stat[0, 1])
    elif metric == "pearson":
        return float(pearsonr(a, b)[0])
    elif metric == "mcc":
        return float(matthews_corrcoef(a, b))
    elif metric == "ac1":
        return gwet_ac1(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def aggregate_corr(corr_mat: pd.DataFrame, grp1, grp2) -> float:
    """Mean off-diagonal (upper triangle if grp1==grp2, else all) entries."""
    if list(grp1) == list(grp2):
        mask = np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
        flat = corr_mat.where(mask).stack()
    else:
        flat = corr_mat.stack()
    return float(flat.dropna().mean())


def interrater_agreement(
    answers_pivot: pd.DataFrame,
    grp1,
    grp2,
    metric: str = "spearman",
    n_boot: int = 200,
    title: str = "Agreement",
) -> dict:
    """
    Compute pairwise inter-rater agreement with jackknife CI.

    Parameters
    ----------
    answers_pivot : wide DataFrame, columns = rater IDs, rows = questions
    grp1, grp2    : rater subsets to correlate
    metric        : one of 'spearman', 'pearson', 'kendall', 'mcc', 'ac1'
    n_boot        : kept for API compatibility (jackknife is used instead)
    title         : label for the result dict

    Returns
    -------
    dict with keys: title, metric, mean_r, ci_2p5, ci_97p5, n_pairs, corr_mat
    """
    corr_mat = pd.DataFrame(np.nan, index=grp1, columns=grp2)
    for r1 in grp1:
        for r2 in grp2:
            if r1 == r2:
                continue
            valid = answers_pivot[[r1, r2]].dropna()
            if len(valid) < 5:
                continue
            corr_mat.loc[r1, r2] = pairwise_metric(valid[r1], valid[r2], metric)

    mean_r = aggregate_corr(corr_mat, grp1, grp2)

    # Jackknife SE
    theta_loo = []
    for r in grp1:
        loo_mat = corr_mat.drop(index=r, columns=r, errors="ignore")
        g1_loo = [x for x in grp1 if x != r]
        g2_loo = [x for x in grp2 if x != r]
        theta_loo.append(aggregate_corr(loo_mat, g1_loo, g2_loo))

    theta_loo = np.array(theta_loo)
    n = len(theta_loo)
    se = np.sqrt((n - 1) / n * np.sum((theta_loo - mean_r) ** 2)) if n > 1 else 0.0
    ci = mean_r + np.array([-1, 1]) * 1.96 * se

    return {
        "title":    title,
        "metric":   metric,
        "mean_r":   mean_r,
        "ci_2p5":   float(ci[0]),
        "ci_97p5":  float(ci[1]),
        "n_pairs":  len(grp1),
        "corr_mat": corr_mat,
    }


def get_all_cm(res_HH: dict, res_MM: dict, res_HM: dict, fill_diag: bool = True) -> pd.DataFrame:
    """Assemble a combined human+model correlation matrix from three block results."""
    human_cols = list(res_HH["corr_mat"].columns)
    model_cols = list(res_MM["corr_mat"].columns)
    all_cols = human_cols + model_cols

    cm = pd.DataFrame(np.nan, index=all_cols, columns=all_cols)
    cm.loc[human_cols, human_cols] = res_HH["corr_mat"].loc[human_cols, human_cols]
    cm.loc[model_cols, model_cols] = res_MM["corr_mat"].loc[model_cols, model_cols]
    cm.loc[human_cols, model_cols] = res_HM["corr_mat"].loc[human_cols, model_cols]
    cm.loc[model_cols, human_cols] = res_HM["corr_mat"].loc[human_cols, model_cols].T

    if fill_diag:
        np.fill_diagonal(cm.values, 1.0)
    return cm
