import numpy as np
from scipy.stats import wasserstein_distance, spearmanr, ks_2samp
from scipy.spatial.distance import jensenshannon


def compute_distribution_alignment(
    H,
    M,
    n_bins=20,
    compute_ks=True,
):
    """
    Compute distributional alignment metrics between human and model
    per-question aggregated scores.

    Parameters
    ----------
    H : array-like
        Per-question human scores (e.g., mean accuracy or similarity).
    M : array-like
        Per-question model scores (same questions as H).
    n_bins : int
        Number of bins for JS divergence (only affects JS).
    compute_ks : bool
        Whether to compute KS statistic (descriptive only).

    Returns
    -------
    dict
        Dictionary of distributional alignment metrics.
    """

    H = np.asarray(H, dtype=float)
    M = np.asarray(M, dtype=float)

    # Drop NaNs pairwise
    mask = ~np.isnan(H) & ~np.isnan(M)
    H = H[mask]
    M = M[mask]

    assert len(H) == len(M), "H and M must have same length after NaN removal"

    # ---------------------------
    # Core metrics
    # ---------------------------
    delta_mean = float(M.mean() - H.mean())
    wasserstein = float(wasserstein_distance(H, M))
    spearman = float(spearmanr(H, M).statistic)

    # ---------------------------
    # Jensen–Shannon divergence
    # (computed on binned distributions)
    # ---------------------------
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    H_hist, _ = np.histogram(H, bins=bins, density=True)
    M_hist, _ = np.histogram(M, bins=bins, density=True)

    # Add small epsilon for numerical stability
    eps = 1e-8
    H_hist = H_hist + eps
    M_hist = M_hist + eps

    js_div = float(jensenshannon(H_hist, M_hist, base=2.0) ** 2)

    # ---------------------------
    # Optional KS statistic (descriptive)
    # ---------------------------
    if compute_ks:
        ks_stat, ks_p = ks_2samp(H, M)
        ks_stat = float(ks_stat)
        ks_p = float(ks_p)
    else:
        ks_stat, ks_p = None, None

    return {
        "n_questions": int(len(H)),
        "human_mean": float(H.mean()),
        "model_mean": float(M.mean()),
        "delta_mean": delta_mean,
        "wasserstein": wasserstein,
        "spearman": spearman,
        "js_divergence": js_div,
        "ks_stat": ks_stat,
        "ks_p": ks_p,  # include but DO NOT emphasize in paper
    }
