from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef


# ---------------------------
# Pairwise metric
# ---------------------------
def pairwise_metric(a, b, metric="spearman"):
    """
    a, b: pandas Series (aligned, no NaNs)
    """
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan

    if metric == "spearman":
        return spearmanr(a, b).statistic
    elif metric == "pearson":
        return a.corr(b)
    elif metric == "mcc":
        return matthews_corrcoef(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------
# Core agreement function
# ---------------------------
def interrater_agreement(
    answers_pivot: pd.DataFrame,
    grp1,
    grp2,
    metric="spearman",
    n_boot=200,
    title="Agreement",
    plot=True,
    save=None,
):
    """
    answers_pivot:
        rows = questions
        columns = raters (humans or models)
        values = accuracy (0/1) or continuous score

    grp1, grp2:
        lists of column names
        - same list → within-group (HH or MM)
        - different lists → cross-group (HM)

    Returns:
        dict with mean agreement + bootstrap CI
    """

    # ---------------------------
    # Pairwise correlation matrix
    # ---------------------------
    corr_mat = pd.DataFrame(np.nan, index=grp1, columns=grp2)

    for r1 in grp1:
        for r2 in grp2:
            if r1 == r2:
                continue

            valid = answers_pivot[[r1, r2]].dropna()
            if len(valid) < 5:
                continue

            corr_mat.loc[r1, r2] = pairwise_metric(
                valid[r1], valid[r2], metric=metric
            )

    # ---------------------------
    # Flatten correlations
    # ---------------------------
    if list(grp1) == list(grp2): 
        # within-group: use upper triangle only
        mask = np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
        flat_corrs = corr_mat.where(mask).stack()
    else:
        # cross-group: use all pairs
        flat_corrs = corr_mat.stack()

    flat_corrs = flat_corrs.dropna()

    # ---------------------------
    # Fisher z aggregation
    # ---------------------------
    z = np.arctanh(np.clip(flat_corrs.values, -0.999, 0.999))
    mean_r = np.tanh(z.mean())

    # ---------------------------
    # Bootstrap CI over questions
    # ---------------------------
    q_ids = answers_pivot.index.values
    boot_means = []

    for _ in tqdm(range(n_boot)):
        sampled_qs = np.random.choice(q_ids, size=len(q_ids), replace=True)
        boot = answers_pivot.loc[sampled_qs]

        tmp = []
        for r1 in grp1:
            for r2 in grp2:
                if r1 == r2:
                    continue
                valid = boot[[r1, r2]].dropna()
                if len(valid) < 5:
                    continue
                r = pairwise_metric(valid[r1], valid[r2], metric=metric)
                if not np.isnan(r):
                    tmp.append(r)

        if len(tmp) > 0:
            z_tmp = np.arctanh(np.clip(tmp, -0.999, 0.999))
            boot_means.append(np.tanh(z_tmp.mean()))

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # ---------------------------
    # Plot heatmap (optional)
    # ---------------------------
    if plot:
        n = len(grp1)
        fig_size = 6 + 0.4 * n
        plt.figure(figsize=(fig_size, fig_size * 0.7))
        sns.heatmap(
            corr_mat,
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            annot=True,
            square=True,
        )
        plt.title(f"{title}\n{metric} mean r={mean_r:.2f}")
        plt.tight_layout()
        if save:
            plt.savefig(f"{save}/{title}.png", dpi=300)
        plt.close()

    # ---------------------------
    # Return summary
    # ---------------------------
    return {
        "title": title,
        "metric": metric,
        "mean_r": mean_r,
        "ci_2p5": ci_low,
        "ci_97p5": ci_high,
        "min_r": flat_corrs.min(),
        "max_r": flat_corrs.max(),
        "n_pairs": len(flat_corrs),
        "corr_mat": corr_mat,   
    }
