from collections import Counter
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 


def gwet_ac1(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    po = np.mean(a == b)

    cats = np.unique(np.concatenate([a, b]))
    pa = Counter(a)
    pb = Counter(b)

    n = len(a)
    pe = 0.0
    for c in cats:
        pa_c = pa[c] / n
        pb_c = pb[c] / n
        pe += pa_c * (1 - pb_c) + pb_c * (1 - pa_c)
    pe *= 0.5

    return (po - pe) / (1 - pe) if (1 - pe) > 0 else np.nan
    
    
def interrater_agreement_ac1(
    answers_pivot: pd.DataFrame,
    grp1,
    grp2,
    n_boot=200,
    title="Agreement (AC1)",
    plot=True,
    save=None,
):
    """
    answers_pivot:
        rows = questions
        columns = raters (humans or models)
        values = categorical answers

    grp1, grp2:
        lists of column names
        - same list → within-group (HH or MM)
        - different lists → cross-group (HM)

    Returns:
        dict with mean AC1 + bootstrap CI
    """

    # ---------------------------
    # Pairwise AC1 matrix
    # ---------------------------
    ac1_mat = pd.DataFrame(np.nan, index=grp1, columns=grp2)

    for r1 in grp1:
        for r2 in grp2:
            if r1 == r2:
                continue

            valid = answers_pivot[[r1, r2]].dropna()
            if len(valid) < 5:
                continue

            ac1_mat.loc[r1, r2] = gwet_ac1(valid[r1], valid[r2])

    # ---------------------------
    # Flatten AC1 values
    # ---------------------------
    if list(grp1) == list(grp2):
        mask = np.triu(np.ones(ac1_mat.shape), k=1).astype(bool)
        flat_vals = ac1_mat.where(mask).stack()
    else:
        flat_vals = ac1_mat.stack()

    flat_vals = flat_vals.dropna()

    mean_ac1 = flat_vals.mean()

    # ---------------------------
    # Bootstrap CI over questions
    # ---------------------------
    q_ids = answers_pivot.index.values
    ci_low = np.nan
    ci_high = np.nan

    if n_boot:
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
                    v = gwet_ac1(valid[r1], valid[r2])
                    if not np.isnan(v):
                        tmp.append(v)

            if len(tmp) > 0:
                boot_means.append(np.mean(tmp))

        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # ---------------------------
    # Plot heatmap (optional)
    # ---------------------------
    if plot:
        n = len(grp1)
        fig_size = 6 + 0.4 * n
        plt.figure(figsize=(fig_size, fig_size * 0.7))
        sns.heatmap(
            ac1_mat,
            vmin=0,
            vmax=1,
            cmap="viridis",
            annot=True,
            square=True,
        )
        plt.title(f"{title}\nmean AC1={mean_ac1:.2f}")
        plt.tight_layout()
        if save:
            plt.savefig(f"{save}/{title}.png", dpi=300)
        plt.close()

    # ---------------------------
    # Return summary (same keys)
    # ---------------------------
    return {
        "title": title,
        "metric": "AC1",
        "mean_r": mean_ac1,        # kept for API compatibility
        "ci_2p5": ci_low,
        "ci_97p5": ci_high,
        "min_r": flat_vals.min(),
        "max_r": flat_vals.max(),
        "n_pairs": len(flat_vals),
        "corr_mat": ac1_mat,       # same key, different semantics
    }