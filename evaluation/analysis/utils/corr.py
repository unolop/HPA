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

def transform_corr_table(df): 
    df["comparison"] = df["title"].str.extract(
        r"(Human–Human|Human–Model|Model–Model)"
    )

    df["signal"] = df["title"].str.extract(
        r"(Accuracy|Embedding Similarity)"
    )

    # Clean signal naming
    df["signal"] = df["signal"].replace({
        "Embedding Similarity": "Embedding"
    })

    # 2. Create formatted mean [CI] column
    df["rho_ci"] = (
        df["mean_r"].round(3).astype(str)
        + " ["
        + df["ci_2p5"].round(3).astype(str)
        + ", "
        + df["ci_97p5"].round(3).astype(str)
        + "]"
    )
    # 3. Pivot table
    pivot = (
        df.pivot(
            index="comparison",
            columns="signal",
            values="rho_ci"
        )
        .reindex(
            ["Human–Human", "Human–Model", "Model–Model"]
        )
    )
    # 4. Rename columns for LaTeX
    pivot = pivot.rename(columns={
        "Accuracy": "Accuracy $\\rho$ ",
        "Embedding": "Embedding $\\rho$ "
    })
    return pivot 


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
    ci_low = 0
    ci_high = 0
    if n_boot : 
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


def get_agreements(human_vqa, model_vqa, metrics="correct", n_boot=200): 
    human_acc = human_vqa.pivot(
        index="question_id",
        columns="participant_id",
        values=metrics
    )

    score_type = 'Accuracy' if metrics =='correct' else "Embedding Similarity"

    res_HH = interrater_agreement(
        human_acc,
        grp1=human_acc.columns,
        grp2=human_acc.columns,
        n_boot=n_boot,
        metric="spearman",
        title=f"Human–Human {score_type}",
    )

    model_acc = model_vqa.pivot(
        index="question_id",
        columns="model",
        values=metrics
    )

    res_MM = interrater_agreement(
        model_acc,
        grp1=model_acc.columns,
        grp2=model_acc.columns,
        n_boot=n_boot,
        metric="spearman",
        title=f"Model–Model {score_type}",
    )

    human_acc = human_vqa.pivot(
        index="question_id",
        columns="participant_id",
        values=metrics
    )

    model_acc = model_vqa.pivot(
        index="question_id",
        columns="model",
        values=metrics 
    )

    answers_pivot = human_acc.join(model_acc, how="inner")

    res_HM = interrater_agreement(
        answers_pivot,
        grp1=human_acc.columns,
        grp2=model_acc.columns,
        n_boot=n_boot,
        metric="spearman",
        title=f"Human–Model {score_type}",
    )
    return res_HH, res_MM, res_HM 

def plot_unified_agreement_heatmap(
    corr_mat,
    human_cols,
    model_cols,
    title="Unified Human–Model Agreement",
    metric_label="Spearman $\\rho$",
    cmap="viridis",
    margin=0.02,
    figsize_base=6,
    figsize_scale=0.35,
    save=None,
):
    """
    Plot a unified correlation heatmap including humans and models.

    Parameters
    ----------
    corr_mat : pd.DataFrame
        Square correlation matrix indexed by all raters (humans + models).
    human_cols : list
        Column names corresponding to human raters.
    model_cols : list
        Column names corresponding to models.
    title : str
        Figure title.
    metric_label : str
        Label for the colorbar.
    cmap : str
        Sequential colormap (e.g., 'viridis', 'mako', 'crest').
    margin : float
        Extra margin added to vmin/vmax for visual contrast.
    figsize_base : float
        Base figure size.
    figsize_scale : float
        Additional size per rater.
    save : str or None
        Path to save figure. If None, figure is shown.
    """

    # --------------------------------------------------
    # 1. Reorder matrix: humans first, models second
    # --------------------------------------------------
    ordered = list(human_cols) + list(model_cols)
    corr_mat = corr_mat.loc[ordered, ordered]

    # --------------------------------------------------
    # 2. Compute color limits from data
    # --------------------------------------------------
    vals = corr_mat.values
    vals = vals[np.isfinite(vals)]

    vmin = vals.min() - margin
    vmax = vals.max() + margin

    # --------------------------------------------------
    # 3. Figure size
    # --------------------------------------------------
    n = len(ordered)
    fig_size = figsize_base + figsize_scale * n

    plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(
        corr_mat,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"label": metric_label},
        linewidths=0.3,
        linecolor="white",
    )

    # --------------------------------------------------
    # 4. Draw separators between humans and models
    # --------------------------------------------------
    h = len(human_cols)
    ax.axhline(h, color="black", lw=2)
    ax.axvline(h, color="black", lw=2)

    # --------------------------------------------------
    # 5. Style tick labels by group
    # --------------------------------------------------
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        if label.get_text() in human_cols:
            label.set_color("tab:blue")
            label.set_fontweight("bold")
        else:
            label.set_color("tab:orange")

    for label in ax.get_yticklabels():
        if label.get_text() in human_cols:
            label.set_color("tab:blue")
            label.set_fontweight("bold")
        else:
            label.set_color("tab:orange")

    # --------------------------------------------------
    # 6. Optional group annotations
    # --------------------------------------------------
    ax.text(
        h / 2,
        -1.5,
        "Humans",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="tab:blue",
    )
    ax.text(
        h + (n - h) / 2,
        -1.5,
        "Models",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="tab:orange",
    )

    # --------------------------------------------------
    # 7. Final formatting
    # --------------------------------------------------
    plt.title(title, fontsize=16)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def get_all_cm(res_HH, res_MM, res_HM, fill_diag=True):
    """
    Combine HH, MM, and HM correlation matrices into a unified matrix.

    Assumes:
    - res_HH["corr_mat"]: Human × Human
    - res_MM["corr_mat"]: Model × Model
    - res_HM["corr_mat"]: Human × Model
    """

    human_cols = list(res_HH["corr_mat"].columns)
    model_cols = list(res_MM["corr_mat"].columns)

    all_cols = human_cols + model_cols

    # Initialize empty matrix
    corr_mat = pd.DataFrame(
        np.nan,
        index=all_cols,
        columns=all_cols
    )

    # -----------------------
    # Human–Human block
    # -----------------------
    corr_mat.loc[human_cols, human_cols] = (
        res_HH["corr_mat"]
        .loc[human_cols, human_cols]
    )

    # -----------------------
    # Model–Model block
    # -----------------------
    corr_mat.loc[model_cols, model_cols] = (
        res_MM["corr_mat"]
        .loc[model_cols, model_cols]
    )

    # -----------------------
    # Human–Model block
    # -----------------------
    corr_mat.loc[human_cols, model_cols] = (
        res_HM["corr_mat"]
        .loc[human_cols, model_cols]
    )

    # -----------------------
    # Model–Human block (transpose)
    # -----------------------
    corr_mat.loc[model_cols, human_cols] = (
        res_HM["corr_mat"]
        .loc[human_cols, model_cols]
        .T
    )

    # -----------------------
    # Optional: fill diagonal
    # -----------------------
    if fill_diag:
        np.fill_diagonal(corr_mat.values, 1.0)

    return corr_mat