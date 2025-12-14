import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

def js_divergence_scores(a, b, bins=20):
    a = np.asarray(a); b = np.asarray(b)
    ha, edges = np.histogram(a, bins=bins, range=(0,1), density=True)
    hb, _     = np.histogram(b, bins=edges, density=True)
    eps = 1e-8
    ha = ha + eps; hb = hb + eps
    ha /= ha.sum(); hb /= hb.sum()
    return (jensenshannon(ha, hb, base=2) ** 2)

def alignment_metrics(human, model, bins=20):
    human = np.asarray(human)
    model = np.asarray(model)

    # Question-wise agreement
    rho, _ = spearmanr(human, model)
    r, _   = pearsonr(human, model)

    # Magnitude error
    mae  = np.mean(np.abs(model - human))
    rmse = np.sqrt(np.mean((model - human)**2))

    # Distribution shape distances
    wd  = wasserstein_distance(human, model)
    jsd = js_divergence_scores(human, model, bins=bins)

    # Distribution test
    ks, ks_p = ks_2samp(human, model)

    # Bias summaries
    dmean = float(np.mean(model) - np.mean(human))
    dstd  = float(np.std(model) - np.std(human))

    return {
        "spearman_rho": float(rho),
        "pearson_r": float(r),
        "mae": float(mae),
        "rmse": float(rmse),
        "wasserstein": float(wd),
        "jsd_hist": float(jsd),
        "ks": float(ks),
        "ks_p": float(ks_p),
        "delta_mean": dmean,
        "delta_std": dstd,
        "n": int(len(human)),
    }

def make_alignment_table(df, group_cols=("dataset","condition","model"), bins=20):
    rows = []
    for keys, sub in df.groupby(list(group_cols)):
        sub = sub.dropna(subset=["human_score","model_score"])
        if len(sub) < 5:
            continue
        m = alignment_metrics(sub["human_score"].values, sub["model_score"].values, bins=bins)
        row = dict(zip(group_cols, keys))
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)

def plot_pooled_hist(df, dataset, condition, bins=20):
    sub = df[(df["dataset"]==dataset) & (df["condition"]==condition)].dropna(
        subset=["human_score","model_score"]
    )

    human = sub["human_score"].values
    pooled_model = sub["model_score"].values  # pooled across all models

    plt.figure(figsize=(6,4))
    sns.histplot(human, bins=bins, stat="density", kde=True, alpha=0.5, label="Human")
    sns.histplot(pooled_model, bins=bins, stat="density", kde=True, alpha=0.5, label="Models (pooled)")
    plt.xlim(0,1)
    plt.title(f"{dataset} — {condition}: Human vs pooled models")
    plt.xlabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
