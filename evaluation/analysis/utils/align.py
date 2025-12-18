import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
from math import atanh, tanh, sqrt 
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import cohen_kappa_score
from math import atanh, tanh, sqrt

def js_divergence_scores(a, b, bins=20):
    a = np.asarray(a); b = np.asarray(b)
    ha, edges = np.histogram(a, bins=bins, range=(0,1), density=True)
    hb, _     = np.histogram(b, bins=edges, density=True)
    eps = 1e-8
    ha = ha + eps; hb = hb + eps
    ha /= ha.sum(); hb /= hb.sum()

    return (jensenshannon(ha, hb, base=2) ** 2)

def calculate_alignment_suite(human, model, bins=20):
    h = np.array(human)
    m = np.array(model)
    n = len(h)

    # 1. CONSISTENCY & PRECISION (Pearson + 95% CI)
    r_val, p_val = pearsonr(h, m)
    r_clip = np.clip(r_val, -0.9999, 0.9999)
    z = atanh(r_clip)
    se = 1 / sqrt(n - 3)
    ci_low, ci_high = tanh(z - 1.96 * se), tanh(z + 1.96 * se)
    
    rho, _ = spearmanr(h, m)

    # 2. AGREEMENT (Quadratic Kappa)
    # Bins scores into deciles to measure ordinal agreement
    h_binned = np.round(h / 10).astype(int)
    m_binned = np.round(m / 10).astype(int)
    kappa = cohen_kappa_score(h_binned, m_binned, weights='quadratic')

    # 3. DISTRIBUTIONAL DISTANCE (TVD, JS, KS)
    bin_edges = np.linspace(0, 100, bins + 1)
    h_counts, _ = np.histogram(h, bins=bin_edges)
    m_counts, _ = np.histogram(m, bins=bin_edges)
    
    # Probability Mass Function (Sum to 1)
    h_pmf = h_counts / (np.sum(h_counts) + 1e-10)
    m_pmf = m_counts / (np.sum(m_counts) + 1e-10)
    
    tvd = 0.5 * np.sum(np.abs(h_pmf - m_pmf))
    js_dist = jensenshannon(h_pmf, m_pmf, base=2)
    ks_stat, ks_p = ks_2samp(h, m)
    wd = wasserstein_distance(h, m)

    # 4. BIAS SUMMARY
    d_mean = np.mean(m) - np.mean(h)

    return {
        "Pearson_r": round(r_val, 3),
        "r_95_CI": [round(ci_low, 3), round(ci_high, 3)],
        "Spearman_rho": round(rho, 3),
        "Kappa_Quad": round(kappa, 3),
        "TVD": round(tvd, 3),
        "JS_Dist": round(js_dist, 3),
        "KS_Stat": round(ks_stat, 3),
        "Wasserstein": round(wd, 3),
        "Delta_Mean": round(d_mean, 3)
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