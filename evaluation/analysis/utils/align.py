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


def get_pearsonr_correlation(values_dict):   
    """
    dictionary of 2 keys each with an array of same legnth
    """  
    x_name,y_name = values_dict.keys()  
    x = values_dict[x_name]
    y = values_dict[y_name] 

    n = len(x) 
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))  
    r_val, p_val = pearsonr(x, y) 

    # 95% CI via Fisher z
    z = atanh(r_val)
    se = 1 / sqrt(n - 3)
    z_crit = 1.96  # 95% CI

    ci_low = tanh(z - z_crit * se)
    ci_high = tanh(z + z_crit * se)
    return {
        f"mean_{x_name}": x_mean, 
        f"mean_{y_name}": y_mean,  
        "method": "pearson",
        "r": round(float(r_val), 3),
        "p_value": round(float(p_val), 4),
        "n": n,
        "ci_95": [round(float(ci_low), 3), round(float(ci_high), 3)]
    } 

def distributional_alignment(
    human_probs: torch.Tensor,
    model_probs: torch.Tensor,
    mode: str = "JS",
    eps: float = 1e-12,
) -> float:
    """
    Compute distributional alignment between human and model distributions.

    Args:
        human_probs: Tensor [K] — human answer distribution (sums to 1)
        model_probs: Tensor [K] — model answer distribution (sums to 1)
        mode: "JS" or "CE"
        eps: numerical stability

    Returns:
        Scalar alignment loss (lower = more aligned)
    """
    h = human_probs.clamp(min=eps)
    m = model_probs.clamp(min=eps)

    if mode == "CE":
        # Cross-entropy H(h, m)
        return float(-(h * torch.log(m)).sum())

    if mode == "JS":
        # Jensen–Shannon divergence
        mid = 0.5 * (h + m)
        js = 0.5 * (h * (torch.log(h) - torch.log(mid))).sum() + \
             0.5 * (m * (torch.log(m) - torch.log(mid))).sum()
        return float(js)

    raise ValueError(f"Unknown mode: {mode}")


def question_alignment(
    human_confidences: list,
    model_counts: list,
    mode: str = "JS",
) -> float:
    """
    Compute alignment for one question.

    Args:
        human_confidences: list[float], e.g. [0.5, 0.3, 0.2]
        model_counts: list[int], number of times model produced each answer
        mode: "JS" or "CE"

    Returns:
        alignment score
    """
    human = torch.tensor(human_confidences, dtype=torch.float32)
    human = human / human.sum()

    model = torch.tensor(model_counts, dtype=torch.float32)
    model = model / model.sum()

    return distributional_alignment(human, model, mode=mode)

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