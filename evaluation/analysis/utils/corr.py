from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from utils.df import result_to_df 

def human_vs_consensus(human_correct):
    H, Q = human_correct.shape
    rhos = []

    for i in range(H):
        hi = human_correct[i]
        consensus_minus_i = (
            human_correct.sum(axis=0) - hi
        ) / (H - 1)

        rho, _ = spearmanr(hi, consensus_minus_i)
        rhos.append(rho)

    return np.array(rhos) 

def get_corr_from_pivot(pivoted, dataset, metric):
    return (
        pivoted[('theta_hat', dataset, metric)].to_numpy(),
        pivoted[('ci_low', dataset, metric)].to_numpy(),
        pivoted[('ci_high', dataset, metric)].to_numpy(),
    )

def pairwise_metric(a, b, metric="spearman"):
    """
    a, b: pandas Series (aligned, no NaNs)
    """

    if metric == "spearman":
        return spearmanr(a, b).statistic
    elif metric == "pearson":
        return a.corr(b)
    elif metric == "mcc":
        return matthews_corrcoef(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def bootstrap_spearman(a, b, n_boot=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(a)

    boot_stats = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)   # resample indices
        boot_stats[i] = spearmanr(a[idx], b[idx]).statistic

    return boot_stats

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

    if list(grp1) == list(grp2): 
        # within-group: use upper triangle only
        mask = np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
        flat_corrs = corr_mat.where(mask).stack()
    else:
        # cross-group: use all pairs
        flat_corrs = corr_mat.stack()

    flat_corrs = flat_corrs.dropna()
    z = np.arctanh(np.clip(flat_corrs.values, -0.999, 0.999))
    mean_r = np.tanh(z.mean())
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
    score_type = 'Accuracy' if metrics =='correct' else "Embedding Similarity"

    human_acc = human_vqa.pivot(
        index="question_id",
        columns="participant_id",
        values=metrics
    )

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

    answers_pivot = human_acc.join(model_acc, how="inner")
    res_HM = interrater_agreement(
        answers_pivot,
        grp1=human_acc.columns,
        grp2=model_acc.columns,
        n_boot=n_boot,
        metric="spearman",
        title=f"Human–Model {score_type}",
    )

    if n_boot:  
        transform_corr_table(pd.concat([result_to_df(res_HH), result_to_df(res_MM), result_to_df(res_HM)]))
        # .to_latex('/home/work/yuna/HPA/evaluation/analysis/tables/3_spearman-interrater-vqa.tex', float_format="%.3f")    
    
    return res_HH, res_MM, res_HM 

def corr_by_regime(
    df,
    x_col="acc_corr",
    y_col="MG_Acc",
    regime_col="regime",
):
    rows = []

    for reg, sub in df.groupby(regime_col):
        # drop NaNs pairwise
        print(reg, sub.model.unique())
        sub = sub[[x_col, y_col]].dropna()

        if len(sub) < 3:
            # too few points for meaningful correlation
            r, rho = float("nan"), float("nan")
        else:
            r, _ = pearsonr(sub[x_col], sub[y_col])
            rho, _ = spearmanr(sub[x_col], sub[y_col])

        rows.append({
            "Regime": reg,
            "models": len(sub), 
            "Pearson r": r,
            "Spearman ρ": rho,
            "R²": r**2 if pd.notna(r) else float("nan"),
        })

    return pd.DataFrame(rows)

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

    if metric == "spearman":
        return spearmanr(a, b).statistic
    elif metric == "pearson":
        return a.corr(b)
    elif metric == "mcc":
        return matthews_corrcoef(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")



def get_all_cm(res_HH, res_MM, res_HM, fill_diag=True):

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