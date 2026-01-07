from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau 
from sklearn.metrics import matthews_corrcoef
from utils.plots import scatterplot, get_family, parse_size  
from utils.df import result_to_df 

# from __future__ import annotations
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple, Union

Label = Union[str, int, float]
from collections import Counter
from typing import List, Optional, Sequence, Tuple, Union

Label = Union[str, int, float]


def gwet_ac1(
    rater_a: Sequence[Optional[Label]],
    rater_b: Sequence[Optional[Label]],
    *,
    categories: Optional[Sequence[Label]] = None,
    drop_missing: bool = True,
) -> float:
    """
    Compute Gwet's AC1 (two raters, nominal categories).

    References:
      Gwet (AC1): Pe = (1/(q-1)) * sum_k π_k(1-π_k),
      π_k = (p_{k+} + p_{+k})/2  (q = number of categories)
    """

    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length.")

    def is_missing(x) -> bool:
        if x is None:
            return True
        try:
            return isinstance(x, float) and x != x  # NaN
        except Exception:
            return False

    paired: List[Tuple[Label, Label]] = []
    for a, b in zip(rater_a, rater_b):
        if is_missing(a) or is_missing(b):
            if drop_missing:
                continue
            raise ValueError("Missing value encountered; set drop_missing=True to drop.")
        paired.append((a, b))

    n = len(paired)
    if n == 0:
        raise ValueError("No valid (non-missing) paired ratings to score.")

    # Categories
    if categories is None:
        cats = sorted({x for ab in paired for x in ab}, key=lambda x: (str(type(x)), str(x)))
    else:
        cats = list(categories)

    q = len(cats)
    if q < 2:
        raise ValueError("AC1 requires at least 2 categories.")

    cat_set = set(cats)
    for a, b in paired:
        if a not in cat_set or b not in cat_set:
            raise ValueError(f"Found label not in categories: {(a, b)}")

    # Observed agreement Po
    po = sum(1 for a, b in paired if a == b) / n

    # π_k: pooled marginal across the two raters (equivalent to (p_{k+}+p_{+k})/2)
    counts = Counter()
    for a, b in paired:
        counts[a] += 1
        counts[b] += 1
    pi = {k: counts[k] / (2 * n) for k in cats}

    # Correct Pe for AC1 (note the 1/(q-1) factor)
    pe = (1 / (q - 1)) * sum(pi[k] * (1 - pi[k]) for k in cats)

    denom = 1 - pe
    if denom == 0:
        return 1.0 if po == 1.0 else 0.0

    ac1 = (po - pe) / denom

    # Numerical safety (should not be needed, but prevents tiny float overshoots)
    if ac1 > 1 and ac1 < 1 + 1e-12:
        ac1 = 1.0
    if ac1 < -1 and ac1 > -1 - 1e-12:
        ac1 = -1.0

    return ac1


def pairwise_metric(a, b, metric="spearman"):
    """
    a, b: pandas Series (aligned, no NaNs)
    """

    if metric == "spearman":
        return spearmanr(a, b).statistic
    elif metric == 'kendall':
        return kendalltau(a, b).statistic 
    elif metric == "pearson":
        return a.corr(b)
    elif metric == "mcc":
        return matthews_corrcoef(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _extract_matrix(interrater_output):
    """Extract correlation matrix from interrater_agreement output."""
    return interrater_output['corr_mat']
    
def get_corr(human_vqa, model_vqa, test_subjects, correlation_type="spearman", metrics="correct", n_boot=0):
    model_corr = []

    # -----------------------
    # Humans wide matrix
    # -----------------------
    human_acc = human_vqa.pivot(
        index="question_id",
        columns="participant_id",
        values=metrics
    )

    # keep only requested subjects that actually exist
    test_subjects = [s for s in test_subjects if s in human_acc.columns]
    if not test_subjects:
        raise ValueError("No valid test_subjects found in human_vqa data")
    
    human_acc = human_acc[test_subjects]

    # HH
    res_HH_out = interrater_agreement(
        human_acc,
        grp1=test_subjects,
        grp2=test_subjects,
        n_boot=n_boot,
        metric=correlation_type,
        title=f"Human-Human {metrics}",
    )
    res_HH = _extract_matrix(res_HH_out)

    # -----------------------
    # LLM (Qwen vs humans)
    # -----------------------
    qwen_models = ["Qwen3 (8B)", "Qwen3 (4B)"]
    qwen_subset = model_vqa[model_vqa["model"].isin(qwen_models)]
    
    llm = None
    if not qwen_subset.empty:
        qwen_acc = qwen_subset.pivot(
            index="question_id",
            columns="model",
            values=metrics
        )
        
        llm_mat = human_acc.join(qwen_acc, how="inner")
        
        if not llm_mat.empty:
            llm_out = interrater_agreement(
                llm_mat,
                grp1=test_subjects,
                grp2=qwen_models,
                n_boot=n_boot,
                metric="spearman",
                title="LLM (Qwen vs Humans)",
            )
            llm = _extract_matrix(llm_out)

    # -----------------------
    # Pretrained models vs humans / vs models
    # -----------------------
    pretrained = model_vqa[model_vqa["finetuned"] == "Pretrained"].copy()

    res_HM = None
    res_MM = None
    
    if not pretrained.empty:
        pretrained_acc = pretrained.pivot(
            index="question_id",
            columns="model",
            values=metrics
        )
        pretrained_models = list(pretrained_acc.columns)

        hm_mat = human_acc.join(pretrained_acc, how="inner")
        
        if not hm_mat.empty and pretrained_models:
            res_HM_out = interrater_agreement(
                hm_mat,
                grp1=test_subjects,
                grp2=pretrained_models,
                n_boot=n_boot,
                metric="spearman",
                title=f"Human–PretrainedModel {metrics}",
            )
            res_HM = _extract_matrix(res_HM_out)

            res_MM_out = interrater_agreement(
                hm_mat,
                grp1=pretrained_models,
                grp2=pretrained_models,
                n_boot=n_boot,
                metric="spearman",
                title=f"PretrainedModel–PretrainedModel {metrics}",
            )
            res_MM = _extract_matrix(res_MM_out)

    # -----------------------
    # Per (model, finetuned): human-model correlation summary
    # -----------------------
    for model in model_vqa["model"].unique():
        if model in qwen_models:
            continue

        submodel = model_vqa[model_vqa["model"] == model]
        for finetuned in submodel["finetuned"].unique():
            one = submodel[submodel["finetuned"] == finetuned]
            
            if one.empty:
                continue

            # Create single column DataFrame for this model variant
            colname = f"{model} :: {finetuned}"
            model_acc = one[["question_id", metrics]].set_index("question_id")
            model_acc = model_acc.rename(columns={metrics: colname})

            mat = human_acc.join(model_acc, how="inner")
            
            if mat.empty or colname not in mat.columns:
                continue

            try:
                out = interrater_agreement(
                    mat,
                    grp1=test_subjects,
                    grp2=[colname],
                    n_boot=n_boot,
                    metric="spearman",
                    title=f"Human–Model {metrics}",
                )
                m = _extract_matrix(out)

                # mean correlation across humans for this model variant
                mean_hm = m[colname].mean()
                model_corr.append({
                    "model": model,
                    "finetuned": finetuned,
                    "mean_human_model_corr": float(mean_hm),
                })
            except Exception as e:
                print(f"Warning: Failed for {colname}: {e}")
                continue

    model_corr = pd.DataFrame(model_corr)
    if not model_corr.empty:
        model_corr["family"] = model_corr["model"].map(get_family)
        model_corr["model_size"] = model_corr["model"].map(parse_size).astype(int)

    return res_HH_out, llm_out, model_corr, res_HM, res_MM

def aggregate_corr(corr_mat, grp1, grp2):
    if list(grp1) == list(grp2):
        mask = np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
        flat = corr_mat.where(mask).stack()
    else:
        flat = corr_mat.stack()

    flat = flat.dropna()
    return flat.mean()

def interrater_agreement(
    answers_pivot: pd.DataFrame,
    grp1,
    grp2,
    metric="spearman",
    n_boot=200,
    title="Agreement",
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

    mean_r = aggregate_corr(corr_mat, grp1, grp2) 
    theta_loo = []

    for r in grp1:
        corr_mat_loo = corr_mat.drop(index=r, columns=r, errors="ignore")
        grp1_loo = [x for x in grp1 if x != r]
        grp2_loo = [x for x in grp2 if x != r]

        theta_loo.append(aggregate_corr(corr_mat_loo, grp1_loo, grp2_loo))

    theta_loo = np.array(theta_loo)
    n = len(theta_loo)
    se = np.sqrt((n - 1) / n * np.sum((theta_loo - mean_r) ** 2))
    ci = mean_r + np.array([-1, 1]) * 1.96 * se

    return {
        "title": title,
        "metric": metric,
        "mean_r": mean_r,
        "ci_2p5": ci[0],
        "ci_97p5": ci[1], 
        "n_pairs": len(grp1),
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

    res_HM = interrater_agreement(
        human_acc.join(model_acc, how="inner"),
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
    elif metric == "ac1":
        return gwet_ac1(a, b) 
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

def plot_correlation_heatmap(res_HH, llm, res_HM, res_MM, test_subjects, save_path=None):
    """
    Create unified correlation heatmap with proper structure.
    Fill diagonals and match the reference format.
    """
    
    # Create mapping: original participant_id -> number
    human_map = {pid: i+1 for i, pid in enumerate(sorted(test_subjects))}
    
    # Get all unique raters (humans + models)
    all_raters = set()
    
    # Collect all rater names
    for mat in [res_HH, llm, res_HM, res_MM]:
        if mat is not None:
            all_raters.update(mat.index)
            all_raters.update(mat.columns)
    
    # Separate humans and models, humans as integers for proper sorting
    humans_ids = sorted([human_map[r] for r in all_raters if r in human_map])
    models = sorted([r for r in all_raters if r not in human_map])
    
    # Order: humans first (as numbers), then models
    ordered_raters = humans_ids + models
    
    # Create empty correlation matrix
    n = len(ordered_raters)
    combined = pd.DataFrame(
        np.nan,  # Initialize with NaN
        index=ordered_raters,
        columns=ordered_raters
    )
    
    # Fill diagonal with 1.0
    for i in range(n):
        combined.iloc[i, i] = 1.0
    
    # Reverse mapping for filling
    reverse_human_map = {v: k for k, v in human_map.items()}
    
    # Helper to fill matrix
    def fill_correlations(mat, human_map, reverse_human_map):
        if mat is None:
            return
        
        for i in mat.index:
            for j in mat.columns:
                # Map to numbers for humans, keep names for models
                i_mapped = human_map.get(i, i)
                j_mapped = human_map.get(j, j)
                
                if i_mapped in combined.index and j_mapped in combined.columns:
                    combined.loc[i_mapped, j_mapped] = mat.loc[i, j]
                    # Make symmetric
                    if i != j:  # Don't double-fill diagonal
                        combined.loc[j_mapped, i_mapped] = mat.loc[i, j]
    
    # Fill all correlations
    fill_correlations(res_HH, human_map, reverse_human_map)
    fill_correlations(llm, human_map, reverse_human_map)
    fill_correlations(res_HM, human_map, reverse_human_map)
    fill_correlations(res_MM, human_map, reverse_human_map)
    
    # Replace remaining NaN with 0 (or keep as NaN for white cells)
    combined = combined.fillna(1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(18, 16))
    
    sns.heatmap(
        combined,
        cmap='YlGnBu',
        vmin=-1,
        vmax=1.0,
        cbar_kws={'label': 'Spearman Correlation', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='white',
        square=True,
        ax=ax
    )
    
    # Add dividing line between humans and models
    n_humans = len(humans_ids)
    ax.axhline(n_humans, color='red', linewidth=3)
    ax.axvline(n_humans, color='red', linewidth=3)
    
    # Add text labels for sections
    ax.text(n_humans/2, 0, 'Humans', ha='center', va='bottom', 
        fontsize=16, weight='bold', color='navy')

    ax.text(n_humans + len(models)/2, 0, 'Models', ha='center', va='bottom',
        fontsize=16, weight='bold', color='orange')
    
    # plt.title('Human-Model Agreement (Spearman Correlation)', fontsize=18, pad=35, weight='bold')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=90, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return combined, human_map 