from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from config import MODELS_ALL, MODEL_GROUP, MIN_ANSWERS_DEFAULT
from helpers import load_cleaned_pair_cache, load_human_subset, read_export, filter_abstained_pairs
from utils.constants import GROUP_COLORS, GROUP_HOLLOW, MODEL_SIZE_B
from analysis.utils.abstention import classify, is_abstained

OUT_DIR = ROOT / "figures/scale_main_panels_vABC_inst_blind"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_OUT = ROOT / "latex/AAAI2026/LaTeX/figures/scale_metrics_vABC_inst_blind"
LATEX_OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["C", "B", "A"]
EXCLUDED_MODELS = {"Mistral-7B", "Phi-3.5-mini"}
MODELS_SIZED = [m for m in MODELS_ALL if m in MODEL_SIZE_B and m not in EXCLUDED_MODELS]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

FAMILY_MARKER = {
    "InternVL": "o",
    "Qwen3": "X",
    "LLaVA-1.5": "D",
    "Mistral": "^",
    "Vicuna": "v",
    "Qwen3 (think)": "P",
    "Qwen2.5": "s",
    "Other": "*",
}
FAMILY_ORDER = ["InternVL", "Qwen3", "LLaVA-1.5", "Mistral", "Vicuna", "Qwen3 (think)", "Qwen2.5"]


def save_dual(fig, local_name: str, latex_name: str | None = None):
    fig.savefig(OUT_DIR / local_name, dpi=220, bbox_inches="tight")
    fig.savefig(LATEX_OUT / (latex_name or local_name), dpi=220, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / local_name}")


def _get_family(model: str) -> str:
    if "InternVL" in model:
        return "InternVL"
    if "Qwen3-VL" in model:
        return "Qwen3"
    if "Qwen3" in model and "think" in model.lower():
        return "Qwen3 (think)"
    if "Qwen3" in model:
        return "Qwen3"
    if "Qwen2.5" in model:
        return "Qwen2.5"
    if "LLaVA-1.5" in model or "llava-1.5" in model.lower():
        return "LLaVA-1.5"
    if "Mistral" in model:
        return "Mistral"
    if "Vicuna" in model or "vicuna" in model.lower():
        return "Vicuna"
    return "Other"


def _fisher_r_ci(r: float, n: int) -> float:
    if not np.isfinite(r) or n <= 3:
        return np.nan
    r_clip = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r_clip)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = 1.959963984540054
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return float(max(abs(r - lo), abs(hi - r)))


def _x_with_shift(size: float, family: str) -> float:
    fam_list = list(FAMILY_MARKER.keys())
    idx = fam_list.index(family) if family in fam_list else 0
    n = len(fam_list)
    shift = (idx / (n - 1) - 0.5) * 0.06
    return float(size) * (1.0 + shift)


def _configure_xaxis(ax):
    sizes = sorted(set(MODEL_SIZE_B[m] for m in MODELS_SIZED))
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(int(s)) if s == int(s) else str(s) for s in sizes], fontsize=9)
    ax.set_xlabel("Parameters (B)", fontsize=11)
    margin = 0.15
    ax.set_xlim(sizes[0] * (10 ** -margin), sizes[-1] * (10 ** margin))


def _plot_by_groups_r(ax, stats: pd.DataFrame, ylabel: str):
    stats = stats.copy()
    stats["family"] = stats["model"].apply(_get_family)
    group_order = ["VLM", "VLM backbone decoder", "standalone LLM (think)", "standalone LLM"]
    group_alpha = {"standalone LLM (think)": 0.55}
    plotted_families = set()

    for grp in group_order:
        gdf = stats[stats["group"] == grp]
        if gdf.empty:
            continue
        color = GROUP_COLORS.get(grp, "#888888")
        hollow = GROUP_HOLLOW.get(grp, False)
        alpha = group_alpha.get(grp, 0.88)

        for fam, fdf in gdf.groupby("family"):
            mkr = FAMILY_MARKER.get(fam, "*")
            mfc = "none" if hollow else color
            mec = color if hollow else "white"
            mew = 1.4 if hollow else 0.5
            x = np.array([_x_with_shift(s, fam) for s in fdf["size"].values], dtype=float)
            ms_arr = [4 + 8 * (np.log10(max(s, 0.3)) / np.log10(32)) for s in fdf["size"].values]
            for xi, yi, ei, msi in zip(x, fdf["mean"].values, fdf["ci"].values, ms_arr):
                if not np.isfinite(yi):
                    continue
                ax.errorbar(
                    xi, yi, yerr=None if not np.isfinite(ei) else ei,
                    fmt=mkr, color=color, ms=msi, capsize=3, capthick=0.8,
                    elinewidth=0.8, alpha=alpha, markerfacecolor=mfc,
                    markeredgecolor=mec, markeredgewidth=mew, label=None,
                )
            if len(np.unique(x)) >= 2 and len(fdf) >= 2:
                coeffs = np.polyfit(np.log(x), fdf["mean"].values, 1)
                x_rng = np.logspace(np.log10(x.min()), np.log10(x.max()), 40)
                ax.plot(
                    x_rng, np.polyval(coeffs, np.log(x_rng)),
                    color=color, ls=":" if hollow else "--", lw=1.1,
                    alpha=0.35 if hollow else 0.45,
                )
            plotted_families.add(fam)

    legend_handles = []
    for fam in FAMILY_ORDER:
        if fam not in plotted_families:
            continue
        mkr = FAMILY_MARKER.get(fam, "*")
        legend_handles.append(
            mlines.Line2D([], [], marker=mkr, color="#555555", markerfacecolor="#555555",
                          markersize=7, linestyle="none", label=fam)
        )
    ax.axhline(0.0, color="#888888", lw=1.2, ls=":", zorder=0)
    ax.set_ylabel(ylabel, fontsize=11)
    _configure_xaxis(ax)
    ax.set_ylim(-0.05, 1.02)
    ax.legend(handles=legend_handles, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=True)


def model_stats_agreement_r(src_pair_cache: pd.DataFrame, col: str) -> pd.DataFrame:
    sub = src_pair_cache[src_pair_cache["pair_type"] == "HM"].copy()
    model_q = sub.groupby(["subject_2", "question_id"])[col].mean().reset_index()
    human_q = (
        src_pair_cache[src_pair_cache["pair_type"] == "HH"]
        .groupby("question_id")[col]
        .mean()
        .reset_index()
        .rename(columns={col: "human"})
    )
    rows = []
    for m in MODELS_SIZED:
        mq = model_q[model_q["subject_2"] == m][["question_id", col]].rename(columns={col: "model"})
        merged = human_q.merge(mq, on="question_id", how="inner").dropna()
        if len(merged) < 4 or merged["human"].nunique() < 2 or merged["model"].nunique() < 2:
            continue
        r = pearsonr(merged["human"].values, merged["model"].values).statistic
        rows.append({
            "model": m,
            "mean": float(r),
            "ci": _fisher_r_ci(float(r), len(merged)),
            "size": MODEL_SIZE_B[m],
            "group": MODEL_GROUP.get(m, "standalone LLM"),
            "n": len(merged),
        })
    return pd.DataFrame(rows)


def model_stats_accuracy_r(src_raw_acc: pd.DataFrame, human_df_full: pd.DataFrame, common_qids: list[int]) -> pd.DataFrame:
    model_q = src_raw_acc.groupby(["model", "question_id"])["accuracy"].mean().reset_index()
    human_q = (
        human_df_full[
            human_df_full["question_id"].isin(common_qids) &
            human_df_full["variant"].isin(VARIANTS)
        ]
        .groupby("question_id")["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "human"})
    )
    rows = []
    for m in MODELS_SIZED:
        mq = model_q[model_q["model"] == m][["question_id", "accuracy"]].rename(columns={"accuracy": "model"})
        merged = human_q.merge(mq, on="question_id", how="inner").dropna()
        if len(merged) < 4 or merged["human"].nunique() < 2 or merged["model"].nunique() < 2:
            continue
        r = pearsonr(merged["human"].values, merged["model"].values).statistic
        rows.append({
            "model": m,
            "mean": float(r),
            "ci": _fisher_r_ci(float(r), len(merged)),
            "size": MODEL_SIZE_B[m],
            "group": MODEL_GROUP.get(m, "standalone LLM"),
            "n": len(merged),
        })
    return pd.DataFrame(rows)


print(f"Loading human data (min_answers={MIN_ANSWERS_DEFAULT})…")
participants, common_qids, human_df_full, _ = load_human_subset(
    ROOT, min_answers=MIN_ANSWERS_DEFAULT, translate=False, verbose=True
)
print(f"  common_qids: {len(common_qids)}  |  participants: {len(participants)}")

pair_cache = load_cleaned_pair_cache(ROOT, condition="inst_blind", include_yesno=True, verbose=True)
pair_cache = pair_cache[pair_cache["question_id"].isin(common_qids) & pair_cache["variant"].isin(VARIANTS)].copy()
raw_acc = read_export(ROOT, "responses_model_inst_blind.csv")
raw_acc = raw_acc[raw_acc["question_id"].isin(common_qids) & raw_acc["variant"].isin(VARIANTS)].copy()

print("Generating raw r-panels…")
stats_acc_r = model_stats_accuracy_r(raw_acc, human_df_full, common_qids)
fig, ax = plt.subplots(figsize=(8, 5))
_plot_by_groups_r(ax, stats_acc_r, "Pearson r with human acc. (avg. C+B+A)")
plt.tight_layout()
save_dual(fig, "accuracy_r_groups_q113_h40_yesno_raw.png", "accuracy_r_groups_vABC_q113_h40_yesno_raw.png")
plt.close(fig)

stats_sbert_r = model_stats_agreement_r(pair_cache, "sbert_score")
fig, ax = plt.subplots(figsize=(8, 5))
_plot_by_groups_r(ax, stats_sbert_r, "Pearson r with human SBERT (avg. C+B+A)")
plt.tight_layout()
save_dual(fig, "agreement_r_groups_sbert_q113_h40_yesno_raw.png", "agreement_r_groups_sbert_vABC_q113_h40_yesno_raw.png")
plt.close(fig)

print("Generating abstention-filtered r-panels…")
pair_cache_filt = filter_abstained_pairs(pair_cache)
raw_acc_filt = raw_acc[~raw_acc["response"].fillna("").astype(str).apply(lambda x: is_abstained(classify(x, None)))].copy()

stats_acc_r_filt = model_stats_accuracy_r(raw_acc_filt, human_df_full, common_qids)
fig, ax = plt.subplots(figsize=(8, 5))
_plot_by_groups_r(ax, stats_acc_r_filt, "Pearson r with human acc. (avg. C+B+A, abstention filtered)")
plt.tight_layout()
save_dual(fig, "accuracy_r_groups_q113_h40_yesno_abstfiltered.png", "accuracy_r_groups_vABC_q113_h40_yesno_abstfiltered.png")
plt.close(fig)

stats_sbert_r_filt = model_stats_agreement_r(pair_cache_filt, "sbert_score")
fig, ax = plt.subplots(figsize=(8, 5))
_plot_by_groups_r(ax, stats_sbert_r_filt, "Pearson r with human SBERT (avg. C+B+A, abstention filtered)")
plt.tight_layout()
save_dual(fig, "agreement_r_groups_sbert_q113_h40_yesno_abstfiltered.png", "agreement_r_groups_sbert_vABC_q113_h40_yesno_abstfiltered.png")
plt.close(fig)

print(f"Done. Saved to: {OUT_DIR} and {LATEX_OUT}")
