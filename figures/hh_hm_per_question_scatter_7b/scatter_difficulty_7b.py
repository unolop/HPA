"""Scatter: human difficulty (HH mean) vs model difficulty (HM mean) per (question, variant).
Each point = one (question_id, variant). Spearman ρ matches the table values exactly.
Free-text questions only (answer_type == 'other'), inst_blind condition, 7/8B models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))
import plot_style  # noqa: F401

EXPORTS = ROOT / "analysis/session2/exports"
OUTDIR  = Path(__file__).resolve().parent

# Free-text question IDs (answer_type == "other")
import json
ANN_PATH = ROOT / "dataset/vqa/v2_mscoco_val2014_annotations.json"
with open(ANN_PATH) as f:
    anns = json.load(f)["annotations"]
ft_qids = {a["question_id"] for a in anns if a.get("answer_type") == "other"}

pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
hh = pc[pc["pair_type"] == "HH"]
hm = pc[pc["pair_type"] == "HM"]

# Filter to free-text questions
hh = hh[hh["question_id"].astype(int).isin(ft_qids)]
hm = hm[hm["question_id"].astype(int).isin(ft_qids)]

# Human difficulty reference: per (question_id, variant) mean HH SBERT
ref = hh.groupby(["question_id", "variant"])["sbert_score"].mean().rename("hh_mean")

models_7b = [
    # VLM
    "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
    # Backbone decoder
    "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    # Standalone LLM
    "Qwen3-8B", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B",
]
available = hm["subject_2"].unique()
models_7b = [m for m in models_7b if m in available]

variant_colors = {"C": "#1f77b4", "B": "#ff7f0e", "A": "#2ca02c"}
variant_labels = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}

ncols = 4
nrows = int(np.ceil(len(models_7b) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.6 * nrows), squeeze=False)

for idx, model in enumerate(models_7b):
    ax = axes[idx // ncols][idx % ncols]

    # Per-(question,variant) mean HM SBERT for this model
    mm = (
        hm[hm["subject_2"] == model]
        .groupby(["question_id", "variant"])["sbert_score"]
        .mean()
        .rename("hm_mean")
    )
    df = mm.to_frame().join(ref, how="inner").reset_index()

    for v in ["C", "B", "A"]:
        vs = df[df["variant"] == v]
        ax.scatter(
            vs["hh_mean"], vs["hm_mean"],
            c=variant_colors[v], label=variant_labels[v],
            alpha=0.55, s=18, edgecolors="none",
        )

    # Spearman ρ overall and per variant
    rho_all, p_all = stats.spearmanr(df["hh_mean"], df["hm_mean"])
    corr_text = f"ρ={rho_all:.3f} (p={p_all:.1e})\n"
    for v, short in [("C", "Orig"), ("B", "Weak"), ("A", "Pron")]:
        vs = df[df["variant"] == v]
        if len(vs) > 2:
            rho_v, _ = stats.spearmanr(vs["hh_mean"], vs["hm_mean"])
            corr_text += f"{short}: ρ={rho_v:.3f}\n"

    ax.set_title(model, fontsize=10, fontweight="bold")
    ax.text(
        0.02, 0.98, corr_text.strip(), transform=ax.transAxes,
        fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88),
    )

    lims = [df[["hh_mean", "hm_mean"]].min().min() - 0.02,
            df[["hh_mean", "hm_mean"]].max().max() + 0.02]
    ax.plot(lims, lims, "--", color="gray", alpha=0.4, lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    if idx % ncols == 0:
        ax.set_ylabel("Model difficulty (HM SBERT mean)")
    if idx // ncols == nrows - 1:
        ax.set_xlabel("Human difficulty (HH SBERT mean)")

for idx in range(len(models_7b), nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

row_labels = ["VLM", "Backbone Decoder", "Standalone LLM"]
for row_idx, label in enumerate(row_labels):
    if row_idx < nrows:
        axes[row_idx][0].annotate(
            label, xy=(-0.38, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Human difficulty vs. Model difficulty per question (free-text, inst_blind, 7/8B)", fontsize=11, y=1.04)
plt.tight_layout()
fig.subplots_adjust(left=0.09)

out = OUTDIR / "scatter_difficulty_7b.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")

print(f"\n{'Model':<25} {'ρ (all)':>9} {'p':>10} {'Orig ρ':>8} {'Weak ρ':>8} {'Pron ρ':>8}")
for model in models_7b:
    mm = (
        hm[hm["subject_2"] == model]
        .groupby(["question_id", "variant"])["sbert_score"]
        .mean()
        .rename("hm_mean")
    )
    df = mm.to_frame().join(ref, how="inner").reset_index()
    rho_all, p_all = stats.spearmanr(df["hh_mean"], df["hm_mean"])
    rhos = {}
    for v in ["C", "B", "A"]:
        vs = df[df["variant"] == v]
        if len(vs) > 2:
            rhos[v], _ = stats.spearmanr(vs["hh_mean"], vs["hm_mean"])
    print(f"{model:<25} {rho_all:>9.3f} {p_all:>10.1e} {rhos.get('C', float('nan')):>8.3f} {rhos.get('B', float('nan')):>8.3f} {rhos.get('A', float('nan')):>8.3f}")
