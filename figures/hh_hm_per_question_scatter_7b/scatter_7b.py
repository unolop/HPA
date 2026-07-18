"""Scatter: HH SBERT vs HM SBERT per question, colored by variant, one panel per 7B model."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "figures"))
import plot_style  # noqa: F401 — sets 10pt Times New Roman

EXPORTS = ROOT / "analysis/session2/exports"
OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)

pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")

# --- HH aggregate ---
hh = pc[pc["pair_type"] == "HH"]
hh_agg = (
    hh.groupby(["question_id", "variant"])
    .agg(hh_sbert=("sbert_score", "mean"), op=("op", "first"))
    .reset_index()
)

# --- HM aggregate ---
hm = pc[pc["pair_type"] == "HM"]
hm_agg = (
    hm.groupby(["question_id", "variant", "subject_2"])
    .agg(hm_sbert=("sbert_score", "mean"))
    .reset_index()
)

# 7B-scale models — grouped by row: VLM, Backbone decoder, Standalone LLM
models_7b = [
    # VLM
    "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
    # Backbone decoder
    "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    # Standalone LLM
    "Qwen3-8B", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B",
]
models_7b = [m for m in models_7b if m in hm_agg["subject_2"].unique()]

merged = hm_agg[hm_agg["subject_2"].isin(models_7b)].merge(
    hh_agg[["question_id", "variant", "hh_sbert"]],
    on=["question_id", "variant"],
)

variant_colors = {"C": "#1f77b4", "B": "#ff7f0e", "A": "#2ca02c"}
variant_labels = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}

ncols = 4
nrows = int(np.ceil(len(models_7b) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.6 * nrows), squeeze=False)

for idx, model in enumerate(models_7b):
    ax = axes[idx // ncols][idx % ncols]
    sub = merged[merged["subject_2"] == model]

    for v in ["C", "B", "A"]:
        vs = sub[sub["variant"] == v]
        ax.scatter(
            vs["hh_sbert"], vs["hm_sbert"],
            c=variant_colors[v], label=variant_labels[v],
            alpha=0.55, s=18, edgecolors="none",
        )

    # Correlations
    r_all, p_all = stats.pearsonr(sub["hh_sbert"], sub["hm_sbert"])
    corr_text = f"r={r_all:.3f} (p={p_all:.1e})\n"
    for v, short in [("C", "Orig"), ("B", "Weak"), ("A", "Pron")]:
        vs = sub[sub["variant"] == v]
        if len(vs) > 2:
            r_v, _ = stats.pearsonr(vs["hh_sbert"], vs["hm_sbert"])
            corr_text += f"{short}: r={r_v:.3f}\n"

    ax.set_title(model, fontsize=10, fontweight="bold")
    ax.text(
        0.02, 0.98, corr_text.strip(), transform=ax.transAxes,
        fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88),
    )

    ax.plot([0.1, 0.95], [0.1, 0.95], "--", color="gray", alpha=0.4, lw=1)
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0.1, 0.95)

    if idx % ncols == 0:
        ax.set_ylabel("HM SBERT")
    if idx // ncols == nrows - 1:
        ax.set_xlabel("HH SBERT")

for idx in range(len(models_7b), nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

# Row labels
row_labels = ["VLM", "Backbone Decoder", "Standalone LLM"]
for row_idx, label in enumerate(row_labels):
    if row_idx < nrows:
        axes[row_idx][0].annotate(
            label, xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Human\u2013Human vs Human\u2013Model SBERT (7B scale)", fontsize=11, y=1.04)
plt.tight_layout()
fig.subplots_adjust(left=0.09)

out = OUTDIR / "scatter_7b.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")

# --- Print correlation summary ---
print(f"\n{'Model':<25} {'All r':>7} {'All p':>10} {'Orig r':>8} {'Weak r':>8} {'Pron r':>8}")
for model in models_7b:
    sub = merged[merged["subject_2"] == model]
    r_all, p_all = stats.pearsonr(sub["hh_sbert"], sub["hm_sbert"])
    rs = {}
    for v in ["C", "B", "A"]:
        vs = sub[sub["variant"] == v]
        if len(vs) > 2:
            rs[v], _ = stats.pearsonr(vs["hh_sbert"], vs["hm_sbert"])
    print(f"{model:<25} {r_all:>7.3f} {p_all:>10.1e} {rs.get('C', float('nan')):>8.3f} {rs.get('B', float('nan')):>8.3f} {rs.get('A', float('nan')):>8.3f}")
