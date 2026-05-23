"""
Export SBERT-versus-accuracy-gap figures for the fair human-study subset.

This script reproduces the notebook-15 analysis in a reusable form and saves
paper-ready figures under:

  figures/sbert_accuracy_corr/

It focuses on the inst_blind condition by default and joins:
  - model accuracy (vs human average accuracy)
  - human-model semantic agreement (SBERT) from pair_cache

Outputs
-------
Figures:
  {condition}_v{variant}_questiongroup_by_group_q{N}_h{H}.png
  {condition}_v{variant}_questiongroup_pooled_q{N}_h{H}.png
  {condition}_v{variant}_modellevel_by_group_q{N}_h{H}.png
  {condition}_v{variant}_modellevel_by_family_q{N}_h{H}.png

Tables:
  {condition}_v{variant}_questiongroup_corr_q{N}_h{H}.csv
  {condition}_v{variant}_modellevel_corr_q{N}_h{H}.csv
  {condition}_v{variant}_modellevel_summary_q{N}_h{H}.csv

Run from repo root:
  conda run -n zero python figures/sbert_accuracy_correlation.py
  conda run -n zero python figures/sbert_accuracy_correlation.py --include_yesno
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from config import MODEL_GROUP, MODEL_LABEL_SHORT as LABEL_MAP, MIN_ANSWERS_DEFAULT
from helpers import (
    get_exports_dir,
    load_human_subset,
    load_pair_cache,
    read_response_exports,
)
from utils.constants import (
    GROUP_COLORS,
    GROUP_MARKER,
    GROUP_ORDER,
    MODEL_FAMILY,
    MODEL_FAMILY_COLORS,
)


parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="C", choices=["A", "B", "C"])
parser.add_argument(
    "--condition",
    default="inst_blind",
    choices=["blind", "inst_blind"],
    help="Model response condition to analyze.",
)
parser.add_argument(
    "--include_yesno",
    action="store_true",
    help="Extend the HM pair cache with yes/no questions and use the q113 human-study subset.",
)
parser.add_argument("--min_answers", type=int, default=MIN_ANSWERS_DEFAULT)
args = parser.parse_args()


EXPORTS = get_exports_dir(ROOT)
OUT_DIR = ROOT / "figures/sbert_accuracy_corr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  [sbert_accuracy_corr] {name}")


def _corr_row(df: pd.DataFrame, x_col: str, y_col: str, label: str) -> dict[str, object]:
    sub = df[[x_col, y_col]].dropna()
    n = len(sub)
    if n < 3:
        return {
            "slice": label,
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    x = sub[x_col].to_numpy()
    y = sub[y_col].to_numpy()
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "slice": label,
        "n": n,
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_rho": sr,
        "spearman_p": sp,
    }


def _annotate_subset(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    label_col: str,
    color_col: str,
    max_labels: int = 8,
) -> None:
    if df.empty:
        return
    ranked = (
        df.assign(priority=df[y_col].abs() + df[x_col])
        .sort_values("priority", ascending=False)
        .head(max_labels)
    )
    for _, row in ranked.iterrows():
        ax.annotate(
            row[label_col],
            (row[x_col], row[y_col]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
            color=row[color_col],
        )


print(f"\nLoading human subset (min_answers={args.min_answers})…")
participants, common_qids, human_df, _ = load_human_subset(
    ROOT, min_answers=args.min_answers, translate=False, verbose=True
)
n_humans = len(participants)

exports = read_response_exports(ROOT, subset_qids=common_qids, variant=args.variant)
model_key = "model_inst_blind" if args.condition == "inst_blind" else "model_blind"
model_df = exports[model_key].copy()

pair_cache = load_pair_cache(ROOT, include_yesno=args.include_yesno, verbose=True)
pair_cache = pair_cache[
    (pair_cache["variant"] == args.variant)
    & (pair_cache["question_id"].isin(common_qids))
    & (pair_cache["pair_type"] == "HM")
].copy()

if args.condition == "blind" and "condition_2" in pair_cache.columns:
    pair_cache = pair_cache[pair_cache["condition_2"] == "blind"].copy()
elif args.condition == "inst_blind" and "condition_2" in pair_cache.columns:
    pair_cache = pair_cache[pair_cache["condition_2"] == "inst_blind"].copy()

model_df["acc_gap"] = model_df["accuracy"] - model_df["human_avg_acc"]

hm_sbert = (
    pair_cache.groupby(["question_id", "subject_2", "subject_group_2"])
    .agg(
        mean_sbert=("sbert_score", "mean"),
        mean_exact=("exact_score", "mean"),
        n_pairs=("sbert_score", "count"),
    )
    .reset_index()
    .rename(columns={"subject_2": "model", "subject_group_2": "model_group"})
)

joined = model_df.merge(
    hm_sbert[["question_id", "model", "model_group", "mean_sbert", "mean_exact"]],
    on=["question_id", "model"],
    how="inner",
)
joined["model_family"] = joined["model"].map(MODEL_FAMILY).fillna("Unknown")
joined["model_group"] = joined["model"].map(MODEL_GROUP).fillna(
    joined.get("model_group_x", joined.get("model_group_y"))
)
joined["model_label"] = joined["model"].map(LABEL_MAP).fillna(joined["model"])

q_group = (
    joined.groupby(
        ["question_id", "model_group", "ent", "op", "question_en", "gt", "human_avg_acc"],
        dropna=False,
    )
    .agg(
        mean_model_acc=("accuracy", "mean"),
        mean_acc_gap=("acc_gap", "mean"),
        mean_sbert=("mean_sbert", "mean"),
    )
    .reset_index()
)

model_summary = (
    joined.groupby(["model", "model_label", "model_group", "model_family"], dropna=False)
    .agg(
        mean_model_acc=("accuracy", "mean"),
        mean_human_acc=("human_avg_acc", "mean"),
        mean_acc_gap=("acc_gap", "mean"),
        mean_sbert=("mean_sbert", "mean"),
        n_questions=("question_id", "nunique"),
    )
    .reset_index()
)

n_questions = joined["question_id"].nunique()
suffix = f"_q{n_questions}_h{n_humans}"
if args.include_yesno:
    suffix += "_yesno"
stem = f"{args.condition}_v{args.variant}"

print(f"  joined rows: {len(joined)}")
print(f"  question-group rows: {len(q_group)}")
print(f"  model rows: {len(model_summary)}")

# Correlation tables
q_corr_rows = [_corr_row(q_group, "mean_sbert", "mean_acc_gap", "pooled")]
for grp in GROUP_ORDER:
    sub = q_group[q_group["model_group"] == grp]
    if not sub.empty:
        q_corr_rows.append(_corr_row(sub, "mean_sbert", "mean_acc_gap", grp))
q_corr = pd.DataFrame(q_corr_rows)
q_corr.to_csv(OUT_DIR / f"{stem}_questiongroup_corr{suffix}.csv", index=False)

m_corr_rows = [_corr_row(model_summary, "mean_sbert", "mean_acc_gap", "pooled")]
for grp in GROUP_ORDER:
    sub = model_summary[model_summary["model_group"] == grp]
    if not sub.empty:
        m_corr_rows.append(_corr_row(sub, "mean_sbert", "mean_acc_gap", grp))
m_corr = pd.DataFrame(m_corr_rows)
m_corr.to_csv(OUT_DIR / f"{stem}_modellevel_corr{suffix}.csv", index=False)
model_summary.sort_values(["model_group", "mean_acc_gap"], ascending=[True, False]).to_csv(
    OUT_DIR / f"{stem}_modellevel_summary{suffix}.csv", index=False
)

# Figure 1: question-group small multiples by group
fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(16, 4.2), sharey=True)
for ax, grp in zip(axes, GROUP_ORDER):
    sub = q_group[q_group["model_group"] == grp]
    color = GROUP_COLORS.get(grp, "#666666")
    ax.scatter(
        sub["mean_sbert"],
        sub["mean_acc_gap"],
        s=18,
        alpha=0.45,
        color=color,
        edgecolors="none",
    )
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.set_title(grp.replace("standalone ", "SA-"))
    ax.set_xlabel("Mean SBERT")
    if not sub.empty and len(sub) >= 3:
        corr = _corr_row(sub, "mean_sbert", "mean_acc_gap", grp)
        ax.text(
            0.03,
            0.96,
            f"r={corr['pearson_r']:.2f}\nrho={corr['spearman_rho']:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color=color,
        )
    ax.grid(True, alpha=0.15)
axes[0].set_ylabel("Accuracy gap (model - human)")
fig.suptitle(f"{args.condition} v{args.variant}: question-level SBERT vs accuracy gap by model group")
plt.tight_layout()
_save(fig, f"{stem}_questiongroup_by_group{suffix}.png")

# Figure 2: pooled question-group scatter
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for grp in GROUP_ORDER:
    sub = q_group[q_group["model_group"] == grp]
    if sub.empty:
        continue
    color = GROUP_COLORS.get(grp, "#666666")
    marker = GROUP_MARKER.get(grp, "o")
    ax.scatter(
        sub["mean_sbert"],
        sub["mean_acc_gap"],
        s=22,
        alpha=0.38,
        color=color,
        marker=marker,
        edgecolors="none",
        label=grp,
    )
ax.axhline(0, color="#777777", lw=1.0, ls="--")
pooled_corr = _corr_row(q_group, "mean_sbert", "mean_acc_gap", "pooled")
ax.set_title(
    f"{args.condition} v{args.variant}: pooled question-level SBERT vs accuracy gap\n"
    f"Pearson r={pooled_corr['pearson_r']:.2f}, Spearman rho={pooled_corr['spearman_rho']:.2f}"
)
ax.set_xlabel("Mean SBERT (model ↔ human)")
ax.set_ylabel("Accuracy gap (model - human)")
ax.legend(frameon=True)
ax.grid(True, alpha=0.15)
plt.tight_layout()
_save(fig, f"{stem}_questiongroup_pooled{suffix}.png")

# Figure 3: model-level by group
fig, ax = plt.subplots(figsize=(8.0, 5.8))
for grp in GROUP_ORDER:
    sub = model_summary[model_summary["model_group"] == grp]
    if sub.empty:
        continue
    color = GROUP_COLORS.get(grp, "#666666")
    marker = GROUP_MARKER.get(grp, "o")
    ax.scatter(
        sub["mean_sbert"],
        sub["mean_acc_gap"],
        s=90,
        color=color,
        marker=marker,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.95,
        label=grp,
    )
    sub = sub.assign(label_color=color)
    _annotate_subset(
        ax,
        sub,
        x_col="mean_sbert",
        y_col="mean_acc_gap",
        label_col="model_label",
        color_col="label_color",
        max_labels=3,
    )
ax.axhline(0, color="#777777", lw=1.0, ls="--")
model_corr = _corr_row(model_summary, "mean_sbert", "mean_acc_gap", "pooled")
ax.set_title(
    f"{args.condition} v{args.variant}: model-level SBERT vs accuracy gap\n"
    f"Pearson r={model_corr['pearson_r']:.2f}, Spearman rho={model_corr['spearman_rho']:.2f}"
)
ax.set_xlabel("Mean SBERT (model ↔ human)")
ax.set_ylabel("Mean accuracy gap (model - human)")
ax.legend(frameon=True)
ax.grid(True, alpha=0.15)
plt.tight_layout()
_save(fig, f"{stem}_modellevel_by_group{suffix}.png")

# Figure 4: model-level by family
families = [f for f in model_summary["model_family"].dropna().unique() if f != "Unknown"]
fig, ax = plt.subplots(figsize=(8.0, 5.8))
for fam in sorted(families):
    sub = model_summary[model_summary["model_family"] == fam]
    if sub.empty:
        continue
    color = MODEL_FAMILY_COLORS.get(fam, "#666666")
    for _, row in sub.iterrows():
        marker = GROUP_MARKER.get(row["model_group"], "o")
        ax.scatter(
            row["mean_sbert"],
            row["mean_acc_gap"],
            s=90,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.7,
            alpha=0.95,
        )
    fam_df = sub.assign(label_color=color)
    _annotate_subset(
        ax,
        fam_df,
        x_col="mean_sbert",
        y_col="mean_acc_gap",
        label_col="model_label",
        color_col="label_color",
        max_labels=2,
    )

family_handles = [
    mlines.Line2D([], [], color=MODEL_FAMILY_COLORS.get(fam, "#666666"), marker="o", ls="none", ms=7, label=fam)
    for fam in sorted(families)
]
group_handles = [
    mlines.Line2D([], [], color="#555555", marker=GROUP_MARKER.get(grp, "o"), ls="none", ms=7, label=grp)
    for grp in GROUP_ORDER
]
ax.axhline(0, color="#777777", lw=1.0, ls="--")
ax.set_title(f"{args.condition} v{args.variant}: model-level SBERT vs accuracy gap by family")
ax.set_xlabel("Mean SBERT (model ↔ human)")
ax.set_ylabel("Mean accuracy gap (model - human)")
ax.legend(handles=family_handles + group_handles, frameon=True, loc="best", ncol=2)
ax.grid(True, alpha=0.15)
plt.tight_layout()
_save(fig, f"{stem}_modellevel_by_family{suffix}.png")

print("\nSaved figures to:", OUT_DIR)
