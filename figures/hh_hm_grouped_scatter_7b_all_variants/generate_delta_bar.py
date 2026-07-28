"""
2×2 grouped bar chart: rows = op/ent, columns = SBERT Δ / Accuracy Δ.
Bars = architecture class (VLM / Backbone / SA-LLM), error bars = 95% CI.
Individual model Δ overlaid as family markers.
Human HH Δ shown as black diamond per group.
Groups ordered by human SBERT Δ ascending (consistent across columns).

Saved as: 7b_sbert_delta_bar_abstfiltered.png
"""
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))
import plot_style  # noqa: F401
from helpers import filter_abstained_pairs

OUTDIR    = Path(__file__).resolve().parent
LATEX_DIR = ROOT / "latex/AAAI2026/LaTeX/figures/hh_hm_grouped_scatter_7b_all_variants"
LATEX_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS   = ROOT / "analysis/session2/exports"

OP_MERGE  = {"cause": "other", "comp": "other", "temp": "other",
             "text": "other", "know": "other"}
ENT_MERGE = {"place": "other", "product": "other",
             "vehicle": "other", "text": "other"}

OP_LABELS = {
    "act": "Action", "attr": "Attribute", "count": "Count",
    "exist": "Existence", "ident": "Identity", "spat": "Spatial",
    "other": "Other",
}
ENT_LABELS = {
    "animal": "Animal", "food": "Food", "object": "Object",
    "other": "Other", "person": "Person",
}

MODEL_GROUPS = {
    "VLM":      ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone": ["InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)",
                 "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"],
    "SA-LLM":   ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct",
                 "Vicuna-7B", "Mistral-7B"],
}
CLS_ORDER  = ["VLM", "Backbone", "SA-LLM"]
CLS_COLORS = {"VLM": "#4878CF", "Backbone": "#6ACC65", "SA-LLM": "#D65F5F"}

MODEL_FAMILIES = [
    ("o", "InternVL-8B",       "InternVL-8B (LM)",    None,                  "InternVL"),
    ("X", "Qwen3-VL-8B",       "Qwen3-VL-8B (LM)",    "Qwen3-8B",            "Qwen3"),
    ("D", "LLaVA-1.5-7B",      "LLaVA-1.5 (LM)",      None,                  "LLaVA-1.5"),
    ("^", "LLaVA-Mistral",      "LLaVA-Mistral (LM)",  "Mistral-7B",          "Mistral"),
    ("v", "LLaVA-Vicuna",       "LLaVA-Vicuna (LM)",   "Vicuna-7B",           "Vicuna"),
    ("P", None,                 None,                   "Qwen3-8B (think)",    "Qwen3 (think)"),
    ("s", None,                 None,                   "Qwen2.5-7B-Instruct", "Qwen2.5-Instruct"),
]
MODEL_STYLE = {}
for marker, vlm, dec, llm, _ in MODEL_FAMILIES:
    if vlm: MODEL_STYLE[vlm] = marker
    if dec: MODEL_STYLE[dec] = marker
    if llm: MODEL_STYLE[llm] = marker


def load_data():
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)
    pc = pc[~pc["subject_2"].str.contains("32B|32b", na=False)]
    pc["op"]  = pc["op"].replace(OP_MERGE)
    pc["ent"] = pc["ent"].replace(ENT_MERGE)
    cls_map = {m: g for g, ms in MODEL_GROUPS.items() for m in ms}
    hh = pc[pc["pair_type"] == "HH"].copy()
    hm = pc[(pc["pair_type"] == "HM") & pc["subject_2"].isin(cls_map)].copy()
    hm["cls"] = hm["subject_2"].map(cls_map)
    return hh, hm


def delta_stats(df, group_col, score_col, subject_col=None):
    """Return per-group Δ(C→A): mean and 95% CI."""
    grp_cols = [group_col] if subject_col is None else [group_col, subject_col]
    pivot = (df.groupby(grp_cols + ["variant"])[score_col]
               .mean().unstack("variant"))
    if "C" not in pivot.columns or "A" not in pivot.columns:
        return pd.DataFrame()
    delta = (pivot["A"] - pivot["C"]).dropna()
    if subject_col:
        out = delta.groupby(level=group_col).agg(
            mean=("mean"),
            ci=lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
        )
    else:
        q_pivot = (df.groupby(["question_id", group_col, "variant"])[score_col]
                    .mean().unstack("variant"))
        q_delta = (q_pivot["A"] - q_pivot["C"]).dropna()
        out = q_delta.groupby(level=group_col).agg(
            mean=("mean"),
            ci=lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
        )
    return out


def per_model_delta(df, group_col, score_col):
    """Return per-model per-group Δ(C→A)."""
    pivot = (df.groupby(["subject_2", group_col, "variant"])[score_col]
               .mean().unstack("variant"))
    if "C" not in pivot.columns or "A" not in pivot.columns:
        return pd.Series(dtype=float)
    return (pivot["A"] - pivot["C"]).dropna()


def draw_bars(ax, groups_ordered, labels_map, hh_stats, hm_stats_by_cls, hm_model_delta):
    n      = len(groups_ordered)
    n_cls  = len(CLS_ORDER)
    width  = 0.20
    offsets = np.linspace(-(n_cls - 1) / 2 * width, (n_cls - 1) / 2 * width, n_cls)
    xs = np.arange(n)

    for ci_idx, cls in enumerate(CLS_ORDER):
        stats = hm_stats_by_cls[cls]
        means = [stats.loc[g, "mean"] if g in stats.index else np.nan for g in groups_ordered]
        cis   = [stats.loc[g, "ci"]   if g in stats.index else 0.0    for g in groups_ordered]
        ax.bar(xs + offsets[ci_idx], means, width,
               color=CLS_COLORS[cls], alpha=0.75, zorder=3, linewidth=0)
        ax.errorbar(xs + offsets[ci_idx], means, yerr=cis,
                    fmt="none", ecolor="#333", elinewidth=0.8, capsize=2.5, zorder=4)

        model_delta = hm_model_delta[cls]
        for gi, g in enumerate(groups_ordered):
            grp_vals = (model_delta.xs(g, level=0)
                        if g in model_delta.index.get_level_values(0)
                        else pd.Series(dtype=float))
            for model_name, val in grp_vals.items():
                mkr = MODEL_STYLE.get(model_name, "o")
                ax.scatter(xs[gi] + offsets[ci_idx], val, s=18,
                           marker=mkr, color="white",
                           edgecolors=CLS_COLORS[cls], linewidth=0.8, zorder=5)

    hh_means = [hh_stats.loc[g, "mean"] if g in hh_stats.index else np.nan for g in groups_ordered]
    hh_cis   = [hh_stats.loc[g, "ci"]   if g in hh_stats.index else 0.0    for g in groups_ordered]
    ax.scatter(xs, hh_means, marker="D", s=28, color="black", zorder=6)
    ax.errorbar(xs, hh_means, yerr=hh_cis,
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, zorder=5)

    ax.axhline(0, color="#888", lw=0.8, ls="-", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels_map.get(g, g) for g in groups_ordered],
                       rotation=35, ha="right", fontsize=7.5)
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_delta_bar():
    hh, hm = load_data()

    fig, axes = plt.subplots(
        2, 2, figsize=(9.0, 5.6),
        gridspec_kw={"hspace": 0.52, "wspace": 0.28,
                     "left": 0.07, "right": 0.98,
                     "top": 0.80, "bottom": 0.10},
    )

    col_titles = ["SBERT $\\Delta$(Orig.$\\to$Pron.)", "Accuracy $\\Delta$(Orig.$\\to$Pron.)"]
    score_cols = ["sbert_score", "exact_score"]
    row_dims   = [("op", OP_LABELS, "By operation type"),
                  ("ent", ENT_LABELS, "By entity type")]

    # Pre-compute group ordering from SBERT HH Δ (consistent across columns)
    sbert_orders = {}
    for dim, labels_map, _ in row_dims:
        hh_stats = delta_stats(hh.dropna(subset=[dim]), dim, "sbert_score")
        sbert_orders[dim] = (hh_stats["mean"]
                             .reindex([k for k in hh_stats.index if k in labels_map])
                             .sort_values()
                             .index.tolist())

    for row_idx, (dim, labels_map, row_title) in enumerate(row_dims):
        groups_ordered = sbert_orders[dim]

        for col_idx, (score_col, col_title) in enumerate(zip(score_cols, col_titles)):
            ax = axes[row_idx, col_idx]

            hh_stats = delta_stats(hh.dropna(subset=[dim]), dim, score_col)
            hm_stats_by_cls = {
                cls: delta_stats(hm[hm["cls"] == cls].dropna(subset=[dim]), dim, score_col, "subject_2")
                for cls in CLS_ORDER
            }
            hm_model_delta = {
                cls: per_model_delta(hm[hm["cls"] == cls].dropna(subset=[dim]), dim, score_col)
                for cls in CLS_ORDER
            }

            draw_bars(ax, groups_ordered, labels_map, hh_stats, hm_stats_by_cls, hm_model_delta)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=9.0, fontweight="bold", pad=4)
            if col_idx == 0:
                ax.set_ylabel(row_title, fontsize=8.5)

    # ── Top legends ──────────────────────────────────────────────────────────
    cls_handles = [mpatches.Patch(color=CLS_COLORS[c], alpha=0.75, label=c) for c in CLS_ORDER]
    cls_handles.append(mlines.Line2D([0], [0], marker="D", color="black",
                                     markerfacecolor="black", linestyle="None",
                                     markersize=5, label="Human (HH)"))
    fig.legend(handles=cls_handles, loc="lower center",
               bbox_to_anchor=(0.50, 0.88), ncol=4,
               fontsize=8.0, frameon=False,
               handletextpad=0.4, columnspacing=0.8)

    family_handles = sorted([
        mlines.Line2D([0], [0], marker=mk, color="#555", markerfacecolor="white",
                      markeredgecolor="#555", markeredgewidth=0.8,
                      linestyle="None", markersize=5, label=lbl)
        for mk, _, _, _, lbl in MODEL_FAMILIES
    ], key=lambda h: h.get_label().lower())
    fig.legend(handles=family_handles, loc="lower center",
               bbox_to_anchor=(0.50, 0.82), ncol=7,
               fontsize=7.5, frameon=False,
               handletextpad=0.3, columnspacing=0.6)

    out = OUTDIR / "7b_sbert_delta_bar_abstfiltered.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_DIR / "7b_sbert_delta_bar_abstfiltered.png")
    plt.close()
    print(f"Saved: {out}")


plot_delta_bar()
