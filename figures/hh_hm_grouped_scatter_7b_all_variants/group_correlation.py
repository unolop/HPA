"""
Within-group Pearson r: model HM scores vs human HH scores across (question, variant) pairs.

For each (group, model):
  - Take all (question_id, variant) pairs in that group
  - Compute mean HM SBERT per (q,v) across all human raters
  - Compute mean HH SBERT per (q,v) (= human difficulty reference)
  - Pearson r between these two vectors

Human range per group (LOOCV-style):
  - For each participant h: r(h's per-(q,v) HH scores, mean of all OTHER participants)
  - Shown as a boxplot across participants

Groups with < MIN_QV (q,v) pairs are merged into "Other (merged)".

Main story: which model class is indistinguishable from humans
            (i.e., falls within the human boxplot)?
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))
import plot_style  # noqa: F401

OUTDIR  = Path(__file__).resolve().parent
EXPORTS = ROOT / "analysis/session2/exports"

MIN_QV = 15   # (q,v) pairs below this threshold get merged into "Other"

MODEL_FAMILIES = [
    ("o", "InternVL-8B",      "InternVL-8B (LM)",   None,                  "InternVL"),
    ("X", "Qwen3-VL-8B",      "Qwen3-VL-8B (LM)",   "Qwen3-8B",            "Qwen3"),
    ("D", "LLaVA-1.5-7B",     "LLaVA-1.5 (LM)",     None,                  "LLaVA-1.5"),
    ("^", "LLaVA-Mistral",     "LLaVA-Mistral (LM)", "Mistral-7B",          "Mistral"),
    ("v", "LLaVA-Vicuna",      "LLaVA-Vicuna (LM)",  "Vicuna-7B",           "Vicuna"),
    ("P", None,                None,                  "Qwen3-8B (think)",    "Qwen3 (think)"),
    ("s", None,                None,                  "Qwen2.5-7B-Instruct", "Qwen2.5-Inst"),
]
MODEL_STYLE = {}
for marker, vlm, dec, llm, _ in MODEL_FAMILIES:
    for m in [vlm, dec, llm]:
        if m:
            MODEL_STYLE[m] = marker

MODEL_GROUPS = {
    "VLM":              ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone Decoder": ["InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)",
                         "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"],
    "Standalone LLM":   ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct",
                         "Vicuna-7B", "Mistral-7B"],
}
GROUP_COLORS = {
    "VLM":              "#C0392B",
    "Backbone Decoder": "#E67E22",
    "Standalone LLM":   "#27AE60",
}

OP_FULL = {
    "act": "Action", "attr": "Attribute", "cause": "Causality",
    "comp": "Comparison", "count": "Count", "exist": "Existence",
    "ident": "Identity", "know": "World Know.", "spat": "Spatial",
    "temp": "Temporal", "text": "Text Reading", "other": "Other",
}
ENT_FULL = {
    "animal": "Animal", "food": "Food", "object": "Object",
    "other": "Other", "person": "Person", "place": "Place",
    "product": "Product", "text": "Text", "vehicle": "Vehicle",
}
MERGED_LABEL = "Other\n(merged)"


def load_data():
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    hh = pc[pc["pair_type"] == "HH"].copy()
    hm = pc[pc["pair_type"] == "HM"].copy()
    keep = [m for fam in MODEL_GROUPS.values() for m in fam]
    hm = hm[hm["subject_2"].isin(keep)].copy()
    return hh, hm


def _count_qv(df: pd.DataFrame, dim: str, grp: str) -> int:
    sub = df[df[dim] == grp]
    return sub[["question_id", "variant"]].drop_duplicates().shape[0]


def compute_group_r(hh: pd.DataFrame, hm: pd.DataFrame, dim: str) -> dict:
    """
    Returns {group_label: {"model_rs": {model: r}, "human_rs": [r_per_participant]}}
    Small groups (< MIN_QV) are merged into MERGED_LABEL.
    """
    all_groups = hh[dim].unique()

    # Tag each row with merged label if group is too small
    qv_counts = {g: _count_qv(hh, dim, g) for g in all_groups}
    def grp_label(g):
        return g if qv_counts[g] >= MIN_QV else "_merged_"

    hh2 = hh.copy()
    hm2 = hm.copy()
    hh2["_grp"] = hh2[dim].map(grp_label)
    hm2["_grp"] = hm2[dim].map(grp_label)

    # Mean HH per (q,v) per effective group
    hh_mean_qv = (
        hh2.groupby(["question_id", "variant", "_grp"])["sbert_score"]
        .mean().reset_index().rename(columns={"sbert_score": "hh_mean"})
    )
    # Per-participant HH per (q,v) per effective group
    hh_part_qv = (
        hh2.groupby(["subject_1", "question_id", "variant", "_grp"])["sbert_score"]
        .mean().reset_index().rename(columns={"sbert_score": "hh_part"})
    )
    # Mean HM per (q,v) per model per effective group
    hm_mean_qv = (
        hm2.groupby(["subject_2", "question_id", "variant", "_grp"])["sbert_score"]
        .mean().reset_index().rename(columns={"sbert_score": "hm_mean", "subject_2": "model"})
    )

    effective_groups = hh2["_grp"].unique()
    results = {}

    for grp in effective_groups:
        label = MERGED_LABEL if grp == "_merged_" else grp
        grp_hh = hh_mean_qv[hh_mean_qv["_grp"] == grp]
        n_qv = grp_hh[["question_id", "variant"]].drop_duplicates().shape[0]
        ref_vec = grp_hh.set_index(["question_id", "variant"])["hh_mean"]

        model_rs = {}
        grp_hm = hm_mean_qv[hm_mean_qv["_grp"] == grp]
        for model, mdf in grp_hm.groupby("model"):
            vec = mdf.set_index(["question_id", "variant"])["hm_mean"]
            common = ref_vec.index.intersection(vec.index)
            if len(common) < MIN_QV:
                continue
            r, _ = stats.pearsonr(ref_vec[common], vec[common])
            model_rs[model] = float(r)

        # Human LOOCV r
        grp_part = hh_part_qv[hh_part_qv["_grp"] == grp]
        human_rs = []
        for part, pdf in grp_part.groupby("subject_1"):
            vec_h = pdf.set_index(["question_id", "variant"])["hh_part"]
            others = grp_part[grp_part["subject_1"] != part]
            ref_others = others.groupby(["question_id", "variant"])["hh_part"].mean()
            common = vec_h.index.intersection(ref_others.index)
            if len(common) < MIN_QV:
                continue
            r, _ = stats.pearsonr(vec_h[common], ref_others[common])
            human_rs.append(float(r))

        results[label] = {"model_rs": model_rs, "human_rs": human_rs, "n_qv": n_qv}

    return results


def plot_correlation_boxplot(results_op: dict, results_ent: dict):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True)

    for ax, results, full_names, title in [
        (axes[0], results_op, OP_FULL, "Operation"),
        (axes[1], results_ent, ENT_FULL, "Entity"),
    ]:
        # Sort groups by human median r; put merged last
        def sort_key(label):
            if label == MERGED_LABEL:
                return 99.0
            hrs = results[label]["human_rs"]
            return float(np.median(hrs)) if hrs else -99.0

        groups = sorted(results.keys(), key=sort_key)
        x = np.arange(len(groups))

        # ── Human boxplots ───────────────────────────────────────────────────
        box_data = []
        for grp in groups:
            hrs = results[grp]["human_rs"]
            box_data.append(hrs if hrs else [np.nan])

        bp = ax.boxplot(
            box_data, positions=x, widths=0.35,
            patch_artist=True, showfliers=True,
            flierprops=dict(marker=".", markersize=3, color="#999999", alpha=0.5),
            medianprops=dict(color="#333333", lw=1.8),
            boxprops=dict(facecolor="#dddddd", edgecolor="#888888", linewidth=0.9, alpha=0.45),
            whiskerprops=dict(color="#888888", lw=0.9),
            capprops=dict(color="#888888", lw=0.9),
            zorder=1,
        )

        # ── Model dots ───────────────────────────────────────────────────────
        n_cls = len(MODEL_GROUPS)
        offsets = np.linspace(-0.28, 0.28, n_cls)
        for ci, (cls, models) in enumerate(MODEL_GROUPS.items()):
            color = GROUP_COLORS[cls]
            for i, grp in enumerate(groups):
                vals = [results[grp]["model_rs"][m]
                        for m in models if m in results[grp]["model_rs"]]
                if not vals:
                    continue
                mean_r = float(np.mean(vals))
                ax.scatter(i + offsets[ci], mean_r, s=45, color=color,
                           marker="o", edgecolors="white", linewidth=0.6,
                           zorder=4, alpha=0.95)

        ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [full_names.get(g, g) if g != MERGED_LABEL else MERGED_LABEL for g in groups],
            fontsize=9, rotation=20, ha="right",
        )
        ax.set_ylabel("Pearson $r$", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(-0.7, len(groups) - 0.3)
        ax.grid(axis="y", lw=0.5, alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[cls], alpha=0.85, label=cls)
        for cls in MODEL_GROUPS
    ] + [
        plt.Rectangle((0, 0), 1, 1, facecolor="#dddddd", edgecolor="#888888",
                       linewidth=0.9, label="Human range (boxplot)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(handles), fontsize=8.5, frameon=False,
               handletextpad=0.4, columnspacing=1.0)

    out = OUTDIR / "group_correlation_r.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    print("Loading data...")
    hh, hm = load_data()
    print("Computing op correlations...")
    results_op  = compute_group_r(hh, hm, "op")
    print("Computing ent correlations...")
    results_ent = compute_group_r(hh, hm, "ent")
    print("Plotting...")
    plot_correlation_boxplot(results_op, results_ent)
