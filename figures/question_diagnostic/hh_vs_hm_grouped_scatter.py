"""Scatter: HH vs HM SBERT at group level.
Layout: 2 rows (Operation / Entity) × 3 columns (VLM / Backbone Decoder / Standalone LLM).
Colors = group categories (op or ent types), markers = model families.
VLM shown as filled markers, Decoder as open markers (same shape per family).
Single unified model legend."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_style  # noqa: F401 — sets 10pt Times New Roman

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "analysis/session2/exports"
OUTDIR = Path(__file__).resolve().parent

# Model families: marker shape shared across VLM, decoder, standalone
# (marker, VLM, Decoder, Standalone LLM, display label)
MODEL_FAMILIES = [
    ("o", "InternVL-8B",    "InternVL-8B (LM)", None,                  "InternVL"),
    ("s", "Qwen3-VL-8B",   "Qwen3-VL-8B (LM)", "Qwen3-8B",           "Qwen3"),
    ("D", "LLaVA-1.5-7B",  "LLaVA-1.5 (LM)",   None,                  "LLaVA-1.5"),
    ("^", "LLaVA-Mistral",  "LLaVA-Mistral (LM)", "Mistral-7B",       "Mistral"),
    ("v", "LLaVA-Vicuna",   "LLaVA-Vicuna (LM)", "Vicuna-7B",         "Vicuna"),
    ("P", None,              None,                "Qwen3-8B (think)",   "Qwen3 (think)"),
    ("X", None,              None,                "Qwen2.5-7B-Instruct", "Qwen2.5"),
]

# Build lookup: model name -> (marker, fill_mode)
MODEL_STYLE = {}
for marker, vlm, dec, llm, label in MODEL_FAMILIES:
    if vlm:
        MODEL_STYLE[vlm] = (marker, "filled")
    if dec:
        MODEL_STYLE[dec] = (marker, "open")
    if llm:
        MODEL_STYLE[llm] = (marker, "filled")

# Column model lists
MODEL_GROUPS = {
    "VLM": ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone Decoder": ["InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)",
                         "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"],
    "Standalone LLM": ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct",
                        "Vicuna-7B", "Mistral-7B"],
}

# Use tab20 with spaced indices so all groups get distinct colors
_tab20 = plt.colormaps["tab20"]
# 11 op groups: pick every other index from tab20 (0,2,4,...,20)
OP_PALETTE = [_tab20(i) for i in range(0, 22, 2)][:11]
# 9 ent groups: pick odd indices (1,3,5,...,17)
ENT_PALETTE = [_tab20(i) for i in range(1, 19, 2)][:9]


def make_grouped_scatter(pc: pd.DataFrame, suffix: str = "", title_extra: str = ""):
    hh = pc[pc["pair_type"] == "HH"]
    hm = pc[pc["pair_type"] == "HM"]

    data = {}
    for group_type, col in [("op", "op"), ("ent", "ent")]:
        hh_grp = hh.groupby(col).agg(hh_sbert=("sbert_score", "mean")).reset_index()
        hh_grp.rename(columns={col: "group"}, inplace=True)
        hm_grp = (
            hm.groupby([col, "subject_2"])
            .agg(hm_sbert=("sbert_score", "mean"))
            .reset_index()
        )
        hm_grp.rename(columns={col: "group"}, inplace=True)
        data[group_type] = hm_grp.merge(hh_grp, on="group")

    op_groups = sorted(data["op"]["group"].unique())
    ent_groups = sorted(data["ent"]["group"].unique())
    op_colors = {g: OP_PALETTE[i] for i, g in enumerate(op_groups)}
    ent_colors = {g: ENT_PALETTE[i] for i, g in enumerate(ent_groups)}

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    row_labels = ["Operation", "Entity"]
    group_configs = [
        ("op", op_colors),
        ("ent", ent_colors),
    ]

    for row_idx, (group_type, color_map) in enumerate(group_configs):
        df = data[group_type]

        for col_idx, (col_title, models) in enumerate(MODEL_GROUPS.items()):
            ax = axes[row_idx][col_idx]
            available = [m for m in models if m in df["subject_2"].unique()]

            for model in available:
                marker, fill_mode = MODEL_STYLE[model]
                sub = df[df["subject_2"] == model]

                for _, row in sub.iterrows():
                    c = color_map[row["group"]]
                    if fill_mode == "filled":
                        ax.scatter(row["hh_sbert"], row["hm_sbert"],
                                   c=[c], marker=marker, s=50, alpha=0.8,
                                   edgecolors="white", linewidth=0.4, zorder=3)
                    else:
                        ax.scatter(row["hh_sbert"], row["hm_sbert"],
                                   facecolors="none", marker=marker, s=50, alpha=0.8,
                                   edgecolors=[c], linewidth=1.2, zorder=3)

            lims = [0.15, 0.75]
            ax.plot(lims, lims, "--", color="gray", alpha=0.4, lw=1, zorder=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            if row_idx == 0:
                ax.set_title(col_title, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("HH SBERT (group mean)")
            if col_idx == 0:
                ax.set_ylabel(f"HM SBERT\n({row_labels[row_idx]})")

    # --- Right-side legends (stacked: Operations, Model Family, Entity) ---
    legend_kw = dict(fontsize=9, framealpha=0.9, handletextpad=0.2,
                     borderpad=0.3, labelspacing=0.25, handlelength=1.2)

    # Operations legend (top right)
    op_handles = [Line2D([0], [0], marker="o", color=op_colors[g],
                         linestyle="None", markersize=5, label=g) for g in op_groups]
    op_leg = fig.legend(
        handles=op_handles, loc="upper right",
        bbox_to_anchor=(0.995, 0.97), title="Operations", title_fontsize=9,
        **legend_kw,
    )
    fig.add_artist(op_leg)

    # Model family legend (middle right, between op and ent)
    model_handles = []
    for marker, vlm, dec, llm, label in MODEL_FAMILIES:
        model_handles.append(Line2D(
            [0], [0], marker=marker, color="gray", markerfacecolor="gray",
            linestyle="None", markersize=5, label=label,
        ))
    model_leg = fig.legend(
        handles=model_handles, loc="center right",
        bbox_to_anchor=(0.995, 0.50), title="Model", title_fontsize=9,
        **legend_kw,
    )
    fig.add_artist(model_leg)

    # Entity legend (bottom right)
    ent_handles = [Line2D([0], [0], marker="o", color=ent_colors[g],
                          linestyle="None", markersize=5, label=g) for g in ent_groups]
    fig.legend(
        handles=ent_handles, loc="lower right",
        bbox_to_anchor=(0.995, 0.03), title="Entity", title_fontsize=9,
        **legend_kw,
    )

    plt.tight_layout()
    fig.subplots_adjust(left=0.07, right=0.87)

    out = OUTDIR / f"hh_vs_hm_grouped_scatter_7b_sbert{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")

    human = pd.read_csv(EXPORTS / "responses_human.csv")
    vC = human[human["variant"] == "C"]
    q_yn = vC.groupby("question_id").apply(
        lambda g: (g["response"].str.lower().str.strip().isin(["yes", "no"])).mean() > 0.5,
        include_groups=False,
    )
    yn_qids = set(q_yn[q_yn].index)

    make_grouped_scatter(pc, "_all", " \u2014 all questions")

    pc_ft = pc[~pc["question_id"].isin(yn_qids)]
    make_grouped_scatter(pc_ft, "_freetext", " \u2014 free-text only")
