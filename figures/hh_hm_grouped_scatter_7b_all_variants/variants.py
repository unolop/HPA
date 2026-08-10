"""Group-level HH vs HM SBERT scatter figures for the vABC slices.

Outputs are written under `figures/hh_hm_grouped_scatter_7b_all_variants/`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))
import plot_style  # noqa: F401 — sets 10pt Times New Roman
EXPORTS = ROOT / "analysis/session2/exports"
OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)

# Model families: marker shape shared across VLM, decoder, standalone
# (marker, VLM, Decoder, Standalone LLM, display label)
MODEL_FAMILIES = [
    ("o", "InternVL-8B",    "InternVL-8B (LM)", None,                  "InternVL"),
    ("X", "Qwen3-VL-8B",   "Qwen3-VL-8B (LM)", "Qwen3-8B",           "Qwen3"),
    ("D", "LLaVA-1.5-7B",  "LLaVA-1.5 (LM)",   None,                  "LLaVA-1.5"),
    ("^", "LLaVA-Mistral",  "LLaVA-Mistral (LM)", "Mistral-7B",       "Mistral"),
    ("v", "LLaVA-Vicuna",   "LLaVA-Vicuna (LM)", "Vicuna-7B",         "Vicuna"),
    ("P", None,              None,                "Qwen3-8B (think)",   "Qwen3 (think)"),
    ("s", None,              None,                "Qwen2.5-7B-Instruct", "Qwen2.5"),
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
    "VLM": ["InternVL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna", "Qwen3-VL-8B"],
    "Backbone Decoder": ["InternVL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)",
                         "LLaVA-Vicuna (LM)", "Qwen3-VL-8B (LM)"],
    "Standalone LLM": ["Qwen2.5-7B-Instruct", "Qwen3-8B", "Qwen3-8B (think)",
                        "Vicuna-7B", "Mistral-7B"],
}

# Use tab20 with spaced indices so all groups get distinct colors
_tab20 = plt.colormaps["tab20"]
# 11 op groups: pick every other index from tab20 (0,2,4,...,20)
OP_PALETTE = [_tab20(i) for i in range(0, 22, 2)][:11]
# 9 ent groups: pick odd indices (1,3,5,...,17)
ENT_PALETTE = [_tab20(i) for i in range(1, 19, 2)][:9]


def make_scatter(pc: pd.DataFrame, suffix: str = "", title_extra: str = ""):
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
                         linestyle="None", markersize=8, label=g) for g in op_groups]
    op_leg = fig.legend(
        handles=op_handles, loc="upper right",
        bbox_to_anchor=(0.995, 0.97), title="Operations", title_fontsize=9,
        **legend_kw,
    )
    fig.add_artist(op_leg)

    # Entity legend (bottom right)
    ent_handles = [Line2D([0], [0], marker="o", color=ent_colors[g],
                          linestyle="None", markersize=8, label=g) for g in ent_groups]
    fig.legend(
        handles=ent_handles, loc="lower right",
        bbox_to_anchor=(0.995, 0.03), title="Entity", title_fontsize=9,
        **legend_kw,
    )

    plt.tight_layout()
    fig.subplots_adjust(left=0.07, right=0.87)

    out = OUTDIR / f"7b{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ── All-scales scatter ────────────────────────────────────────────────────────
# Includes all Qwen3 parameter sizes; marker size encodes scale.

MODEL_SIZE_B_SCATTER = {
    # VLM
    "InternVL-1B": 1, "InternVL-2B": 2, "InternVL-8B": 8,
    "Qwen3-VL-2B": 2, "Qwen3-VL-4B": 4, "Qwen3-VL-8B": 8, "Qwen3-VL-32B": 32,
    "LLaVA-1.5-7B": 7, "LLaVA-Mistral": 7, "LLaVA-Vicuna": 7, "LLaVA-Vicuna-13B": 13,
    # Backbone
    "InternVL-1B (LM)": 1, "InternVL-2B (LM)": 2, "InternVL-8B (LM)": 8,
    "Qwen3-VL-2B (LM)": 2, "Qwen3-VL-4B (LM)": 4, "Qwen3-VL-8B (LM)": 8, "Qwen3-VL-32B (LM)": 32,
    "LLaVA-1.5 (LM)": 7, "LLaVA-Mistral (LM)": 7, "LLaVA-Vicuna (LM)": 7, "LLaVA-Vicuna-13B (LM)": 13,
    # Standalone
    "Qwen3-0.6B": 0.6, "Qwen3-1.7B": 1.7, "Qwen3-4B": 4, "Qwen3-8B": 8, "Qwen3-32B": 32,
    "Qwen3-0.6B (think)": 0.6, "Qwen3-1.7B (think)": 1.7, "Qwen3-4B (think)": 4,
    "Qwen3-8B (think)": 8, "Qwen3-32B (think)": 32,
    "Qwen2.5-7B-Instruct": 7, "Vicuna-7B": 7, "Vicuna-13B": 13, "Mistral-7B": 7,
}

def _size_to_marker_s(size_b: float) -> float:
    """Map parameter count (B) to scatter marker area."""
    return 20 + 60 * (np.log10(max(size_b, 0.3)) / np.log10(32))

# Extended model style covering all scales
MODEL_STYLE_ALL = {}
for marker, vlm, dec, llm, label in MODEL_FAMILIES:
    if vlm:
        for v in [vlm, vlm.replace("-8B", "-2B"), vlm.replace("-8B", "-4B"),
                  vlm.replace("-8B", "-32B"), vlm.replace("-8B", "-1B")]:
            if v:
                MODEL_STYLE_ALL[v] = (marker, "filled")
    if dec:
        for d in [dec, dec.replace("-8B", "-2B"), dec.replace("-8B", "-4B"),
                  dec.replace("-8B", "-32B"), dec.replace("-8B", "-1B")]:
            if d:
                MODEL_STYLE_ALL[d] = (marker, "open")
    if llm:
        for l in [llm, llm.replace("-8B", "-0.6B"), llm.replace("-8B", "-1.7B"),
                  llm.replace("-8B", "-4B"), llm.replace("-8B", "-32B")]:
            if l:
                MODEL_STYLE_ALL[l] = (marker, "filled")
# Vicuna-13B
MODEL_STYLE_ALL["Vicuna-13B"] = ("v", "filled")
MODEL_STYLE_ALL["LLaVA-Vicuna-13B"] = ("v", "filled")
MODEL_STYLE_ALL["LLaVA-Vicuna-13B (LM)"] = ("v", "open")

MODEL_GROUPS_ALL = {
    "VLM": ["InternVL-1B", "InternVL-2B", "InternVL-8B",
            "Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B", "Qwen3-VL-32B",
            "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna", "LLaVA-Vicuna-13B"],
    "Backbone Decoder": ["InternVL-1B (LM)", "InternVL-2B (LM)", "InternVL-8B (LM)",
                         "Qwen3-VL-2B (LM)", "Qwen3-VL-4B (LM)", "Qwen3-VL-8B (LM)", "Qwen3-VL-32B (LM)",
                         "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)", "LLaVA-Vicuna-13B (LM)"],
    "Standalone LLM": ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-32B",
                       "Qwen3-0.6B (think)", "Qwen3-1.7B (think)", "Qwen3-4B (think)",
                       "Qwen3-8B (think)", "Qwen3-32B (think)",
                       "Qwen2.5-7B-Instruct", "Vicuna-7B", "Vicuna-13B"],
}


def make_scatter_all_scales(pc: pd.DataFrame, suffix: str = ""):
    """Like make_scatter but includes all model scales; marker size encodes param count."""
    hh = pc[pc["pair_type"] == "HH"]
    hm = pc[pc["pair_type"] == "HM"]

    data = {}
    for group_type in ["op", "ent"]:
        hh_grp = hh.groupby(group_type).agg(hh_sbert=("sbert_score", "mean")).reset_index()
        hh_grp.rename(columns={group_type: "group"}, inplace=True)
        hm_grp = (hm.groupby([group_type, "subject_2"])
                    .agg(hm_sbert=("sbert_score", "mean")).reset_index())
        hm_grp.rename(columns={group_type: "group"}, inplace=True)
        data[group_type] = hm_grp.merge(hh_grp, on="group")

    op_groups  = sorted(data["op"]["group"].unique())
    ent_groups = sorted(data["ent"]["group"].unique())
    op_colors  = {g: OP_PALETTE[i]  for i, g in enumerate(op_groups)}
    ent_colors = {g: ENT_PALETTE[i] for i, g in enumerate(ent_groups)}

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)

    for row_idx, (group_type, color_map) in enumerate([("op", op_colors), ("ent", ent_colors)]):
        df = data[group_type]
        for col_idx, (col_title, models) in enumerate(MODEL_GROUPS_ALL.items()):
            ax = axes[row_idx][col_idx]
            available = [m for m in models if m in df["subject_2"].unique()]

            for model in available:
                style = MODEL_STYLE_ALL.get(model)
                if style is None:
                    continue
                marker, fill_mode = style
                size_b = MODEL_SIZE_B_SCATTER.get(model, 7)
                ms = _size_to_marker_s(size_b)
                sub = df[df["subject_2"] == model]

                for _, row in sub.iterrows():
                    c = color_map[row["group"]]
                    if fill_mode == "filled":
                        ax.scatter(row["hh_sbert"], row["hm_sbert"],
                                   c=[c], marker=marker, s=ms, alpha=0.75,
                                   edgecolors="white", linewidth=0.3, zorder=3)
                    else:
                        ax.scatter(row["hh_sbert"], row["hm_sbert"],
                                   facecolors="none", marker=marker, s=ms, alpha=0.75,
                                   edgecolors=[c], linewidth=1.1, zorder=3)

            lims = [0.15, 0.75]
            ax.plot(lims, lims, "--", color="gray", alpha=0.4, lw=1, zorder=1)
            ax.set_xlim(lims); ax.set_ylim(lims)
            if row_idx == 0:
                ax.set_title(col_title, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("HH SBERT (group mean)")
            if col_idx == 0:
                ax.set_ylabel(f"HM SBERT\n({['Operation','Entity'][row_idx]})")

    legend_kw = dict(fontsize=9, framealpha=0.9, handletextpad=0.2,
                     borderpad=0.3, labelspacing=0.25, handlelength=1.2)
    op_handles = [Line2D([0],[0], marker="o", color=op_colors[g],
                         linestyle="None", markersize=8, label=g) for g in op_groups]
    op_leg = fig.legend(handles=op_handles, loc="upper right",
        bbox_to_anchor=(0.995, 0.97), title="Operations", title_fontsize=9, **legend_kw)
    fig.add_artist(op_leg)

    ent_handles = [Line2D([0],[0], marker="o", color=ent_colors[g],
                          linestyle="None", markersize=8, label=g) for g in ent_groups]
    fig.legend(handles=ent_handles, loc="lower right",
        bbox_to_anchor=(0.995, 0.03), title="Entity", title_fontsize=9, **legend_kw)

    plt.tight_layout()
    fig.subplots_adjust(left=0.07, right=0.87)
    out = OUTDIR / f"allscales{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


_ROW_SHORT = {
    "VLM": "VLM",
    "Backbone Decoder": "Backbone",
    "Standalone LLM": "SA-LLM",
}


def make_scatter_all_scales_transposed(pc: pd.DataFrame, suffix: str = ""):
    """Transposed layout: 3 rows (model groups) × 2 cols (Op / Entity).
    x-axis (HH SBERT group mean) is shared across model-group rows within each
    question-grouping column, making Op vs Entity directly comparable per row.
    """
    hh = pc[pc["pair_type"] == "HH"]
    hm = pc[pc["pair_type"] == "HM"]

    data = {}
    for group_type in ["op", "ent"]:
        hh_grp = hh.groupby(group_type).agg(hh_sbert=("sbert_score", "mean")).reset_index()
        hh_grp.rename(columns={group_type: "group"}, inplace=True)
        hm_grp = (hm.groupby([group_type, "subject_2"])
                    .agg(hm_sbert=("sbert_score", "mean")).reset_index())
        hm_grp.rename(columns={group_type: "group"}, inplace=True)
        data[group_type] = hm_grp.merge(hh_grp, on="group")

    op_groups  = sorted(data["op"]["group"].unique())
    ent_groups = sorted(data["ent"]["group"].unique())
    op_colors  = {g: OP_PALETTE[i]  for i, g in enumerate(op_groups)}
    ent_colors = {g: ENT_PALETTE[i] for i, g in enumerate(ent_groups)}

    col_configs = [
        ("op",  op_colors,  "Operation"),
        ("ent", ent_colors, "Entity"),
    ]
    row_configs = list(MODEL_GROUPS_ALL.items())  # (title, models)

    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex="col", sharey=True)

    for row_idx, (row_title, models) in enumerate(row_configs):
        for col_idx, (group_type, color_map, col_label) in enumerate(col_configs):
            ax = axes[row_idx][col_idx]
            df = data[group_type]
            available = [m for m in models if m in df["subject_2"].unique()]

            for model in available:
                style = MODEL_STYLE_ALL.get(model)
                if style is None:
                    continue
                marker, fill_mode = style
                size_b = MODEL_SIZE_B_SCATTER.get(model, 7)
                ms = _size_to_marker_s(size_b)
                sub = df[df["subject_2"] == model]

                for _, row in sub.iterrows():
                    c = color_map[row["group"]]
                    if fill_mode == "filled":
                        ax.scatter(row["hh_sbert"], row["hm_sbert"],
                                   c=[c], marker=marker, s=ms, alpha=0.75,
                                   edgecolors="white", linewidth=0.3, zorder=3)
                    else:
                        ax.scatter(row["hh_sbert"], row["hm_sbert"],
                                   facecolors="none", marker=marker, s=ms, alpha=0.75,
                                   edgecolors=[c], linewidth=1.1, zorder=3)

            lims = [0.15, 0.75]
            ax.plot(lims, lims, "--", color="gray", alpha=0.4, lw=1, zorder=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            if row_idx == 0:
                ax.set_title(col_label, fontweight="bold")
            if row_idx == len(row_configs) - 1:
                ax.set_xlabel("HH SBERT (group mean)")
            if col_idx == 0:
                ax.set_ylabel(f"{_ROW_SHORT[row_title]}\n(HM SBERT)")

    legend_kw = dict(fontsize=9, framealpha=0.9, handletextpad=0.2,
                     borderpad=0.3, labelspacing=0.25, handlelength=1.2)
    op_handles = [Line2D([0], [0], marker="o", color=op_colors[g],
                         linestyle="None", markersize=8, label=g) for g in op_groups]
    op_leg = fig.legend(handles=op_handles, loc="upper right",
        bbox_to_anchor=(0.995, 0.97), title="Operations", title_fontsize=9, **legend_kw)
    fig.add_artist(op_leg)

    ent_handles = [Line2D([0], [0], marker="o", color=ent_colors[g],
                          linestyle="None", markersize=8, label=g) for g in ent_groups]
    fig.legend(handles=ent_handles, loc="lower right",
        bbox_to_anchor=(0.995, 0.03), title="Entity", title_fontsize=9, **legend_kw)

    plt.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.82)
    out = OUTDIR / f"allscales{suffix}_T.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def make_scatter_compact(pc: pd.DataFrame, suffix: str = ""):
    """Compact 3×1 figure: rows = model groups, one point per operation category.

    Operation categories colored; one point per (op, model-class) averaged across all models.
    Designed to fit as a single-column figure in two-column paper layout.
    """
    hh = pc[pc["pair_type"] == "HH"]
    hm = pc[pc["pair_type"] == "HM"]

    hh_grp = hh.groupby("op").agg(hh_sbert=("sbert_score", "mean")).reset_index()

    model_class_map = {}
    for col_title, models in MODEL_GROUPS_ALL.items():
        for m in models:
            model_class_map[m] = col_title

    hm_tagged = hm.copy()
    hm_tagged["model_class"] = hm_tagged["subject_2"].map(model_class_map)
    hm_tagged = hm_tagged.dropna(subset=["model_class"])

    hm_grp = (hm_tagged.groupby(["op", "model_class"])
              .agg(hm_sbert=("sbert_score", "mean")).reset_index())
    merged = hm_grp.merge(hh_grp, on="op")

    op_groups = sorted(merged["op"].unique())
    op_colors_compact = {g: OP_PALETTE[i] for i, g in enumerate(op_groups)}

    row_order = ["VLM", "Backbone Decoder", "Standalone LLM"]
    row_labels = {"VLM": "VLM", "Backbone Decoder": "Backbone\nDecoder",
                  "Standalone LLM": "Standalone\nLLM"}

    lims = [0.15, 0.75]

    fig, axes = plt.subplots(3, 1, figsize=(3.2, 5.5),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.18, "left": 0.18,
                                          "right": 0.97, "top": 0.97, "bottom": 0.26})

    for row_idx, row_title in enumerate(row_order):
        ax = axes[row_idx]
        sub = merged[merged["model_class"] == row_title]

        for _, row in sub.iterrows():
            c = op_colors_compact[row["op"]]
            ax.scatter(row["hh_sbert"], row["hm_sbert"],
                       s=40, alpha=0.85, color=c, edgecolors="white",
                       linewidth=0.5, zorder=3)

        ax.plot(lims, lims, "--", color="gray", alpha=0.35, lw=1, zorder=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_ylabel(row_labels[row_title], fontsize=7.5)
        if row_idx == len(row_order) - 1:
            ax.set_xlabel("HH SBERT", fontsize=7.5)

    # Operation legend centered below plots
    op_handles = [Line2D([0], [0], marker="o", color=op_colors_compact[g],
                         linestyle="None", markersize=8, label=g) for g in op_groups]
    fig.legend(handles=op_handles, loc="lower center",
               bbox_to_anchor=(0.57, 0.0), title="Operation", title_fontsize=7,
               ncol=3, fontsize=6.5, frameon=False,
               handletextpad=0.2, borderpad=0.2, labelspacing=0.10,
               columnspacing=0.5)

    out = OUTDIR / f"allscales{suffix}_compact.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return out


def make_scatter_compact_single(pc: pd.DataFrame, group_type: str, suffix: str = ""):
    """Compact 3x1 figure for a single grouping axis (operation or entity)."""
    assert group_type in {"op", "ent"}
    hh = pc[pc["pair_type"] == "HH"]
    hm = pc[pc["pair_type"] == "HM"]

    hh_grp = hh.groupby(group_type).agg(hh_sbert=("sbert_score", "mean")).reset_index()

    model_class_map = {}
    for col_title, models in MODEL_GROUPS_ALL.items():
        for m in models:
            model_class_map[m] = col_title

    hm_tagged = hm.copy()
    hm_tagged["model_class"] = hm_tagged["subject_2"].map(model_class_map)
    hm_tagged = hm_tagged.dropna(subset=["model_class"])
    hm_grp = (hm_tagged.groupby([group_type, "model_class"])
              .agg(hm_sbert=("sbert_score", "mean")).reset_index())
    merged = hm_grp.merge(hh_grp, on=group_type)

    groups_sorted = sorted(merged[group_type].unique())
    palette = OP_PALETTE if group_type == "op" else ENT_PALETTE
    color_map = {g: palette[i] for i, g in enumerate(groups_sorted)}

    row_order = ["VLM", "Backbone Decoder", "Standalone LLM"]
    row_labels = {"VLM": "VLM", "Backbone Decoder": "Backbone", "Standalone LLM": "SA-LLM"}
    title = "Operation" if group_type == "op" else "Entity"
    lims = [0.15, 0.75]
    row_colors = {
        "VLM": "#1565C0",
        "Backbone Decoder": "#42A5F5",
        "Standalone LLM": "#2E7D32",
    }

    fig, axes = plt.subplots(
        3, 1, figsize=(2.65, 4.55),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.14, "left": 0.22, "right": 0.73, "top": 0.95, "bottom": 0.09},
    )

    for row_idx, row_title in enumerate(row_order):
        ax = axes[row_idx]
        sub = merged[merged["model_class"] == row_title]

        for _, row in sub.iterrows():
            c = color_map[row[group_type]]
            ax.scatter(
                row["hh_sbert"], row["hm_sbert"],
                s=28, alpha=0.88, color=c, edgecolors="white",
                linewidth=0.45, zorder=3,
            )

        ax.plot(lims, lims, "--", color="gray", alpha=0.35, lw=1, zorder=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(labelsize=8)
        ax.set_ylabel(row_labels[row_title], fontsize=9)
        if row_idx == 0:
            ax.set_title(title, fontsize=10, fontweight="bold")
        if row_idx == len(row_order) - 1:
            ax.set_xlabel("HH SBERT", fontsize=9)

        ax.spines["left"].set_color(row_colors[row_title])
        ax.spines["left"].set_linewidth(1.5)

    q_handles = [
        Line2D([0], [0], marker="o", color=color_map[g],
               linestyle="None", markersize=4.6, label=g)
        for g in groups_sorted
    ]
    fig.legend(
        handles=q_handles, loc="center right",
        bbox_to_anchor=(1.03, 0.50),
        fontsize=7.2, frameon=True, framealpha=0.92,
        handletextpad=0.25, borderpad=0.35, labelspacing=0.22,
    )

    base = "op" if group_type == "op" else "ent"
    out = OUTDIR / f"allscales{suffix}_compact_{base}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    import shutil
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "figures"))
    from helpers import filter_abstained_pairs

    LATEX_FIGURES = ROOT / "latex/AAAI2026/LaTeX/figures/hh_hm_grouped_scatter_7b_all_variants"
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)

    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")

    human = pd.read_csv(EXPORTS / "responses_human.csv")
    vC = human[human["variant"] == "C"]
    q_yn = vC.groupby("question_id").apply(
        lambda g: (g["response"].str.lower().str.strip().isin(["yes", "no"])).mean() > 0.5,
        include_groups=False,
    )
    yn_qids = set(q_yn[q_yn].index)

    print("\nFiltering abstentions (new standard)…")
    pc_filt = filter_abstained_pairs(pc)

    # 7B figures — unfiltered and abstention-filtered
    make_scatter(pc, "_all_raw", " — all questions")
    make_scatter(pc[~pc["question_id"].isin(yn_qids)], "_freetext_raw", " — free-text only")
    make_scatter(pc_filt, "_all_abstfiltered", " — all questions (abstention-filtered)")
    make_scatter(pc_filt[~pc_filt["question_id"].isin(yn_qids)], "_freetext_abstfiltered", " — free-text only (abstention-filtered)")

    # All-scales figures — unfiltered and abstention-filtered
    make_scatter_all_scales(pc, "_all_raw")
    shutil.copy(OUTDIR / "allscales_all_raw.png", LATEX_FIGURES / "allscales_all_raw.png")
    make_scatter_all_scales(pc[~pc["question_id"].isin(yn_qids)], "_freetext_raw")
    make_scatter_all_scales(pc_filt, "_all_abstfiltered")
    shutil.copy(OUTDIR / "allscales_all_abstfiltered.png", LATEX_FIGURES / "allscales_all_abstfiltered.png")
    make_scatter_all_scales(pc_filt[~pc_filt["question_id"].isin(yn_qids)], "_freetext_abstfiltered")

    # Transposed layout (3 rows model groups × 2 cols Op/Entity)
    make_scatter_all_scales_transposed(pc, "_all_raw")
    make_scatter_all_scales_transposed(pc[~pc["question_id"].isin(yn_qids)], "_freetext_raw")
    make_scatter_all_scales_transposed(pc_filt, "_all_abstfiltered")
    make_scatter_all_scales_transposed(pc_filt[~pc_filt["question_id"].isin(yn_qids)], "_freetext_abstfiltered")

    # Compact 1×3 figure for main paper
    compact_out = make_scatter_compact(pc_filt, "_all_abstfiltered")
    shutil.copy(compact_out, LATEX_FIGURES / compact_out.name)
    print(f"Copied compact figure to LaTeX")

    # Separate compact single-grouping figures
    compact_op = make_scatter_compact_single(pc_filt, "op", "_all_abstfiltered")
    compact_ent = make_scatter_compact_single(pc_filt, "ent", "_all_abstfiltered")
    shutil.copy(compact_op, LATEX_FIGURES / compact_op.name)
    shutil.copy(compact_ent, LATEX_FIGURES / compact_ent.name)
    print("Copied compact op/entity figures to LaTeX")
