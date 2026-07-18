"""
grouped_distribution_bars.py

Horizontal bar chart showing each model's Mean JS alignment to human
grouped answer distributions, faceted by answer type (yes/no / number / text).

One figure for entity grouping (ent) and one for operation grouping (op).

Run from repo root:
  conda run -n zero python figures/grouped_distribution/grouped_distribution_bars.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "figures/grouped_distribution"
LATEX_OUT = ROOT / "latex/AAAI2026/LaTeX/figures/grouped_distribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_OUT.mkdir(parents=True, exist_ok=True)

DATA_ENT = ROOT / "notebooks/exports/alignment_tables/grouped_distribution_ent_7b_all_variants.csv"
DATA_OP  = ROOT / "notebooks/exports/alignment_tables/grouped_distribution_op_7b_all_variants.csv"

GROUP_ORDER  = ["VLM", "Backbone Decoder", "Standalone LLM"]
GROUP_COLORS = {
    "VLM":              "#9E4A3A",
    "Backbone Decoder": "#C0843D",
    "Standalone LLM":   "#5E8C61",
    "Human baseline":   "#424242",
}
GROUP_LABELS = {
    "VLM":              "VLM",
    "Backbone Decoder": "Backbone",
    "Standalone LLM":   "SA-LLM",
}

ANSWER_TYPE_LABELS = {
    "yes/no":  "Yes/No",
    "number":  "Count",
    "text":    "Text",
    # op-type labels
    "attr":    "Attr",
    "count":   "Count",
    "ident":   "Identity",
    "spat":    "Spatial",
    "exist":   "Exist",
    "act":     "Action",
    "other":   "Other",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
})


def make_bar_figure(csv_path: Path, answer_types: list[str], fname: str, title: str):
    df = pd.read_csv(csv_path)

    # Separate human baseline
    human_row = df[df["Group"] == "Human baseline"].iloc[0]
    models_df = df[df["Group"] != "Human baseline"].copy()

    # Sort within each group by overall mean JS (ascending = best first)
    js_cols = [f"{at} | Mean JS" for at in answer_types]
    models_df["_mean_js"] = models_df[js_cols].mean(axis=1)
    models_df = models_df.sort_values(["Group", "_mean_js"], ascending=[True, True])

    # Build ordered model list: group by GROUP_ORDER, best (lowest JS) at top within group
    ordered_models = []
    group_boundaries = {}
    y = 0
    for grp in GROUP_ORDER:
        sub = models_df[models_df["Group"] == grp]
        if sub.empty:
            continue
        group_boundaries[grp] = (y, y + len(sub))
        for _, row in sub.iterrows():
            ordered_models.append(row)
        y += len(sub)

    n_models = len(ordered_models)
    n_types  = len(answer_types)

    fig, axes = plt.subplots(
        1, n_types,
        figsize=(n_types * 2.8 + 0.4, n_models * 0.38 + 0.9),
        sharey=True,
    )
    if n_types == 1:
        axes = [axes]

    y_positions = np.arange(n_models)
    y_labels = [row["Model"] for row in ordered_models]

    for ax_idx, at in enumerate(answer_types):
        ax = axes[ax_idx]
        col = f"{at} | Mean JS"
        human_val = float(human_row[col])

        for y_pos, row in enumerate(ordered_models):
            grp = row["Group"]
            color = GROUP_COLORS[grp]
            val = float(row[col])
            ax.barh(y_pos, val, color=color, height=0.62, alpha=0.85, zorder=2)

        # Human LOO reference line
        ax.axvline(human_val, color="#424242", lw=1.2, ls="--", zorder=3,
                   label=f"Human LOO ({human_val:.3f})")

        # Group separator lines
        prev = 0
        for grp in GROUP_ORDER:
            if grp not in group_boundaries:
                continue
            lo, hi = group_boundaries[grp]
            if lo > prev:
                ax.axhline(lo - 0.5, color="#ccc", lw=0.8, zorder=1)
            prev = hi

        at_label = ANSWER_TYPE_LABELS.get(at, at)
        ax.set_title(at_label, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Mean JS ↓", fontsize=8)
        ax.set_xlim(left=0)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", lw=0.5, alpha=0.4, zorder=0)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=7.5)
    axes[0].invert_yaxis()

    # Shaded background bands per group on all axes
    for ax in axes:
        for grp in GROUP_ORDER:
            if grp not in group_boundaries:
                continue
            lo, hi = group_boundaries[grp]
            ax.axhspan(lo - 0.5, hi - 0.5, color=GROUP_COLORS[grp],
                       alpha=0.06, zorder=0)

    # Group labels inside the first panel, right-aligned at each band midpoint
    for grp in GROUP_ORDER:
        if grp not in group_boundaries:
            continue
        lo, hi = group_boundaries[grp]
        mid = (lo + hi - 1) / 2
        axes[0].text(
            axes[0].get_xlim()[1] * 0.97, mid,
            GROUP_LABELS[grp],
            fontsize=7, fontweight="bold",
            color=GROUP_COLORS[grp],
            ha="right", va="center", zorder=5,
        )

    # Legend
    legend_handles = [
        mlines.Line2D([], [], color="#424242", lw=1.2, ls="--", label="Human LOO"),
    ] + [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g], alpha=0.85,
                       label=GROUP_LABELS[g])
        for g in GROUP_ORDER
    ]
    fig.legend(handles=legend_handles, fontsize=7.5, ncol=len(legend_handles),
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               frameon=False, handletextpad=0.5, columnspacing=1.0)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out = OUT_DIR / fname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_OUT / fname)
    print(f"Saved: {out}")
    plt.close(fig)


# Entity grouping: yes/no, number (count), text
make_bar_figure(
    DATA_ENT,
    answer_types=["yes/no", "number", "text"],
    fname="grouped_dist_ent_bars.png",
    title="Model–Human Distribution Alignment by Answer Type (Entity Groups)",
)

# Operation grouping
make_bar_figure(
    DATA_OP,
    answer_types=["yes/no", "number", "text"],
    fname="grouped_dist_op_bars.png",
    title="Model–Human Distribution Alignment by Answer Type (Operation Groups)",
)

print("\nDone.")
