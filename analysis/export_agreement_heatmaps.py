"""
Export pairwise inter-rater agreement heatmaps for every metric.

For each metric, generates two files in figures/agreement_heatmap/:
  vC_{metric}_all.png    — all raters (36 humans + all model groups)
  vC_{metric}_groups.png — compact version: human block + model group means

Run from the repo root:
  conda run -n zero python analysis/session2/export_agreement_heatmaps.py

Outputs saved to:
  latex/AnonymousSubmission/LaTeX/appendix/agreement/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "analysis/session2/exports/pair_cache.parquet"
OUTPUT_DIR = ROOT / "latex/AnonymousSubmission/LaTeX/figures/agreement_heatmap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
VARIANT = "C"          # which control variant to use
METRICS = {
    "exact":    ("exact_score",    "Exact Match",         (0, 1)),
    "jaccard":  ("jaccard_score",  "Token Jaccard",       (0, 1)),
    "rouge1":   ("rouge1_score",   "ROUGE-1 F1",          (0, 1)),
    "chrf":     ("chrf_score",     "chrF",                (0, 1)),
    "sbert":    ("sbert_score",    "SBERT Cosine",        (0.2, 1.0)),
    "simcse":   ("simcse_score",   "SimCSE Cosine",       (0.2, 1.0)),
    "bertscore":("bertscore_f1",   "BERTScore F1",        (0.7, 1.0)),
}

# Display-name order and group colours
GROUP_ORDER = ["human", "VLM", "VLM backbone decoder",
               "standalone LLM", "standalone LLM (think)"]
GROUP_COLORS = {
    "human":                 "#1565C0",
    "VLM":                   "#E53935",
    "VLM backbone decoder":  "#E67E22",
    "standalone LLM":        "#2E7D32",
    "standalone LLM (think)":"#8E24AA",
}

# Short display labels for individual raters
def short_label(group, subject):
    if group == "human":
        return subject[:6]          # first 6 chars of anonymous ID
    return subject


def build_matrix(df_v, metric_col, subjects, subject_index):
    """Build symmetric pairwise mean-score matrix (N × N)."""
    n = len(subjects)
    mat = np.full((n, n), np.nan)
    # diagonal = self-agreement = max of scale (set later per metric)
    for _, row in df_v.iterrows():
        i = subject_index.get(row["subject_1"])
        j = subject_index.get(row["subject_2"])
        if i is None or j is None:
            continue
        v = row[metric_col]
        if np.isnan(mat[i, j]):
            mat[i, j] = v
            mat[j, i] = v
        else:
            mat[i, j] = (mat[i, j] + v) / 2
            mat[j, i] = mat[i, j]
    # set diagonal to 1.0 (perfect self-agreement)
    np.fill_diagonal(mat, 1.0)
    return mat


def plot_heatmap(mat, labels, groups, title, metric_range, out_path,
                 figsize=None, fontsize_tick=7, fontsize_title=11,
                 show_values=False):
    n = len(labels)
    if figsize is None:
        figsize = (max(8, n * 0.32 + 2), max(7, n * 0.32 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = metric_range
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")

    # Axis ticks
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90,
                                                  fontsize=fontsize_tick)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=fontsize_tick)

    # Colour tick labels by group
    for tick, grp in zip(ax.get_xticklabels(), groups):
        tick.set_color(GROUP_COLORS.get(grp, "black"))
    for tick, grp in zip(ax.get_yticklabels(), groups):
        tick.set_color(GROUP_COLORS.get(grp, "black"))

    # Draw group-boundary lines
    boundary_positions = []
    prev = groups[0]
    for idx, grp in enumerate(groups):
        if grp != prev:
            boundary_positions.append(idx - 0.5)
            prev = grp
    for bp in boundary_positions:
        ax.axhline(bp, color="white", lw=1.5)
        ax.axvline(bp, color="white", lw=1.5)

    if show_values and n <= 20:
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=5, color="black" if mat[i,j] > (vmin+vmax)/2 else "white")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title, fontsize=fontsize_title, pad=10)

    # Group legend
    legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
                      for g in GROUP_ORDER if g in set(groups)]
    ax.legend(handles=legend_patches, loc="upper left",
              bbox_to_anchor=(1.12, 1.0), fontsize=7, frameon=True,
              title="Group", title_fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def build_group_means_matrix(mat, subjects_df):
    """
    Collapse the full matrix into a group-mean matrix.
    Rows/cols: humans (aggregated as one block mean) + each model group.
    Returns (group_mat, group_labels, group_colors_list).
    """
    groups_present = subjects_df["group"].unique().tolist()
    # Keep groups in defined order
    ordered = [g for g in GROUP_ORDER if g in groups_present]

    indices_by_group = {
        g: subjects_df.index[subjects_df["group"] == g].tolist()
        for g in ordered
    }

    n = len(ordered)
    gmat = np.full((n, n), np.nan)
    for i, g1 in enumerate(ordered):
        for j, g2 in enumerate(ordered):
            idxs1 = indices_by_group[g1]
            idxs2 = indices_by_group[g2]
            block = mat[np.ix_(idxs1, idxs2)]
            if i == j:
                # within-group: exclude diagonal (self-pairs)
                mask = ~np.eye(len(idxs1), dtype=bool) if len(idxs1) > 1 else np.ones((1,1), dtype=bool)
                vals = block[mask]
            else:
                vals = block.flatten()
            valid = vals[~np.isnan(vals)]
            gmat[i, j] = valid.mean() if len(valid) > 0 else np.nan

    glabels = [g.replace(" backbone decoder", "\nbackbone") for g in ordered]
    gcolors = [GROUP_COLORS[g] for g in ordered]
    return gmat, glabels, gcolors, ordered


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading pair cache from {CACHE} …")
df = pd.read_parquet(CACHE)
df_v = df[df["variant"] == VARIANT].copy()
print(f"  {len(df_v):,} pairs (variant={VARIANT})")

# Build ordered subject list
s1 = df_v[["subject_group_1", "subject_1"]].rename(
    columns={"subject_group_1": "group", "subject_1": "subject"})
s2 = df_v[["subject_group_2", "subject_2"]].rename(
    columns={"subject_group_2": "group", "subject_2": "subject"})
subjects_df = (pd.concat([s1, s2])
               .drop_duplicates()
               .reset_index(drop=True))
# Sort by group order then subject name
subjects_df["_order"] = subjects_df["group"].map(
    {g: i for i, g in enumerate(GROUP_ORDER)})
subjects_df = subjects_df.sort_values(["_order", "subject"]).reset_index(drop=True)
subjects_df.drop(columns="_order", inplace=True)

subject_index = {row["subject"]: idx for idx, row in subjects_df.iterrows()}
labels = [short_label(row["group"], row["subject"])
          for _, row in subjects_df.iterrows()]
groups = subjects_df["group"].tolist()

print(f"  {len(subjects_df)} raters: "
      + ", ".join(f"{g}={sum(subjects_df['group']==g)}"
                  for g in GROUP_ORDER if g in subjects_df["group"].values))

# ── Generate one heatmap set per metric ───────────────────────────────────────
for metric_key, (col, display_name, mrange) in METRICS.items():
    print(f"\n── {display_name} ({col}) ──")
    mat = build_matrix(df_v, col, subjects_df["subject"].tolist(), subject_index)

    # Full heatmap (all raters)
    plot_heatmap(
        mat, labels, groups,
        title=f"Pairwise {display_name} — all raters (variant {VARIANT})",
        metric_range=mrange,
        out_path=OUTPUT_DIR / f"v{VARIANT}_{metric_key}_all.png",
        figsize=(14, 13),
        fontsize_tick=6.5,
    )

    # Group-mean compact heatmap
    gmat, glabels, gcolors, gordered = build_group_means_matrix(mat, subjects_df)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    norm2 = Normalize(vmin=mrange[0], vmax=mrange[1])
    im2 = ax2.imshow(gmat, cmap="RdYlGn", norm=norm2, aspect="auto")
    n2 = len(glabels)
    ax2.set_xticks(range(n2)); ax2.set_xticklabels(glabels, rotation=30,
                                                     ha="right", fontsize=8)
    ax2.set_yticks(range(n2)); ax2.set_yticklabels(glabels, fontsize=8)
    for tick, c in zip(ax2.get_xticklabels(), gcolors):
        tick.set_color(c)
    for tick, c in zip(ax2.get_yticklabels(), gcolors):
        tick.set_color(c)
    for i in range(n2):
        for j in range(n2):
            if not np.isnan(gmat[i, j]):
                ax2.text(j, i, f"{gmat[i,j]:.3f}", ha="center", va="center",
                         fontsize=8.5, fontweight="bold",
                         color="black" if gmat[i,j] > (mrange[0]+mrange[1])/2 else "white")
    plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    ax2.set_title(f"Group-mean {display_name} (variant {VARIANT})",
                  fontsize=10, pad=8)
    plt.tight_layout()
    out2 = OUTPUT_DIR / f"v{VARIANT}_{metric_key}_groups.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  saved: {out2.name}")

print(f"\nAll heatmaps saved to {OUTPUT_DIR}")
