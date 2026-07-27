"""
Subject×subject mean SBERT matrix (variant C, instruction-blind, abstention-filtered).

Rows and columns = all 77 subjects (40 humans + 37 models), ordered by group.
Cell value = mean pairwise SBERT across all 113 questions for that subject pair.
The diagonal (self-pairs) is filled with the within-subject mean SBERT
(mean over all HH/HM/MM pairs involving that subject, averaged with its partner).

Output: figures/hh/subject_correlation_matrix.png
        latex/AAAI2026/LaTeX/figures/subject_correlation_matrix.png
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPORTS  = ROOT / "analysis" / "session2" / "exports"
OUT_DIR  = ROOT / "figures" / "hh"
LATEX_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

# ── group ordering and display labels ────────────────────────────────────────
GROUP_ORDER = [
    "human",
    "VLM",
    "VLM backbone decoder",
    "standalone LLM",
    "standalone LLM (think)",
]
GROUP_LABELS = {
    "human":                    "Humans",
    "VLM":                      "VLMs",
    "VLM backbone decoder":     "LM Backbone",
    "standalone LLM":           "SA-LLM",
    "standalone LLM (think)":   "Think",
}
GROUP_COLORS = {
    "human":                    "#4C72B0",
    "VLM":                      "#DD8452",
    "VLM backbone decoder":     "#55A868",
    "standalone LLM":           "#C44E52",
    "standalone LLM (think)":   "#8172B2",
}

# ── load variant C, abstention-filtered pair cache ───────────────────────────
pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
pc = pc[pc["variant"] == "C"].copy()

# build subject→group lookup
s1 = pc[["subject_1", "subject_group_1"]].rename(columns={"subject_1": "subject", "subject_group_1": "group"})
s2 = pc[["subject_2", "subject_group_2"]].rename(columns={"subject_2": "subject", "subject_group_2": "group"})
subj_meta = pd.concat([s1, s2]).drop_duplicates().set_index("subject")["group"].to_dict()

# sort subjects: group order, then alphabetical within group
def sort_key(s):
    g = subj_meta.get(s, "z")
    return (GROUP_ORDER.index(g) if g in GROUP_ORDER else 99, s)

all_subjects = sorted(subj_meta.keys(), key=sort_key)
n = len(all_subjects)
idx = {s: i for i, s in enumerate(all_subjects)}

# ── build per-subject display labels ─────────────────────────────────────────
human_counter = 0
display_labels: list[str] = []
for s in all_subjects:
    g = subj_meta.get(s, "")
    if g == "human":
        human_counter += 1
        display_labels.append(str(human_counter))
    elif g == "VLM backbone decoder":
        # strip trailing " (LM)" suffix
        display_labels.append(s.replace(" (LM)", ""))
    else:
        display_labels.append(s)

# ── build mean SBERT matrix ───────────────────────────────────────────────────
mat = np.full((n, n), np.nan)

means = (
    pc.groupby(["subject_1", "subject_2"])["sbert_score"]
    .mean()
    .reset_index()
)

for _, row in means.iterrows():
    i = idx.get(row["subject_1"])
    j = idx.get(row["subject_2"])
    if i is not None and j is not None:
        mat[i, j] = row["sbert_score"]
        mat[j, i] = row["sbert_score"]   # symmetric

# diagonal: mean of all pairs involving this subject
for s, i in idx.items():
    vals = mat[i, :].copy()
    vals[i] = np.nan
    mat[i, i] = np.nanmean(vals)

# ── figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 5,
    "axes.labelsize": 7,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
})

fig, ax = plt.subplots(figsize=(9.0, 8.0))

cmap = LinearSegmentedColormap.from_list(
    "sbert_heat", ["#f7f7f7", "#fddbc7", "#d6604d", "#67001f"], N=256
)
im = ax.imshow(mat, cmap=cmap, vmin=0.2, vmax=0.85, aspect="auto")

# ── group boundary lines ──────────────────────────────────────────────────────
group_spans: dict[str, tuple[int, int]] = {}
for g in GROUP_ORDER:
    members = [s for s in all_subjects if subj_meta.get(s) == g]
    if members:
        lo = idx[members[0]]
        hi = idx[members[-1]]
        group_spans[g] = (lo, hi)

# draw boundary lines between groups
for g in GROUP_ORDER[:-1]:
    if g in group_spans:
        boundary = group_spans[g][1] + 0.5
        ax.axhline(boundary, color="white", lw=1.5)
        ax.axvline(boundary, color="white", lw=1.5)

# ── individual tick labels ────────────────────────────────────────────────────
tick_positions = list(range(n))

# x-axis: angled labels leaning left
ax.set_xticks(tick_positions)
ax.set_xticklabels(display_labels, rotation=45, fontsize=4.5, ha="right", rotation_mode="anchor")

# y-axis: horizontal labels
ax.set_yticks(tick_positions)
ax.set_yticklabels(display_labels, fontsize=4.5)

# ── vertical group label annotations on left side only ───────────────────────
for g in GROUP_ORDER:
    if g not in group_spans:
        continue
    lo, hi = group_spans[g]
    mid = (lo + hi) / 2
    label = GROUP_LABELS[g]
    color = GROUP_COLORS[g]
    ax.annotate(
        label, xy=(0, mid), xycoords=("axes fraction", "data"),
        xytext=(-52, 0), textcoords="offset points",
        ha="center", va="center", fontsize=6.5, fontweight="bold", color=color,
        rotation=90, annotation_clip=False,
    )

ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(n - 0.5, -0.5)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Mean SBERT", fontsize=7)
cbar.ax.tick_params(labelsize=6)

plt.tight_layout()

out = OUT_DIR / "subject_correlation_matrix.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
shutil.copy(out, LATEX_DIR / "subject_correlation_matrix.png")
plt.close(fig)
print(f"Saved: {out}")
print(f"Copied to: {LATEX_DIR / 'subject_correlation_matrix.png'}")
