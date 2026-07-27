"""
Per-question SBERT heatmap: humans (HH) + all models (HM), variant C.
Rows = subjects grouped by architecture; cols = 113 questions sorted easy→hard.
Left colour strip shows group membership; blue curve = mean HH SBERT.
"""

import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PAIR_CACHE = os.path.join(ROOT, "analysis/session2/exports/pair_cache_cleaned.csv")
OUT        = os.path.join(ROOT,
    "latex/AAAI2026/LaTeX/figures/sbert_heatmap_all_subjects.png")

VARIANT = "C"

GROUP_ORDER  = ["human", "VLM", "VLM backbone decoder",
                "standalone LLM", "standalone LLM (think)"]
GROUP_LABELS = {
    "human":                  "Humans\n(HH)",
    "VLM":                    "Full\nVLMs",
    "VLM backbone decoder":   "Backbone\nDecoders",
    "standalone LLM":         "Standalone\nLLMs",
    "standalone LLM (think)": "Thinking\nModels",
}
GROUP_COLORS = {
    "human":                  "#3a7bbf",
    "VLM":                    "#c0392b",
    "VLM backbone decoder":   "#d4850a",
    "standalone LLM":         "#27ae60",
    "standalone LLM (think)": "#8e44ad",
}

# ── Load ──────────────────────────────────────────────────────────────────────
hh_by_q           = defaultdict(list)
hh_by_participant = defaultdict(lambda: defaultdict(list))
hm_by_model       = defaultdict(lambda: defaultdict(list))

print("Loading…")
with open(PAIR_CACHE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["variant"] != VARIANT:
            continue
        qid   = row["question_id"]
        score = float(row["sbert_score"]) if row["sbert_score"] else np.nan
        pt    = row["pair_type"]
        if pt == "HH":
            hh_by_q[qid].append(score)
            hh_by_participant[row["subject_1"]][qid].append(score)
            hh_by_participant[row["subject_2"]][qid].append(score)
        elif pt == "HM":
            grp   = row["subject_group_2"]
            model = row["subject_2"]
            hm_by_model[(grp, model)][qid].append(score)

q_all = sorted(hh_by_q, key=lambda q: np.nanmean(hh_by_q[q]), reverse=True)
nQ    = len(q_all)
mean_hh = np.array([np.nanmean(hh_by_q[q]) for q in q_all])

print(f"  {nQ} questions | {len(hh_by_participant)} humans | "
      f"{len(hm_by_model)} models")

# ── Build row list ────────────────────────────────────────────────────────────
rows = []   # (group, label, array[nQ])

for pid in sorted(hh_by_participant):
    arr = np.array([np.nanmean(hh_by_participant[pid].get(q, [np.nan]))
                    for q in q_all])
    rows.append(("human", pid, arr))

for grp in GROUP_ORDER[1:]:
    for (g, m) in sorted(hm_by_model):
        if g != grp:
            continue
        arr = np.array([np.nanmean(hm_by_model[(g, m)].get(q, [np.nan]))
                        for q in q_all])
        rows.append((grp, m, arr))

nS     = len(rows)
mat    = np.vstack([r for _, _, r in rows])
groups = [r[0] for r in rows]

# Group boundaries
bounds = [0]
for i in range(1, nS):
    if groups[i] != groups[i - 1]:
        bounds.append(i)
bounds.append(nS)

print(f"  Matrix: {nS} rows × {nQ} cols")

# ── Figure with GridSpec: [colour strip | heatmap] ───────────────────────────
fig = plt.figure(figsize=(18, 9))
gs  = GridSpec(1, 2, figure=fig,
               width_ratios=[0.06, 1],
               left=0.02, right=0.88,
               top=0.90, bottom=0.08,
               wspace=0.01)

ax_strip = fig.add_subplot(gs[0])
ax_heat  = fig.add_subplot(gs[1])

# ── Colour strip ──────────────────────────────────────────────────────────────
ax_strip.set_xlim(0, 1)
ax_strip.set_ylim(0, nS)
ax_strip.axis("off")
for i, grp in enumerate(groups):
    ax_strip.add_patch(mpatches.Rectangle(
        (0, i), 1, 1,
        facecolor=GROUP_COLORS[grp], edgecolor="none"
    ))

# Group label annotations centred in strip
for k in range(len(bounds) - 1):
    s, e  = bounds[k], bounds[k + 1]
    mid   = (s + e) / 2
    grp   = groups[s]
    label = GROUP_LABELS[grp] + f"\n(n={e-s})"
    ax_strip.text(0.5, mid, label,
                  ha="center", va="center", fontsize=7,
                  color="white", fontweight="bold", rotation=90,
                  multialignment="center")

# ── Heatmap ───────────────────────────────────────────────────────────────────
im = ax_heat.imshow(mat, aspect="auto", cmap="YlOrRd",
                    vmin=0.0, vmax=1.0,
                    interpolation="nearest", origin="upper")

# Group separator lines
for b in bounds[1:-1]:
    ax_heat.axhline(b - 0.5, color="white", linewidth=1.5)

ax_heat.set_yticks([])

# X-axis
ax_heat.set_xticks(range(0, nQ, 10))
ax_heat.set_xticklabels(range(1, nQ + 1, 10), fontsize=7)
ax_heat.set_xlabel(
    "Questions sorted by human agreement (HH SBERT), easy → hard", fontsize=9)

# Mean HH SBERT curve on a twin x-axis at the top
ax2 = ax_heat.twiny()
ax2.plot(range(nQ), mean_hh, color="#3a7bbf", lw=1.5, alpha=0.9)
ax2.fill_between(range(nQ), mean_hh, 0, color="#3a7bbf", alpha=0.08)
ax2.set_xlim(-0.5, nQ - 0.5)
ax2.set_ylim(0, 1)
ax2.tick_params(axis="x", labeltop=False, top=False)
ax2.tick_params(axis="y", labelsize=7, colors="#3a7bbf",
                right=True, labelright=True, left=False, labelleft=False)
ax2.yaxis.set_label_position("right")
ax2.set_ylabel("Mean HH SBERT →", color="#3a7bbf", fontsize=8,
               rotation=270, labelpad=14)

# Colorbar
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.15)
cbar.set_label("SBERT  (HH for humans · HM for models)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

ax_heat.set_title(
    "Per-question SBERT agreement — humans (HH) and models (HM) "
    "vs. human reference   [variant C, instruction-blind]",
    fontsize=9, pad=8)

fig.savefig(OUT, dpi=150)
print(f"Saved → {OUT}")
