"""
InternVL-8B VLM vs backbone LM — sHM by entity group.

Left panel : grouped horizontal bars (VLM, LM) per entity.
Right panel: difference bar (LM − VLM) per entity, coloured by sign.

Entities sorted by (LM − VLM) ascending (most negative at top → vehicle first).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── project paths ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent   # HPA/
FIGURES_DIR = REPO / "figures"
sys.path.insert(0, str(FIGURES_DIR))

import plot_style  # noqa  — sets global rcParams
from helpers import filter_abstained_pairs

# ── constants ──────────────────────────────────────────────────────────────────
PAIR_CACHE = REPO / "analysis/session2/exports/pair_cache_cleaned.parquet"
VQA_ANNOTS  = REPO / "dataset/vqa/v2_mscoco_val2014_annotations.json"
OUT_DIR     = Path(__file__).resolve().parent

MODEL_VLM = "InternVL-8B"
MODEL_LM  = "InternVL-8B (LM)"

ENT_LABELS = {
    "animal":  "Animal",
    "food":    "Food",
    "object":  "Object",
    "other":   "Other",
    "person":  "Person",
    "place":   "Place",
    "product": "Product",
    "text":    "Text",
    "vehicle": "Vehicle",
}

COLOR_VLM    = "#0072B2"   # blue   (Okabe-Ito)
COLOR_LM     = "#56B4E9"   # light-blue (Okabe-Ito)
COLOR_POS    = "#009E73"   # teal-green (LM better)
COLOR_NEG    = "#D55E00"   # orange-red  (VLM better)
BG_COLOR     = "#f7f7f7"

FONTSIZE     = 8.5

# ── load & filter data ─────────────────────────────────────────────────────────
print("Loading pair cache …")
pc = pd.read_parquet(PAIR_CACHE)

# HM pairs only
pc = pc[pc["pair_type"] == "HM"].copy()
print(f"  HM pairs: {len(pc):,}")

# Filter abstentions
pc = filter_abstained_pairs(pc)

# Keep only free-text questions (answer_type == "other")
print("Loading VQA annotations …")
with open(VQA_ANNOTS) as f:
    vqa_data = json.load(f)
other_qids = {
    a["question_id"]
    for a in vqa_data["annotations"]
    if a["answer_type"] == "other"
}
pc = pc[pc["question_id"].isin(other_qids)].copy()
print(f"  After 'other' filter: {len(pc):,} rows, "
      f"{pc['question_id'].nunique()} questions")

# Keep only InternVL-8B models
pc = pc[pc["subject_2"].isin([MODEL_VLM, MODEL_LM])].copy()
print(f"  After model filter: {len(pc):,} rows")
print(f"  Models: {pc['subject_2'].value_counts().to_dict()}")

# ── compute mean sHM per (model, ent) ─────────────────────────────────────────
grouped = (
    pc.groupby(["subject_2", "ent"])["sbert_score"]
    .mean()
    .reset_index()
    .rename(columns={"sbert_score": "mean_sHM"})
)

# Pivot to wide
pivot = grouped.pivot(index="ent", columns="subject_2", values="mean_sHM").reset_index()
pivot.columns.name = None

# Some entities may be missing for LM (fewer questions covered) — fill with NaN
for col in [MODEL_VLM, MODEL_LM]:
    if col not in pivot.columns:
        pivot[col] = float("nan")

pivot["diff"] = pivot[MODEL_LM] - pivot[MODEL_VLM]  # positive → LM better

# Sort by diff ascending (most negative first → vehicle at top of horizontal bars)
pivot = pivot.sort_values("diff", ascending=True).reset_index(drop=True)

ents_sorted  = [ENT_LABELS.get(e, e) for e in pivot["ent"]]
vals_vlm     = pivot[MODEL_VLM].values
vals_lm      = pivot[MODEL_LM].values
diffs        = pivot["diff"].values
n            = len(ents_sorted)

print("\nSorted entities + diff:")
for e, d in zip(ents_sorted, diffs):
    print(f"  {e:<10}  diff={d:+.4f}")

# ── figure ─────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(
    1, 2,
    figsize=(5.5, 3.2),
    gridspec_kw={"width_ratios": [2.2, 1.8], "wspace": 0.55},
)

y        = np.arange(n)
bar_h    = 0.35
tick_kw  = dict(fontsize=FONTSIZE)

# ── Left panel: grouped horizontal bars ───────────────────────────────────────
ax_left.set_facecolor(BG_COLOR)
ax_left.grid(axis="x", color="white", linewidth=0.8, zorder=0)

bars_vlm = ax_left.barh(
    y + bar_h / 2, vals_vlm, height=bar_h,
    color=COLOR_VLM, label=MODEL_VLM, zorder=3,
)
bars_lm = ax_left.barh(
    y - bar_h / 2, vals_lm, height=bar_h,
    color=COLOR_LM, label=MODEL_LM, zorder=3,
)

ax_left.set_yticks(y)
ax_left.set_yticklabels(ents_sorted, fontsize=FONTSIZE)
ax_left.set_xlabel("Mean sHM", fontsize=FONTSIZE)
ax_left.set_xlim(0, 0.60)
ax_left.tick_params(axis="x", labelsize=FONTSIZE)
ax_left.set_title("sHM by Entity Group", fontsize=FONTSIZE, pad=4)

# Legend inside left panel
patch_vlm = mpatches.Patch(color=COLOR_VLM, label=MODEL_VLM)
patch_lm  = mpatches.Patch(color=COLOR_LM,  label=MODEL_LM)
ax_left.legend(
    handles=[patch_vlm, patch_lm],
    fontsize=FONTSIZE - 1,
    loc="lower right",
    framealpha=0.85,
    edgecolor="none",
)

# ── Right panel: difference bars ──────────────────────────────────────────────
ax_right.set_facecolor(BG_COLOR)
ax_right.grid(axis="x", color="white", linewidth=0.8, zorder=0)

bar_colors = [COLOR_POS if d >= 0 else COLOR_NEG for d in diffs]
ax_right.barh(y, diffs, height=0.55, color=bar_colors, zorder=3)

# Vertical line at x=0
ax_right.axvline(0, color="#444444", linewidth=0.8, zorder=4)

# Numeric labels on each bar
x_pad = 0.003
for i, d in enumerate(diffs):
    label = f"{d:+.3f}"
    ha = "left" if d >= 0 else "right"
    offset = x_pad if d >= 0 else -x_pad
    ax_right.text(
        d + offset, y[i], label,
        va="center", ha=ha, fontsize=FONTSIZE - 1.5,
    )

ax_right.set_yticks(y)
ax_right.set_yticklabels(ents_sorted, fontsize=FONTSIZE)
ax_right.set_xlabel("LM − VLM", fontsize=FONTSIZE)
ax_right.tick_params(axis="x", labelsize=FONTSIZE)
ax_right.set_title("Difference (LM − VLM)", fontsize=FONTSIZE, pad=4)

# Patch legend for right panel
patch_pos = mpatches.Patch(color=COLOR_POS, label="LM better")
patch_neg = mpatches.Patch(color=COLOR_NEG, label="VLM better")
ax_right.legend(
    handles=[patch_neg, patch_pos],
    fontsize=FONTSIZE - 1,
    loc="lower right",
    framealpha=0.85,
    edgecolor="none",
)

# ── save ───────────────────────────────────────────────────────────────────────
OUT_NAME = "internvl_entity_vlm_lm.png"

out_path_main = OUT_DIR / OUT_NAME
fig.savefig(out_path_main, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_path_main}")

latex_dir = REPO / "latex/AAAI2026/LaTeX/figures/internvl_entity_vlm_lm"
latex_dir.mkdir(parents=True, exist_ok=True)
out_path_latex = latex_dir / OUT_NAME
fig.savefig(out_path_latex, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path_latex}")

plt.close(fig)
print("Done.")
