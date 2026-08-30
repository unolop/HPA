"""
Generate rt_sbert_scatter.png:
  Per-question mean RT (s) vs HH SBERT agreement, colored by variant,
  with Pearson r values annotated.

Run from repo root:
  ~/miniconda3/envs/zero/bin/python figures/confidence_human/rt_sbert_scatter.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from utils.constants import VARIANT_COLORS, VARIANT_ORDER

HUMAN_DIR  = ROOT / "evaluation" / "humans" / "by_participant"
EXPORTS    = ROOT / "analysis" / "session2" / "exports"
OUT_DIR    = Path(__file__).parent
LATEX_DIR  = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures_paper" / "confidence"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
VARIANT_SHORT  = {"C": "Orig", "B": "Weak", "A": "Pron"}

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "axes.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# ── load human RT (per participant × question × variant) ─────────────────────
records = []
for fpath in sorted(HUMAN_DIR.glob("*.json")):
    d = json.load(open(fpath))
    for ans in d.get("answers", []):
        records.append({
            "participant":   d.get("code", fpath.stem),
            "question_id":   ans["question_id"],
            "variant":       ans["variant"],
            "confidence":    (ans["confidence"] - 1) / 4.0,
            "time_s":        ans["time_spent_ms"] / 1000.0,
        })
human_df = pd.DataFrame(records)

# cap RT at 99th percentile
p99 = human_df["time_s"].quantile(0.99)
human_df["time_s"] = human_df["time_s"].clip(upper=p99)

# per-question × variant mean RT and confidence
human_qv = (
    human_df.groupby(["question_id", "variant"])
    .agg(mean_rt=("time_s", "mean"), mean_conf=("confidence", "mean"))
    .reset_index()
)

# ── load HH SBERT (pair_cache_cleaned.parquet) ───────────────────────────────
pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
hh = pc[pc["pair_type"] == "HH"].copy()
hh_qv = (
    hh[hh["variant"].isin(["A", "B", "C"])]
    .groupby(["question_id", "variant"])["sbert_score"]
    .mean()
    .reset_index()
    .rename(columns={"sbert_score": "hh_sbert"})
)

# ── merge ────────────────────────────────────────────────────────────────────
merged = human_qv.merge(hh_qv, on=["question_id", "variant"], how="inner")
print(f"Merged rows: {len(merged)} | questions: {merged['question_id'].nunique()}")


def sig_stars(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.8))

for variant in VARIANT_ORDER:
    sub = merged[merged["variant"] == variant]
    color = VARIANT_COLORS[variant]
    label = VARIANT_LABELS[variant]

    x = sub["mean_rt"].values
    y = sub["mean_conf"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    ax.scatter(x, y, s=12, alpha=0.45, color=color, edgecolors="none", label=label)

    if len(x) >= 5:
        slope, intercept, *_ = stats.linregress(x, y)
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, intercept + slope * xfit, color=color, lw=1.6)

# overall regression line (pooled across variants)
x_all = merged["mean_rt"].values
y_all = merged["mean_conf"].values
mask_all = np.isfinite(x_all) & np.isfinite(y_all)
x_all, y_all = x_all[mask_all], y_all[mask_all]
slope_all, intercept_all, *_ = stats.linregress(x_all, y_all)
xfit_all = np.linspace(x_all.min(), x_all.max(), 100)
ax.plot(xfit_all, intercept_all + slope_all * xfit_all, color="black", lw=1.8, ls="--")

ax.set_xlabel("Mean RT (s)")
ax.set_ylabel("Mean confidence")

# r-value annotation
r_all, p_all = stats.pearsonr(x_all, y_all)
lines = [f"r(All)={r_all:.2f}{sig_stars(p_all)}"]
for variant in VARIANT_ORDER:
    sub = merged[merged["variant"] == variant]
    x = sub["mean_rt"].to_numpy()
    y = sub["mean_conf"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) >= 3:
        r_val, p_val = stats.pearsonr(x, y)
        lines.append(f"r({VARIANT_SHORT[variant]})={r_val:.2f}{sig_stars(p_val)}")

ax.text(
    0.97, 0.97, "\n".join(lines),
    transform=ax.transAxes,
    ha="right", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86, linewidth=0.0),
)

plt.tight_layout(pad=0.3)

out = OUT_DIR / "rt_sbert_scatter.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
shutil.copy(out, LATEX_DIR / "rt_sbert_scatter.png")
plt.close(fig)
print(f"Saved: {out}")
print(f"Copied to: {LATEX_DIR / 'rt_sbert_scatter.png'}")
