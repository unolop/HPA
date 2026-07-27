"""
Generate conf_rt_sbert_triangle.png:
  Left panel:  per-question mean confidence vs HH SBERT (by variant)
  Right panel: per-question mean RT (s)   vs HH SBERT (by variant)

Run from repo root:
  ~/miniconda3/envs/zero/bin/python figures/confidence_human/conf_rt_sbert_triangle.py
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
LATEX_DIR  = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "confidence"
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

# ── load human confidence + RT (per participant × question × variant) ────────
records = []
for fpath in sorted(HUMAN_DIR.glob("*.json")):
    d = json.load(open(fpath))
    for ans in d.get("answers", []):
        records.append({
            "participant":   d.get("code", fpath.stem),
            "question_id":   ans["question_id"],
            "variant":       ans["variant"],      # A/B/C
            "confidence":    (ans["confidence"] - 1) / 4.0,   # Likert 1-5 → [0,1]
            "time_s":        ans["time_spent_ms"] / 1000.0,
        })
human_df = pd.DataFrame(records)

# cap RT at 99th percentile
p99 = human_df["time_s"].quantile(0.99)
human_df["time_s"] = human_df["time_s"].clip(upper=p99)

# per-question × variant aggregates
human_qv = (
    human_df.groupby(["question_id", "variant"])
    .agg(mean_conf=("confidence", "mean"), mean_rt=("time_s", "mean"))
    .reset_index()
)

# ── load HH SBERT (pair_cache_cleaned.parquet) ────────────────────────────────
pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
hh = pc[pc["pair_type"] == "HH"].copy()
hh_qv = (
    hh[hh["variant"].isin(["A", "B", "C"])]
    .groupby(["question_id", "variant"])["sbert_score"]
    .mean()
    .reset_index()
    .rename(columns={"sbert_score": "hh_sbert"})
)

# ── merge ─────────────────────────────────────────────────────────────────────
merged = human_qv.merge(hh_qv, on=["question_id", "variant"], how="inner")
print(f"Merged rows: {len(merged)} | questions: {merged['question_id'].nunique()}")


def sig_stars(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def variant_r_text(df: pd.DataFrame, xcol: str) -> str:
    lines = []
    for variant in VARIANT_ORDER:
        sub = df[df["variant"] == variant]
        x = sub[xcol].to_numpy()
        y = sub["hh_sbert"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) >= 3:
            r_val, p_val = stats.pearsonr(x, y)
            lines.append(f"r({VARIANT_SHORT[variant]})={r_val:.2f}{sig_stars(p_val)}")
    return "\n".join(lines)

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_conf, ax_rt) = plt.subplots(
    1, 2,
    figsize=(5.0, 2.4),
    sharey=True,
    gridspec_kw={"wspace": 0.08},
)

for variant in VARIANT_ORDER:
    sub = merged[merged["variant"] == variant]
    color = VARIANT_COLORS[variant]
    label = VARIANT_LABELS[variant]

    for ax, xcol, xlabel in [
        (ax_conf, "mean_conf", "Mean confidence"),
        (ax_rt,   "mean_rt",   "Mean RT (s)"),
    ]:
        x = sub[xcol].values
        y = sub["hh_sbert"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        ax.scatter(x, y, s=12, alpha=0.45, color=color, edgecolors="none",
                   label=label if ax is ax_conf else None)

        if len(x) >= 5:
            slope, intercept, *_ = stats.linregress(x, y)
            xfit = np.linspace(x.min(), x.max(), 100)
            ax.plot(xfit, intercept + slope * xfit, color=color, lw=1.6)

for ax, xlabel in [(ax_conf, "Mean confidence"), (ax_rt, "Mean RT (s)")]:
    ax.set_xlabel(xlabel)

ax_conf.set_ylabel("HH SBERT agreement")
ax_rt.set_ylabel("")

ax_conf.text(
    0.03, 0.08, variant_r_text(merged, "mean_conf"),
    transform=ax_conf.transAxes,
    ha="left", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86, linewidth=0.0),
)
ax_rt.text(
    0.03, 0.08, variant_r_text(merged, "mean_rt"),
    transform=ax_rt.transAxes,
    ha="left", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86, linewidth=0.0),
)

# shared legend above both panels
handles, labels = ax_conf.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.50, 1.03),
    markerscale=2.0,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])

out = OUT_DIR / "conf_rt_sbert_triangle.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
shutil.copy(out, LATEX_DIR / "conf_rt_sbert_triangle.png")
plt.close(fig)
print(f"Saved: {out}")
print(f"Copied to: {LATEX_DIR / 'conf_rt_sbert_triangle.png'}")
