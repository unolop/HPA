"""
Human participant statistics figures for supplementary.

Three individual figures:
  human_study_statistics_a.png  — A. Response time distribution
  human_study_statistics_c.png  — C. Answer length
  human_study_statistics_f.png  — F. Per-participant mean confidence
"""

from pathlib import Path
import json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
HUMAN_DIR = ROOT / "evaluation" / "humans" / "by_participant"
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

VARIANT_ORDER = ["C", "B", "A"]
VARIANT_LABELS = {"C": "C (Original)", "B": "B (Weaker)", "A": "A (Pronominalized)"}
from utils.constants import VARIANT_COLORS

BAR_BLUE = "#5c85d6"

# ── Load all responses ─────────────────────────────────────────────────────────
records = []
for fpath in sorted(HUMAN_DIR.glob("*.json")):
    d = json.load(open(fpath))
    pid = d.get("code", fpath.stem)
    for ans in d.get("answers", []):
        records.append({
            "participant": pid,
            "variant": ans["variant"],
            "response": ans["answer_text"],
            "confidence": ans["confidence"],
            "time_ms": ans["time_spent_ms"],
        })

df = pd.DataFrame(records)
df["resp"] = df["response"].fillna("").str.strip().str.lower()
df["word_count"] = df["resp"].str.split().str.len().fillna(0).astype(int)

SOFT_RE = re.compile(
    r"\b(don'?t know|cannot|can'?t tell|no idea|not sure|unsure|"
    r"unknown|unanswerable|nothing|nowhere|none|idk|n/a)\b|^(idk|dk|na|-)$",
    re.IGNORECASE,
)
df["soft_abstain"] = df["resp"].apply(lambda s: bool(SOFT_RE.search(s)))

p99 = df["time_ms"].quantile(0.99)
df["time_s_cap"] = df["time_ms"].clip(upper=p99) / 1000

pp = df.groupby("participant").agg(
    median_rt=("time_s_cap", "median"),
    mean_conf=("confidence", "mean"),
).reset_index().sort_values("median_rt")


FIG_W = 4.5
FIG_H = 3.2

def save(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── A: Response time distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.hist(df["time_s_cap"], bins=40, color=BAR_BLUE, edgecolor="white",
        linewidth=0.4, density=True)
ax.axvline(df["time_s_cap"].median(), color="#c0392b", lw=1.2,
           label=f"Median = {df['time_s_cap'].median():.1f}s")
ax.axvline(df["time_s_cap"].mean(), color="#e67e22", lw=1.2, ls="--",
           label=f"Mean = {df['time_s_cap'].mean():.1f}s")
ax.set_xlabel("Response time (s, capped at 99th pct)")
ax.set_ylabel("Density")
ax.legend(frameon=False, fontsize=7)
ax.text(0.97, 0.97, f"N={len(df):,}", transform=ax.transAxes,
        ha="right", va="top", fontsize=7)
save(fig, "human_study_statistics_a.png")


# ── C: Word count distribution ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
wc_counts = df["word_count"].value_counts().sort_index()
wc_pct = wc_counts / wc_counts.sum() * 100
top_bins = wc_pct[wc_pct.index <= 6]
rest = wc_pct[wc_pct.index > 6].sum()
xs = list(top_bins.index) + [7]
ys = list(top_bins.values) + [rest]
xlabels = [str(x) for x in top_bins.index] + ["7+"]
ax.bar(range(len(xs)), ys, color=BAR_BLUE, edgecolor="white", linewidth=0.4)
ax.set_xticks(range(len(xs)))
ax.set_xticklabels(xlabels)
ax.set_xlabel("Word count per answer")
ax.set_ylabel("% of responses")
mean_wc = df["word_count"].mean()
ax.text(0.97, 0.97, f"Mean = {mean_wc:.2f} words\n(Median = 1)",
        transform=ax.transAxes, ha="right", va="top", fontsize=7)
save(fig, "human_study_statistics_c.png")


# ── F: Per-participant mean confidence (sorted) ────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
pp_conf = df.groupby("participant")["confidence"].mean().sort_values().values
ax.bar(range(len(pp_conf)), pp_conf, color=BAR_BLUE, edgecolor="none", width=0.8)
ax.axhline(pp_conf.mean(), color="#c0392b", lw=1.0, ls="--",
           label=f"Mean = {pp_conf.mean():.2f}")
ax.set_xlabel("Participant (sorted by mean conf.)")
ax.set_ylabel("Mean confidence (1–5)")
ax.set_xticks([])
ax.legend(frameon=False, fontsize=7)
ax.set_ylim(1, 5)
save(fig, "human_study_statistics_f.png")

# ── Print summary stats ────────────────────────────────────────────────────────
print(f"\n=== Summary ===")
print(f"N responses: {len(df):,}  |  Participants: {df['participant'].nunique()}")
print(f"Median RT: {df['time_s_cap'].median():.1f}s  |  Mean RT: {df['time_s_cap'].mean():.1f}s")
overall_abst = df["soft_abstain"].mean() * 100
abst_by_v = df.groupby("variant")["soft_abstain"].mean() * 100
print(f"Soft abstention overall: {overall_abst:.3f}%")
print(f"  By variant: C={abst_by_v['C']:.3f}%  B={abst_by_v['B']:.3f}%  A={abst_by_v['A']:.3f}%")
print(f"Mean word count: {df['word_count'].mean():.2f}  |  1-word responses: {(df['word_count']==1).mean()*100:.1f}%")
print(f"Mean confidence: {df['confidence'].mean():.2f}  |  Std: {df['confidence'].std():.2f}")
