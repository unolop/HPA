"""
Human participant statistics figures for supplementary.

Combined figure — human_study_statistics.png (panels A–F, 2 rows):
  Row 1: A. Response time distribution  B. Per-participant median RT  C. Answer length
  Row 2: D. Confidence by variant       E. Soft abstention rate       F. Per-participant confidence
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
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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


# ══════════════════════════════════════════════════════════════════════════════
# Combined figure: 2 rows × 3 cols (panels A–F)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.4),
                         gridspec_kw={"wspace": 0.42, "hspace": 0.52,
                                      "left": 0.09, "right": 0.97,
                                      "top": 0.94, "bottom": 0.10})

ax_a, ax_b, ax_c = axes[0]
ax_d, ax_e, ax_f = axes[1]

# A: Response time distribution
ax_a.hist(df["time_s_cap"], bins=40, color=BAR_BLUE, edgecolor="white",
          linewidth=0.4, density=True)
ax_a.axvline(df["time_s_cap"].median(), color="#c0392b", lw=1.2,
             label=f"Median = {df['time_s_cap'].median():.1f}s")
ax_a.axvline(df["time_s_cap"].mean(), color="#e67e22", lw=1.2, ls="--",
             label=f"Mean = {df['time_s_cap'].mean():.1f}s")
ax_a.set_xlabel("Response time (s, capped at 99th pct)")
ax_a.set_ylabel("Density")
ax_a.set_title("A. Response Time Distribution")
ax_a.legend(frameon=False, fontsize=7)
ax_a.text(0.97, 0.97, f"N={len(df):,}", transform=ax_a.transAxes,
          ha="right", va="top", fontsize=7)

# B: Per-participant median RT
y_pos = np.arange(len(pp))
ax_b.barh(y_pos, pp["median_rt"].values, color=BAR_BLUE,
          edgecolor="none", height=0.7)
ax_b.set_yticks([])
ax_b.set_xlabel("Median RT (s)")
ax_b.set_title("B. Per-Participant Median RT")
ax_b.axvline(pp["median_rt"].median(), color="#c0392b", lw=1.0, ls="--",
             label=f"Median={pp['median_rt'].median():.1f}s")
ax_b.legend(frameon=False, fontsize=7)
ax_b.set_xlim(0, pp["median_rt"].max() * 1.15)

# C: Word count distribution
wc_counts = df["word_count"].value_counts().sort_index()
wc_pct = wc_counts / wc_counts.sum() * 100
top_bins = wc_pct[wc_pct.index <= 6]
rest = wc_pct[wc_pct.index > 6].sum()
xs = list(top_bins.index) + [7]
ys = list(top_bins.values) + [rest]
xlabels = [str(x) for x in top_bins.index] + ["7+"]
ax_c.bar(range(len(xs)), ys, color=BAR_BLUE, edgecolor="white", linewidth=0.4)
ax_c.set_xticks(range(len(xs)))
ax_c.set_xticklabels(xlabels)
ax_c.set_xlabel("Word count per answer")
ax_c.set_ylabel("% of responses")
ax_c.set_title("C. Answer Length (Words)")
mean_wc = df["word_count"].mean()
ax_c.text(0.97, 0.97, f"Mean = {mean_wc:.2f} words\n(Median = 1)",
          transform=ax_c.transAxes, ha="right", va="top", fontsize=7)

# D: Confidence distribution by variant
conf_levels = sorted(df["confidence"].unique())
width = 0.25
offsets = [-0.25, 0, 0.25]
for offset, vk in zip(offsets, VARIANT_ORDER):
    sub = df[df["variant"] == vk]
    counts = sub["confidence"].value_counts(normalize=True).reindex(conf_levels, fill_value=0) * 100
    ax_d.bar([c + offset for c in conf_levels], counts.values,
             width=width, label=VARIANT_LABELS[vk], color=VARIANT_COLORS[vk],
             edgecolor="white", linewidth=0.3, alpha=0.9)
ax_d.set_xlabel("Confidence level (1–5)")
ax_d.set_ylabel("% of responses")
ax_d.set_title("D. Confidence by Variant")
ax_d.set_xticks(conf_levels)
ax_d.legend(frameon=False, fontsize=6.5)

# E: Soft abstention rate by variant
abst_by_v = df.groupby("variant")["soft_abstain"].mean() * 100
abst_by_v = abst_by_v.reindex(VARIANT_ORDER)
ax_e.bar(VARIANT_ORDER, abst_by_v.values,
         color=[VARIANT_COLORS[v] for v in VARIANT_ORDER],
         edgecolor="white", linewidth=0.4)
ax_e.set_xlabel("Variant")
ax_e.set_ylabel("Soft abstention rate (%)")
ax_e.set_title("E. Soft Abstention Rate")
overall_abst = df["soft_abstain"].mean() * 100
ax_e.text(0.5, 0.92, f"Overall: {overall_abst:.2f}%",
          transform=ax_e.transAxes, ha="center", va="top", fontsize=7.5,
          color="#c0392b")
for i, (v, val) in enumerate(zip(VARIANT_ORDER, abst_by_v.values)):
    ax_e.text(i, val + 0.002, f"{val:.2f}%", ha="center", va="bottom", fontsize=7)
ax_e.set_ylim(0, max(abst_by_v.values) * 3.5)

# F: Per-participant mean confidence (sorted)
pp_conf = df.groupby("participant")["confidence"].mean().sort_values().values
ax_f.bar(range(len(pp_conf)), pp_conf, color=BAR_BLUE,
         edgecolor="none", width=0.8)
ax_f.axhline(pp_conf.mean(), color="#c0392b", lw=1.0, ls="--",
             label=f"Mean = {pp_conf.mean():.2f}")
ax_f.set_xlabel("Participant (sorted by mean conf.)")
ax_f.set_ylabel("Mean confidence (1–5)")
ax_f.set_title("F. Per-Participant Confidence")
ax_f.set_xticks([])
ax_f.legend(frameon=False, fontsize=7)
ax_f.set_ylim(1, 5)

out = OUT_DIR / "human_study_statistics.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Print summary stats ────────────────────────────────────────────────────────
print(f"\n=== Summary ===")
print(f"N responses: {len(df):,}  |  Participants: {df['participant'].nunique()}")
print(f"Median RT: {df['time_s_cap'].median():.1f}s  |  Mean RT: {df['time_s_cap'].mean():.1f}s")
print(f"Soft abstention overall: {overall_abst:.3f}%")
print(f"  By variant: C={abst_by_v['C']:.3f}%  B={abst_by_v['B']:.3f}%  A={abst_by_v['A']:.3f}%")
print(f"Mean word count: {df['word_count'].mean():.2f}  |  1-word responses: {(df['word_count']==1).mean()*100:.1f}%")
print(f"Mean confidence: {df['confidence'].mean():.2f}  |  Std: {df['confidence'].std():.2f}")
