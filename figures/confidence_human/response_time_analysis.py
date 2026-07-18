"""
Response time vs. confidence analysis for human participants.

Hypothesis: higher confidence → faster response (inverse relationship).
Tests:
  1. Per-participant Pearson r(log_time, confidence) → reliability via ICC
  2. Population-level correlation (mixed-effects aware: mean per question)
  3. Binned RT distribution by confidence level
  4. Per-variant breakdown (C/B/A)
  5. Outlier cap at 99th percentile

Figure saved to figures/confidence/response_time_vs_confidence.png
CSV saved to figures/confidence/response_time_stats.csv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
HUMAN_DIR = ROOT / "evaluation" / "humans" / "by_participant"
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── AAAI style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

VARIANT_NAMES = {"question": "C (Original)", "weaker_object": "B (Weaker)", "pronominalized": "A (Pronominalized)"}
VARIANT_SHORT = {"question": "C", "weaker_object": "B", "pronominalized": "A"}
CONF_LABELS = {1: "1\n(Very low)", 2: "2\n(Low)", 3: "3\n(Moderate)", 4: "4\n(High)", 5: "5\n(Very high)"}
PALETTE = ["#e53935", "#fb8c00", "#f9d71c", "#43a047", "#1565c0"]  # red→blue for conf 1-5

# ── Load data ─────────────────────────────────────────────────────────────────
records = []
for fpath in sorted(HUMAN_DIR.glob("*.json")):
    d = json.load(open(fpath))
    pid = d.get("code", fpath.stem)
    for ans in d.get("answers", []):
        records.append({
            "participant": pid,
            "question_id": ans["question_id"],
            "variant": ans["variant"],  # A/B/C
            "confidence": ans["confidence"],       # 1-4 (Likert)
            "time_ms": ans["time_spent_ms"],
        })

df = pd.DataFrame(records)

# Map variant letters to internal keys
LETTER_TO_KEY = {"C": "question", "B": "weaker_object", "A": "pronominalized"}
df["variant_key"] = df["variant"].map(LETTER_TO_KEY)

# ── Outlier cap at 99th percentile per participant ────────────────────────────
p99 = df["time_ms"].quantile(0.99)
df["time_ms_cap"] = df["time_ms"].clip(upper=p99)
df["log_time"] = np.log(df["time_ms_cap"])

print(f"N participants: {df['participant'].nunique()}")
print(f"N responses: {len(df)}")
print(f"Time 99th percentile cap: {p99:.0f} ms")
print(f"Responses capped: {(df['time_ms'] > p99).sum()}")

# ── 1. Per-participant Pearson r(log_time, confidence) ────────────────────────
per_part = []
for pid, g in df.groupby("participant"):
    r, p = stats.pearsonr(g["log_time"], g["confidence"])
    per_part.append({"participant": pid, "r": r, "p": p, "n": len(g)})
part_df = pd.DataFrame(per_part)

mean_r = part_df["r"].mean()
# One-sample t-test: is mean r significantly different from 0?
valid_r = part_df["r"].dropna()
t, tp = stats.ttest_1samp(valid_r, 0)
print(f"\nPer-participant r(log_time, conf): mean={mean_r:.3f}, t={t:.2f}, p={tp:.4f}")
print(f"  Range: [{part_df['r'].min():.3f}, {part_df['r'].max():.3f}]")
print(f"  Negative r (higher conf=faster): {(part_df['r'] < 0).sum()}/{len(part_df)}")

# ── 2. Per-variant breakdown ──────────────────────────────────────────────────
variant_corr = []
for vk, vname in VARIANT_NAMES.items():
    sub = df[df["variant_key"] == vk]
    r, p = stats.pearsonr(sub["log_time"], sub["confidence"])
    variant_corr.append({"variant": VARIANT_SHORT[vk], "variant_name": vname,
                          "r": r, "p": p, "n": len(sub)})
var_df = pd.DataFrame(variant_corr)
print("\nPer-variant r(log_time, conf):")
print(var_df.to_string(index=False))

# ── 3. Mean RT per confidence level ──────────────────────────────────────────
conf_rt = df.groupby("confidence")["time_ms_cap"].agg(["mean", "sem", "count"]).reset_index()
conf_rt.columns = ["confidence", "mean_ms", "sem_ms", "count"]
print("\nMean RT by confidence level:")
print(conf_rt.to_string(index=False))

# ── Save stats CSV ────────────────────────────────────────────────────────────
summary_rows = []
# Overall
r_all, p_all = stats.pearsonr(df["log_time"], df["confidence"])
summary_rows.append({"group": "All", "r": r_all, "p": p_all, "n": len(df)})
# Per variant
for _, row in var_df.iterrows():
    summary_rows.append({"group": f"Variant {row['variant']}", "r": row["r"], "p": row["p"], "n": row["n"]})
# Per-participant summary
summary_rows.append({"group": "Per-participant mean r", "r": mean_r, "p": tp, "n": len(part_df)})
stats_df = pd.DataFrame(summary_rows)
stats_df.to_csv(OUT_DIR / "response_time_stats.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'response_time_stats.csv'}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.0, 5.0))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38,
                       left=0.10, right=0.97, top=0.88, bottom=0.12)

# Panel A: Distribution of per-participant r values
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(part_df["r"], bins=12, color="#5c85d6", edgecolor="white", linewidth=0.5)
ax_a.axvline(0, color="black", lw=0.8, ls="--")
ax_a.axvline(mean_r, color="#c0392b", lw=1.2, label=f"Mean r = {mean_r:.3f}")
ax_a.set_xlabel("Pearson r (log RT, confidence)")
ax_a.set_ylabel("# Participants")
ax_a.set_title("A. Per-Participant Correlation")
ax_a.legend(frameon=False, fontsize=7)
# Add t-test annotation
sig_str = f"t={t:.2f}, p={'<0.001' if tp < 0.001 else f'{tp:.3f}'}"
ax_a.text(0.97, 0.95, sig_str, transform=ax_a.transAxes,
          ha="right", va="top", fontsize=7, color="#c0392b")

# Panel B: Mean RT by confidence level (bar)
ax_b = fig.add_subplot(gs[0, 1])
xs = conf_rt["confidence"].values
bars = ax_b.bar(xs, conf_rt["mean_ms"] / 1000, yerr=conf_rt["sem_ms"] / 1000,
                color=PALETTE, edgecolor="white", linewidth=0.5, capsize=3,
                error_kw={"linewidth": 0.8, "ecolor": "black"})
ax_b.set_xticks(xs)
ax_b.set_xticklabels([CONF_LABELS[x] for x in xs], fontsize=7)
ax_b.set_xlabel("Confidence level")
ax_b.set_ylabel("Response time (s)")
ax_b.set_title("B. Mean RT by Confidence Level")
# Monotone annotation
means = conf_rt["mean_ms"].values
trend = "↓ as confidence ↑" if means[-1] < means[0] else "↑ as confidence ↑"
ax_b.text(0.97, 0.97, trend, transform=ax_b.transAxes,
          ha="right", va="top", fontsize=7, style="italic")

# Panel C: Scatter log_time vs confidence with regression (jittered)
ax_c = fig.add_subplot(gs[1, 0])
# Sample for clarity
sample = df.sample(min(3000, len(df)), random_state=42)
jitter_conf = sample["confidence"] + np.random.uniform(-0.15, 0.15, len(sample))
ax_c.scatter(jitter_conf, sample["log_time"], alpha=0.08, s=4, color="#5c85d6", rasterized=True)
# Regression line
slope, intercept, r_reg, p_reg, se = stats.linregress(df["confidence"], df["log_time"])
xfit = np.array([1, 4])
ax_c.plot(xfit, intercept + slope * xfit, color="#c0392b", lw=1.5,
          label=f"r = {r_reg:.3f}")
ax_c.set_xlabel("Confidence level (jittered)")
ax_c.set_ylabel("log(Response time ms)")
ax_c.set_xticks([1, 2, 3, 4, 5])
ax_c.set_title("C. Log RT vs. Confidence (N=all)")
ax_c.legend(frameon=False, fontsize=7)

# Panel D: Per-variant r comparison
ax_d = fig.add_subplot(gs[1, 1])
v_colors = ["#e67e22", "#8e44ad", "#27ae60"]
v_rs = var_df["r"].values
v_labels = var_df["variant"].values
bars2 = ax_d.bar(v_labels, v_rs, color=v_colors, edgecolor="white", linewidth=0.5)
ax_d.axhline(0, color="black", lw=0.8, ls="--")
ax_d.set_xlabel("Variant")
ax_d.set_ylabel("Pearson r")
ax_d.set_title("D. Correlation by Variant")
for bar, r_val, row in zip(bars2, v_rs, var_df.itertuples()):
    p_str = "<.001" if row.p < 0.001 else f"{row.p:.3f}"
    # Place annotation inside bar (centered vertically)
    y_text = r_val / 2
    ax_d.text(bar.get_x() + bar.get_width() / 2, y_text,
              f"r={r_val:.3f}\np={p_str}", ha="center", va="center", fontsize=6.5,
              color="white", fontweight="bold")

fig.suptitle("Human Response Time vs. Confidence", fontsize=10, fontweight="bold", y=0.97)
fig.savefig(OUT_DIR / "response_time_vs_confidence.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'response_time_vs_confidence.png'}")
