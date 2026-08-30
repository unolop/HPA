"""
think_abs_vs_length.py

Generates:
  think_abs_sbert_accuracy.png
  think_abs_confidence.png

Each panel: x = mean think words, y = absolute metric value.
Marker size = model scale, filled = w/ inst, open = w/o inst.
Dotted arrows show blind → inst_blind transition.
HH baseline bands added to SBERT and accuracy panels.

Run from repo root:
  conda run -n zero python figures/think_eda/think_abs_vs_length.py
"""
from __future__ import annotations
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OUT_DIR    = Path(__file__).parent
LATEX_OUT  = Path(__file__).resolve().parents[2] / "latex/AAAI2026/LaTeX/figures_supp/think_eda"
LATEX_OUT.mkdir(parents=True, exist_ok=True)
CSV        = OUT_DIR / "think_delta_vs_length_with_blind.csv"

VARIANT_ORDER  = ["C", "B", "A"]
VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
VARIANT_COLORS = {"C": "#2a4e6e", "B": "#678cac", "A": "#bdccde"}

SIZE_MARKER = {0.6: 20, 1.7: 38, 4.0: 68, 8.0: 105, 32.0: 160}
SIZE_LABEL  = {0.6: "0.6B", 1.7: "1.7B", 4.0: "4B", 8.0: "8B", 32.0: "32B"}

# Human LOO CIs (free-text, abstention-filtered, p5–p95)
import json as _json
_hh_ci = _json.loads((OUT_DIR / "hh_ci_freetext.json").read_text())
HH_SBERT_BAND    = (_hh_ci["sbert_lo"],   _hh_ci["sbert_hi"])
HH_ACCURACY_BAND = (_hh_ci["accuracy_lo"], _hh_ci["accuracy_hi"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df = pd.read_csv(CSV)
df_valid = df[df["mean_think_words"] > 0].copy()

PANEL_SPECS = [
    ("think_hm_sbert",   "HM SBERT",            "think_trace_sbert.png",      HH_SBERT_BAND),
    ("think_confidence", "Mean answer logprob",  "think_trace_confidence.png", None),
]

# Build shared legend handles
variant_handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=VARIANT_COLORS[v], markeredgecolor=VARIANT_COLORS[v],
           markersize=7, label=VARIANT_LABELS[v])
    for v in VARIANT_ORDER
]
# Size legend handles (marker area → display radius via sqrt)
_SIZE_LEGEND = [(0.6, "0.6B"), (4.0, "4B"), (8.0, "8B"), (32.0, "32B")]
size_handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor="#999", markeredgecolor="#999",
           markersize=(SIZE_MARKER[s] ** 0.5) * 0.85, label=lbl)
    for s, lbl in _SIZE_LEGEND
]
def plot_metric_panel(ax, metric: str, ylabel: str, hh_band: tuple[float, float] | None) -> None:
    # HH baseline band (human LOO p5–p95 CI, free-text)
    if hh_band is not None:
        lo, hi = hh_band
        ax.axhspan(lo, hi, color="#bbb", alpha=0.18, zorder=0)
        ax.axhline(lo, color="#999", lw=0.9, ls="--", zorder=1)
        ax.axhline(hi, color="#999", lw=0.9, ls="--", zorder=1)

    # Dotted arrows: blind → inst_blind
    for size_b, g in df_valid.groupby("size_b"):
        conds = set(g["condition"].unique())
        if "blind" not in conds or "inst_blind" not in conds:
            continue
        for variant in VARIANT_ORDER:
            row_b = g[(g["condition"] == "blind")      & (g["variant"] == variant)]
            row_i = g[(g["condition"] == "inst_blind") & (g["variant"] == variant)]
            if row_b.empty or row_i.empty:
                continue
            x0, y0 = float(row_b["mean_think_words"].iloc[0]), float(row_b[metric].iloc[0])
            x1, y1 = float(row_i["mean_think_words"].iloc[0]), float(row_i[metric].iloc[0])
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", linestyle="dotted",
                    color=VARIANT_COLORS[variant],
                    lw=2.4, mutation_scale=12,
                ),
                zorder=2,
            )

    # Scatter points
    for _, row in df_valid.iterrows():
        c      = VARIANT_COLORS[row["variant"]]
        filled = row["condition"] == "inst_blind"
        ms     = SIZE_MARKER.get(row["size_b"], 60)
        ax.scatter(
            row["mean_think_words"], row[metric],
            s=ms,
            facecolors=c if filled else "none",
            edgecolors=c,
            linewidths=1.2,
            zorder=3,
        )

    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)
    ax.grid(axis="x", alpha=0.08)
    # Add top padding so highest points are not clipped
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.06)
    # Add right x-axis padding
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(xlo, xhi + (xhi - xlo) * 0.12)

fig, (ax_sbert, ax_acc) = plt.subplots(
    2, 1,
    figsize=(4.35, 4.35),
    sharex=True,
    gridspec_kw={"hspace": 0.08},
)
plot_metric_panel(ax_sbert, "think_hm_sbert", "HM SBERT", HH_SBERT_BAND)
plot_metric_panel(ax_acc, "think_accuracy", "Accuracy", HH_ACCURACY_BAND)
ax_acc.set_xlabel("Mean think words")

fig.legend(
        handles=variant_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=0.9,
)
fig.tight_layout(rect=[0.06, 0.08, 1, 0.98])

combo_out = OUT_DIR / "think_abs_sbert_accuracy.png"
fig.savefig(combo_out, dpi=200, bbox_inches="tight")
shutil.copy(combo_out, LATEX_OUT / "think_abs_sbert_accuracy.png")
plt.close(fig)
print(f"Saved: {combo_out}")

# Horizontal variant: 3 panels, slightly wider, half height
fig, (ax_sbert, ax_acc, ax_conf) = plt.subplots(
    1, 3,
    figsize=(6.5, 1.8),
    sharey=False,
    gridspec_kw={"wspace": 0.55},
)
plot_metric_panel(ax_sbert, "think_hm_sbert",   "HM SBERT",          HH_SBERT_BAND)
plot_metric_panel(ax_acc,   "think_accuracy",    "Accuracy",          HH_ACCURACY_BAND)
plot_metric_panel(ax_conf,  "think_confidence",  "Mean answer logprob", None)
ax_sbert.set_xlabel("Mean think words")
ax_acc.set_xlabel("Mean think words")
ax_conf.set_xlabel("Mean think words")
for ax, label in zip([ax_sbert, ax_acc, ax_conf], ["(a)", "(b)", "(c)"]):
    ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")

fig.legend(
        handles=variant_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=0.9,
)
fig.tight_layout(rect=[0, 0.06, 1, 0.92])

horiz_out = OUT_DIR / "think_abs_sbert_accuracy_horiz.png"
fig.savefig(horiz_out, dpi=200, bbox_inches="tight")
shutil.copy(horiz_out, LATEX_OUT / "think_abs_sbert_accuracy_horiz.png")
plt.close(fig)
print(f"Saved: {horiz_out}")

# Vertical variant: sHM + confidence, for appendix
fig, (ax_sbert, ax_conf) = plt.subplots(
    2, 1,
    figsize=(3.2, 4.0),
    sharex=True,
    gridspec_kw={"hspace": 0.10},
)
plot_metric_panel(ax_sbert, "think_hm_sbert",   "HM SBERT",            HH_SBERT_BAND)
plot_metric_panel(ax_conf,  "think_confidence",  "Mean answer logprob", None)
ax_sbert.set_xlabel("")
ax_sbert.tick_params(labelbottom=False)
ax_conf.set_xlabel("Mean think words")

# Variant legend (top)
fig.legend(
        handles=variant_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=0.9,
)
# Size legend (bottom, below x-axis label)
fig.legend(
        handles=size_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=0.7,
)
fig.tight_layout(rect=[0.04, 0.04, 1, 0.95])

vert_out = OUT_DIR / "think_abs_sbert_confidence_vert.png"
fig.savefig(vert_out, dpi=200, bbox_inches="tight")
shutil.copy(vert_out, LATEX_OUT / "think_abs_sbert_confidence_vert.png")
plt.close(fig)
print(f"Saved: {vert_out}")

for metric, ylabel, fname, hh_vals in PANEL_SPECS:
    show_scale = (metric == "think_confidence")
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.2))
    plot_metric_panel(ax, metric, ylabel, hh_vals)
    if show_scale:
        # Confidence: xlabel + scale legend at bottom only
        ax.set_xlabel("Mean think words")
        fig.legend(
            handles=size_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            frameon=False,
            fontsize=7.5,
            handletextpad=0.4,
            columnspacing=0.7,
        )
        fig.tight_layout(rect=[0.06, 0.04, 1, 0.97])
    else:
        # sHM: variant legend on top, no xlabel
        ax.tick_params(labelbottom=False)
        fig.legend(
            handles=variant_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
            fontsize=7.5,
            handletextpad=0.4,
            columnspacing=0.9,
        )
        fig.tight_layout(rect=[0.06, 0, 1, 0.93])

    out = OUT_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    shutil.copy(out, LATEX_OUT / fname)
    plt.close(fig)
    print(f"Saved: {out}")

print("Done.")
