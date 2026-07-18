"""
Abstention scatter: blind vs inst_blind overlaid on same axes (q113 subset).
Filled markers = inst_blind  |  Hollow markers = blind (no instruction)
2×2 layout: VLMs | Backbone Decoders / Standalone LLMs | Think Models

Data source: responses_model_blind.csv / responses_model_inst_blind.csv
(113-question human-study subset, variant C)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent
LATEX_OUT = ROOT / "latex/AAAI2026/LaTeX/figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

GROUP_COLORS = {
    "VLM":                    "#E53935",
    "VLM backbone decoder":   "#E67E22",
    "standalone LLM":         "#2E7D32",
    "standalone LLM (think)": "#1E88E5",
}
FAMILY_MARKER = {
    "InternVL": "o", "Qwen3-VL": "X", "LLaVA-1.5": "D",
    "LLaVA-Mistral": "^", "LLaVA-Vicuna": "v",
    "Qwen3": "X", "Qwen3 (think)": "P",
    "Qwen2.5": "s", "Mistral": "^", "Vicuna": "v", "Phi": "h",
}
MODEL_SIZE_B = {
    "InternVL-1B": 1.0, "InternVL-2B": 2.0, "InternVL-8B": 8.0,
    "InternVL-1B (LM)": 1.0, "InternVL-2B (LM)": 2.0, "InternVL-8B (LM)": 8.0,
    "Qwen3-VL-2B": 2.0, "Qwen3-VL-4B": 4.0, "Qwen3-VL-8B": 8.0, "Qwen3-VL-32B": 32.0,
    "Qwen3-VL-2B (LM)": 2.0, "Qwen3-VL-4B (LM)": 4.0, "Qwen3-VL-8B (LM)": 8.0,
    "Qwen3-VL-32B (LM)": 32.0,
    "LLaVA-1.5-7B": 7.0, "LLaVA-Mistral": 7.0, "LLaVA-Vicuna": 7.0,
    "LLaVA-Vicuna-13B": 13.0,
    "LLaVA-1.5 (LM)": 7.0, "LLaVA-Mistral (LM)": 7.0,
    "LLaVA-Vicuna (LM)": 7.0, "LLaVA-Vicuna-13B (LM)": 13.0,
    "Qwen3-0.6B": 0.6, "Qwen3-1.7B": 1.7, "Qwen3-4B": 4.0, "Qwen3-8B": 8.0, "Qwen3-32B": 32.0,
    "Qwen3-0.6B (think)": 0.6, "Qwen3-1.7B (think)": 1.7,
    "Qwen3-4B (think)": 4.0, "Qwen3-8B (think)": 8.0, "Qwen3-32B (think)": 32.0,
    "Qwen2.5-7B-Instruct": 7.0, "Mistral-7B": 7.0,
    "Phi-3.5-mini": 3.8, "Vicuna-7B": 7.0, "Vicuna-13B": 13.0,
}

# Model group mapping (from model_group column in CSV, but map to our keys)
GROUP_MAP = {
    "VLM": "VLM",
    "VLM backbone decoder": "VLM backbone decoder",
    "standalone LLM": "standalone LLM",
    "standalone LLM (think)": "standalone LLM (think)",
}

def _get_family(name):
    n = name.replace(" (LM)", "").replace(" (think)", "").strip()
    if "InternVL" in n:      return "InternVL"
    if "Qwen3-VL" in n:      return "Qwen3-VL"
    if "LLaVA-1.5" in n:    return "LLaVA-1.5"
    if "LLaVA-Mistral" in n: return "LLaVA-Mistral"
    if "LLaVA-Vicuna" in n:  return "LLaVA-Vicuna"
    if "Qwen3" in n and "think" in name: return "Qwen3 (think)"
    if "Qwen3" in n:         return "Qwen3"
    if "Qwen2.5" in n:       return "Qwen2.5"
    if "Mistral" in n:       return "Mistral"
    if "Vicuna" in n:        return "Vicuna"
    if "Phi" in n:           return "Phi"
    return "Other"

SOFT_WORDS = {
    "nothing", "none", "unknown", "unclear", "unanswerable", "unsure",
    "uncertain", "impossible", "indeterminate", "unspecified", "unidentifiable",
    "invisible", "hidden", "unreadable", "illegible", "ambiguous", "n/a", "na",
    "blank", "empty",
}

def compute_abstention_from_csv(csv_path, variant="C"):
    df = pd.read_csv(csv_path)
    df = df[df["variant"] == variant].copy()
    df["abst"] = df["response"].apply(
        lambda a: pd.isna(a) or str(a).strip().lower() in SOFT_WORDS or str(a).strip() == ""
    )
    rates = df.groupby(["model", "model_group"])["abst"].mean().mul(100).reset_index()
    rates.columns = ["model", "group", "abst_pct"]
    return rates

# Load and compute
exports_dir = ROOT / "analysis/session2/exports"
inst_rates  = compute_abstention_from_csv(exports_dir / "responses_model_inst_blind.csv")
blind_rates = compute_abstention_from_csv(exports_dir / "responses_model_blind.csv")

N_Q = 113
print(f"inst_blind: {len(inst_rates)} model-group pairs")
print(f"blind: {len(blind_rates)} model-group pairs")

# 2×2 panel groups
PANELS = [
    ("A. Full VLMs",         "VLM",                    15.0),
    ("B. Backbone Decoders", "VLM backbone decoder",   10.0),
    ("C. Standalone LLMs",   "standalone LLM",         25.0),
    ("D. Think Models",      "standalone LLM (think)", 100.0),
]

max_size = 32.0
JITTER = {"inst": -0.04, "blind": +0.04}
size_ticks = [0.6, 1, 2, 4, 7, 8, 13, 32]

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
fig.subplots_adjust(left=0.10, right=0.74, top=0.90, bottom=0.10,
                    wspace=0.42, hspace=0.48)
axes_flat = axes.flatten()

for ax, (title, group_name, ymax) in zip(axes_flat, PANELS):
    color = GROUP_COLORS[group_name]

    for cond, rates_df, filled, jitter_key in [
        ("inst_blind", inst_rates,  True,  "inst"),
        ("blind",      blind_rates, False, "blind"),
    ]:
        sub = rates_df[rates_df["group"] == group_name]
        for _, row in sub.iterrows():
            name = row["model"]
            size = MODEL_SIZE_B.get(name)
            if size is None:
                continue
            fam = _get_family(name)
            mrk = FAMILY_MARKER.get(fam, "o")
            ms  = 32 + 55 * (np.log10(size + 0.3) / np.log10(max_size + 0.3)) ** 0.8
            x   = size * (1.0 + JITTER[jitter_key])
            if filled:
                ax.scatter(x, row["abst_pct"], s=ms, marker=mrk,
                           color=color, alpha=0.85, zorder=4,
                           edgecolors="white", linewidths=0.4)
            else:
                ax.scatter(x, row["abst_pct"], s=ms, marker=mrk,
                           facecolors="white", edgecolors=color,
                           linewidths=1.2, alpha=0.85, zorder=3)

    ax.axhline(0, color="#555", lw=0.9, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(size_ticks)
    ax.set_xticklabels(["0.6", "1", "2", "4", "7", "8", "13", "32"], fontsize=7)
    ax.set_xlim(0.4, 45)
    ax.set_ylim(-ymax * 0.06, ymax)
    ax.set_xlabel("Parameters (B)", fontsize=7.5)
    ax.set_ylabel("Total abstention rate (%)", fontsize=7.5)
    ax.set_title(title, fontweight="bold", fontsize=8.5, pad=4)

fig.suptitle(
    f"Abstention Rate vs. Scale: Blind vs. Instruction-Aware (N={N_Q} questions)",
    fontweight="bold", fontsize=9.0, y=0.97,
)

# Legends
cond_handles = [
    mlines.Line2D([], [], marker="o", color="#555", markersize=7,
                  markerfacecolor="#555", label="Inst-blind (filled)"),
    mlines.Line2D([], [], marker="o", color="#555", markersize=7,
                  markerfacecolor="white", markeredgewidth=1.2,
                  label="Blind (hollow)"),
]
family_order = ["InternVL", "Qwen3-VL", "LLaVA-1.5", "LLaVA-Mistral",
                "LLaVA-Vicuna", "Qwen3", "Qwen3 (think)", "Qwen2.5",
                "Mistral", "Vicuna", "Phi"]
family_handles = [
    mlines.Line2D([], [], marker=FAMILY_MARKER.get(f, "o"), color="#444",
                  markersize=7, linestyle="None", label=f)
    for f in family_order if f in FAMILY_MARKER
]

fig.legend(handles=cond_handles, title="Condition", frameon=True,
           fontsize=7, title_fontsize=7.5,
           bbox_to_anchor=(0.75, 0.99), loc="upper left", borderaxespad=0)
fig.legend(handles=family_handles, title="Family", frameon=True,
           fontsize=7, title_fontsize=7.5,
           bbox_to_anchor=(0.75, 0.72), loc="upper left", borderaxespad=0)

out = OUT_DIR / "abstention_blind_vs_inst_scatter_q113.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

import shutil
dst = LATEX_OUT / out.name
shutil.copy(out, dst)
print(f"Copied to: {dst}")
