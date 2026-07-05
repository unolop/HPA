"""
Human vs Model confidence comparison figures for the paper.

Outputs (figures/confidence_comparison/):

  1. confidence_distribution_human_vs_model.png
     KDE overlay: human self-reported confidence vs model token probability
     Side-by-side per model group, human shown as reference in each panel.

  2. reliability_human_vs_model.png
     ECE-style reliability diagram: binned confidence vs empirical accuracy
     One curve per model group + one for humans.  Diagonal = perfect calibration.

Run from repo root:
  ~/miniconda3/envs/zero/bin/python figures/confidence_comparison.py
"""

from __future__ import annotations

import json
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from utils.constants import GROUP_COLORS, GROUP_ORDER
from utils.vqa import VQAAnswerMapper, vqa_accuracy

OUT_DIR = ROOT / "figures" / "confidence_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# AAAI formatting: min 9pt, Times/Helvetica, 300 dpi, no hairlines
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
})

HUMAN_COLOR = "#7B1FA2"  # purple — distinct from all group colors

# ── Load human confidence data ─────────────────────────────────────────────────
print("Loading human confidence data…")
human_files = sorted(glob.glob(str(ROOT / "evaluation/humans/by_participant/*.json")))
mapper = VQAAnswerMapper()

human_rows = []
for fp in human_files:
    with open(fp) as f:
        data = json.load(f)
    pid = data.get("code", Path(fp).stem)
    for ans in data.get("answers", []):
        qid = ans["question_id"]
        variant = ans.get("variant", "C")
        raw_conf = ans.get("confidence", 3)  # Likert 1-5
        conf_norm = (raw_conf - 1) / 4.0     # → [0, 1]
        answer_text = ans.get("answer_text", "")
        gt = mapper.get_answers(qid)
        acc = vqa_accuracy(answer_text, gt) if gt else 0.0
        human_rows.append({
            "question_id": qid,
            "participant": pid,
            "variant": variant,
            "confidence": conf_norm,
            "accuracy": acc,
        })

human_df = pd.DataFrame(human_rows)
# Restrict to variant C (original question) for fair comparison
human_df = human_df[human_df["variant"] == "C"].copy()
human_qids = set(human_df["question_id"].unique())
print(f"  Human: {len(human_df)} rows, {human_df['participant'].nunique()} participants, "
      f"{len(human_qids)} questions (variant C)")

# ── Load model confidence data ────────────────────────────────────────────────
print("Loading model confidence data…")

VLM_DIR_TO_MODEL = {
    "InternVL3_5-1B": ("InternVL-1B", "VLM"),
    "InternVL3_5-2B": ("InternVL-2B", "VLM"),
    "InternVL3_5-8B": ("InternVL-8B", "VLM"),
    "llava-1.5-7b-hf": ("LLaVA-1.5-7B", "VLM"),
    "llava-v1.6-mistral-7b-hf": ("LLaVA-Mistral", "VLM"),
    "llava-v1.6-vicuna-7b-hf": ("LLaVA-Vicuna", "VLM"),
    "llava-v1.6-vicuna-13b-hf": ("LLaVA-Vicuna-13B", "VLM"),
    "Qwen3-VL-2B-Instruct": ("Qwen3-VL-2B", "VLM"),
    "Qwen3-VL-4B-Instruct": ("Qwen3-VL-4B", "VLM"),
    "Qwen3-VL-8B-Instruct": ("Qwen3-VL-8B", "VLM"),
    "Qwen3-VL-32B-Instruct": ("Qwen3-VL-32B", "VLM"),
}
LM_DECODER_DIR_TO_MODEL = {
    "InternVL3_5-1B": ("InternVL-1B (LM)", "VLM backbone decoder"),
    "InternVL3_5-2B": ("InternVL-2B (LM)", "VLM backbone decoder"),
    "InternVL3_5-8B": ("InternVL-8B (LM)", "VLM backbone decoder"),
    "llava-1.5-7b-hf": ("LLaVA-1.5 (LM)", "VLM backbone decoder"),
    "llava-v1.6-mistral-7b-hf": ("LLaVA-Mistral (LM)", "VLM backbone decoder"),
    "llava-v1.6-vicuna-7b-hf": ("LLaVA-Vicuna (LM)", "VLM backbone decoder"),
    "llava-v1.6-vicuna-13b-hf": ("LLaVA-Vicuna-13B (LM)", "VLM backbone decoder"),
    "Qwen3-VL-2B-Instruct": ("Qwen3-VL-2B (LM)", "VLM backbone decoder"),
    "Qwen3-VL-4B-Instruct": ("Qwen3-VL-4B (LM)", "VLM backbone decoder"),
    "Qwen3-VL-8B-Instruct": ("Qwen3-VL-8B (LM)", "VLM backbone decoder"),
    "Qwen3-VL-32B-Instruct": ("Qwen3-VL-32B (LM)", "VLM backbone decoder"),
}
BACKBONE_DIR_TO_MODEL = {
    "Mistral-7B-Instruct-v0.2": ("Mistral-7B", "standalone LLM"),
    "Phi-3.5-mini-instruct": ("Phi-3.5-mini", "standalone LLM"),
    "Qwen2.5-7B-Instruct": ("Qwen2.5-7B-Instruct", "standalone LLM"),
    "Qwen3-0.6B": ("Qwen3-0.6B", "standalone LLM"),
    "Qwen3-1.7B": ("Qwen3-1.7B", "standalone LLM"),
    "Qwen3-4B": ("Qwen3-4B", "standalone LLM"),
    "Qwen3-8B": ("Qwen3-8B", "standalone LLM"),
    "Qwen3-32B": ("Qwen3-32B", "standalone LLM"),
    "vicuna-7b-v1.5": ("Vicuna-7B", "standalone LLM"),
    "vicuna-13b-v1.5": ("Vicuna-13B", "standalone LLM"),
}
BACKBONE_THINK_DIR_TO_MODEL = {
    "Qwen3-0.6B_think": ("Qwen3-0.6B (think)", "standalone LLM (think)"),
    "Qwen3-1.7B_think": ("Qwen3-1.7B (think)", "standalone LLM (think)"),
    "Qwen3-4B_think": ("Qwen3-4B (think)", "standalone LLM (think)"),
    "Qwen3-8B_think": ("Qwen3-8B (think)", "standalone LLM (think)"),
    "Qwen3-32B_think": ("Qwen3-32B (think)", "standalone LLM (think)"),
}

LOGIT_SOURCES = [
    (ROOT / "evaluation/logits/vlm/pretrained", VLM_DIR_TO_MODEL),
    (ROOT / "evaluation/logits/lm_decoder/pretrained", LM_DECODER_DIR_TO_MODEL),
    (ROOT / "evaluation/logits/backbone/pretrained", BACKBONE_DIR_TO_MODEL),
    (ROOT / "evaluation/logits/backbone/pretrained", BACKBONE_THINK_DIR_TO_MODEL),
]

EOS_TOKENS = {"<|im_end|>", "</s>", "<eos>", "<|endoftext|>", "<|end|>"}

model_rows = []
for base_dir, dir_to_model in LOGIT_SOURCES:
    for dir_name, (display, group) in dir_to_model.items():
        fpath = base_dir / dir_name / "vqa_1k_control_inst_blind.jsonl"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                logits = ex.get("generated_logits", {})
                logit_data = logits.get("question")
                if not logit_data:
                    continue
                tokens = logit_data.get("content", [])
                probs = [np.exp(t["logprob"]) for t in tokens if t["token"] not in EOS_TOKENS]
                if not probs:
                    continue
                qid = ex["question_id"]
                answer = ex.get("generated_answers", {}).get("question", "")
                gt = mapper.get_answers(qid)
                acc = vqa_accuracy(answer, gt) if gt else 0.0
                model_rows.append({
                    "question_id": qid,
                    "model": display,
                    "model_group": group,
                    "confidence": float(np.mean(probs)),
                    "accuracy": acc,
                })

model_df = pd.DataFrame(model_rows)
# Filter to the same question set as humans for fair comparison
model_df_matched = model_df[model_df["question_id"].isin(human_qids)].copy()
print(f"  Model: {len(model_df)} total rows, {model_df['model'].nunique()} models")
print(f"  Model (matched to human q): {len(model_df_matched)} rows, "
      f"{model_df_matched['question_id'].nunique()} questions")


def save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Confidence distribution comparison (histogram)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Confidence distributions (histogram)…")

groups_present = [g for g in GROUP_ORDER if g in model_df_matched["model_group"].unique()]
n_panels = len(groups_present)
fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.0), sharey=True)
if n_panels == 1:
    axes = [axes]

# 5 bins matching the human Likert scale (1-5 → 0.05, 0.25, 0.5, 0.75, 1.0)
hist_bins = np.array([0.0, 0.15, 0.375, 0.625, 0.875, 1.001])
human_conf = human_df["confidence"].values

for ax, grp in zip(axes, groups_present):
    # Human histogram (normalized to density for comparability)
    ax.hist(human_conf, bins=hist_bins, density=True, alpha=0.45, color=HUMAN_COLOR,
            edgecolor="white", linewidth=0.6,
            label=f"Human (μ={human_conf.mean():.2f})")

    # Model group histogram
    grp_data = model_df_matched[model_df_matched["model_group"] == grp]["confidence"].values
    if len(grp_data) > 10:
        color = GROUP_COLORS.get(grp, "#888")
        ax.hist(grp_data, bins=hist_bins, density=True, alpha=0.45, color=color,
                edgecolor="white", linewidth=0.6,
                label=f"{grp} (μ={grp_data.mean():.2f})")

    short_grp = grp.replace("standalone LLM", "SA-LLM").replace("VLM backbone decoder", "Backbone")
    ax.set_title(short_grp, fontweight="bold")
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="upper left", frameon=True)

axes[0].set_ylabel("Density")
fig.suptitle("Confidence Distributions: Human vs Model (Inst-Blind, Variant C)",
             fontsize=12, y=1.02)
plt.tight_layout()
save(fig, "confidence_distribution_human_vs_model.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: ECE reliability diagram — human + model groups
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 2: Reliability / ECE diagram…")

N_BINS = 8
edges = np.linspace(0.0, 1.0, N_BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])


def compute_reliability(conf_arr, acc_arr):
    """Bin by confidence, return (mean_conf_per_bin, mean_acc_per_bin, count, ece)."""
    idx = np.digitize(conf_arr, edges[1:-1], right=False)
    bins = []
    for b in range(N_BINS):
        mask = idx == b
        if mask.sum() == 0:
            continue
        bins.append({
            "mean_conf": conf_arr[mask].mean(),
            "mean_acc": acc_arr[mask].mean(),
            "count": int(mask.sum()),
        })
    bin_df = pd.DataFrame(bins)
    n_total = len(conf_arr)
    ece = float(np.sum((bin_df["count"] / n_total) * np.abs(bin_df["mean_acc"] - bin_df["mean_conf"])))
    return bin_df, ece


fig, ax = plt.subplots(figsize=(6.5, 5.5))

# Human reliability curve
h_conf = human_df["confidence"].values
h_acc = human_df["accuracy"].values
h_bins, h_ece = compute_reliability(h_conf, h_acc)
ax.plot(h_bins["mean_conf"], h_bins["mean_acc"], marker="s", ms=7, lw=2.2,
        color=HUMAN_COLOR, label=f"Human (ECE={h_ece:.3f})", zorder=5)

# Model groups
for grp in groups_present:
    sub = model_df_matched[model_df_matched["model_group"] == grp]
    if sub.empty:
        continue
    m_bins, m_ece = compute_reliability(sub["confidence"].values, sub["accuracy"].values)
    color = GROUP_COLORS.get(grp, "#888")
    short = grp.replace("standalone LLM", "SA-LLM").replace("VLM backbone decoder", "Backbone")
    ax.plot(m_bins["mean_conf"], m_bins["mean_acc"], marker="o", ms=6, lw=2,
            color=color, label=f"{short} (ECE={m_ece:.3f})")

# Perfect calibration
ax.plot([0, 1], [0, 1], color="#888888", lw=1.2, ls="--", label="Perfect calibration")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Confidence")
ax.set_ylabel("Empirical Accuracy")
ax.set_title("Reliability Diagram: Human vs Model (Inst-Blind, Variant C)")
ax.legend(loc="lower right", frameon=True)
ax.set_aspect("equal")
plt.tight_layout()
save(fig, "reliability_human_vs_model.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Per-question scatter — mean confidence vs human accuracy
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 3: Per-question confidence vs accuracy scatter…")
from scipy import stats

# Human: per-question mean confidence and accuracy
hq = human_df.groupby("question_id").agg(
    h_confidence=("confidence", "mean"),
    h_accuracy=("accuracy", "mean"),
).reset_index()

# Model: per-question × group mean
mq = model_df_matched.groupby(["question_id", "model_group"]).agg(
    m_confidence=("confidence", "mean"),
    m_accuracy=("accuracy", "mean"),
).reset_index()

fig, axes = plt.subplots(1, len(groups_present) + 1, figsize=(4.2 * (len(groups_present) + 1), 4.2),
                         sharey=True)

# Panel 0: Human
ax = axes[0]
x, y = hq["h_confidence"].values, hq["h_accuracy"].values
ax.scatter(x, y, alpha=0.45, s=18, color=HUMAN_COLOR, edgecolors="none")
r, p = stats.pearsonr(x, y)
m_slope, b_int = np.polyfit(x, y, 1)
xr = np.linspace(x.min(), x.max(), 100)
ax.plot(xr, m_slope * xr + b_int, color=HUMAN_COLOR, lw=1.8)
ax.set_title(f"Human\nr={r:.2f}", fontsize=10)
ax.set_xlabel("Mean confidence")
ax.set_ylabel("Mean accuracy")

# Model panels
for i, grp in enumerate(groups_present):
    ax = axes[i + 1]
    sub = mq[mq["model_group"] == grp]
    x, y = sub["m_confidence"].values, sub["m_accuracy"].values
    color = GROUP_COLORS.get(grp, "#888")
    ax.scatter(x, y, alpha=0.45, s=18, color=color, edgecolors="none")
    r, p = stats.pearsonr(x, y)
    m_slope, b_int = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m_slope * xr + b_int, color=color, lw=1.8)
    short = grp.replace("standalone LLM", "SA-LLM").replace("VLM backbone decoder", "Backbone")
    ax.set_title(f"{short}\nr={r:.2f}", fontsize=10)
    ax.set_xlabel("Mean confidence")

fig.suptitle("Per-Question: Confidence vs Accuracy (Inst-Blind, Variant C)", fontsize=12, y=1.02)
plt.tight_layout()
save(fig, "confidence_vs_accuracy_scatter.png")

print("\nDone. All figures saved to:", OUT_DIR)
