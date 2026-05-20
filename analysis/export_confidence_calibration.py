"""
Export calibration-style confidence figures for the paper.

Outputs
-------
latex/AnonymousSubmission/LaTeX/figures/confidence_calibration/

  inst_blind_vC_group_reliability_accuracy_q113_h40.png
      Reliability-style plot:
      x = mean token probability bin
      y = empirical answer accuracy
      one curve per model group
      legend includes group-level ECE

  inst_blind_vC_group_confidence_hm_sbert_q88_h40.png
      Confidence-vs-human-alignment plot:
      x = mean token probability bin
      y = mean HM SBERT (clipped cosine)
      one curve per model group
      dashed horizontal line = HH SBERT reference

Run from repo root:
  ~/miniconda3/envs/aide/bin/python analysis/export_confidence_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from utils.constants import GROUP_COLORS, GROUP_ORDER
from export_helpers import read_export

OUT_DIR = ROOT / "latex/AnonymousSubmission/LaTeX/figures/confidence_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VLM_DIR_TO_MODEL = {
    "InternVL3_5-1B": "InternVL-1B",
    "InternVL3_5-2B": "InternVL-2B",
    "InternVL3_5-8B": "InternVL-8B",
    "llava-1.5-7b-hf": "LLaVA-1.5-7B",
    "llava-v1.6-mistral-7b-hf": "LLaVA-Mistral",
    "llava-v1.6-vicuna-13b-hf": "LLaVA-Vicuna",
    "llava-v1.6-vicuna-7b-hf": "LLaVA-Vicuna",
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen3-VL-32B-Instruct": "Qwen3-VL-32B",
}

LM_DECODER_DIR_TO_MODEL = {
    "llava-1.5-7b-hf": "LLaVA-1.5 (LM)",
    "llava-v1.6-mistral-7b-hf": "LLaVA-Mistral (LM)",
    "llava-v1.6-vicuna-13b-hf": "LLaVA-Vicuna (LM)",
    "llava-v1.6-vicuna-7b-hf": "LLaVA-Vicuna (LM)",
}

BACKBONE_NOTHINK_DIR_TO_MODEL = {
    "Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "Phi-3.5-mini-instruct": "Phi-3.5-mini",
    "Qwen2.5-7B": "Qwen2.5-7B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen3-0.6B": "Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen3-4B": "Qwen3-4B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-32B": "Qwen3-32B",
    "vicuna-13b-v1.5": "Vicuna-13B",
    "vicuna-7b-v1.5": "Vicuna-13B",
}

BACKBONE_THINK_DIR_TO_MODEL = {
    "Qwen3-0.6B": "Qwen3-0.6B (think)",
    "Qwen3-1.7B": "Qwen3-1.7B (think)",
    "Qwen3-4B": "Qwen3-4B (think)",
    "Qwen3-8B": "Qwen3-8B (think)",
    "Qwen3-32B": "Qwen3-32B (think)",
}

LOGIT_SOURCES = [
    (ROOT / "evaluation/logits/vlm/pretrained", VLM_DIR_TO_MODEL),
    (ROOT / "evaluation/logits/lm_decoder/pretrained", LM_DECODER_DIR_TO_MODEL),
    (ROOT / "evaluation/logits/backbone_nothink/pretrained", BACKBONE_NOTHINK_DIR_TO_MODEL),
    (ROOT / "evaluation/logits/backbone_think/pretrained", BACKBONE_THINK_DIR_TO_MODEL),
]

DATASET_FILENAME = "vqa_1k_control_inst_blind.jsonl"
CONTROL_TYPE = "question"  # variant C
N_BINS = 8

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
})


def save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  [confidence_calibration] {name}")


def apply_axis_style(ax, grid_axis="both"):
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=3.5, width=0.8, color="#555555")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def load_jsonl_examples(path: Path):
    out = []
    bad = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  WARN {path.name}: skipped {bad} malformed lines")
    return out


def assign_bin(series: pd.Series, n_bins: int = N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(series.to_numpy(), edges[1:-1], right=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pd.Series(idx, index=series.index), centers


def compute_ece(bin_df: pd.DataFrame, n_total: int) -> float:
    if n_total == 0 or bin_df.empty:
        return np.nan
    return float(np.sum((bin_df["count"] / n_total) * np.abs(bin_df["metric"] - bin_df["mean_conf"])))


print("Loading model confidence rows…")
rows = []
for base_dir, dir_to_model in LOGIT_SOURCES:
    for dir_name, model in dir_to_model.items():
        fpath = base_dir / dir_name / DATASET_FILENAME
        if not fpath.exists():
            continue
        examples = load_jsonl_examples(fpath)
        for ex in examples:
            logits = ex.get("generated_logits", {})
            logit_data = logits.get(CONTROL_TYPE)
            if not logit_data:
                continue
            tokens = logit_data.get("content", [])
            if not tokens:
                continue
            conf = float(np.exp([t["logprob"] for t in tokens]).mean())
            rows.append({
                "model": model,
                "question_id": ex.get("question_id"),
                "confidence": conf,
            })

conf_df = pd.DataFrame(rows)
if conf_df.empty:
    raise RuntimeError("No confidence rows loaded for inst_blind/question.")

model_df = read_export(ROOT, "responses_model_inst_blind.csv", variant="C")
human_df = read_export(ROOT, "responses_human.csv", variant="C")
pair_df = read_export(ROOT, "answer_pairs_sbert_text.csv", variant="C")

N_Q_ACC = model_df["question_id"].nunique()
N_H = human_df["participant"].nunique()

model_df = model_df[["question_id", "model", "model_group", "accuracy", "response"]].copy()
merged = conf_df.merge(model_df, on=["question_id", "model"], how="inner")
merged = merged[merged["model_group"].isin(GROUP_ORDER)].copy()

merged["bin"], centers = assign_bin(merged["confidence"])

print(f"  merged accuracy rows: {len(merged)} | questions={N_Q_ACC} | humans={N_H}")

# Reliability / ECE-style accuracy plot
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ece_handles = []
for grp in GROUP_ORDER:
    sub = merged[merged["model_group"] == grp].copy()
    if sub.empty:
        continue
    agg = (sub.groupby("bin")
             .agg(mean_conf=("confidence", "mean"),
                  metric=("accuracy", "mean"),
                  count=("accuracy", "size"))
             .reset_index())
    ax.plot(agg["mean_conf"], agg["metric"], marker="o", ms=6, lw=2,
            color=GROUP_COLORS[grp], label=grp)
    ax.scatter(agg["mean_conf"], agg["metric"], s=20 + 0.35 * agg["count"],
               color=GROUP_COLORS[grp], alpha=0.8)
    ece = compute_ece(agg, len(sub))
    ece_handles.append(mlines.Line2D([], [], color=GROUP_COLORS[grp], marker="o", lw=2,
                                     label=f"{grp} (ECE={ece:.3f})"))

ax.plot([0, 1], [0, 1], color="#666666", lw=1.2, ls="--", label="perfect calibration")
apply_axis_style(ax, "both")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Mean token probability")
ax.set_ylabel("Empirical accuracy")
ax.set_title("Reliability by Model Group — Inst-Blind, Variant C")
ax.legend(handles=ece_handles + [mlines.Line2D([], [], color="#666666", lw=1.2, ls="--", label="perfect calibration")],
          loc="lower right", frameon=True)
fig.tight_layout()
save(fig, f"inst_blind_vC_group_reliability_accuracy_q{N_Q_ACC}_h{N_H}.png")

# Confidence vs HM SBERT alignment
hm = pair_df[pair_df["pair_type"] == "HM"].copy()
hm = (hm.groupby(["question_id", "subject_2"])["sbert_score_clip"]
        .mean()
        .reset_index()
        .rename(columns={"subject_2": "model", "sbert_score_clip": "hm_sbert"}))

hm_merged = merged.merge(hm, on=["question_id", "model"], how="inner")
N_Q_HM = hm_merged["question_id"].nunique()

hh = pair_df[pair_df["pair_type"] == "HH"].copy()
hh_ref = float(hh["sbert_score_clip"].mean()) if not hh.empty else np.nan

fig, ax = plt.subplots(figsize=(6.2, 5.2))
for grp in GROUP_ORDER:
    sub = hm_merged[hm_merged["model_group"] == grp].copy()
    if sub.empty:
        continue
    agg = (sub.groupby("bin")
             .agg(mean_conf=("confidence", "mean"),
                  hm_sbert=("hm_sbert", "mean"),
                  count=("hm_sbert", "size"))
             .reset_index())
    ax.plot(agg["mean_conf"], agg["hm_sbert"], marker="o", ms=6, lw=2,
            color=GROUP_COLORS[grp], label=grp)
    ax.scatter(agg["mean_conf"], agg["hm_sbert"], s=20 + 0.35 * agg["count"],
               color=GROUP_COLORS[grp], alpha=0.8)

if not np.isnan(hh_ref):
    ax.axhline(hh_ref, color="#222222", lw=1.4, ls="--", label=f"HH SBERT={hh_ref:.3f}")

apply_axis_style(ax, "both")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Mean token probability")
ax.set_ylabel("Mean HM SBERT (clipped cosine)")
ax.set_title("Confidence vs Human Alignment — Inst-Blind, Variant C")
ax.legend(loc="lower right", frameon=True)
fig.tight_layout()
save(fig, f"inst_blind_vC_group_confidence_hm_sbert_q{N_Q_HM}_h{N_H}.png")

print("\nDone. Figures saved to:", OUT_DIR)
