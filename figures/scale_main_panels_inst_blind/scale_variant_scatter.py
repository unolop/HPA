"""
scale_variant_scatter.py

Scale scatter for SBERT agreement and accuracy, broken down by control variant (C/B/A).

Layout per metric file: 2×2 subplots, sharey='row'
  Row 0: VLM (col 0) | Backbone decoder (col 1)
  Row 1: SA-LLM (col 0) | Think (col 1)

Color:  variant C=dark blue, B=amber, A=dark red
Shape:  model family
Human:  per-variant dashed horizontal lines
"""

from __future__ import annotations
import json
import glob
import shutil
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from config import MODELS_ALL, MODEL_GROUP, MIN_ANSWERS_DEFAULT
from helpers import get_exports_dir, load_cleaned_pair_cache, load_human_subset, read_export, filter_abstained_pairs
from utils.constants import MODEL_SIZE_B, VARIANT_COLORS
from analysis.utils.abstention import classify, is_abstained

OUT_DIR   = ROOT / "figures/scale_main_panels"
LATEX_OUT = ROOT / "latex/AAAI2026/LaTeX/figures/scale_alignment_variant"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_OUT.mkdir(parents=True, exist_ok=True)

EXCLUDED_MODELS = {"Mistral-7B", "Phi-3.5-mini"}
MODELS_SIZED    = [m for m in MODELS_ALL if m in MODEL_SIZE_B and m not in EXCLUDED_MODELS]

VARIANTS = ["C", "B", "A"]
VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
JITTER         = {"C": -0.04, "B": 0.0, "A": +0.04}

FAMILY_MARKER = {
    "InternVL":      "o",
    "Qwen3":         "X",
    "LLaVA-1.5":     "D",
    "Mistral":       "^",
    "Vicuna":        "v",
    "Qwen3 (think)": "X",
    "Qwen2.5":       "s",
    "Other":         "*",
}
FAMILY_ORDER = ["InternVL", "Qwen3", "LLaVA-1.5", "Mistral", "Vicuna",
                "Qwen2.5"]

# 2×2 panel layout: ROW_DEFS[row][col] = (title, [groups])
ROW_DEFS = [
    [
        ("VLM",              ["VLM"]),
        ("Backbone decoder", ["VLM backbone decoder"]),
    ],
    [
        ("SA-LLM",           ["standalone LLM"]),
        ("Think",            ["standalone LLM (think)"]),
    ],
]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
})


def _ci95(vals: np.ndarray) -> float:
    n = len(vals)
    return 1.96 * np.std(vals) / np.sqrt(n) if n >= 2 else 0.0


def _get_family(model: str) -> str:
    if "InternVL" in model:   return "InternVL"
    if "Qwen3-VL" in model:   return "Qwen3"
    if "Qwen3" in model and "think" in model.lower(): return "Qwen3 (think)"
    if "Qwen3" in model:      return "Qwen3"
    if "Qwen2.5" in model:    return "Qwen2.5"
    if "LLaVA-1.5" in model or "llava-1.5" in model.lower(): return "LLaVA-1.5"
    if "Mistral" in model:    return "Mistral"
    if "Vicuna" in model or "vicuna" in model.lower(): return "Vicuna"
    return "Other"


def _marker_size(size: float, max_size: float) -> float:
    return 25 + 65 * (np.log10(size + 0.3) / np.log10(max_size + 0.3)) ** 0.8


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nLoading human data (min_answers={MIN_ANSWERS_DEFAULT})…")
participants, common_qids, human_df_full, _ = load_human_subset(
    ROOT, min_answers=MIN_ANSWERS_DEFAULT, translate=False, verbose=True
)
print(f"  common_qids: {len(common_qids)}  |  participants: {len(participants)}")

print("Loading pair cache (all variants)…")
pair_cache_full = load_cleaned_pair_cache(
    ROOT, condition="inst_blind", include_yesno=True, verbose=True
)
pair_cache_h   = pair_cache_full[pair_cache_full["question_id"].isin(common_qids)].copy()
pair_cache_all = pair_cache_full.copy()

print("Loading model accuracy (all variants)…")
raw_acc_full = read_export(ROOT, "responses_model_inst_blind.csv")
raw_acc_h    = raw_acc_full[raw_acc_full["question_id"].isin(common_qids)].copy()
raw_acc_all  = raw_acc_full.copy()
print("Loading blind model accuracy (all variants)…")
raw_acc_blind_full = read_export(ROOT, "responses_model_blind.csv")
raw_acc_blind_h    = raw_acc_blind_full[raw_acc_blind_full["question_id"].isin(common_qids)].copy()
raw_acc_blind_all  = raw_acc_blind_full.copy()

max_size   = max(MODEL_SIZE_B[m] for m in MODELS_SIZED)
SIZE_TICKS = [1, 2, 4, 7, 8, 13, 32]


# ── Compute per-model, per-variant stats ──────────────────────────────────────
def model_sbert_by_variant(pair_cache: pd.DataFrame) -> pd.DataFrame:
    sub = pair_cache[pair_cache["pair_type"] == "HM"]
    rows = []
    for v in VARIANTS:
        vsub    = sub[sub["variant"] == v]
        q_means = (vsub.groupby(["subject_2", "question_id"])["sbert_score"]
                   .mean().reset_index())
        for m in MODELS_SIZED:
            mq = q_means[q_means["subject_2"] == m]["sbert_score"].values
            if len(mq) == 0:
                continue
            rows.append({
                "model": m, "variant": v, "mean": mq.mean(), "ci": _ci95(mq),
                "size": MODEL_SIZE_B[m],
                "group": MODEL_GROUP.get(m, "standalone LLM"),
            })
    return pd.DataFrame(rows)


def model_accuracy_by_variant(raw_acc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        vsub    = raw_acc[raw_acc["variant"] == v]
        q_means = (vsub.groupby(["model", "question_id"])["accuracy"]
                   .mean().reset_index())
        for m in MODELS_SIZED:
            mq = q_means[q_means["model"] == m]["accuracy"].values
            if len(mq) == 0:
                continue
            rows.append({
                "model": m, "variant": v, "mean": mq.mean(), "ci": _ci95(mq),
                "size": MODEL_SIZE_B[m],
                "group": MODEL_GROUP.get(m, "standalone LLM"),
            })
    return pd.DataFrame(rows)


def human_sbert_by_variant(pair_cache: pd.DataFrame) -> dict:
    hh = pair_cache[pair_cache["pair_type"] == "HH"]
    return {
        v: hh[hh["variant"] == v].groupby("question_id")["sbert_score"]
           .mean().pipe(lambda s: (s.mean(), _ci95(s.values)))
        for v in VARIANTS
    }


def human_accuracy_by_variant() -> dict:
    result = {}
    for v in VARIANTS:
        hv = human_df_full[
            (human_df_full["variant"] == v) &
            (human_df_full["question_id"].isin(common_qids))
        ]
        q_vals = hv.groupby("question_id")["accuracy"].mean().values
        result[v] = (q_vals.mean(), _ci95(q_vals))
    return result


def load_confidence_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    sem_df = pd.read_json(ROOT / "dataset/vqa/vqa1k_semantics.jsonl", lines=True)[["question_id", "ent", "op"]]
    human_files = sorted(glob.glob(str(ROOT / "evaluation/humans/by_participant/*.json")))
    human_rows = []
    for fp in human_files:
        with open(fp) as f:
            data = json.load(f)
        pid = data.get("code", Path(fp).stem)
        for ans in data.get("answers", []):
            raw_conf = ans.get("confidence", 3)
            human_rows.append({
                "question_id": ans["question_id"],
                "participant": pid,
                "variant": ans.get("variant", "C"),
                "confidence": (raw_conf - 1) / 4.0,
            })
    human_df = pd.DataFrame(human_rows).merge(sem_df, on="question_id", how="left")

    ct_to_variant = {"question": "C", "weaker_object": "B", "pronominalized": "A"}
    eos_tokens = {"<|im_end|>", "</s>", "<eos>", "<|endoftext|>", "<|end|>"}
    logit_sources = [
        (ROOT / "evaluation/logits/vlm/pretrained", {
            "InternVL3_5-8B": "InternVL-8B",
            "llava-1.5-7b-hf": "LLaVA-1.5-7B",
            "llava-v1.6-mistral-7b-hf": "LLaVA-Mistral",
            "llava-v1.6-vicuna-7b-hf": "LLaVA-Vicuna",
            "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
        }),
        (ROOT / "evaluation/logits/lm_decoder/pretrained", {
            "InternVL3_5-8B": "InternVL-8B (LM)",
            "llava-1.5-7b-hf": "LLaVA-1.5 (LM)",
            "llava-v1.6-mistral-7b-hf": "LLaVA-Mistral (LM)",
            "llava-v1.6-vicuna-7b-hf": "LLaVA-Vicuna (LM)",
            "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B (LM)",
        }),
        (ROOT / "evaluation/logits/backbone/pretrained", {
            "Mistral-7B-Instruct-v0.2": "Mistral-7B",
            "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
            "Qwen3-8B": "Qwen3-8B",
            "vicuna-7b-v1.5": "Vicuna-7B",
        }),
        (ROOT / "evaluation/logits/backbone/pretrained", {
            "Qwen3-8B_think": "Qwen3-8B (think)",
        }),
    ]
    human_qids = set(human_df["question_id"].unique())
    rows = []
    for base_dir, mapping in logit_sources:
        for dir_name, display in mapping.items():
            fpath = base_dir / dir_name / "vqa_1k_control_inst_blind.jsonl"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    qid = ex["question_id"]
                    if qid not in human_qids:
                        continue
                    logits = ex.get("generated_logits", {})
                    for ct, variant in ct_to_variant.items():
                        logit_data = logits.get(ct)
                        if not logit_data:
                            continue
                        probs = [
                            np.exp(t["logprob"])
                            for t in logit_data.get("content", [])
                            if t.get("token") not in eos_tokens
                        ]
                        if not probs:
                            continue
                        rows.append({
                            "question_id": qid,
                            "model": display,
                            "variant": variant,
                            "confidence": float(np.mean(probs)),
                        })
    model_df = pd.DataFrame(rows).merge(sem_df, on="question_id", how="left")
    return human_df, model_df


def model_confidence_by_variant(model_conf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        vsub = model_conf[model_conf["variant"] == v]
        q_means = (vsub.groupby(["model", "question_id"])["confidence"]
                   .mean().reset_index())
        for m in MODELS_SIZED:
            mq = q_means[q_means["model"] == m]["confidence"].values
            if len(mq) == 0:
                continue
            rows.append({
                "model": m, "variant": v, "mean": mq.mean(), "ci": _ci95(mq),
                "size": MODEL_SIZE_B[m],
                "group": MODEL_GROUP.get(m, "standalone LLM"),
            })
    return pd.DataFrame(rows)


def human_confidence_by_variant(human_conf: pd.DataFrame) -> dict:
    result = {}
    for v in VARIANTS:
        hv = human_conf[
            (human_conf["variant"] == v) &
            (human_conf["question_id"].isin(common_qids))
        ]
        q_vals = hv.groupby("question_id")["confidence"].mean().values
        result[v] = (q_vals.mean(), _ci95(q_vals))
    return result


print("Computing stats (human subset, 113q)…")
sbert_stats_h = model_sbert_by_variant(pair_cache_h)
acc_stats_h   = model_accuracy_by_variant(raw_acc_h)
print("Computing blind stats (human subset, 113q)…")
pair_cache_blind_full = load_cleaned_pair_cache(
    ROOT, condition="blind", include_yesno=True, verbose=True
)
pair_cache_blind_h   = pair_cache_blind_full[pair_cache_blind_full["question_id"].isin(common_qids)].copy()
sbert_stats_blind_h  = model_sbert_by_variant(pair_cache_blind_h)
acc_stats_blind_h    = model_accuracy_by_variant(raw_acc_blind_h)
human_conf_full, model_conf_full = load_confidence_frames()
human_conf_h = human_conf_full[human_conf_full["question_id"].isin(common_qids)].copy()
model_conf_h = model_conf_full[model_conf_full["question_id"].isin(common_qids)].copy()
conf_stats_h = model_confidence_by_variant(model_conf_h)
hh_sbert      = human_sbert_by_variant(pair_cache_h)
hh_acc        = human_accuracy_by_variant()
hh_conf       = human_confidence_by_variant(human_conf_h)
print(f"  SBERT rows: {len(sbert_stats_h)}  |  Accuracy rows: {len(acc_stats_h)}  |  Confidence rows: {len(conf_stats_h)}")

print("Computing stats (full question set, ~1000q)…")
sbert_stats_all = model_sbert_by_variant(pair_cache_all)
acc_stats_all   = model_accuracy_by_variant(raw_acc_all)
sbert_stats_blind_all = model_sbert_by_variant(pair_cache_blind_full)
acc_stats_blind_all   = model_accuracy_by_variant(raw_acc_blind_all)
n_q_all = raw_acc_all["question_id"].nunique()
print(f"  SBERT rows: {len(sbert_stats_all)}  |  Accuracy rows: {len(acc_stats_all)}  |  questions: {n_q_all}")


# ── 2×2 plot function (one metric per file) ───────────────────────────────────
def _plot_variant_scatter(
    stats_df: pd.DataFrame,
    human_stats: dict,
    ylabel: str,
    fname: str,
    overlay_df: pd.DataFrame | None = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(4.9, 4.5), sharey="row", sharex=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.18,
                        wspace=0.22, hspace=0.36)

    for row_idx, row_panels in enumerate(ROW_DEFS):
        for col_idx, (panel_title, groups) in enumerate(row_panels):
            ax = axes[row_idx, col_idx]
            sub = stats_df[stats_df["group"].isin(groups)]

            for _, row in sub.iterrows():
                col    = VARIANT_COLORS[row["variant"]]
                mrk    = FAMILY_MARKER.get(_get_family(row["model"]), "o")
                ms     = _marker_size(row["size"], max_size)
                x      = row["size"] * (1.0 + JITTER[row["variant"]])
                ax.scatter(x, row["mean"], s=ms, marker=mrk,
                           color=col, alpha=0.78, zorder=3,
                           edgecolors="white", linewidths=0.4)

            if overlay_df is not None:
                sub_overlay = overlay_df[overlay_df["group"].isin(groups)]
                for _, row in sub_overlay.iterrows():
                    col    = VARIANT_COLORS[row["variant"]]
                    mrk    = FAMILY_MARKER.get(_get_family(row["model"]), "o")
                    ms     = _marker_size(row["size"], max_size)
                    x      = row["size"] * (1.0 + JITTER[row["variant"]])
                    ax.scatter(
                        x, row["mean"], s=ms, marker=mrk,
                        facecolors="none", edgecolors=col,
                        alpha=0.55, zorder=2, linewidths=1.1,
                    )

            # Per-variant human baselines
            for v in VARIANTS:
                hm, _ = human_stats[v]
                ax.axhline(hm, color=VARIANT_COLORS[v], lw=1.1, ls="--",
                           alpha=0.50, zorder=1)

            ax.set_xscale("log")
            ax.set_xticks(SIZE_TICKS)
            ax.set_xticklabels([str(t) for t in SIZE_TICKS])
            ax.set_xlim(0.7, 45)
            ax.set_title(panel_title, fontsize=9)

            ax.set_ylabel("")

    # ── Combined legend at bottom (2 rows) ───────────────────────────────────
    variant_handles = [
        mpatches.Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
        for v in VARIANTS
    ] + [
        mlines.Line2D([], [], color="#888", ls="--", lw=1.1, alpha=0.6,
                      label="Human baseline")
    ]

    family_handles = sorted([
        mlines.Line2D([], [], marker=FAMILY_MARKER.get(f, "o"), color="#444",
                      markersize=7, linestyle="None", label=f)
        for f in FAMILY_ORDER if f in FAMILY_MARKER
    ], key=lambda h: h.get_label())

    # variant_handles (4 items) + family handles in a wider legend grid
    all_handles = variant_handles + family_handles
    fig.legend(handles=all_handles, frameon=True, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.50, 0.03),
               ncol=5, handletextpad=0.4, columnspacing=0.8)

    out = OUT_DIR / fname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_OUT / fname)
    print(f"Saved: {out}")
    print(f"Copied to: {LATEX_OUT / fname}")
    plt.close(fig)


# ── Generate figures ──────────────────────────────────────────────────────────
# Human subset (113q, 40 participants) — raw
_plot_variant_scatter(
    sbert_stats_h, hh_sbert,
    ylabel="SBERT cosine",
    fname="sbert_vABC_q113_h40_raw.png",
)
_plot_variant_scatter(
    sbert_stats_h, hh_sbert,
    ylabel="SBERT cosine",
    fname="sbert_vABC_q113_h40_raw_with_blind.png",
    overlay_df=sbert_stats_blind_h,
)
_plot_variant_scatter(
    acc_stats_h, hh_acc,
    ylabel="Mean accuracy",
    fname="accuracy_vABC_q113_h40_raw.png",
)
_plot_variant_scatter(
    acc_stats_h, hh_acc,
    ylabel="Mean accuracy",
    fname="accuracy_vABC_q113_h40_raw_with_blind.png",
    overlay_df=acc_stats_blind_h,
)
_plot_variant_scatter(
    conf_stats_h, hh_conf,
    ylabel="Mean confidence",
    fname="confidence_vABC_q113_h40_raw.png",
)

# Full question set (~1000q)
_plot_variant_scatter(
    sbert_stats_all, hh_sbert,
    ylabel="SBERT cosine",
    fname=f"sbert_vABC_q{n_q_all}.png",
)
_plot_variant_scatter(
    sbert_stats_all, hh_sbert,
    ylabel="SBERT cosine",
    fname=f"sbert_vABC_q{n_q_all}_with_blind.png",
    overlay_df=sbert_stats_blind_all,
)
_plot_variant_scatter(
    acc_stats_all, hh_acc,
    ylabel="Mean accuracy",
    fname=f"accuracy_vABC_q{n_q_all}.png",
)
_plot_variant_scatter(
    acc_stats_all, hh_acc,
    ylabel="Mean accuracy",
    fname=f"accuracy_vABC_q{n_q_all}_with_blind.png",
    overlay_df=acc_stats_blind_all,
)

# ── Abstention-filtered (113q subset only) ────────────────────────────────────
print("\nGenerating abstention-filtered variants (113q)…")
pair_cache_h_filt = filter_abstained_pairs(pair_cache_h)
raw_acc_h_filt = raw_acc_h[~raw_acc_h['response'].fillna('').astype(str).apply(
    lambda x: is_abstained(classify(x, None)))].copy()

sbert_stats_h_filt = model_sbert_by_variant(pair_cache_h_filt)
acc_stats_h_filt   = model_accuracy_by_variant(raw_acc_h_filt)
conf_stats_h_filt  = model_confidence_by_variant(model_conf_h)

_plot_variant_scatter(
    sbert_stats_h_filt, hh_sbert,
    ylabel="SBERT cosine",
    fname="sbert_vABC_q113_h40_abstfiltered.png",
)
pair_cache_blind_h_filt = filter_abstained_pairs(pair_cache_blind_h)
raw_acc_blind_h_filt = raw_acc_blind_h[~raw_acc_blind_h['response'].fillna('').astype(str).apply(
    lambda x: is_abstained(classify(x, None)))].copy()
sbert_stats_blind_h_filt = model_sbert_by_variant(pair_cache_blind_h_filt)
acc_stats_blind_h_filt   = model_accuracy_by_variant(raw_acc_blind_h_filt)
_plot_variant_scatter(
    sbert_stats_h_filt, hh_sbert,
    ylabel="SBERT cosine",
    fname="sbert_vABC_q113_h40_abstfiltered_with_blind.png",
    overlay_df=sbert_stats_blind_h_filt,
)
_plot_variant_scatter(
    acc_stats_h_filt, hh_acc,
    ylabel="Mean accuracy",
    fname="accuracy_vABC_q113_h40_abstfiltered.png",
)
_plot_variant_scatter(
    acc_stats_h_filt, hh_acc,
    ylabel="Mean accuracy",
    fname="accuracy_vABC_q113_h40_abstfiltered_with_blind.png",
    overlay_df=acc_stats_blind_h_filt,
)
_plot_variant_scatter(
    conf_stats_h_filt, hh_conf,
    ylabel="Mean confidence",
    fname="confidence_vABC_q113_h40_abstfiltered.png",
)

print(f"\nDone. Saved to: {OUT_DIR}")
