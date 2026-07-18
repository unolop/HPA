"""
Human vs Model confidence comparison figures for the paper.

Outputs (figures/confidence_human/):

  1. confidence_distribution_human_vs_model.png
     Histogram: human self-reported confidence vs model token probability
     Side-by-side per model group, bins match human Likert scale.

  2. alignment_by_variant.png
     Per-question scatter: mean human confidence vs mean model confidence
     Panels = model groups, colors/markers = variants (C/B/A).

  3. correlation_by_group.png
     Bar chart: Pearson r(human conf, model conf) per model group × variant,
     broken down by entity type and operation type.

Run from repo root:
  ~/miniconda3/envs/zero/bin/python figures/confidence_human/comparison.py
"""

from __future__ import annotations

import json
import glob
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from utils.constants import (
    GROUP_COLORS, GROUP_ORDER, VARIANT_ORDER, VARIANT_COLORS, VARIANT_LABELS,
    MODEL_FAMILY, MODEL_FAMILY_COLORS,
)
from utils.vqa import VQAAnswerMapper, vqa_accuracy

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "confidence"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

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

HUMAN_COLOR = "#7B1FA2"  # purple

# Control-type → variant mapping
CT_TO_VARIANT = {"question": "C", "weaker_object": "B", "pronominalized": "A"}

# ── Load semantics for entity / op grouping ────────────────────────────────────
sem_df = pd.read_json(ROOT / "dataset/vqa/vqa1k_semantics.jsonl", lines=True)[
    ["question_id", "ent", "op"]
]

# ── Load human confidence data (all variants) ─────────────────────────────────
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
        human_rows.append({
            "question_id": qid,
            "participant": pid,
            "variant": variant,
            "confidence": conf_norm,
        })

human_df = pd.DataFrame(human_rows)
human_qids = set(human_df["question_id"].unique())
print(f"  Human: {len(human_df)} rows, {human_df['participant'].nunique()} participants, "
      f"{len(human_qids)} questions, variants={sorted(human_df['variant'].unique())}")

# ── Load model confidence data (all 3 variants) ──────────────────────────────
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
                qid = ex["question_id"]
                for ct, variant in CT_TO_VARIANT.items():
                    logit_data = logits.get(ct)
                    if not logit_data:
                        continue
                    tokens = logit_data.get("content", [])
                    probs = [np.exp(t["logprob"]) for t in tokens
                             if t["token"] not in EOS_TOKENS]
                    if not probs:
                        continue
                    model_rows.append({
                        "question_id": qid,
                        "model": display,
                        "model_group": group,
                        "variant": variant,
                        "confidence": float(np.mean(probs)),
                    })

model_df = pd.DataFrame(model_rows)
model_df = model_df[model_df["question_id"].isin(human_qids)].copy()
model_df["family"] = model_df["model"].map(MODEL_FAMILY)
model_df = model_df.merge(sem_df, on="question_id", how="left")
human_df = human_df.merge(sem_df, on="question_id", how="left")

print(f"  Model (matched): {len(model_df)} rows, {model_df['model'].nunique()} models, "
      f"variants={sorted(model_df['variant'].unique())}")
print(f"  Families: {sorted(model_df['family'].dropna().unique())}")


def save(fig, name: str, latex_aliases: list[str] | None = None):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    shutil.copy(path, LATEX_DIR / name)
    for alias in latex_aliases or []:
        shutil.copy(path, LATEX_DIR / alias)
    plt.close(fig)
    print(f"  → {name}")


def ecdf_xy(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def safe_pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(stats.pearsonr(x, y)[0])


def bootstrap_corr_ci(df: pd.DataFrame, x_col: str, y_col: str,
                      n_boot: int = 1000, seed: int = 0):
    vals = []
    if len(df) < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(df)
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    point = safe_pearsonr(x, y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = safe_pearsonr(x[idx], y[idx])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


groups_present = [g for g in GROUP_ORDER if g in model_df["model_group"].unique()]

# Families with enough data (exclude single-model families with <2 sizes)
FAMILY_ORDER = [f for f in ['InternVL', 'LLaVA-1.5', 'LLaVA-Mistral', 'LLaVA-Vicuna',
                            'Qwen3-VL', 'Qwen3', 'Mistral', 'Vicuna', 'Phi', 'Qwen2.5']
                if f in model_df['family'].values]


def short_grp(g):
    return (g.replace("standalone LLM", "SA-LLM")
             .replace("VLM backbone decoder", "Backbone"))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Confidence distribution histogram (variant C only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Confidence distributions (histogram)…")

n_panels = len(groups_present)
fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.4), sharey=True)
if n_panels == 1:
    axes = [axes]

hist_bins = np.array([0.0, 0.15, 0.375, 0.625, 0.875, 1.001])
human_conf_c = human_df[human_df["variant"] == "C"]["confidence"].values

for ax, grp in zip(axes, groups_present):
    ax.hist(human_conf_c, bins=hist_bins, density=True, alpha=0.45, color=HUMAN_COLOR,
            edgecolor="white", linewidth=0.6,
            label=f"Human (μ={human_conf_c.mean():.2f})")

    grp_data = model_df[(model_df["model_group"] == grp)
                         & (model_df["variant"] == "C")]["confidence"].values
    if len(grp_data) > 10:
        color = GROUP_COLORS.get(grp, "#888")
        ax.hist(grp_data, bins=hist_bins, density=True, alpha=0.45, color=color,
                edgecolor="white", linewidth=0.6,
                label=f"{short_grp(grp)} (μ={grp_data.mean():.2f})")

    ax.set_title(short_grp(grp), fontweight="bold")
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9, loc="upper left", frameon=True)

axes[0].set_ylabel("Density")
plt.tight_layout(pad=0.6, w_pad=0.5)
save(
    fig,
    "distribution_human_vs_model.png",
    latex_aliases=["confidence_distribution_human_vs_model.png"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1b: Confidence distribution ECDF (all variants)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 1b: Confidence distributions (ECDF)…")

fig, axes = plt.subplots(1, len(VARIANT_ORDER),
                         figsize=(4.3 * len(VARIANT_ORDER), 3.5), sharey=True)
if len(VARIANT_ORDER) == 1:
    axes = [axes]

for ax, variant in zip(axes, VARIANT_ORDER):
    human_conf = human_df[human_df["variant"] == variant]["confidence"].values
    hx, hy = ecdf_xy(human_conf)
    ax.step(hx, hy, where="post", color=HUMAN_COLOR, lw=2.3, ls="--",
            label=f"Human (μ={human_conf.mean():.2f})")

    for grp in groups_present:
        grp_data = model_df[(model_df["model_group"] == grp)
                            & (model_df["variant"] == variant)]["confidence"].values
        if len(grp_data) <= 10:
            continue
        gx, gy = ecdf_xy(grp_data)
        color = GROUP_COLORS.get(grp, "#888")
        ax.step(gx, gy, where="post", color=color, lw=2.0,
                label=f"{short_grp(grp)} (μ={grp_data.mean():.2f})")

    ax.set_title(VARIANT_LABELS[variant], fontweight="bold")
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(fontsize=8, loc="lower right", frameon=True)

axes[0].set_ylabel("ECDF")
fig.suptitle("Confidence ECDFs: Human vs Model (Inst-Blind)",
             fontsize=12, y=1.02)
plt.tight_layout(pad=0.6, w_pad=0.5)
save(
    fig,
    "distribution_human_vs_model_ecdf.png",
    latex_aliases=["confidence_distribution_human_vs_model_ecdf.png"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Per-question confidence alignment scatter — all variants
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 2: Per-question confidence alignment by variant…")

# Aggregate per question × variant (group level — for group figures)
hq = (human_df.groupby(["question_id", "variant"])["confidence"]
      .mean().reset_index().rename(columns={"confidence": "h_conf"}))
mq = (model_df.groupby(["question_id", "variant", "model_group"])["confidence"]
      .mean().reset_index().rename(columns={"confidence": "m_conf"}))
merged = mq.merge(hq, on=["question_id", "variant"])

# Family-level merge: keep individual model + family for per-family figures
mq_fam = (model_df.groupby(["question_id", "variant", "model_group", "family"])["confidence"]
           .mean().reset_index().rename(columns={"confidence": "m_conf"}))
merged_fam = mq_fam.merge(hq, on=["question_id", "variant"])

# Individual model merge for per-model overlays on grouped plots
mq_model = (model_df.groupby(["question_id", "variant", "model_group", "model", "family"])["confidence"]
            .mean().reset_index().rename(columns={"confidence": "m_conf"}))
merged_model = mq_model.merge(hq, on=["question_id", "variant"])

VARIANT_MARKERS = {"C": "o", "B": "s", "A": "^"}

fig, axes = plt.subplots(1, len(groups_present),
                         figsize=(4.0 * len(groups_present), 3.8), sharey=True)
if len(groups_present) == 1:
    axes = [axes]

for ax, grp in zip(axes, groups_present):
    sub = merged[merged["model_group"] == grp]
    for v in VARIANT_ORDER:
        vs = sub[sub["variant"] == v]
        if vs.empty:
            continue
        ax.scatter(vs["h_conf"], vs["m_conf"], alpha=0.5, s=22,
                   color=VARIANT_COLORS[v], marker=VARIANT_MARKERS[v],
                   edgecolors="none", label=VARIANT_LABELS[v])
        r, p = stats.pearsonr(vs["h_conf"], vs["m_conf"])
        # add r to legend label
        handles, labels = ax.get_legend_handles_labels()
        handles[-1].set_label(f"{VARIANT_LABELS[v]} (r={r:.2f})")

    ax.plot([0, 1], [0, 1], color="#888", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(short_grp(grp), fontweight="bold")
    ax.set_xlabel("Human confidence (mean)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left", frameon=True)

axes[0].set_ylabel("Model confidence (mean)")
plt.tight_layout(pad=0.6, w_pad=0.5)
save(
    fig,
    "alignment_by_variant.png",
    latex_aliases=["confidence_alignment_by_variant.png"],
)


# Add semantics to both merged frames
merged = merged.merge(sem_df, on="question_id", how="left",
                      suffixes=("", "_dup"))
for c in [c for c in merged.columns if c.endswith("_dup")]:
    merged.drop(columns=c, inplace=True)

merged_fam = merged_fam.merge(sem_df, on="question_id", how="left",
                               suffixes=("", "_dup"))
for c in [c for c in merged_fam.columns if c.endswith("_dup")]:
    merged_fam.drop(columns=c, inplace=True)

ENT_GROUPS = {
    "object": "object", "person": "person", "animal": "animal",
    "food": "food", "vehicle": "visual-other", "product": "visual-other",
    "place": "visual-other", "text": "text", "other": "text",
}
merged["ent_group"] = merged["ent"].map(ENT_GROUPS)
merged_fam["ent_group"] = merged_fam["ent"].map(ENT_GROUPS)
ent_values = ["object", "person", "animal", "food", "visual-other", "text"]

OP_GROUPS = {
    "attr": "attr", "count": "count", "ident": "ident", "spat": "spat",
    "exist": "exist", "act": "act", "know": "other", "text": "other",
    "temp": "other", "comp": "other", "other": "other", "cause": "other",
}
merged["op_group"] = merged["op"].map(OP_GROUPS)
merged_fam["op_group"] = merged_fam["op"].map(OP_GROUPS)
op_values = ["attr", "count", "ident", "spat", "exist", "act", "other"]


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Per-family confidence alignment scatter (all variants)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 3: Per-family confidence alignment scatter…")

families_present = [f for f in FAMILY_ORDER if f in merged_fam["family"].values]
VARIANT_MARKERS = {"C": "o", "B": "s", "A": "^"}

n_fam = len(families_present)
n_cols = min(5, n_fam)
n_rows = (n_fam + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 4.0 * n_rows),
                         squeeze=False)

for idx, fam in enumerate(families_present):
    row, col = divmod(idx, n_cols)
    ax = axes[row][col]
    sub = merged_fam[merged_fam["family"] == fam]
    for v in VARIANT_ORDER:
        vs = sub[sub["variant"] == v]
        if len(vs) < 5:
            continue
        r, _ = stats.pearsonr(vs["h_conf"], vs["m_conf"])
        ax.scatter(vs["h_conf"], vs["m_conf"], alpha=0.45, s=20,
                   color=VARIANT_COLORS[v], marker=VARIANT_MARKERS[v],
                   edgecolors="none",
                   label=f"{VARIANT_LABELS[v]} (r={r:.2f})")
    ax.plot([0, 1], [0, 1], color="#888", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(fam, fontweight="bold",
                 color=MODEL_FAMILY_COLORS.get(fam, "#333"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Human conf." if row == n_rows - 1 else "")
    if col == 0:
        ax.set_ylabel("Model conf.")
    ax.legend(fontsize=7, loc="upper left", frameon=True)

# Hide unused axes
for idx in range(n_fam, n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    axes[row][col].set_visible(False)

fig.suptitle("Per-Family Confidence Alignment: Human vs Model (all variants)",
             fontsize=12, y=1.01)
plt.tight_layout()
save(
    fig,
    "alignment_by_family.png",
    latex_aliases=["confidence_alignment_by_family.png"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Correlation by variant — per family (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 4: Correlation by variant × family…")

fig, ax = plt.subplots(figsize=(max(8, len(families_present) * 1.2), 5))

n_variants = len(VARIANT_ORDER)
n_fams = len(families_present)
bar_w = 0.8 / n_variants
x_base = np.arange(n_fams)

for vi, v in enumerate(VARIANT_ORDER):
    vals = []
    for fam in families_present:
        sub = merged_fam[(merged_fam["family"] == fam) & (merged_fam["variant"] == v)]
        if len(sub) < 5:
            vals.append(np.nan)
        else:
            r, _ = stats.pearsonr(sub["h_conf"], sub["m_conf"])
            vals.append(r)
    ax.bar(x_base + vi * bar_w, vals, width=bar_w,
           color=VARIANT_COLORS[v], alpha=0.85, label=VARIANT_LABELS[v])
    for xi, val in zip(x_base + vi * bar_w, vals):
        if not np.isnan(val):
            ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=45)

ax.set_xticks(x_base + bar_w * (n_variants - 1) / 2)
ax.set_xticklabels(families_present, fontsize=9, rotation=25, ha="right")
ax.set_ylabel("Pearson r (human–model confidence)")
ax.set_title("Human–Model Confidence Correlation by Family and Variant",
             fontweight="bold")
ax.axhline(0, color="#888", lw=0.6)
ax.legend(fontsize=9, loc="upper right", frameon=True)
plt.tight_layout()
save(
    fig,
    "correlation_by_family_variant.png",
    latex_aliases=["confidence_correlation_by_family_variant.png"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Correlation by entity/op — per family (heatmap style)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 5: Correlation by family × entity and operation…")

def family_corr_matrix(cat_col, cat_values):
    """Return DataFrame: rows=families, cols=categories, values=r (mean over variants)."""
    data = {}
    for fam in families_present:
        row = {}
        for cat in cat_values:
            sub = merged_fam[(merged_fam["family"] == fam)
                             & (merged_fam[cat_col] == cat)]
            if len(sub) < 5:
                row[cat] = np.nan
            else:
                r, _ = stats.pearsonr(sub["h_conf"], sub["m_conf"])
                row[cat] = r
        data[fam] = row
    return pd.DataFrame(data).T


import matplotlib.colors as mcolors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(families_present) * 0.55)))

cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.6)

for ax, cat_col, cat_values, title in [
    (ax1, "ent_group", ent_values, "By Entity Group"),
    (ax2, "op_group", op_values, "By Operation Type"),
]:
    mat = family_corr_matrix(cat_col, cat_values)
    im = ax.imshow(mat.values.astype(float), cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(len(cat_values)))
    ax.set_xticklabels(cat_values, fontsize=9, rotation=35, ha="right")
    ax.set_yticks(range(len(families_present)))
    ax.set_yticklabels(families_present, fontsize=9)
    ax.set_title(title, fontweight="bold")

fig.colorbar(im, ax=[ax1, ax2], label="Pearson r", shrink=0.8, pad=0.02)
fig.suptitle("Human–Model Confidence Correlation by Family (mean across variants)",
             fontsize=12, y=1.02)
plt.tight_layout()
save(
    fig,
    "correlation_heatmap_family.png",
    latex_aliases=["confidence_correlation_heatmap_family.png"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 (kept): Overall correlation by variant × model group
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 6: Overall correlation by variant × model group…")

fig, ax = plt.subplots(figsize=(7, 4.5))

n_variants = len(VARIANT_ORDER)
n_groups = len(groups_present)
bar_w = 0.8 / n_groups
x_base = np.arange(n_variants)

family_markers = {
    "InternVL": "o",
    "LLaVA-1.5": "s",
    "LLaVA-Mistral": "D",
    "LLaVA-Vicuna": "P",
    "Qwen3-VL": "^",
    "Qwen3": "v",
    "Mistral": "X",
    "Vicuna": "*",
    "Phi": "h",
    "Qwen2.5": ">",
}

for gi, grp in enumerate(groups_present):
    vals, err_lo, err_hi = [], [], []
    for v in VARIANT_ORDER:
        sub = merged[(merged["model_group"] == grp) & (merged["variant"] == v)]
        if len(sub) < 5:
            vals.append(np.nan)
            err_lo.append(np.nan)
            err_hi.append(np.nan)
        else:
            r, lo, hi = bootstrap_corr_ci(sub, "h_conf", "m_conf",
                                          n_boot=1000, seed=gi * 100 + ord(v))
            vals.append(r)
            err_lo.append(r - lo if np.isfinite(lo) else np.nan)
            err_hi.append(hi - r if np.isfinite(hi) else np.nan)
    color = GROUP_COLORS.get(grp, "#888")
    xpos = x_base + gi * bar_w
    ax.bar(xpos, vals, width=bar_w, color=color,
           alpha=0.35, edgecolor=color, linewidth=1.0, label=short_grp(grp),
           yerr=np.vstack([err_lo, err_hi]), capsize=2.5,
           error_kw={"elinewidth": 1.0, "ecolor": color})
    for xi, val in zip(x_base + gi * bar_w, vals):
        if not np.isnan(val):
            ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8, color=color)

        # Overlay per-model correlations
    for vi, v in enumerate(VARIANT_ORDER):
        model_sub = merged_model[(merged_model["model_group"] == grp) &
                                 (merged_model["variant"] == v)]
        if model_sub.empty:
            continue
        models_here = []
        for (model_name, family), ms in model_sub.groupby(["model", "family"]):
            r_model = safe_pearsonr(ms["h_conf"], ms["m_conf"])
            if np.isfinite(r_model):
                models_here.append((model_name, family, r_model))
        if not models_here:
            continue
        base_x = xpos[vi]
        spread = np.linspace(-bar_w * 0.28, bar_w * 0.28, len(models_here))
        for dx, (_, family, r_model) in zip(spread, models_here):
            marker = family_markers.get(family, "o")
            ax.scatter(base_x + dx, r_model, s=40, marker=marker,
                       facecolors="white", edgecolors=color, linewidths=1.0,
                       alpha=0.95, zorder=5)

ax.set_xticks(x_base + bar_w * (n_groups - 1) / 2)
ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], fontsize=10)
ax.set_ylabel("Pearson r (human–model confidence)")
ax.set_title("Human–Model Confidence Correlation by Variant", fontweight="bold")
ax.axhline(0, color="#888", lw=0.6)
ax.legend(fontsize=9, loc="upper right", frameon=True)
plt.tight_layout()
save(
    fig,
    "correlation_by_variant.png",
    latex_aliases=["confidence_correlation_by_variant.png"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: Confidence vs accuracy scatter — one panel per VLM 7B model
#           Saved as vs_accuracy_scatter_{model_name}.png
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 7: Confidence vs accuracy scatter — VLM 7B, per model…")

VLM_7B_MODELS = [
    "Qwen3-VL-8B", "InternVL-8B",
    "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
]

# Load accuracy from model response export (variant C, inst_blind)
resp_df = pd.read_csv(
    ROOT / "analysis/session2/exports/responses_model_inst_blind.csv"
)[["question_id", "model", "accuracy", "variant"]].query("variant == 'C'").copy()

# Per-question confidence (variant C only) for each individual model
conf_c = (model_df[model_df["variant"] == "C"]
          .groupby(["question_id", "model"])["confidence"]
          .mean().reset_index())

# Merge with accuracy
scatter_src = conf_c.merge(resp_df[["question_id", "model", "accuracy"]],
                           on=["question_id", "model"], how="inner")
scatter_src = scatter_src[scatter_src["model"].isin(VLM_7B_MODELS)].copy()

# Per-question mean accuracy per model
scatter_pq = (scatter_src.groupby(["question_id", "model"])
              .agg(confidence=("confidence", "mean"), accuracy=("accuracy", "mean"))
              .reset_index())

# ── Build summary statistics table ───────────────────────────────────────────
table_rows = []
for model_name in VLM_7B_MODELS:
    sub = scatter_pq[scatter_pq["model"] == model_name]
    if sub.empty:
        continue
    x, y = sub["confidence"].values, sub["accuracy"].values
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    # Significance stars
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    table_rows.append({
        "Model": model_name,
        "n": len(sub),
        "Mean conf.": round(float(x.mean()), 3),
        "Mean acc.": round(float(y.mean()), 3),
        "Slope": round(float(slope), 3),
        "r": round(float(r), 3),
        "p": round(float(p), 4),
        "Sig.": sig,
    })

summary_tbl = pd.DataFrame(table_rows)
print("\nVLM 7B — Confidence vs Accuracy Summary:")
print(summary_tbl.to_string(index=False))
summary_tbl.to_csv(OUT_DIR / "vs_accuracy_stats_vlm7b.csv", index=False)

# ── Render as a figure table ──────────────────────────────────────────────────
col_labels = list(summary_tbl.columns)
cell_vals = summary_tbl.values.tolist()

fig_tbl, ax_tbl = plt.subplots(figsize=(len(col_labels) * 1.5, len(table_rows) * 0.55 + 0.8))
ax_tbl.axis("off")
tbl = ax_tbl.table(
    cellText=cell_vals,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.6)

# Color header row
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Color significance column and highlight sig rows
for i, row in enumerate(table_rows):
    sig_val = row["Sig."]
    bg = "#d5e8d4" if sig_val != "n.s." else "#f8d7da"
    for j in range(len(col_labels)):
        tbl[i + 1, j].set_facecolor(bg)
    # Bold the r column
    tbl[i + 1, col_labels.index("r")].set_text_props(fontweight="bold")
    tbl[i + 1, col_labels.index("Sig.")].set_text_props(
        fontweight="bold",
        color="#2e7d32" if sig_val != "n.s." else "#c62828"
    )

ax_tbl.set_title("VLM 7B: Confidence vs Accuracy (inst-blind, variant C)",
                 fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
save(fig_tbl, "vs_accuracy_table_vlm7b.png")

n_models = len(VLM_7B_MODELS)
fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.8), sharey=True)

for ax, model_name in zip(axes, VLM_7B_MODELS):
    sub = scatter_pq[scatter_pq["model"] == model_name]
    if sub.empty:
        ax.set_visible(False)
        continue

    x = sub["confidence"].values
    y = sub["accuracy"].values
    r, p = stats.pearsonr(x, y)
    color = MODEL_FAMILY_COLORS.get(MODEL_FAMILY.get(model_name, ""), "#555")

    ax.scatter(x, y, alpha=0.5, s=22, color=color, edgecolors="none")
    m_slope, b_int = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m_slope * xr + b_int, color=color, lw=1.8)
    ax.axhline(y.mean(), color="#aaa", lw=0.8, ls="--")

    ax.set_xlabel("Mean token probability")
    ax.set_title(f"{model_name}\nr={r:.2f}  p={p:.3f}", fontweight="bold")

axes[0].set_ylabel("Accuracy (per question mean)")
fig.suptitle("Confidence vs Accuracy — VLM 7B models (inst-blind, variant C)",
             fontsize=12, y=1.02)
plt.tight_layout()
save(fig, "vs_accuracy_scatter_vlm7b.png")


print("\nDone. All figures saved to:", OUT_DIR)
