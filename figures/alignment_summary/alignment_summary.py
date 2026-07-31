"""
Comprehensive alignment summary figure.

Computes per-model (7/8B set) scores on:
  1. Yes/No JS divergence (inst_blind, lower = better)
  2. Yes/No Mode match   (inst_blind, higher = better)
  3. Count JS divergence (inst_blind, lower = better)
  4. Count Mode match    (inst_blind, higher = better)
  5. Free-text HM SBERT  (inst_blind, higher = better)
  6. Spearman rho        (human vs model Delta across op groups, higher = better)

Normalises each metric to [0,1] (1 = best), computes composite mean,
and renders a heatmap with a final composite column.

Output: figures/alignment_summary/alignment_summary.png
        latex/AAAI2026/LaTeX/figures/alignment_summary/alignment_summary.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from notebooks.helpers import (
    load_human_responses,
    _clean_answer_series,
    _answer_type_subset,
    _mark_abstentions,
    _prepare_distribution_category,
    _distribution_stats_from_counts,
)
from analysis.utils.vqa import VQAAnswerMapper

# ── constants ─────────────────────────────────────────────────────────────────
EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "figures/alignment_summary"
OUT_DIR.mkdir(exist_ok=True)

MODEL_ORDER = {
    "VLM": [
        "Qwen3-VL-8B", "InternVL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"
    ],
    "Backbone Decoder": [
        "Qwen3-VL-8B (LM)", "InternVL-8B (LM)", "LLaVA-1.5 (LM)",
        "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    ],
    "Standalone LLM": [
        "Qwen2.5-7B-Instruct", "Qwen3-8B", "Qwen3-8B (think)", "Mistral-7B", "Vicuna-7B"
    ],
}
GROUP_COLORS = {
    "VLM": "#4878CF",
    "Backbone Decoder": "#6ACC65",
    "Standalone LLM": "#D65F5F",
}
ALL_MODELS = [m for grp in MODEL_ORDER.values() for m in grp]

DISPLAY = {
    "Qwen3-VL-8B": "Qwen3-VL",
    "InternVL-8B": "InternVL",
    "LLaVA-1.5-7B": "LLaVA-1.5",
    "LLaVA-Mistral": "LLaVA-Mistral",
    "LLaVA-Vicuna": "LLaVA-Vicuna",
    "Qwen3-VL-8B (LM)": "Qwen3-VL (LM)",
    "InternVL-8B (LM)": "InternVL (LM)",
    "LLaVA-1.5 (LM)": "LLaVA-1.5 (LM)",
    "LLaVA-Mistral (LM)": "LLaVA-Mistral (LM)",
    "LLaVA-Vicuna (LM)": "LLaVA-Vicuna (LM)",
    "Qwen2.5-7B-Instruct": "Qwen2.5",
    "Qwen3-8B": "Qwen3",
    "Qwen3-8B (think)": "Qwen3 (think)",
    "Mistral-7B": "Mistral",
    "Vicuna-7B": "Vicuna",
}

OP_GROUPS = ["act", "attr", "cause", "comp", "count", "ident", "know", "spat", "temp", "text"]


# ── helpers ───────────────────────────────────────────────────────────────────
def _qid2atype() -> dict[int, str]:
    mapper = VQAAnswerMapper()
    mapper._load()
    return {int(qid): ann.get("answer_type", "other") for qid, ann in mapper.annotations.items()}


def _read_model_responses(condition: str) -> pd.DataFrame:
    return pd.read_csv(EXPORTS / f"responses_model_{condition}.csv")


# ── metric 1-4: JS and Mode for yes/no and count ──────────────────────────────
def compute_js_mode(condition: str) -> dict[str, dict]:
    qid2atype = _qid2atype()
    human_all = load_human_responses()
    model_all = _read_model_responses(condition)
    variants = ("C", "B", "A")
    model_keep = set(ALL_MODELS)

    human_all = human_all[human_all["variant"].isin(variants)].copy()
    model_all = model_all[
        model_all["variant"].isin(variants) & model_all["model"].isin(model_keep)
    ].copy()
    human_all["answer_type"] = human_all["question_id"].astype(int).map(qid2atype).fillna("other")
    model_all["answer_type"] = model_all["question_id"].astype(int).map(qid2atype).fillna("other")
    human_all["clean_response"] = _clean_answer_series(human_all["response"])
    model_all["clean_response"] = _clean_answer_series(model_all["response"])
    human_all = _mark_abstentions(human_all)
    model_all = _mark_abstentions(model_all)

    result: dict[str, dict] = {}
    for model_name, model_sub_all in model_all.groupby("model"):
        if model_name not in model_keep:
            continue
        row: dict = {}
        for atype in ("yes/no", "number"):
            human_sub, model_sub, category_order = _prepare_distribution_category(
                human_all.copy(), model_sub_all.copy(), atype
            )
            model_sub = model_sub[~model_sub["is_abstained"]].copy()
            kept = set(zip(model_sub["question_id"].astype(int), model_sub["variant"].astype(str)))
            human_sub = human_sub[
                [(int(q), str(v)) in kept
                 for q, v in zip(human_sub["question_id"], human_sub["variant"])]
            ].copy()
            human_sub = human_sub[~human_sub["is_abstained"]].copy()
            js_vals, mode_hits = [], []
            for (qid, variant), hq in human_sub.groupby(["question_id", "variant"]):
                mq = model_sub[
                    (model_sub["question_id"].astype(int) == int(qid))
                    & (model_sub["variant"].astype(str) == str(variant))
                ]
                if hq.empty or mq.empty:
                    continue
                hc = hq["category"].value_counts().reindex(category_order, fill_value=0)
                mc = mq["category"].value_counts().reindex(category_order, fill_value=0)
                if hc.sum() == 0 or mc.sum() == 0:
                    continue
                _, _, js, _, _ = _distribution_stats_from_counts(hc, mc)
                js_vals.append(js)
                modal = set(hc[hc == hc.max()].index)
                mode_hits.append(float(mc.idxmax() in modal))
            row[(atype, "JS")] = float(np.mean(js_vals)) if js_vals else float("nan")
            row[(atype, "Mode")] = float(np.mean(mode_hits)) if mode_hits else float("nan")
        result[model_name] = row
    return result


# ── metric 5: mean free-text HM SBERT ─────────────────────────────────────────
def compute_sbert(pair_cache: pd.DataFrame) -> dict[str, float]:
    hm = pair_cache[pair_cache["pair_type"] == "HM"].copy()
    result = {}
    for model, grp in hm.groupby("subject_2"):
        if model not in ALL_MODELS:
            continue
        result[model] = float(grp["sbert_score"].mean())
    return result


# ── metric 6: Spearman rho (human vs model Delta across op groups) ────────────
def compute_rho(pair_cache_inst: pd.DataFrame, pair_cache_blind: pd.DataFrame) -> dict[str, float]:
    """
    For each model and each op group, compute mean HM SBERT for variant C and A.
    Delta = mean_A - mean_C.
    Correlate model Delta vector with human HH Delta vector via Spearman rho.
    Free-text (answer_type='other') excluded to avoid distribution questions.
    Include all questions (no answer_type filter) as the paper does for this measure.
    """
    qid2atype = _qid2atype()

    def _op_delta(pc: pd.DataFrame, pair_type: str, subject_col: str) -> pd.DataFrame:
        sub = pc[(pc["pair_type"] == pair_type) & (pc["variant"].isin(["C", "A"]))].copy()
        sub["atype"] = sub["question_id"].astype(int).map(qid2atype).fillna("other")
        sub = sub[sub["atype"] == "other"]  # free-text only for SBERT
        rows = []
        for (subj, op), grp in sub.groupby([subject_col, "op"]):
            for v in ("C", "A"):
                vg = grp[grp["variant"] == v]
                if vg.empty:
                    continue
                rows.append({"subject": subj, "op": op, "variant": v,
                              "mean_sbert": vg["sbert_score"].mean()})
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame()
        pivot = df.pivot_table(index=["subject", "op"], columns="variant",
                               values="mean_sbert").reset_index()
        if "C" not in pivot.columns or "A" not in pivot.columns:
            return pd.DataFrame()
        pivot["delta"] = pivot["A"] - pivot["C"]
        return pivot[["subject", "op", "delta"]]

    # Human delta (HH, use subject_1 as the "subject")
    hh_delta = _op_delta(pair_cache_inst, "HH", "subject_1")
    if hh_delta.empty:
        return {}
    human_mean = hh_delta.groupby("op")["delta"].mean()

    result = {}
    # Combine inst and blind pair caches for models (use inst as primary)
    hm_delta = _op_delta(pair_cache_inst, "HM", "subject_2")
    if hm_delta.empty:
        return result
    for model, grp in hm_delta.groupby("subject"):
        if model not in ALL_MODELS:
            continue
        model_mean = grp.groupby("op")["delta"].mean()
        common_ops = human_mean.index.intersection(model_mean.index)
        if len(common_ops) < 4:
            continue
        rho, _ = spearmanr(human_mean[common_ops], model_mean[common_ops])
        result[model] = float(rho)
    return result


# ── normalise and rank ─────────────────────────────────────────────────────────
def normalise(vals: np.ndarray, higher_is_better: bool) -> np.ndarray:
    finite = vals[np.isfinite(vals)]
    if len(finite) < 2:
        return np.full_like(vals, 0.5)
    lo, hi = finite.min(), finite.max()
    if hi == lo:
        return np.full_like(vals, 0.5)
    normed = (vals - lo) / (hi - lo)
    return normed if higher_is_better else (1.0 - normed)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading data...")
    pair_cache_inst  = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    pair_cache_blind = pd.read_parquet(EXPORTS / "pair_cache_cleaned_blind.parquet")

    # Free-text only for SBERT
    qid2atype = _qid2atype()
    ft_qids = {qid for qid, atype in qid2atype.items() if atype == "other"}
    pc_inst_ft = pair_cache_inst[pair_cache_inst["question_id"].isin(ft_qids)].copy()

    print("Computing JS/Mode...")
    js_mode = compute_js_mode("inst_blind")

    print("Computing SBERT...")
    sbert_scores = compute_sbert(pc_inst_ft)

    print("Computing Spearman rho...")
    rho_scores = compute_rho(pair_cache_inst, pair_cache_blind)

    # Build metric matrix
    METRICS = [
        ("Y/N JS",    "yes/no", "JS",   False),
        ("Y/N Mode",  "yes/no", "Mode", True),
        ("Count JS",  "number", "JS",   False),
        ("Count Mode","number", "Mode", True),
        ("SBERT",     None,     None,   True),
        ("ρ (op)",    None,     None,   True),
    ]

    raw: dict[str, list] = {m: [] for name, *_ in METRICS for m in [name]}
    rows_data = []
    for model in ALL_MODELS:
        row = {}
        for name, atype, metric, _ in METRICS:
            if name == "SBERT":
                row[name] = sbert_scores.get(model, float("nan"))
            elif name == "ρ (op)":
                row[name] = rho_scores.get(model, float("nan"))
            else:
                row[name] = js_mode.get(model, {}).get((atype, metric), float("nan"))
        rows_data.append(row)

    df = pd.DataFrame(rows_data, index=ALL_MODELS)

    # Normalise per metric
    norm_df = df.copy()
    for (name, _, _, higher), col in zip(METRICS, df.columns):
        norm_df[col] = normalise(df[col].values.astype(float), higher)

    # Composite: mean of all normalised scores
    norm_df["Composite"] = norm_df.mean(axis=1)
    df["Composite"] = norm_df["Composite"]

    # ── plot ──────────────────────────────────────────────────────────────────
    col_labels = [name for name, *_ in METRICS] + ["Composite"]
    n_models = len(ALL_MODELS)
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Heatmap
    plot_data = norm_df[col_labels].values.astype(float)
    cmap = plt.cm.RdYlGn
    im = ax.imshow(plot_data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Cell text
    for i, model in enumerate(ALL_MODELS):
        for j, col in enumerate(col_labels):
            raw_val = df.at[model, col]
            txt = f"{raw_val:.2f}" if np.isfinite(raw_val) else "—"
            normed = plot_data[i, j]
            text_color = "white" if (normed < 0.25 or normed > 0.75) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="normal")

    # Axes
    display_models = [DISPLAY[m] for m in ALL_MODELS]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(display_models, fontsize=8.5)

    # Group colour bars on left
    group_boundaries = []
    idx = 0
    for grp, models in MODEL_ORDER.items():
        start = idx
        end = idx + len(models)
        group_boundaries.append((start, end, grp))
        idx = end

    for start, end, grp in group_boundaries:
        color = GROUP_COLORS[grp]
        mid = (start + end - 1) / 2
        ax.annotate("", xy=(-0.55, start - 0.48), xytext=(-0.55, end - 0.52),
                    xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", color=color, lw=4),
                    annotation_clip=False)
        ax.text(-0.015, mid, grp, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8, color=color, fontweight="bold",
                rotation=0)

    # Separator line before Composite column
    ax.axvline(x=n_cols - 1.5, color="black", linewidth=1.5, linestyle="--", alpha=0.5)

    # Separator lines between groups
    for _, end, _ in group_boundaries[:-1]:
        ax.axhline(y=end - 0.5, color="white", linewidth=1.5)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Normalised score (1 = best)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Human Alignment Summary — 7/8B Models (with instruction)", fontsize=10, pad=10)

    plt.tight_layout()

    out_path = OUT_DIR / "alignment_summary.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also copy to LaTeX figures dir
    latex_dir = ROOT / "latex/AAAI2026/LaTeX/figures/alignment_summary"
    latex_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(out_path, latex_dir / "alignment_summary.png")
    print(f"Copied to: {latex_dir / 'alignment_summary.png'}")

    # Print summary table
    print("\n=== Raw values ===")
    print(df.round(3).to_string())
    print("\n=== Normalised scores ===")
    print(norm_df[col_labels].round(3).to_string())


if __name__ == "__main__":
    main()
