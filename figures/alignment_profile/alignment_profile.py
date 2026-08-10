"""
Alignment profile lineplot.

Shows all 6 alignment metrics on the x-axis, with:
  - Colored lines per model (grouped by architecture class)
  - Black star markers for individual human participants (LOO computation)
  - Gray band = p5–p95 of per-participant LOO values

Metrics (all normalised to [0,1], higher = more human-like):
  1. Yes/No JS divergence  (inverted: lower JS → higher score)
  2. Yes/No Mode match
  3. Count JS divergence   (inverted)
  4. Count Mode match
  5. Free-text HM SBERT
  6. Spearman ρ (human–model Δ across op groups)

Output: figures/alignment_profile/alignment_profile.png
        latex/AAAI2026/LaTeX/figures/alignment_profile/alignment_profile.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from notebooks.helpers import (
    load_human_responses,
    _clean_answer_series,
    _mark_abstentions,
    _prepare_distribution_category,
    _distribution_stats_from_counts,
)
from analysis.utils.vqa import VQAAnswerMapper

# ── constants ─────────────────────────────────────────────────────────────────
EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "figures/alignment_profile"
OUT_DIR.mkdir(exist_ok=True)

MODEL_ORDER = {
    "VLM": ["InternVL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna", "Qwen3-VL-8B"],
    "Backbone Decoder": [
        "InternVL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)",
        "LLaVA-Vicuna (LM)", "Qwen3-VL-8B (LM)",
    ],
    "Standalone LLM": [
        "Qwen2.5-7B-Instruct", "Qwen3-8B", "Qwen3-8B (think)", "Mistral-7B", "Vicuna-7B",
    ],
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

GROUP_COLORS = {
    "VLM":              "#E53935",  # canonical from analysis/utils/constants.py
    "Backbone Decoder": "#E67E22",
    "Standalone LLM":   "#2E7D32",
}

# Markers per model within each group (same order as MODEL_ORDER lists)
GROUP_MARKERS = {
    "VLM":             ["o", "s", "^", "D", "v"],
    "Backbone Decoder":["o", "s", "^", "D", "v"],
    "Standalone LLM":  ["o", "s", "^", "D", "v"],
}

METRIC_LABELS = ["Y/N JS↓", "Y/N Mode↑", "Count JS↓", "Count Mode↑", "SBERT↑", "ρ (op)↑"]
HIGHER_IS_BETTER = [False, True, False, True, True, True]

# ── helpers ───────────────────────────────────────────────────────────────────
def _qid2atype() -> dict[int, str]:
    mapper = VQAAnswerMapper()
    mapper._load()
    return {int(qid): ann.get("answer_type", "other") for qid, ann in mapper.annotations.items()}


def _read_model_responses(condition: str) -> pd.DataFrame:
    return pd.read_csv(EXPORTS / f"responses_model_{condition}.csv")


# ── model JS/Mode (inst_blind, pooled across variants) ────────────────────────
def compute_model_js_mode(condition: str = "inst_blind") -> dict[str, dict]:
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


# ── per-participant JS/Mode LOO ───────────────────────────────────────────────
def compute_human_js_mode_loo(human_all: pd.DataFrame) -> dict[str, dict]:
    """
    For each participant, treat their answers as a 'model' and compute JS / Mode
    against the LOO pool (remaining participants), per question-variant, averaged.
    """
    qid2atype = _qid2atype()
    human_all = human_all.copy()
    human_all["answer_type"] = human_all["question_id"].astype(int).map(qid2atype).fillna("other")
    human_all["clean_response"] = _clean_answer_series(human_all["response"])
    human_all = _mark_abstentions(human_all)
    human_all = human_all[~human_all["is_abstained"]].copy()

    participants = human_all["participant"].unique()
    result: dict[str, dict] = {}

    for atype in ("yes/no", "number"):
        sub = human_all[human_all["answer_type"] == atype].copy()
        if atype == "yes/no":
            category_order = ["yes", "no"]
            sub["category"] = sub["clean_response"].apply(
                lambda x: "yes" if x in {"yes"} else ("no" if x in {"no"} else None)
            )
        else:  # number
            def _to_count_bin(x):
                try:
                    v = int(float(x))
                    if v == 0: return "0"
                    if v == 1: return "1"
                    if v <= 3: return "2-3"
                    return "4+"
                except Exception:
                    return None
            sub["category"] = sub["clean_response"].apply(_to_count_bin)
            category_order = ["0", "1", "2-3", "4+"]

        sub = sub[sub["category"].notna()].copy()

        for p in participants:
            p_sub   = sub[sub["participant"] == p]
            loo_sub = sub[sub["participant"] != p]
            js_vals, mode_hits = [], []
            for (qid, variant), loo_grp in loo_sub.groupby(["question_id", "variant"]):
                p_grp = p_sub[
                    (p_sub["question_id"].astype(int) == int(qid))
                    & (p_sub["variant"].astype(str) == str(variant))
                ]
                if p_grp.empty or loo_grp.empty:
                    continue
                loo_c = loo_grp["category"].value_counts().reindex(category_order, fill_value=0)
                p_c   = p_grp["category"].value_counts().reindex(category_order, fill_value=0)
                if loo_c.sum() == 0 or p_c.sum() == 0:
                    continue
                # JS between participant (singleton or small) and LOO pool
                p_dist   = p_c.values.astype(float);   p_dist   /= p_dist.sum()
                loo_dist = loo_c.values.astype(float); loo_dist /= loo_dist.sum()
                js = float(jensenshannon(p_dist, loo_dist) ** 2)
                js_vals.append(js)
                modal = set(loo_c[loo_c == loo_c.max()].index)
                mode_hits.append(float(p_c.idxmax() in modal))

            if p not in result:
                result[p] = {}
            result[p][(atype, "JS")]   = float(np.mean(js_vals))   if js_vals   else float("nan")
            result[p][(atype, "Mode")] = float(np.mean(mode_hits)) if mode_hits else float("nan")

    return result


# ── per-question Pearson r: HM/HH SBERT level correlation ────────────────────
def compute_pearson_r(pair_cache: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """
    For each model: Pearson r between per-(question, variant) mean HM SBERT
    and the corresponding per-(question, variant) mean HH SBERT (human reference).
    For each human participant: same but treating them as the 'model' vs group mean.
    Returns (model_r, human_r).
    """
    from scipy.stats import pearsonr as _pearsonr
    qid2atype = _qid2atype()
    ft_qids = {qid for qid, atype in qid2atype.items() if atype == "other"}

    hh = pair_cache[
        (pair_cache["pair_type"] == "HH")
        & (pair_cache["question_id"].isin(ft_qids))
    ].copy()
    hm = pair_cache[
        (pair_cache["pair_type"] == "HM")
        & (pair_cache["question_id"].isin(ft_qids))
    ].copy()

    # Group mean HH SBERT per (question_id, variant) — reference curve
    ref = hh.groupby(["question_id", "variant"])["sbert_score"].mean()

    # Model r
    model_r: dict[str, float] = {}
    for model, grp in hm.groupby("subject_2"):
        if model not in ALL_MODELS:
            continue
        m_mean = grp.groupby(["question_id", "variant"])["sbert_score"].mean()
        common = m_mean.index.intersection(ref.index)
        if len(common) < 20:
            continue
        r, _ = _pearsonr(m_mean[common], ref[common])
        model_r[model] = float(r)

    # Per-participant r (LOO: participant vs remaining group mean)
    human_r: dict[str, float] = {}
    for p, grp in hh.groupby("subject_1"):
        p_mean  = grp.groupby(["question_id", "variant"])["sbert_score"].mean()
        loo_ref = hh[hh["subject_1"] != p].groupby(
            ["question_id", "variant"])["sbert_score"].mean()
        common  = p_mean.index.intersection(loo_ref.index)
        if len(common) < 20:
            continue
        r, _ = _pearsonr(p_mean[common], loo_ref[common])
        human_r[p] = float(r)

    return model_r, human_r


# ── Spearman ρ: op-group Δ(C→A) concordance ─────────────────────────────────
def _op_delta_df(pc: pd.DataFrame, pair_type: str, subject_col: str,
                 ft_qids: set) -> pd.DataFrame:
    sub = pc[
        (pc["pair_type"] == pair_type)
        & (pc["variant"].isin(["C", "A"]))
        & (pc["question_id"].isin(ft_qids))
    ].copy()
    rows = []
    for (subj, op), grp in sub.groupby([subject_col, "op"]):
        for v in ("C", "A"):
            vg = grp[grp["variant"] == v]
            if vg.empty:
                continue
            rows.append({"subject": subj, "op": op, "variant": v,
                         "mean_sbert": vg["sbert_score"].mean()})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["subject", "op"], columns="variant",
                           values="mean_sbert").reset_index()
    if "C" not in pivot.columns or "A" not in pivot.columns:
        return pd.DataFrame()
    pivot["delta"] = pivot["A"] - pivot["C"]
    return pivot[["subject", "op", "delta"]]


def compute_rho(pair_cache: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Returns (model_rho, human_participant_rho)."""
    qid2atype = _qid2atype()
    ft_qids = {qid for qid, atype in qid2atype.items() if atype == "other"}

    hh_delta = _op_delta_df(pair_cache, "HH", "subject_1", ft_qids)
    hm_delta = _op_delta_df(pair_cache, "HM", "subject_2", ft_qids)
    if hh_delta.empty or hm_delta.empty:
        return {}, {}

    human_mean = hh_delta.groupby("op")["delta"].mean()

    # Per-model rho
    model_rho: dict[str, float] = {}
    for model, grp in hm_delta.groupby("subject"):
        if model not in ALL_MODELS:
            continue
        mm = grp.groupby("op")["delta"].mean()
        common = human_mean.index.intersection(mm.index)
        if len(common) < 4:
            continue
        rho, _ = spearmanr(human_mean[common], mm[common])
        model_rho[model] = float(rho)

    # Per-participant rho (LOO: each participant vs remaining mean)
    human_rho: dict[str, float] = {}
    participants = hh_delta["subject"].unique()
    for p in participants:
        p_delta  = hh_delta[hh_delta["subject"] == p].groupby("op")["delta"].mean()
        loo_mean = hh_delta[hh_delta["subject"] != p].groupby("op")["delta"].mean()
        common   = p_delta.index.intersection(loo_mean.index)
        if len(common) < 4:
            continue
        rho, _ = spearmanr(loo_mean[common], p_delta[common])
        human_rho[p] = float(rho)

    return model_rho, human_rho


# ── normalise ─────────────────────────────────────────────────────────────────
def normalise_col(vals: np.ndarray, higher_is_better: bool,
                  lo: float | None = None, hi: float | None = None) -> np.ndarray:
    finite = vals[np.isfinite(vals)]
    if len(finite) < 2:
        return np.full_like(vals, 0.5)
    lo = finite.min() if lo is None else lo
    hi = finite.max() if hi is None else hi
    if hi == lo:
        return np.full_like(vals, 0.5)
    normed = (vals - lo) / (hi - lo)
    return np.clip(normed if higher_is_better else (1.0 - normed), 0.0, 1.0)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading data...")
    pair_cache = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    human_all  = load_human_responses()

    print("Computing model JS/Mode...")
    model_js_mode = compute_model_js_mode("inst_blind")

    print("Computing per-participant JS/Mode LOO...")
    human_js_mode = compute_human_js_mode_loo(human_all)

    print("Computing per-question Pearson r...")
    model_pearsonr, human_pearsonr = compute_pearson_r(pair_cache)

    # ── assemble raw score matrices ───────────────────────────────────────────
    METRICS = [
        ("yes/no",   "JS", False),  # Y/N JS
        ("number",   "JS", False),  # Count JS
        ("pearsonr", None, True),   # per-question Pearson r (HM vs HH SBERT)
    ]

    def _model_raw(model: str) -> list[float]:
        row = []
        for atype, metric, _ in METRICS:
            if atype == "pearsonr":
                row.append(model_pearsonr.get(model, float("nan")))
            else:
                row.append(model_js_mode.get(model, {}).get((atype, metric), float("nan")))
        return row

    def _human_raw(p: str) -> list[float]:
        row = []
        for atype, metric, _ in METRICS:
            if atype == "pearsonr":
                row.append(human_pearsonr.get(p, float("nan")))
            else:
                row.append(human_js_mode.get(p, {}).get((atype, metric), float("nan")))
        return row

    model_mat = np.array([_model_raw(m) for m in ALL_MODELS], dtype=float)  # (15, 6)
    participants = sorted(set(human_js_mode) | set(human_pearsonr))
    human_mat   = np.array([_human_raw(p) for p in participants], dtype=float)  # (N, 6)

    # Normalise each metric using the joint min–max of models + humans → always [0, 1].
    # No direction: human-like means landing in the human target band, not scoring high/low.
    norm_model = np.empty_like(model_mat)
    norm_human = np.empty_like(human_mat)
    human_lo  = np.empty(len(METRICS))  # p10 of human dist (normalised)
    human_hi  = np.empty(len(METRICS))  # p90 of human dist (normalised)
    human_med = np.empty(len(METRICS))  # median of human dist (normalised)
    for j in range(len(METRICS)):
        combined_col = np.concatenate([model_mat[:, j], human_mat[:, j]])
        finite = combined_col[np.isfinite(combined_col)]
        lo = finite.min() if len(finite) else 0.0
        hi = finite.max() if len(finite) else 1.0
        if hi == lo:
            norm_model[:, j] = 0.5
            norm_human[:, j] = 0.5
            human_lo[j] = human_hi[j] = human_med[j] = 0.5
        else:
            norm_model[:, j] = np.clip((model_mat[:, j] - lo) / (hi - lo), 0.0, 1.0)
            norm_human[:, j] = np.clip((human_mat[:, j] - lo) / (hi - lo), 0.0, 1.0)
            h_finite = human_mat[:, j][np.isfinite(human_mat[:, j])]
            human_lo[j]  = (np.percentile(h_finite, 5) - lo) / (hi - lo)
            human_hi[j]  = (np.percentile(h_finite, 95) - lo) / (hi - lo)
            human_med[j] = (np.median(h_finite) - lo) / (hi - lo)

    # ── plot (horizontal: metrics on y-axis, score on x-axis) ────────────────
    n_metrics = len(METRICS)
    y = np.arange(n_metrics)
    METRIC_NAMES = ["Y/N JS", "Count JS", "SBERT r"]

    # Markers consistent with the scatter figure (MODEL_FAMILIES in generate_variant_scatter.py)
    MODEL_STYLE = {
        "InternVL-8B":        "o", "InternVL-8B (LM)":   "o",
        "Qwen3-VL-8B":        "X", "Qwen3-VL-8B (LM)":  "X", "Qwen3-8B":          "X",
        "LLaVA-1.5-7B":       "D", "LLaVA-1.5 (LM)":    "D",
        "LLaVA-Mistral":      "^", "LLaVA-Mistral (LM)": "^", "Mistral-7B":        "^",
        "LLaVA-Vicuna":       "v", "LLaVA-Vicuna (LM)":  "v", "Vicuna-7B":         "v",
        "Qwen3-8B (think)":   "P",
        "Qwen2.5-7B-Instruct":"s",
    }

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.subplots_adjust(bottom=0.52)  # room for two legend rows below

    # Human target zone: shaded strip (p10–p90) + median line per metric
    for j in range(n_metrics):
        ax.barh(y[j], human_hi[j] - human_lo[j], left=human_lo[j],
                height=0.55, color="#dddddd", edgecolor="#aaaaaa",
                linewidth=0.8, zorder=2, align="center")
        ax.plot([human_med[j], human_med[j]],
                [y[j] - 0.275, y[j] + 0.275],
                color="black", lw=2.0, zorder=3)

    # Human participant stars (visible in front)
    for i, p_vals in enumerate(norm_human):
        ax.scatter(p_vals, y, marker="*", s=60, color="#555555",
                   alpha=0.45, zorder=6, linewidths=0.0)

    # Individual model dots — jittered within ±0.22 band, one row per metric
    legend_handles = []
    all_model_list = [(grp, m)
                      for grp, models in MODEL_ORDER.items()
                      for m in models]
    n_models_total = len(all_model_list)
    y_offsets = np.linspace(-0.30, 0.30, n_models_total)

    for i, (grp, model) in enumerate(all_model_list):
        color  = GROUP_COLORS[grp]
        mk     = MODEL_STYLE.get(model, "o")
        idx    = ALL_MODELS.index(model)
        vals   = norm_model[idx]
        offset = y_offsets[i]
        ax.scatter(vals, y + offset, color=color, marker=mk,
                   s=38, alpha=0.85, zorder=4,
                   linewidths=0.5, edgecolors="white")
        legend_handles.append(
            mlines.Line2D([], [], color=color, marker=mk, lw=0,
                          markersize=6, markeredgecolor="white",
                          markeredgewidth=0.4, label=DISPLAY[model])
        )

    # Axes
    ax.set_yticks(y)
    ax.set_yticklabels(METRIC_NAMES, fontsize=10)
    ax.set_xlabel("Normalised score  (shaded band = human 95% range)", fontsize=9)
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.65, n_metrics - 0.35)
    ax.grid(axis="x", alpha=0.20, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    # Legend row 1: colors = group
    color_handles = [
        mlines.Line2D([], [], color=c, marker="o", lw=0, markersize=7,
                      markeredgecolor="white", markeredgewidth=0.5, label=g)
        for g, c in GROUP_COLORS.items()
    ]
    human_star_h = mlines.Line2D([], [], color="#555555", marker="*", lw=0,
                                  markersize=7, alpha=0.7, label="Human participants")
    box_handle = plt.Rectangle((0, 0), 1, 1, fc="#dddddd", ec="#aaaaaa",
                                linewidth=0.8, label="Human 95% range")
    # Row 1: marker shapes = model family
    FAMILY_LEGEND = [
        ("o", "InternVL"), ("X", "Qwen3-VL / Qwen3"), ("D", "LLaVA-1.5"),
        ("^", "LLaVA-Mistral / Mistral"), ("v", "LLaVA-Vicuna / Vicuna"),
        ("P", "Qwen3 (think)"), ("s", "Qwen2.5"),
    ]
    marker_handles = [
        mlines.Line2D([], [], color="#555555", marker=mk, lw=0,
                      markersize=7, markeredgecolor="white", markeredgewidth=0.5,
                      label=name)
        for mk, name in FAMILY_LEGEND
    ]
    leg1 = ax.legend(handles=marker_handles,
                     loc="upper center", bbox_to_anchor=(0.5, -0.22),
                     ncol=4, fontsize=8, frameon=False,
                     handlelength=1.2, columnspacing=0.9, handletextpad=0.5)
    ax.add_artist(leg1)

    # Row 2: colors = group + human reference
    ax.legend(handles=color_handles + [human_star_h, box_handle],
              loc="upper center", bbox_to_anchor=(0.5, -0.42),
              ncol=5, fontsize=8, frameon=False,
              handlelength=1.2, columnspacing=0.9, handletextpad=0.5)


    plt.tight_layout()
    out_path = OUT_DIR / "alignment_profile.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")

    latex_dir = ROOT / "latex/AAAI2026/LaTeX/figures/alignment_profile"
    latex_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(out_path, latex_dir / "alignment_profile.png")
    print(f"Copied to: {latex_dir / 'alignment_profile.png'}")

    # Print summary
    print("\n=== Raw model scores ===")
    df = pd.DataFrame(model_mat, index=ALL_MODELS,
                      columns=["YN-JS", "YN-Mode", "Ct-JS", "Ct-Mode", "SBERT", "rho"])
    print(df.round(3).to_string())
    print("\n=== Human participant score ranges ===")
    hdf = pd.DataFrame(human_mat, index=participants,
                       columns=["YN-JS", "YN-Mode", "Ct-JS", "Ct-Mode", "SBERT", "rho"])
    print(hdf.describe().round(3).to_string())


if __name__ == "__main__":
    main()
