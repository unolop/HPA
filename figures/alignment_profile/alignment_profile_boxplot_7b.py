"""
Boxplot alignment profile — 3 metric panels, one box per group (VLM / Backbone / SA-LLM).
Each box shows the distribution of per-model mean values within that group.
Individual model points overlaid with family color coding.
"""
from __future__ import annotations

import shutil, sys
from pathlib import Path

import matplotlib, matplotlib.pyplot as _mplplt
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import pearsonr as _pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import figures.plot_style  # noqa: F401

# Reuse data loading and constants from barplot script
sys.path.insert(0, str(ROOT / "figures/alignment_profile"))
from alignment_profile_barplot_7b import (
    load_metrics,
    MODEL_ORDER_7B,
    MODEL_HATCH,
    _FAMILY_COLOR,
    _FAMILY_MARKER,
    _fam,
    GROUP_ALPHA,
    HUMAN_YESNO_CI,
    HUMAN_COUNT_CI,
    LBL_YN_JS,
    LBL_COUNT_JS,
    LBL_PEARSONR,
    EXPORTS,
)

OUT_DIR   = ROOT / "figures/alignment_profile"
LATEX_DIR = ROOT / "latex/AAAI2026/LaTeX/figures/alignment_profile"
OUT_DIR.mkdir(exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

GROUPS      = list(MODEL_ORDER_7B.keys())
GROUP_SHORT = {"VLM": "VLM", "Backbone Decoder": "Backbone", "Standalone LLM": "SA-LLM"}

# VLM at top (y=2), Backbone y=1, SA-LLM y=0
Y_CENTER = {grp: float(len(GROUPS) - 1 - i) for i, grp in enumerate(GROUPS)}

_MEDIAN_KEY = {
    "yesno_js":  "yn_median",
    "count_js":  "num_median",
    "pearson_r": "r_median",
    "ft_sbert":  "ft_median",
}


def _draw_hbox(ax, vals_arr, y_c, box_h=0.30, color="#d0d0d0",
               edge_color="#888888", median_color="#333333", lw=0.8):
    """Draw a horizontal box-and-whisker at y_c from a 1-D array of values."""
    q1, q2, q3 = np.percentile(vals_arr, [25, 50, 75])
    iqr = q3 - q1
    wlo = max(vals_arr.min(), q1 - 1.5 * iqr)
    whi = min(vals_arr.max(), q3 + 1.5 * iqr)

    # Box
    rect = plt.Rectangle(
        (q1, y_c - box_h / 2), q3 - q1, box_h,
        facecolor=color, edgecolor=edge_color, linewidth=lw, zorder=2,
    )
    ax.add_patch(rect)
    # Median
    ax.plot([q2, q2], [y_c - box_h / 2, y_c + box_h / 2],
            color=median_color, lw=1.4, zorder=3)
    # Whiskers
    cap_h = box_h * 0.45
    for wx, qx in [(wlo, q1), (whi, q3)]:
        ax.plot([wx, qx], [y_c, y_c], color=edge_color, lw=lw, zorder=2)
        ax.plot([wx, wx], [y_c - cap_h / 2, y_c + cap_h / 2],
                color=edge_color, lw=lw, zorder=2)


def load_bootstrap_pearson_r(n_boot: int = 500, seed: int = 0,
                              ft_only: bool = True) -> tuple[dict, dict]:
    """Bootstrap Pearson r for each model.

    ft_only=True  : free-text questions only (matches group boxplot)
    ft_only=False : all question types
    Returns (boot_r, human_refs) where boot_r[model] = array of n_boot r values.
    """
    from analysis.utils.vqa import VQAAnswerMapper
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    hm = pc[pc["pair_type"] == "HM"]
    hh = pc[pc["pair_type"] == "HH"]

    if ft_only:
        mapper = VQAAnswerMapper(); mapper._load()
        ft_qids = {int(qid) for qid, ann in mapper.annotations.items()
                   if ann.get("answer_type", "other") == "other"}
        hm_ft = hm[hm["question_id"].astype(int).isin(ft_qids)]
        hh_ft = hh[hh["question_id"].astype(int).isin(ft_qids)]
    else:
        hm_ft = hm
        hh_ft = hh

    # Human per-question mean SBERT (reference curve)
    hh_perq = hh_ft.groupby("question_id")["sbert_score"].mean()

    all_models = [m for ms in MODEL_ORDER_7B.values() for m in ms]
    rng = np.random.default_rng(seed)
    boot_r: dict[str, np.ndarray] = {}

    for m in all_models:
        hm_perq = hm_ft[hm_ft["subject_2"] == m].groupby("question_id")["sbert_score"].mean()
        common  = hh_perq.index.intersection(hm_perq.index)
        if len(common) < 10:
            boot_r[m] = np.array([np.nan])
            continue
        x = hh_perq[common].values
        y = hm_perq[common].values
        rs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(x), size=len(x))
            r, _ = _pearsonr(x[idx], y[idx])
            rs.append(r)
        boot_r[m] = np.array(rs)

    # Human LOO bootstrap CI (same method)
    hh_boot: list[float] = []
    participants = hh_ft["subject_1"].unique()
    for p in participants:
        hm_p  = hh_ft[hh_ft["subject_1"] == p].groupby("question_id")["sbert_score"].mean()
        loo_p = hh_ft[hh_ft["subject_1"] != p].groupby("question_id")["sbert_score"].mean()
        common = hm_p.index.intersection(loo_p.index)
        if len(common) < 10:
            continue
        r, _ = _pearsonr(hm_p[common].values, loo_p[common].values)
        hh_boot.append(r)

    human_refs = {
        "r_p5":    float(np.percentile(hh_boot, 5)),
        "r_p95":   float(np.percentile(hh_boot, 95)),
        "r_median":float(np.median(hh_boot)),
    }
    return boot_r, human_refs


def _compute_pearson_r_variants(hm: pd.DataFrame, hh: pd.DataFrame,
                                 qid_filter: set | None = None) -> tuple[dict, dict]:
    """Core: pooled Pearson r over all (question, variant) pairs, plus human LOO refs.

    qid_filter: set of question_ids to keep (None = all questions).
    Returns (v_r, human_r_refs) where v_r[model] = [r_pooled] (single value list
    for API compatibility with callers that do np.mean).
    """
    if qid_filter is not None:
        hm = hm[hm["question_id"].astype(int).isin(qid_filter)]
        hh = hh[hh["question_id"].astype(int).isin(qid_filter)]

    all_models = [m for ms in MODEL_ORDER_7B.values() for m in ms]
    v_r: dict[str, list[float]] = {m: [] for m in all_models}

    # Pool all (question_id, variant) pairs — maximally stable, equal per-pair weight
    ref_pooled = hh.groupby(["question_id", "variant"])["sbert_score"].mean()
    for m in all_models:
        mg = hm[hm["subject_2"] == m]
        mm = mg.groupby(["question_id", "variant"])["sbert_score"].mean()
        common = mm.index.intersection(ref_pooled.index)
        if len(common) >= 20:
            r, _ = _pearsonr(mm[common].values, ref_pooled[common].values)
            v_r[m].append(float(r))

    h_r_vals: list[float] = []
    for p, grp in hh.groupby("subject_1"):
        pm  = grp.groupby(["question_id", "variant"])["sbert_score"].mean()
        loo = hh[hh["subject_1"] != p].groupby(
            ["question_id", "variant"])["sbert_score"].mean()
        common = pm.index.intersection(loo.index)
        if len(common) < 20:
            continue
        r, _ = _pearsonr(pm[common].values, loo[common].values)
        h_r_vals.append(float(r))

    human_r_refs = {
        "r_p5":       float(np.percentile(h_r_vals, 5)),
        "r_p95":      float(np.percentile(h_r_vals, 95)),
        "r_median":   float(np.median(h_r_vals)),
        "r_individual": h_r_vals,
    }
    return v_r, human_r_refs


def load_allq_pearson_r(abstention_filtered: bool = False) -> tuple[dict, dict, dict]:
    """Returns (metrics_r, v_r, human_r_refs) for all question types.

    abstention_filtered=True: drop HM pairs where the model abstained before
    computing Pearson r (mirrors the filtering in the distribution tables).
    """
    from analysis.utils.abstention import classify, is_abstained as _is_abs
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    hm = pc[pc["pair_type"] == "HM"].copy()
    hh = pc[pc["pair_type"] == "HH"]
    if abstention_filtered:
        abs_class = hm["answer_2_clean"].apply(lambda x: classify(str(x), None))
        hm = hm[~abs_class.apply(_is_abs)]
    v_r, human_r_refs = _compute_pearson_r_variants(hm, hh, qid_filter=None)
    all_models = [m for ms in MODEL_ORDER_7B.values() for m in ms]
    metrics_r = {m: {"pearson_r": float(np.mean(v_r[m])) if v_r[m] else np.nan}
                 for m in all_models}
    return metrics_r, v_r, human_r_refs


def load_ft_pearson_r() -> tuple[dict, dict, dict]:
    """Returns (metrics_r, v_r, human_r_refs) for free-text questions only."""
    from analysis.utils.vqa import VQAAnswerMapper
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    hm = pc[pc["pair_type"] == "HM"]
    hh = pc[pc["pair_type"] == "HH"]
    mapper = VQAAnswerMapper(); mapper._load()
    ft_qids = {int(qid) for qid, ann in mapper.annotations.items()
               if ann.get("answer_type", "other") == "other"}
    v_r, human_r_refs = _compute_pearson_r_variants(hm, hh, qid_filter=ft_qids)
    all_models = [m for ms in MODEL_ORDER_7B.values() for m in ms]
    metrics_r = {m: {"pearson_r": float(np.mean(v_r[m])) if v_r[m] else np.nan}
                 for m in all_models}
    return metrics_r, v_r, human_r_refs


VLM_BACKBONE_PAIRS = [
    ("Qwen3-VL-8B",   "Qwen3-VL-8B (LM)"),
    ("InternVL-8B",   "InternVL-8B (LM)"),
    ("LLaVA-1.5-7B",  "LLaVA-1.5 (LM)"),
    ("LLaVA-Mistral", "LLaVA-Mistral (LM)"),
    ("LLaVA-Vicuna",  "LLaVA-Vicuna (LM)"),
]


def make_figure(metrics: dict, human_refs: dict,
                metric_keys: list | None = None,
                overlay_metrics: dict | None = None,
                overlay_human_refs: dict | None = None,
                connect_pairs: bool = False,
                human_individual_r: list | None = None) -> plt.Figure:
    """
    metrics / human_refs       : primary data (solid markers + box)
    overlay_metrics / _refs    : secondary data (hollow markers + dashed ref line)
    """
    _all_cfg = [
        ("yesno_js",  LBL_YN_JS,    (human_refs["yn_p5"],  human_refs["yn_p95"]), None),
        ("count_js",  LBL_COUNT_JS, (human_refs["num_p5"], human_refs["num_p95"]), (0.3, 1.0)),
        ("pearson_r", LBL_PEARSONR, (human_refs["r_p5"],   human_refs["r_p95"]),  (0.5, 1.0)),
    ]
    if metric_keys is not None:
        key_set = set(metric_keys)
        metric_cfg = [t for t in _all_cfg if t[0] in key_set]
    else:
        metric_cfg = _all_cfg

    n_panels = len(metric_cfg)
    fig_w = 5.2 if n_panels == 1 else 2.5 * n_panels
    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_w, 1.9),
        squeeze=False,
        gridspec_kw={"wspace": 0.06, "left": 0.20 if n_panels == 1 else 0.15,
                     "right": 0.97, "top": 0.78, "bottom": 0.12},
    )
    axes = axes[0]

    rng = np.random.default_rng(0)

    for ax_i, (key, xlabel, ci, xlim) in enumerate(metric_cfg):
        ax = axes[ax_i]

        # Human CI shading + primary median line (dotted)
        if ci is not None:
            lo, hi = ci
            ax.axvspan(lo, hi, color="#aaaaaa", alpha=0.18, zorder=0)
            mkey = _MEDIAN_KEY.get(key)
            if mkey and mkey in human_refs:
                ax.axvline(human_refs[mkey], color="#000000", lw=1.0,
                           linestyle=":", alpha=0.85, zorder=3)

        # Overlay (ft-only) human median line (dashed)
        if overlay_human_refs is not None:
            mkey = _MEDIAN_KEY.get(key)
            if mkey and mkey in overlay_human_refs:
                ax.axvline(overlay_human_refs[mkey], color="#000000", lw=1.0,
                           linestyle="--", alpha=0.55, zorder=3)

        dot_pos: dict[str, tuple[float, float]] = {}  # model -> (x, y) for connection lines

        for grp in GROUPS:
            y_c    = Y_CENTER[grp]
            models = MODEL_ORDER_7B[grp]
            vals   = np.array([metrics[m][key] for m in models
                                if np.isfinite(metrics[m][key])])

            if len(vals) >= 2:
                _draw_hbox(ax, vals, y_c)

            # Deterministic vertical spread: use fixed MODEL_ORDER_7B index
            n = len(models)
            spread = 0.22
            y_offsets = {m: float(np.linspace(-spread, spread, n)[i])
                         for i, m in enumerate(models)}

            for m in models:
                color  = _FAMILY_COLOR[_fam(m)]
                marker = _FAMILY_MARKER[_fam(m)]
                jitter = y_offsets.get(m, 0.0)

                val = metrics[m][key]
                if np.isfinite(val):
                    ax.scatter(val, y_c + jitter,
                               color=color, s=90, zorder=4,
                               marker=marker, edgecolors="white",
                               linewidths=0.4, alpha=0.75)
                    dot_pos[m] = (val, y_c + jitter)

                # Hollow marker — overlay (ft-only)
                if overlay_metrics is not None:
                    val_ov = overlay_metrics[m][key]
                    if np.isfinite(val_ov):
                        ax.scatter(val_ov, y_c + jitter,
                                   facecolors="none", edgecolors=color,
                                   s=42, zorder=4, marker=marker,
                                   linewidths=1.0, alpha=0.85)

        # Draw VLM↔Backbone connecting lines
        if connect_pairs:
            for vlm_m, bb_m in VLM_BACKBONE_PAIRS:
                if vlm_m in dot_pos and bb_m in dot_pos:
                    x0, y0 = dot_pos[vlm_m]
                    x1, y1 = dot_pos[bb_m]
                    color = _FAMILY_COLOR[_fam(vlm_m)]
                    ax.plot([x0, x1], [y0, y1], color=color,
                            lw=0.9, alpha=0.55, zorder=3, linestyle="-")

        # Human row (stars) at top
        y_human = float(len(GROUPS))
        if human_individual_r is not None and key == "pearson_r":
            for val in human_individual_r:
                if np.isfinite(val):
                    ax.scatter(val, y_human,
                               color="#888888", s=70, zorder=4,
                               marker="*", edgecolors="white",
                               linewidths=0.3, alpha=0.80)

        # Group separator lines
        sep_ys = [0.5, 1.5]
        if human_individual_r is not None and key == "pearson_r":
            sep_ys.append(y_human - 0.5)
        for sep_y in sep_ys:
            ax.axhline(sep_y, color="#cccccc", lw=0.7, linestyle="--", zorder=1)

        ax.set_xlabel(xlabel, fontsize=11.0)
        if xlim is not None:
            ax.set_xlim(*xlim)
            ax.autoscale(enable=False, axis="x")
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        else:
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
        y_top = (y_human + 0.55) if (human_individual_r is not None and key == "pearson_r") \
                else (len(GROUPS) - 1 + 0.55)
        ax.set_ylim(-0.55, y_top)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=11.0)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.18, linewidth=0.5, zorder=0)
        ax.set_yticks([])

        if ax_i == 0:
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_color("#cccccc")
            for grp in GROUPS:
                ax.text(-0.03, Y_CENTER[grp], GROUP_SHORT[grp],
                        ha="right", va="center", fontsize=10.5,
                        transform=ax.get_yaxis_transform(), clip_on=False)
            if human_individual_r is not None and key == "pearson_r":
                ax.text(-0.03, y_human, "Human",
                        ha="right", va="center", fontsize=10.5,
                        transform=ax.get_yaxis_transform(), clip_on=False)
        else:
            ax.spines["left"].set_visible(False)

    # ── Legend: 2-row layout with family markers ─────────────────────────────
    from matplotlib.lines import Line2D
    leg_items = [
        ("InternVL",     "InternVL"),
        ("LLaVA-1.5",   "LLaVA-1.5"),
        ("Mistral",      "Mistral"),
        ("Vicuna",       "Vicuna"),
        ("Qwen2.5",      "Qwen2.5"),
        ("Qwen",         "Qwen3"),
        ("Qwen (think)", "Qwen3 (think)"),
    ]
    leg_kw = dict(loc="upper center", fontsize=10.5, frameon=False,
                  handlelength=0.8, handletextpad=0.4, columnspacing=0.9)
    anchor_x = 0.50
    handles = [
        Line2D([0], [0], marker=_FAMILY_MARKER[fam], color=_FAMILY_COLOR[fam],
               linestyle="none", markersize=6, markeredgecolor="white",
               markeredgewidth=0.4, label=lbl)
        for fam, lbl in leg_items
    ]
    row1, row2 = handles[:4], handles[4:]
    leg1 = fig.legend(handles=row1, bbox_to_anchor=(anchor_x, 1.07), ncol=4, **leg_kw)
    leg2 = fig.legend(handles=row2, bbox_to_anchor=(anchor_x, 1.01), ncol=3, **leg_kw)
    fig.add_artist(leg1)

    return fig


MODEL_DISPLAY = {
    "Qwen3-VL-8B": "Qwen3-VL", "InternVL-8B": "InternVL",
    "LLaVA-1.5-7B": "LLaVA-1.5", "LLaVA-Mistral": "LLaVA-Mistral",
    "LLaVA-Vicuna": "LLaVA-Vicuna",
    "Qwen3-VL-8B (LM)": "Qwen3-VL (LM)", "InternVL-8B (LM)": "InternVL (LM)",
    "LLaVA-1.5 (LM)": "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)": "Mistral (LM)",
    "LLaVA-Vicuna (LM)": "Vicuna (LM)",
    "Qwen2.5-7B-Instruct": "Qwen2.5", "Qwen3-8B": "Qwen3",
    "Qwen3-8B (think)": "Qwen3 (think)", "Mistral-7B": "Mistral",
    "Vicuna-7B": "Vicuna",
}


def make_figure_per_model(perq: dict, human_refs: dict) -> plt.Figure:
    """One box per model, distribution over per-question mean SBERT scores."""
    # y positions
    y_pos: dict[str, float] = {}
    y = 0.0
    grp_sep_y: list[float] = []
    grp_label_y: list[tuple[str, float]] = []
    for grp, ms in MODEL_ORDER_7B.items():
        y_start = y
        for m in ms:
            y_pos[m] = y
            y += 0.72
        grp_label_y.append((grp, (y_start + y - 0.72) / 2))
        grp_sep_y.append(y - 0.72 + 0.54)
        y += 0.45

    fig_h = max(3.5, y * 0.34 + 0.6)
    fig, ax = plt.subplots(figsize=(5.2, fig_h),
                           gridspec_kw={"left": 0.32, "right": 0.97,
                                        "top": 0.97, "bottom": 0.08})

    ci = (human_refs["r_p5"], human_refs["r_p95"])
    ax.axvspan(*ci, color="#aaaaaa", alpha=0.18, zorder=0)
    ax.axvline(human_refs["r_median"], color="#000000", lw=1.0,
               linestyle=":", alpha=0.85, zorder=3)

    box_h = 0.28
    for ms in MODEL_ORDER_7B.values():
        for m in ms:
            y_c   = y_pos[m]
            color = _FAMILY_COLOR[_fam(m)]
            vals  = perq.get(m, np.array([]))
            vals  = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                _draw_hbox(ax, vals, y_c, box_h=box_h,
                           color=color, edge_color=color, median_color="white")

    # Group separator lines
    grp_short = {"VLM": "VLM", "Backbone Decoder": "Backbone", "Standalone LLM": "SA-LLM"}
    for sep in grp_sep_y[:-1]:
        ax.axhline(sep, color="#cccccc", lw=0.7, linestyle="--", zorder=1)

    all_models = [m for ms in MODEL_ORDER_7B.values() for m in ms]
    ax.set_yticks([y_pos[m] for m in all_models])
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in all_models], fontsize=8.5)
    ax.tick_params(axis="y", length=0)

    for grp, y_mid in grp_label_y:
        ax.text(-0.38, y_mid, grp_short[grp],
                ha="right", va="center", fontsize=9.0, fontweight="bold",
                transform=ax.get_yaxis_transform(), clip_on=False)

    ax.invert_yaxis()
    ax.set_xlabel(LBL_PEARSONR + " (bootstrap)", fontsize=11.0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="both"))
    ax.tick_params(axis="x", labelsize=10.0)
    ax.grid(axis="x", alpha=0.18, linewidth=0.5, zorder=0)

    return fig


def main():
    print("Loading metrics…")
    metrics, errors, human_refs, *_ = load_metrics()

    # Version: free-text Pearson r only
    fig = make_figure(metrics, human_refs)
    out = OUT_DIR / "boxplot_r_7b.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    shutil.copy(out, LATEX_DIR / "boxplot_r_7b.png")
    print(f"Saved: {out}")
    plt.close(fig)

    # Version: all-question Pearson r — unfiltered
    print("Computing all-question Pearson r (unfiltered)…")
    allq_r, allq_v_r, allq_r_refs = load_allq_pearson_r(abstention_filtered=False)
    metrics_allq = {m: dict(v) for m, v in metrics.items()}
    for m, rd in allq_r.items():
        metrics_allq[m]["pearson_r"] = rd["pearson_r"]
    human_refs_allq = dict(human_refs)
    human_refs_allq.update(allq_r_refs)

    fig2 = make_figure(metrics_allq, human_refs_allq, metric_keys=["pearson_r"],
                       human_individual_r=allq_r_refs.get("r_individual"))
    out2 = OUT_DIR / "boxplot_allq_r_7b.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    shutil.copy(out2, LATEX_DIR / "boxplot_allq_r_7b.png")
    print(f"Saved: {out2}")
    plt.close(fig2)

    # Version: all-question Pearson r — abstention-filtered
    print("Computing all-question Pearson r (abstention-filtered)…")
    allq_r_f, _, allq_r_refs_f = load_allq_pearson_r(abstention_filtered=True)
    metrics_allq_f = {m: dict(v) for m, v in metrics.items()}
    for m, rd in allq_r_f.items():
        metrics_allq_f[m]["pearson_r"] = rd["pearson_r"]
    human_refs_allq_f = dict(human_refs)
    human_refs_allq_f.update(allq_r_refs_f)

    fig2f = make_figure(metrics_allq_f, human_refs_allq_f, metric_keys=["pearson_r"],
                        human_individual_r=allq_r_refs_f.get("r_individual"))
    out2f = OUT_DIR / "boxplot_allq_r_filtered_7b.png"
    fig2f.savefig(out2f, dpi=180, bbox_inches="tight")
    shutil.copy(out2f, LATEX_DIR / "boxplot_allq_r_filtered_7b.png")
    print(f"Saved: {out2f}")
    plt.close(fig2f)

    # Version: all-q (solid) + ft-only (hollow) overlay, group-level
    fig3 = make_figure(metrics_allq, human_refs_allq, metric_keys=["pearson_r"],
                       overlay_metrics=metrics, overlay_human_refs=human_refs)
    out3 = OUT_DIR / "boxplot_allq_ft_overlay_r_7b.png"
    fig3.savefig(out3, dpi=180, bbox_inches="tight")
    shutil.copy(out3, LATEX_DIR / "boxplot_allq_ft_overlay_r_7b.png")
    print(f"Saved: {out3}")
    plt.close(fig3)

    # Version: per-model bootstrap Pearson r — free-text only
    print("Computing bootstrap Pearson r (ft-only)…")
    boot_r, boot_human_refs = load_bootstrap_pearson_r(ft_only=True)
    fig4 = make_figure_per_model(boot_r, boot_human_refs)
    out4 = OUT_DIR / "boxplot_per_model_r_7b.png"
    fig4.savefig(out4, dpi=180, bbox_inches="tight")
    shutil.copy(out4, LATEX_DIR / "boxplot_per_model_r_7b.png")
    print(f"Saved: {out4}")
    plt.close(fig4)

    # Version: per-model bootstrap Pearson r — all questions
    print("Computing bootstrap Pearson r (all-q)…")
    boot_r_allq, boot_human_refs_allq = load_bootstrap_pearson_r(ft_only=False)
    fig5 = make_figure_per_model(boot_r_allq, boot_human_refs_allq)
    out5 = OUT_DIR / "boxplot_per_model_allq_r_7b.png"
    fig5.savefig(out5, dpi=180, bbox_inches="tight")
    shutil.copy(out5, LATEX_DIR / "boxplot_per_model_allq_r_7b.png")
    print(f"Saved: {out5}")
    plt.close(fig5)


if __name__ == "__main__":
    main()
