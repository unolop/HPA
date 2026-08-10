"""
Barplot alignment profile — 4 metric panels side by side.
Metrics: Yes/No JS ↓, Count JS ↓, Free-text SBERT ↑, Abstention rate ↓
Y-axis: models grouped as VLM / Backbone / SA-LLM (same as strip plot)
Color:  family hue (same as strip plot); hatch encodes group type
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import figures.plot_style  # noqa: F401
from analysis.utils.vqa import VQAAnswerMapper
from scipy.stats import pearsonr as _pearsonr
from scipy.spatial.distance import jensenshannon
from notebooks.helpers import (
    load_human_responses,
    _clean_answer_series,
    _mark_abstentions,
    _answer_type_subset,
    _distribution_stats_from_counts,
    _prepare_distribution_category,
)
sys.path.insert(0, str(ROOT / "scripts/latex_table_generators"))
from gen_focus_compact_tables import (
    _distribution_base_table_variants_with_models,
    _compute_js_mode_per_model,
)

EXPORTS   = ROOT / "analysis/session2/exports"
OUT_DIR   = ROOT / "figures/alignment_profile"
LATEX_DIR = ROOT / "latex/AAAI2026/LaTeX/figures/alignment_profile"
OUT_DIR.mkdir(exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

# ── 7B model order (consistent with strip plot) ───────────────────────────────
MODEL_ORDER_7B = {
    "VLM":              ["InternVL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral",
                         "LLaVA-Vicuna", "Qwen3-VL-8B"],
    "Backbone Decoder": ["InternVL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)",
                         "LLaVA-Vicuna (LM)", "Qwen3-VL-8B (LM)"],
    "Standalone LLM":   ["Qwen2.5-7B-Instruct", "Qwen3-8B",
                         "Qwen3-8B (think)", "Vicuna-7B"],
}
MODEL_DISPLAY = {
    "Qwen3-VL-8B": "Qwen3-VL", "InternVL-8B": "InternVL",
    "LLaVA-1.5-7B": "LLaVA-1.5", "LLaVA-Mistral": "Mistral",
    "LLaVA-Vicuna": "Vicuna",
    "Qwen3-VL-8B (LM)": "Qwen3-VL", "InternVL-8B (LM)": "InternVL",
    "LLaVA-1.5 (LM)": "LLaVA-1.5", "LLaVA-Mistral (LM)": "Mistral",
    "LLaVA-Vicuna (LM)": "Vicuna",
    "Qwen2.5-7B-Instruct": "Qwen2.5-Instruct", "Qwen3-8B": "Qwen3",
    "Qwen3-8B (think)": "Qwen3 (think)", "Mistral-7B": "Mistral",
    "Vicuna-7B": "Vicuna",
}

# ── family colors & group visual encoding ─────────────────────────────────────
_FAMILY_COLOR = {
    "InternVL":      "#0072B2",  # blue
    "LLaVA-1.5":     "#3DBFAB",  # mint
    "Mistral":       "#E69F00",  # orange
    "Vicuna":        "#D55E00",  # vermillion
    "Qwen2.5":       "#9B59B6",  # purple
    "Qwen":          "#F48FB1",  # pastel pink
    "Qwen (think)":  "#C0152A",  # red
}
_FAMILY_MARKER = {
    "InternVL":     "o",
    "LLaVA-1.5":   "D",
    "Mistral":      "^",
    "Vicuna":       "v",
    "Qwen2.5":      "s",
    "Qwen":         "X",
    "Qwen (think)": "P",
}
MODEL_HATCH: dict[str, str] = {}
# ── Metric axis labels (keep in sync with LaTeX macros in macros.tex) ────────
LBL_YN_JS    = r"Yes/No $D_\mathrm{JS}$"
LBL_COUNT_JS = r"Count $D_\mathrm{JS}$"
LBL_PEARSONR = r"Pearson $r$"
LBL_SHM      = r"$S_\mathrm{HM}$"

GROUP_HATCH     = {"VLM": "",    "Backbone Decoder": "",   "Standalone LLM": ""}
GROUP_ALPHA     = {"VLM": 0.58, "Backbone Decoder": 0.58, "Standalone LLM": 0.58}
GROUP_HATCH_LW  = {"VLM": 1.0,  "Backbone Decoder": 1.0,  "Standalone LLM": 1.0}

def _fam(m: str) -> str:
    if m.startswith("InternVL"):   return "InternVL"
    if m.startswith("Qwen2.5"):    return "Qwen2.5"
    if "(think)" in m:             return "Qwen (think)"
    if "Qwen" in m:                return "Qwen"
    if m.startswith("LLaVA-1.5"): return "LLaVA-1.5"
    if "Mistral" in m:             return "Mistral"
    if "Vicuna" in m:              return "Vicuna"
    return "LLaVA"

# ── JS values from paper table (pooled C+B+A, abstention-filtered) ────────────
# Source: tables/hm_grouped_distribution_7b_compact_filtered.tex
JS_DATA = {
    # model: (yesno_js, count_js)
    "Qwen3-VL-8B":        (0.277, 0.788),
    "InternVL-8B":        (0.230, 0.586),
    "LLaVA-1.5-7B":       (0.421, 0.824),
    "LLaVA-Mistral":      (0.357, 0.787),
    "LLaVA-Vicuna":       (0.385, 0.837),
    "Qwen3-VL-8B (LM)":   (0.230, 0.745),
    "InternVL-8B (LM)":   (0.209, 0.599),
    "LLaVA-1.5 (LM)":     (0.241, 0.663),
    "LLaVA-Mistral (LM)": (0.275, 0.671),
    "LLaVA-Vicuna (LM)":  (0.223, 0.693),
    "Qwen2.5-7B-Instruct": (0.440, 0.688),
    "Qwen3-8B":            (0.531, 0.721),
    "Qwen3-8B (think)":    (0.397, 0.610),
    "Mistral-7B":          (0.848, 0.901),
    "Vicuna-7B":           (0.225, 0.788),
}

# ── JS human LOOCV CI (per-question basis, matching figure's model JS method) ─
# Computed via LOOCV over all participants, per-question JS averaged then p5/p95
HUMAN_YESNO_CI    = (0.219, 0.375)   # [p5, p95]
HUMAN_YESNO_MED   = 0.256
HUMAN_COUNT_CI    = (0.395, 0.693)
HUMAN_COUNT_MED   = 0.506

def _js(model_ans, human_ans, classify_fn, cats):
    m = np.array([sum(1 for a in model_ans if classify_fn(a) == c) for c in cats], dtype=float)
    h = np.array([sum(1 for a in human_ans if classify_fn(a) == c) for c in cats], dtype=float)
    if m.sum() == 0 or h.sum() == 0:
        return np.nan
    return float(jensenshannon(m / m.sum(), h / h.sum()) ** 2)

def _classify_yn(a):
    a = str(a).lower().strip()
    return "yes" if a == "yes" else ("no" if a == "no" else "other")

def _classify_num(a):
    try:
        return str(min(int(round(float(str(a).strip()))), 5))
    except Exception:
        return "other"


def _collect_per_question_js(human_all: pd.DataFrame, model_all: pd.DataFrame) -> dict:
    """Return {model: {key: [js per (qid,variant)]}} — raw per-question JS values."""
    h_all = _mark_abstentions(human_all.copy())
    m_all = _mark_abstentions(model_all.copy())
    result = {}
    for model_name, model_sub_all in m_all.groupby("model"):
        result[model_name] = {"yesno_js": [], "count_js": []}
        for answer_type, key in [("yes/no", "yesno_js"), ("number", "count_js")]:
            human_sub, model_sub, category_order = _prepare_distribution_category(
                h_all.copy(), model_sub_all.copy(), answer_type)
            model_sub = model_sub[~model_sub["is_abstained"]].copy()
            kept_qvs = set(zip(model_sub["question_id"].astype(int),
                               model_sub["variant"].astype(str)))
            human_sub = human_sub[
                [qv in kept_qvs for qv in
                 zip(human_sub["question_id"].astype(int),
                     human_sub["variant"].astype(str))]].copy()
            human_sub = human_sub[~human_sub["is_abstained"]].copy()
            for (qid, variant), hq in human_sub.groupby(["question_id", "variant"]):
                mq = model_sub[
                    (model_sub["question_id"].astype(int) == int(qid)) &
                    (model_sub["variant"].astype(str) == str(variant))]
                if hq.empty or mq.empty:
                    continue
                hc = hq["category"].value_counts().reindex(category_order, fill_value=0)
                mc = mq["category"].value_counts().reindex(category_order, fill_value=0)
                if hc.sum() == 0 or mc.sum() == 0:
                    continue
                _, _, js, _, _ = _distribution_stats_from_counts(hc, mc)
                result[model_name][key].append(float(js))
    return result


def load_metrics() -> tuple[dict, dict, dict]:
    """Returns (metrics, errors, human_refs).
    metrics[model][key] = mean value
    errors[model][key]  = SD across variants C/B/A (only for sbert/r; JS from table)
    JS values come from JS_DATA (operation-grouped, matching paper table).
    SBERT and Pearson r are computed from pair_cache per variant.
    Human JS CIs come from paper table caption; human SBERT/r CIs computed via LOO.
    """
    pc_cleaned = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    hm_cln = pc_cleaned[pc_cleaned["pair_type"] == "HM"]
    hh_cln = pc_cleaned[pc_cleaned["pair_type"] == "HH"]

    mapper = VQAAnswerMapper(); mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other")
                 for qid, ann in mapper.annotations.items()}
    ft_qids = {qid for qid, at in qid2atype.items() if at == "other"}
    yn_qids = {qid for qid, at in qid2atype.items() if at == "yes/no"}
    num_qids= {qid for qid, at in qid2atype.items() if at == "number"}

    hm_ft  = hm_cln[hm_cln["question_id"].astype(int).isin(ft_qids)]
    hh_ft  = hh_cln[hh_cln["question_id"].astype(int).isin(ft_qids)]
    hh_yn  = hh_cln[hh_cln["question_id"].astype(int).isin(yn_qids)]
    hh_num = hh_cln[hh_cln["question_id"].astype(int).isin(num_qids)]

    EXCLUDE: set[str] = set()
    VARIANTS = ["C", "B", "A"]

    # ── Load responses for per-question JS (exact paper table method) ────────
    human_all_full, model_inst_full = _distribution_base_table_variants_with_models(
        "inst_blind", ("C", "B", "A"))
    human_all_full = _mark_abstentions(human_all_full)

    # ── Human reference ranges ────────────────────────────────────────────────
    # JS: from paper table caption (operation-grouped method)
    human_refs = {
        "yn_p5":  HUMAN_YESNO_CI[0], "yn_p95":  HUMAN_YESNO_CI[1],
        "num_p5": HUMAN_COUNT_CI[0],  "num_p95": HUMAN_COUNT_CI[1],
        "yn_median":  HUMAN_YESNO_MED,
        "num_median": HUMAN_COUNT_MED,
    }

    # SBERT LOO p5/p95/median
    h_sbert = hh_ft.groupby("subject_1")["sbert_score"].mean().values
    human_refs.update({
        "ft_p5":     float(np.percentile(h_sbert, 5)),
        "ft_p95":    float(np.percentile(h_sbert, 95)),
        "ft_median": float(np.median(h_sbert)),
    })

    # r LOO p5/p95/median
    h_r_vals = []
    for p, grp in hh_ft.groupby("subject_1"):
        pm  = grp.groupby(["question_id", "variant"])["sbert_score"].mean()
        loo = hh_ft[hh_ft["subject_1"] != p].groupby(
            ["question_id", "variant"])["sbert_score"].mean()
        common = pm.index.intersection(loo.index)
        if len(common) < 20: continue
        r, _ = _pearsonr(pm[common], loo[common])
        h_r_vals.append(r)
    human_refs.update({
        "r_p5":    float(np.percentile(h_r_vals, 5)),
        "r_p95":   float(np.percentile(h_r_vals, 95)),
        "r_median":float(np.median(h_r_vals)),
    })

    # ── Per-variant SBERT, r, and JS per model ───────────────────────────────
    all_models = [m for ms in MODEL_ORDER_7B.values() for m in ms]
    v_data: dict[str, dict[str, list]] = {
        m: {"ft_sbert": [], "pearson_r": [], "yesno_js": [], "count_js": []}
        for m in all_models
    }

    for v in VARIANTS:
        hm_ft_v = hm_ft[hm_ft["variant"] == v]

        # Per-variant JS using same per-question method as paper table
        h_v   = human_all_full[human_all_full["variant"] == v].copy()
        m_v   = _mark_abstentions(model_inst_full[model_inst_full["variant"] == v].copy())
        js_v  = _compute_js_mode_per_model(h_v, m_v)

        for m in all_models:
            # SBERT
            s = hm_ft_v[hm_ft_v["subject_2"] == m]["sbert_score"].mean()
            v_data[m]["ft_sbert"].append(float(s) if np.isfinite(s) else np.nan)

            # JS
            if m in js_v:
                v_data[m]["yesno_js"].append(js_v[m].get(("yes/no", "JS"), np.nan))
                v_data[m]["count_js"].append(js_v[m].get(("number",  "JS"), np.nan))

    # ── Pooled Pearson r over all (question, variant) pairs ──────────────────
    # More stable than averaging 3 per-variant r values; human LOO ref is also pooled
    ref_pooled = hh_ft.groupby(["question_id", "variant"])["sbert_score"].mean()
    for m in all_models:
        mg = hm_ft[hm_ft["subject_2"] == m]
        mm = mg.groupby(["question_id", "variant"])["sbert_score"].mean()
        common = mm.index.intersection(ref_pooled.index)
        if len(common) >= 20:
            r, _ = _pearsonr(mm[common].values, ref_pooled[common].values)
            v_data[m]["pearson_r"] = [float(r)]
        else:
            v_data[m]["pearson_r"] = [float("nan")]

    # ── Aggregate mean + SD across variants ───────────────────────────────────
    metrics: dict[str, dict] = {}
    errors:  dict[str, dict] = {}
    for m in all_models:
        metrics[m] = {}
        errors[m]  = {}
        for key in ["ft_sbert", "pearson_r", "yesno_js", "count_js"]:
            vals = [v for v in v_data[m][key] if np.isfinite(v)]
            metrics[m][key] = float(np.mean(vals)) if vals else np.nan
            errors[m][key]  = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0

    # Mistral-7B is excluded from MODEL_ORDER_7B entirely (>72% abstention
    # across all question types after verbose-refusal filtering)

    # Print for sanity-check against paper table
    print("Model JS sanity check (yesno_js | count_js):")
    for m in all_models:
        yn = metrics[m]['yesno_js']
        cnt = metrics[m]['count_js']
        print(f"  {m:30s}  yn={'---' if np.isnan(yn) else f'{yn:.3f}'}  cnt={'---' if np.isnan(cnt) else f'{cnt:.3f}'}")

    # ── Per-question distributions for CI-based boxplot & error bars ─────────
    q_js = _collect_per_question_js(human_all_full, model_inst_full)
    q_data: dict[str, dict] = {}
    for m in all_models:
        q_data[m] = {
            "yesno_js": q_js.get(m, {}).get("yesno_js", []),
            "count_js":  q_js.get(m, {}).get("count_js",  []),
        }
        sub_ft = hm_ft[hm_ft["subject_2"] == m]
        q_data[m]["ft_sbert"] = sub_ft.groupby("question_id")["sbert_score"].mean().dropna().tolist()
        # Pearson r has no per-question analog; reuse per-variant r values from v_data
        q_data[m]["pearson_r"] = [v for v in v_data[m]["pearson_r"] if np.isfinite(v)]

    # Recompute errors as 95% CI half-width from per-question data
    # (much more data points → tight, meaningful intervals)
    def _ci95(vals):
        arr = np.array([v for v in vals if np.isfinite(v)])
        if len(arr) < 2:
            return 0.0
        return float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))

    for m in all_models:
        for key in ["yesno_js", "count_js", "ft_sbert"]:
            errors[m][key] = _ci95(q_data[m][key])
        # pearson_r: keep variant-based std (r is a global stat, not per-question)
        # errors[m]["pearson_r"] already set above from v_data

    return metrics, errors, human_refs, v_data, q_data


def make_figure(metrics: dict, human_refs: dict,
                metric_keys: list | None = None,
                errors: dict | None = None) -> plt.Figure:
    all_models_ordered = [m for ms in MODEL_ORDER_7B.values() for m in ms]

    # y positions with gap between groups
    y_pos: dict[str, float] = {}
    y = 0.0
    for grp, ms in MODEL_ORDER_7B.items():
        for m in ms:
            y_pos[m] = y
            y += 0.72
        y += 0.35  # inter-group gap

    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 2.9),
        gridspec_kw={"wspace": 0.06, "left": 0.15, "right": 0.97,
                     "top": 0.74, "bottom": 0.11},
    )

    if metric_keys is None:
        metric_keys = [
            ("yesno_js", "Yes/No JS", HUMAN_YESNO_CI, None),
            ("count_js",  "Count JS",  HUMAN_COUNT_CI, None),
            ("ft_sbert",  "SBERT",     (human_refs["ft_p5"], human_refs["ft_p95"]), None),
        ]
    # Normalise to 4-tuples (key, label, ci, xlim)
    METRIC_CFG = [t if len(t) == 4 else (*t, None) for t in metric_keys]

    # Mapping key → human_refs median key
    _MEDIAN_KEY = {
        "yesno_js": "yn_median", "count_js": "num_median",
        "pearson_r": "r_median", "ft_sbert": "ft_median",
    }

    bar_h = 0.52

    for ax_i, (key, xlabel, ci, xlim) in enumerate(METRIC_CFG):
        ax = axes[ax_i]

        # Human CI shading
        if ci is not None:
            lo, hi = ci
            ax.axvspan(lo, hi, color="#aaaaaa", alpha=0.18, zorder=0)
            # Human median line (dark blue)
            mkey = _MEDIAN_KEY.get(key)
            if mkey and mkey in human_refs:
                ax.axvline(human_refs[mkey], color="#000000", lw=1.0,
                           linestyle=":", alpha=0.85, zorder=3)

        # Find model closest to human median for this metric
        mkey = _MEDIAN_KEY.get(key)
        median_val = human_refs.get(mkey) if mkey else None
        all_models_flat = [m for ms in MODEL_ORDER_7B.values() for m in ms]
        if median_val is not None:
            closest_m = min(
                (m for m in all_models_flat if np.isfinite(metrics[m][key])),
                key=lambda m: abs(metrics[m][key] - median_val)
            )
        else:
            closest_m = None

        matplotlib.rcParams["hatch.linewidth"] = 1.5
        for grp, ms in MODEL_ORDER_7B.items():
            for m in ms:
                val = metrics[m][key]
                if not np.isfinite(val):
                    continue
                y = y_pos[m]
                color = _FAMILY_COLOR[_fam(m)]
                err = errors[m][key] if errors else 0.0
                is_closest = (m == closest_m)
                ax.barh(y, val, height=bar_h,
                        color=color, alpha=GROUP_ALPHA[grp],
                        hatch=MODEL_HATCH.get(m, ""),
                        edgecolor="white", linewidth=0.2,
                        zorder=2)
                if ci is not None and np.isfinite(val):
                    lo, hi = ci
                    if lo <= val <= hi:
                        ax.barh(y, val, height=bar_h,
                                color="none", edgecolor="#444444",
                                linewidth=0.5, zorder=5)
                if is_closest:
                    ax.barh(y, val, height=bar_h,
                            color="none", edgecolor="#E8294A",
                            linewidth=1.4, zorder=6)
                # Error bar — match bar color
                if errors and np.isfinite(err) and err > 0:
                    ax.errorbar(val, y, xerr=err, fmt="none",
                                ecolor=color, elinewidth=0.7,
                                capsize=1.2, capthick=0.7, zorder=4)

        # Group separator lines
        for grp_i, (grp, ms) in enumerate(MODEL_ORDER_7B.items()):
            if grp_i < len(MODEL_ORDER_7B) - 1:
                sep_y = y_pos[ms[-1]] + 0.54
                ax.axhline(sep_y, color="#cccccc", lw=0.7, linestyle="--", zorder=1)

        ax.set_xlabel(xlabel, fontsize=11.0)
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(left=0)
        ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
        ax.tick_params(axis="x", labelsize=11.0)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.18, linewidth=0.5, zorder=0)

        ax.set_yticks([])

        if ax_i == 0:
            # Group labels as y-axis text at center of each group block
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_color("#cccccc")
            grp_short = {"VLM": "VLM", "Backbone Decoder": "Backbone",
                         "Standalone LLM": "SA-LLM"}
            for grp, ms in MODEL_ORDER_7B.items():
                y_center = np.mean([y_pos[m] for m in ms])
                ax.text(-0.03, y_center, grp_short[grp],
                        ha="right", va="center", fontsize=10.5, fontweight="normal",
                        transform=ax.get_yaxis_transform(), clip_on=False)
        else:
            ax.spines["left"].set_visible(False)

    # ── Legend: 2 rows — row1: non-Qwen, row2: all Qwen ─────────────────────
    row1_items = [
        ("InternVL",      "InternVL",  ""),
        ("LLaVA-1.5",     "LLaVA-1.5",""),
        ("LLaVA-Mistral", "Mistral",   ""),
        ("LLaVA-Vicuna",  "Vicuna",   ""),
    ]
    row1 = [mpatches.Patch(facecolor=_FAMILY_COLOR[fam], hatch=htch,
                            edgecolor="white", linewidth=0.2,
                            alpha=GROUP_ALPHA["VLM"], label=lbl)
            for lbl, fam, htch in row1_items]
    row2_items = [
        ("Qwen2.5-Instruct","Qwen2.5",""),
        ("Qwen3",           "Qwen",   ""),
        ("Qwen3 (think)",   "Qwen (think)", ""),
    ]
    row2 = [mpatches.Patch(facecolor=_FAMILY_COLOR[fam], hatch=htch,
                            edgecolor="white", linewidth=0.2,
                            alpha=GROUP_ALPHA["Standalone LLM"], label=lbl)
            for lbl, fam, htch in row2_items]
    leg_kw = dict(loc="upper center", fontsize=10.5, frameon=False,
                  handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
    anchor_x = 0.50
    leg1 = fig.legend(handles=row1, bbox_to_anchor=(anchor_x, 0.94), ncol=4, **leg_kw)
    leg2 = fig.legend(handles=row2, bbox_to_anchor=(anchor_x, 0.88), ncol=3, **leg_kw)
    fig.add_artist(leg1)

    return fig


def make_figure_vertical(metrics: dict, human_refs: dict,
                          metric_keys: list | None = None,
                          errors: dict | None = None) -> plt.Figure:
    """Vertical bar version: models on x-axis, metric panels stacked vertically."""
    all_models_ordered = [m for ms in MODEL_ORDER_7B.values() for m in ms]

    x_pos: dict[str, float] = {}
    x = 0.0
    grp_mids: list[tuple[float, str]] = []
    for grp, ms in MODEL_ORDER_7B.items():
        x_start = x
        for m in ms:
            x_pos[m] = x
            x += 0.9
        grp_mids.append(((x_start + x - 0.9) / 2, grp))
        x += 0.55

    if metric_keys is None:
        metric_keys = [
            ("yesno_js", "Yes/No JS", (human_refs["yn_p5"],  human_refs["yn_p95"]), None),
            ("count_js",  "Count JS",  (human_refs["num_p5"], human_refs["num_p95"]), (0.4, 1.0)),
            ("pearson_r", "Pearson r", (human_refs["r_p5"],   human_refs["r_p95"]),  (0.3, 1.0)),
        ]
    METRIC_CFG = [t if len(t) == 4 else (*t, None) for t in metric_keys]
    _MEDIAN_KEY = {"yesno_js": "yn_median", "count_js": "num_median",
                   "pearson_r": "r_median", "ft_sbert": "ft_median"}

    n = len(METRIC_CFG)
    fig, axes = plt.subplots(n, 1, figsize=(5.5, 2.1), sharex=True,
                              gridspec_kw={"hspace": 0.08, "left": 0.09, "right": 0.97,
                                           "top": 0.92, "bottom": 0.26})
    if n == 1: axes = [axes]

    bar_w = 0.60
    x_max = max(x_pos.values()) + 0.55

    for ax_i, (key, ylabel, ci, ylim) in enumerate(METRIC_CFG):
        ax = axes[ax_i]
        if ci is not None:
            lo, hi = ci
            ax.axhspan(lo, hi, color="#9ecae1", alpha=0.30, zorder=0)
            mkey = _MEDIAN_KEY.get(key)
            if mkey and mkey in human_refs:
                ax.axhline(human_refs[mkey], color="#1a3a6b", lw=1.0,
                           linestyle=":", alpha=0.85, zorder=3)

        matplotlib.rcParams["hatch.linewidth"] = 1.5
        for grp, ms in MODEL_ORDER_7B.items():
            for m in ms:
                val = metrics[m][key]
                if not np.isfinite(val): continue
                xv = x_pos[m]
                color = _FAMILY_COLOR[_fam(m)]
                err = errors[m][key] if errors else 0.0
                ax.bar(xv, val, width=bar_w, color=color, alpha=GROUP_ALPHA[grp],
                       hatch=MODEL_HATCH.get(m, ""), edgecolor="white", linewidth=0.2, zorder=2)
                if errors and np.isfinite(err) and err > 0:
                    ax.errorbar(xv, val, yerr=err, fmt="none", ecolor=color,
                                elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=4)

        for grp_i, (grp, ms) in enumerate(MODEL_ORDER_7B.items()):
            if grp_i < len(MODEL_ORDER_7B) - 1:
                sep_x = x_pos[ms[-1]] + 0.72
                ax.axvline(sep_x, color="#cccccc", lw=0.7, linestyle="--", zorder=1)

        ax.set_ylabel(ylabel, fontsize=7.5)
        if ylim is not None: ax.set_ylim(*ylim)
        else: ax.set_ylim(bottom=0)
        ax.set_xlim(-0.5, x_max)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=0.18, linewidth=0.5, zorder=0)

    # x-tick labels: model names on bottom panel
    xtick_vals = [x_pos[m] for m in all_models_ordered]
    axes[-1].set_xticks(xtick_vals)
    axes[-1].set_xticklabels(
        [MODEL_DISPLAY.get(m, m) for m in all_models_ordered],
        fontsize=6.5, rotation=40, ha="right")

    # Group labels above top panel
    ax_top = axes[0]
    import matplotlib.transforms as _mt
    grp_short = {"VLM": "VLM", "Backbone Decoder": "Backbone", "Standalone LLM": "SA-LLM"}
    for mid_x, grp in grp_mids:
        trans = _mt.blended_transform_factory(ax_top.transData, ax_top.transAxes)
        ax_top.text(mid_x, 1.04, grp_short[grp], ha="center", va="bottom",
                    fontsize=7.0, fontweight="bold", transform=trans, clip_on=False)

    return fig


def make_boxplot_figure(q_data: dict, human_refs: dict,
                        metric_keys: list | None = None) -> plt.Figure:
    """Horizontal boxplot: each model gets a box showing CI across per-question JS/SBERT values."""
    y_pos: dict[str, float] = {}
    y = 0.0
    for grp, ms in MODEL_ORDER_7B.items():
        for m in ms:
            y_pos[m] = y
            y += 0.72
        y += 0.35

    _MEDIAN_KEY = {
        "yesno_js": "yn_median", "count_js": "num_median",
        "pearson_r": "r_median",  "ft_sbert": "ft_median",
    }

    if metric_keys is None:
        metric_keys = [
            ("yesno_js",  LBL_YN_JS,    (human_refs["yn_p5"],  human_refs["yn_p95"]),  None),
            ("count_js",  LBL_COUNT_JS, (human_refs["num_p5"], human_refs["num_p95"]), None),
            ("pearson_r", LBL_PEARSONR, (human_refs["r_p5"],   human_refs["r_p95"]),   None),
        ]
    METRIC_CFG = [t if len(t) == 4 else (*t, None) for t in metric_keys]

    box_h = 0.44
    fig, axes = plt.subplots(
        1, len(METRIC_CFG), figsize=(6.5, 2.9),
        gridspec_kw={"wspace": 0.06, "left": 0.15, "right": 0.97,
                     "top": 0.74, "bottom": 0.11},
    )
    if len(METRIC_CFG) == 1:
        axes = [axes]

    for ax_i, (key, xlabel, ci, xlim) in enumerate(METRIC_CFG):
        ax = axes[ax_i]

        if ci is not None:
            lo, hi = ci
            ax.axvspan(lo, hi, color="#aaaaaa", alpha=0.18, zorder=0)
            mkey = _MEDIAN_KEY.get(key)
            if mkey and mkey in human_refs:
                ax.axvline(human_refs[mkey], color="#000000", lw=1.0,
                           linestyle=":", alpha=0.85, zorder=3)

        for grp, ms in MODEL_ORDER_7B.items():
            for m in ms:
                vals = [v for v in q_data.get(m, {}).get(key, []) if np.isfinite(v)]
                if len(vals) == 0:
                    continue
                y = y_pos[m]
                color = _FAMILY_COLOR[_fam(m)]
                alpha = GROUP_ALPHA[grp]

                arr = np.array(vals)
                stats = [{
                    "med":    float(np.median(arr)),
                    "q1":     float(np.percentile(arr, 25)),
                    "q3":     float(np.percentile(arr, 75)),
                    "whislo": float(arr.min()),
                    "whishi": float(arr.max()),
                    "fliers": [],
                }]
                bp = ax.bxp(stats, positions=[y], vert=False, widths=box_h,
                            patch_artist=True, manage_ticks=False,
                            boxprops=dict(facecolor=color, alpha=alpha,
                                          edgecolor="white", linewidth=0.3,
                                          hatch=MODEL_HATCH.get(m, "")),
                            medianprops=dict(color="white", linewidth=1.2),
                            whiskerprops=dict(color=color, linewidth=0.8, alpha=alpha),
                            capprops=dict(color=color, linewidth=0.8, alpha=alpha),
                            flierprops=dict(marker=""),
                            zorder=2)

                # mean dot
                ax.scatter(float(arr.mean()), y, s=18, color=color,
                           edgecolors="#333333", linewidth=0.4, zorder=4)

        # group separator lines
        for grp_i, (grp, ms) in enumerate(MODEL_ORDER_7B.items()):
            if grp_i < len(MODEL_ORDER_7B) - 1:
                sep_y = y_pos[ms[-1]] + 0.54
                ax.axhline(sep_y, color="#cccccc", lw=0.7, linestyle="--", zorder=1)

        ax.set_xlabel(xlabel, fontsize=11.0)
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(left=0)
        ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
        ax.tick_params(axis="x", labelsize=11.0)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.18, linewidth=0.5, zorder=0)
        ax.set_yticks([])

        if ax_i == 0:
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_color("#cccccc")
            grp_short = {"VLM": "VLM", "Backbone Decoder": "Backbone",
                         "Standalone LLM": "SA-LLM"}
            for grp, ms in MODEL_ORDER_7B.items():
                y_center = np.mean([y_pos[m] for m in ms])
                ax.text(-0.03, y_center, grp_short[grp],
                        ha="right", va="center", fontsize=10.5,
                        transform=ax.get_yaxis_transform(), clip_on=False)
        else:
            ax.spines["left"].set_visible(False)

    # Legend (same as barplot)
    row1_items = [
        ("InternVL",      "InternVL",  ""),
        ("LLaVA-1.5",     "LLaVA-1.5",""),
        ("LLaVA-Mistral", "Mistral",   ""),
        ("LLaVA-Vicuna",  "Vicuna",   ""),
    ]
    row1 = [mpatches.Patch(facecolor=_FAMILY_COLOR[fam], hatch=htch,
                            edgecolor="white", linewidth=0.2,
                            alpha=GROUP_ALPHA["VLM"], label=lbl)
            for lbl, fam, htch in row1_items]
    row2_items = [
        ("Qwen2.5-Instruct","Qwen2.5",""),
        ("Qwen3",           "Qwen",   ""),
        ("Qwen3 (think)",   "Qwen (think)", ""),
    ]
    row2 = [mpatches.Patch(facecolor=_FAMILY_COLOR[fam], hatch=htch,
                            edgecolor="white", linewidth=0.2,
                            alpha=GROUP_ALPHA["Standalone LLM"], label=lbl)
            for lbl, fam, htch in row2_items]
    leg_kw = dict(loc="upper center", fontsize=10.5, frameon=False,
                  handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
    anchor_x = 0.50
    leg1 = fig.legend(handles=row1, bbox_to_anchor=(anchor_x, 0.94), ncol=4, **leg_kw)
    leg2 = fig.legend(handles=row2, bbox_to_anchor=(anchor_x, 0.88), ncol=3, **leg_kw)
    fig.add_artist(leg1)

    return fig


def main():
    print("Loading metrics…")
    metrics, errors, human_refs, v_data, q_data = load_metrics()

    yn_ci  = (human_refs["yn_p5"],  human_refs["yn_p95"])
    num_ci = (human_refs["num_p5"], human_refs["num_p95"])
    r_ci   = (human_refs["r_p5"],  human_refs["r_p95"])
    ft_ci  = (human_refs["ft_p5"], human_refs["ft_p95"])

    # Version 1: horizontal barplot, JS + Pearson r  (primary)
    fig1 = make_figure(metrics, human_refs, errors=errors, metric_keys=[
        ("yesno_js",  LBL_YN_JS,    yn_ci,  None),
        ("count_js",  LBL_COUNT_JS, num_ci, (0.2, 1.0)),
        ("pearson_r", LBL_PEARSONR, r_ci,   (0.3, 1.0)),
    ])
    out1 = OUT_DIR / "barplot_r_7b.png"
    fig1.savefig(out1, dpi=180, bbox_inches="tight")
    shutil.copy(out1, LATEX_DIR / "barplot_r_7b.png")
    print(f"Saved: {out1}")
    plt.close(fig1)

    # Version 2: horizontal barplot, JS + SBERT
    fig2 = make_figure(metrics, human_refs, errors=errors, metric_keys=[
        ("yesno_js",  LBL_YN_JS,    yn_ci,  None),
        ("count_js",  LBL_COUNT_JS, num_ci, (0.2, 1.0)),
        ("ft_sbert",  LBL_SHM,      ft_ci,  (0.3, 0.6)),
    ])
    out2 = OUT_DIR / "barplot_7b.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    shutil.copy(out2, LATEX_DIR / "barplot_7b.png")
    print(f"Saved: {out2}")
    plt.close(fig2)

    # Version 3: vertical barplot, Pearson r only
    fig3 = make_figure_vertical(metrics, human_refs, errors=errors, metric_keys=[
        ("pearson_r", LBL_PEARSONR, r_ci, (0.3, 1.0)),
    ])
    out3 = OUT_DIR / "barplot_vertical_r_7b.png"
    fig3.savefig(out3, dpi=180, bbox_inches="tight")
    shutil.copy(out3, LATEX_DIR / "barplot_vertical_r_7b.png")
    print(f"Saved: {out3}")
    plt.close(fig3)

    # Version 4: horizontal boxplot, JS + Pearson r (per-question CI)
    fig4 = make_boxplot_figure(q_data, human_refs, metric_keys=[
        ("yesno_js",  LBL_YN_JS,    yn_ci,  None),
        ("count_js",  LBL_COUNT_JS, num_ci, (0.2, 1.0)),
        ("pearson_r", LBL_PEARSONR, r_ci,   (0.3, 1.0)),
    ])
    out4 = OUT_DIR / "boxplot_7b.png"
    fig4.savefig(out4, dpi=180, bbox_inches="tight")
    shutil.copy(out4, LATEX_DIR / "boxplot_7b.png")
    print(f"Saved: {out4}")
    plt.close(fig4)



if __name__ == "__main__":
    main()
