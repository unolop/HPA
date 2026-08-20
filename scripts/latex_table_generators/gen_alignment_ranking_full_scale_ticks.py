r"""
Full-scale alignment ranking table — all model sizes, tick variant.

Same layout as gen_alignment_ranking_full_scale.py but with a different visual style:
  - No green/orange/gray cell colors for JS/sHM columns.
    Instead, a \checkmark is appended to the rank when the model's value is within
    the human LOO 95 % CI.
  - Abstention columns show actual percentage values (not ranks) with gray shading
    proportional to "goodness" (higher blind = darker; lower inst = darker).
  - Arrow labels in abstention headers: w/o ↑  and  w/ ↓

Requires \usepackage{amssymb} in the LaTeX preamble for \checkmark.

Output: latex/AAAI2026/LaTeX/tables/supp/alignment_ranking_full_scale_ticks.tex
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/latex_table_generators"))

from figures.helpers import filter_abstained_pairs
from notebooks.helpers import (
    _mark_abstentions,
    _prepare_distribution_category,
    _distribution_stats_from_counts,
)
from analysis.utils.vqa import VQAAnswerMapper
from analysis.utils.abstention import classify, is_abstained

EXPORTS = ROOT / "analysis/session2/exports"
LOGITS  = ROOT / "evaluation/logits"
OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/alignment_ranking_full_scale_ticks.tex"

# Maps our model label → (logit_subdir, pretrained_folder)
MODEL_LOGIT_MAP: dict[str, tuple[str, str]] = {
    "InternVL-1B":           ("vlm",        "InternVL3_5-1B"),
    "InternVL-2B":           ("vlm",        "InternVL3_5-2B"),
    "InternVL-8B":           ("vlm",        "InternVL3_5-8B"),
    "Qwen3-VL-2B":           ("vlm",        "Qwen3-VL-2B-Instruct"),
    "Qwen3-VL-4B":           ("vlm",        "Qwen3-VL-4B-Instruct"),
    "Qwen3-VL-8B":           ("vlm",        "Qwen3-VL-8B-Instruct"),
    "Qwen3-VL-32B":          ("vlm",        "Qwen3-VL-32B-Instruct"),
    "LLaVA-1.5-7B":          ("vlm",        "llava-1.5-7b-hf"),
    "LLaVA-Mistral":         ("vlm",        "llava-v1.6-mistral-7b-hf"),
    "LLaVA-Vicuna":          ("vlm",        "llava-v1.6-vicuna-7b-hf"),
    "LLaVA-Vicuna-13B":      ("vlm",        "llava-v1.6-vicuna-13b-hf"),
    "InternVL-1B (LM)":      ("lm_decoder", "InternVL3_5-1B"),
    "InternVL-2B (LM)":      ("lm_decoder", "InternVL3_5-2B"),
    "InternVL-8B (LM)":      ("lm_decoder", "InternVL3_5-8B"),
    "Qwen3-VL-2B (LM)":      ("lm_decoder", "Qwen3-VL-2B-Instruct"),
    "Qwen3-VL-4B (LM)":      ("lm_decoder", "Qwen3-VL-4B-Instruct"),
    "Qwen3-VL-8B (LM)":      ("lm_decoder", "Qwen3-VL-8B-Instruct"),
    "Qwen3-VL-32B (LM)":     ("lm_decoder", "Qwen3-VL-32B-Instruct"),
    "LLaVA-1.5 (LM)":        ("lm_decoder", "llava-1.5-7b-hf"),
    "LLaVA-Mistral (LM)":    ("lm_decoder", "llava-v1.6-mistral-7b-hf"),
    "LLaVA-Vicuna (LM)":     ("lm_decoder", "llava-v1.6-vicuna-7b-hf"),
    "LLaVA-Vicuna-13B (LM)": ("lm_decoder", "llava-v1.6-vicuna-13b-hf"),
    "Qwen3-0.6B":            ("backbone",   "Qwen3-0.6B"),
    "Qwen3-1.7B":            ("backbone",   "Qwen3-1.7B"),
    "Qwen3-4B":              ("backbone",   "Qwen3-4B"),
    "Qwen3-8B":              ("backbone",   "Qwen3-8B"),
    "Qwen3-32B":             ("backbone",   "Qwen3-32B"),
    "Qwen2.5-7B-Instruct":   ("backbone",   "Qwen2.5-7B-Instruct"),
    "Vicuna-7B":             ("backbone",   "vicuna-7b-v1.5"),
    "Vicuna-13B":            ("backbone",   "vicuna-13b-v1.5"),
    "Mistral-7B":            ("backbone",   "Mistral-7B-Instruct-v0.2"),
    "Qwen3-1.7B (think)":    ("backbone",   "Qwen3-1.7B_think"),
    "Qwen3-8B (think)":      ("backbone",   "Qwen3-8B_think"),
    "Qwen3-32B (think)":     ("backbone",   "Qwen3-32B_think"),
}

_CT_MAP = {"question": "C", "weaker_object": "B", "pronominalized": "A"}
_STRIP_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

HUMAN_YESNO_CI  = (0.219, 0.375)
HUMAN_YESNO_MED = 0.256
HUMAN_COUNT_CI  = (0.395, 0.693)
HUMAN_COUNT_MED = 0.506

STATIC_CI: dict[str, tuple[float, float] | None] = {
    "yesno_js":        (0.260, 0.282),
    "count_js":        (0.493, 0.554),
    "ft_sbert":        None,   # filled at runtime
    "spearman_r_ft":   None,   # filled at runtime
    "spearman_r_all":  None,   # filled at runtime
    "abs_blind":       None,
    "abs_inst":        None,
    "abs_blind_1k":    None,
    "abs_inst_1k":     None,
}
STATIC_MED: dict[str, float | None] = {
    "yesno_js":       0.271,
    "count_js":       0.524,
    "ft_sbert":       None,
    "spearman_r_ft":  None,
    "spearman_r_all": None,
    "abs_blind":      None,
    "abs_inst":       None,
    "abs_blind_1k":   None,
    "abs_inst_1k":    None,
}

VLM_FAMILIES: list[tuple[str, list[tuple[str, str, str | None]]]] = [
    ("InternVL3.5", [
        ("1B",  "InternVL-1B",      "InternVL-1B (LM)"),
        ("2B",  "InternVL-2B",      "InternVL-2B (LM)"),
        ("8B",  "InternVL-8B",      "InternVL-8B (LM)"),
    ]),
    ("Qwen3-VL", [
        ("2B",  "Qwen3-VL-2B",     "Qwen3-VL-2B (LM)"),
        ("4B",  "Qwen3-VL-4B",     "Qwen3-VL-4B (LM)"),
        ("8B",  "Qwen3-VL-8B",     "Qwen3-VL-8B (LM)"),
        ("32B", "Qwen3-VL-32B",    "Qwen3-VL-32B (LM)"),
    ]),
    ("LLaVA-1.5", [
        ("7B",  "LLaVA-1.5-7B",    "LLaVA-1.5 (LM)"),
    ]),
    ("LLaVA-Mistral", [
        ("7B",  "LLaVA-Mistral",   "LLaVA-Mistral (LM)"),
    ]),
    ("LLaVA-Vicuna", [
        ("7B",  "LLaVA-Vicuna",      "LLaVA-Vicuna (LM)"),
        ("13B", "LLaVA-Vicuna-13B",  "LLaVA-Vicuna-13B (LM)"),
    ]),
]

SA_MODELS: list[tuple[str, str, str]] = [
    ("Qwen3",   "0.6B",  "Qwen3-0.6B"),
    ("Qwen3",   "1.7B",  "Qwen3-1.7B"),
    ("Qwen3",   "4B",    "Qwen3-4B"),
    ("Qwen3",   "8B",    "Qwen3-8B"),
    ("Qwen3",   "32B",   "Qwen3-32B"),
    ("Qwen2.5", "7B",    "Qwen2.5-7B-Instruct"),
    ("Vicuna",  "7B",    "Vicuna-7B"),
    ("Vicuna",  "13B",   "Vicuna-13B"),
    ("Mistral", "7B",    "Mistral-7B"),
]

THINK_MODELS: list[tuple[str, str, str]] = [
    ("Qwen3", "1.7B", "Qwen3-1.7B (think)"),
    ("Qwen3", "8B",   "Qwen3-8B (think)"),
    ("Qwen3", "32B",  "Qwen3-32B (think)"),
]

N_YESNO_PAIRS = 78   # 26 yes/no questions × 3 variants
N_COUNT_PAIRS = 66   # 22 count questions × 3 variants
N_STUDY_PAIRS = 339  # 113 study questions × 3 variants
N_1K_PAIRS    = 3000 # 1000 questions × 3 variants

GROUP_LABELS = {
    "VLM":              "VLMs",
    "Backbone Decoder": "Backbone Decoders",
    "Standalone LLM":   "SA-LLMs",
    "Thinking":         "Thinking Models",
}

METRIC_CFG = [
    ("yesno_js",       r"Y/N",    True,  True,  ".3f"),
    ("count_js",       r"Cnt",    True,  True,  ".3f"),
    ("ft_sbert",       r"$\sHM$", True,  False, ".3f"),
    ("spearman_r_ft",  r"FT",     True,  False, ".3f"),
    ("spearman_r_all", r"All",    True,  False, ".3f"),
    ("abs_inst",       r"Study",  False, True,  ".1f"),
    ("abs_inst_1k",    r"1kQ",    False, True,  ".1f"),
]
# abs_blind / abs_blind_1k stay in metrics dict for delta, but not rendered as columns
VALUE_KEYS = {"yesno_js", "count_js"}  # ft_sbert/spearman handled via SBERT_DELTA_MAP
ABS_INST_KEYS = {"abs_inst", "abs_inst_1k"}
DJS_BLIND_MAP = {"yesno_js": "yesno_js_blind", "count_js": "count_js_blind"}
SBERT_DELTA_MAP = {
    "ft_sbert":       "ft_sbert_blind",
    "spearman_r_ft":  "spearman_r_ft_blind",
    "spearman_r_all": "spearman_r_all_blind",
}
DELTA_KEYS = set()


# ── Ranking helpers ───────────────────────────────────────────────────────────

def compute_ranks(all_models, metrics, rank_by_median, lower_is_better,
                  key, human_median):
    vals = [(m, metrics[m].get(key, np.nan)) for m in all_models]
    finite = [(m, v) for m, v in vals if np.isfinite(v)]
    ranks = {m: np.nan for m in all_models}
    if not finite:
        return ranks
    if rank_by_median and human_median is not None:
        sorted_m = sorted(finite, key=lambda x: abs(x[1] - human_median))
    else:
        sign = 1 if lower_is_better else -1
        sorted_m = sorted(finite, key=lambda x: sign * x[1])
    i = 0
    while i < len(sorted_m):
        j = i
        while j < len(sorted_m) - 1 and sorted_m[j+1][1] == sorted_m[i][1]:
            j += 1
        avg = ((i + 1) + (j + 1)) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_m[k][0]] = avg
        i = j + 1
    return ranks


def rank_gray(rank, n_models, max_int=20):
    if not np.isfinite(rank) or n_models <= 1:
        return 0
    raw = (1.0 - (rank - 1) / (n_models - 1)) * max_int
    return max(0, round(raw / 5) * 5)


def rank_str(r):
    if not np.isfinite(r):
        return "---"
    return str(int(r)) if r == int(r) else f"{r:.1f}"


# ── Cell formatters ───────────────────────────────────────────────────────────

_BG_BLUE   = r"\cellcolor[HTML]{D9EAF3}"   # within CI
_BG_ORANGE = r"\cellcolor[HTML]{F7EBD9}"   # over-aligned (JS below lo, or SBERT above hi)
_SYM_IN    = r""                           # within CI
_SYM_OVER  = r""                           # over-aligned (below human LOO range)


def fmt_metric_cell(val, rank, ci_bounds):
    """JS rank cell: blue+checkmark if within CI, orange+▽ if below CI, plain if above CI."""
    if not np.isfinite(val):
        return "---"
    rs = rank_str(rank)
    if np.isfinite(rank):
        if rank <= 1.5:
            rs = rf"\textbf{{{rs}}}"
        elif rank <= 2.5:
            rs = rf"\underline{{{rs}}}"
    if ci_bounds is not None:
        lo, hi = ci_bounds
        if val < lo:          # over-aligned: JS lower than human LOO range
            return _BG_ORANGE + rs + _SYM_OVER
        if lo <= val <= hi:   # within CI
            return _BG_BLUE + rs + _SYM_IN
    return rs


def fmt_value_cell(val, all_vals, higher_better, ci_bounds, fmt=".3f", rank=np.nan):
    """Value cell: blue+checkmark if within CI, orange+▽ if over-aligned, plain otherwise.
    For higher-better (SBERT/rho): over-aligned = val > hi. For lower-better: over-aligned = val < lo.
    Bold = best (rank ≤ 1.5); underline = second-best (rank ≤ 2.5)."""
    if not np.isfinite(val):
        return "---"
    raw = format(val, fmt)
    val_str = raw.lstrip("0") if raw.startswith("0.") and len(raw) > 3 else raw
    if np.isfinite(rank):
        if rank <= 1.5:
            val_str = rf"\textbf{{{val_str}}}"
        elif rank <= 2.5:
            val_str = rf"\underline{{{val_str}}}"
    if ci_bounds is not None:
        lo, hi = ci_bounds
        if higher_better and val > hi:
            return _BG_ORANGE + val_str + _SYM_OVER
        if not higher_better and val < lo:
            return _BG_ORANGE + val_str + _SYM_OVER
        if lo <= val <= hi:
            return _BG_BLUE + val_str + _SYM_IN
    return val_str


def fmt_delta_cell(val, fmt="+.1f"):
    """Delta abstention cell: colored brackets — blue for negative (good), orange for positive (bad)."""
    if not np.isfinite(val):
        return "---"
    s = format(val, fmt)
    if val < 0:
        return rf"\textcolor[HTML]{{2166AC}}{{({s})}}"
    elif val > 0:
        return rf"\textcolor[HTML]{{D6610A}}{{({s})}}"
    else:
        return f"({s})"


def fmt_abs_cell(val, all_vals, higher_better, fmt=".1f"):
    """Abstention cell: gray shading proportional to goodness; no CI coloring."""
    if not np.isfinite(val):
        return "---"
    finite = [v for v in all_vals if np.isfinite(v)]
    mn, mx = (min(finite), max(finite)) if finite else (0, 1)
    intens = 0
    if mx > mn:
        frac = (val - mn) / (mx - mn) if higher_better else (mx - val) / (mx - mn)
        intens = round(frac * 4) * 5  # 0, 5, 10, 15, 20
    val_str = format(val, fmt)
    if intens > 0:
        return rf"\cellcolor{{gray!{intens}}}{val_str}"
    return val_str


def fmt_abs_inst_with_delta(inst_val, blind_val, all_inst_vals, fmt=".1f"):
    """Abstention w/ instruction: gray shading + colored Δ=(inst−blind) in parentheses."""
    if not np.isfinite(inst_val):
        return "---"
    finite = [v for v in all_inst_vals if np.isfinite(v)]
    mn, mx = (min(finite), max(finite)) if finite else (0, 1)
    intens = 0
    if mx > mn:
        frac = (mx - inst_val) / (mx - mn)   # lower inst = better
        intens = round(frac * 4) * 5
    val_str = format(inst_val, fmt)
    if intens > 0:
        val_str = rf"\cellcolor{{gray!{intens}}}{val_str}"
    if np.isfinite(blind_val):
        delta = inst_val - blind_val
        sign = "+" if delta >= 0 else ""
        color = "[HTML]{CC7A00}" if delta > 0 else "[HTML]{0072B2}"
        val_str += rf"\,{{\scriptsize(\textcolor{color}{{{sign}{delta:.1f}}})}}"
    return val_str


def fmt_js_cell_with_delta(inst_val, blind_val, all_vals, higher_better, ci_bounds,
                           fmt=".3f", rank=np.nan):
    """DJS value cell: CI coloring/checkmark as usual, plus colored Δ=(inst−blind) in brackets."""
    base = fmt_value_cell(inst_val, all_vals, higher_better=higher_better,
                          ci_bounds=ci_bounds, fmt=fmt, rank=rank)
    if not np.isfinite(blind_val):
        return base
    delta = inst_val - blind_val
    sign = "+" if delta >= 0 else ""
    # For DJS: lower is better, so higher inst vs blind is worsening (orange)
    color = "[HTML]{CC7A00}" if delta > 0 else "[HTML]{0072B2}"
    return base + rf"\,{{\scriptsize(\textcolor{color}{{{sign}{delta:.3f}}})}}"


def fmt_sbert_cell_with_delta(inst_val, blind_val, all_vals, higher_better, ci_bounds,
                              fmt=".3f", rank=np.nan):
    """SBERT/ρ value cell: CI coloring/checkmark as usual, plus colored Δ=(inst−blind) in brackets.
    For higher-better metrics: positive delta (inst > blind) = improvement = blue."""
    base = fmt_value_cell(inst_val, all_vals, higher_better=higher_better,
                          ci_bounds=ci_bounds, fmt=fmt, rank=rank)
    if not np.isfinite(blind_val):
        return base
    delta = inst_val - blind_val
    sign = "+" if delta >= 0 else ""
    # For SBERT/ρ: higher is better, so positive delta (inst > blind) is improvement (blue)
    color = "[HTML]{0072B2}" if delta > 0 else "[HTML]{CC7A00}"
    return base + rf"\,{{\scriptsize(\textcolor{color}{{{sign}{delta:.3f}}})}}"


# ── Data loading (identical to gen_alignment_ranking_full_scale.py) ───────────

def _all_model_ids():
    vlm_ids   = [vlm_id for _, sizes in VLM_FAMILIES for _, vlm_id, _ in sizes]
    lm_ids    = [lm_id  for _, sizes in VLM_FAMILIES for _, _, lm_id in sizes if lm_id is not None]
    sa_ids    = [mid    for _, _, mid in SA_MODELS]
    think_ids = [mid    for _, _, mid in THINK_MODELS]
    return vlm_ids + lm_ids + sa_ids + think_ids


def compute_abstention_rates(all_models):
    blind_df = pd.read_csv(EXPORTS / "responses_model_blind.csv")
    inst_df  = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")
    result = {}
    for model in all_models:
        b_rows = blind_df[blind_df["model"] == model]["response"].fillna("").astype(str)
        i_rows = inst_df[ inst_df["model"]  == model]["response"].fillna("").astype(str)

        def _rate(rows):
            if len(rows) == 0:
                return float("nan")
            abst = sum(is_abstained(classify(r, [])) for r in rows)
            return float(abst / len(rows) * 100.0)

        result[model] = (_rate(b_rows), _rate(i_rows))
    return result


def _compute_js_from_csv(all_models, responses_csv, out_keys):
    """Shared JS computation: load model responses from csv, compute YN and Count JS vs human."""
    from notebooks.helpers import load_human_responses, _clean_answer_series

    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other")
                 for qid, ann in mapper.annotations.items()}

    human_raw = load_human_responses()
    model_raw = pd.read_csv(EXPORTS / responses_csv)

    variants = ("C", "B", "A")
    human_raw = human_raw[human_raw["variant"].isin(variants)].copy()
    model_raw = model_raw[model_raw["variant"].isin(variants)].copy()

    qvs = set(zip(human_raw["question_id"].astype(int), human_raw["variant"].astype(str)))
    model_raw = model_raw[
        [qv in qvs for qv in zip(model_raw["question_id"].astype(int),
                                  model_raw["variant"].astype(str))]
    ].copy()

    human_raw["answer_type"] = human_raw["question_id"].astype(int).map(qid2atype).fillna("other")
    model_raw["answer_type"] = model_raw["question_id"].astype(int).map(qid2atype).fillna("other")
    human_raw["clean_response"] = _clean_answer_series(human_raw["response"])
    model_raw["clean_response"] = _clean_answer_series(model_raw["response"])

    h_all = _mark_abstentions(human_raw.copy())
    m_all = _mark_abstentions(model_raw.copy())
    m_all = m_all[m_all["model"].isin(all_models)].copy()

    yn_key, cnt_key = out_keys
    result = {m: {yn_key: np.nan, cnt_key: np.nan} for m in all_models}

    for model_name, model_sub_all in m_all.groupby("model"):
        if model_name not in all_models:
            continue
        for answer_type, key in [("yes/no", yn_key), ("number", cnt_key)]:
            human_sub, model_sub, category_order = _prepare_distribution_category(
                h_all.copy(), model_sub_all.copy(), answer_type)
            model_sub = model_sub[~model_sub["is_abstained"]].copy()
            kept_qvs = set(zip(model_sub["question_id"].astype(int),
                               model_sub["variant"].astype(str)))
            human_sub = human_sub[
                [qv in kept_qvs for qv in
                 zip(human_sub["question_id"].astype(int),
                     human_sub["variant"].astype(str))]
            ].copy()
            human_sub = human_sub[~human_sub["is_abstained"]].copy()
            js_vals = []
            for (qid, variant), hq in human_sub.groupby(["question_id", "variant"]):
                mq = model_sub[
                    (model_sub["question_id"].astype(int) == int(qid)) &
                    (model_sub["variant"].astype(str) == str(variant))
                ]
                if hq.empty or mq.empty:
                    continue
                hc = hq["category"].value_counts().reindex(category_order, fill_value=0)
                mc = mq["category"].value_counts().reindex(category_order, fill_value=0)
                if hc.sum() == 0 or mc.sum() == 0:
                    continue
                _, _, js, _, _ = _distribution_stats_from_counts(hc, mc)
                js_vals.append(float(js))
            if js_vals:
                result[model_name][key] = float(np.mean(js_vals))

    return result


def compute_js_all_models(all_models):
    return _compute_js_from_csv(all_models, "responses_model_inst_blind.csv",
                                ("yesno_js", "count_js"))


def compute_js_blind_all_models(all_models):
    return _compute_js_from_csv(all_models, "responses_model_blind.csv",
                                ("yesno_js_blind", "count_js_blind"))


def compute_sbert(all_models):
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)

    hm = pc[pc["pair_type"] == "HM"]
    hh = pc[pc["pair_type"] == "HH"]

    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other")
                 for qid, ann in mapper.annotations.items()}
    ft_qids = {qid for qid, at in qid2atype.items() if at == "other"}

    hm_ft = hm[hm["question_id"].astype(int).isin(ft_qids)]
    hh_ft = hh[hh["question_id"].astype(int).isin(ft_qids)]

    h_sbert = hh_ft.groupby("subject_1")["sbert_score"].mean().values
    human_refs = {
        "ft_p5":     float(np.percentile(h_sbert, 5)),
        "ft_p95":    float(np.percentile(h_sbert, 95)),
        "ft_median": float(np.median(h_sbert)),
    }

    result = {m: {"ft_sbert": np.nan} for m in all_models}
    for model in all_models:
        mg = hm_ft[hm_ft["subject_2"] == model]
        if mg.empty:
            continue
        result[model]["ft_sbert"] = float(mg["sbert_score"].mean())

    return result, human_refs


def compute_sbert_blind(all_models):
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned_blind.parquet")
    pc = filter_abstained_pairs(pc)

    hm = pc[pc["pair_type"] == "HM"]

    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other")
                 for qid, ann in mapper.annotations.items()}
    ft_qids = {qid for qid, at in qid2atype.items() if at == "other"}

    hm_ft = hm[hm["question_id"].astype(int).isin(ft_qids)]

    result = {m: {"ft_sbert_blind": np.nan} for m in all_models}
    for model in all_models:
        mg = hm_ft[hm_ft["subject_2"] == model]
        if mg.empty:
            continue
        result[model]["ft_sbert_blind"] = float(mg["sbert_score"].mean())

    return result


def compute_spearman_r_blind(all_models, ft_only=False):
    from scipy.stats import spearmanr as _sr

    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned_blind.parquet")
    pc = filter_abstained_pairs(pc)
    hm = pc[pc["pair_type"] == "HM"]
    hh = pc[pc["pair_type"] == "HH"]

    if ft_only:
        mapper = VQAAnswerMapper()
        mapper._load()
        qid2atype = {int(qid): ann.get("answer_type", "other")
                     for qid, ann in mapper.annotations.items()}
        ft_qids = {qid for qid, at in qid2atype.items() if at == "other"}
        hm = hm[hm["question_id"].astype(int).isin(ft_qids)]
        hh = hh[hh["question_id"].astype(int).isin(ft_qids)]

    ref_pooled = hh.groupby(["question_id", "variant"])["sbert_score"].mean()

    rho_per_model: dict[str, float] = {}
    for m in all_models:
        mg = hm[hm["subject_2"] == m]
        if mg.empty:
            rho_per_model[m] = float("nan")
            continue
        mm = mg.groupby(["question_id", "variant"])["sbert_score"].mean()
        common = mm.index.intersection(ref_pooled.index)
        if len(common) >= 20:
            rho_per_model[m] = float(_sr(mm[common].values, ref_pooled[common].values)[0])
        else:
            rho_per_model[m] = float("nan")
    return rho_per_model


# ── Full-1k abstention ───────────────────────────────────────────────────────

def compute_abstention_rates_1k(all_models):
    """Abstention over all 1000 questions × 3 variants from raw JSONL files.
    Returns nan for any model whose JSONL is missing or has < 1000 rows."""

    def _rate(path: Path) -> float:
        if not path.exists():
            return float("nan")
        with open(path) as f:
            lines = [json.loads(l) for l in f]
        if len(lines) < 1000:
            return float("nan")
        total = abstained = 0
        for ex in lines:
            ga = ex.get("generated_answers", {}) or {}
            for ct in _CT_MAP:
                raw = ga.get(ct) or ""
                ans = _STRIP_THINK.sub("", raw).strip()
                total += 1
                if is_abstained(classify(ans, [])):
                    abstained += 1
        return float(abstained / total * 100.0) if total > 0 else float("nan")

    result: dict[str, tuple[float, float]] = {}
    for m in all_models:
        if m not in MODEL_LOGIT_MAP:
            result[m] = (float("nan"), float("nan"))
            continue
        subdir, folder = MODEL_LOGIT_MAP[m]
        base = LOGITS / subdir / "pretrained" / folder
        result[m] = (
            _rate(base / "vqa_1k_control_blind.jsonl"),
            _rate(base / "vqa_1k_control_inst_blind.jsonl"),
        )
    return result


# ── Spearman ρ ────────────────────────────────────────────────────────────────

def compute_spearman_r(all_models, ft_only=False):
    from scipy.stats import spearmanr as _sr

    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)
    hm = pc[pc["pair_type"] == "HM"]
    hh = pc[pc["pair_type"] == "HH"]

    if ft_only:
        mapper = VQAAnswerMapper()
        mapper._load()
        qid2atype = {int(qid): ann.get("answer_type", "other")
                     for qid, ann in mapper.annotations.items()}
        ft_qids = {qid for qid, at in qid2atype.items() if at == "other"}
        hm = hm[hm["question_id"].astype(int).isin(ft_qids)]
        hh = hh[hh["question_id"].astype(int).isin(ft_qids)]

    ref_pooled = hh.groupby(["question_id", "variant"])["sbert_score"].mean()
    h_rho_vals = []
    for p, grp in hh.groupby("subject_1"):
        pm  = grp.groupby(["question_id", "variant"])["sbert_score"].mean()
        loo = hh[hh["subject_1"] != p].groupby(["question_id", "variant"])["sbert_score"].mean()
        common = pm.index.intersection(loo.index)
        if len(common) >= 20:
            rho, _ = _sr(pm[common].values, loo[common].values)
            h_rho_vals.append(float(rho))

    rho_refs = {
        "spearman_r":    float(np.median(h_rho_vals)),
        "spearman_r_lo": float(np.percentile(h_rho_vals, 5)),
        "spearman_r_hi": float(np.percentile(h_rho_vals, 95)),
    }

    rho_per_model: dict[str, float] = {}
    n_vals: list[int] = []
    for m in all_models:
        mg = hm[hm["subject_2"] == m]
        if mg.empty:
            rho_per_model[m] = float("nan")
            continue
        mm = mg.groupby(["question_id", "variant"])["sbert_score"].mean()
        common = mm.index.intersection(ref_pooled.index)
        if len(common) >= 20:
            rho_per_model[m] = float(_sr(mm[common].values, ref_pooled[common].values)[0])
            n_vals.append(len(common))
        else:
            rho_per_model[m] = float("nan")
    typical_n = int(np.median(n_vals)) if n_vals else 0
    return rho_per_model, rho_refs, typical_n


# ── Table builder ─────────────────────────────────────────────────────────────

def _fmt_ref(val, lo, hi, fmt=".3f"):
    """Human reference cell: value on first line, CI in tiny font on second line."""
    if not np.isfinite(val):
        return "---"
    def _strip(x):
        raw = format(x, fmt)
        return raw.lstrip("0") if raw.startswith("0.") and len(raw) > 3 else raw
    return rf"\makecell{{{_strip(val)} \\ {{\tiny [{_strip(lo)},{_strip(hi)}]}}}}"


def build_table(metrics, human_refs):
    all_models = _all_model_ids()

    ci  = dict(STATIC_CI)
    med = dict(STATIC_MED)
    if "ft_p5" in human_refs:
        ci["ft_sbert"]  = (human_refs["ft_p5"], human_refs["ft_p95"])
        med["ft_sbert"] = human_refs["ft_median"]
    if "spearman_r_ft_lo" in human_refs:
        ci["spearman_r_ft"]  = (human_refs["spearman_r_ft_lo"], human_refs["spearman_r_ft_hi"])
        med["spearman_r_ft"] = human_refs["spearman_r_ft"]
    if "spearman_r_all_lo" in human_refs:
        ci["spearman_r_all"]  = (human_refs["spearman_r_all_lo"], human_refs["spearman_r_all_hi"])
        med["spearman_r_all"] = human_refs["spearman_r_all"]

    # Per-metric ranks
    all_ranks: dict[str, dict[str, float]] = {}
    for key, _, rank_by_med, lower_better, _ in METRIC_CFG:
        all_ranks[key] = compute_ranks(
            all_models, metrics, rank_by_med, lower_better, key, med.get(key)
        )

    # Per-column value lists for shade normalization
    col_vals: dict[str, list[float]] = {
        key: [metrics[m].get(key, np.nan) for m in all_models]
        for key, *_ in METRIC_CFG
    }

    n_metrics = len(METRIC_CFG)
    col_spec = r">{\raggedright\arraybackslash}p{2.2cm}c" + "c" * n_metrics
    n_total = 2 + n_metrics
    # human LOO row needs multiline cells → need array package p-col wrapper
    # handled by \renewcommand{\arraystretch}

    n_ft  = int(human_refs.get("n_ft",  0))
    n_all = int(human_refs.get("n_all", 0))

    # Pre-build col_vals for inst keys (needed for shade normalisation)
    inst_col_vals = {
        "abs_inst":    [metrics[m].get("abs_inst",    np.nan) for m in all_models],
        "abs_inst_1k": [metrics[m].get("abs_inst_1k", np.nan) for m in all_models],
    }

    def _model_row(m, size, fam_cell=""):
        cells = []
        for key, _, _, lower_better, fmt in METRIC_CFG:
            val  = metrics[m].get(key, np.nan)
            rank = all_ranks[key].get(m, np.nan)
            if key in ABS_INST_KEYS:
                blind_key = "abs_blind" if key == "abs_inst" else "abs_blind_1k"
                blind_val = metrics[m].get(blind_key, np.nan)
                cells.append(fmt_abs_inst_with_delta(
                    val, blind_val, inst_col_vals[key], fmt=fmt))
            elif key in DJS_BLIND_MAP:
                blind_val = metrics[m].get(DJS_BLIND_MAP[key], np.nan)
                cells.append(fmt_js_cell_with_delta(
                    val, blind_val, col_vals[key],
                    higher_better=not lower_better,
                    ci_bounds=ci.get(key),
                    fmt=fmt,
                    rank=rank,
                ))
            elif key in SBERT_DELTA_MAP:
                blind_val = metrics[m].get(SBERT_DELTA_MAP[key], np.nan)
                cells.append(fmt_sbert_cell_with_delta(
                    val, blind_val, col_vals[key],
                    higher_better=not lower_better,
                    ci_bounds=ci.get(key),
                    fmt=fmt,
                    rank=rank,
                ))
            elif key in VALUE_KEYS:
                cells.append(fmt_value_cell(
                    val, col_vals[key],
                    higher_better=not lower_better,
                    ci_bounds=ci.get(key),
                    fmt=fmt,
                    rank=rank,
                ))
            else:
                cells.append(fmt_metric_cell(val, rank, ci.get(key)))
        return (fam_cell + " & " + size + " & " +
                " & ".join(cells) + r" \\")

    def _human_loo_row():
        cells = []
        for key, _, _, _, fmt in METRIC_CFG:
            if key == "yesno_js":
                cells.append(_fmt_ref(STATIC_MED["yesno_js"],
                                      STATIC_CI["yesno_js"][0], STATIC_CI["yesno_js"][1], fmt))
            elif key == "count_js":
                cells.append(_fmt_ref(STATIC_MED["count_js"],
                                      STATIC_CI["count_js"][0], STATIC_CI["count_js"][1], fmt))
            elif key == "ft_sbert":
                cells.append(_fmt_ref(human_refs.get("ft_median", np.nan),
                                      human_refs.get("ft_p5", np.nan),
                                      human_refs.get("ft_p95", np.nan), fmt))
            elif key == "spearman_r_ft":
                cells.append(_fmt_ref(human_refs.get("spearman_r_ft", np.nan),
                                      human_refs.get("spearman_r_ft_lo", np.nan),
                                      human_refs.get("spearman_r_ft_hi", np.nan), fmt))
            elif key == "spearman_r_all":
                cells.append(_fmt_ref(human_refs.get("spearman_r_all", np.nan),
                                      human_refs.get("spearman_r_all_lo", np.nan),
                                      human_refs.get("spearman_r_all_hi", np.nan), fmt))
            else:
                cells.append("---")
        return (r"\textit{Human LOO-CV} & --- & " + " & ".join(cells) + r" \\")

    def _sa_group(model_list, label_key):
        lines = []
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_total}}}{{c}}{{\textit{{{GROUP_LABELS[label_key]}}}}}\\ ")
        lines.append(r"\midrule")
        fam_counts: dict[str, int] = {}
        for fam, _, _ in model_list:
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
        fam_printed: set[str] = set()
        for fam, sz, mid in model_list:
            fam_cell = fam if fam not in fam_printed else ""
            fam_printed.add(fam)
            lines.append(_model_row(mid, sz, fam_cell))
        return lines

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.96}")
    lines.append(
        r"\caption{Full-scale alignment ranking across all evaluated model sizes. "
        r"All questions $\times$ variants pooled; abstention classifier applied. "
        r"All columns show values under the blind+instruction condition; "
        r"{\scriptsize(colored $\Delta$)} = change from blind-only. "
        r"For $\djs$: {\textcolor[HTML]{CC7A00}{orange}}$+$ = more divergence with instruction; {\textcolor[HTML]{0072B2}{blue}}$-$ = less. "
        r"For $\sHM$/$\rho$: {\textcolor[HTML]{0072B2}{blue}}$+$ = improvement with instruction; {\textcolor[HTML]{CC7A00}{orange}}$-$ = degradation. "
        r"\colorbox[HTML]{D9EAF3}{Blue} = within human LOO-CV 95\,\% CI; "
        r"\colorbox[HTML]{F7EBD9}{Orange} = over-aligned (below CI lower bound for $\djs$, above upper bound for $\sHM$/$\rho$). "
        r"$\rho$ = Spearman correlation between per-question model and human difficulty. "
        r"Abst(\%) = abstention rate under the instruction condition with $\Delta$ from blind-only. "
        r"Gray shading = lower is better. "
        r"Human LOO-CV row: median with {\tiny[5th, 95th percentile]} CI across leave-one-out raters.}"
    )
    lines.append(r"\label{tab:alignment_ranking_full_scale_ticks}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # ── Two-level header ──────────────────────────────────────────────────────
    # Top row: djs(2) | sHM(1) | rho(2) | Abst(2 combined)
    lines.append(
        r"\multirow{2}{*}{\textbf{Model}} "
        r"& \multirow{2}{*}{\textbf{Sz}} "
        r"& \multicolumn{2}{c}{$\djs$} "
        r"& $\sHM$ "
        r"& \multicolumn{2}{c}{$\rho$} "
        r"& \multicolumn{2}{c}{Abst\,(\%)} \\"
    )
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(
        rf" & & Y/N\,($N{{\leq}}{N_YESNO_PAIRS}$) & Cnt\,($N{{\leq}}{N_COUNT_PAIRS}$)"
        rf" & ($N{{\leq}}{n_ft}$)"
        rf" & FT & All"
        rf" & $N{{\leq}}{N_STUDY_PAIRS}$"
        rf" & $N{{\leq}}{N_1K_PAIRS}$ \\"
    )
    lines.append(r"\midrule")

    # ── Human LOO-CV reference row ────────────────────────────────────────────
    lines.append(_human_loo_row())

    # ── VLMs ──────────────────────────────────────────────────────────────────
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{n_total}}}{{c}}{{\textit{{{GROUP_LABELS['VLM']}}}}}\\ ")
    lines.append(r"\midrule")

    for fam_name, sizes in VLM_FAMILIES:
        n = len(sizes)
        for i, (sz, vlm_id, _) in enumerate(sizes):
            fam_cell = fam_name if i == 0 else ""
            lines.append(_model_row(vlm_id, sz, fam_cell))
        lines.append(r"\addlinespace[3pt]")

    # ── Backbone Decoders ──────────────────────────────────────────────────────
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{n_total}}}{{c}}{{\textit{{{GROUP_LABELS['Backbone Decoder']}}}}}\\ ")
    lines.append(r"\midrule")

    for fam_name, sizes in VLM_FAMILIES:
        lm_sizes = [(sz, lm_id) for sz, _, lm_id in sizes if lm_id is not None]
        if not lm_sizes:
            continue
        n = len(lm_sizes)
        for i, (sz, lm_id) in enumerate(lm_sizes):
            fam_cell = fam_name if i == 0 else ""
            lines.append(_model_row(lm_id, sz, fam_cell))
        lines.append(r"\addlinespace[3pt]")

    # ── SA-LLMs (non-thinking) ────────────────────────────────────────────────
    lines.extend(_sa_group(SA_MODELS, "Standalone LLM"))

    # ── Thinking Models ───────────────────────────────────────────────────────
    lines.extend(_sa_group(THINK_MODELS, "Thinking"))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_models = _all_model_ids()
    print(f"Total models: {len(all_models)}")

    print("Computing JS scores (inst_blind)…")
    js_data = compute_js_all_models(all_models)

    print("Computing JS scores (blind)…")
    js_blind_data = compute_js_blind_all_models(all_models)

    print("Computing SBERT (inst_blind)…")
    sbert_data, human_refs = compute_sbert(all_models)

    print("Computing SBERT (blind)…")
    sbert_blind_data = compute_sbert_blind(all_models)

    print("Computing Spearman ρ (free-text only)…")
    rho_ft, rho_refs_ft, n_ft = compute_spearman_r(all_models, ft_only=True)

    print("Computing Spearman ρ (all questions)…")
    rho_all, rho_refs_all, n_all = compute_spearman_r(all_models, ft_only=False)

    print("Computing Spearman ρ blind (free-text only)…")
    rho_ft_blind = compute_spearman_r_blind(all_models, ft_only=True)

    print("Computing Spearman ρ blind (all questions)…")
    rho_all_blind = compute_spearman_r_blind(all_models, ft_only=False)

    print("Computing abstention rates (study subset)…")
    abs_data = compute_abstention_rates(all_models)

    print("Computing abstention rates (full 1k)…")
    abs_data_1k = compute_abstention_rates_1k(all_models)

    metrics: dict[str, dict[str, float]] = {}
    for m in all_models:
        metrics[m] = {}
        metrics[m].update(js_data.get(m, {}))
        metrics[m].update(js_blind_data.get(m, {}))
        metrics[m].update(sbert_data.get(m, {}))
        metrics[m].update(sbert_blind_data.get(m, {}))
        metrics[m]["spearman_r_ft"]       = rho_ft.get(m, float("nan"))
        metrics[m]["spearman_r_all"]      = rho_all.get(m, float("nan"))
        metrics[m]["spearman_r_ft_blind"]  = rho_ft_blind.get(m, float("nan"))
        metrics[m]["spearman_r_all_blind"] = rho_all_blind.get(m, float("nan"))
        blind_pct, inst_pct = abs_data.get(m, (np.nan, np.nan))
        metrics[m]["abs_blind"] = blind_pct
        metrics[m]["abs_inst"]  = inst_pct
        blind_1k, inst_1k = abs_data_1k.get(m, (np.nan, np.nan))
        metrics[m]["abs_blind_1k"]  = blind_1k
        metrics[m]["abs_inst_1k"]   = inst_1k

    human_refs["spearman_r_ft"]     = rho_refs_ft["spearman_r"]
    human_refs["spearman_r_ft_lo"]  = rho_refs_ft["spearman_r_lo"]
    human_refs["spearman_r_ft_hi"]  = rho_refs_ft["spearman_r_hi"]
    human_refs["spearman_r_all"]    = rho_refs_all["spearman_r"]
    human_refs["spearman_r_all_lo"] = rho_refs_all["spearman_r_lo"]
    human_refs["spearman_r_all_hi"] = rho_refs_all["spearman_r_hi"]
    human_refs["n_ft"]  = n_ft
    human_refs["n_all"] = n_all

    # Print ρ values for inspection
    print(f"\nHuman LOO ρ (ft)  CI: [{human_refs['spearman_r_ft_lo']:.3f}, {human_refs['spearman_r_ft_hi']:.3f}]  median={human_refs['spearman_r_ft']:.3f}")
    print(f"Human LOO ρ (all) CI: [{human_refs['spearman_r_all_lo']:.3f}, {human_refs['spearman_r_all_hi']:.3f}]  median={human_refs['spearman_r_all']:.3f}")
    print(f"\n{'Model':<35} {'ρ (ft)':>8} {'ρ (all)':>8}")
    for m in all_models:
        rft  = metrics[m].get("spearman_r_ft",  float("nan"))
        rall = metrics[m].get("spearman_r_all", float("nan"))
        ft_s  = f"{rft:.3f}"  if np.isfinite(rft)  else "nan"
        all_s = f"{rall:.3f}" if np.isfinite(rall) else "nan"
        print(f"  {m:<33} {ft_s:>8} {all_s:>8}")

    tex = build_table(metrics, human_refs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
