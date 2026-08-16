r"""
Full-scale alignment ranking table — all model sizes.

Extends gen_alignment_ranking_table.py to cover all VLMs, Backbone Decoders, and SA-LLMs
at every evaluated scale.

Columns: Size | Y/N JS | Count JS | S_HM | r | Blind Abs% | Inst Abs% | Avg Rank

Visual encoding:
  green!20  = within human LOO 95% CI
  orange!30 = over-concentrated (JS below CI; sHM/r above CI)
  gray grad = outside CI in wrong direction; darker = closer
  Bold = rank 1, underline = rank 2 (global across all models)

Output: latex/AAAI2026/LaTeX/tables/supp/alignment_ranking_full_scale.tex
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/latex_table_generators"))

from scipy.stats import pearsonr as _pearsonr
from figures.helpers import filter_abstained_pairs
from notebooks.helpers import (
    _mark_abstentions,
    _prepare_distribution_category,
    _distribution_stats_from_counts,
)
from analysis.utils.vqa import VQAAnswerMapper
from analysis.utils.abstention import classify, is_abstained

EXPORTS = ROOT / "analysis/session2/exports"
OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/alignment_ranking_full_scale.tex"

# ── Human reference CIs (same constants as barplot_7b) ───────────────────────
HUMAN_YESNO_CI  = (0.219, 0.375)
HUMAN_YESNO_MED = 0.256
HUMAN_COUNT_CI  = (0.395, 0.693)
HUMAN_COUNT_MED = 0.506

STATIC_CI: dict[str, tuple[float, float] | None] = {
    "yesno_js":  (0.260, 0.282),  # from paper table caption
    "count_js":  (0.493, 0.554),
    "ft_sbert":  None,   # filled at runtime from HH LOO
    "pearson_r": None,   # filled at runtime from HH LOO
    "abs_blind": None,
    "abs_inst":  None,
}
STATIC_MED: dict[str, float | None] = {
    "yesno_js":  0.271,
    "count_js":  0.524,
    "ft_sbert":  None,
    "pearson_r": None,
    "abs_blind": None,
    "abs_inst":  None,
}

# ── Model layout ──────────────────────────────────────────────────────────────

# VLM families: (family_name, [(size, vlm_id, lm_id_or_None)])
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
        ("13B", "LLaVA-Vicuna-13B",  None),   # no LM counterpart
    ]),
]

# SA-LLMs: (family, size, model_id)
SA_MODELS: list[tuple[str, str, str]] = [
    ("Qwen3",   "0.6B", "Qwen3-0.6B"),
    ("Qwen3",   "1.7B", "Qwen3-1.7B"),
    ("Qwen3",   "1.7B (think)", "Qwen3-1.7B (think)"),
    ("Qwen3",   "4B",   "Qwen3-4B"),
    ("Qwen3",   "8B",   "Qwen3-8B"),
    ("Qwen3",   "8B (think)",   "Qwen3-8B (think)"),
    ("Qwen3",   "32B",  "Qwen3-32B"),
    ("Qwen3",   "32B (think)",  "Qwen3-32B (think)"),
    ("Qwen2.5", "7B",   "Qwen2.5-7B-Instruct"),
    ("Vicuna",  "7B",   "Vicuna-7B"),
    ("Vicuna",  "13B",  "Vicuna-13B"),
    ("Mistral", "7B",   "Mistral-7B"),
]

GROUP_LABELS = {
    "VLM":              "VLMs",
    "Backbone Decoder": "Backbone Decoders",
    "Standalone LLM":   "SA-LLMs",
}

# ── Metric config ─────────────────────────────────────────────────────────────
# (key, header, rank_by_median, lower_is_better, format_spec)
METRIC_CFG = [
    ("yesno_js",   r"Y/N $\djs$",  True,  True,  ".3f"),
    ("count_js",   r"Cnt $\djs$",  True,  True,  ".3f"),
    ("ft_sbert",   r"$\sHM$",      True,  False, ".3f"),
    ("abs_blind",  r"Blind Abs\%", False, False, ".1f"),
    ("abs_inst",   r"Inst Abs\%",  False, True,  ".1f"),
]


# ── Helpers (verbatim from gen_alignment_ranking_table.py) ───────────────────

def compute_ranks(all_models: list[str], metrics: dict,
                  rank_by_median: bool, lower_is_better: bool,
                  key: str, human_median: float | None) -> dict[str, float]:
    """Global ranks 1..N; ties get average rank."""
    vals = [(m, metrics[m].get(key, np.nan)) for m in all_models]
    finite = [(m, v) for m, v in vals if np.isfinite(v)]
    ranks: dict[str, float] = {m: np.nan for m in all_models}
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
        while j < len(sorted_m) - 1 and sorted_m[j + 1][1] == sorted_m[i][1]:
            j += 1
        avg = ((i + 1) + (j + 1)) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_m[k][0]] = avg
        i = j + 1
    return ranks


def gray_intensity(dist: float, max_dist: float, max_int: int = 20) -> int:
    """Gradient gray intensity (0..max_int). dist=0 → max_int (darkest)."""
    if max_dist <= 0 or not np.isfinite(dist):
        return 0
    raw = (1.0 - dist / max_dist) * max_int
    return max(0, min(max_int, round(raw / 5) * 5))


def rank_gray(rank: float, n_models: int, max_int: int = 20) -> int:
    """Rank-based gray (rank=1 → max_int, rank=N → 0). Used for abstention cols."""
    if not np.isfinite(rank) or n_models <= 1:
        return 0
    raw = (1.0 - (rank - 1) / (n_models - 1)) * max_int
    return max(0, round(raw / 5) * 5)


def get_color(val: float, ci: tuple[float, float] | None,
              lower_is_better: bool, all_finite: list[float]) -> str:
    r"""Return \cellcolor{...} prefix or empty string."""
    if not np.isfinite(val) or ci is None:
        return ""
    lo, hi = ci
    if lo <= val <= hi:
        return r"\cellcolor{green!20}"
    if val < lo:
        if lower_is_better:
            return r"\cellcolor{orange!30}"
        else:
            bad_vals = [v for v in all_finite if v < lo]
            max_d = max(abs(v - lo) for v in bad_vals) if bad_vals else 0
            intens = gray_intensity(abs(val - lo), max_d)
            return rf"\cellcolor{{gray!{intens}}}" if intens > 0 else ""
    else:  # val > hi
        if lower_is_better:
            bad_vals = [v for v in all_finite if v > hi]
            max_d = max(abs(v - hi) for v in bad_vals) if bad_vals else 0
            intens = gray_intensity(abs(val - hi), max_d)
            return rf"\cellcolor{{gray!{intens}}}" if intens > 0 else ""
        else:
            return r"\cellcolor{orange!30}"


def rank_str(r: float) -> str:
    """Format rank as integer when possible, else one decimal."""
    if not np.isfinite(r):
        return "---"
    return str(int(r)) if r == int(r) else f"{r:.1f}"


# ── Data loading ──────────────────────────────────────────────────────────────

def _all_model_ids() -> list[str]:
    """Flat list of all model IDs in order: VLMs, Backbone Decoders, SA-LLMs."""
    vlm_ids = [vlm_id for _, sizes in VLM_FAMILIES for _, vlm_id, _ in sizes]
    lm_ids  = [lm_id  for _, sizes in VLM_FAMILIES for _, _, lm_id in sizes if lm_id is not None]
    sa_ids  = [mid    for _, _, mid in SA_MODELS]
    return vlm_ids + lm_ids + sa_ids


def compute_abstention_rates(all_models: list[str]) -> dict[str, tuple[float, float]]:
    """Compute (blind_pct, inst_pct) abstention rates for every model."""
    blind_df = pd.read_csv(EXPORTS / "responses_model_blind.csv")
    inst_df  = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")

    result: dict[str, tuple[float, float]] = {}
    for model in all_models:
        b_rows = blind_df[blind_df["model"] == model]["response"].fillna("").astype(str)
        i_rows = inst_df[ inst_df["model"]  == model]["response"].fillna("").astype(str)

        def _rate(rows: pd.Series) -> float:
            if len(rows) == 0:
                return float("nan")
            abst = sum(is_abstained(classify(r, [])) for r in rows)
            return float(abst / len(rows) * 100.0)

        result[model] = (_rate(b_rows), _rate(i_rows))
    return result


def compute_js_all_models(all_models: list[str]) -> dict[str, dict[str, float]]:
    """Compute per-question-mean JS for yes/no and count questions, all models.

    Loads inst_blind condition (variants C, B, A) and uses the same
    abstention-filtered per-question JS method as the paper table.
    """
    from notebooks.helpers import load_human_responses, _clean_answer_series

    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other")
                 for qid, ann in mapper.annotations.items()}

    human_raw = load_human_responses()
    model_raw = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")

    variants = ("C", "B", "A")
    human_raw = human_raw[human_raw["variant"].isin(variants)].copy()
    model_raw = model_raw[model_raw["variant"].isin(variants)].copy()

    # Keep only question/variant pairs that appear in human data
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

    # Filter to only the models we need
    m_all = m_all[m_all["model"].isin(all_models)].copy()

    result: dict[str, dict[str, float]] = {m: {"yesno_js": np.nan, "count_js": np.nan}
                                            for m in all_models}

    for model_name, model_sub_all in m_all.groupby("model"):
        if model_name not in all_models:
            continue
        for answer_type, key in [("yes/no", "yesno_js"), ("number", "count_js")]:
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


def compute_sbert_r(all_models: list[str]) -> tuple[dict[str, dict[str, float]], dict]:
    """Compute ft_sbert, pearson_r for all models, plus human LOO reference CIs."""
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

    # Human SBERT LOO CI
    h_sbert = hh_ft.groupby("subject_1")["sbert_score"].mean().values
    human_refs = {
        "ft_p5":     float(np.percentile(h_sbert, 5)),
        "ft_p95":    float(np.percentile(h_sbert, 95)),
        "ft_median": float(np.median(h_sbert)),
    }

    # Human r LOO CI
    ref_pooled = hh_ft.groupby(["question_id", "variant"])["sbert_score"].mean()
    h_r_vals = []
    for p, grp in hh_ft.groupby("subject_1"):
        pm  = grp.groupby(["question_id", "variant"])["sbert_score"].mean()
        loo = hh_ft[hh_ft["subject_1"] != p].groupby(
            ["question_id", "variant"])["sbert_score"].mean()
        common = pm.index.intersection(loo.index)
        if len(common) < 20:
            continue
        r, _ = _pearsonr(pm[common], loo[common])
        h_r_vals.append(r)
    human_refs.update({
        "r_p5":     float(np.percentile(h_r_vals, 5)),
        "r_p95":    float(np.percentile(h_r_vals, 95)),
        "r_median": float(np.median(h_r_vals)),
    })

    # Per-model SBERT and Pearson r
    result: dict[str, dict[str, float]] = {m: {"ft_sbert": np.nan, "pearson_r": np.nan}
                                            for m in all_models}
    for model in all_models:
        mg = hm_ft[hm_ft["subject_2"] == model]
        if mg.empty:
            continue
        result[model]["ft_sbert"] = float(mg["sbert_score"].mean())

        mm = mg.groupby(["question_id", "variant"])["sbert_score"].mean()
        common = mm.index.intersection(ref_pooled.index)
        if len(common) >= 20:
            r, _ = _pearsonr(mm[common].values, ref_pooled[common].values)
            result[model]["pearson_r"] = float(r)

    return result, human_refs


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(metrics: dict, human_refs: dict) -> str:
    all_models = _all_model_ids()

    # Fill CIs from human_refs
    ci  = dict(STATIC_CI)
    med = dict(STATIC_MED)
    if "ft_p5" in human_refs:
        ci["ft_sbert"]  = (human_refs["ft_p5"], human_refs["ft_p95"])
        med["ft_sbert"] = human_refs["ft_median"]
    if "r_p5" in human_refs:
        ci["pearson_r"]  = (human_refs["r_p5"], human_refs["r_p95"])
        med["pearson_r"] = human_refs["r_median"]

    # Per-metric global ranks
    all_ranks: dict[str, dict[str, float]] = {}
    for key, _, rank_by_med, lower_better, _ in METRIC_CFG:
        human_median = med.get(key)
        all_ranks[key] = compute_ranks(
            all_models, metrics, rank_by_med, lower_better, key, human_median
        )

    # Average rank
    avg_rank: dict[str, float] = {}
    for m in all_models:
        rv = [all_ranks[key][m] for key, *_ in METRIC_CFG
              if np.isfinite(all_ranks[key].get(m, np.nan))]
        avg_rank[m] = float(np.mean(rv)) if rv else np.nan

    sorted_avgs = sorted(
        [(m, avg_rank[m]) for m in all_models if np.isfinite(avg_rank[m])],
        key=lambda x: x[1]
    )

    n_metrics = len(METRIC_CFG)
    # col spec: family (p{2.2cm}), size (c), then metrics, then avg
    col_spec = r">{\raggedright\arraybackslash}p{2.2cm}c" + "c" * n_metrics + "c"

    def _fmt_cell(m: str, key: str, rank_by_med: bool, lower_better: bool, spec: str) -> str:
        val  = metrics[m].get(key, np.nan)
        rank = all_ranks[key].get(m, np.nan)
        bounds = ci.get(key)

        rs = rank_str(rank)
        if np.isfinite(rank) and rank <= 1.5:
            rs = rf"\textbf{{{rs}}}"
        elif np.isfinite(rank) and rank <= 2.5:
            rs = rf"\underline{{{rs}}}"

        if bounds is not None:
            all_finite = [metrics[mm].get(key, np.nan)
                          for mm in all_models
                          if np.isfinite(metrics[mm].get(key, np.nan))]
            color = get_color(val, bounds, lower_better, all_finite)
            return f"{color}{rs}"
        else:
            n_finite = sum(1 for mm in all_models
                           if np.isfinite(metrics[mm].get(key, np.nan)))
            intens = rank_gray(rank, n_finite)
            return (rf"\cellcolor{{gray!{intens}}}{rs}" if intens > 0 else rs)

    def _avg_cell(m: str) -> str:
        ar = avg_rank[m]
        if not np.isfinite(ar):
            return "---"
        avg_rank_pos = next(
            (i + 1 for i, (mm, _) in enumerate(sorted_avgs) if mm == m), np.nan
        )
        intens = rank_gray(avg_rank_pos, len(sorted_avgs))
        val_str = f"{ar:.1f}"
        if avg_rank_pos == 1:
            val_str = rf"\textbf{{{val_str}}}"
        elif avg_rank_pos == 2:
            val_str = rf"\underline{{{val_str}}}"
        return (rf"\cellcolor{{gray!{intens}}}{val_str}" if intens > 0 else val_str)

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.96}")
    lines.append(
        r"\caption{Full-scale alignment ranking table covering all evaluated model sizes "
        r"(VLMs, Backbone Decoders, and SA-LLMs). "
        r"All questions $\times$ variants pooled; abstention classifier applied. "
        r"Cells show rank (1\,=\,best for all metrics). "
        r"\colorbox{green!20}{Green} = within human LOO 95\,\% CI. "
        r"\colorbox{orange!30}{Orange} = over-concentrated ($<$CI for $\djs$; $>$CI for $\sHM$). "
        r"\colorbox{gray!15}{Gray} = outside CI in the wrong direction; darker = closer. "
        r"Blind Abs\%: abstention without instruction (higher = model spontaneously detects missing image). "
        r"Inst Abs\%: abstention with instruction (lower = model follows the instruction to answer). "
        r"Avg = mean rank across all five metrics.}"
    )
    lines.append(r"\label{tab:alignment_ranking_full_scale}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Two-level header
    lines.append(
        r"\multirow{2}{*}{\textbf{Model}} "
        r"& \multirow{2}{*}{\textbf{Sz}} "
        r"& \multicolumn{2}{c}{$\djs$} "
        r"& \multirow{2}{*}{$\sHM$} "
        r"& \multicolumn{2}{c}{Abst(\%)} "
        r"& \multirow{2}{*}{\textbf{Avg}} \\"
    )
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}")
    lines.append(r" & & Y/N & Count & & w/o & w/ & \\")

    n_total = 2 + n_metrics + 1   # model + size + metrics + avg

    def _model_row(m: str, size: str, fam_cell: str = "") -> str:
        cells = [_fmt_cell(m, key, rbm, lb, sp)
                 for key, _, rbm, lb, sp in METRIC_CFG]
        return (fam_cell + " & " + size + " & " +
                " & ".join(cells) + " & " + _avg_cell(m) + r" \\")

    # ── VLMs ──────────────────────────────────────────────────────────────────
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{n_total}}}{{c}}{{\textit{{{GROUP_LABELS['VLM']}}}}}\\ ")
    lines.append(r"\midrule")

    for fam_name, sizes in VLM_FAMILIES:
        n = len(sizes)
        for i, (sz, vlm_id, _) in enumerate(sizes):
            fam_cell = (r"\multirow{" + str(n) + r"}{*}{" + fam_name + "}") if i == 0 else ""
            lines.append(_model_row(vlm_id, sz, fam_cell))
        lines.append(r"\addlinespace[1pt]")

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
            fam_cell = (r"\multirow{" + str(n) + r"}{*}{" + fam_name + "}") if i == 0 else ""
            lines.append(_model_row(lm_id, sz, fam_cell))
        lines.append(r"\addlinespace[1pt]")

    # ── SA-LLMs ───────────────────────────────────────────────────────────────
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{n_total}}}{{c}}{{\textit{{{GROUP_LABELS['Standalone LLM']}}}}}\\ ")
    lines.append(r"\midrule")

    prev_fam = None
    i_in_fam = 0
    # Count per family for multirow
    fam_counts: dict[str, int] = {}
    for fam, sz, mid in SA_MODELS:
        fam_counts[fam] = fam_counts.get(fam, 0) + 1

    fam_printed: dict[str, bool] = {}
    for fam, sz, mid in SA_MODELS:
        if fam not in fam_printed:
            fam_cell = (r"\multirow{" + str(fam_counts[fam]) + r"}{*}{" + fam + "}")
            fam_printed[fam] = True
        else:
            fam_cell = ""
        lines.append(_model_row(mid, sz, fam_cell))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_models = _all_model_ids()
    print(f"Total models: {len(all_models)}")

    print("Computing JS scores…")
    js_data = compute_js_all_models(all_models)

    print("Computing SBERT and Pearson r…")
    sbert_r_data, human_refs = compute_sbert_r(all_models)

    print("Computing abstention rates…")
    abs_data = compute_abstention_rates(all_models)

    # Assemble metrics dict
    metrics: dict[str, dict[str, float]] = {}
    for m in all_models:
        metrics[m] = {}
        metrics[m].update(js_data.get(m, {}))
        metrics[m].update(sbert_r_data.get(m, {}))
        blind_pct, inst_pct = abs_data.get(m, (np.nan, np.nan))
        metrics[m]["abs_blind"] = blind_pct
        metrics[m]["abs_inst"]  = inst_pct

    # Print summary
    print("\nMetrics summary:")
    for m in all_models:
        v = metrics[m]
        print(f"  {m:35s}  yn={v.get('yesno_js', float('nan')):.3f}  "
              f"cnt={v.get('count_js', float('nan')):.3f}  "
              f"sbert={v.get('ft_sbert', float('nan')):.3f}  "
              f"r={v.get('pearson_r', float('nan')):.3f}  "
              f"abs_b={v.get('abs_blind', float('nan')):.1f}%  "
              f"abs_i={v.get('abs_inst', float('nan')):.1f}%")

    tex = build_table(metrics, human_refs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
