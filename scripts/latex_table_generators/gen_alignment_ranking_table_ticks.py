r"""
Alignment ranking table — 7B models, tick variant.

Same layout as gen_alignment_ranking_table.py but with a different visual style:
  - No green/orange/gray cell colors for JS/sHM/rho columns.
    Instead, a \checkmark is appended to the rank when the model's value is within
    the human LOO 95 % CI.
  - Abstention columns show actual percentage values (not ranks) with gray shading
    proportional to "goodness" (higher blind = darker; lower inst = darker).
  - Arrow labels in abstention headers: w/o ↑  and  w/ ↓

Requires \usepackage{amssymb} in the LaTeX preamble for \checkmark.

Output: latex/AAAI2026/LaTeX/tables/paper/alignment_ranking_7b_ticks.tex
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/latex_table_generators"))

from figures.alignment_profile.alignment_profile_barplot_7b import (
    load_metrics, MODEL_ORDER_7B,
)
from figures.helpers import filter_abstained_pairs
from analysis.utils.vqa import VQAAnswerMapper

OUT         = ROOT / "latex/AAAI2026/LaTeX/tables/paper/alignment_ranking_7b_ticks.tex"
OUT_COLORED = ROOT / "latex/AAAI2026/LaTeX/tables/paper/alignment_ranking_7b_ticks_colored.tex"

GROUP_LABELS = {
    "VLM":              "VLMs",
    "Backbone Decoder": "Backbone Decoders",
    "Standalone LLM":   "SA-LLMs",
}

TABLE_DISPLAY = {
    "InternVL-8B":        "InternVL3.5",
    "LLaVA-1.5-7B":       "LLaVA-1.5",
    "LLaVA-Mistral":      "LLaVA-Mistral",
    "LLaVA-Vicuna":       "LLaVA-Vicuna",
    "Qwen3-VL-8B":        "Qwen3-VL",
    "InternVL-8B (LM)":   "InternVL3.5",
    "LLaVA-1.5 (LM)":    "LLaVA-1.5",
    "LLaVA-Mistral (LM)": "LLaVA-Mistral",
    "LLaVA-Vicuna (LM)":  "LLaVA-Vicuna",
    "Qwen3-VL-8B (LM)":   "Qwen3-VL",
    "Qwen2.5-7B-Instruct": "Qwen2.5-Instruct",
    "Qwen3-8B":            "Qwen3",
    "Qwen3-8B (think)":   "Qwen3 (think)",
    "Vicuna-7B":           "Vicuna",
}

ABS_COMPUTED: dict[str, tuple[float, float]] = {
    "InternVL-8B":        ( 0.00,  2.07),
    "LLaVA-1.5-7B":       ( 2.95,  3.54),
    "LLaVA-Mistral":      ( 2.65,  1.77),
    "Qwen3-VL-8B":        ( 5.33,  3.87),
    "LLaVA-Vicuna":       ( 2.68,  0.89),
    "InternVL-8B (LM)":   (17.11,  9.47),
    "LLaVA-1.5 (LM)":     ( 0.00,  0.59),
    "LLaVA-Mistral (LM)": ( 0.30,  2.07),
    "Qwen3-VL-8B (LM)":   ( 5.33,  3.25),
    "LLaVA-Vicuna (LM)":  ( 0.30,  0.59),
    "Qwen2.5-7B-Instruct": ( 9.14,  6.51),
    "Qwen3-8B":            (12.72,  4.72),
    "Qwen3-8B (think)":    (12.72,  4.72),
    "Vicuna-7B":           ( 1.18,  1.49),
}

STATIC_CI: dict[str, tuple[float, float] | None] = {
    "yesno_js":   (0.260, 0.282),
    "count_js":   (0.493, 0.554),
    "ft_sbert":   None,
    "spearman_r": None,
    "abs_blind":  None,
    "abs_inst":   None,
}
STATIC_MED: dict[str, float | None] = {
    "yesno_js":   0.271,
    "count_js":   0.524,
    "ft_sbert":   None,
    "spearman_r": None,
    "abs_blind":  None,
    "abs_inst":   None,
}

# rank_by_median=True  → rank by |val − human_median| (only used for JS cols now)
# rank_by_median=False → not used for value-display cols
# VALUE_KEYS: shown as actual values with gray shade, not ranks
VALUE_KEYS = {"ft_sbert", "spearman_r"}
# TICK_KEYS: shown as symbol-only (no rank/value) with blue/orange + CI symbol
TICK_KEYS  = {"ft_sbert", "spearman_r", "yesno_js", "count_js"}
# ABS_KEYS: abstention columns — plain value, bold best, underline second-best, no gray
ABS_KEYS   = {"abs_blind", "abs_inst"}

METRIC_CFG = [
    ("yesno_js",   r"Y/N",              True,  True,  ".3f"),
    ("count_js",   r"Cnt",              True,  True,  ".3f"),
    ("ft_sbert",   r"$\sHM$",           True,  False, ".3f"),
    ("spearman_r", r"$\rho$",           True,  False, ".3f"),
    ("abs_blind",  r"w/o $\uparrow$",   False, False, ".1f"),
    ("abs_inst",   r"w/ $\downarrow$",  False, True,  ".1f"),
]


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

_BG_BLUE   = r"\cellcolor[HTML]{D9EAF3}"
_BG_ORANGE = r"\cellcolor[HTML]{F7EBD9}"
_SYM_IN    = r"\,$\checkmark$"
_SYM_OVER  = r"\,$\triangle$"
_SYM_IN_COLORED  = r"\,\textcolor[HTML]{2E7D32}{$\checkmark$}"
_SYM_OVER_COLORED = r"\,\textcolor[HTML]{CC7A00}{$\triangle$}"

_NO_COLOR = False  # set to True for plain (no cellcolor) output


def fmt_metric_cell(val, rank, ci_bounds):
    """JS rank cell: blue+✓ if within CI, orange+▽ if below CI (over-aligned), plain otherwise."""
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
        if val < lo:
            return _BG_ORANGE + rs + _SYM_OVER
        if lo <= val <= hi:
            return _BG_BLUE + rs + _SYM_IN
    return rs


def fmt_value_cell(val, all_vals, higher_better, ci_bounds, fmt=".3f"):
    """Actual-value cell: gray shading (higher=darker or lower=darker) + checkmark if in CI."""
    if not np.isfinite(val):
        return "---"
    finite = [v for v in all_vals if np.isfinite(v)]
    mn, mx = (min(finite), max(finite)) if finite else (0, 1)
    intens = 0
    if mx > mn:
        frac = (val - mn) / (mx - mn) if higher_better else (mx - val) / (mx - mn)
        intens = round(frac * 4) * 5  # 0, 5, 10, 15, 20
    raw = format(val, fmt)
    # Strip leading zero only for multi-decimal values (e.g., 0.446 → .446)
    val_str = raw.lstrip("0") if raw.startswith("0.") and len(raw) > 3 else raw
    if ci_bounds is not None:
        lo, hi = ci_bounds
        if lo <= val <= hi:
            val_str = val_str + r"\,$\checkmark$"
    if intens > 0:
        return rf"\cellcolor{{gray!{intens}}}{val_str}"
    return val_str


def fmt_abst_cell(val, best_val, second_val):
    """Abstention cell: plain value, bold = best unique value, underline = second-best unique value."""
    if not np.isfinite(val):
        return "---"
    s = f"{val:.1f}"
    if np.isfinite(best_val) and abs(val - best_val) < 1e-9:
        return rf"\textbf{{{s}}}"
    if np.isfinite(second_val) and abs(val - second_val) < 1e-9:
        return rf"\underline{{{s}}}"
    return s


def fmt_tick_cell(val, all_vals, higher_better, ci_bounds, colored=True):
    """Symbol-only cell: checkmark if within CI, triangle if over-aligned, dash otherwise.
    colored=True: cell background colors (for _colored.tex).
    colored=False: colored symbols, no cell background (default plain output)."""
    if not np.isfinite(val):
        return "---"
    if ci_bounds is not None:
        lo, hi = ci_bounds
        if (higher_better and val > hi) or (not higher_better and val < lo):
            if colored:
                return _BG_ORANGE + _SYM_OVER[2:]  # strip leading \,
            return _SYM_OVER_COLORED[2:]
        if lo <= val <= hi:
            if colored:
                return _BG_BLUE + _SYM_IN[2:]  # strip leading \,
            return _SYM_IN_COLORED[2:]
    return "-"


# ── Spearman ρ (all questions, no ft filter) ──────────────────────────────────

def compute_spearman_r_7b(ft_only=False):
    from scipy.stats import spearmanr as _sr

    pc = pd.read_parquet(ROOT / "analysis/session2/exports/pair_cache_cleaned.parquet")
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

    result: dict[str, dict] = {
        "Human LOO": {
            "spearman_r":    float(np.median(h_rho_vals)),
            "spearman_r_lo": float(np.percentile(h_rho_vals, 5)),
            "spearman_r_hi": float(np.percentile(h_rho_vals, 95)),
        }
    }
    all_7b = [m for ms in MODEL_ORDER_7B.values() for m in ms]
    for m in all_7b:
        mg = hm[hm["subject_2"] == m]
        if mg.empty:
            result[m] = {"spearman_r": float("nan")}
            continue
        mm = mg.groupby(["question_id", "variant"])["sbert_score"].mean()
        common = mm.index.intersection(ref_pooled.index)
        rho = (float(_sr(mm[common].values, ref_pooled[common].values)[0])
               if len(common) >= 20 else float("nan"))
        result[m] = {"spearman_r": rho}
    return result


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(metrics, human_refs, colored=True):
    all_models = [m for ms in MODEL_ORDER_7B.values() for m in ms]

    # Inject abstention + Spearman rho
    for m in all_models:
        blind, inst = ABS_COMPUTED.get(m, (np.nan, np.nan))
        metrics[m]["abs_blind"] = blind
        metrics[m]["abs_inst"]  = inst

    sbert_cache = compute_spearman_r_7b(ft_only=False)
    loo_rho = sbert_cache.get("Human LOO", {})
    for m in all_models:
        metrics[m]["spearman_r"] = sbert_cache.get(m, {}).get("spearman_r", np.nan)

    # Fill dynamic CIs
    ci = dict(STATIC_CI)
    med = dict(STATIC_MED)
    if "ft_p5" in human_refs:
        ci["ft_sbert"]  = (human_refs["ft_p5"],  human_refs["ft_p95"])
        med["ft_sbert"] = human_refs["ft_median"]
    if loo_rho.get("spearman_r") is not None:
        ci["spearman_r"]  = (loo_rho["spearman_r_lo"], loo_rho["spearman_r_hi"])
        med["spearman_r"] = loo_rho["spearman_r"]

    # Per-metric ranks (only used for JS cols, kept for consistency)
    all_ranks: dict[str, dict[str, float]] = {}
    for key, _, rank_by_med, lower_better, _ in METRIC_CFG:
        all_ranks[key] = compute_ranks(
            all_models, metrics, rank_by_med, lower_better, key, med.get(key)
        )

    # Pre-compute per-column value lists for shade normalization
    col_vals: dict[str, list[float]] = {
        key: [metrics[m].get(key, np.nan) for m in all_models]
        for key, *_ in METRIC_CFG
    }

    # Pre-compute best/second-best unique values for abstention columns
    abs_best: dict[str, float] = {}
    abs_second: dict[str, float] = {}
    for key, _, _, lower_better, _ in METRIC_CFG:
        if key not in ABS_KEYS:
            continue
        finite = sorted(
            {v for v in col_vals[key] if np.isfinite(v)},
            reverse=not lower_better,  # best first
        )
        abs_best[key]   = finite[0] if finite else float("nan")
        abs_second[key] = finite[1] if len(finite) > 1 else float("nan")

    n_total = 1 + len(METRIC_CFG)  # model col + metrics

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.96}")
    if colored:
        ci_legend = (
            r"\colorbox[HTML]{D9EAF3}{Blue}\,$\checkmark$ = within human LOO 95\,\% CI; "
            r"\colorbox[HTML]{F7EBD9}{orange}\,$\triangle$ = over-aligned. "
        )
    else:
        ci_legend = r"$\checkmark$ = within human LOO 95\,\% CI; $\triangle$ = over-aligned. "
    lines.append(
        r"\caption{Per-model alignment ranking. "
        + ci_legend
        + r"Abst(\%) = spontaneous (w/o\,$\uparrow$) and instructed (w/\,$\downarrow$) abstention rates (\textbf{bold} = best; \underline{underline} = second-best).}"
    )
    lines.append(r"\label{tab:alignment_ranking_7b_ticks}")

    col_spec = r">{\raggedright\arraybackslash}p{2.6cm}" + "c" * len(METRIC_CFG)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{Model}} "
        r"& \multicolumn{2}{c}{$\djs$} "
        r"& \multicolumn{2}{c}{$\sHM$} "
        r"& \multicolumn{2}{c}{Abst(\%)} \\"
    )
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r" & Yes/No & Count & Other & $\rho$ & w/o $\uparrow$ & w/ $\downarrow$ \\")

    for grp, ms in MODEL_ORDER_7B.items():
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{n_total}}}{{c}}{{\textit{{{GROUP_LABELS[grp]}}}}}\\"
        )
        lines.append(r"\midrule")
        for m in ms:
            disp = TABLE_DISPLAY.get(m, m)
            cells: list[str] = []
            for key, _, rank_by_med, lower_better, fmt in METRIC_CFG:
                val  = metrics[m].get(key, np.nan)
                rank = all_ranks[key].get(m, np.nan)
                if key in TICK_KEYS:
                    cells.append(fmt_tick_cell(
                        val, col_vals[key],
                        higher_better=not lower_better,
                        ci_bounds=ci.get(key),
                        colored=colored,
                    ))
                elif key in ABS_KEYS:
                    cells.append(fmt_abst_cell(val, abs_best[key], abs_second[key]))
                elif key in VALUE_KEYS:
                    cells.append(fmt_value_cell(
                        val, col_vals[key],
                        higher_better=not lower_better,
                        ci_bounds=ci.get(key),
                        fmt=fmt,
                    ))
                else:
                    cells.append(fmt_metric_cell(val, rank, ci.get(key)))
            lines.append(rf"{disp} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print("Loading metrics…")
    metrics, errors, human_refs, v_data, q_data = load_metrics()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Plain version (no cell colors) — primary output
    tex_plain = build_table(metrics, human_refs, colored=False)
    OUT.write_text(tex_plain)
    print(f"Saved: {OUT}")
    # Colored version — archived with _colored suffix
    tex_colored = build_table(metrics, human_refs, colored=True)
    OUT_COLORED.write_text(tex_colored)
    print(f"Saved: {OUT_COLORED}")
    print(tex_plain)
