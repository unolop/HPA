r"""
Comprehensive alignment ranking table.
Columns: Yes/No JS | Count JS | S_HM | r | Blind Abs% | Inst Abs% | Avg Rank

Visual encoding:
  green!20  = within human LOO 95% CI
  orange!30 = below CI lower bound (over-concentrated, JS only)
  gray grad = outside CI in wrong direction; darker = farther
  rank-based gray for abstention columns (no human CI)

Cells show rank (1=best) not raw values.
Human LOO reference row sits above the column headers.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/latex_table_generators"))

from figures.alignment_profile.alignment_profile_barplot_7b import (
    load_metrics, MODEL_ORDER_7B,
)

OUT = ROOT / "latex/AAAI2026/LaTeX/tables/alignment_ranking_7b.tex"

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

# ── Abstention rates from classifier (all Q×variant pairs pooled) ─────────────
# Computed via analysis.utils.abstention.classify on pair_cache_cleaned*.parquet
# "0" and all numeric answers are NOT classified as abstention (confirmed).
# Format: (blind_pct, inst_pct)
#   blind_pct : abstention in no-instruction condition
#   inst_pct  : abstention when instruction says no image present
#               (positive alignment signal — model correctly declines)
ABS_COMPUTED: dict[str, tuple[float, float]] = {
    # VLMs  (blind_pct, inst_pct) — computed via soft_abst on responses_model_{blind,inst_blind}.csv
    "InternVL-8B":        ( 0.00,  2.07),
    "LLaVA-1.5-7B":       ( 2.95,  3.54),
    "LLaVA-Mistral":      ( 2.65,  1.77),
    "Qwen3-VL-8B":        ( 5.33,  3.87),
    "LLaVA-Vicuna":       ( 2.68,  0.89),
    # Backbone Decoders
    "InternVL-8B (LM)":   (17.11,  9.47),
    "LLaVA-1.5 (LM)":     ( 0.00,  0.59),
    "LLaVA-Mistral (LM)": ( 0.30,  2.07),
    "Qwen3-VL-8B (LM)":   ( 5.33,  3.25),
    "LLaVA-Vicuna (LM)":  ( 0.30,  0.59),
    # Standalone LLMs
    "Qwen2.5-7B-Instruct": ( 9.14,  6.51),
    "Qwen3-8B":            (12.72,  4.72),
    "Qwen3-8B (think)":    (12.72,  4.72),
    "Vicuna-7B":           ( 1.18,  1.49),
}

# ── Human reference CIs and medians ──────────────────────────────────────────
# CI bounds from paper tables/captions (static); medians and r/SBERT CIs from load_metrics
STATIC_CI: dict[str, tuple[float, float] | None] = {
    "yesno_js":  (0.260, 0.282),
    "count_js":  (0.493, 0.554),
    "ft_sbert":  None,   # filled from human_refs at runtime
    "pearson_r": None,   # filled from human_refs at runtime
    "abs_blind": None,
    "abs_inst":  None,
}
STATIC_MED: dict[str, float | None] = {
    "yesno_js":  0.271,   # LOO median from hm_grouped_distribution table
    "count_js":  0.524,
    "ft_sbert":  None,
    "pearson_r": None,
    "abs_blind": None,
    "abs_inst":  None,
}

# ── Metric configuration: (key, header, rank_by_median, lower_is_better, format)
# rank_by_median=True  → rank by |val − human_median| (rank 1 = closest to median)
# rank_by_median=False → rank by value direction given by lower_is_better
# S_HM and r are adjacent; abstention at the end
METRIC_CFG = [
    ("yesno_js",   r"Y/N $\djs$",  True,  True,  ".3f"),
    ("count_js",   r"Cnt $\djs$",  True,  True,  ".3f"),
    ("ft_sbert",   r"$\sHM$",      True,  False, ".3f"),
    ("pearson_r",  r"$r$",         True,  False, ".3f"),
    ("abs_blind",  r"Blind Abs\%", False, False, ".1f"),
    ("abs_inst",   r"Inst Abs\%",  False, True,  ".1f"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Gradient gray intensity (0..max_int). dist=0 → max_int (darkest), dist=max_dist → 0."""
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
    r"""Return \cellcolor{...} prefix or empty string (3-zone CI logic, no tick)."""
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
            # SBERT/r above CI hi: over-fitted → orange
            return r"\cellcolor{orange!30}"


def rank_str(r: float) -> str:
    """Format rank as integer when possible, else one decimal."""
    if not np.isfinite(r):
        return "---"
    return str(int(r)) if r == int(r) else f"{r:.1f}"


def fmt_human_cell(key: str, ci_dict: dict, med_dict: dict, spec: str) -> str:
    """Human LOO reference cell: median on top, [lo, hi] below in scriptsize."""
    m = med_dict.get(key)
    bounds = ci_dict.get(key)
    if m is None or bounds is None:
        return "---"
    lo, hi = bounds
    m_s  = format(m,  spec).lstrip("0") or "0"
    lo_s = format(lo, spec).lstrip("0") or "0"
    hi_s = format(hi, spec).lstrip("0") or "0"
    return rf"\makecell[r]{{{m_s}\\{{\scriptsize [{lo_s},{hi_s}]}}}}"


# ── Main builder ──────────────────────────────────────────────────────────────

def build_table(metrics: dict, human_refs: dict) -> str:
    all_models = [m for ms in MODEL_ORDER_7B.values() for m in ms]

    # Inject abstention metrics from classifier
    for m in all_models:
        blind, inst = ABS_COMPUTED.get(m, (np.nan, np.nan))
        metrics[m]["abs_blind"] = blind
        metrics[m]["abs_inst"]  = inst

    # Fill dynamic CI/median from loaded human_refs
    ci = dict(STATIC_CI)
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

    # Average rank per model
    avg_rank: dict[str, float] = {}
    for m in all_models:
        rv = [all_ranks[key][m] for key, *_ in METRIC_CFG
              if np.isfinite(all_ranks[key].get(m, np.nan))]
        avg_rank[m] = float(np.mean(rv)) if rv else np.nan

    n_total = len(METRIC_CFG) + 2  # model col + metrics + avg rank

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.96}")
    lines.append(
        r"\caption{Per-model alignment ranks across all criteria (all questions $\times$ variants pooled; "
        r"abstention classifier applied). "
        r"Cells show rank (1\,=\,best for all metrics). "
        r"\colorbox{green!20}{Green} = within human LOO 95\,\% CI. "
        r"\colorbox{orange!30}{Orange} = over-concentrated ($<$CI for $\djs$; $>$CI for $\sHM$/$\pr$). "
        r"\colorbox{gray!15}{Gray} = outside CI in the wrong direction; darker = closer to human range. "
        r"Abst: abstention rate. "
        r"w/o (blind): rank 1\,=\,highest abstention (model spontaneously detects missing image). "
        r"w/ (inst): rank 1\,=\,lowest abstention (model follows the instruction to answer). "
        r"Avg = mean rank across all six metrics.}"
    )
    lines.append(r"\label{tab:alignment_ranking_7b}")

    col_spec = r">{\raggedright\arraybackslash}p{2.6cm}" + "c" * len(METRIC_CFG) + "c"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # ── Two-level header ──────────────────────────────────────────────────────
    lines.append(
        r"\multirow{2}{*}{\textbf{Model}} "
        r"& \multicolumn{2}{c}{$\djs$} "
        r"& \multicolumn{2}{c}{SBERT} "
        r"& \multicolumn{2}{c}{Abst(\%)} "
        r"& \multirow{2}{*}{\textbf{Avg}} \\"
    )
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r" & Y/N & Count & $\sHM$ & $r$ & w/o & w/ & \\")

    for grp, ms in MODEL_ORDER_7B.items():
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{n_total}}}{{c}}{{\textit{{{GROUP_LABELS[grp]}}}}}\\"
        )
        lines.append(r"\midrule")
        for m in ms:
            disp = TABLE_DISPLAY.get(m, m)
            cells: list[str] = []
            for key, _, rank_by_med, lower_better, spec in METRIC_CFG:
                val  = metrics[m].get(key, np.nan)
                rank = all_ranks[key].get(m, np.nan)
                bounds = ci.get(key)

                rs = rank_str(rank)
                if rank <= 1.5:
                    rs = rf"\textbf{{{rs}}}"
                elif rank <= 2.5:
                    rs = rf"\underline{{{rs}}}"

                if bounds is not None:
                    all_finite = [metrics[mm].get(key, np.nan)
                                  for mm in all_models
                                  if np.isfinite(metrics[mm].get(key, np.nan))]
                    color = get_color(val, bounds, lower_better, all_finite)
                    cells.append(f"{color}{rs}")
                else:
                    # Abstention: rank-based gray, show rank
                    n_finite = sum(1 for mm in all_models
                                   if np.isfinite(metrics[mm].get(key, np.nan)))
                    intens = rank_gray(rank, n_finite)
                    cells.append(rf"\cellcolor{{gray!{intens}}}{rs}" if intens > 0 else rs)

            ar = avg_rank[m]
            if np.isfinite(ar):
                sorted_avgs = sorted(
                    [(mm, avg_rank[mm]) for mm in all_models if np.isfinite(avg_rank[mm])],
                    key=lambda x: x[1]
                )
                avg_rank_pos = next(i + 1 for i, (mm, _) in enumerate(sorted_avgs) if mm == m)
                intens = rank_gray(avg_rank_pos, len(sorted_avgs))
                val_str = f"{ar:.1f}"
                if avg_rank_pos == 1:
                    val_str = rf"\textbf{{{val_str}}}"
                elif avg_rank_pos == 2:
                    val_str = rf"\underline{{{val_str}}}"
                ar_str = (rf"\cellcolor{{gray!{intens}}}{val_str}" if intens > 0
                          else val_str)
            else:
                ar_str = "---"
            lines.append(rf"{disp} & " + " & ".join(cells) + rf" & {ar_str} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print("Loading metrics…")
    metrics, errors, human_refs, v_data, q_data = load_metrics()
    tex = build_table(metrics, human_refs)
    OUT.write_text(tex)
    print(f"Saved: {OUT}")
    print(tex)
