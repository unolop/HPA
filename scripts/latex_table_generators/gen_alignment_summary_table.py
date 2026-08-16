r"""
Redesigned alignment summary table for main paper.
Columns: Y/N djs (tick) | Count djs (tick) | sHM (absolute) | Blind Abst% (absolute + arrow) | Inst Abst% (absolute)

Visual encoding:
  green!20  = within human LOO 95% CI (djs and sHM columns)
  orange!30 = over-concentrated (val < CI lo for djs; val > CI hi for sHM)
  gray grad = outside CI in wrong direction
  tick mark (checkmark) indicates within-CI; blank otherwise
  abstention: absolute % values with ↓/↑ arrow showing inst vs blind direction

The old rank table moves to the appendix (gen_alignment_ranking_table.py → tables/supp/).
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

OUT = ROOT / "latex/AAAI2026/LaTeX/tables/paper/alignment_summary_7b.tex"

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

# Abstention rates (blind%, inst%) from classifier
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
    "yesno_js":  (0.260, 0.282),
    "count_js":  (0.493, 0.554),
    "ft_sbert":  None,  # filled at runtime
}
STATIC_MED: dict[str, float | None] = {
    "yesno_js":  0.271,
    "count_js":  0.524,
    "ft_sbert":  None,
}


def get_color(val: float, ci: tuple[float, float],
              lower_is_better: bool, all_finite: list[float]) -> str:
    lo, hi = ci
    if lo <= val <= hi:
        return r"\cellcolor{green!20}"
    if val < lo:
        if lower_is_better:
            return r"\cellcolor{orange!30}"
        bad = [v for v in all_finite if v < lo]
        max_d = max(abs(v - lo) for v in bad) if bad else 0
        intens = _gray(abs(val - lo), max_d)
        return rf"\cellcolor{{gray!{intens}}}" if intens else ""
    else:  # val > hi
        if lower_is_better:
            bad = [v for v in all_finite if v > hi]
            max_d = max(abs(v - hi) for v in bad) if bad else 0
            intens = _gray(abs(val - hi), max_d)
            return rf"\cellcolor{{gray!{intens}}}" if intens else ""
        return r"\cellcolor{orange!30}"


def _gray(dist: float, max_dist: float, max_int: int = 20) -> int:
    if max_dist <= 0 or not np.isfinite(dist):
        return 0
    raw = (1.0 - dist / max_dist) * max_int
    return max(0, min(max_int, round(raw / 5) * 5))


def within_ci(val: float, ci: tuple[float, float] | None) -> bool:
    if ci is None or not np.isfinite(val):
        return False
    return ci[0] <= val <= ci[1]


def build_table(metrics: dict, human_refs: dict) -> str:
    all_models = [m for ms in MODEL_ORDER_7B.values() for m in ms]

    # Inject abstention
    for m in all_models:
        blind, inst = ABS_COMPUTED.get(m, (np.nan, np.nan))
        metrics[m]["abs_blind"] = blind
        metrics[m]["abs_inst"]  = inst

    # Fill dynamic CI/median
    ci = dict(STATIC_CI)
    med = dict(STATIC_MED)
    if "ft_p5" in human_refs:
        ci["ft_sbert"]  = (human_refs["ft_p5"], human_refs["ft_p95"])
        med["ft_sbert"] = human_refs["ft_median"]

    # Precompute all_finite lists for color scaling
    all_finite: dict[str, list[float]] = {}
    for key in ("yesno_js", "count_js", "ft_sbert"):
        all_finite[key] = [
            metrics[m].get(key, np.nan)
            for m in all_models
            if np.isfinite(metrics[m].get(key, np.nan))
        ]

    n_cols = 6  # model + 3 metric cols + 2 abst cols
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.96}")
    lines.append(
        r"\caption{Per-model alignment summary (all questions $\times$ variants pooled; "
        r"abstention classifier applied). "
        r"\colorbox{green!20}{\checkmark} = within human LOO 95\,\% CI; "
        r"blank = outside. "
        r"\colorbox{orange!30}{Orange} = over-concentrated. "
        r"\colorbox{gray!15}{Gray} = outside CI in wrong direction. "
        r"$\sHM$: absolute HM SBERT. "
        r"Abst$_\mathrm{blind}$: abstention without instruction; "
        r"Abst$_\mathrm{inst}$: with instruction. "
        r"Arrow shows direction of change ($\downarrow$ decreases, $\uparrow$ increases).}"
    )
    lines.append(r"\label{tab:alignment_summary_7b}")

    col_spec = r">{\raggedright\arraybackslash}p{2.6cm}ccccc"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    lines.append(
        r"\multirow{2}{*}{\textbf{Model}} "
        r"& \multicolumn{2}{c}{$\djs$} "
        r"& \multirow{2}{*}{$\sHM$} "
        r"& \multicolumn{2}{c}{Abst (\%)} \\"
    )
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}")
    lines.append(r" & Y/N & Count & & Blind & +Inst \\")

    for grp, ms in MODEL_ORDER_7B.items():
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{n_cols}}}{{c}}{{\textit{{{GROUP_LABELS[grp]}}}}}\\"
        )
        lines.append(r"\midrule")
        for m in ms:
            disp = TABLE_DISPLAY.get(m, m)
            cells: list[str] = []

            # ── djs Y/N
            val_yn = metrics[m].get("yesno_js", np.nan)
            color_yn = get_color(val_yn, ci["yesno_js"], True, all_finite["yesno_js"]) if ci["yesno_js"] and np.isfinite(val_yn) else ""
            tick_yn = r"\checkmark" if within_ci(val_yn, ci["yesno_js"]) else ""
            cells.append(f"{color_yn}{tick_yn}")

            # ── djs Count
            val_cnt = metrics[m].get("count_js", np.nan)
            color_cnt = get_color(val_cnt, ci["count_js"], True, all_finite["count_js"]) if ci["count_js"] and np.isfinite(val_cnt) else ""
            tick_cnt = r"\checkmark" if within_ci(val_cnt, ci["count_js"]) else ""
            cells.append(f"{color_cnt}{tick_cnt}")

            # ── sHM (absolute value + color)
            val_s = metrics[m].get("ft_sbert", np.nan)
            color_s = get_color(val_s, ci["ft_sbert"], False, all_finite["ft_sbert"]) if ci["ft_sbert"] and np.isfinite(val_s) else ""
            tick_s = r"\checkmark " if within_ci(val_s, ci["ft_sbert"]) else ""
            if np.isfinite(val_s):
                val_s_str = f"{val_s:.3f}".lstrip("0")
                cells.append(f"{color_s}{tick_s}{val_s_str}")
            else:
                cells.append("---")

            # ── Blind abstention (absolute %)
            blind_pct, inst_pct = ABS_COMPUTED.get(m, (np.nan, np.nan))
            if np.isfinite(blind_pct):
                cells.append(f"{blind_pct:.1f}")
            else:
                cells.append("---")

            # ── Inst abstention (absolute % + arrow showing change direction)
            if np.isfinite(inst_pct) and np.isfinite(blind_pct):
                delta = inst_pct - blind_pct
                arrow = r"$\downarrow$" if delta < -0.05 else (r"$\uparrow$" if delta > 0.05 else "")
                cells.append(f"{inst_pct:.1f}{arrow}")
            else:
                cells.append("---")

            lines.append(f"{disp} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print("Loading metrics…")
    metrics, errors, human_refs, v_data, q_data = load_metrics()
    tex = build_table(metrics, human_refs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"Saved: {OUT}")
    print(tex)
