"""
Full-scale alignment metrics table (all models, all sizes).

Columns: Size | S_HM(Orig) | S_HM(Pron) | Delta | Pearson r

Organized by:
  Part 1: VLMs + Backbone Decoders (side by side, by family)
  Part 2: Standalone LLMs (Qwen3 at all scales, others)

Abstention-filtered throughout.
Output: latex/AAAI2026/LaTeX/tables/scale_alignment_full_table.tex
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figures.helpers import filter_abstained_pairs  # noqa: E402

EXPORTS = ROOT / "analysis/session2/exports"
OUT = ROOT / "latex/AAAI2026/LaTeX/tables/scale_alignment_full_table.tex"

# ── Model layout ─────────────────────────────────────────────────────────────
# (display_name, size_label, vlm_id, lm_id)  -- lm_id=None for SA-LLMs

VLM_FAMILIES = [
    ("InternVL3.5", [
        ("1B",  "InternVL-1B",     "InternVL-1B (LM)"),
        ("2B",  "InternVL-2B",     "InternVL-2B (LM)"),
        ("8B",  "InternVL-8B",     "InternVL-8B (LM)"),
    ]),
    ("Qwen3-VL", [
        ("2B",  "Qwen3-VL-2B",    "Qwen3-VL-2B (LM)"),
        ("4B",  "Qwen3-VL-4B",    "Qwen3-VL-4B (LM)"),
        ("8B",  "Qwen3-VL-8B",    "Qwen3-VL-8B (LM)"),
        ("32B", "Qwen3-VL-32B",   "Qwen3-VL-32B (LM)"),
    ]),
    ("LLaVA-1.5", [
        ("7B",  "LLaVA-1.5-7B",   "LLaVA-1.5 (LM)"),
    ]),
    ("LLaVA-Mistral", [
        ("7B",  "LLaVA-Mistral",  "LLaVA-Mistral (LM)"),
    ]),
    ("LLaVA-Vicuna", [
        ("7B",  "LLaVA-Vicuna",     "LLaVA-Vicuna (LM)"),
        ("13B", "LLaVA-Vicuna-13B", None),  # 13B LM not in data
    ]),
]

SA_MODELS = [
    # (display, size, model_id, think_id or None)
    # Note: 0.6B and 4B think are identical to no-think (same inference data)
    ("Qwen3",   "0.6B",  "Qwen3-0.6B",          None),            # think ≈ identical
    ("Qwen3",   "1.7B",  "Qwen3-1.7B",           "Qwen3-1.7B (think)"),
    ("Qwen3",   "4B",    "Qwen3-4B",              None),            # think = identical
    ("Qwen3",   "8B",    "Qwen3-8B",              "Qwen3-8B (think)"),
    ("Qwen3",   "32B",   "Qwen3-32B",             "Qwen3-32B (think)"),
    ("Qwen2.5", "7B",    "Qwen2.5-7B-Instruct",   None),
    ("Vicuna",  "7B",    "Vicuna-7B",              None),
    ("Vicuna",  "13B",   "Vicuna-13B",             None),
    ("Mistral", "7B",    "Mistral-7B",             None),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)

    hh = pc[pc["pair_type"] == "HH"]
    hm = pc[pc["pair_type"] == "HM"]

    # HH per-question mean (difficulty curve)
    hh_perq = hh.groupby("question_id")["sbert_score"].mean()

    # HM per-variant mean SBERT
    hm_var = hm.groupby(["subject_2", "variant"])["sbert_score"].mean()

    # HH per-variant mean (human reference)
    hh_var = hh.groupby("variant")["sbert_score"].mean()

    return hm, hm_var, hh_perq, hh_var


def get_sbert(hm_var, model: str) -> tuple[float | None, float | None, float | None]:
    """Return (orig_C, pron_A, delta) for a model."""
    try:
        s_c = hm_var.loc[(model, "C")]
    except KeyError:
        s_c = None
    try:
        s_a = hm_var.loc[(model, "A")]
    except KeyError:
        s_a = None
    delta = (s_c - s_a) if (s_c is not None and s_a is not None) else None
    return s_c, s_a, delta


def get_r(hm: pd.DataFrame, hh_perq: pd.Series, model: str) -> float | None:
    """Per-question Pearson r between HM and HH difficulty."""
    hm_perq = hm[hm["subject_2"] == model].groupby("question_id")["sbert_score"].mean()
    common = hh_perq.index.intersection(hm_perq.index)
    if len(common) < 10:
        return None
    r, _ = pearsonr(hh_perq[common], hm_perq[common])
    return float(r)


# ── Formatting ────────────────────────────────────────────────────────────────

def fv(x: float | None, decimals: int = 3) -> str:
    if x is None:
        return r"{\scriptsize ---}"
    return f"{x:.{decimals}f}"


def fd(x: float | None) -> str:
    """Format delta with sign."""
    if x is None:
        return r"{\scriptsize ---}"
    if x >= 0:
        return f"+{x:.3f}"
    else:
        return f"$-${abs(x):.3f}"


def fr(x: float | None) -> str:
    if x is None:
        return r"{\scriptsize ---}"
    return f"{x:.3f}"


# ── Table builders ────────────────────────────────────────────────────────────

def build_table(hm, hm_var, hh_perq, hh_var) -> list[str]:
    # Human reference values
    h_c = hh_var.get("C", None)
    h_a = hh_var.get("A", None)
    h_d = (h_c - h_a) if (h_c and h_a) else None

    lines: list[str] = []

    # ── Preamble ──
    lines += [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\caption{Full-scale alignment metrics for all models"
        r" (abstention-filtered; instruction-blind condition)."
        r" $\sHM$(Orig) and $\sHM$(Pron): mean HM SBERT for"
        r" Original (C) and Pronominalized (A) variants."
        r" $\Delta = \sHM(\text{Orig})-\sHM(\text{Pron})$: positive = worse under pronominalization."
        r" $r$: per-question Pearson correlation between model $\sHM$ and human difficulty $\sHH$."
        r" Backbone Decoder column: LM counterpart at same size ({\scriptsize ---} = not available)."
        r" Think variants shown only where confirmed distinct from no-think."
        r" Human reference: $\sHH$(Orig)\,$=$\,"
        + fv(h_c)
        + r", $\sHH$(Pron)\,$=$\,"
        + fv(h_a)
        + r".}",
        r"\label{tab:scale_alignment_full}",
        r"\begin{tabular}{llcccc|cccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Family}} & \multirow{2}{*}{\textbf{Size}}"
        r" & \multicolumn{4}{c|}{\textbf{VLM}}"
        r" & \multicolumn{4}{c}{\textbf{Backbone Decoder}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r" & & $\sHM$(Orig) & $\sHM$(Pron) & $\Delta$ & $r$"
        r" & $\sHM$(Orig) & $\sHM$(Pron) & $\Delta$ & $r$ \\",
        r"\midrule",
    ]

    for fam_name, sizes in VLM_FAMILIES:
        n = len(sizes)
        for i, (sz, vlm_id, lm_id) in enumerate(sizes):
            fam_cell = (r"\multirow{" + str(n) + r"}{*}{" + fam_name + "}") if i == 0 else ""

            vc, va, vd = get_sbert(hm_var, vlm_id)
            vr = get_r(hm, hh_perq, vlm_id)

            lc, la, ld = get_sbert(hm_var, lm_id)
            lr = get_r(hm, hh_perq, lm_id)

            lines.append(
                f"{fam_cell} & {sz}"
                f" & {fv(vc)} & {fv(va)} & {fd(vd)} & {fr(vr)}"
                f" & {fv(lc)} & {fv(la)} & {fd(ld)} & {fr(lr)}"
                r" \\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}"]

    # ── SA-LLM panel ──
    lines += [
        r"\vspace{4pt}",
        r"\begin{tabular}{llcccc|cccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Family}} & \multirow{2}{*}{\textbf{Size}}"
        r" & \multicolumn{4}{c|}{\textbf{SA-LLM (no think)}}"
        r" & \multicolumn{4}{c}{\textbf{SA-LLM (think)}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r" & & $\sHM$(Orig) & $\sHM$(Pron) & $\Delta$ & $r$"
        r" & $\sHM$(Orig) & $\sHM$(Pron) & $\Delta$ & $r$ \\",
        r"\midrule",
    ]

    prev_fam = None
    for fam_name, sz, model_id, think_id in SA_MODELS:
        if fam_name != prev_fam:
            fam_cell = fam_name
            prev_fam = fam_name
        else:
            fam_cell = ""

        mc, ma, md = get_sbert(hm_var, model_id)
        mr = get_r(hm, hh_perq, model_id)

        if think_id:
            tc, ta, td = get_sbert(hm_var, think_id)
            tr = get_r(hm, hh_perq, think_id)
        else:
            tc, ta, td, tr = None, None, None, None

        lines.append(
            f"{fam_cell} & {sz}"
            f" & {fv(mc)} & {fv(ma)} & {fd(md)} & {fr(mr)}"
            f" & {fv(tc)} & {fv(ta)} & {fd(td)} & {fr(tr)}"
            r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    return lines


def main():
    print("Loading data...")
    hm, hm_var, hh_perq, hh_var = load_data()
    lines = build_table(hm, hm_var, hh_perq, hh_var)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
