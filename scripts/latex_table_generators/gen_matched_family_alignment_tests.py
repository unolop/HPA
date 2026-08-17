"""
Generate matched_family_alignment_tests.tex with exact Wilcoxon p-values.

For each family, compares backbone decoder vs. full VLM and vs. matched SA-LLM
using per-(question, variant) mean HM SBERT, abstention-filtered.
Tests: two-sided Wilcoxon signed-rank, Holm correction across all comparisons.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from figures.helpers import filter_abstained_pairs

OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/matched_family_alignment_tests.tex"

# Matched family definitions: (display_name, decoder, vlm, sa_llm)
FAMILIES = [
    ("InternVL3.5",  "InternVL-8B (LM)",   "InternVL-8B",   "Qwen3-8B"),
    ("Qwen3-VL",     "Qwen3-VL-8B (LM)",   "Qwen3-VL-8B",   "Qwen3-8B"),
    ("LLaVA-1.5",    "LLaVA-1.5 (LM)",     "LLaVA-1.5-7B",  "Vicuna-7B"),
    ("LLaVA-Mistral","LLaVA-Mistral (LM)",  "LLaVA-Mistral", "Mistral-7B"),
    ("LLaVA-Vicuna", "LLaVA-Vicuna (LM)",   "LLaVA-Vicuna",  "Vicuna-7B"),
]


def mean_sbert_per_qv(hm: pd.DataFrame, model: str) -> pd.Series:
    """Mean HM SBERT per (question_id, variant) for one model."""
    sub = hm[hm["subject_2"] == model]
    return sub.groupby(["question_id", "variant"])["sbert_score"].mean()


def fmt_p(p: float) -> str:
    """Format p-value: scientific notation for very small, else 3 sig figs."""
    if p < 0.001:
        exp = int(np.floor(np.log10(p)))
        mantissa = p / 10**exp
        return rf"{mantissa:.1f}{{\times}}10^{{{exp}}}"
    return f"{p:.3f}"


def star(p_holm: float) -> str:
    if p_holm < 0.001:
        return "***"
    if p_holm < 0.01:
        return "**"
    if p_holm < 0.05:
        return "*"
    return "ns"


def main():
    pc = pd.read_parquet(ROOT / "analysis/session2/exports/pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)
    hm = pc[pc["pair_type"] == "HM"].copy()

    # Pre-compute per-(qv) mean for each model needed
    all_models = set()
    for _, dec, vlm, sa in FAMILIES:
        all_models.update([dec, vlm, sa])
    qv_means = {m: mean_sbert_per_qv(hm, m) for m in all_models}

    # Collect all raw p-values for Holm correction (2 tests per family = 10 total)
    raw_results = []  # (family_idx, comparison, dec_mean, cmp_mean, delta, raw_p, n)
    for fi, (name, dec, vlm, sa) in enumerate(FAMILIES):
        dec_s = qv_means[dec]
        for label, cmp_model in [("vlm", vlm), ("sa", sa)]:
            cmp_s = qv_means[cmp_model]
            common = dec_s.index.intersection(cmp_s.index)
            d = dec_s[common].values - cmp_s[common].values
            # Drop zero differences (Wilcoxon requirement)
            nz = d[d != 0]
            if len(nz) < 10:
                raw_p = float("nan")
            else:
                _, raw_p = stats.wilcoxon(dec_s[common].values, cmp_s[common].values,
                                          alternative="two-sided")
            raw_results.append({
                "fi": fi, "label": label,
                "dec_mean": float(dec_s[common].mean()),
                "cmp_mean": float(cmp_s[common].mean()),
                "delta": float(dec_s[common].mean() - cmp_s[common].mean()),
                "raw_p": raw_p,
                "n": len(common),
            })

    # Holm correction across all valid tests
    valid_mask = [not np.isnan(r["raw_p"]) for r in raw_results]
    valid_ps = [r["raw_p"] for r in raw_results if not np.isnan(r["raw_p"])]
    if valid_ps:
        _, p_holm_vals, _, _ = multipletests(valid_ps, method="holm")
    holm_iter = iter(p_holm_vals)
    for r in raw_results:
        r["p_holm"] = float(next(holm_iter)) if not np.isnan(r["raw_p"]) else float("nan")

    # Index results for easy lookup
    res = {}
    for r in raw_results:
        res[(r["fi"], r["label"])] = r

    # Build table
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{%",
        r"\textbf{Matched-family backbone decoder alignment.} "
        r"Each row compares a VLM backbone decoder (the language-model component of the VLM, "
        r"run text-only without its vision encoder) against two reference points: "
        r"(i)~its paired full VLM (same weights, but with the vision encoder active and a blank image), "
        r"and (ii)~a standalone LLM whose base weights match the backbone's language model. "
        r"The \textbf{Decoder} column reports mean abstention-filtered HM SBERT "
        r"across all 339 question--variant pairs; "
        r"\textbf{Score} in each comparison column is the comparator's mean HM SBERT, "
        r"with the parenthetical $\Delta$ = decoder $-$ comparator "
        r"(\textcolor{teal}{teal} = decoder higher; \textcolor{red}{red} = decoder lower). "
        r"Significance is assessed with a two-sided Wilcoxon signed-rank test on paired "
        r"per-(question,~variant) mean SBERT values. "
        r"Holm correction is applied across all 10 tests simultaneously "
        r"(2 comparisons $\times$ 5 families) to control the family-wise error rate. "
        r"$p$ values are Holm-corrected; "
        r"$^{***}p{<}0.001$, $^{**}p{<}0.01$, $^{*}p{<}0.05$.}",
        r"\label{tab:matched_family_alignment_tests}",
        r"\begin{tabular}{l c cc cc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Family}} & \multirow{2}{*}{\textbf{Decoder}} "
        r"& \multicolumn{2}{c}{\textbf{vs.\ Full VLM}} "
        r"& \multicolumn{2}{c}{\textbf{vs.\ SA-LLM}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & Score ($\Delta$) & $p$ & Score ($\Delta$) & $p$ \\",
        r"\midrule",
    ]

    for fi, (name, dec, vlm, sa) in enumerate(FAMILIES):
        rv = res[(fi, "vlm")]
        rs = res[(fi, "sa")]

        dec_val = f"{rv['dec_mean']:.3f}"

        # VLM comparison cell
        d_vlm = rv["delta"]
        sign_vlm = "+" if d_vlm >= 0 else ""
        color_vlm = "teal" if d_vlm >= 0 else "red"
        st_vlm = star(rv["p_holm"])
        vlm_delta = (rf"{rv['cmp_mean']:.3f} {{\scriptsize (\textcolor{{{color_vlm}}}"
                     rf"{{${sign_vlm}{d_vlm:.3f}^{{{st_vlm}}}$}})}}")
        p_vlm = rf"$p{{\!=\!}}{fmt_p(rv['p_holm'])}$"

        # SA-LLM comparison cell
        d_sa = rs["delta"]
        sign_sa = "+" if d_sa >= 0 else ""
        color_sa = "teal" if d_sa >= 0 else "red"
        st_sa = star(rs["p_holm"])
        sa_delta = (rf"{rs['cmp_mean']:.3f} {{\scriptsize (\textcolor{{{color_sa}}}"
                    rf"{{${sign_sa}{d_sa:.3f}^{{{st_sa}}}$}})}}")
        p_sa = rf"$p{{\!=\!}}{fmt_p(rs['p_holm'])}$"

        lines.append(rf"{name} & {dec_val} & {vlm_delta} & {p_vlm} & {sa_delta} & {p_sa} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    tex = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"Saved: {OUT}")
    print(tex)


if __name__ == "__main__":
    main()
