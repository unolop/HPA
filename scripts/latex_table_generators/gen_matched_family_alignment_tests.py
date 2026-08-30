r"""
Matched-family alignment table — models compared against Backbone Decoder.

Reference: Backbone Decoder. Subjects: Full VLM and SA-LLM.
  Col 1 (Full VLM) : Delta = Full VLM - Decoder
  Col 2 (SA-LLM)   : Delta = SA-LLM  - Decoder

Stars: raw paired Wilcoxon (each family is a pre-specified independent comparison).
Pooled row: Wilcoxon signed-rank on all per-(q,v) deltas across families, Holm-corrected.

Generates two versions:
  matched_family_alignment_tests_113q.tex  (all questions)
  matched_family_alignment_tests_61q.tex   (free-text only)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _st

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = ROOT.parent / "human-prior-alignment"
sys.path.insert(0, str(ANALYSIS_ROOT / "analysis"))

from utils.load_data import load_hh_hm, DATA_DIR

OUT_DIR = ROOT / "latex/AAAI2026/LaTeX/tables/supp"

FAMILIES = [
    ("InternVL3.5",   "InternVL-8B (LM)",   "InternVL-8B",   "Qwen3-8B"),
    ("Qwen3-VL",      "Qwen3-VL-8B (LM)",   "Qwen3-VL-8B",   "Qwen3-8B"),
    ("LLaVA-1.5",     "LLaVA-1.5 (LM)",     "LLaVA-1.5-7B",  "Vicuna-7B"),
    ("LLaVA-Mistral", "LLaVA-Mistral (LM)",  "LLaVA-Mistral", "Mistral-7B"),
    ("LLaVA-Vicuna",  "LLaVA-Vicuna (LM)",   "LLaVA-Vicuna",  "Vicuna-7B"),
]


def per_qv_mean(df: pd.DataFrame, model: str) -> pd.Series:
    return (
        df[df["subject_2"] == model]
        .groupby(["question_id", "variant"])["sbert_score"]
        .mean()
    )


def wilcoxon_pair(s_subj: pd.Series, s_ref: pd.Series):
    """Δ = subject − reference. Returns (delta, raw_p, n)."""
    common = s_subj.index.intersection(s_ref.index)
    v_subj, v_ref = s_subj[common].values, s_ref[common].values
    delta = float(v_subj.mean() - v_ref.mean())
    nz = v_subj - v_ref
    nz = nz[nz != 0]
    if len(nz) < 5:
        return delta, float("nan"), len(common)
    _, p = _st.wilcoxon(v_subj, v_ref, alternative="two-sided")
    return delta, float(p), len(common)


def stars(p: float) -> str:
    if p != p:      return ""
    if p < 0.001:   return "^{***}"
    if p < 0.01:    return "^{**}"
    if p < 0.05:    return "^{*}"
    return ""


def holm_bonferroni(pvals: list[float]) -> list[float]:
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    corrected = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        cummax = max(cummax, adj)
        corrected[orig_idx] = min(cummax, 1.0)
    return corrected


def fmt_delta(val: float, sup: str = "") -> str:
    sign = "+" if val >= 0 else "-"
    s = sign + "." + f"{abs(val):.3f}".split(".")[1]
    return f"${s}{sup}$"


def generate_table(hm: pd.DataFrame, nq: int, label_suffix: str, caption_scope: str):
    """Generate one version of the table."""
    all_models = {m for _, dec, vlm, sa in FAMILIES for m in (dec, vlm, sa)}
    qv = {m: per_qv_mean(hm, m) for m in all_models}

    rows = []
    for name, dec, vlm, sa in FAMILIES:
        d_vlm, p_vlm, n_vlm = wilcoxon_pair(qv[vlm], qv[dec])
        d_sa,  p_sa,  n_sa  = wilcoxon_pair(qv[sa],  qv[dec])
        rows.append(dict(Family=name,
                         d_vlm=d_vlm, p_vlm=p_vlm, n_vlm=n_vlm,
                         d_sa=d_sa,   p_sa=p_sa,   n_sa=n_sa))

    res = pd.DataFrame(rows).set_index("Family")
    print(res[["d_vlm", "p_vlm", "n_vlm", "d_sa", "p_sa", "n_sa"]].round(4).to_string())

    # Pooled Wilcoxon across all families
    vlm_deltas, sa_deltas = [], []
    for name, dec, vlm, sa in FAMILIES:
        common_v = qv[vlm].index.intersection(qv[dec].index)
        vlm_deltas.extend((qv[vlm][common_v] - qv[dec][common_v]).values)
        common_s = qv[sa].index.intersection(qv[dec].index)
        sa_deltas.extend((qv[sa][common_s] - qv[dec][common_s]).values)
    vlm_deltas = np.array(vlm_deltas)
    sa_deltas = np.array(sa_deltas)

    nz_vlm = vlm_deltas[vlm_deltas != 0]
    nz_sa = sa_deltas[sa_deltas != 0]
    _, p_pool_vlm = _st.wilcoxon(nz_vlm, alternative="two-sided")
    _, p_pool_sa = _st.wilcoxon(nz_sa, alternative="two-sided")
    avg_d_vlm = float(vlm_deltas.mean())
    avg_d_sa = float(sa_deltas.mean())
    n_pool_vlm = len(vlm_deltas)
    n_pool_sa = len(sa_deltas)

    p_vlm_adj, p_sa_adj = holm_bonferroni([p_pool_vlm, p_pool_sa])

    print(f"\nPooled Wilcoxon — VLM: Δ={avg_d_vlm:+.4f} N={n_pool_vlm} p={p_pool_vlm:.6f} Holm={p_vlm_adj:.6f}")
    print(f"Pooled Wilcoxon — SA:  Δ={avg_d_sa:+.4f} N={n_pool_sa} p={p_pool_sa:.6f} Holm={p_sa_adj:.6f}")

    tex_rows = []
    for name, *_ in FAMILIES:
        r = res.loc[name]
        tex_rows.append(
            f"{name} & {fmt_delta(r['d_vlm'], stars(r['p_vlm']))} & {int(r['n_vlm'])}"
            f" & {fmt_delta(r['d_sa'], stars(r['p_sa']))} & {int(r['n_sa'])} \\\\"
        )

    tex = (
        r"\begin{table}[ht]" "\n"
        r"\centering\small\setlength{\tabcolsep}{5pt}" "\n"
        r"\begin{tabular}{l cc cc}" "\n"
        r"\toprule" "\n"
        r"& \multicolumn{2}{c}{\textbf{Full VLM}} & \multicolumn{2}{c}{\textbf{SA-LLM}} \\" "\n"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
        r"\textbf{Family} & $\Delta$ & $N$ & $\Delta$ & $N$ \\" "\n"
        r"\midrule" "\n"
        + "\n".join(tex_rows) + "\n"
        r"\midrule" "\n"
        f"\\textit{{Pooled}} & {fmt_delta(avg_d_vlm, stars(p_vlm_adj))} & {n_pool_vlm} & {fmt_delta(avg_d_sa, stars(p_sa_adj))} & {n_pool_sa} \\\\\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        f"\\caption{{Matched-family alignment relative to the backbone decoder ({caption_scope}; $N$ = matched question-variant pairs).\n"
        r"$\Delta$ = model in comparison $-$ corresponding backbone decoder (LM) on per-question-variant $\sHM$ on average; positive values indicate the model leads." "\n"
        r"Stars on per-family rows: paired Wilcoxon (raw $p$). \textit{Pooled} row: Wilcoxon signed-rank on all per-question-variant deltas pooled across families, Holm--Bonferroni corrected across the two tests. ${}^{***}p{<}.001$, ${}^{**}p{<}.01$, ${}^{*}p{<}.05$; unlabelled\,=\,not significant.}" "\n"
        f"\\label{{tab:matched_family_alignment_tests{label_suffix}}}\n"
        r"\end{table}"
    )

    out = OUT_DIR / f"matched_family_alignment_tests_{nq}q{label_suffix}.tex"
    out.write_text(tex)
    print(f"\nWritten to {out}\n")
    return tex


def main():
    _at = pd.read_csv(DATA_DIR / "vqa_answer_types.csv")
    qid2atype = dict(zip(_at["question_id"].astype(int), _at["answer_type"]))
    other_qids = {qid for qid, at in qid2atype.items() if at == "other"}

    for filt in [False, True]:
        filt_tag = "absf" if filt else "raw"
        filt_label = "abstention-filtered" if filt else "unfiltered"
        _, hm = load_hh_hm(filtered=filt)

        hm_ft = hm[hm["question_id"].isin(other_qids)]
        hm_c = hm[hm["variant"] == "C"]
        hm_ft_c = hm_ft[hm_ft["variant"] == "C"]

        configs = [
            (hm,      339, f"_{filt_tag}", f"all 113 questions $\\times$ 3 variants, {filt_label}"),
            (hm_ft,   183, f"_ft_{filt_tag}", f"61 free-text questions $\\times$ 3 variants, {filt_label}"),
            (hm_c,    113, f"_orig_{filt_tag}", f"all 113 questions, original variant only, {filt_label}"),
            (hm_ft_c,  61, f"_ft_orig_{filt_tag}", f"61 free-text questions, original variant only, {filt_label}"),
        ]

        for data, nqv, suffix, caption in configs:
            print("=" * 60)
            print(f"{caption.upper()} ({nqv} qv pairs)")
            print("=" * 60)
            generate_table(data, nqv, suffix, caption)


if __name__ == "__main__":
    main()
