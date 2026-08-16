"""
Full-scale alignment metrics table (all models, all sizes).

Columns: Size | S_HM(Orig) | S_HM(Pron) | Delta

Coloring (per sHM column):
  green!20  = within human LOO 95% CI
  no color  = outside CI
  Bold      = closest to human LOO mean per column
  Underline = second-closest per column

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figures.helpers import filter_abstained_pairs  # noqa: E402

EXPORTS = ROOT / "analysis/session2/exports"
OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/scale_alignment_full_table.tex"

# ── Model layout ─────────────────────────────────────────────────────────────

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
        ("13B", "LLaVA-Vicuna-13B", None),
    ]),
]

SA_MODELS = [
    # (display, size, model_id, think_id or None)
    ("Qwen3",   "0.6B",  "Qwen3-0.6B",          None),
    ("Qwen3",   "1.7B",  "Qwen3-1.7B",           "Qwen3-1.7B (think)"),
    ("Qwen3",   "4B",    "Qwen3-4B",              None),
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

    hm_var = hm.groupby(["subject_2", "variant"])["sbert_score"].mean()
    hh_var = hh.groupby("variant")["sbert_score"].mean()

    return hh, hm_var, hh_var


def compute_hh_loo_ci(hh: pd.DataFrame, variant: str):
    """Leave-one-human-out CI (p5, p95) and mean for sHH in a given variant."""
    hh_v = hh[hh["variant"] == variant]
    humans = pd.concat([hh_v["subject_1"], hh_v["subject_2"]]).unique()
    loo_means = []
    for h in humans:
        sub = hh_v[(hh_v["subject_1"] != h) & (hh_v["subject_2"] != h)]
        if len(sub) > 0:
            loo_means.append(sub["sbert_score"].mean())
    loo = np.array(loo_means)
    return float(np.percentile(loo, 5)), float(np.percentile(loo, 95)), float(np.mean(loo))


# ── Formatting ────────────────────────────────────────────────────────────────

DASH = r"{\scriptsize ---}"


def fd(x: float | None) -> str:
    """Format delta with sign."""
    if x is None:
        return r"{\scriptsize ---}"
    if x >= 0:
        return f"+{x:.3f}"
    return f"$-${abs(x):.3f}"


def fmt_sbert(val: float | None, ci_lo: float, ci_hi: float,
              rank: int) -> str:
    """Format an sHM cell with green coloring and bold/underline for rank."""
    if val is None:
        return r"{\scriptsize ---}"
    color = r"\cellcolor{green!20}" if ci_lo <= val <= ci_hi else ""
    s = f"{val:.3f}"
    if rank == 1:
        s = rf"\textbf{{{s}}}"
    elif rank == 2:
        s = rf"\underline{{{s}}}"
    return f"{color}{s}"


def rank_by_distance(vals: list[tuple[str, float | None]], ref: float) -> dict[str, int]:
    """Rank model ids by distance to ref; closest = rank 1."""
    finite = [(mid, v) for mid, v in vals if v is not None]
    finite.sort(key=lambda x: abs(x[1] - ref))
    return {mid: i + 1 for i, (mid, _) in enumerate(finite)}


# ── Table builders ────────────────────────────────────────────────────────────

def build_table(hh, hm_var, hh_var) -> list[str]:
    h_c = hh_var.get("C")
    h_a = hh_var.get("A")
    h_d = (h_c - h_a) if (h_c is not None and h_a is not None) else None

    # Human LOO CIs
    ci_c_lo, ci_c_hi, ci_c_mean = compute_hh_loo_ci(hh, "C")
    ci_a_lo, ci_a_hi, ci_a_mean = compute_hh_loo_ci(hh, "A")

    def get_sbert(model):
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

    # Collect all model ids to compute per-column ranks
    all_vlm_ids = [vlm_id for _, sizes in VLM_FAMILIES
                   for _, vlm_id, _ in sizes]
    all_lm_ids  = [lm_id for _, sizes in VLM_FAMILIES
                   for _, _, lm_id in sizes if lm_id is not None]
    all_sa_ids  = [mid for _, _, mid, _ in SA_MODELS]
    all_think_ids = [tid for _, _, _, tid in SA_MODELS if tid is not None]
    all_ids = all_vlm_ids + all_lm_ids + all_sa_ids + all_think_ids

    orig_vals = [(mid, get_sbert(mid)[0]) for mid in all_ids]
    pron_vals = [(mid, get_sbert(mid)[1]) for mid in all_ids]
    rank_orig = rank_by_distance(orig_vals, ci_c_mean)
    rank_pron = rank_by_distance(pron_vals, ci_a_mean)

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
        r" Human LOO reference: $\sHH$(Orig)\,$=$\,"
        + f"{ci_c_mean:.3f}"
        + r"\ [" + f"{ci_c_lo:.3f}, {ci_c_hi:.3f}" + r"];"
        + r" $\sHH$(Pron)\,$=$\,"
        + f"{ci_a_mean:.3f}"
        + r"\ [" + f"{ci_a_lo:.3f}, {ci_a_hi:.3f}" + r"]."
        + r" \colorbox{green!20}{Green} = within human LOO 95\,\% CI."
        + r" \textbf{Bold} = closest to LOO mean; \underline{underline} = second-closest."
        + r" Backbone Decoder column: LM counterpart at same size ({\scriptsize ---} = not available)."
        + r" Think variants shown only where confirmed distinct from no-think.}",
        r"\label{tab:scale_alignment_full}",
        r"\begin{tabular}{llccc|ccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Family}} & \multirow{2}{*}{\textbf{Size}}"
        r" & \multicolumn{3}{c|}{\textbf{VLM}}"
        r" & \multicolumn{3}{c}{\textbf{Backbone Decoder}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & $\sHM$(Orig) & $\sHM$(Pron) & $\Delta$"
        r" & $\sHM$(Orig) & $\sHM$(Pron) & $\Delta$ \\",
        r"\midrule",
    ]

    for fam_name, sizes in VLM_FAMILIES:
        n = len(sizes)
        for i, (sz, vlm_id, lm_id) in enumerate(sizes):
            fam_cell = (r"\multirow{" + str(n) + r"}{*}{" + fam_name + "}") if i == 0 else ""

            vc, va, vd = get_sbert(vlm_id)
            lc, la, ld = get_sbert(lm_id) if lm_id else (None, None, None)

            lines.append(
                f"{fam_cell} & {sz}"
                f" & {fmt_sbert(vc, ci_c_lo, ci_c_hi, rank_orig.get(vlm_id, 99))}"
                f" & {fmt_sbert(va, ci_a_lo, ci_a_hi, rank_pron.get(vlm_id, 99))}"
                f" & {fd(vd)}"
                f" & {fmt_sbert(lc, ci_c_lo, ci_c_hi, rank_orig.get(lm_id, 99)) if lm_id else DASH}"
                f" & {fmt_sbert(la, ci_a_lo, ci_a_hi, rank_pron.get(lm_id, 99)) if lm_id else DASH}"
                f" & {fd(ld) if lm_id else DASH}"
                r" \\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}"]

    # ── SA-LLM panel ──
    lines += [
        r"\vspace{4pt}",
        r"\begin{tabular}{llccc|ccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Family}} & \multirow{2}{*}{\textbf{Size}}"
        r" & \multicolumn{3}{c|}{\textbf{SA-LLM (no think)}}"
        r" & \multicolumn{3}{c}{\textbf{SA-LLM (think)}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & $\sHM$(Orig) & $\sHM$(Pron) & $\Delta$"
        r" & $\sHM$(Orig) & $\sHM$(Pron) & $\Delta$ \\",
        r"\midrule",
    ]

    prev_fam = None
    for fam_name, sz, model_id, think_id in SA_MODELS:
        fam_cell = fam_name if fam_name != prev_fam else ""
        prev_fam = fam_name

        mc, ma, md = get_sbert(model_id)
        tc, ta, td = get_sbert(think_id) if think_id else (None, None, None)

        lines.append(
            f"{fam_cell} & {sz}"
            f" & {fmt_sbert(mc, ci_c_lo, ci_c_hi, rank_orig.get(model_id, 99))}"
            f" & {fmt_sbert(ma, ci_a_lo, ci_a_hi, rank_pron.get(model_id, 99))}"
            f" & {fd(md)}"
            f" & {fmt_sbert(tc, ci_c_lo, ci_c_hi, rank_orig.get(think_id, 99)) if think_id else DASH}"
            f" & {fmt_sbert(ta, ci_a_lo, ci_a_hi, rank_pron.get(think_id, 99)) if think_id else DASH}"
            f" & {fd(td) if think_id else DASH}"
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
    hh, hm_var, hh_var = load_data()
    lines = build_table(hh, hm_var, hh_var)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
