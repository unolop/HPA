"""
Generate degradation_wilcoxon.tex — group-level pronominalization degradation
(variant C vs. variant A) with Holm-corrected Wilcoxon signed-rank p-values.

Unit of analysis: per-question mean SBERT (variant C minus variant A).
  - Human: mean HH SBERT per question per variant, then Wilcoxon across questions.
  - Model groups: mean HM SBERT per question across all models in the group,
    then Wilcoxon across questions.

Holm correction applied across all group × subject tests simultaneously.
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
OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/degradation_wilcoxon.tex"

MODEL_GROUPS = {
    "VLM":       ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B",
                  "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone":  ["InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)",
                  "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"],
    "SA-LLM":    ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B"],
}

OP_GROUPS = {
    "attr":  "Attribute", "count": "Counting",  "spat": "Spatial",
    "ident": "Identity",  "act":   "Action",     "exist": "Existence",
    "other": "Other",
}
ENT_GROUPS = {
    "person": "Person", "animal": "Animal", "object": "Object",
    "food":   "Food",   "other":  "Other",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pc = pd.read_parquet(ROOT / "analysis/session2/exports/pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)
    hh = pc[pc["pair_type"] == "HH"].copy()
    all_models = {m for ms in MODEL_GROUPS.values() for m in ms}
    hm = pc[(pc["pair_type"] == "HM") & pc["subject_2"].isin(all_models)].copy()
    return hh, hm


def per_question_means(df: pd.DataFrame, dim_col: str, dim_val: str) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned (C_scores, A_scores) arrays, one entry per question_id."""
    sub = df[df[dim_col] == dim_val]
    c = sub[sub["variant"] == "C"].groupby("question_id")["sbert_score"].mean()
    a = sub[sub["variant"] == "A"].groupby("question_id")["sbert_score"].mean()
    common = c.index.intersection(a.index)
    return c[common].values, a[common].values


def run_wilcoxon(c_vals: np.ndarray, a_vals: np.ndarray) -> tuple[float, float, int]:
    """Return (delta, raw_p, n_questions)."""
    if len(c_vals) < 5:
        return float("nan"), float("nan"), len(c_vals)
    delta = float(c_vals.mean() - a_vals.mean())
    diffs = c_vals - a_vals
    nz = diffs[diffs != 0]
    if len(nz) < 5:
        return delta, float("nan"), len(c_vals)
    _, p = stats.wilcoxon(c_vals, a_vals, alternative="two-sided")
    return delta, float(p), len(c_vals)


def star(p_holm: float) -> str:
    if np.isnan(p_holm):    return ""
    if p_holm < 0.001:      return "***"
    if p_holm < 0.01:       return "**"
    if p_holm < 0.05:       return "*"
    return ""


def fmt_delta(delta: float, p_holm: float, human_delta: float) -> str:
    """Format delta cell with significance star and bold if matches human significance."""
    if np.isnan(delta):
        return "---"
    s = star(p_holm)
    sign = "+" if delta >= 0 else ""
    num = f"{sign}{delta:.3f}"
    cell = rf"{num}$^{{{s}}}$" if s else num
    # Bold if human is also significant (≥ * level, i.e. p < .05)
    if not np.isnan(human_delta) and not np.isnan(p_holm):
        return rf"\textbf{{{cell}}}" if s else cell
    return cell


def fmt_human(delta: float, p_holm: float) -> str:
    if np.isnan(delta):
        return "---"
    s = star(p_holm)
    sign = "+" if delta >= 0 else ""
    num = f"{sign}{delta:.3f}"
    return rf"{num}$^{{{s}}}$" if s else num


def main():
    hh, hm = load_data()

    # Collect all tests: (dim_col, dim_val, subject_key, c_vals, a_vals)
    records = []

    for dim_col, groups, section in [("op", OP_GROUPS, "op"), ("ent", ENT_GROUPS, "ent")]:
        for gkey in groups:
            # Human (HH)
            c, a = per_question_means(hh, dim_col, gkey)
            delta, raw_p, n = run_wilcoxon(c, a)
            records.append(dict(section=section, gkey=gkey, subject="Human",
                                delta=delta, raw_p=raw_p, n=n))
            # Model groups (HM — average across models in group first)
            for grp_name, models in MODEL_GROUPS.items():
                sub = hm[hm["subject_2"].isin(models)]
                c2, a2 = per_question_means(sub, dim_col, gkey)
                delta2, raw_p2, n2 = run_wilcoxon(c2, a2)
                records.append(dict(section=section, gkey=gkey, subject=grp_name,
                                    delta=delta2, raw_p=raw_p2, n=n2))

    df = pd.DataFrame(records)

    # Human: no correction (single test per group, standalone question)
    # Model groups: Holm within each row across 3 model groups (VLM, Backbone, SA-LLM)
    # Family = the 3 model group comparisons for a given question group
    df["p_holm"] = float("nan")
    # Human column: p_holm = raw_p (uncorrected)
    mask_h = (df["subject"] == "Human") & df["raw_p"].notna()
    df.loc[mask_h, "p_holm"] = df.loc[mask_h, "raw_p"]
    # Model groups: Holm per row
    for (section, gkey), _ in df[df["subject"] != "Human"].groupby(["section", "gkey"]):
        mask = ((df["section"] == section) & (df["gkey"] == gkey)
                & (df["subject"] != "Human") & df["raw_p"].notna())
        if mask.sum() > 0:
            _, p_holm_arr, _, _ = multipletests(df.loc[mask, "raw_p"].values, method="holm")
            df.loc[mask, "p_holm"] = p_holm_arr

    # Build table
    SUBJECTS = ["Human", "VLM", "Backbone", "SA-LLM"]
    col_headers = r"& \textbf{Human} & \textbf{VLM} & \textbf{Backbone} & \textbf{SA-LLM} \\"
    n_cols = 5  # group name + 4 subjects

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\caption{Per-group pronominalization degradation "
        r"($\Delta\sHM = \sHM[\vOrig] - \sHM[\vPron]$) "
        r"for humans (HH) and each model group (HM). "
        r"Two-sided Wilcoxon signed-rank; Holm correction applied within each row "
        r"across the 3 model groups (3 tests per question group); human column uncorrected. "
        r"$^{***}p{<}0.001$, $^{**}p{<}0.01$, $^{*}p{<}0.05$. "
        r"\textbf{Bold}: model significance matches human.}",
        r"\label{tab:degradation_wilcoxon}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Group} & \textbf{Human} & \textbf{VLM} & \textbf{Backbone} & \textbf{SA-LLM} \\",
        r"& $\Delta\sHH$ & $\Delta\sHM$ & $\Delta\sHM$ & $\Delta\sHM$ \\",
        r"\midrule",
    ]

    for section, groups, sec_label in [
        ("op",  OP_GROUPS,  r"\textit{Operation groups}"),
        ("ent", ENT_GROUPS, r"\textit{Entity groups}"),
    ]:
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{{sec_label}}} \\")
        lines.append(r"\midrule")
        for gkey, glabel in groups.items():
            row_data = {
                r["subject"]: r
                for _, r in df[(df["section"] == section) & (df["gkey"] == gkey)].iterrows()
            }
            human_r = row_data.get("Human", {})
            h_delta = human_r.get("delta", float("nan"))
            h_holm  = human_r.get("p_holm", float("nan"))
            h_sig   = not np.isnan(h_holm) and h_holm < 0.05

            cells = [fmt_human(h_delta, h_holm)]
            for grp in ["VLM", "Backbone", "SA-LLM"]:
                r = row_data.get(grp, {})
                d  = r.get("delta", float("nan"))
                ph = r.get("p_holm", float("nan"))
                s  = star(ph)
                sign = "+" if (not np.isnan(d) and d >= 0) else ""
                num = f"{sign}{d:.3f}" if not np.isnan(d) else "---"
                cell = rf"{num}$^{{{s}}}$" if s else num
                # Bold if model significant AND human significant
                if s and h_sig:
                    cell = rf"\textbf{{{cell}}}"
                cells.append(cell)

            lines.append(rf"{glabel} & {' & '.join(cells)} \\")
        lines.append(r"\midrule")

    # Remove trailing \midrule before \bottomrule
    if lines[-1] == r"\midrule":
        lines.pop()

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"Saved: {OUT}")
    print(tex)


if __name__ == "__main__":
    main()
