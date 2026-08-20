"""
Generate degradation_wilcoxon.tex — transposed pronominalization degradation
table with all matched 7/8B models shown as rows.

Rows:
  - Human LOO
  - group mean rows (VLM / Backbone / SA-LLM) using the same averaged values
    as the previous group-level table
  - individual model rows within each group

Columns:
  - operation groups
  - entity groups

Significance stars are retained for the human and group-mean rows.
Individual model rows additionally report raw paired Wilcoxon stars.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figures.helpers import filter_abstained_pairs

OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/degradation_wilcoxon.tex"

MODEL_GROUPS = {
    "VLM": [
        "InternVL-8B",
        "Qwen3-VL-8B",
        "LLaVA-1.5-7B",
        "LLaVA-Mistral",
        "LLaVA-Vicuna",
    ],
    "Backbone": [
        "InternVL-8B (LM)",
        "Qwen3-VL-8B (LM)",
        "LLaVA-1.5 (LM)",
        "LLaVA-Mistral (LM)",
        "LLaVA-Vicuna (LM)",
    ],
    "SA-LLM": [
        "Qwen3-8B",
        "Qwen3-8B (think)",
        "Qwen2.5-7B-Instruct",
        "Vicuna-7B",
    ],
}

OP_GROUPS = [
    (["attr"], "Attr."),
    (["count"], "Count"),
    (["spat"], "Spat."),
    (["ident"], "Ident."),
    (["act"], "Act."),
    (["exist"], "Exist."),
    (["text", "know", "temp", "comp", "cause"], "Other"),
]

ENT_GROUPS = [
    (["person"], "Person"),
    (["animal"], "Animal"),
    (["object"], "Object"),
    (["food"], "Food"),
    (["other", "place", "product", "vehicle", "text"], "Other"),
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pc = pd.read_parquet(ROOT / "analysis/session2/exports/pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)
    hh = pc[pc["pair_type"] == "HH"].copy()
    all_models = {m for ms in MODEL_GROUPS.values() for m in ms}
    hm = pc[(pc["pair_type"] == "HM") & pc["subject_2"].isin(all_models)].copy()
    return hh, hm


def per_question_means(df: pd.DataFrame, dim_col: str, dim_val: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df[dim_col].isin(dim_val)]
    c = sub[sub["variant"] == "C"].groupby("question_id")["sbert_score"].mean()
    a = sub[sub["variant"] == "A"].groupby("question_id")["sbert_score"].mean()
    common = c.index.intersection(a.index)
    return c[common].values, a[common].values


def run_wilcoxon(c_vals: np.ndarray, a_vals: np.ndarray) -> tuple[float, float, int]:
    if len(c_vals) < 5:
        return float("nan"), float("nan"), len(c_vals)
    delta = float(c_vals.mean() - a_vals.mean())
    diffs = c_vals - a_vals
    nz = diffs[diffs != 0]
    if len(nz) < 5:
        return delta, float("nan"), len(c_vals)
    _, p = stats.wilcoxon(c_vals, a_vals, alternative="two-sided")
    return delta, float(p), len(c_vals)


def star(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return r"***"
    if p < 0.01:
        return r"**"
    if p < 0.05:
        return r"*"
    return ""


def holm_correct(pvals: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    m = len(arr)
    order = np.argsort(arr)
    sorted_p = arr[order]
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for i, p in enumerate(sorted_p):
        val = (m - i) * p
        running = max(running, val)
        adjusted[i] = min(running, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return out


def fmt_delta(delta: float) -> str:
    if np.isnan(delta):
        return "---"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def colorize(delta: float, text: str) -> str:
    if np.isnan(delta):
        return text
    if delta >= 0.10:
        return rf"\cellcolor{{orange!24}}{text}"
    if delta >= 0.02:
        return rf"\cellcolor{{orange!12}}{text}"
    if delta <= -0.10:
        return rf"\cellcolor{{blue!20}}{text}"
    if delta <= -0.02:
        return rf"\cellcolor{{blue!10}}{text}"
    return text


def fmt_delta_star(delta: float, p: float) -> str:
    base = fmt_delta(delta)
    s = star(p)
    text = rf"{base}$^{{{s}}}$" if s else base
    return colorize(delta, text)


def main() -> None:
    hh, hm = load_data()

    # all_groups: (dim_type, gid, gvals, label)
    # gid is a string key for DataFrame indexing; gvals is the list for data filtering
    all_groups = [("op", "|".join(vals), vals, label) for vals, label in OP_GROUPS] + \
                 [("ent", "|".join(vals), vals, label) for vals, label in ENT_GROUPS]

    # Collect human and group-average rows with p-values
    summary_records: list[dict] = []
    for dim_type, gid, gvals, _ in all_groups:
        dim_col = "op" if dim_type == "op" else "ent"

        c_h, a_h = per_question_means(hh, dim_col, gvals)
        d_h, p_h, _ = run_wilcoxon(c_h, a_h)
        summary_records.append({"row": "Human LOO", "group_type": dim_type, "gid": gid, "delta": d_h, "raw_p": p_h})

        for group_name, models in MODEL_GROUPS.items():
            sub = hm[hm["subject_2"].isin(models)]
            c_g, a_g = per_question_means(sub, dim_col, gvals)
            d_g, p_g, _ = run_wilcoxon(c_g, a_g)
            summary_records.append({"row": f"{group_name} Mean", "group_type": dim_type, "gid": gid, "delta": d_g, "raw_p": p_g})

    summary_df = pd.DataFrame(summary_records)
    summary_df["p_holm"] = np.nan

    # Human row: uncorrected
    human_mask = summary_df["row"] == "Human LOO"
    summary_df.loc[human_mask, "p_holm"] = summary_df.loc[human_mask, "raw_p"]

    # Group-mean rows: use raw p-values (same as human rows) for fair comparison
    non_human_mask = (summary_df["row"] != "Human LOO") & summary_df["raw_p"].notna()
    summary_df.loc[non_human_mask, "p_holm"] = summary_df.loc[non_human_mask, "raw_p"]

    # Collect individual model rows: deltas + raw per-model Wilcoxon p-values
    model_records: list[dict] = []
    for group_name, models in MODEL_GROUPS.items():
        for model in models:
            sub_model = hm[hm["subject_2"] == model]
            for dim_type, gid, gvals, _ in all_groups:
                dim_col = "op" if dim_type == "op" else "ent"
                c_m, a_m = per_question_means(sub_model, dim_col, gvals)
                d_m, p_m, _ = run_wilcoxon(c_m, a_m)
                model_records.append(
                    {
                        "row": model,
                        "group_name": group_name,
                        "group_type": dim_type,
                        "gid": gid,
                        "delta": d_m,
                        "raw_p": p_m,
                    }
                )

    model_df = pd.DataFrame(model_records)

    op_columns = [label for _, label in OP_GROUPS]
    ent_columns = [label for _, label in ENT_GROUPS]
    columns = op_columns + ent_columns
    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\caption{Pronominalization degradation by question group. "
        r"All rows (Human LOO and group means) report raw paired Wilcoxon $p$-values; individual model rows likewise report raw Wilcoxon stars. "
        r"Warm cells indicate positive degradation (alignment worsens); blue cells indicate negative values. "
        r"$^{***}p{<}0.001$, $^{**}p{<}0.01$, $^{*}p{<}0.05$.}",
        r"\label{tab:degradation_wilcoxon}",
        r"\begin{tabular}{l" + "c" * len(columns) + "}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Row}} & \multicolumn{" + str(len(op_columns)) + r"}{c}{\textbf{Operation groups}} & \multicolumn{" + str(len(ent_columns)) + r"}{c}{\textbf{Entity groups}} \\",
        r"\cmidrule(lr){2-" + str(1 + len(op_columns)) + r"}\cmidrule(lr){" + str(2 + len(op_columns)) + "-" + str(1 + len(op_columns) + len(ent_columns)) + r"}",
        r"& " + " & ".join(rf"\rotatebox{{60}}{{\textbf{{{c}}}}}" for c in columns) + r" \\",
        r"\midrule",
    ]

    # Human row
    human_cells = []
    for dim_type, gid, gvals, _ in all_groups:
        rec = summary_df[(summary_df["row"] == "Human LOO") & (summary_df["group_type"] == dim_type) & (summary_df["gid"] == gid)].iloc[0]
        human_cells.append(fmt_delta_star(rec["delta"], rec["p_holm"]))
    lines.append(r"\textbf{Human LOO} & " + " & ".join(human_cells) + r" \\")
    lines.append(r"\midrule")

    # Group sections
    for group_name, models in MODEL_GROUPS.items():
        mean_cells = []
        for dim_type, gid, gvals, _ in all_groups:
            rec = summary_df[
                (summary_df["row"] == f"{group_name} Mean")
                & (summary_df["group_type"] == dim_type)
                & (summary_df["gid"] == gid)
            ].iloc[0]
            mean_cells.append(fmt_delta_star(rec["delta"], rec["p_holm"]))
        lines.append(rf"\rowcolor{{gray!12}}\textit{{{group_name} Mean}} & " + " & ".join(mean_cells) + r" \\")
        for model in models:
            cells = []
            for dim_type, gid, gvals, _ in all_groups:
                rec = model_df[
                    (model_df["row"] == model)
                    & (model_df["group_type"] == dim_type)
                    & (model_df["gid"] == gid)
                ].iloc[0]
                cells.append(fmt_delta_star(rec["delta"], rec["raw_p"]))
            lines.append(rf"\quad {model} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
