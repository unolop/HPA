import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from notebooks.helpers import grouped_distribution_long_table

OUT_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables"

GROUP_ORDER = ["Human baseline", "Backbone Decoder", "VLM", "Standalone LLM"]
ANSWER_TYPE_ORDER = ["yes/no", "number", "text"]
ANSWER_TYPE_LABEL = {"yes/no": "Yes/No", "number": "Count", "text": "Text"}

SHORT = {
    "Qwen3-VL-8B": "Qwen3-VL",
    "LLaVA-1.5-7B": "LLaVA-1.5",
    "LLaVA-Mistral": "LLaVA-Mistral",
    "LLaVA-Vicuna": "LLaVA-Vicuna",
    "InternVL-8B": "InternVL",
    "Qwen3-VL-8B (LM)": "Qwen3-VL",
    "LLaVA-1.5 (LM)": "LLaVA-1.5",
    "LLaVA-Mistral (LM)": "LLaVA-Mistral",
    "LLaVA-Vicuna (LM)": "LLaVA-Vicuna",
    "InternVL-8B (LM)": "InternVL",
    "Qwen3-8B": "Qwen3",
    "Qwen2.5-7B-Instruct": "Qwen2.5",
    "Mistral-7B": "Mistral",
    "Vicuna-7B": "Vicuna",
}
SECTION_TITLES = {
    "Human baseline": "Human baseline",
    "Backbone Decoder": "Backbone decoders",
    "VLM": "VLMs",
    "Standalone LLM": "Standalone LLMs",
}


def fmt(v: float) -> str:
    return "--" if pd.isna(v) else f"{v:.3f}"


def build_table(dimension: str, filter_abstention: bool) -> pd.DataFrame:
    df = grouped_distribution_long_table(
        dimension,
        condition="inst_blind",
        variants=("C", "B", "A"),
        filter_abstention=filter_abstention,
        answer_types=tuple(ANSWER_TYPE_ORDER),
        include_human_baseline=True,
    ).copy()
    df["Group"] = pd.Categorical(df["Group"], categories=GROUP_ORDER, ordered=True)
    df["Answer type"] = pd.Categorical(df["Answer type"], categories=ANSWER_TYPE_ORDER, ordered=True)
    return df.sort_values(["Answer type", "Group", "Mean JS", "Mean TV", "Model"]).reset_index(drop=True)


def render_table(df: pd.DataFrame, caption: str, label: str) -> str:
    lines = []
    lines.append(r"\begin{table*}[p]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Group & \% sig slices & Mean Cramer's V & Mean JS & Mean TV & Model abstention \% \\")
    lines.append(r"\midrule")

    for answer_type in ANSWER_TYPE_ORDER:
        sub = df[df["Answer type"] == answer_type].copy()
        if sub.empty:
            continue
        lines.append(rf"\multicolumn{{7}}{{c}}{{\textbf{{{ANSWER_TYPE_LABEL[answer_type]}}}}} \\")
        lines.append(r"\midrule")
        for _, row in sub.iterrows():
            vals = [
                fmt(row["% sig slices"]),
                fmt(row["Mean Cramer's V"]),
                fmt(row["Mean JS"]),
                fmt(row["Mean TV"]),
                fmt(row["Model abstention %"]),
            ]
            lines.append(
                f"{row['Model']} & {row['Group']} & " + " & ".join(vals) + r" \\"
            )
        if answer_type != ANSWER_TYPE_ORDER[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def render_compact_js_table(df: pd.DataFrame, caption: str, label: str) -> str:
    mat = (
        df.pivot_table(index=["Model", "Group"], columns="Answer type", values="Mean JS", aggfunc="first")
        .reindex(columns=ANSWER_TYPE_ORDER)
        .reset_index()
    )
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{2.8pt}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Yes/No & Count & Text \\")
    lines.append(r"\midrule")
    for group in GROUP_ORDER:
        sub = mat[mat["Group"] == group].copy()
        if sub.empty:
            continue
        lines.append(rf"\multicolumn{{4}}{{c}}{{\textbf{{{SECTION_TITLES[group]}}}}} \\")
        lines.append(r"\midrule")
        for _, row in sub.iterrows():
            model = SHORT.get(row["Model"], row["Model"])
            vals = [fmt(row[a]) for a in ANSWER_TYPE_ORDER]
            vals = [v[1:] if v.startswith("0") else v for v in vals]
            lines.append(f"{model} & " + " & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_one(dimension: str, filter_abstention: bool) -> Path:
    df = build_table(dimension, filter_abstention)
    grouping = "operation-group" if dimension == "op" else "entity-group"
    shared = (
        "shared substantive-response version"
        if filter_abstention
        else "inclusive version"
    )
    caption = (
        f"Matched 7/8B {grouping} pooled-distribution alignment to humans across all control variants "
        f"({shared}). Rows are grouped by answer type, with the leave-one-out human baseline shown first in each block. "
        "Lower JS/TV indicate closer alignment to the corresponding human grouped distribution."
    )
    if filter_abstention:
        caption += " Abstentions are removed and human/model distributions are compared on the shared non-abstaining subset."
    else:
        caption += " Model abstention rate is reported for the full pooled response set."

    suffix = "_filtered" if filter_abstention else ""
    out = OUT_DIR / f"hm_grouped_distribution_7b_{dimension}_all_variants{suffix}.tex"
    label = f"tab:hm_grouped_distribution_7b_{dimension}" + ("_filtered" if filter_abstention else "")
    out.write_text(render_table(df, caption, label))
    return out


if __name__ == "__main__":
    for dimension in ("op", "ent"):
        for filter_abstention in (False, True):
            print(write_one(dimension, filter_abstention))
    compact_df = build_table("ent", True)
    compact_caption = (
        "Matched 7/8B entity-group pooled-distribution alignment to humans across all control variants "
        "(shared substantive-response version). Entries are Mean JS divergence (lower is better) by answer type."
    )
    compact_out = OUT_DIR / "hm_grouped_distribution_7b_ent_js_compact_filtered.tex"
    compact_out.write_text(
        render_compact_js_table(
            compact_df,
            compact_caption,
            "tab:hm_grouped_distribution_7b_ent_js_compact_filtered",
        )
    )
    print(compact_out)
