"""
Generate an inst_blind abstention-behavior LaTeX table.

Usage:
    python scripts/latex_table_generators/gen_abstention_inst_blind.py
    python scripts/latex_table_generators/gen_abstention_inst_blind.py --write
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from utils.abstention import classify

EXPORTS = ROOT / "analysis/session2/exports"
mr = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")

VARIANTS = ["C", "B", "A"]

SIZE_NUM = {
    'Qwen3-VL-2B': 2, 'Qwen3-VL-4B': 4, 'Qwen3-VL-8B': 8, 'Qwen3-VL-32B': 32,
    'InternVL-1B': 1, 'InternVL-2B': 2, 'InternVL-8B': 8,
    'LLaVA-1.5-7B': 7, 'LLaVA-Mistral': 7.01, 'LLaVA-Vicuna': 7.02, 'LLaVA-Vicuna-13B': 13,
    'Qwen3-VL-2B (LM)': 2, 'Qwen3-VL-4B (LM)': 4, 'Qwen3-VL-8B (LM)': 8, 'Qwen3-VL-32B (LM)': 32,
    'InternVL-1B (LM)': 1, 'InternVL-2B (LM)': 2, 'InternVL-8B (LM)': 8,
    'LLaVA-1.5 (LM)': 7, 'LLaVA-Vicuna (LM)': 7.02, 'LLaVA-Mistral (LM)': 7.01, 'LLaVA-Vicuna-13B (LM)': 13,
    'Qwen3-0.6B': 0.6, 'Qwen3-1.7B': 1.7, 'Qwen3-4B': 4, 'Qwen3-8B': 8, 'Qwen3-32B': 32,
    'Qwen2.5-7B-Instruct': 7, 'Mistral-7B': 7.01, 'Vicuna-7B': 7.02, 'Vicuna-13B': 13,
    'Phi-3.5-mini': 3.8,
    'Qwen3-0.6B (think)': 0.6, 'Qwen3-1.7B (think)': 1.7, 'Qwen3-4B (think)': 4,
    'Qwen3-8B (think)': 8, 'Qwen3-32B (think)': 32,
}

SIZE_LABEL = {
    'Qwen3-VL-2B': '2B', 'Qwen3-VL-4B': '4B', 'Qwen3-VL-8B': '8B', 'Qwen3-VL-32B': '32B',
    'InternVL-1B': '1B', 'InternVL-2B': '2B', 'InternVL-8B': '8B',
    'LLaVA-1.5-7B': '7B', 'LLaVA-Mistral': '7B', 'LLaVA-Vicuna': '7B', 'LLaVA-Vicuna-13B': '13B',
    'Qwen3-VL-2B (LM)': '2B', 'Qwen3-VL-4B (LM)': '4B', 'Qwen3-VL-8B (LM)': '8B', 'Qwen3-VL-32B (LM)': '32B',
    'InternVL-1B (LM)': '1B', 'InternVL-2B (LM)': '2B', 'InternVL-8B (LM)': '8B',
    'LLaVA-1.5 (LM)': '7B', 'LLaVA-Vicuna (LM)': '7B', 'LLaVA-Mistral (LM)': '7B', 'LLaVA-Vicuna-13B (LM)': '13B',
    'Qwen3-0.6B': '0.6B', 'Qwen3-1.7B': '1.7B', 'Qwen3-4B': '4B', 'Qwen3-8B': '8B', 'Qwen3-32B': '32B',
    'Qwen2.5-7B-Instruct': '7B', 'Mistral-7B': '7B', 'Vicuna-7B': '7B', 'Vicuna-13B': '13B',
    'Phi-3.5-mini': '3.8B',
    'Qwen3-0.6B (think)': '0.6B', 'Qwen3-1.7B (think)': '1.7B', 'Qwen3-4B (think)': '4B',
    'Qwen3-8B (think)': '8B', 'Qwen3-32B (think)': '32B',
}

DISPLAY_NAME = {
    'LLaVA-1.5-7B': 'LLaVA-1.5',
    'LLaVA-Mistral': 'LLaVA-1.6-Mistral',
    'LLaVA-Vicuna': 'LLaVA-1.6-Vicuna',
    'LLaVA-Vicuna-13B': 'LLaVA-1.6-Vicuna',
    'LLaVA-1.5 (LM)': 'LLaVA-1.5',
    'LLaVA-Vicuna (LM)': 'LLaVA-1.6-Vicuna',
    'LLaVA-Mistral (LM)': 'LLaVA-1.6-Mistral',
    'LLaVA-Vicuna-13B (LM)': 'LLaVA-1.6-Vicuna',
}


def family_key(model):
    base = DISPLAY_NAME.get(model, model).replace(" (LM)", "").replace(" (think)", "")
    if base.startswith('InternVL'):
        return 'InternVL'
    if base.startswith('LLaVA'):
        return 'LLaVA'
    if base.startswith('Mistral'):
        return 'Mistral'
    if base.startswith('Phi'):
        return 'Phi'
    if base.startswith('Qwen2.5'):
        return 'Qwen2.5'
    if base.startswith('Qwen3-VL'):
        return 'Qwen3-VL'
    if base.startswith('Qwen3'):
        return 'Qwen3'
    if base.startswith('Vicuna'):
        return 'Vicuna'
    return base


def sort_models(models):
    return sorted(models, key=lambda m: (family_key(m).lower(), SIZE_NUM[m], DISPLAY_NAME.get(m, m).lower()))


def row_name(model, group):
    name = DISPLAY_NAME.get(model, model)
    if group == 'VLM backbone decoder':
        name = name.replace(' (LM)', '')
    if group == 'standalone LLM (think)':
        name = name.replace(' (think)', '')
    name = re.sub(r'-(?:0\.6|1|1\.7|2|3\.8|4|7|8|13|32)B(?=-|$)', '', name)
    name = name.replace('--', '-').strip('- ')
    return name


def emit_rows_with_multirow(lines, rows):
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and rows[j]["name"] == rows[i]["name"]:
            j += 1
        span = j - i
        for k in range(i, j):
            name_cell = rf"\multirow{{{span}}}{{*}}{{{rows[i]['name']}}}" if k == i and span > 1 else (rows[i]["name"] if k == i else "")
            row = rows[k]
            lines.append(f"{name_cell} & {row['size']} & " + " & ".join(row["cols"]) + r" \\")
        i = j


def fmt(val):
    if np.isnan(val):
        return "--"
    return f"{val * 100:.1f}"


def rankings_for_rows(rows):
    arr = np.array(rows, dtype=float)
    rankings = []
    for col_idx in range(arr.shape[1]):
        valid = arr[:, col_idx][~np.isnan(arr[:, col_idx])]
        if len(valid) == 0:
            rankings.append((np.nan, np.nan))
            continue
        uniq = sorted(set(valid.tolist()), reverse=True)
        rankings.append((uniq[0], uniq[1] if len(uniq) > 1 else np.nan))
    return rankings


def format_cell(value, best, second):
    text = fmt(value)
    if text == "--" or np.isnan(value):
        return text
    if not np.isnan(best) and np.isclose(value, best, equal_nan=False):
        return rf"\textbf{{{text}}}"
    if not np.isnan(second) and np.isclose(value, second, equal_nan=False):
        return rf"\underline{{{text}}}"
    return text


def compute_rows(model):
    sub = mr[mr["model"] == model].copy()
    if sub.empty:
        return [np.nan] * 9
    sub["cls"] = sub["response"].fillna("").astype(str).apply(lambda x: classify(x, None))
    vals = []
    for variant in VARIANTS:
        vsub = sub[sub["variant"] == variant]
        if len(vsub) == 0:
            vals.extend([np.nan, np.nan, np.nan])
            continue
        vals.extend([
            (vsub["cls"] == "soft_abstained").mean(),
            (vsub["cls"] == "hard_abstained").mean(),
            (vsub["cls"] == "degenerate").mean(),
        ])
    return vals


vlm_models = sort_models([
    'Qwen3-VL-2B', 'Qwen3-VL-4B', 'Qwen3-VL-8B', 'Qwen3-VL-32B',
    'InternVL-1B', 'InternVL-2B', 'InternVL-8B',
    'LLaVA-1.5-7B', 'LLaVA-Vicuna', 'LLaVA-Mistral', 'LLaVA-Vicuna-13B',
])

dec_models = sort_models([
    'Qwen3-VL-2B (LM)', 'Qwen3-VL-4B (LM)', 'Qwen3-VL-8B (LM)', 'Qwen3-VL-32B (LM)',
    'InternVL-1B (LM)', 'InternVL-2B (LM)', 'InternVL-8B (LM)',
    'LLaVA-1.5 (LM)', 'LLaVA-Vicuna (LM)', 'LLaVA-Mistral (LM)', 'LLaVA-Vicuna-13B (LM)',
])

llm_models = sort_models([
    'Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-32B',
    'Qwen2.5-7B-Instruct', 'Mistral-7B', 'Vicuna-7B', 'Vicuna-13B',
    'Phi-3.5-mini',
])

think_models = sort_models([
    'Qwen3-0.6B (think)', 'Qwen3-1.7B (think)', 'Qwen3-4B (think)',
    'Qwen3-8B (think)', 'Qwen3-32B (think)',
])


def build_table():
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\scriptsize")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lc|ccc|ccc|ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Size} "
        r"& \multicolumn{3}{c|}{\textbf{Original}}"
        r"& \multicolumn{3}{c|}{\textbf{Weaker}}"
        r"& \multicolumn{3}{c}{\textbf{Pronominalized}} \\"
    )
    lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}")
    lines.append(
        r" & & \textbf{Soft} & \textbf{Hard} & \textbf{Deg.}"
        r"& \textbf{Soft} & \textbf{Hard} & \textbf{Deg.}"
        r"& \textbf{Soft} & \textbf{Hard} & \textbf{Deg.} \\"
    )
    lines.append(r"\midrule")

    def emit_section(title, models, group):
        lines.append(r"\multicolumn{11}{c}{\textit{" + title + r"}} \\")
        lines.append(r"\midrule")
        numeric_rows = [compute_rows(model) for model in models]
        ranks = rankings_for_rows(numeric_rows)
        rows = []
        for model, vals in zip(models, numeric_rows):
            cols = [format_cell(v, *ranks[idx]) for idx, v in enumerate(vals)]
            rows.append({"name": row_name(model, group), "size": SIZE_LABEL[model], "cols": cols})
        emit_rows_with_multirow(lines, rows)

    emit_section("Vision-Language Models", vlm_models, "VLM")
    lines.append(r"\midrule")
    emit_section("VLM Backbone Decoders", dec_models, "VLM backbone decoder")
    lines.append(r"\midrule")
    emit_section("Standalone LLMs", llm_models, "standalone LLM")
    lines.append(r"\midrule")
    emit_section("Standalone LLMs with Thinking", think_models, "standalone LLM (think)")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{\small Inst\_blind abstention behavior across evaluation variants (Original, Weaker, Pronominalized). "
        r"Soft = implicit abstention rate; Hard = explicit refusal rate; Deg. = degenerate/empty output rate. "
        r"Bold marks the highest value and underline marks the second-highest value within each model-group column.}"
    )
    lines.append(r"\label{tab:abstention_inst_blind}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write output to abstention_inst_blind.tex")
    args = parser.parse_args()
    table = build_table()
    if args.write:
        out = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables" / "abstention_inst_blind.tex"
        out.write_text(table + "\n")
        print(f"Wrote {out}")
    else:
        print(table)
