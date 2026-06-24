"""
Generate the comprehensive blind-response LaTeX table.

Usage:
    python gen_comprehensive_inst_blind.py           # prints to stdout
    python gen_comprehensive_inst_blind.py --write   # writes to comprehensive_inst_blind.tex
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]   # HPA/
sys.path.insert(0, str(ROOT / 'analysis'))

EXPORTS = ROOT / 'analysis/session2/exports'
pc = pd.read_parquet(EXPORTS / 'pair_cache_cleaned.parquet')
mr = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')
hr = pd.read_csv(EXPORTS / 'responses_human.csv')

VARIANTS = ['C', 'B', 'A']
hh = pc[pc['pair_type'] == 'HH']
hm = pc[pc['pair_type'] == 'HM']

# Soft abstention regex
ABSTAIN_RE = r'\b(?:blank|nothing|none|nowhere|unanswerable|unknown|n/a|not applicable|cannot|can.t tell|no image|not possible|i don.t know|not sure)\b'

# ── Model metadata ──

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

# ── Metrics ──

METRIC_KEYS = [
    ('Acc', None, 'acc'),
    ('Len', None, 'length'),
    ('SBERT', 'sbert_score', 'hm'),
    ('chrF', 'chrf_score', 'hm'),
    ('R-1', 'rouge1_score', 'hm'),
    ('Exact', 'exact_score', 'hm'),
]

VARIANT_LABELS = {
    'C': 'Original',
    'B': 'Weaker',
    'A': 'Pronominalized',
}


def get_vals(model, group):
    result = {}
    for label, col, src in METRIC_KEYS:
        vals = []
        for v in VARIANTS:
            if src == 'acc':
                if group == 'Human':
                    val = hr[hr['variant']==v].groupby('question_id')['accuracy'].mean().mean()
                else:
                    d = mr[(mr['model']==model) & (mr['variant']==v)]['accuracy']
                    val = d.mean() if len(d) > 0 else np.nan
            elif src == 'length':
                if group == 'Human':
                    resp = hr[hr['variant']==v]['response'].fillna('')
                    val = resp.str.split().apply(len).mean() / 100.0
                else:
                    resp = mr[(mr['model']==model) & (mr['variant']==v)]['response'].fillna('')
                    wc = resp.str.split().apply(len)
                    val = (wc.mean() if len(wc) > 0 else np.nan) / 100.0
            else:
                if group == 'Human':
                    d = hh[hh['variant']==v][col]
                else:
                    d = hm[(hm['subject_2']==model) & (hm['variant']==v)][col]
                val = d.mean() if len(d) > 0 else np.nan
            vals.append(val)
        result[label] = vals
    return result


def fmt(val):
    if np.isnan(val):
        return "--"
    return f"{val*100:.1f}"

def fmt_len(val):
    if np.isnan(val):
        return "--"
    raw = val * 100
    return f"{raw:.0f}" if raw >= 10 else f"{raw:.1f}"


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
    return sorted(
        models,
        key=lambda m: (family_key(m).lower(), SIZE_NUM[m], DISPLAY_NAME.get(m, m).lower()),
    )


def row_cells(vals):
    cells = []
    for variant_idx, _variant in enumerate(VARIANTS):
        for metric_label, _col, _src in METRIC_KEYS:
            metric_vals = vals[metric_label]
            cell = fmt_len(metric_vals[variant_idx]) if metric_label == 'Len' else fmt(metric_vals[variant_idx])
            cells.append(cell)
    return cells


def flat_numeric_cells(vals):
    cells = []
    for variant_idx, _variant in enumerate(VARIANTS):
        for metric_label, _col, _src in METRIC_KEYS:
            cells.append(vals[metric_label][variant_idx])
    return cells


def row_name(model, group):
    name = DISPLAY_NAME.get(model, model)
    if group == 'VLM backbone decoder':
        name = name.replace(' (LM)', '')
    if group == 'standalone LLM (think)':
        name = name.replace(' (think)', '')
    # Remove terminal scale markers because size already has its own column.
    name = re.sub(r'-(?:0\.6|1|1\.7|2|3\.8|4|7|8|13|32)B(?=-|$)', '', name)
    name = name.replace('--', '-').strip('- ')
    return name


def section_column_rankings(models, group):
    rows = [flat_numeric_cells(get_vals(model, group)) for model in models]
    rankings = []
    for col_idx in range(len(rows[0])):
        col = np.array([row[col_idx] for row in rows], dtype=float)
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            rankings.append((np.nan, np.nan))
            continue
        uniq = sorted(set(valid.tolist()), reverse=True)
        best = uniq[0]
        second = uniq[1] if len(uniq) > 1 else np.nan
        rankings.append((best, second))
    return rankings


def format_highlighted_cell(value, metric_label, best_value, second_value):
    text = fmt_len(value) if metric_label == 'Len' else fmt(value)
    if text == "--" or np.isnan(value):
        return text
    if not np.isnan(best_value) and np.isclose(value, best_value, equal_nan=False):
        return rf"\textbf{{{text}}}"
    if not np.isnan(second_value) and np.isclose(value, second_value, equal_nan=False):
        return rf"\underline{{{text}}}"
    return text


def highlighted_row_cells(vals, rankings):
    cells = []
    flat_idx = 0
    for variant_idx, _variant in enumerate(VARIANTS):
        for metric_label, _col, _src in METRIC_KEYS:
            value = vals[metric_label][variant_idx]
            best_value, second_value = rankings[flat_idx]
            cells.append(format_highlighted_cell(value, metric_label, best_value, second_value))
            flat_idx += 1
    return cells


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


# ── Model lists per section, sorted alphabetically by family then size ──

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


# ── Build table ──

def build_table():
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\scriptsize")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lc|cccccc|cccccc|cccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Size} "
        r"& \multicolumn{6}{c|}{\textbf{Original}}"
        r"& \multicolumn{6}{c|}{\textbf{Weaker}}"
        r"& \multicolumn{6}{c}{\textbf{Pronominalized}} \\"
    )
    lines.append(r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}\cmidrule(lr){15-20}")
    lines.append(
        r" & "
        r"& \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{chrF} & \textbf{ROUGE-1} & \textbf{Exact}"
        r"& \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{chrF} & \textbf{ROUGE-1} & \textbf{Exact}"
        r"& \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{chrF} & \textbf{ROUGE-1} & \textbf{Exact} \\"
    )
    lines.append(r"\midrule")

    # Human row
    hvals = get_vals('Human (N=40)', 'Human')
    cols = row_cells(hvals)
    lines.append(r"\textbf{Human (N=40)} & -- & " + " & ".join(cols) + r" \\")
    lines.append(r"\midrule")

    def emit_section(title, models, group):
        lines.append(r"\multicolumn{20}{c}{\textit{" + title + r"}} \\")
        lines.append(r"\midrule")
        rankings = section_column_rankings(models, group)
        rows = []
        for model in models:
            vals = get_vals(model, group)
            name = row_name(model, group)
            size = SIZE_LABEL.get(model, '?')
            cols = highlighted_row_cells(vals, rankings)
            rows.append({"name": name, "size": size, "cols": cols})
        emit_rows_with_multirow(lines, rows)

    emit_section("Vision-Language Models", vlm_models, 'VLM')
    lines.append(r"\midrule")
    emit_section("VLM Backbone Decoders", dec_models, 'VLM backbone decoder')
    lines.append(r"\midrule")
    emit_section("Standalone LLMs", llm_models, 'standalone LLM')
    lines.append(r"\midrule")
    emit_section("Standalone LLMs with Thinking", think_models, 'standalone LLM (think)')

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\caption{\small Blind performance across evaluation variants "
        r"(Original, Weaker, Pronominalized). "
        r"Acc = VQA accuracy (\%); Len = mean answer length in words; "
        r"SBERT, chrF, ROUGE-1, Exact = human--model (HM) agreement (\%). "
        r"Human row shows human--human (HH) agreement.}")
    lines.append(r"\label{tab:comprehensive_inst_blind}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write output to comprehensive_inst_blind.tex")
    args = parser.parse_args()

    table = build_table()

    if args.write:
        out = Path(__file__).resolve().parent / "comprehensive_inst_blind.tex"
        out.write_text(table + "\n")
        print(f"Wrote {out}")
    else:
        print(table)
