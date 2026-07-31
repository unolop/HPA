"""
Generate the comprehensive blind-response LaTeX table.

Usage:
    python scripts/latex_table_generators/gen_comprehensive_inst_blind.py                      # prints to stdout
    python scripts/latex_table_generators/gen_comprehensive_inst_blind.py --write              # writes to the LaTeX tables directory
    python scripts/latex_table_generators/gen_comprehensive_inst_blind.py --cleaned --max_words 5 --output_tag w5 --output_dir latex/AAAI2026/LaTeX/tables_5words --write
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # HPA/
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))
from helpers import load_pair_cache, load_cleaned_pair_cache

EXPORTS = ROOT / 'analysis/session2/exports'

# Parse early so we know whether to filter
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--filtered", action="store_true")
_pre.add_argument("--cleaned", action="store_true")
_pre.add_argument("--max_words", type=int, default=None)
_pre.add_argument("--output_tag", default=None)
_pre_args, _ = _pre.parse_known_args()

if _pre_args.filtered or _pre_args.cleaned:
    pc = load_cleaned_pair_cache(
        ROOT,
        include_yesno=True,
        max_words=_pre_args.max_words,
        output_tag=_pre_args.output_tag,
        verbose=False,
    )
else:
    pc = load_pair_cache(ROOT, include_yesno=True, verbose=False)
mr = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')
hr = pd.read_csv(EXPORTS / 'responses_human.csv')

if _pre_args.filtered:
    from utils.abstention import classify, is_abstained
    from utils.vqa import preprocess_answer
    mr = mr.copy()
    mr['_clean'] = mr['response'].fillna('').astype(str).apply(
        lambda x: preprocess_answer(x, strip_think=True))
    mr['_is_subst'] = mr['_clean'].apply(
        lambda x: not is_abstained(classify(x, None)))
    mr = mr[mr['_is_subst']].copy()
    mr.drop(columns=['_clean', '_is_subst'], inplace=True)

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

# Human LOOCV p5–p95 ranges (percent units) for within-human highlighting.
# Computed via leave-one-participant-out: per-participant mean score vs all others.
HUMAN_LOOCV_RANGES = {
    # (metric_label, variant) → (p5, p95)  [in % units]
    ('Acc',   'C'): (19.2, 33.7), ('Acc',   'B'): (17.1, 29.0), ('Acc',   'A'): (16.2, 28.9),
    ('SBERT', 'C'): (54.9, 63.8), ('SBERT', 'B'): (53.2, 62.4), ('SBERT', 'A'): (52.4, 61.2),
    ('chrF',  'C'): (23.1, 32.5), ('chrF',  'B'): (21.5, 32.8), ('chrF',  'A'): (19.8, 30.3),
    ('R-1',   'C'): (18.8, 30.6), ('R-1',   'B'): (18.1, 30.4), ('R-1',   'A'): (16.0, 27.8),
    ('Exact', 'C'): (17.5, 27.8), ('Exact', 'B'): (16.7, 28.7), ('Exact', 'A'): (15.8, 26.7),
}


def get_vals(model, group, variants):
    result = {}
    for label, col, src in METRIC_KEYS:
        vals = []
        for v in variants:
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
                    d = hh[hh['variant']==v]
                else:
                    d = hm[(hm['subject_2']==model) & (hm['variant']==v)]
                if len(d) > 0:
                    val = d.groupby('question_id')[col].mean().mean()
                else:
                    val = np.nan
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
    for variant_idx in range(len(vals['Acc'])):
        for metric_label, _col, _src in METRIC_KEYS:
            metric_vals = vals[metric_label]
            cell = fmt_len(metric_vals[variant_idx]) if metric_label == 'Len' else fmt(metric_vals[variant_idx])
            cells.append(cell)
    return cells


def flat_numeric_cells(vals):
    cells = []
    for variant_idx in range(len(vals['Acc'])):
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


def section_column_rankings(models, group, variants):
    rows = [flat_numeric_cells(get_vals(model, group, variants)) for model in models]
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


def _in_human_range(value, metric_label, variant):
    """True if value (in [0,1] scale) falls within the human LOOCV p5–p95 range."""
    if metric_label == 'Len' or np.isnan(value):
        return False
    key = (metric_label, variant)
    if key not in HUMAN_LOOCV_RANGES:
        return False
    p5, p95 = HUMAN_LOOCV_RANGES[key]
    return p5 <= value * 100 <= p95


def format_highlighted_cell(value, metric_label, best_value, variant=None):
    text = fmt_len(value) if metric_label == 'Len' else fmt(value)
    if text == "--" or np.isnan(value):
        return text
    in_range = _in_human_range(value, metric_label, variant)
    is_best = not np.isnan(best_value) and np.isclose(value, best_value, equal_nan=False)
    if is_best and in_range:
        return rf"\cellcolor{{gray!15}}\textbf{{{text}}}"
    if is_best:
        return rf"\textbf{{{text}}}"
    if in_range:
        return rf"\cellcolor{{gray!15}}{text}"
    return text


def highlighted_row_cells(vals, rankings, variants):
    cells = []
    flat_idx = 0
    for variant_idx, variant in enumerate(variants):
        for metric_label, _col, _src in METRIC_KEYS:
            value = vals[metric_label][variant_idx]
            best_value, _second = rankings[flat_idx]
            cells.append(format_highlighted_cell(value, metric_label, best_value, variant=variant))
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

def variant_block_spec(variants):
    if len(variants) == 1:
        title = VARIANT_LABELS[variants[0]]
        return (
            r"\begin{tabular}{lc|cccccc}",
            r"\textbf{Model} & \textbf{Size} "
            rf"& \multicolumn{{6}}{{c}}{{\textbf{{{title}}}}} \\",
            r"\cmidrule(lr){3-8}",
            r" & "
            r"& \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{chrF} & \textbf{ROUGE-1} & \textbf{Exact} \\",
        )
    return (
        r"\begin{tabular}{lc|cccccc|cccccc|cccccc}",
        r"\textbf{Model} & \textbf{Size} "
        r"& \multicolumn{6}{c|}{\textbf{Original}}"
        r"& \multicolumn{6}{c|}{\textbf{Weaker}}"
        r"& \multicolumn{6}{c}{\textbf{Pronominalized}} \\",
        r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}\cmidrule(lr){15-20}",
        r" & "
        r"& \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{chrF} & \textbf{ROUGE-1} & \textbf{Exact}"
        r"& \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{chrF} & \textbf{ROUGE-1} & \textbf{Exact}"
        r"& \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{chrF} & \textbf{ROUGE-1} & \textbf{Exact} \\",
    )


def build_table(variants=VARIANTS, label="tab:comprehensive_inst_blind"):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\scriptsize")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    tabular_spec, header1, cmidrule, header2 = variant_block_spec(variants)
    lines.append(tabular_spec)
    lines.append(r"\toprule")
    lines.append(header1)
    lines.append(cmidrule)
    lines.append(header2)
    lines.append(r"\midrule")

    # Human row
    hvals = get_vals('Human (N=40)', 'Human', variants)
    cols = row_cells(hvals)
    lines.append(r"\textbf{Human (N=40)} & -- & " + " & ".join(cols) + r" \\")
    lines.append(r"\midrule")

    def emit_section(title, models, group):
        ncols = 2 + len(variants) * len(METRIC_KEYS)
        lines.append(rf"\multicolumn{{{ncols}}}{{c}}{{\textit{{{title}}}}} \\")
        lines.append(r"\midrule")
        rankings = section_column_rankings(models, group, variants)
        rows = []
        for model in models:
            vals = get_vals(model, group, variants)
            name = row_name(model, group)
            size = SIZE_LABEL.get(model, '?')
            cols = highlighted_row_cells(vals, rankings, variants)
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
    filt_note = r" Abstaining responses are excluded." if _pre_args.filtered else ""
    clean_note = r" All answers are length-normalized ($\leq$5 words) before scoring." if (_pre_args.cleaned or _pre_args.max_words) else ""
    if len(variants) == 1:
        caption = (
            r"\caption{\small Blind+Inst performance on the original question variant only. "
            r"Acc = VQA accuracy (\%); Len = mean answer length in words; "
            r"SBERT, chrF, ROUGE-1, Exact = human--model (HM) agreement (\%). "
            r"Human row shows human--human (HH) agreement."
            + clean_note + filt_note + r"}"
        )
    else:
        caption = (
            r"\caption{\small Blind performance across evaluation variants "
            r"(Original, Weaker, Pronominalized). "
            r"Acc = VQA accuracy (\%); Len = mean answer length in words; "
            r"SBERT, chrF, ROUGE-1, Exact = human--model (HM) agreement (\%). "
            r"Human row shows human--human (HH) agreement."
            + clean_note + filt_note + r"}"
        )
    lines.append(caption)
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write output to comprehensive_inst_blind.tex")
    parser.add_argument("--variant", choices=VARIANTS,
                        help="Emit a single-variant table only.")
    parser.add_argument("--filtered", action="store_true",
                        help="Filter out abstentions before computing statistics.")
    parser.add_argument("--cleaned", action="store_true",
                        help="Use the cleaned agreement cache instead of the raw cache.")
    parser.add_argument("--max_words", type=int, default=None,
                        help="Optional max words used by the cleaned cache.")
    parser.add_argument("--output_tag", default=None,
                        help="Optional cleaned-cache tag (e.g. w5).")
    parser.add_argument("--output_dir", default=".",
                        help="Directory to write the .tex file into when --write is set.")
    args = parser.parse_args()

    filtered_tag = "_filtered" if args.filtered else ""
    cleaned_tag = f"_{args.output_tag}" if args.output_tag else ""
    variants = [args.variant] if args.variant else VARIANTS
    base_label = "tab:comprehensive_inst_blind_vc" if args.variant == "C" else "tab:comprehensive_inst_blind"
    label = f"{base_label}{cleaned_tag}{filtered_tag}"
    table = build_table(variants=variants, label=label)

    if args.write:
        if args.variant == "C":
            filename = f"comprehensive_inst_blind_vC{cleaned_tag}{filtered_tag}.tex"
        else:
            filename = f"comprehensive_inst_blind{cleaned_tag}{filtered_tag}.tex"
        out_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / filename
        out.write_text(table + "\n")
        print(f"Wrote {out}")
    else:
        print(table)
