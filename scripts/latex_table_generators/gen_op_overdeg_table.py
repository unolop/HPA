"""Generate op_overdeg_per_model.tex — over-degradation gap per model per operation group."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "analysis" / "session2" / "exports" / "answer_pairs_sbert_text.csv"
OUT_PATH = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables" / "op_overdeg_per_model.tex"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_7B = {
    'InternVL-8B': 'VLM', 'Qwen3-VL-8B': 'VLM', 'LLaVA-1.5-7B': 'VLM',
    'LLaVA-Mistral': 'VLM', 'LLaVA-Vicuna': 'VLM',
    'InternVL-8B (LM)': 'Backbone', 'Qwen3-VL-8B (LM)': 'Backbone',
    'LLaVA-1.5 (LM)': 'Backbone', 'LLaVA-Mistral (LM)': 'Backbone',
    'LLaVA-Vicuna (LM)': 'Backbone',
    'Qwen3-8B': 'SA-LLM', 'Qwen3-8B (think)': 'SA-LLM',
    'Qwen2.5-7B-Instruct': 'SA-LLM', 'Vicuna-7B': 'SA-LLM',
}

OPS = ['cause', 'spat', 'count', 'know', 'ident', 'text', 'act']
OP_LABELS = {
    'cause': 'Caus.', 'spat': 'Spat.', 'count': 'Count',
    'know': 'Know.', 'ident': 'Ident.', 'text': 'Text', 'act': 'Act.',
}

MODEL_LABELS = {
    'InternVL-8B': 'InternVL', 'Qwen3-VL-8B': 'Qwen3-VL', 'LLaVA-1.5-7B': 'LLaVA-1.5',
    'LLaVA-Mistral': 'LLaVA-Mist.', 'LLaVA-Vicuna': 'LLaVA-Vic.',
    'InternVL-8B (LM)': 'InternVL', 'Qwen3-VL-8B (LM)': 'Qwen3-VL',
    'LLaVA-1.5 (LM)': 'LLaVA-1.5',
    'LLaVA-Mistral (LM)': 'LLaVA-Mist.', 'LLaVA-Vicuna (LM)': 'LLaVA-Vic.',
    'Qwen3-8B': 'Qwen3', 'Qwen3-8B (think)': 'Qwen3 (think)',
    'Qwen2.5-7B-Instruct': 'Qwen2.5', 'Vicuna-7B': 'Vicuna',
}

GROUP_ORDER = ['VLM', 'Backbone', 'SA-LLM']
GROUP_TITLES = {'VLM': 'VLMs', 'Backbone': 'Backbone Decoders', 'SA-LLM': 'Standalone LLMs'}

# Model order within each group (preserves MODELS_7B insertion order)
GROUP_MODELS: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
for m, g in MODELS_7B.items():
    GROUP_MODELS[g].append(m)

# ---------------------------------------------------------------------------
# Load & compute
# ---------------------------------------------------------------------------

print("Loading data …")
df = pd.read_csv(DATA_PATH)

# Keep only variants A and C (delta = A - C)
df = df[df['variant'].isin(['A', 'C'])].copy()

# --- HH delta per op ---
hh = df[df['pair_type'] == 'HH']
hh_pivot = (
    hh.groupby(['op', 'variant'])['sbert_score']
    .mean()
    .unstack('variant')
)
# delta = mean(A) - mean(C)
hh_delta: dict[str, float] = {}
for op in OPS:
    if op in hh_pivot.index and 'A' in hh_pivot.columns and 'C' in hh_pivot.columns:
        hh_delta[op] = hh_pivot.loc[op, 'A'] - hh_pivot.loc[op, 'C']
    else:
        hh_delta[op] = float('nan')

print("HH delta per op:")
for op, v in hh_delta.items():
    print(f"  {op}: {v:+.4f}")

# --- HM delta per (model, op) ---
hm = df[(df['pair_type'] == 'HM') & (df['subject_2'].isin(MODELS_7B))].copy()
hm_pivot = (
    hm.groupby(['subject_2', 'op', 'variant'])['sbert_score']
    .mean()
    .unstack('variant')
)

# Build overdeg matrix: rows=models, cols=ops
overdeg: dict[str, dict[str, float]] = {}
for model in MODELS_7B:
    overdeg[model] = {}
    for op in OPS:
        try:
            row = hm_pivot.loc[(model, op)]
            if 'A' in row.index and 'C' in row.index and not (np.isnan(row['A']) or np.isnan(row['C'])):
                hm_d = row['A'] - row['C']
                overdeg[model][op] = hm_d - hh_delta.get(op, float('nan'))
            else:
                overdeg[model][op] = float('nan')
        except KeyError:
            overdeg[model][op] = float('nan')

# ---------------------------------------------------------------------------
# Cell formatting helpers
# ---------------------------------------------------------------------------

def cell_color(val: float) -> str:
    """Return LaTeX cellcolor command string (without surrounding braces)."""
    if np.isnan(val) or abs(val) < 0.02:
        return r'\cellcolor{gray!10}'
    intensity = min(40, max(10, int(abs(val) / 0.3 * 40)))
    if val > 0:
        return rf'\cellcolor{{blue!{intensity}}}'
    else:
        return rf'\cellcolor{{red!{intensity}}}'


def fmt_val(val: float) -> str:
    if np.isnan(val):
        return r'---'
    sign = '+' if val >= 0 else ''
    return f'{sign}{val:.2f}'


def cell(val: float) -> str:
    color = cell_color(val)
    text = fmt_val(val)
    return f'{color}{text}'


# ---------------------------------------------------------------------------
# Build LaTeX
# ---------------------------------------------------------------------------

col_spec = 'l' + 'r' * len(OPS)
op_header = ' & '.join(r'\textbf{' + OP_LABELS[op] + '}' for op in OPS)

lines: list[str] = []
lines.append(r'\begin{table*}[t]')
lines.append(r'\centering')
lines.append(r'\small')
lines.append(r'\setlength{\tabcolsep}{3.5pt}')
lines.append(r'\caption{Over-degradation gap (variant~A $-$ variant~C, relative to human'
             r' $\Delta_{\text{HH}}$) per model and operation group.'
             r' \colorbox{blue!20}{Blue}: model--human gap narrows (models converge faster'
             r' than humans scatter).'
             r' \colorbox{red!20}{Red}: gap widens (models degrade more than humans).'
             r' \colorbox{gray!10}{Gray}: near zero ($<0.02$).'
             r' Darker shading = larger magnitude.}')
lines.append(r'\label{tab:op_overdeg_per_model}')
lines.append(rf'\begin{{tabular}}{{{col_spec}}}')
lines.append(r'\toprule')
lines.append(r'\textbf{Model} & ' + op_header + r' \\')
lines.append(r'\midrule')

for g in GROUP_ORDER:
    title = GROUP_TITLES[g]
    n_cols = 1 + len(OPS)
    lines.append(rf'\multicolumn{{{n_cols}}}{{l}}{{\textit{{{title}}}}} \\')
    for model in GROUP_MODELS[g]:
        label = MODEL_LABELS[model]
        cells = ' & '.join(cell(overdeg[model][op]) for op in OPS)
        lines.append(rf'{label} & {cells} \\')
    lines.append(r'\midrule')

# Remove last \midrule, replace with \bottomrule
lines[-1] = r'\bottomrule'

lines.append(r'\end{tabular}')
lines.append(r'\end{table*}')

tex = '\n'.join(lines) + '\n'

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(tex, encoding='utf-8')
print(f"\nWritten to {OUT_PATH}")
print("\n--- Generated LaTeX ---")
print(tex)
