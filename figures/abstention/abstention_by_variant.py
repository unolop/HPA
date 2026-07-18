from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from analysis.utils.constants import VARIANT_COLORS
from analysis.utils.abstention import classify
from analysis.utils.load_session import clean_answer
from analysis.utils.model_registry import MODEL_TYPE, default_all_models
from helpers import read_response_exports

OUT_DIR = Path(__file__).parent
LATEX_OUT = ROOT / 'latex/AAAI2026/LaTeX/figures'
LATEX_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

GROUP_ORDER = ['VLM', 'VLM backbone decoder', 'standalone LLM', 'standalone LLM (think)']
GROUP_COLORS = {
    'VLM': '#9E4A3A',
    'VLM backbone decoder': '#C0843D',
    'standalone LLM': '#5E8C61',
    'standalone LLM (think)': '#5B84B8',
}
FAMILY_MARKER = {
    'InternVL': 'o', 'Qwen3-VL': 'X', 'LLaVA-1.5': 'D', 'LLaVA-Mistral': '^',
    'LLaVA-Vicuna': 'v', 'Qwen3': 'X', 'Qwen3 (think)': 'P', 'Qwen2.5': 's',
    'Mistral': '^', 'Vicuna': 'v', 'Phi': 'h',
}
MODEL_SIZE_B = {
    'InternVL-1B': 1.0, 'InternVL-2B': 2.0, 'InternVL-8B': 8.0,
    'InternVL-1B (LM)': 1.0, 'InternVL-2B (LM)': 2.0, 'InternVL-8B (LM)': 8.0,
    'Qwen3-VL-2B': 2.0, 'Qwen3-VL-4B': 4.0, 'Qwen3-VL-8B': 8.0, 'Qwen3-VL-32B': 32.0,
    'Qwen3-VL-2B (LM)': 2.0, 'Qwen3-VL-4B (LM)': 4.0, 'Qwen3-VL-8B (LM)': 8.0, 'Qwen3-VL-32B (LM)': 32.0,
    'LLaVA-1.5-7B': 7.0, 'LLaVA-Mistral': 7.0, 'LLaVA-Vicuna': 7.0, 'LLaVA-Vicuna-13B': 13.0,
    'LLaVA-1.5 (LM)': 7.0, 'LLaVA-Mistral (LM)': 7.0, 'LLaVA-Vicuna (LM)': 7.0, 'LLaVA-Vicuna-13B (LM)': 13.0,
    'Qwen3-0.6B': 0.6, 'Qwen3-1.7B': 1.7, 'Qwen3-4B': 4.0, 'Qwen3-8B': 8.0, 'Qwen3-32B': 32.0,
    'Qwen3-0.6B (think)': 0.6, 'Qwen3-1.7B (think)': 1.7, 'Qwen3-4B (think)': 4.0, 'Qwen3-8B (think)': 8.0, 'Qwen3-32B (think)': 32.0,
    'Qwen2.5-7B-Instruct': 7.0, 'Mistral-7B': 7.0, 'Phi-3.5-mini': 3.8, 'Vicuna-7B': 7.0, 'Vicuna-13B': 13.0,
}
CT_TO_VARIANT = {'question': 'C', 'weaker_object': 'B', 'pronominalized': 'A'}
X = [0, 1, 2]
XLABELS = ['Original', 'Weaker', 'Pronominalized']
JITTER = {'C': -0.04, 'B': 0.0, 'A': +0.04}
SIZE_TICKS = [0.6, 1, 2, 4, 7, 8, 13, 32]
MAX_SIZE = 32.0
PANELS = [
    ('Full VLMs', 'VLM', 35.0),
    ('Backbone Decoders', 'VLM backbone decoder', 10.0),
    ('Standalone LLMs', 'standalone LLM', 35.0),
    ('Think Models', 'standalone LLM (think)', 95.0),
]


def _family(name: str) -> str:
    n = name.replace(' (LM)', '').replace(' (think)', '').strip()
    if 'InternVL' in n:
        return 'InternVL'
    if 'Qwen3-VL' in n:
        return 'Qwen3-VL'
    if 'LLaVA-1.5' in n:
        return 'LLaVA-1.5'
    if 'LLaVA-Mistral' in n:
        return 'LLaVA-Mistral'
    if 'LLaVA-Vicuna' in n:
        return 'LLaVA-Vicuna'
    if 'Qwen3' in n and '(think)' in name:
        return 'Qwen3 (think)'
    if 'Qwen3' in n:
        return 'Qwen3'
    if 'Qwen2.5' in n:
        return 'Qwen2.5'
    if 'Mistral' in n:
        return 'Mistral'
    if 'Vicuna' in n:
        return 'Vicuna'
    if 'Phi' in n:
        return 'Phi'
    return 'Other'


def _is_merged_abstained(resp: str) -> bool:
    cls = classify(resp, None)
    return cls in ('hard_abstained', 'soft_abstained')


def _full_rates() -> pd.DataFrame:
    rows = []
    for label, (rdir, mdir) in default_all_models(ROOT).items():
        path_blind = rdir / mdir / 'vqa_1k_control_blind.jsonl'
        path_inst = rdir / mdir / 'vqa_1k_control_inst_blind.jsonl'
        for condition, path in [('blind', path_blind), ('inst_blind', path_inst)]:
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qid = int(ex['question_id'])
                    ga = ex.get('generated_answers', {})
                    for ct, variant in CT_TO_VARIANT.items():
                        resp = clean_answer(str(ga.get(ct, '')))
                        rows.append({
                            'question_id': qid,
                            'model': label,
                            'group': MODEL_TYPE.get(label, 'unknown'),
                            'condition': condition,
                            'variant': variant,
                            'response': resp,
                            'abstained': _is_merged_abstained(resp),
                        })
    df = pd.DataFrame(rows)
    return df.groupby(['model', 'group', 'condition', 'variant'], as_index=False)['abstained'].mean().assign(abst_pct=lambda d: d['abstained'] * 100.0)


def _subset_rates() -> pd.DataFrame:
    rows = []
    for variant in ['C', 'B', 'A']:
        exp = read_response_exports(ROOT, variant=variant, check_completeness=False)
        for condition, key in [('blind', 'model_blind'), ('inst_blind', 'model_inst_blind')]:
            df = exp[key].copy()
            if df.empty:
                continue
            df['response'] = df['response'].fillna('').astype(str).apply(clean_answer)
            df['abstained'] = df['response'].apply(_is_merged_abstained)
            if 'model_group' in df.columns:
                grp_col = 'model_group'
            elif 'subject_group_2' in df.columns:
                grp_col = 'subject_group_2'
            else:
                grp_col = None
            grp_map = df[grp_col] if grp_col else df['model'].map(MODEL_TYPE)
            tmp = pd.DataFrame({
                'model': df['model'],
                'group': grp_map,
                'condition': condition,
                'variant': variant,
                'abstained': df['abstained'],
            })
            rows.append(tmp)
    all_df = pd.concat(rows, ignore_index=True)
    return all_df.groupby(['model', 'group', 'condition', 'variant'], as_index=False)['abstained'].mean().assign(abst_pct=lambda d: d['abstained'] * 100.0)


def _plot_scale_scatter(df: pd.DataFrame, out_name: str, ymax_override: dict[str, float] | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(4.4, 4.6), sharex=True, sharey=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.24, wspace=0.38, hspace=0.35)
    axes_flat = axes.flatten()
    for ax_idx, (ax, (title, group_name, default_ymax)) in enumerate(zip(axes_flat, PANELS)):
        ymax = ymax_override.get(group_name, default_ymax) if ymax_override else default_ymax
        is_top_row = ax_idx < 2
        for condition, filled in [('inst_blind', True), ('blind', False)]:
            sub = df[(df['group'] == group_name) & (df['condition'] == condition)]
            for _, row in sub.iterrows():
                size = MODEL_SIZE_B.get(row['model'])
                if size is None:
                    continue
                mrk = FAMILY_MARKER.get(_family(row['model']), 'o')
                ms = 28 + 55 * (max(0.0, np.log10(size + 0.3) / np.log10(MAX_SIZE + 0.3))) ** 0.8
                x = size * (1.0 + JITTER[row['variant']])
                col = VARIANT_COLORS[row['variant']]
                if filled:
                    ax.scatter(x, row['abst_pct'], s=ms, marker=mrk, color=col, alpha=0.80, zorder=3, edgecolors='white', linewidths=0.4)
                else:
                    ax.scatter(x, row['abst_pct'], s=ms, marker=mrk, facecolors='white', edgecolors=col, linewidths=1.1, alpha=0.85, zorder=3)
        ax.axhline(0, color='#555', lw=0.9, ls=':', zorder=1)
        ax.grid(True, lw=0.4, alpha=0.30, zorder=0)
        ax.set_xscale('log')
        ax.set_xticks(SIZE_TICKS)
        ax.set_xlim(0.4, 45)
        ax.set_ylim(-ymax * 0.06, ymax)
        ax.set_title(title, fontweight='bold', fontsize=8.5, pad=4)
        ax.tick_params(labelsize=7)
        if is_top_row:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xticklabels(['0.6', '1', '2', '4', '7', '8', '13', '32'], fontsize=7)
            ax.set_xlabel('Parameters (B)', fontsize=7.5)
    leg_handles = [
        mlines.Line2D([], [], marker='o', color=VARIANT_COLORS['C'], linestyle='None', markersize=6, label='Original'),
        mlines.Line2D([], [], marker='o', color=VARIANT_COLORS['B'], linestyle='None', markersize=6, label='Weaker'),
        mlines.Line2D([], [], marker='o', color=VARIANT_COLORS['A'], linestyle='None', markersize=6, label='Pronominalized'),
        mlines.Line2D([], [], marker='o', color='#666', markerfacecolor='#666', linestyle='None', markersize=6, label='Inst-blind'),
        mlines.Line2D([], [], marker='o', color='#666', markerfacecolor='white', linestyle='None', markersize=6, label='Blind'),
    ]
    fig.legend(handles=leg_handles, ncol=5, frameon=True, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 0.01), borderaxespad=0)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    shutil.copy(out, LATEX_OUT / out_name)
    print(f'Saved: {out}')


def main():
    full = _full_rates()
    subset = _subset_rates()
    _plot_scale_scatter(full, 'abstention_by_variant_scatter.png')
    _plot_scale_scatter(subset, 'abstention_by_variant_scatter_q113_merged.png', {
        'VLM': 15.0,
        'VLM backbone decoder': 10.0,
        'standalone LLM': 35.0,
        'standalone LLM (think)': 95.0,
    })


if __name__ == '__main__':
    main()
