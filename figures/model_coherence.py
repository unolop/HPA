"""
Supplementary: within-group model-model (MM) coherence across control variants.

Shows how much models within each group agree with each other (SBERT cosine)
as the question is progressively weakened: Original (C) → Weaker (B) → Pronoun. (A).

Key finding: backbone decoders are most internally coherent (shared weights) but drop
steepest at variant A; standalone LLMs have lowest coherence throughout.
The steep B→A drop in aggregate MM is driven by abstention divergence — some models
stop answering when the entity name is removed, others keep guessing.

Outputs:
  figures/model_coherence/
    within_group_MM_{metric}_lineplot_q{N}_h{H}.png   — one line per group
    within_group_MM_{metric}_heatmap_q{N}_h{H}.png    — group × variant heatmap

Metrics: sbert (default), optionally all metrics via --all_metrics

Run from repo root:
  conda run -n zero python figures/model_coherence.py
  conda run -n zero python figures/model_coherence.py --all_metrics
  conda run -n zero python figures/model_coherence.py --include_yesno
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from utils.constants import GROUP_COLORS, GROUP_ORDER
from helpers import clear_output_plots, get_exports_dir, load_pair_cache
from config import MODELS_7B

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--all_metrics',   action='store_true',
                    help='Export all metrics (default: sbert only)')
parser.add_argument('--include_yesno', action='store_true',
                    help='Extend pair cache with yes/no question pairs')
parser.add_argument('--overwrite', action='store_true',
                    help='Delete existing plot files in the output folder before exporting.')
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = ROOT / 'figures/model_coherence'
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)

pair_df = load_pair_cache(ROOT, include_yesno=args.include_yesno, verbose=True)

n_questions = pair_df[pair_df['pair_type'] == 'HH']['question_id'].nunique()
n_humans    = len(
    set(pair_df.loc[pair_df['pair_type'] == 'HH', 'subject_1'])
    | set(pair_df.loc[pair_df['pair_type'] == 'HH', 'subject_2'])
)
SUFFIX = f'_q{n_questions}_h{n_humans}' + ('_yesno' if args.include_yesno else '')

# ── Constants ─────────────────────────────────────────────────────────────────
VARIANTS   = ['C', 'B', 'A']
VAR_LABELS = {'C': 'Original (C)', 'B': 'Weaker (B)', 'A': 'Pronoun. (A)'}

GROUP_SHORT = {
    'VLM':                    'VLM',
    'VLM backbone decoder':   'Backbone decoder',
    'standalone LLM':         'LLM',
    'standalone LLM (think)': 'LLM (think)',
}

METRICS = {'sbert': ('SBERT cosine', 'sbert_score')}
if args.all_metrics:
    METRICS.update({
        'chrf':      ('chrF',          'chrf_score'),
        'rouge1':    ('ROUGE-1',       'rouge1_score'),
        'jaccard':   ('Token Jaccard', 'jaccard_score'),
        'exact':     ('Exact match',   'exact_score'),
        'simcse':    ('SimCSE cosine', 'simcse_score'),
        'bertscore': ('BERTScore F1',  'bertscore_f1'),
    })

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
    'axes.labelsize':    11,
    'axes.titlesize':    13,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   9,
})


def _apply_axis_style(ax):
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(length=3.5, width=0.8, color='#555555')
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ── Within-group MM pairs only ────────────────────────────────────────────────
mm = pair_df[pair_df['pair_type'] == 'MM'].copy()
mm_within = mm[mm['subject_group_1'] == mm['subject_group_2']].copy()
mm_within = mm_within.rename(columns={'subject_group_1': 'group'})
mm_within_7b = mm_within[
    mm_within['subject_1'].isin(MODELS_7B) & mm_within['subject_2'].isin(MODELS_7B)
].copy()

print(f'\nWithin-group MM pairs: {len(mm_within):,}  '
      f'(dropped {len(mm) - len(mm_within):,} cross-group pairs)')
print(mm_within.groupby('group').size().to_string())
print(f'Within-group MM 7B pairs: {len(mm_within_7b):,}')
if not mm_within_7b.empty:
    print(mm_within_7b.groupby('group').size().to_string())

# ── Export ────────────────────────────────────────────────────────────────────
for label, (metric_name, col) in METRICS.items():
    print(f'\n── {metric_name} ──')

    # ── Lineplot: one line per model group ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _apply_axis_style(ax)

    for grp in GROUP_ORDER:
        g = mm_within[mm_within['group'] == grp]
        if g.empty:
            continue
        ys  = [g[g['variant'] == v][col].mean() for v in VARIANTS]
        cis = [1.96 * g[g['variant'] == v][col].sem() for v in VARIANTS]
        ax.fill_between(range(3),
                        [y - c for y, c in zip(ys, cis)],
                        [y + c for y, c in zip(ys, cis)],
                        color=GROUP_COLORS[grp], alpha=0.12)
        ax.plot(range(3), ys,
                color=GROUP_COLORS[grp], lw=2, marker='o', markersize=7,
                label=GROUP_SHORT[grp],
                markeredgecolor='white', markeredgewidth=0.8)
        ax.text(2.08, ys[-1], f'{ys[-1]:.3f}',
                va='center', fontsize=7.5, color=GROUP_COLORS[grp])

        # Print delta summary
        print(f'  {GROUP_SHORT[grp]:22s}  C={ys[0]:.3f}  B={ys[1]:.3f}  A={ys[2]:.3f}'
              f'  |  C→B={ys[0]-ys[1]:+.3f}  B→A={ys[1]-ys[2]:+.3f}')

    ax.set_xticks(range(3))
    ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=9)
    ax.set_ylabel(f'Mean {metric_name} (within-group model pairs)', fontsize=10)
    ax.set_title(f'Model-Model Coherence Across Control Variants\n'
                 f'(inst_blind, q={n_questions}, h={n_humans})', fontsize=12)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, frameon=True)
    ax.set_xlim(-0.3, 2.7)
    fig.tight_layout()
    _save(fig, f'within_group_MM_{label}_lineplot{SUFFIX}.png')

    # ── Heatmap: group × variant ──────────────────────────────────────────────
    data = {}
    for grp in GROUP_ORDER:
        g = mm_within[mm_within['group'] == grp]
        if g.empty:
            continue
        data[GROUP_SHORT[grp]] = {
            VAR_LABELS[v]: g[g['variant'] == v][col].mean()
            for v in VARIANTS
        }

    heat_df = pd.DataFrame(data).T[list(VAR_LABELS.values())]
    vmin, vmax = heat_df.min().min(), heat_df.max().max()

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    sns.heatmap(heat_df, ax=ax, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='white',
                annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
    ax.set_title(f'Model-Model Coherence — {metric_name}\n'
                 f'(within-group, inst_blind, q={n_questions}, h={n_humans})',
                 fontsize=12)
    ax.set_xlabel('Control variant', fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    _save(fig, f'within_group_MM_{label}_heatmap{SUFFIX}.png')

    # ── 7B matched-size subset: same plots filtered to 7/8B models ──────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _apply_axis_style(ax)

    for grp in GROUP_ORDER:
        g = mm_within_7b[mm_within_7b['group'] == grp]
        if g.empty:
            continue
        ys  = [g[g['variant'] == v][col].mean() for v in VARIANTS]
        cis = [1.96 * g[g['variant'] == v][col].sem() for v in VARIANTS]
        ax.fill_between(range(3),
                        [y - c for y, c in zip(ys, cis)],
                        [y + c for y, c in zip(ys, cis)],
                        color=GROUP_COLORS[grp], alpha=0.12)
        ax.plot(range(3), ys,
                color=GROUP_COLORS[grp], lw=2, marker='o', markersize=7,
                label=GROUP_SHORT[grp],
                markeredgecolor='white', markeredgewidth=0.8)
        ax.text(2.08, ys[-1], f'{ys[-1]:.3f}',
                va='center', fontsize=7.5, color=GROUP_COLORS[grp])

        print(f'  {GROUP_SHORT[grp]:22s} [7B]  C={ys[0]:.3f}  B={ys[1]:.3f}  A={ys[2]:.3f}'
              f'  |  C→B={ys[0]-ys[1]:+.3f}  B→A={ys[1]-ys[2]:+.3f}')

    ax.set_xticks(range(3))
    ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=9)
    ax.set_ylabel(f'Mean {metric_name} (within-group model pairs)', fontsize=10)
    ax.set_title(f'Model-Model Coherence Across Control Variants — 7/8B Subset\n'
                 f'(inst_blind, q={n_questions}, h={n_humans})', fontsize=12)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, frameon=True)
    ax.set_xlim(-0.3, 2.7)
    fig.tight_layout()
    _save(fig, f'within_group_MM_{label}_lineplot{SUFFIX}_7b.png')

    data7 = {}
    for grp in GROUP_ORDER:
        g = mm_within_7b[mm_within_7b['group'] == grp]
        if g.empty:
            continue
        data7[GROUP_SHORT[grp]] = {
            VAR_LABELS[v]: g[g['variant'] == v][col].mean()
            for v in VARIANTS
        }

    heat7 = pd.DataFrame(data7).T[list(VAR_LABELS.values())]
    vmin7, vmax7 = heat7.min().min(), heat7.max().max()

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    sns.heatmap(heat7, ax=ax, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=vmin7, vmax=vmax7, linewidths=0.5, linecolor='white',
                annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
    ax.set_title(f'Model-Model Coherence — {metric_name} — 7/8B Subset\n'
                 f'(within-group, inst_blind, q={n_questions}, h={n_humans})',
                 fontsize=12)
    ax.set_xlabel('Control variant', fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    _save(fig, f'within_group_MM_{label}_heatmap{SUFFIX}_7b.png')

print(f'\nDone. Outputs → {OUT_DIR}')
