"""
Export agreement-by-control-variant figures for the paper.

For each agreement metric, produces two figures:
  figures/agreement_lineplot/  — mean HM/HH/MM agreement per group across variants C→B→A
  figures/agreement_heatmap/   — group×variant heatmap (HM pairs only)

Metrics: sbert, simcse, bertscore, chrf, rouge1, jaccard, exact

Filenames: inst_blind_vABC_{metric}_groups[_yesno].png

Run from repo root:
  conda run -n zero python analysis/export_agreement_variants.py
  conda run -n zero python analysis/export_agreement_variants.py --include_yesno
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.constants import GROUP_COLORS, GROUP_ORDER
from config import extend_pair_cache_with_yesno

parser = argparse.ArgumentParser()
parser.add_argument('--include_yesno', action='store_true',
                    help='Extend pair cache with yes/no question pairs.')
args = parser.parse_args()

LINEPLOT_DIR = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/agreement_lineplot'
HEATMAP_DIR  = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/agreement_heatmap'
LINEPLOT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

EXPORTS = ROOT / 'analysis/session2/exports'
pair_df = pd.read_parquet(EXPORTS / 'pair_cache.parquet')
if args.include_yesno:
    print('Extending pair_cache with yes/no question pairs…')
    pair_df = extend_pair_cache_with_yesno(pair_df, EXPORTS)

SUFFIX = '_yesno' if args.include_yesno else ''

# ── Constants ─────────────────────────────────────────────────────────────────
VARIANTS    = ['C', 'B', 'A']
VAR_LABELS  = {'C': 'Original (C)', 'B': 'Weaker (B)', 'A': 'Pronoun. (A)'}

GROUP_SHORT = {
    'VLM backbone decoder':   'Backbone\ndecoder',
    'VLM':                    'VLM',
    'standalone LLM (think)': 'LLM\n(think)',
    'standalone LLM':         'LLM',
}

METRICS = {
    'sbert':       ('SBERT cosine',      'sbert_score'),
    'simcse':      ('SimCSE cosine',     'simcse_score'),
    'bertscore':   ('BERTScore F1',      'bertscore_f1'),
    'chrf':        ('chrF',              'chrf_score'),
    'rouge1':      ('ROUGE-1',           'rouge1_score'),
    'jaccard':     ('Token Jaccard',     'jaccard_score'),
    'exact':       ('Exact match',       'exact_score'),
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        False,
})

HH_COLOR = '#1565C0'
MM_COLOR = '#757575'   # neutral gray for model–model baseline

def save(fig, name, out_dir):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    folder = out_dir.name
    print(f'  [{folder}] {name}')


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute: mean score per (variant × group × pair_type)
# ─────────────────────────────────────────────────────────────────────────────
hm = pair_df[pair_df['pair_type'] == 'HM'].copy()
hh = pair_df[pair_df['pair_type'] == 'HH'].copy()
mm = pair_df[pair_df['pair_type'] == 'MM'].copy()

for label, (metric_name, col) in METRICS.items():
    print(f'\n── {metric_name} ({label}) ──')

    # ── 1. Line plot: HM per group + HH ceiling + MM baseline ────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    hh_means = hh.groupby('variant')[col].mean()
    mm_means = mm.groupby('variant')[col].mean()

    for grp in GROUP_ORDER:
        g_hm = hm[hm['subject_group_2'] == grp]
        if g_hm.empty:
            continue
        ys = [g_hm[g_hm['variant'] == v][col].mean() for v in VARIANTS]
        ax.plot(range(3), ys,
                color=GROUP_COLORS[grp], lw=2, marker='o', markersize=7,
                label=GROUP_SHORT[grp].replace('\n', ' '),
                markeredgecolor='white', markeredgewidth=0.8)
        ax.text(2.08, ys[-1], f'{ys[-1]:.3f}',
                va='center', fontsize=7.5, color=GROUP_COLORS[grp])

    # HH ceiling
    hh_ys = [hh_means.get(v, np.nan) for v in VARIANTS]
    ax.plot(range(3), hh_ys, color=HH_COLOR, lw=1.8, ls='--',
            marker='*', markersize=10, label='Human–Human',
            markeredgecolor='white', markeredgewidth=0.6)
    ax.text(2.08, hh_ys[-1], f'{hh_ys[-1]:.3f}',
            va='center', fontsize=7.5, color=HH_COLOR)

    # MM baseline
    mm_ys = [mm_means.get(v, np.nan) for v in VARIANTS]
    ax.plot(range(3), mm_ys, color=MM_COLOR, lw=1.6, ls=':',
            marker='D', markersize=6, label='Model–Model',
            markeredgecolor='white', markeredgewidth=0.6)
    ax.text(2.08, mm_ys[-1], f'{mm_ys[-1]:.3f}',
            va='center', fontsize=7.5, color=MM_COLOR)

    ax.set_xticks(range(3))
    ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=9)
    ax.set_ylabel(f'Mean {metric_name}', fontsize=10)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=5, frameon=True)
    ax.set_xlim(-0.3, 2.7)
    plt.tight_layout()
    save(fig, f'inst_blind_vABC_{label}_groups{SUFFIX}.png', LINEPLOT_DIR)

    # ── 2. Heatmap: group × variant (HM pairs, mean score) + HH + MM rows ────
    data = {}
    for grp in GROUP_ORDER:
        g_hm = hm[hm['subject_group_2'] == grp]
        data[GROUP_SHORT[grp].replace('\n', ' ')] = {
            VAR_LABELS[v]: g_hm[g_hm['variant'] == v][col].mean()
            for v in VARIANTS
        }
    data['Human–Human'] = {VAR_LABELS[v]: hh_means.get(v, np.nan) for v in VARIANTS}
    data['Model–Model'] = {VAR_LABELS[v]: mm_means.get(v, np.nan) for v in VARIANTS}

    heat_df = pd.DataFrame(data).T[list(VAR_LABELS.values())]

    vmin = heat_df.min().min()
    vmax = heat_df.max().max()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(heat_df, ax=ax, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='white',
                annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Control variant', fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    save(fig, f'inst_blind_vABC_{label}_groups{SUFFIX}.png', HEATMAP_DIR)

print(f'\nDone. Lineplots → {LINEPLOT_DIR}\n      Heatmaps  → {HEATMAP_DIR}')
