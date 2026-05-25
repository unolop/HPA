"""
Export agreement-by-control-variant figures for the paper.

For each agreement metric, produces two figures:
  figures/agreement_vABC_group_lineplot/  — mean HM/HH agreement per group across variants C→B→A
  figures/agreement_vABC_group_heatmap/   — group×variant heatmap (HM pairs only)

Metrics: sbert, simcse, bertscore, chrf, rouge1, jaccard, exact

Filenames:
  inst_blind_{metric}_q{N}_h{H}[_yesno].png

Note: Model-Model (MM) within-group coherence figures are in figures/model_coherence.py.

Run from repo root:
  conda run -n zero python figures/agreement_variants.py
  conda run -n zero python figures/agreement_variants.py --include_yesno
"""

import argparse
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from utils.constants import GROUP_COLORS, GROUP_ORDER, GROUP_MARKER, GROUP_HOLLOW, GROUP_LINESTYLE
from helpers import clear_output_plots, get_exports_dir, load_pair_cache
from config import MODELS_7B

parser = argparse.ArgumentParser()
parser.add_argument('--include_yesno', action='store_true',
                    help='Extend pair cache with yes/no question pairs.')
parser.add_argument('--overwrite', action='store_true',
                    help='Delete existing plot files in the output folder before exporting.')
args = parser.parse_args()

LINEPLOT_DIR = ROOT / 'figures/agreement_vABC_group_lineplot'
HEATMAP_DIR  = ROOT / 'figures/agreement_vABC_group_heatmap'
LINEPLOT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(LINEPLOT_DIR, overwrite=args.overwrite)
if args.overwrite and HEATMAP_DIR.exists():
    shutil.rmtree(HEATMAP_DIR)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Cleared nested heatmap outputs from {HEATMAP_DIR}')
else:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

EXPORTS = get_exports_dir(ROOT)
pair_df = load_pair_cache(ROOT, include_yesno=args.include_yesno, verbose=True)

def hh_participant_count(df: pd.DataFrame) -> int:
    hh = df[df['pair_type'] == 'HH']
    return len(set(hh['subject_1']).union(set(hh['subject_2'])))

n_questions = pair_df[pair_df['pair_type'] == 'HH']['question_id'].nunique()
n_humans = hh_participant_count(pair_df)
SUFFIX = f'_q{n_questions}_h{n_humans}' + ('_yesno' if args.include_yesno else '')
QSET_TAG = f'q{n_questions}_h{n_humans}' + ('_yesno' if args.include_yesno else '')

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
    'axes.labelsize':   11,
    'axes.titlesize':   13,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  9,
})

HH_COLOR = '#1565C0'

def save(fig, name, out_dir):
    path = out_dir / name
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    folder = out_dir.name
    print(f'  [{folder}] {name}')


def heatmap_out_dir(metric_key: str, is_7b: bool) -> Path:
    scale_tag = 'scale_7b' if is_7b else 'scale_all'
    out_dir = HEATMAP_DIR / QSET_TAG / scale_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def apply_axis_style(ax, grid_axis='y'):
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(length=3.5, width=0.8, color='#555555')
    if grid_axis:
        ax.grid(axis=grid_axis, color='#D9D9D9', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute: mean score per (variant × group × pair_type)
# ─────────────────────────────────────────────────────────────────────────────
hm    = pair_df[pair_df['pair_type'] == 'HM'].copy()
hh    = pair_df[pair_df['pair_type'] == 'HH'].copy()
hm_7b = hm[hm['subject_2'].isin(MODELS_7B)]

for label, (metric_name, col) in METRICS.items():
    print(f'\n── {metric_name} ({label}) ──')

    # ── 1. Line plot: HM per group + HH ceiling ──────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    apply_axis_style(ax, grid_axis='y')

    hh_means = hh.groupby('variant')[col].mean()
    hh_sems  = hh.groupby('variant')[col].sem()

    for grp in GROUP_ORDER:
        g_hm = hm[hm['subject_group_2'] == grp]
        if g_hm.empty:
            continue
        ys   = [g_hm[g_hm['variant'] == v][col].mean() for v in VARIANTS]
        cis  = [1.96 * g_hm[g_hm['variant'] == v][col].sem() for v in VARIANTS]
        ax.fill_between(range(3),
                        [y - c for y, c in zip(ys, cis)],
                        [y + c for y, c in zip(ys, cis)],
                        color=GROUP_COLORS[grp], alpha=0.12)
        mkr = GROUP_MARKER.get(grp, 'o')
        hollow = GROUP_HOLLOW.get(grp, False)
        ls = GROUP_LINESTYLE.get(grp, '-')
        ax.plot(range(3), ys,
                color=GROUP_COLORS[grp], lw=2, ls=ls, marker=mkr, markersize=7,
                label=GROUP_SHORT[grp].replace('\n', ' '),
                markerfacecolor='none' if hollow else GROUP_COLORS[grp],
                markeredgecolor=GROUP_COLORS[grp], markeredgewidth=1.0)
        ax.text(2.08, ys[-1], f'{ys[-1]:.3f}',
                va='center', fontsize=7.5, color=GROUP_COLORS[grp])

    # HH ceiling
    hh_ys  = [hh_means.get(v, np.nan) for v in VARIANTS]
    hh_cis = [1.96 * hh_sems.get(v, 0) for v in VARIANTS]
    ax.fill_between(range(3),
                    [y - c for y, c in zip(hh_ys, hh_cis)],
                    [y + c for y, c in zip(hh_ys, hh_cis)],
                    color=HH_COLOR, alpha=0.10)
    ax.plot(range(3), hh_ys, color=HH_COLOR, lw=1.8, ls='--',
            marker='*', markersize=10, label='Human–Human',
            markeredgecolor='white', markeredgewidth=0.6)
    ax.text(2.08, hh_ys[-1], f'{hh_ys[-1]:.3f}',
            va='center', fontsize=7.5, color=HH_COLOR)

    ax.set_xticks(range(3))
    ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=9)
    ax.set_ylabel(f'Mean {metric_name}', fontsize=10)
    ax.set_title(f'{metric_name} Across Control Variants\n(inst_blind, q={n_questions}, h={n_humans})',
                 fontsize=12)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=5, frameon=True)
    ax.set_xlim(-0.3, 2.7)
    fig.tight_layout()
    save(fig, f'inst_blind_{label}{SUFFIX}.png', LINEPLOT_DIR)

    # ── 2. Heatmap: group × variant (HM pairs, mean score) + HH + MM rows ────
    data = {}
    for grp in GROUP_ORDER:
        g_hm = hm[hm['subject_group_2'] == grp]
        data[GROUP_SHORT[grp].replace('\n', ' ')] = {
            VAR_LABELS[v]: g_hm[g_hm['variant'] == v][col].mean()
            for v in VARIANTS
        }
    data['Human–Human'] = {VAR_LABELS[v]: hh_means.get(v, np.nan) for v in VARIANTS}

    heat_df = pd.DataFrame(data).T[list(VAR_LABELS.values())]

    vmin = heat_df.min().min()
    vmax = heat_df.max().max()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(heat_df, ax=ax, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='white',
                annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
    ax.set_title(f'{metric_name} Heatmap\n(inst_blind, q={n_questions}, h={n_humans})',
                 fontsize=12)
    ax.set_xlabel('Control variant', fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    save(fig, f'inst_blind_{label}{SUFFIX}.png', heatmap_out_dir(label, is_7b=False))

    # ── 7B matched-size subset: same plots filtered to 7–8 B models ───────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    apply_axis_style(ax, grid_axis='y')

    for grp in GROUP_ORDER:
        g_hm7 = hm_7b[hm_7b['subject_group_2'] == grp]
        if g_hm7.empty:
            continue
        ys  = [g_hm7[g_hm7['variant'] == v][col].mean() for v in VARIANTS]
        cis = [1.96 * g_hm7[g_hm7['variant'] == v][col].sem() for v in VARIANTS]
        ax.fill_between(range(3),
                        [y - c for y, c in zip(ys, cis)],
                        [y + c for y, c in zip(ys, cis)],
                        color=GROUP_COLORS[grp], alpha=0.12)
        mkr = GROUP_MARKER.get(grp, 'o')
        hollow = GROUP_HOLLOW.get(grp, False)
        ls = GROUP_LINESTYLE.get(grp, '-')
        ax.plot(range(3), ys,
                color=GROUP_COLORS[grp], lw=2, ls=ls, marker=mkr, markersize=7,
                label=GROUP_SHORT[grp].replace('\n', ' '),
                markerfacecolor='none' if hollow else GROUP_COLORS[grp],
                markeredgecolor=GROUP_COLORS[grp], markeredgewidth=1.0)
        ax.text(2.08, ys[-1], f'{ys[-1]:.3f}',
                va='center', fontsize=7.5, color=GROUP_COLORS[grp])

    hh_ys  = [hh_means.get(v, np.nan) for v in VARIANTS]
    hh_cis = [1.96 * hh_sems.get(v, 0) for v in VARIANTS]
    ax.fill_between(range(3),
                    [y - c for y, c in zip(hh_ys, hh_cis)],
                    [y + c for y, c in zip(hh_ys, hh_cis)],
                    color=HH_COLOR, alpha=0.10)
    ax.plot(range(3), hh_ys, color=HH_COLOR, lw=1.8, ls='--',
            marker='*', markersize=10, label='Human–Human',
            markeredgecolor='white', markeredgewidth=0.6)
    ax.text(2.08, hh_ys[-1], f'{hh_ys[-1]:.3f}',
            va='center', fontsize=7.5, color=HH_COLOR)

    ax.set_xticks(range(3))
    ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=9)
    ax.set_ylabel(f'Mean {metric_name}', fontsize=10)
    ax.set_title(f'{metric_name} Across Control Variants — 7/8B Subset\n'
                 f'(inst_blind, q={n_questions}, h={n_humans})', fontsize=12)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=5, frameon=True)
    ax.set_xlim(-0.3, 2.7)
    fig.tight_layout()
    save(fig, f'inst_blind_{label}{SUFFIX}_7b.png', LINEPLOT_DIR)

    # 7B heatmap
    data7 = {}
    for grp in GROUP_ORDER:
        g_hm7 = hm_7b[hm_7b['subject_group_2'] == grp]
        if g_hm7.empty:
            continue
        data7[GROUP_SHORT[grp].replace('\n', ' ')] = {
            VAR_LABELS[v]: g_hm7[g_hm7['variant'] == v][col].mean()
            for v in VARIANTS
        }
    data7['Human–Human'] = {VAR_LABELS[v]: hh_means.get(v, np.nan) for v in VARIANTS}

    heat7 = pd.DataFrame(data7).T[list(VAR_LABELS.values())]
    vmin7, vmax7 = heat7.min().min(), heat7.max().max()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(heat7, ax=ax, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=vmin7, vmax=vmax7, linewidths=0.5, linecolor='white',
                annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
    ax.set_title(f'{metric_name} Heatmap — 7/8B Subset\n'
                 f'(inst_blind, q={n_questions}, h={n_humans})', fontsize=12)
    ax.set_xlabel('Control variant', fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    save(fig, f'inst_blind_{label}{SUFFIX}_7b.png', heatmap_out_dir(label, is_7b=True))

print(f'\nDone. Lineplots → {LINEPLOT_DIR}\n      Heatmaps  → {HEATMAP_DIR}')
