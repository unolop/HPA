"""
Export agreement-by-control-variant figures for the paper.

For each agreement metric, produces two figures:
  figures/agreement_variants/  — mean HM/HH/MM agreement per group across variants C→B→A
  figures/agreement_variants/  — group×variant heatmap (HM pairs only)

Metrics: sbert, simcse, bertscore, chrf, rouge1, jaccard, exact

Filenames:
  inst_blind_vABC_{metric}_lineplot_q{N}_h{H}[_yesno].png
  inst_blind_vABC_{metric}_heatmap_q{N}_h{H}[_yesno].png

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

from utils.constants import GROUP_COLORS, GROUP_ORDER, GROUP_MARKER, GROUP_HOLLOW, GROUP_LINESTYLE
from export_helpers import get_exports_dir, load_pair_cache
from config import MODELS_7B

parser = argparse.ArgumentParser()
parser.add_argument('--include_yesno', action='store_true',
                    help='Extend pair cache with yes/no question pairs.')
args = parser.parse_args()

LINEPLOT_DIR    = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/agreement_variants'
HEATMAP_DIR     = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/agreement_variants'
LINEPLOT_DIR_7B = LINEPLOT_DIR / '7b'
HEATMAP_DIR_7B  = HEATMAP_DIR  / '7b'
LINEPLOT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
LINEPLOT_DIR_7B.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR_7B.mkdir(parents=True, exist_ok=True)

EXPORTS = get_exports_dir(ROOT)
pair_df = load_pair_cache(ROOT, include_yesno=args.include_yesno, verbose=True)

def hh_participant_count(df: pd.DataFrame) -> int:
    hh = df[df['pair_type'] == 'HH']
    return len(set(hh['subject_1']).union(set(hh['subject_2'])))

n_questions = pair_df[pair_df['pair_type'] == 'HH']['question_id'].nunique()
n_humans = hh_participant_count(pair_df)
SUFFIX = f'_q{n_questions}_h{n_humans}' + ('_yesno' if args.include_yesno else '')

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
MM_COLOR = '#757575'   # neutral gray for model–model baseline

def save(fig, name, out_dir):
    path = out_dir / name
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    folder = out_dir.name
    print(f'  [{folder}] {name}')


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
mm    = pair_df[pair_df['pair_type'] == 'MM'].copy()
hm_7b = hm[hm['subject_2'].isin(MODELS_7B)]
mm_7b = mm[mm['subject_2'].isin(MODELS_7B)]

for label, (metric_name, col) in METRICS.items():
    print(f'\n── {metric_name} ({label}) ──')

    # ── 1. Line plot: HM per group + HH ceiling + MM baseline ────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    apply_axis_style(ax, grid_axis='y')

    hh_means = hh.groupby('variant')[col].mean()
    hh_sems  = hh.groupby('variant')[col].sem()
    mm_means = mm.groupby('variant')[col].mean()
    mm_sems  = mm.groupby('variant')[col].sem()

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
        ax.plot(range(3), ys,
                color=GROUP_COLORS[grp], lw=2, marker='o', markersize=7,
                label=GROUP_SHORT[grp].replace('\n', ' '),
                markeredgecolor='white', markeredgewidth=0.8)
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

    # MM baseline
    mm_ys  = [mm_means.get(v, np.nan) for v in VARIANTS]
    mm_cis = [1.96 * mm_sems.get(v, 0) for v in VARIANTS]
    ax.fill_between(range(3),
                    [y - c for y, c in zip(mm_ys, mm_cis)],
                    [y + c for y, c in zip(mm_ys, mm_cis)],
                    color=MM_COLOR, alpha=0.10)
    ax.plot(range(3), mm_ys, color=MM_COLOR, lw=1.6, ls=':',
            marker='D', markersize=6, label='Model–Model',
            markeredgecolor='white', markeredgewidth=0.6)
    ax.text(2.08, mm_ys[-1], f'{mm_ys[-1]:.3f}',
            va='center', fontsize=7.5, color=MM_COLOR)

    ax.set_xticks(range(3))
    ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=9)
    ax.set_ylabel(f'Mean {metric_name}', fontsize=10)
    ax.set_title(f'{metric_name} Across Control Variants\n(inst_blind, q={n_questions}, h={n_humans})',
                 fontsize=12)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=5, frameon=True)
    ax.set_xlim(-0.3, 2.7)
    fig.tight_layout()
    save(fig, f'inst_blind_vABC_{label}_lineplot{SUFFIX}.png', LINEPLOT_DIR)

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
    ax.set_title(f'{metric_name} Heatmap\n(inst_blind, q={n_questions}, h={n_humans})',
                 fontsize=12)
    ax.set_xlabel('Control variant', fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    save(fig, f'inst_blind_vABC_{label}_heatmap{SUFFIX}.png', HEATMAP_DIR)

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
        ax.plot(range(3), ys,
                color=GROUP_COLORS[grp], lw=2, marker='o', markersize=7,
                label=GROUP_SHORT[grp].replace('\n', ' '),
                markeredgecolor='white', markeredgewidth=0.8)
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

    mm_means_7b = mm_7b.groupby('variant')[col].mean()
    mm_sems_7b  = mm_7b.groupby('variant')[col].sem()
    mm_ys_7b  = [mm_means_7b.get(v, np.nan) for v in VARIANTS]
    mm_cis_7b = [1.96 * mm_sems_7b.get(v, 0) for v in VARIANTS]
    ax.fill_between(range(3),
                    [y - c for y, c in zip(mm_ys_7b, mm_cis_7b)],
                    [y + c for y, c in zip(mm_ys_7b, mm_cis_7b)],
                    color=MM_COLOR, alpha=0.10)
    ax.plot(range(3), mm_ys_7b, color=MM_COLOR, lw=1.6, ls=':',
            marker='D', markersize=6, label='Model–Model',
            markeredgecolor='white', markeredgewidth=0.6)
    ax.text(2.08, mm_ys_7b[-1], f'{mm_ys_7b[-1]:.3f}',
            va='center', fontsize=7.5, color=MM_COLOR)

    ax.set_xticks(range(3))
    ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=9)
    ax.set_ylabel(f'Mean {metric_name}', fontsize=10)
    ax.set_title(f'{metric_name} Across Control Variants — 7/8B Subset\n'
                 f'(inst_blind, q={n_questions}, h={n_humans})', fontsize=12)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=5, frameon=True)
    ax.set_xlim(-0.3, 2.7)
    fig.tight_layout()
    save(fig, f'inst_blind_vABC_{label}_lineplot{SUFFIX}.png', LINEPLOT_DIR_7B)

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
    data7['Model–Model'] = {VAR_LABELS[v]: mm_means.get(v, np.nan) for v in VARIANTS}

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
    save(fig, f'inst_blind_vABC_{label}_heatmap{SUFFIX}.png', HEATMAP_DIR_7B)

print(f'\nDone. Lineplots → {LINEPLOT_DIR}\n      Heatmaps  → {HEATMAP_DIR}')
