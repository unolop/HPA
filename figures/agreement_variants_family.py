"""
Export family-level C→B→A agreement subplot figure.

Layout: 2 rows × 3 columns
  Row 1 (VLM + Backbone): Qwen3-VL | InternVL | LLaVA (all)
  Row 2 (SA-LLM + Think): Qwen3   | Others   | Group summary

Within each panel:
  - Solid line = base group (VLM / SA-LLM)
  - Dotted line = derivative (backbone / think)
  - Line weight/alpha scales with parameter count
  - HH ceiling (dashed blue) in every panel

Output: figures/agreement/vABC_family/

Run from repo root:
  conda run -n zero python figures/agreement_variants_family.py
  conda run -n zero python figures/agreement_variants_family.py --include_yesno
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from utils.constants import (
    GROUP_COLORS, GROUP_ORDER, GROUP_MARKER, GROUP_HOLLOW, GROUP_LINESTYLE,
    MODEL_FAMILY, MODEL_FAMILY_COLORS, MODEL_SIZE_B,
)
from helpers import clear_output_plots, load_pair_cache
from config import MODEL_LABEL_SHORT

parser = argparse.ArgumentParser()
parser.add_argument('--include_yesno', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--metric', default='sbert',
                    choices=['sbert', 'simcse', 'bertscore', 'chrf',
                             'rouge1', 'jaccard', 'exact'])
args = parser.parse_args()

OUT_DIR = ROOT / 'figures/agreement/vABC_family'
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)

pair_df = load_pair_cache(ROOT, include_yesno=args.include_yesno, verbose=True)

SUFFIX = '_yesno' if args.include_yesno else ''

METRICS = {
    'sbert':     ('SBERT cosine',  'sbert_score'),
    'simcse':    ('SimCSE cosine', 'simcse_score'),
    'bertscore': ('BERTScore F1',  'bertscore_f1'),
    'chrf':      ('chrF',          'chrf_score'),
    'rouge1':    ('ROUGE-1',       'rouge1_score'),
    'jaccard':   ('Token Jaccard', 'jaccard_score'),
    'exact':     ('Exact match',   'exact_score'),
}

VARIANTS = ['C', 'B', 'A']
VAR_LABELS = {'C': 'Original (C)', 'B': 'Weaker (B)', 'A': 'Pronoun. (A)'}
HH_COLOR = '#1565C0'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
})

# ── Pre-split ────────────────────────────────────────────────────────────────
hm = pair_df[pair_df['pair_type'] == 'HM'].copy()
hh = pair_df[pair_df['pair_type'] == 'HH'].copy()

# Filter to models with complete data (expected question count per variant)
expected_qids = hm['question_id'].nunique()
_model_completeness = (hm.groupby('subject_2')['question_id'].nunique()
                       .reset_index()
                       .rename(columns={'subject_2': 'model', 'question_id': 'n_qids'}))
complete_models = set(_model_completeness[
    _model_completeness['n_qids'] >= expected_qids * 0.9]['model'])
hm = hm[hm['subject_2'].isin(complete_models)].copy()
print(f'Models after completeness filter: {len(complete_models)}/{len(_model_completeness)}')

models_df = (hm[['subject_2', 'subject_group_2']]
             .drop_duplicates()
             .rename(columns={'subject_2': 'model', 'subject_group_2': 'group'}))

# ── Panel definitions ────────────────────────────────────────────────────────
# Row 1: VLM + Backbone
PANEL_ROW1 = [
    {
        'title': 'Qwen3-VL',
        'families': ['Qwen3-VL'],
        'groups': ['VLM', 'VLM backbone decoder'],
    },
    {
        'title': 'InternVL',
        'families': ['InternVL'],
        'groups': ['VLM', 'VLM backbone decoder'],
    },
    {
        'title': 'LLaVA',
        'families': ['LLaVA-1.5', 'LLaVA-Mistral', 'LLaVA-Vicuna'],
        'groups': ['VLM', 'VLM backbone decoder'],
    },
]

# Row 2: SA-LLM + Think
PANEL_ROW2 = [
    {
        'title': 'Qwen3',
        'families': ['Qwen3'],
        'groups': ['standalone LLM', 'standalone LLM (think)'],
    },
    {
        'title': 'Others',
        'families': ['Vicuna', 'Mistral', 'Phi', 'Qwen2.5'],
        'groups': ['standalone LLM', 'standalone LLM (think)'],
    },
    {
        'title': 'Group Summary',
        'summary': True,
    },
]

PANELS = [PANEL_ROW1, PANEL_ROW2]


def _scale_style(model: str) -> dict:
    """Line weight/alpha based on parameter count."""
    size = MODEL_SIZE_B.get(model, 7.0)
    # Map size to alpha and linewidth
    if size <= 1.0:
        return {'alpha': 0.45, 'lw': 1.0, 'ms': 4}
    elif size <= 2.0:
        return {'alpha': 0.55, 'lw': 1.2, 'ms': 5}
    elif size <= 4.0:
        return {'alpha': 0.65, 'lw': 1.4, 'ms': 5.5}
    elif size <= 8.0:
        return {'alpha': 0.80, 'lw': 1.8, 'ms': 6.5}
    elif size <= 13.0:
        return {'alpha': 0.85, 'lw': 2.0, 'ms': 7}
    else:  # 32B+
        return {'alpha': 0.95, 'lw': 2.4, 'ms': 8}


def _short(model: str) -> str:
    return MODEL_LABEL_SHORT.get(model, model)


def _size_label(model: str) -> str:
    """Short label: just the size."""
    size = MODEL_SIZE_B.get(model)
    if size is None:
        return _short(model)
    if size == int(size):
        return f'{int(size)}B'
    return f'{size}B'


def plot_panel(ax, panel_def, col, hh_means):
    """Plot one panel: models from specified families/groups."""
    # HH ceiling
    hh_ys = [hh_means.get(v, np.nan) for v in VARIANTS]
    ax.plot(range(3), hh_ys, color=HH_COLOR, lw=1.8, ls='--',
            marker='*', markersize=9, zorder=4,
            markeredgecolor='white', markeredgewidth=0.5)

    families = panel_def['families']
    groups = panel_def['groups']

    # Get matching models
    panel_models = models_df[
        (models_df['model'].map(lambda m: MODEL_FAMILY.get(m, '')).isin(families)) &
        (models_df['group'].isin(groups))
    ].copy()

    if panel_models.empty:
        return

    # Sort by group (base first), then family, then size
    panel_models['size'] = panel_models['model'].map(lambda m: MODEL_SIZE_B.get(m, 0))
    panel_models['fam'] = panel_models['model'].map(lambda m: MODEL_FAMILY.get(m, ''))
    panel_models = panel_models.sort_values(['group', 'fam', 'size'])

    for _, row in panel_models.iterrows():
        model = row['model']
        grp = row['group']
        fam = MODEL_FAMILY.get(model, '')

        sub = hm[hm['subject_2'] == model]
        if sub.empty:
            continue

        ys = [sub[sub['variant'] == v][col].mean() for v in VARIANTS]
        if all(np.isnan(y) for y in ys):
            continue

        color = MODEL_FAMILY_COLORS.get(fam, '#888888')
        hollow = GROUP_HOLLOW.get(grp, False)
        ls = ':' if hollow else '-'
        mkr = 'o'
        style = _scale_style(model)

        ax.plot(range(3), ys,
                color=color, lw=style['lw'], ls=ls, alpha=style['alpha'],
                marker=mkr, markersize=style['ms'],
                markerfacecolor='none' if hollow else color,
                markeredgecolor=color, markeredgewidth=0.8,
                zorder=3)

        # Label at right end
        label = _size_label(model)
        if hollow:
            label += ' (dec)' if 'backbone' in grp else ' (think)'
        ax.text(2.06, ys[-1], label,
                va='center', fontsize=6, color=color, alpha=style['alpha'])


def plot_summary(ax, col, hh_means):
    """Plot group-mean summary panel."""
    # HH ceiling
    hh_ys = [hh_means.get(v, np.nan) for v in VARIANTS]
    hh_sems = hh.groupby('variant')[col].sem()

    ax.fill_between(range(3),
                    [y - 1.96 * hh_sems.get(v, 0) for y, v in zip(hh_ys, VARIANTS)],
                    [y + 1.96 * hh_sems.get(v, 0) for y, v in zip(hh_ys, VARIANTS)],
                    color=HH_COLOR, alpha=0.10)
    ax.plot(range(3), hh_ys, color=HH_COLOR, lw=1.8, ls='--',
            marker='*', markersize=9, label='Human–Human', zorder=4,
            markeredgecolor='white', markeredgewidth=0.5)

    GROUP_SHORT = {
        'VLM': 'VLM',
        'VLM backbone decoder': 'Backbone',
        'standalone LLM': 'SA-LLM',
        'standalone LLM (think)': 'SA-LLM (think)',
    }

    for grp in GROUP_ORDER:
        g_hm = hm[hm['subject_group_2'] == grp]
        if g_hm.empty:
            continue
        ys = [g_hm[g_hm['variant'] == v][col].mean() for v in VARIANTS]
        cis = [1.96 * g_hm[g_hm['variant'] == v][col].sem() for v in VARIANTS]

        color = GROUP_COLORS[grp]
        mkr = GROUP_MARKER.get(grp, 'o')
        hollow = GROUP_HOLLOW.get(grp, False)
        ls = GROUP_LINESTYLE.get(grp, '-')

        ax.fill_between(range(3),
                        [y - c for y, c in zip(ys, cis)],
                        [y + c for y, c in zip(ys, cis)],
                        color=color, alpha=0.10)
        ax.plot(range(3), ys,
                color=color, lw=2, ls=ls, marker=mkr, markersize=7,
                markerfacecolor='none' if hollow else color,
                markeredgecolor=color, markeredgewidth=1.0,
                zorder=3)
        ax.text(2.06, ys[-1], GROUP_SHORT.get(grp, grp),
                va='center', fontsize=7, color=color, fontweight='bold')


def build_legend_handles(row_panels):
    """Build legend handles for a row of panels."""
    handles = []
    seen_fam = set()

    for panel_def in row_panels:
        if panel_def.get('summary'):
            continue
        families = panel_def['families']
        groups = panel_def['groups']

        for fam in families:
            if fam in seen_fam:
                continue
            seen_fam.add(fam)
            color = MODEL_FAMILY_COLORS.get(fam, '#888')
            handles.append(mlines.Line2D(
                [], [], color=color, marker='o', ms=6, ls='-',
                markerfacecolor=color, label=fam))

    # Add base/derivative style indicators
    base_grp = row_panels[0]['groups'][0]  # VLM or standalone LLM
    deriv_grp = row_panels[0]['groups'][1] if len(row_panels[0]['groups']) > 1 else None

    base_label = 'VLM' if 'VLM' in base_grp else 'SA-LLM'
    deriv_label = 'Backbone' if 'backbone' in (deriv_grp or '') else 'Think'

    handles.append(mlines.Line2D(
        [], [], color='#666', marker='o', ms=6, ls='-',
        markerfacecolor='#666', label=f'{base_label} (solid)'))
    if deriv_grp:
        handles.append(mlines.Line2D(
            [], [], color='#666', marker='o', ms=6, ls=':',
            markerfacecolor='none', markeredgecolor='#666',
            label=f'{deriv_label} (dotted)'))

    handles.append(mlines.Line2D(
        [], [], color=HH_COLOR, ls='--', marker='*', ms=8,
        markeredgecolor='white', label='Human–Human'))

    return handles


# ── Generate figure ──────────────────────────────────────────────────────────
metric_name, col = METRICS[args.metric]
print(f'\n── {metric_name} ──')

hh_means = hh.groupby('variant')[col].mean()

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey='row')

for row_idx, row_panels in enumerate(PANELS):
    for col_idx, panel_def in enumerate(row_panels):
        ax = axes[row_idx, col_idx]

        if panel_def.get('summary'):
            plot_summary(ax, col, hh_means)
        else:
            plot_panel(ax, panel_def, col, hh_means)

        ax.set_title(panel_def['title'], fontsize=11, fontweight='bold')
        ax.set_xticks(range(3))
        ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=8)
        ax.set_xlim(-0.2, 2.55)

        if col_idx == 0:
            ax.set_ylabel(f'Mean {metric_name}', fontsize=10)

# Build legends per row
for row_idx, row_panels in enumerate(PANELS):
    handles = build_legend_handles(row_panels)
    # Place legend below the row
    axes[row_idx, 1].legend(
        handles=handles, fontsize=7.5,
        loc='upper center', bbox_to_anchor=(0.5, -0.18),
        ncol=len(handles), frameon=True,
        handletextpad=0.4, columnspacing=0.7)

plt.tight_layout(h_pad=3.0)

n_questions = pair_df[pair_df['pair_type'] == 'HH']['question_id'].nunique()
out_name = f'inst_blind_{args.metric}_family_vABC_q{n_questions}{SUFFIX}.png'
out_path = OUT_DIR / out_name
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_path}')

print(f'\nDone. Output → {OUT_DIR}')
