"""
Export per-model agreement-by-control-variant lineplots.

For each metric, one figure with a line per model (colour = model family,
marker = group shape), plus Human–Human ceiling and Model–Model baseline.

Output: figures/agreement_variants_by_models/inst_blind_vABC_{metric}_models[_yesno].png

Run from repo root:
  conda run -n zero python analysis/export_agreement_variants_by_models.py
  conda run -n zero python analysis/export_agreement_variants_by_models.py --include_yesno
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

from utils.constants import (
    GROUP_COLORS, GROUP_ORDER, GROUP_MARKER,
    MODEL_FAMILY, MODEL_FAMILY_COLORS,
)
from build_pair_cache import build_pair_cache
from config import MODEL_LABEL_SHORT as LABEL_MAP, extend_pair_cache_with_yesno

parser = argparse.ArgumentParser()
parser.add_argument('--include_yesno', action='store_true',
                    help='Extend pair cache with yes/no question pairs.')
args = parser.parse_args()

OUT_DIR = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/agreement_variants_by_models'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORTS = ROOT / 'analysis/session2/exports'
pair_df = build_pair_cache(ROOT, EXPORTS, verbose=True)
if args.include_yesno:
    print('Extending pair_cache with yes/no question pairs…')
    pair_df = extend_pair_cache_with_yesno(pair_df, EXPORTS)

SUFFIX = '_yesno' if args.include_yesno else ''

# ── Constants ─────────────────────────────────────────────────────────────────
VARIANTS   = ['C', 'B', 'A']
VAR_LABELS = {'C': 'Original (C)', 'B': 'Weaker (B)', 'A': 'Pronoun. (A)'}

METRICS = {
    'sbert':     ('SBERT cosine',  'sbert_score'),
    'simcse':    ('SimCSE cosine', 'simcse_score'),
    'bertscore': ('BERTScore F1',  'bertscore_f1'),
    'chrf':      ('chrF',          'chrf_score'),
    'rouge1':    ('ROUGE-1',       'rouge1_score'),
    'jaccard':   ('Token Jaccard', 'jaccard_score'),
    'exact':     ('Exact match',   'exact_score'),
}

HH_COLOR = '#1565C0'
MM_COLOR = '#757575'


plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [agreement_variants_by_models] {name}')


# ── Pre-split pair_df ─────────────────────────────────────────────────────────
hm = pair_df[pair_df['pair_type'] == 'HM'].copy()
hh = pair_df[pair_df['pair_type'] == 'HH'].copy()
mm = pair_df[pair_df['pair_type'] == 'MM'].copy()

# All models present in HM pairs, sorted by group then name
models_df = (hm[['subject_2', 'subject_group_2']]
             .drop_duplicates()
             .rename(columns={'subject_2': 'model', 'subject_group_2': 'group'})
             .sort_values(['group', 'model']))


for label, (metric_name, col) in METRICS.items():
    print(f'\n── {metric_name} ({label}) ──')

    hh_means = hh.groupby('variant')[col].mean()
    mm_means = mm.groupby('variant')[col].mean()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # ── Per-model lines ───────────────────────────────────────────────────────
    for _, row in models_df.iterrows():
        model = row['model']
        grp   = row['group']
        sub   = hm[hm['subject_2'] == model]
        if sub.empty:
            continue
        ys = [sub[sub['variant'] == v][col].mean() for v in VARIANTS]
        if all(np.isnan(y) for y in ys):
            continue

        fam   = MODEL_FAMILY.get(model, 'Unknown')
        color = MODEL_FAMILY_COLORS.get(fam, '#888888')
        mkr   = GROUP_MARKER.get(grp, 'o')
        lbl   = LABEL_MAP.get(model, model)

        ax.plot(range(3), ys, color=color, lw=1.4, ls='-', alpha=0.82,
                marker=mkr, markersize=6,
                markeredgecolor='white', markeredgewidth=0.5,
                zorder=2)
        # Label at the right end
        ax.text(2.08, ys[-1], lbl,
                va='center', fontsize=6.5, color=color)

    # ── HH ceiling ───────────────────────────────────────────────────────────
    hh_ys = [hh_means.get(v, np.nan) for v in VARIANTS]
    ax.plot(range(3), hh_ys, color=HH_COLOR, lw=2, ls='--',
            marker='*', markersize=10, label='Human–Human',
            markeredgecolor='white', markeredgewidth=0.6, zorder=4)
    ax.text(2.08, hh_ys[-1], f'{hh_ys[-1]:.3f}',
            va='center', fontsize=7, color=HH_COLOR, fontweight='bold')

    # ── MM baseline ──────────────────────────────────────────────────────────
    mm_ys = [mm_means.get(v, np.nan) for v in VARIANTS]
    ax.plot(range(3), mm_ys, color=MM_COLOR, lw=1.8, ls=':',
            marker='D', markersize=6, label='Model–Model',
            markeredgecolor='white', markeredgewidth=0.6, zorder=4)
    ax.text(2.08, mm_ys[-1], f'{mm_ys[-1]:.3f}',
            va='center', fontsize=7, color=MM_COLOR, fontweight='bold')

    ax.set_xticks(range(3))
    ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=9)
    ax.set_ylabel(f'Mean {metric_name}', fontsize=10)
    ax.set_xlim(-0.3, 2.7)

    # ── Legend: families (color) + groups (marker shape) + baselines ─────────
    # Collect unique families present
    present_fams = sorted(set(
        MODEL_FAMILY.get(m, '') for m in models_df['model']
    ) - {''})
    fam_handles = [
        mlines.Line2D([], [], color=MODEL_FAMILY_COLORS.get(f, '#888'),
                      marker='o', ms=6, ls='-', label=f)
        for f in present_fams
    ]
    grp_handles = [
        mlines.Line2D([], [], color='gray',
                      marker=GROUP_MARKER.get(g, 'o'), ms=7, ls='none',
                      label=g.replace('standalone LLM', 'SA-LLM')
                             .replace('VLM backbone decoder', 'Backbone'))
        for g in GROUP_ORDER
        if g in models_df['group'].values
    ]
    baseline_handles = [
        mlines.Line2D([], [], color=HH_COLOR, ls='--', marker='*', ms=8, label='Human–Human'),
        mlines.Line2D([], [], color=MM_COLOR, ls=':',  marker='D', ms=6, label='Model–Model'),
    ]
    all_handles = fam_handles + grp_handles + baseline_handles
    ax.legend(handles=all_handles, fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=True)

    plt.tight_layout()
    save(fig, f'inst_blind_vABC_{label}_models{SUFFIX}.png')

print('\nDone. All figures saved to:', OUT_DIR)
