"""
Export per-model agreement-by-control-variant lineplots.

For each metric, one figure with a line per model (colour = model family,
marker = group shape), plus Human–Human ceiling and Model–Model baseline.

Output: figures/agreement/vABC_by_models/inst_blind_{metric}[_yesno].png

Run from repo root:
  conda run -n zero python figures/agreement_variants_by_models.py
  conda run -n zero python figures/agreement_variants_by_models.py --include_yesno
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
    MODEL_FAMILY, MODEL_FAMILY_COLORS, SCALE_STYLE_GROUPS, model_scale_style,
)
from helpers import clear_output_plots, get_exports_dir, load_pair_cache
from config import MODEL_LABEL_SHORT as LABEL_MAP

parser = argparse.ArgumentParser()
parser.add_argument('--include_yesno', action='store_true',
                    help='Extend pair cache with yes/no question pairs.')
parser.add_argument('--overwrite', action='store_true',
                    help='Delete existing plot files in the output folder before exporting.')
args = parser.parse_args()

OUT_DIR = ROOT / 'figures/agreement/vABC_by_models'
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)

EXPORTS = get_exports_dir(ROOT)
if args.include_yesno:
    pair_df = load_pair_cache(ROOT, include_yesno=True, verbose=True)
else:
    pair_df = pd.read_parquet(EXPORTS / 'pair_cache.parquet')

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
FAMILY_SHORT = {
    'Qwen3-VL': 'Qwen3-VL',
    'Qwen3': 'Qwen3',
    'LLaVA-1.5': 'LLaVA-1.5',
    'LLaVA-Mistral': 'LLaVA-M',
    'LLaVA-Vicuna': 'LLaVA-V',
    'InternVL': 'InternVL',
    'Mistral': 'Mistral',
    'Vicuna': 'Vicuna',
    'Phi': 'Phi',
    'Qwen2.5': 'Qwen2.5',
}


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

# All models present in HM pairs, sorted by group then name
models_df = (hm[['subject_2', 'subject_group_2']]
             .drop_duplicates()
             .rename(columns={'subject_2': 'model', 'subject_group_2': 'group'})
             .sort_values(['group', 'model']))
llm_reference_models = models_df[models_df['group'].isin(SCALE_STYLE_GROUPS)]['model'].tolist()


for label, (metric_name, col) in METRICS.items():
    print(f'\n── {metric_name} ({label}) ──')

    hh_means = hh.groupby('variant')[col].mean()

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
        hollow = GROUP_HOLLOW.get(grp, False)
        ls = GROUP_LINESTYLE.get(grp, '-')
        lbl   = LABEL_MAP.get(model, model)
        scale_style = (
            model_scale_style(model, llm_reference_models)
            if grp in SCALE_STYLE_GROUPS
            else {'alpha': 0.82, 'markersize': 6.0}
        )

        ax.plot(range(3), ys, color=color, lw=1.4, ls=ls, alpha=scale_style['alpha'],
                marker=mkr, markersize=scale_style['markersize'],
                markerfacecolor='none' if hollow else color,
                markeredgecolor=color, markeredgewidth=1.0,
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
                      marker='o', ms=6, ls='-', label=FAMILY_SHORT.get(f, f))
        for f in present_fams
    ]
    grp_handles = [
        mlines.Line2D([], [], color='gray',
                      marker=GROUP_MARKER.get(g, 'o'), ms=7,
                      ls=GROUP_LINESTYLE.get(g, '-'),
                      markerfacecolor='none' if GROUP_HOLLOW.get(g, False) else 'gray',
                      markeredgecolor='gray', markeredgewidth=1.0,
                      label=g.replace('standalone LLM', 'LLM')
                             .replace('VLM backbone decoder', 'Backbone'))
        for g in GROUP_ORDER
        if g in models_df['group'].values
    ]
    baseline_handles = [
        mlines.Line2D([], [], color=HH_COLOR, ls='--', marker='*', ms=8, label='Human–Human'),
    ]
    all_handles = fam_handles + grp_handles + baseline_handles
    ax.legend(handles=all_handles, fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True,
              handletextpad=0.5, columnspacing=0.8)

    plt.tight_layout()
    save(fig, f'inst_blind_{label}{SUFFIX}.png')

print('\nDone. All figures saved to:', OUT_DIR)
