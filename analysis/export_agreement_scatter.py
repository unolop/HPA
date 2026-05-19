"""
Export model-level human-model agreement scatter figures.

One figure per variant (A, B, C):
  inst_blind_v{V}_sbert_exact_models_q{N}_h{H}.png
      x = mean SBERT cosine similarity to humans
      y = mean exact-match agreement with humans
      one point per model, coloured by group

Scale plots (human alignment vs. model size) live in export_scale_plots.py
and are saved to figures/agreement_scale/.

Before plotting, auto-detects models with complete inst_blind JSONL data
not yet in pair_cache and computes their HM pairs on the fly.

Run from repo root:
  conda run -n zero python analysis/export_agreement_scatter.py
  conda run -n zero python analysis/export_agreement_scatter.py --include_yesno
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.constants import GROUP_COLORS, GROUP_ORDER
from build_pair_cache import build_pair_cache
from config import MODEL_LABEL_SHORT as LABEL_MAP, extend_pair_cache_with_yesno

parser = argparse.ArgumentParser()
parser.add_argument('--include_yesno', action='store_true',
                    help='Extend pair cache with yes/no question pairs.')
args = parser.parse_args()

EXPORTS = ROOT / 'analysis/session2/exports'
OUT_DIR = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/agreement_scatter'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

VARIANTS      = ['C', 'B', 'A']
VARIANT_LABEL = {'C': 'Original (C)', 'B': 'Weaker (B)', 'A': 'Pronominalized (A)'}


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [agreement_scatter] {name}')


# ── Manual layout hints (model-specific, variant-independent) ─────────────────
JITTER = {
    'Qwen3-8B (think)':   (0, +0.003),
    'Qwen3-8B':           (0, -0.003),
    'Qwen3-4B (think)':   (0, +0.003),
    'Qwen3-4B':           (0, -0.003),
    'Qwen3-0.6B (think)': (0, +0.003),
    'Qwen3-0.6B':         (0, -0.003),
}

LABEL_OFFSETS = {
    'LLaVA-1.5 (LM)':    (-0.060,  0.004),
    'LLaVA-Mistral (LM)': ( 0.004,  0.005),
    'LLaVA-Vicuna (LM)':  ( 0.004, -0.010),
    'InternVL-8B':         ( 0.004,  0.000),
    'LLaVA-1.5-7B':      (-0.040,  0.006),
    'LLaVA-Mistral':     (-0.040,  0.000),
    'LLaVA-Vicuna':      (-0.040, -0.007),
    'Qwen3-VL-8B':       ( 0.004,  0.004),
    'InternVL-2B':       ( 0.004, -0.007),
    'InternVL-1B':       (-0.050, -0.010),
    'Mistral-7B':        (-0.005,  0.006),
    'Vicuna-13B':        (-0.005, -0.007),
    'Phi-3.5-mini':      ( 0.004,  0.004),
    'Qwen2.5-7B':        ( 0.004,  0.003),
    'Qwen3-32B':         (-0.050,  0.000),
    'Qwen3-32B (think)': ( 0.004,  0.000),
}

# ── Load / update pair_cache ──────────────────────────────────────────────────
print('Building / updating pair_cache…')
pair_df = build_pair_cache(ROOT, EXPORTS, verbose=True)
if args.include_yesno:
    print('Extending pair_cache with yes/no question pairs…')
    pair_df = extend_pair_cache_with_yesno(pair_df, EXPORTS)

n_questions = pair_df[pair_df['pair_type'] == 'HH']['question_id'].nunique()
n_humans    = pair_df[pair_df['pair_type'] == 'HH']['subject_1'].nunique()
YESNO_TAG   = '_yesno' if args.include_yesno else ''
SUFFIX      = f'_q{n_questions}_h{n_humans}{YESNO_TAG}'

print(f'  questions: {n_questions}  |  humans: {n_humans}')
print(f'  variants:  {sorted(pair_df["variant"].unique())}')

# ── Per-variant scatter ───────────────────────────────────────────────────────
for var in VARIANTS:
    print(f'\n── Variant {var} ({VARIANT_LABEL[var]}) ──')

    hm = pair_df[(pair_df['variant'] == var) & (pair_df['pair_type'] == 'HM')]
    hh = pair_df[(pair_df['variant'] == var) & (pair_df['pair_type'] == 'HH')]

    per_model = (hm.groupby(['subject_2', 'subject_group_2'])
                   .agg(sbert=('sbert_score', 'mean'),
                        exact=('exact_score', 'mean'))
                   .reset_index()
                   .rename(columns={'subject_2': 'model',
                                    'subject_group_2': 'group'}))

    hh_sbert = hh['sbert_score'].mean()
    hh_exact = hh['exact_score'].mean()
    print(f'  models: {len(per_model)}  |  '
          f'HH sbert={hh_sbert:.3f}, exact={hh_exact:.3f}')

    fig, ax = plt.subplots(figsize=(8, 6.5))

    ax.scatter([hh_sbert], [hh_exact], marker='*', s=220,
               color='#1565C0', zorder=5, label='Human–Human ceiling',
               linewidths=0)
    ax.axvline(hh_sbert, color='#1565C0', lw=0.9, ls='--', alpha=0.4)
    ax.axhline(hh_exact, color='#1565C0', lw=0.9, ls=':', alpha=0.4)
    ax.text(hh_sbert + 0.001, hh_exact + 0.003, 'HH',
            fontsize=7, color='#1565C0', alpha=0.7)

    for grp in GROUP_ORDER:
        sub = per_model[per_model['group'] == grp]
        if sub.empty:
            continue
        xs = sub['sbert'] + sub['model'].map(
            lambda m: JITTER.get(m, (0, 0))[0])
        ys = sub['exact'] + sub['model'].map(
            lambda m: JITTER.get(m, (0, 0))[1])
        ax.scatter(xs, ys, color=GROUP_COLORS[grp], s=65, zorder=3,
                   label=grp, edgecolors='white', linewidths=0.7)

    for _, row in per_model.iterrows():
        label = LABEL_MAP.get(row['model'], row['model'])
        jx, jy = JITTER.get(row['model'], (0, 0))
        px, py = row['sbert'] + jx, row['exact'] + jy
        dx, dy = LABEL_OFFSETS.get(row['model'], (0.004, 0.003))
        ax.annotate(label, xy=(px, py), xytext=(px + dx, py + dy),
                    fontsize=6.5, color='#333333',
                    ha='left' if dx >= 0 else 'right', va='center')

    ax.set_xlabel('Mean SBERT cosine similarity to humans', fontsize=11)
    ax.set_ylabel('Mean exact-match agreement with humans', fontsize=11)
    ax.set_title(f'Human–Model Answer Agreement\n'
                 f'({VARIANT_LABEL[var]}, inst_blind)', fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, frameon=True,
              loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    plt.tight_layout()
    save(fig, f'inst_blind_v{var}_sbert_exact_models{SUFFIX}.png')

print(f'\nDone. Saved to: {OUT_DIR}')
