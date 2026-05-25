"""
Export entity-type analysis figures for the human-study question subset.

Figures are saved to figures/entity_analysis/ with _m{MIN_ANSWERS} suffix.

Sections
--------
1. SBERT HM semantic agreement by entity type per model group
2. Instruction sensitivity (response change rate) by entity type

Run from repo root:
  conda run -n zero python figures/entity_analysis.py
  conda run -n zero python figures/entity_analysis.py --min_answers 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from utils.constants import GROUP_COLORS, VARIANT_ORDER
from helpers import clear_output_plots, load_human_subset, load_pair_cache, read_response_exports

# ─────────────────────────────────────────────────────────────────────────────
# CLI argument
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--min_answers', type=int, default=348,
                    help='Minimum answers to qualify a participant (default: 348)')
parser.add_argument('--overwrite', action='store_true',
                    help='Delete existing plot files in the output folder before exporting.')
args = parser.parse_args()
MIN_ANSWERS = args.min_answers
SUFFIX = f'_m{MIN_ANSWERS}'

OUT_DIR = ROOT / 'figures/entity_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)

plt.rcParams.update({
    'font.family':   'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})
sns.set_style('whitegrid')

# Groups shown in all multi-group plots (drop 'think' to reduce clutter unless needed)
GROUPS_SHOW = ['VLM', 'VLM backbone decoder', 'standalone LLM']
GROUP_PALETTE = {g: GROUP_COLORS[g] for g in GROUPS_SHOW}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Get common_qids via load_human_data (respects min_answers)
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nLoading human data (min_answers={MIN_ANSWERS})…')
_, common_qids, _, _ = load_human_subset(ROOT, min_answers=MIN_ANSWERS,
                                        translate=False, verbose=True)
print(f'Common question IDs: {len(common_qids)}')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load pre-processed exports and filter to common_qids
# ─────────────────────────────────────────────────────────────────────────────
print('\nLoading exports…')
exports = read_response_exports(ROOT, subset_qids=common_qids)
human       = exports['human']
model_blind = exports['model_blind']
model_inst  = exports['model_inst_blind']
pair_df     = load_pair_cache(ROOT, subset_qids=common_qids, verbose=True)

print(f'  human rows: {len(human)} | model_blind: {len(model_blind)} '
      f'| model_inst: {len(model_inst)} | pairs: {len(pair_df)}')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def ent_order(df):
    """Sort entity types by descending frequency in human study variant C."""
    counts = (human[human['variant'] == 'C'][['question_id', 'ent']]
              .drop_duplicates()['ent']
              .value_counts()
              .index.tolist())
    cat = pd.Categorical(df['ent'], categories=counts, ordered=True)
    return df.assign(ent=cat).sort_values('ent')


def save(fig, name):
    path = OUT_DIR / (name + SUFFIX + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [entity_analysis] {path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — SBERT HM semantic agreement by entity type  [variant C]
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig_entity_sbert_groups ──')
hm = pair_df[(pair_df['pair_type'] == 'HM') & (pair_df['variant'].isin(VARIANT_ORDER))].copy()
entity_sem = (hm.groupby(['subject_group_2', 'ent', 'variant'], dropna=False)
               .agg(sbert=('sbert_score_clip', 'mean'),
                    n_q=('question_id', 'nunique'))
               .reset_index()
               .rename(columns={'subject_group_2': 'model_group'}))

sem_c = ent_order(
    entity_sem[(entity_sem['variant'] == 'C')
               & (entity_sem['model_group'].isin(GROUPS_SHOW))
               & (entity_sem['n_q'] >= 3)]
)

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=sem_c, x='ent', y='sbert', hue='model_group',
            palette=GROUP_PALETTE, ax=ax)
ax.set_xlabel('Entity type', fontsize=10)
ax.set_ylabel('Mean HM SBERT cosine', fontsize=10)
ax.legend(title=None, frameon=True, loc='upper right', fontsize=8)
for tick in ax.get_xticklabels():
    tick.set_rotation(35); tick.set_ha('right')
plt.tight_layout()
save(fig, 'fig_entity_sbert_groups')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Instruction sensitivity (response change rate) by entity type [C]
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig_entity_instruction_delta ──')
resp_b = model_blind[['question_id', 'ent', 'op', 'variant',
                       'model', 'model_group', 'response']].copy()
resp_i = model_inst[['question_id', 'ent', 'op', 'variant',
                      'model', 'model_group', 'response']].copy()
resp_m = resp_b.merge(resp_i,
                      on=['question_id', 'ent', 'op', 'variant', 'model', 'model_group'],
                      suffixes=('_blind', '_inst'))
resp_m['b_norm'] = resp_m['response_blind'].fillna('').astype(str).str.strip().str.lower()
resp_m['i_norm'] = resp_m['response_inst'].fillna('').astype(str).str.strip().str.lower()
resp_m['changed'] = (resp_m['b_norm'] != resp_m['i_norm']).astype(float)

change_ent = (resp_m.groupby(['model_group', 'ent', 'variant'], dropna=False)['changed']
              .mean().reset_index())

inst_c = ent_order(
    change_ent[(change_ent['variant'] == 'C')
               & (change_ent['model_group'].isin(GROUPS_SHOW))]
)

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=inst_c, x='ent', y='changed', hue='model_group',
            palette=GROUP_PALETTE, ax=ax)
ax.set_xlabel('Entity type', fontsize=10)
ax.set_ylabel('Response change rate (blind → inst_blind)', fontsize=10)
ax.legend(title=None, frameon=True, loc='upper right', fontsize=8)
for tick in ax.get_xticklabels():
    tick.set_rotation(35); tick.set_ha('right')
plt.tight_layout()
save(fig, 'fig_entity_instruction_delta')

print(f'\nDone. Outputs in: {OUT_DIR}')
