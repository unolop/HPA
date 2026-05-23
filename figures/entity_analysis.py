"""
Export entity-type analysis figures for the human-study question subset.

Figures are saved to figures/entity_analysis/ with _m{MIN_ANSWERS} suffix.

Sections
--------
1. Entity distribution (n questions per entity type)
2. Pearson r (human vs model accuracy) by entity type per model group

3. SBERT HM semantic agreement by entity type per model group
4. Control-variant degradation (C→A drop) by entity type
5. Instruction sensitivity (response change rate) by entity type

Run from repo root:
  conda run -n zero python figures/entity_analysis.py
  conda run -n zero python figures/entity_analysis.py --min_answers 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from utils.constants import GROUP_COLORS, VARIANT_ORDER
from helpers import load_human_subset, load_pair_cache, read_response_exports

# ─────────────────────────────────────────────────────────────────────────────
# CLI argument
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--min_answers', type=int, default=348,
                    help='Minimum answers to qualify a participant (default: 348)')
args = parser.parse_args()
MIN_ANSWERS = args.min_answers
SUFFIX = f'_m{MIN_ANSWERS}'

OUT_DIR = ROOT / 'figures/entity_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
model_ctrl  = exports['model_control']
pair_df     = load_pair_cache(ROOT, subset_qids=common_qids, verbose=True)

print(f'  human rows: {len(human)} | model_blind: {len(model_blind)} '
      f'| model_inst: {len(model_inst)} | pairs: {len(pair_df)}')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pearson_safe(x, y):
    x, y = np.asarray(list(x), float), np.asarray(list(y), float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    if np.allclose(x, x.mean()) or np.allclose(y, y.mean()):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def summarize_q(df, who='model'):
    gc = ['question_id', 'ent', 'op', 'variant']
    if who == 'model':
        gc += ['model_group', 'model']
    return (df.groupby(gc, dropna=False)['accuracy']
              .mean()
              .reset_index()
              .rename(columns={'accuracy': f'{who}_acc'}))


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
# Pre-compute question-level accuracy tables
# ─────────────────────────────────────────────────────────────────────────────
human_q = summarize_q(human, 'human')
blind_q  = summarize_q(model_blind, 'model')
inst_q   = summarize_q(model_inst, 'model')
ctrl_q   = summarize_q(model_ctrl, 'model')

ent_counts = (human[human['variant'] == 'C'][['question_id', 'ent']]
              .drop_duplicates()['ent']
              .value_counts()
              .rename_axis('ent')
              .reset_index(name='n_questions'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Entity type distribution
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig_entity_dist ──')
ec = ent_counts.sort_values('n_questions', ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(ec['ent'].str.capitalize(), ec['n_questions'],
               color='#5C6BC0', edgecolor='white', height=0.65)
for bar, n in zip(bars, ec['n_questions']):
    ax.text(n + 0.3, bar.get_y() + bar.get_height() / 2,
            str(n), va='center', fontsize=9)
ax.set_xlabel(f'Number of questions (human study, n={len(common_qids)} total)', fontsize=10)
ax.set_xlim(0, ec['n_questions'].max() * 1.15)
plt.tight_layout()
save(fig, 'fig_entity_dist')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Pearson r (human vs model accuracy) by entity type  [variant C]
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig_entity_corr_groups ──')
merged_inst = inst_q.merge(
    human_q[['question_id', 'ent', 'op', 'variant', 'human_acc']],
    on=['question_id', 'ent', 'op', 'variant'], how='left')

corr_rows = []
for (mg, ent, var), sub in merged_inst.groupby(['model_group', 'ent', 'variant'],
                                                dropna=False):
    corr_rows.append({
        'model_group': mg, 'ent': ent, 'variant': var,
        'n_questions': sub['question_id'].nunique(),
        'pearson_r': pearson_safe(sub['human_acc'], sub['model_acc']),
    })
entity_corr = pd.DataFrame(corr_rows)

plot_c = ent_order(
    entity_corr[(entity_corr['variant'] == 'C')
                & (entity_corr['model_group'].isin(GROUPS_SHOW))
                & (entity_corr['n_questions'] >= 3)]
)

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=plot_c, x='ent', y='pearson_r', hue='model_group',
            palette=GROUP_PALETTE, ax=ax)
ax.axhline(0, color='gray', lw=1, alpha=0.6)
ax.set_xlabel('Entity type', fontsize=10)
ax.set_ylabel('Pearson r: human vs model accuracy', fontsize=10)
ax.legend(title=None, frameon=True, loc='upper right', fontsize=8)
for tick in ax.get_xticklabels():
    tick.set_rotation(35); tick.set_ha('right')
plt.tight_layout()
save(fig, 'fig_entity_corr_groups')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — SBERT HM semantic agreement by entity type  [variant C]
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
# Fig 4 — Control-variant degradation (C→A drop) by entity type
# Uses blind condition (all model groups have blind data; control is VLM-only)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig_entity_degradation ──')
blind_ent = (blind_q.groupby(['model_group', 'ent', 'variant'], dropna=False)['model_acc']
             .mean().reset_index())
blind_piv = blind_ent.pivot_table(index=['model_group', 'ent'],
                                  columns='variant', values='model_acc')
for col in ['A', 'B', 'C']:
    if col not in blind_piv.columns:
        blind_piv[col] = np.nan
blind_piv = blind_piv.reset_index()
blind_piv['drop_C_to_A'] = blind_piv['C'] - blind_piv['A']

deg_plot = ent_order(
    blind_piv[blind_piv['model_group'].isin(GROUPS_SHOW)]
)

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=deg_plot, x='ent', y='drop_C_to_A', hue='model_group',
            palette=GROUP_PALETTE, ax=ax)
ax.axhline(0, color='gray', lw=1, alpha=0.6)
ax.set_xlabel('Entity type', fontsize=10)
ax.set_ylabel('Accuracy drop: C − A (higher = more degradation)', fontsize=10)
ax.legend(title=None, frameon=True, loc='upper right', fontsize=8)
for tick in ax.get_xticklabels():
    tick.set_rotation(35); tick.set_ha('right')
plt.tight_layout()
save(fig, 'fig_entity_degradation')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Instruction sensitivity (response change rate) by entity type [C]
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


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Combined paper figure: entity dist (left) + Pearson r VLM (right)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig_hm_alignment (2-panel: entity dist + Pearson r VLM) ──')

ENT_COLORS = {
    'object':  '#e74c3c', 'person': '#3498db', 'animal': '#f39c12',
    'food':    '#27ae60', 'other':  '#9b59b6', 'place':  '#1abc9c',
    'product': '#e67e22', 'text':   '#95a5a6', 'vehicle':'#2c3e50',
}

# VLM-only Pearson r per entity type (inst_blind, variant C)
# Reuse merged_inst (inst_q joined with human_q, already computed above)
vlm_inst_c = merged_inst[(merged_inst['model_group'] == 'VLM') &
                          (merged_inst['variant'] == 'C')]
ent_r_vlm = (vlm_inst_c.groupby('ent')
             .apply(lambda g: pearson_safe(g['human_acc'], g['model_acc'])
                    if g['question_id'].nunique() >= 5 else np.nan,
                    include_groups=False)
             .dropna()
             .sort_values(ascending=False))
ent_n = ent_counts.set_index('ent')['n_questions']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: entity distribution
ax = axes[0]
ec = ent_counts.sort_values('n_questions', ascending=True)
bar_colors = [ENT_COLORS.get(e, '#aaaaaa') for e in ec['ent']]
ax.barh(ec['ent'].str.capitalize(), ec['n_questions'],
        color=bar_colors, edgecolor='white', height=0.65)
for i, (e, n) in enumerate(zip(ec['ent'], ec['n_questions'])):
    ax.text(n + 0.3, i, str(n), va='center', fontsize=9.5, color='#333333')
ax.set_xlabel(f'Number of questions (out of {len(common_qids)})', fontsize=10)
ax.set_title('Question Distribution by Entity Type\n(human study, variant C)', fontsize=11)
ax.set_xlim(0, ec['n_questions'].max() * 1.3)
ax.set_ylabel('Entity type', fontsize=10)

# Right panel: Pearson r by entity type (VLM, inst_blind)
ax = axes[1]
ents = ent_r_vlm.index.tolist()
vals = ent_r_vlm.values
ax.barh([e.capitalize() for e in ents], vals,
        color=[ENT_COLORS.get(e, '#aaaaaa') for e in ents],
        edgecolor='white', height=0.65, alpha=0.85)
for v, e in zip(vals, ents):
    offset = 0.03 if v >= 0 else -0.03
    ax.text(v + offset, ents.index(e), f'r = {v:.2f}',
            va='center', ha='left' if v >= 0 else 'right', fontsize=9.5)
ax.set_xlim(-1.0, 1.4)
ax.set_xlabel('Pearson r (human vs. VLM accuracy, per question)', fontsize=10)
ax.set_title('Human–Model Difficulty Correlation\nby Entity Type (VLM, inst_blind)', fontsize=11)
ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_ylabel('')

plt.tight_layout()
save(fig, 'fig_hm_alignment')

print(f'\nDone. Outputs in: {OUT_DIR}')
print(f'Suffix used: {SUFFIX}  (common_qids={len(common_qids)}, '
      f'entities={ent_counts["ent"].nunique()})')
