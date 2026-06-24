"""
Compare human–model agreement between blind and inst_blind conditions.

Shows whether adding the "no image" instruction brings model answers
closer to or further from human answers, measured by SBERT cosine.

Generates:
  figures/agreement_instruction_effect/
    instruction_effect_by_model.png      — per-model HM agreement: blind vs inst_blind
    instruction_effect_by_group.png      — per-group mean with error bars
    instruction_effect_per_question.png  — per-question scatter + delta histogram

Run from repo root:
  conda run -n zero python figures/agreement_instruction_effect.py
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))
from helpers import load_pair_cache, save_fig, clear_output_plots
from config import MODEL_GROUP, MODELS_ALL

from utils.constants import GROUP_COLORS, GROUP_ORDER

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--variant', default='C', help='Variant to compare (default: C)')
args = parser.parse_args()

OUT_DIR = ROOT / 'figures' / 'agreement_instruction_effect'
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)

VARIANT = args.variant
SCORE_COL = 'sbert_score_clip'
METRIC_LABEL = 'SBERT cosine (clipped)'
INCLUDE_YESNO = True

# ── Load both pair caches ────────────────────────────────────────────────────
print('Loading inst_blind pair cache…')
pc_inst = load_pair_cache(ROOT, condition='inst_blind', include_yesno=INCLUDE_YESNO, verbose=False)
print(f'  inst_blind: {len(pc_inst):,} pairs')

print('Loading blind pair cache…')
pc_blind = load_pair_cache(ROOT, condition='blind', include_yesno=INCLUDE_YESNO, verbose=False)
print(f'  blind: {len(pc_blind):,} pairs')

# ── Filter to HM pairs, target variant ───────────────────────────────────────
def get_hm_scores(pc, variant, score_col):
    """Extract per-(model, question) HM agreement from pair cache."""
    hm = pc[(pc['pair_type'] == 'HM') & (pc['variant'] == variant)].copy()
    # subject_1 = human, subject_2 = model (convention in build_pair_cache)
    return (
        hm.groupby(['question_id', 'subject_2'])[score_col]
        .mean()
        .rename('hm_score')
        .reset_index()
        .rename(columns={'subject_2': 'model'})
    )

hm_inst = get_hm_scores(pc_inst, VARIANT, SCORE_COL)
hm_blind = get_hm_scores(pc_blind, VARIANT, SCORE_COL)

# Merge on (model, question_id)
merged = hm_inst.merge(
    hm_blind, on=['model', 'question_id'], suffixes=('_inst', '_blind')
)
merged['delta'] = merged['hm_score_inst'] - merged['hm_score_blind']
merged['model_group'] = merged['model'].map(MODEL_GROUP)

# Only keep models present in both conditions
common_models = sorted(merged['model'].unique())
print(f'\nModels in both conditions: {len(common_models)}')

# ── HH reference (same for both conditions — humans are always inst_blind) ──
hh_inst = pc_inst[(pc_inst['pair_type'] == 'HH') & (pc_inst['variant'] == VARIANT)]
hh_mean = hh_inst[SCORE_COL].mean()
print(f'HH reference (inst_blind, variant {VARIANT}): {hh_mean:.3f}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Per-model HM agreement — blind vs inst_blind
# ══════════════════════════════════════════════════════════════════════════════
model_agg = (
    merged.groupby(['model', 'model_group'])['hm_score_inst']
    .mean().rename('inst')
    .reset_index()
)
model_agg['blind'] = (
    merged.groupby('model')['hm_score_blind'].mean().values
)
model_agg['delta'] = model_agg['inst'] - model_agg['blind']
model_agg = model_agg.sort_values('delta', ascending=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(model_agg) * 0.32)))
y = np.arange(len(model_agg))
bar_h = 0.35

colors_blind = [GROUP_COLORS.get(g, '#888') for g in model_agg['model_group']]
colors_inst = [GROUP_COLORS.get(g, '#888') for g in model_agg['model_group']]

ax.barh(y + bar_h / 2, model_agg['blind'], bar_h,
        color=colors_blind, alpha=0.4, edgecolor='white', label='blind')
ax.barh(y - bar_h / 2, model_agg['inst'], bar_h,
        color=colors_inst, alpha=0.85, edgecolor='white', label='inst_blind')

# Delta annotations
for i, (_, row) in enumerate(model_agg.iterrows()):
    sign = '+' if row['delta'] >= 0 else ''
    ax.text(max(row['inst'], row['blind']) + 0.005, i,
            f'{sign}{row["delta"]:.3f}', va='center', fontsize=7,
            color='#2E7D32' if row['delta'] > 0 else '#C62828')

ax.axvline(hh_mean, color='red', ls='--', lw=1.5, alpha=0.7, label=f'HH = {hh_mean:.3f}')
ax.set_yticks(y)
ax.set_yticklabels(model_agg['model'], fontsize=8)
ax.set_xlabel(f'Mean HM {METRIC_LABEL}')
ax.set_title(f'Instruction effect on human–model agreement (variant {VARIANT})\n'
             f'faded = blind, solid = inst_blind, Δ = inst − blind')
ax.legend(loc='lower right', fontsize=8)
ax.set_xlim(0, ax.get_xlim()[1] + 0.06)
plt.tight_layout()
save_fig(fig, OUT_DIR, f'instruction_effect_by_model_v{VARIANT}.png')
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Per-group mean with 95% CI
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

group_data = []
for group in GROUP_ORDER:
    sub = merged[merged['model_group'] == group]
    if sub.empty:
        continue
    # Question-level means (pool all models in group per question, then average)
    q_inst = sub.groupby('question_id')['hm_score_inst'].mean()
    q_blind = sub.groupby('question_id')['hm_score_blind'].mean()
    group_data.append({
        'group': group,
        'inst_mean': q_inst.mean(),
        'inst_sem': q_inst.sem(),
        'blind_mean': q_blind.mean(),
        'blind_sem': q_blind.sem(),
        'delta_mean': (q_inst - q_blind).mean(),
        'delta_sem': (q_inst - q_blind).sem(),
        'n_models': sub['model'].nunique(),
        'n_questions': sub['question_id'].nunique(),
    })

gdf = pd.DataFrame(group_data)
x = np.arange(len(gdf))
width = 0.35

bars_blind = ax.bar(x - width / 2, gdf['blind_mean'], width,
                    yerr=1.96 * gdf['blind_sem'],
                    color=[GROUP_COLORS.get(g, '#888') for g in gdf['group']],
                    alpha=0.4, edgecolor='white', capsize=3, label='blind')
bars_inst = ax.bar(x + width / 2, gdf['inst_mean'], width,
                   yerr=1.96 * gdf['inst_sem'],
                   color=[GROUP_COLORS.get(g, '#888') for g in gdf['group']],
                   alpha=0.85, edgecolor='white', capsize=3, label='inst_blind')

# Delta annotations
for i, row in gdf.iterrows():
    sign = '+' if row['delta_mean'] >= 0 else ''
    ypos = max(row['inst_mean'], row['blind_mean']) + 2 * max(row['inst_sem'], row['blind_sem']) + 0.01
    ax.text(i, ypos, f'Δ={sign}{row["delta_mean"]:.3f}', ha='center', fontsize=9,
            fontweight='bold', color='#2E7D32' if row['delta_mean'] > 0 else '#C62828')

ax.axhline(hh_mean, color='red', ls='--', lw=1.5, alpha=0.7, label=f'HH = {hh_mean:.3f}')
ax.set_xticks(x)
ax.set_xticklabels([f'{g}\n(n={int(row["n_models"])} models)' for g, (_, row) in
                    zip(gdf['group'], gdf.iterrows())], fontsize=8)
ax.set_ylabel(f'Mean HM {METRIC_LABEL}')
ax.set_title(f'Instruction effect on human–model agreement by model group\n'
             f'(variant {VARIANT}, ±95% CI)')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
save_fig(fig, OUT_DIR, f'instruction_effect_by_group_v{VARIANT}.png')
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Per-question analysis — scatter + delta histogram
# ══════════════════════════════════════════════════════════════════════════════
# Average across all models for each question
q_agg = merged.groupby('question_id').agg(
    inst=('hm_score_inst', 'mean'),
    blind=('hm_score_blind', 'mean'),
).assign(delta=lambda d: d['inst'] - d['blind'])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: scatter blind vs inst_blind per question
ax = axes[0]
colors = ['#2E7D32' if d > 0 else '#C62828' for d in q_agg['delta']]
ax.scatter(q_agg['blind'], q_agg['inst'], c=colors, s=25, alpha=0.6, edgecolors='white', linewidth=0.3)
lims = [min(q_agg[['blind', 'inst']].min()), max(q_agg[['blind', 'inst']].max())]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.4, label='y = x (no change)')
ax.set_xlabel(f'HM agreement — blind ({METRIC_LABEL})')
ax.set_ylabel(f'HM agreement — inst_blind ({METRIC_LABEL})')
n_up = (q_agg['delta'] > 0).sum()
n_down = (q_agg['delta'] < 0).sum()
ax.set_title(f'Per-question HM agreement: blind vs inst_blind\n'
             f'(variant {VARIANT}, {len(q_agg)} questions — '
             f'{n_up} ↑, {n_down} ↓ with instruction)')
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# Right: delta histogram
ax = axes[1]
ax.hist(q_agg['delta'], bins=30, color='#1565C0', alpha=0.7, edgecolor='white')
ax.axvline(0, color='black', ls='-', lw=1)
ax.axvline(q_agg['delta'].mean(), color='red', ls='--', lw=2,
           label=f'mean Δ = {q_agg["delta"].mean():+.3f}')
ax.axvline(q_agg['delta'].median(), color='orange', ls=':', lw=2,
           label=f'median Δ = {q_agg["delta"].median():+.3f}')
ax.set_xlabel(f'Δ HM agreement (inst_blind − blind)')
ax.set_ylabel('Number of questions')
ax.set_title(f'Distribution of instruction effect on agreement\n'
             f'(positive = instruction brings model closer to humans)')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
save_fig(fig, OUT_DIR, f'instruction_effect_per_question_v{VARIANT}.png')
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Summary stats
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'INSTRUCTION EFFECT ON AGREEMENT — VARIANT {VARIANT}')
print(f'{"="*60}')
print(f'Questions: {len(q_agg)} | Models: {len(common_models)}')
print(f'HH reference: {hh_mean:.3f}')
print(f'\nPer-question delta (inst − blind):')
print(f'  mean  = {q_agg["delta"].mean():+.4f}')
print(f'  median= {q_agg["delta"].median():+.4f}')
print(f'  std   = {q_agg["delta"].std():.4f}')
print(f'  ↑ {n_up}/{len(q_agg)} questions improved by instruction')
print(f'  ↓ {n_down}/{len(q_agg)} questions hurt by instruction')

print(f'\nPer-model summary:')
for _, row in model_agg.sort_values('delta', ascending=False).iterrows():
    sign = '+' if row['delta'] >= 0 else ''
    print(f'  {row["model"]:30s}  blind={row["blind"]:.3f}  inst={row["inst"]:.3f}  '
          f'Δ={sign}{row["delta"]:.3f}  [{row["model_group"]}]')

print(f'\nPer-group summary:')
for _, row in gdf.iterrows():
    sign = '+' if row['delta_mean'] >= 0 else ''
    print(f'  {row["group"]:30s}  blind={row["blind_mean"]:.3f}  inst={row["inst_mean"]:.3f}  '
          f'Δ={sign}{row["delta_mean"]:.3f}  (n={int(row["n_models"])} models)')
