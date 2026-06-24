"""
Generate entropy_alignment_analysis figure with entity-type hue instead of op-type.

Run from repo root:
  conda run -n zero python figures/question_diagnostic/generate_entropy_entity.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy as sp_entropy

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from figures.helpers import save_fig
from utils.constants import GROUP_COLORS, GROUP_ORDER, VARIANT_ORDER
from config import MODELS_7B

EXPORTS = ROOT / 'analysis/session2/exports'
OUT_DIR = ROOT / 'figures/question_diagnostic'

_7b = set(MODELS_7B)

# ── Build per-question diagnostic table ─────────────────────────────────────
pc = pd.read_parquet(EXPORTS / 'pair_cache.parquet')
human = pd.read_csv(EXPORTS / 'responses_human.csv')
model_ib = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')

rows = []
for v in VARIANT_ORDER:
    sub = pc[pc['variant'] == v]
    hh = sub[sub['pair_type'] == 'HH'].groupby('question_id')['sbert_score'].mean()
    hm = sub[(sub['pair_type'] == 'HM') & (sub['subject_2'].isin(_7b))].groupby('question_id')['sbert_score'].mean()
    for qid in hh.index:
        row = {'question_id': qid, 'variant': v, 'hh_sbert': hh.get(qid, np.nan), 'hm_sbert': hm.get(qid, np.nan)}
        rows.append(row)

qv = pd.DataFrame(rows)
meta = human[human['variant'] == 'C'].drop_duplicates('question_id')[
    ['question_id', 'question_en', 'ent', 'op', 'gt']
].set_index('question_id')

qwide = qv.pivot(index='question_id', columns='variant', values=['hh_sbert', 'hm_sbert'])
qwide.columns = [f'{m}_{v}' for m, v in qwide.columns]
df = qwide.join(meta)
df['hm_drop_CA'] = df['hm_sbert_C'] - df['hm_sbert_A']

# ── Merge low-support entity types ──────────────────────────────────────────
ENT_MERGE = {
    'person': 'person', 'animal': 'animal', 'object': 'object',
    'food': 'food', 'other': 'other',
    'product': 'other', 'place': 'other', 'vehicle': 'other', 'text': 'other',
}
ENT_COLORS = {
    'person': '#E53935', 'animal': '#4CAF50', 'object': '#2196F3',
    'food': '#FF9800', 'other': '#9E9E9E',
}

# ── Entropy calculation ─────────────────────────────────────────────────────
def answer_entropy(responses, answer_col='response'):
    counts = responses[answer_col].value_counts()
    probs = counts / counts.sum()
    return sp_entropy(probs, base=2)

h_ent = human[human['variant'] == 'C'].groupby('question_id').apply(
    lambda x: answer_entropy(x)
).rename('human_entropy')

m_ent_all = model_ib[(model_ib['variant'] == 'C') & (model_ib['model'].isin(_7b))].groupby('question_id').apply(
    lambda x: answer_entropy(x)
).rename('model_entropy')

ent_df = pd.DataFrame({'human_entropy': h_ent, 'model_entropy': m_ent_all}).join(
    df[['hm_sbert_C', 'hh_sbert_C', 'op', 'ent', 'question_en', 'hm_drop_CA']]
)
for c in ['human_entropy', 'model_entropy', 'hm_sbert_C', 'hm_drop_CA']:
    ent_df = ent_df[ent_df[c].notna() & np.isfinite(ent_df[c])]

ent_df['entropy_gap'] = ent_df['model_entropy'] - ent_df['human_entropy']
ent_df['ent_group'] = ent_df['ent'].map(ENT_MERGE).fillna('other')

print(f'Entropy analysis: {len(ent_df)} questions')
print(f'Entity groups: {ent_df["ent_group"].value_counts().to_dict()}')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Figure: Entropy vs Alignment (3-panel, entity hue)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Human entropy vs Model entropy
ax = axes[0]
for ent_g in ['person', 'animal', 'object', 'food', 'other']:
    sub = ent_df[ent_df['ent_group'] == ent_g]
    ax.scatter(sub['human_entropy'], sub['model_entropy'], s=35, alpha=0.65,
               color=ENT_COLORS[ent_g], label=ent_g, edgecolors='white', lw=0.3)
ax.plot([0, 5.5], [0, 5.5], 'k--', alpha=0.3)
ax.set_xlabel('Human answer entropy (bits)', fontsize=10)
ax.set_ylabel('Model answer entropy (bits)', fontsize=10)
ax.set_title('Answer Uncertainty:\nHuman vs Model', fontsize=10)
ax.legend(fontsize=8, loc='upper left')
r1, p1 = stats.pearsonr(ent_df['human_entropy'], ent_df['model_entropy'])
ax.text(0.95, 0.05, f'r={r1:.2f}, p={p1:.1e}', transform=ax.transAxes, fontsize=8, ha='right')

# Panel 2: Human entropy vs HM SBERT
ax = axes[1]
for ent_g in ['person', 'animal', 'object', 'food', 'other']:
    sub = ent_df[ent_df['ent_group'] == ent_g]
    ax.scatter(sub['human_entropy'], sub['hm_sbert_C'], s=35, alpha=0.65,
               color=ENT_COLORS[ent_g], label=ent_g, edgecolors='white', lw=0.3)
ax.set_xlabel('Human answer entropy (bits)', fontsize=10)
ax.set_ylabel('HM SBERT (variant C)', fontsize=10)
ax.set_title('Human Uncertainty vs\nHuman-Model Alignment', fontsize=10)
r2, p2 = stats.pearsonr(ent_df['human_entropy'], ent_df['hm_sbert_C'])
ax.text(0.95, 0.05, f'r={r2:.2f}, p={p2:.1e}', transform=ax.transAxes, fontsize=8, ha='right')

# Panel 3: Entropy gap vs C→A drop
ax = axes[2]
valid = ent_df[ent_df['entropy_gap'].notna() & ent_df['hm_drop_CA'].notna()]
for ent_g in ['person', 'animal', 'object', 'food', 'other']:
    sub = valid[valid['ent_group'] == ent_g]
    ax.scatter(sub['entropy_gap'], sub['hm_drop_CA'], s=35, alpha=0.65,
               color=ENT_COLORS[ent_g], label=ent_g, edgecolors='white', lw=0.3)
ax.set_xlabel('Entropy gap (model − human, bits)', fontsize=10)
ax.set_ylabel('C→A degradation (SBERT)', fontsize=10)
ax.set_title('Entropy Mismatch vs\nEntity-Anchor Dependency', fontsize=10)
ax.axvline(0, color='gray', ls=':', alpha=0.3)
ax.axhline(0, color='gray', ls=':', alpha=0.3)
if len(valid) > 2:
    r3, p3 = stats.pearsonr(valid['entropy_gap'], valid['hm_drop_CA'])
    ax.text(0.95, 0.05, f'r={r3:.2f}, p={p3:.1e}', transform=ax.transAxes, fontsize=8, ha='right')

plt.suptitle('Information-Theoretic View: Answer Entropy and Prior Alignment (7/8B, vC)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, OUT_DIR, 'entropy_alignment_entity.png')
plt.close(fig)

print(f'\nDone. Output → {OUT_DIR}')
