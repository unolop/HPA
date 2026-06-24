"""
Generate entropy alignment figure with per-model-family entropy on y-axis.

Instead of pooling all 16 7/8B models into one entropy value, computes
separate per-question entropy for each model group (VLM, backbone decoder,
standalone LLM) and plots against human entropy. This addresses the
non-independence concern: models within a family share training biases,
so pooled entropy conflates convergence with shared architecture.

Run from repo root:
  conda run -n zero python figures/question_diagnostic/generate_entropy_family.py
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
from utils.constants import GROUP_COLORS, VARIANT_ORDER
from config import MODELS_7B, MODEL_GROUP

EXPORTS = ROOT / 'analysis/session2/exports'
OUT_DIR = ROOT / 'figures/question_diagnostic'

_7b = set(MODELS_7B)

# ── Load data ─────────────────────────────────────────────────────────────────
pc = pd.read_parquet(EXPORTS / 'pair_cache.parquet')
human = pd.read_csv(EXPORTS / 'responses_human.csv')
model_ib = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')

# ── Build per-question HM SBERT and C→A degradation ──────────────────────────
rows = []
for v in VARIANT_ORDER:
    sub = pc[pc['variant'] == v]
    hh = sub[sub['pair_type'] == 'HH'].groupby('question_id')['sbert_score'].mean()
    hm = sub[(sub['pair_type'] == 'HM') & (sub['subject_2'].isin(_7b))].groupby('question_id')['sbert_score'].mean()
    for qid in hh.index:
        rows.append({'question_id': qid, 'variant': v,
                     'hh_sbert': hh.get(qid, np.nan),
                     'hm_sbert': hm.get(qid, np.nan)})

qv = pd.DataFrame(rows)
meta = human[human['variant'] == 'C'].drop_duplicates('question_id')[
    ['question_id', 'question_en', 'ent', 'op']
].set_index('question_id')

qwide = qv.pivot(index='question_id', columns='variant', values=['hh_sbert', 'hm_sbert'])
qwide.columns = [f'{m}_{v}' for m, v in qwide.columns]
df = qwide.join(meta)
df['hm_drop_CA'] = df['hm_sbert_C'] - df['hm_sbert_A']

# ── Entropy calculation ───────────────────────────────────────────────────────
def answer_entropy(responses):
    counts = responses['response'].value_counts()
    probs = counts / counts.sum()
    return sp_entropy(probs, base=2)

# Human entropy (40 participants per question)
h_ent = human[human['variant'] == 'C'].groupby('question_id').apply(
    answer_entropy, include_groups=False
).rename('human_entropy')

# Model entropy per family group
group_to_models = {}
for m in _7b:
    g = MODEL_GROUP.get(m, 'unknown')
    group_to_models.setdefault(g, []).append(m)

# Exclude think (N=1, degenerate entropy)
FAMILY_ORDER = ['VLM', 'VLM backbone decoder', 'standalone LLM']
FAMILY_LABELS = {
    'VLM': 'VLM',
    'VLM backbone decoder': 'Backbone',
    'standalone LLM': 'SA-LLM',
}

model_c = model_ib[(model_ib['variant'] == 'C') & (model_ib['model'].isin(_7b))]

family_entropies = {}
for grp in FAMILY_ORDER:
    mlist = group_to_models[grp]
    grp_resp = model_c[model_c['model'].isin(mlist)]
    fam_ent = grp_resp.groupby('question_id').apply(
        answer_entropy, include_groups=False
    ).rename(f'{grp}_entropy')
    family_entropies[grp] = fam_ent

# ── Merge everything ──────────────────────────────────────────────────────────
ent_df = pd.DataFrame({'human_entropy': h_ent})
for grp in FAMILY_ORDER:
    ent_df = ent_df.join(family_entropies[grp])
ent_df = ent_df.join(df[['hm_sbert_C', 'hm_drop_CA', 'ent', 'op']])

# Drop rows with missing values
keep_cols = ['human_entropy', 'hm_sbert_C', 'hm_drop_CA'] + \
            [f'{g}_entropy' for g in FAMILY_ORDER]
for c in keep_cols:
    ent_df = ent_df[ent_df[c].notna() & np.isfinite(ent_df[c])]

print(f'Questions: {len(ent_df)}')
for grp in FAMILY_ORDER:
    col = f'{grp}_entropy'
    r, p = stats.pearsonr(ent_df['human_entropy'], ent_df[col])
    mean_gap = (ent_df[col] - ent_df['human_entropy']).mean()
    n_below = (ent_df[col] < ent_df['human_entropy']).sum()
    print(f'  {grp}: r={r:.3f}, mean gap={mean_gap:+.3f}, below diagonal={n_below}/{len(ent_df)}')

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ═════════════════════════════════════════════════════════════════════════════
# Figure: 3-panel, per-family entropy
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── Panel 1: Human entropy vs Family entropy (main new panel) ────────────────
ax = axes[0]
for grp in FAMILY_ORDER:
    col = f'{grp}_entropy'
    color = GROUP_COLORS[grp]
    label = FAMILY_LABELS[grp]
    r, p = stats.pearsonr(ent_df['human_entropy'], ent_df[col])
    ax.scatter(ent_df['human_entropy'], ent_df[col], s=25, alpha=0.5,
               color=color, label=f'{label} (r={r:.2f})',
               edgecolors='white', lw=0.3)
ax.plot([0, 5.5], [0, 5.5], 'k--', alpha=0.3, label='y = x')
ax.set_xlabel('Human answer entropy (bits)', fontsize=10)
ax.set_ylabel('Model answer entropy (bits)', fontsize=10)
ax.set_title('Answer Diversity:\nHuman vs Model Family', fontsize=10)
ax.legend(fontsize=7.5, loc='upper left')

# ── Panel 2: Human entropy vs HM SBERT (by family) ──────────────────────────
ax = axes[1]

# Compute per-family HM SBERT
for grp in FAMILY_ORDER:
    mlist = group_to_models[grp]
    grp_hm = pc[(pc['variant'] == 'C') & (pc['pair_type'] == 'HM') &
                (pc['subject_2'].isin(mlist))].groupby('question_id')['sbert_score'].mean()
    col = f'{grp}_hm_sbert'
    ent_df[col] = grp_hm
    valid = ent_df[ent_df[col].notna()]
    color = GROUP_COLORS[grp]
    label = FAMILY_LABELS[grp]
    r, p = stats.pearsonr(valid['human_entropy'], valid[col])
    ax.scatter(valid['human_entropy'], valid[col], s=25, alpha=0.5,
               color=color, label=f'{label} (r={r:.2f})',
               edgecolors='white', lw=0.3)

ax.set_xlabel('Human answer entropy (bits)', fontsize=10)
ax.set_ylabel('HM SBERT (variant C)', fontsize=10)
ax.set_title('Human Uncertainty vs\nHuman-Model Alignment', fontsize=10)
ax.legend(fontsize=7.5, loc='upper right')

# ── Panel 3: Family entropy gap vs C→A degradation ──────────────────────────
ax = axes[2]
for grp in FAMILY_ORDER:
    col = f'{grp}_entropy'
    gap = ent_df[col] - ent_df['human_entropy']
    valid_mask = gap.notna() & ent_df['hm_drop_CA'].notna()
    color = GROUP_COLORS[grp]
    label = FAMILY_LABELS[grp]
    r, p = stats.pearsonr(gap[valid_mask], ent_df.loc[valid_mask, 'hm_drop_CA'])
    ax.scatter(gap[valid_mask], ent_df.loc[valid_mask, 'hm_drop_CA'],
               s=25, alpha=0.5, color=color,
               label=f'{label} (r={r:.2f})',
               edgecolors='white', lw=0.3)
ax.axvline(0, color='gray', ls=':', alpha=0.3)
ax.axhline(0, color='gray', ls=':', alpha=0.3)
ax.set_xlabel('Entropy gap (model − human, bits)', fontsize=10)
ax.set_ylabel('C→A degradation (SBERT)', fontsize=10)
ax.set_title('Entropy Mismatch vs\nEntity-Anchor Dependency', fontsize=10)
ax.legend(fontsize=7.5, loc='upper left')

plt.suptitle('Information-Theoretic View: Per-Family Answer Entropy (7/8B, vC)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, OUT_DIR, 'entropy_alignment_family.png')
plt.close(fig)

print(f'\nDone. Output → {OUT_DIR / "entropy_alignment_family.png"}')
