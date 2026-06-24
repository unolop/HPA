"""
Generate the two missing question-diagnostic figures:
  1. entropy_alignment_analysis.png  (3-panel information-theoretic view)
  2. backbone_vlm_dumbbell_per_question.png  (per-question BB vs VLM)

Run from repo root:
  conda run -n zero python figures/question_diagnostic/generate_missing_figures.py
"""

import argparse
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
from config import MODELS_7B, extend_pair_cache_with_yesno

EXPORTS = ROOT / 'analysis/session2/exports'
OUT_DIR = ROOT / 'figures/question_diagnostic'
OUT_DIR.mkdir(parents=True, exist_ok=True)

_7b = set(MODELS_7B)

parser = argparse.ArgumentParser()
parser.add_argument('--free_text_only', action='store_true')
args = parser.parse_args()

# ── Build per-question diagnostic table ─────────────────────────────────────
pc = pd.read_parquet(EXPORTS / 'pair_cache.parquet')
if not args.free_text_only:
    pc = extend_pair_cache_with_yesno(pc, EXPORTS)
human = pd.read_csv(EXPORTS / 'responses_human.csv')
model_ib = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')

rows = []
for v in VARIANT_ORDER:
    sub = pc[pc['variant'] == v]
    hh = sub[sub['pair_type'] == 'HH'].groupby('question_id')['sbert_score'].mean()
    hm = sub[(sub['pair_type'] == 'HM') & (sub['subject_2'].isin(_7b))].groupby('question_id')['sbert_score'].mean()
    hm_grp = sub[(sub['pair_type'] == 'HM') & (sub['subject_2'].isin(_7b))].groupby(
        ['question_id', 'subject_group_2'])['sbert_score'].mean().unstack()
    for qid in hh.index:
        row = {'question_id': qid, 'variant': v, 'hh_sbert': hh.get(qid, np.nan), 'hm_sbert': hm.get(qid, np.nan)}
        for g in GROUP_ORDER:
            if g in hm_grp.columns:
                row[f'hm_{g}'] = hm_grp[g].get(qid, np.nan)
        rows.append(row)

qv = pd.DataFrame(rows)

h_acc = human[human['variant'] == 'C'].groupby('question_id')['accuracy'].mean().rename('human_acc')
m_acc = model_ib[(model_ib['variant'] == 'C') & (model_ib['model'].isin(_7b))].groupby('question_id')['accuracy'].mean().rename('model_acc')
meta = human[human['variant'] == 'C'].drop_duplicates('question_id')[
    ['question_id', 'question_en', 'ent', 'op', 'gt']
].set_index('question_id')

qwide = qv.pivot(index='question_id', columns='variant', values=['hh_sbert', 'hm_sbert'])
qwide.columns = [f'{m}_{v}' for m, v in qwide.columns]

for g in GROUP_ORDER:
    col = f'hm_{g}'
    if col in qv.columns:
        gpiv = qv.pivot(index='question_id', columns='variant', values=col)
        gpiv.columns = [f'{col}_{v}' for v in gpiv.columns]
        qwide = qwide.join(gpiv)

df = qwide.join(h_acc).join(m_acc).join(meta)
df['hm_drop_CA'] = df['hm_sbert_C'] - df['hm_sbert_A']

bb_col_C = 'hm_VLM backbone decoder_C'
vlm_col_C = 'hm_VLM_C'
if bb_col_C in df.columns and vlm_col_C in df.columns:
    df['bb_vlm_gap'] = df[bb_col_C] - df[vlm_col_C]

print(f'Diagnostic table: {len(df)} questions')

# ── Operation type colors ───────────────────────────────────────────────────
OP_COLORS = {
    'yesno': '#2196F3', 'count': '#E53935', 'attr': '#4CAF50',
    'act': '#FF9800', 'ident': '#9C27B0', 'spat': '#795548',
    'wk': '#607D8B', 'comp': '#00BCD4', 'temp': '#CDDC39',
    'text': '#F44336', 'caus': '#3F51B5',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Entropy vs Alignment (3-panel)
# ═══════════════════════════════════════════════════════════════════════════════

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
    df[['hm_sbert_C', 'hh_sbert_C', 'human_acc', 'model_acc', 'op', 'question_en', 'hm_drop_CA']]
)
for c in ['human_entropy', 'model_entropy', 'hm_sbert_C', 'hm_drop_CA']:
    ent_df = ent_df[ent_df[c].notna() & np.isfinite(ent_df[c])]

ent_df['entropy_gap'] = ent_df['model_entropy'] - ent_df['human_entropy']
print(f'Entropy analysis: {len(ent_df)} questions')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Human entropy vs Model entropy
ax = axes[0]
for op in sorted(ent_df['op'].unique()):
    sub = ent_df[ent_df['op'] == op]
    ax.scatter(sub['human_entropy'], sub['model_entropy'], s=30, alpha=0.6,
               color=OP_COLORS.get(op, '#888'), label=op, edgecolors='white', lw=0.3)
ax.plot([0, 4.5], [0, 4.5], 'k--', alpha=0.3)
ax.set_xlabel('Human answer entropy (bits)', fontsize=10)
ax.set_ylabel('Model answer entropy (bits)', fontsize=10)
ax.set_title('Answer Uncertainty:\nHuman vs Model', fontsize=10)
ax.legend(fontsize=6.5, ncol=2, loc='upper left')
r1, p1 = stats.pearsonr(ent_df['human_entropy'], ent_df['model_entropy'])
ax.text(0.95, 0.05, f'r={r1:.2f}, p={p1:.1e}', transform=ax.transAxes, fontsize=8, ha='right')

# Panel 2: Human entropy vs HM SBERT
ax = axes[1]
for op in sorted(ent_df['op'].unique()):
    sub = ent_df[ent_df['op'] == op]
    ax.scatter(sub['human_entropy'], sub['hm_sbert_C'], s=30, alpha=0.6,
               color=OP_COLORS.get(op, '#888'), label=op, edgecolors='white', lw=0.3)
ax.set_xlabel('Human answer entropy (bits)', fontsize=10)
ax.set_ylabel('HM SBERT (variant C)', fontsize=10)
ax.set_title('Human Uncertainty vs\nHuman-Model Alignment', fontsize=10)
r2, p2 = stats.pearsonr(ent_df['human_entropy'], ent_df['hm_sbert_C'])
ax.text(0.95, 0.05, f'r={r2:.2f}, p={p2:.1e}', transform=ax.transAxes, fontsize=8, ha='right')

# Panel 3: Entropy gap vs C→A drop
ax = axes[2]
valid = ent_df[ent_df['entropy_gap'].notna() & ent_df['hm_drop_CA'].notna()]
for op in sorted(valid['op'].unique()):
    sub = valid[valid['op'] == op]
    ax.scatter(sub['entropy_gap'], sub['hm_drop_CA'], s=30, alpha=0.6,
               color=OP_COLORS.get(op, '#888'), label=op, edgecolors='white', lw=0.3)
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
save_fig(fig, OUT_DIR, 'entropy_alignment_op.png')
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Per-Question Backbone vs VLM Dumbbell
# ═══════════════════════════════════════════════════════════════════════════════

bb_col = 'hm_VLM backbone decoder_C'
vlm_col = 'hm_VLM_C'

if bb_col in df.columns and vlm_col in df.columns:
    plot_df = df[[bb_col, vlm_col, 'op', 'question_en', 'hm_drop_CA']].dropna().copy()
    plot_df['gap'] = plot_df[bb_col] - plot_df[vlm_col]
    plot_df = plot_df.sort_values('gap', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    y = np.arange(len(plot_df))

    for i, (qid, r) in enumerate(plot_df.iterrows()):
        color = '#E67E22' if r['gap'] > 0 else '#E53935'
        ax.plot([r[vlm_col], r[bb_col]], [i, i], color=color, lw=1.5, alpha=0.7)
        ax.scatter(r[vlm_col], i, color='#E53935', s=25, zorder=5, edgecolors='white', lw=0.3)
        ax.scatter(r[bb_col], i, color='#E67E22', s=25, zorder=5, edgecolors='white', lw=0.3)

    q_labels = [f'{str(r.question_en)[:35]}... [{r.op}]' if len(str(r.question_en)) > 35
                else f'{r.question_en} [{r.op}]' for _, r in plot_df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(q_labels, fontsize=6.5)
    ax.set_xlabel('HM SBERT (variant C)', fontsize=10)
    ax.set_title('Per-Question Vision Encoder Effect:\nBackbone Decoder (orange) vs VLM (red)', fontsize=11)
    ax.axvline(plot_df[bb_col].mean(), color='#E67E22', ls='--', alpha=0.4, lw=1)
    ax.axvline(plot_df[vlm_col].mean(), color='#E53935', ls='--', alpha=0.4, lw=1)

    n_bb_wins = (plot_df['gap'] > 0).sum()
    n_total = len(plot_df)
    ax.text(0.02, 0.02, f'Backbone > VLM on {n_bb_wins}/{n_total} questions ({100*n_bb_wins/n_total:.0f}%)',
            transform=ax.transAxes, fontsize=9, va='bottom')

    plt.tight_layout()
    save_fig(fig, OUT_DIR, 'backbone_vlm_dumbbell_per_question.png')
    plt.close(fig)

    print(f'Backbone > VLM: {n_bb_wins}/{n_total} ({100*n_bb_wins/n_total:.0f}%)')
    print(f'Mean gap: {plot_df.gap.mean():.3f}')
else:
    print(f'Missing columns: {bb_col} or {vlm_col}')

print(f'\nDone. Output → {OUT_DIR}')
