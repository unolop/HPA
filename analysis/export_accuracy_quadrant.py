"""
Export per-question human vs. model accuracy quadrant scatter figures.

Two figures saved to figures/accuracy_quadrant/:

  blind_vC_jaccard_vlm.png
      x-axis = human accuracy (inst_blind)
      y-axis = VLM group mean accuracy (blind condition)
      colour  = Jaccard overlap of top-3 answers (human vs VLM)
      One point per question. Shows cross-condition human–model alignment.

  inst_blind_vC_jaccard_vlm_annotated.png
      Same axes but both conditions are inst_blind.
      Annotates shared-failure questions with entity type and question text.

Both are restricted to variant C and use only the human-study subset (common_qids).

Run from repo root:
  conda run -n zero python analysis/export_accuracy_quadrant.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from export_helpers import load_human_subset, read_response_exports
from config import MIN_ANSWERS_DEFAULT
OUT_DIR = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/accuracy_quadrant'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
    'axes.labelsize':    11,
    'axes.titlesize':    13,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
})


def norm_ans(s):
    if pd.isna(s): return ''
    return str(s).strip().lower()


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  [accuracy_quadrant] {name}')


def apply_axis_style(ax):
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(length=3.5, width=0.8, color='#555555')
    ax.grid(axis='both', color='#DEDEDE', linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print('Loading human data…')
participants, common_qids, human_df_full, _ = load_human_subset(
    ROOT, min_answers=MIN_ANSWERS_DEFAULT, translate=False, verbose=True)
n_humans = len(participants)

print('\nLoading model responses…')
exports = read_response_exports(ROOT, subset_qids=common_qids, variant='C')
df_mb = exports['model_blind']
df_mi = exports['model_inst_blind']
df_h  = exports['human']

n_questions = len(common_qids)
SUFFIX = f'_q{n_questions}_h{n_humans}'

for df in [df_mb, df_mi, df_h]:
    df['resp_norm'] = df['response'].apply(norm_ans)

print(f'  questions: {n_questions}  |  humans: {n_humans}')


def jaccard_top3(h_ans, m_ans):
    """Jaccard overlap of top-3 answer tokens between two lists."""
    hc = Counter(h_ans); mc = Counter(m_ans)
    ht = set(list(hc.keys())[:3]); mt = set(list(mc.keys())[:3])
    return len(ht & mt) / len(ht | mt) if (ht | mt) else 0.0


def build_quad_df(df_model):
    """Build per-question (h_acc, m_acc, jacc) table for VLM group."""
    vlm = df_model[df_model['model_group'] == 'VLM']
    h_acc_q = df_h.groupby('question_id')['accuracy'].mean().reset_index()
    h_acc_q.columns = ['question_id', 'h_acc']
    m_acc_q = vlm.groupby('question_id')['accuracy'].mean().reset_index()
    m_acc_q.columns = ['question_id', 'm_acc']

    jacc_rows = []
    for qid in h_acc_q['question_id'].unique():
        h_ans = df_h[df_h['question_id'] == qid]['resp_norm'].tolist()
        m_ans = vlm[vlm['question_id'] == qid]['resp_norm'].tolist()
        jacc_rows.append({'question_id': qid, 'jacc': jaccard_top3(h_ans, m_ans)})

    jacc_q = pd.DataFrame(jacc_rows)
    return h_acc_q.merge(m_acc_q, on='question_id').merge(jacc_q, on='question_id')


def _draw_quadrant_base(ax):
    apply_axis_style(ax)
    ax.axhline(0.5, color='gray', lw=0.7, ls='--', alpha=0.45)
    ax.axvline(0.5, color='gray', lw=0.7, ls='--', alpha=0.45)
    kw = dict(fontsize=8, color='#888888', alpha=0.75, transform=ax.transAxes)
    ax.text(0.04, 0.94, 'Shared\nfailure',  **kw)
    ax.text(0.60, 0.94, 'Human ✓\nModel ✗', **kw)
    ax.text(0.04, 0.04, 'Human ✗\nModel ✓', **kw)
    ax.text(0.60, 0.04, 'Both\ncorrect',    **kw)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — cross-condition: human (inst_blind) vs VLM (blind), coloured by Jaccard
# ─────────────────────────────────────────────────────────────────────────────
print('\n── blind_vC_jaccard_vlm ──')
quad_blind = build_quad_df(df_mb)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
sc = ax.scatter(quad_blind['h_acc'], quad_blind['m_acc'],
                c=quad_blind['jacc'], cmap='RdYlGn', vmin=0, vmax=1,
                s=58, alpha=0.84, edgecolors='white', lw=0.55)
cb = plt.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
cb.set_label('Answer overlap (Jaccard top-3)', fontsize=8.5)
_draw_quadrant_base(ax)
r_val = quad_blind['h_acc'].corr(quad_blind['m_acc'])
ax.text(0.98, 0.02, f'r = {r_val:.3f}', transform=ax.transAxes,
        ha='right', fontsize=9, color='#333333')
ax.set_xlabel('Human accuracy (inst_blind, variant C)', fontsize=10)
ax.set_ylabel('VLM accuracy (blind, variant C)', fontsize=10)
ax.set_title('Per-question Human vs. Model Accuracy\nColour = answer overlap', fontsize=11.5)
fig.tight_layout()
save(fig, f'blind_vC_jaccard_vlm{SUFFIX}.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — same condition: both inst_blind, with entity-type annotation
# ─────────────────────────────────────────────────────────────────────────────
print('\n── inst_blind_vC_jaccard_vlm_annotated ──')
quad_inst = build_quad_df(df_mi)

# Merge entity type from human responses
ent_map = (df_h[['question_id', 'ent']].drop_duplicates()
           .set_index('question_id')['ent'])
quad_inst['ent'] = quad_inst['question_id'].map(ent_map)

# Merge question text for annotation
q_text_map = (df_h[['question_id', 'question_en']].drop_duplicates()
              .set_index('question_id')['question_en']
              if 'question_en' in df_h.columns else pd.Series(dtype=str))

fig, ax = plt.subplots(figsize=(7.5, 6.5))
sc = ax.scatter(quad_inst['h_acc'], quad_inst['m_acc'],
                c=quad_inst['jacc'], cmap='RdYlGn', vmin=0, vmax=1,
                s=62, alpha=0.84, edgecolors='white', lw=0.55)
cb = plt.colorbar(sc, ax=ax, shrink=0.82)
cb.set_label('Answer overlap (Jaccard top-3)', fontsize=9)
_draw_quadrant_base(ax)

# Annotate shared-failure questions with high answer overlap
if not q_text_map.empty:
    highlight = (quad_inst[(quad_inst['jacc'] >= 0.9)
                           & (quad_inst['h_acc'] < 0.5)
                           & (quad_inst['m_acc'] < 0.5)]
                 .head(4))
    for _, r in highlight.iterrows():
        qtext = q_text_map.get(r['question_id'], '')
        txt = qtext[:35] + '...' if len(qtext) > 35 else qtext
        ax.annotate(txt, (r['h_acc'], r['m_acc']), xytext=(8, 8),
                    textcoords='offset points', fontsize=7, color='#555555',
                    arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.7))

r_all = quad_inst['h_acc'].corr(quad_inst['m_acc'])
ax.text(0.98, 0.02, f'r = {r_all:.3f}', transform=ax.transAxes,
        ha='right', fontsize=9, color='#333333')
ax.set_xlabel('Human accuracy (inst_blind)', fontsize=11)
ax.set_ylabel('VLM accuracy (inst_blind)', fontsize=11)
ax.set_title('Per-question Human vs. Model Accuracy\n'
             'Colour = answer distribution overlap', fontsize=12)
fig.tight_layout()
save(fig, f'inst_blind_vC_jaccard_vlm_annotated{SUFFIX}.png')

print(f'\nDone. Saved to: {OUT_DIR}')
