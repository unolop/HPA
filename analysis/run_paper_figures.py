"""
Master figure export script for the AAAI paper.

Generates every figure needed by paper.tex and saves them to:
  latex/AnonymousSubmission/LaTeX/figures/   (main paper)
  latex/AnonymousSubmission/LaTeX/appendix/agreement/  (supplementary)

Run from the repo root:
  conda run -n zero python analysis/run_paper_figures.py

Each section is labelled with the paper figure it produces.
Re-running is safe: all outputs are overwritten.
"""

import sys
import subprocess
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.constants import GROUP_COLORS, GROUP_ORDER, TIER_STYLE

FIG_DIR    = ROOT / "latex/AnonymousSubmission/LaTeX/figures"
AGREE_DIR  = ROOT / "latex/AnonymousSubmission/LaTeX/figures/agreement_C"
FIG_DIR.mkdir(parents=True, exist_ok=True)
AGREE_DIR.mkdir(parents=True, exist_ok=True)

EXPORTS = ROOT / "analysis/session2/exports"

# ─────────────────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

# Short display labels used across figures
LABEL_MAP = {
    'LLaVA-Mistral (LM)': 'LLaVA-M\n(backbone)',
    'LLaVA-Vicuna (LM)':  'LLaVA-V\n(backbone)',
    'LLaVA-1.5 (LM)':     'LLaVA-1.5\n(backbone)',
    'LLaVA-Mistral':      'LLaVA-M',
    'LLaVA-Vicuna':       'LLaVA-V',
    'LLaVA-1.5-7B':       'LLaVA-1.5',
    'Qwen3-VL-8B':        'Qwen3-VL-8B',
    'InternVL-1B':        'IVL-1B',
    'InternVL-2B':        'IVL-2B',
    'InternVL-8B':        'IVL-8B',
    'Qwen3-8B':           'Qwen3-8B',
    'Qwen3-8B (think)':   'Qwen3-8B-T',
    'Qwen3-32B':          'Qwen3-32B',
    'Qwen3-32B (think)':  'Qwen3-32B-T',
    'Qwen3-4B':           'Qwen3-4B',
    'Qwen3-4B (think)':   'Qwen3-4B-T',
    'Qwen3-1.7B':         'Qwen3-1.7B',
    'Qwen3-1.7B (think)': 'Qwen3-1.7B-T',
    'Qwen3-0.6B':         'Qwen3-0.6B',
    'Qwen3-0.6B (think)': 'Qwen3-0.6B-T',
    'Qwen2.5-7B':         'Qwen2.5-7B',
    'Phi-3.5-mini':       'Phi-3.5',
    'Mistral-7B':         'Mistral-7B',
    'Vicuna-13B':         'Vicuna-13B',
}

def norm_ans(s):
    if pd.isna(s): return ''
    return str(s).strip().lower()

def to_int(s):
    try: return int(float(str(s)))
    except: return None

def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [figures] {name}')


# ─────────────────────────────────────────────────────────────────────────────
# Load shared data
# ─────────────────────────────────────────────────────────────────────────────
df_mb  = pd.read_csv(EXPORTS / 'responses_model_blind.csv')
df_mi  = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')
df_h   = pd.read_csv(EXPORTS / 'responses_human.csv')
pair_df = pd.read_parquet(EXPORTS / 'pair_cache.parquet')

with open(ROOT / 'dataset/vqa/vqav2_1k_val.json') as f:
    vqa1k = json.load(f)
vqa_df = pd.DataFrame([{'question_id': q['question_id'],
                         'answer_type': q['answer_type'],
                         'mc_answer':   q['multiple_choice_answer']} for q in vqa1k])

# Merge answer_type into model responses
qids_all = df_mb['question_id'].unique()
sub = vqa_df[vqa_df['question_id'].isin(qids_all)]
df_mb = df_mb.merge(sub[['question_id', 'answer_type', 'mc_answer']], on='question_id', how='left')
df_mi = df_mi.merge(sub[['question_id', 'answer_type', 'mc_answer']], on='question_id', how='left')
df_h  = df_h.merge(sub[['question_id', 'answer_type', 'mc_answer']], on='question_id', how='left')

df_mb['resp_norm'] = df_mb['response'].apply(norm_ans)
df_mi['resp_norm'] = df_mi['response'].apply(norm_ans)
df_h['resp_norm']  = df_h['response'].apply(norm_ans)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1  fig:scatter_agreement
# Per-model SBERT similarity vs exact-match to humans (inst_blind, variant C)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── Fig 1: fig_scatter_agreement ──')

hm = pair_df[(pair_df['variant'] == 'C') & (pair_df['pair_type'] == 'HM')]
hh = pair_df[(pair_df['variant'] == 'C') & (pair_df['pair_type'] == 'HH')]

per_model = (hm.groupby(['subject_2', 'subject_group_2'])
               .agg(sbert=('sbert_score', 'mean'),
                    exact=('exact_score', 'mean'))
               .reset_index()
               .rename(columns={'subject_2': 'model', 'subject_group_2': 'group'}))

hh_sbert = hh['sbert_score'].mean()
hh_exact = hh['exact_score'].mean()

# Slight jitter for exact-overlap pairs (think vs non-think at same coords)
JITTER = {
    'Qwen3-8B (think)':   (0, +0.003),
    'Qwen3-8B':           (0, -0.003),
    'Qwen3-4B (think)':   (0, +0.003),
    'Qwen3-4B':           (0, -0.003),
    'Qwen3-0.6B (think)': (0, +0.003),
    'Qwen3-0.6B':         (0, -0.003),
}

# Manual label offsets (dx, dy in data units, applied to jittered position)
LABEL_OFFSETS = {
    # Top-right backbone cluster — fan labels to avoid overlap
    'LLaVA-1.5 (LM)':    (-0.060,  0.004),   # push left of HH star
    'LLaVA-Mistral (LM)': ( 0.004,  0.005),   # right + up
    'LLaVA-Vicuna (LM)':  ( 0.004, -0.010),   # right + down
    'InternVL-8B':         ( 0.004,  0.000),   # far right, no issue
    # VLM cluster
    'LLaVA-1.5-7B':      (-0.040,  0.006),
    'LLaVA-Mistral':     (-0.040,  0.000),
    'LLaVA-Vicuna':      (-0.040, -0.007),
    'Qwen3-VL-8B':       ( 0.004,  0.004),
    'InternVL-2B':       ( 0.004, -0.007),
    'InternVL-1B':       (-0.050, -0.010),
    # Standalone
    'Mistral-7B':        (-0.005,  0.006),
    'Vicuna-13B':        (-0.005, -0.007),
    'Phi-3.5-mini':      ( 0.004,  0.004),
    'Qwen2.5-7B':        ( 0.004,  0.003),
    'Qwen3-32B':         (-0.050,  0.000),
    'Qwen3-32B (think)': ( 0.004,  0.000),
}

fig, ax = plt.subplots(figsize=(8, 6.5))

# Human–Human ceiling
ax.scatter([hh_sbert], [hh_exact], marker='*', s=220,
           color='#1565C0', zorder=5, label='Human–Human ceiling', linewidths=0)
ax.axvline(hh_sbert, color='#1565C0', lw=0.9, ls='--', alpha=0.4)
ax.axhline(hh_exact, color='#1565C0', lw=0.9, ls=':', alpha=0.4)
ax.text(hh_sbert + 0.001, hh_exact + 0.003, 'HH', fontsize=7, color='#1565C0', alpha=0.7)

# Plot each group (with jitter applied to point positions)
for grp in GROUP_ORDER:
    sub_ = per_model[per_model['group'] == grp]
    if sub_.empty:
        continue
    xs = sub_['sbert'] + sub_['model'].map(lambda m: JITTER.get(m, (0, 0))[0])
    ys = sub_['exact'] + sub_['model'].map(lambda m: JITTER.get(m, (0, 0))[1])
    ax.scatter(xs, ys,
               color=GROUP_COLORS[grp], s=65, zorder=3,
               label=grp, edgecolors='white', linewidths=0.7)

# Annotate every point
for _, row in per_model.iterrows():
    label = LABEL_MAP.get(row['model'], row['model'])
    jx, jy = JITTER.get(row['model'], (0, 0))
    px, py = row['sbert'] + jx, row['exact'] + jy
    dx, dy = LABEL_OFFSETS.get(row['model'], (0.004, 0.003))
    ax.annotate(label,
                xy=(px, py),
                xytext=(px + dx, py + dy),
                fontsize=6.5, color='#333333',
                ha='left' if dx >= 0 else 'right',
                va='center')

ax.set_xlabel('Mean SBERT cosine similarity to humans', fontsize=11)
ax.set_ylabel('Mean exact-match agreement with humans', fontsize=11)
ax.set_title('Human–Model Answer Agreement\n(variant C, inst_blind condition)', fontsize=11)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=8, frameon=True,
          loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)

plt.tight_layout()
save(fig, 'fig_scatter_agreement.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIG_MAIN  fig:overview_scatter  —  per-question human vs model accuracy scatter
# FIG_MAIN  fig:overview_degradation  —  control-variant degradation curves
# (formerly a single 2-panel figure; now saved separately)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── Fig_main: fig_overview_scatter + fig_overview_degradation ──')

# ── Shared data ───────────────────────────────────────────────────────────────
vlm_c = df_mb[(df_mb['model_group'] == 'VLM') & (df_mb['variant'] == 'C')]
h_c   = df_h[(df_h['variant'] == 'C')]

h_acc_q = h_c.groupby('question_id')['accuracy'].mean().reset_index()
h_acc_q.columns = ['question_id', 'h_acc']
m_acc_q = vlm_c.groupby('question_id')['accuracy'].mean().reset_index()
m_acc_q.columns = ['question_id', 'm_acc']

jacc_rows = []
for qid in h_acc_q['question_id'].unique():
    h_ans = h_c[h_c['question_id'] == qid]['resp_norm'].tolist()
    m_ans = vlm_c[vlm_c['question_id'] == qid]['resp_norm'].tolist()
    hc_ = Counter(h_ans); mc_ = Counter(m_ans)
    ht_ = set(list(hc_.keys())[:3]); mt_ = set(list(mc_.keys())[:3])
    jacc_ = len(ht_ & mt_) / len(ht_ | mt_) if (ht_ | mt_) else 0
    jacc_rows.append({'question_id': qid, 'jacc': jacc_})
jacc_q  = pd.DataFrame(jacc_rows)
quad_df = h_acc_q.merge(m_acc_q, on='question_id').merge(jacc_q, on='question_id')

VARIANT_LABEL = {'C': 'Original\n(C)', 'B': 'Weaker\n(B)', 'A': 'Pronoun.\n(A)'}
deg_model  = df_mb.groupby(['variant', 'model_group'])['accuracy'].mean().reset_index()
deg_human  = df_h.groupby(['variant'])['accuracy'].mean().reset_index()
deg_human['model_group'] = 'Human'
VARIANT_ORDER = ['C', 'B', 'A']
X_DEG = [0, 1, 2]

DEG_COLORS = {**GROUP_COLORS, 'Human': '#1565C0'}
DEG_STYLES = {
    'VLM backbone decoder':   ('o', '-'),
    'VLM':                    ('s', '-'),
    'standalone LLM (think)': ('D', '--'),
    'standalone LLM':         ('^', '--'),
    'Human':                  ('*', '-'),
}
DEG_LABEL = {
    'VLM backbone decoder':   'Backbone decoder',
    'VLM':                    'VLM',
    'standalone LLM (think)': 'Standalone LLM (think)',
    'standalone LLM':         'Standalone LLM',
    'Human':                  'Human',
}

# ── fig_overview_scatter: human vs model accuracy scatter ────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sc = ax.scatter(quad_df['h_acc'], quad_df['m_acc'],
                c=quad_df['jacc'], cmap='RdYlGn', vmin=0, vmax=1,
                s=50, alpha=0.80, edgecolors='white', lw=0.4)
cb = plt.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
cb.set_label('Answer overlap (Jaccard top-3)', fontsize=8.5)
ax.axhline(0.5, color='gray', lw=0.7, ls='--', alpha=0.45)
ax.axvline(0.5, color='gray', lw=0.7, ls='--', alpha=0.45)
kw2 = dict(fontsize=8, color='#888888', alpha=0.75)
ax.text(0.04, 0.94, 'Shared\nfailure',  transform=ax.transAxes, **kw2)
ax.text(0.60, 0.94, 'Human ✓\nModel ✗', transform=ax.transAxes, **kw2)
ax.text(0.04, 0.04, 'Human ✗\nModel ✓', transform=ax.transAxes, **kw2)
ax.text(0.60, 0.04, 'Both\ncorrect',    transform=ax.transAxes, **kw2)
r_val = quad_df['h_acc'].corr(quad_df['m_acc'])
ax.text(0.98, 0.02, f'r = {r_val:.3f}', transform=ax.transAxes,
        ha='right', fontsize=9, color='#333333')
ax.set_xlabel('Human accuracy (inst\_blind, variant C)', fontsize=10)
ax.set_ylabel('VLM accuracy (blind, variant C)', fontsize=10)
ax.set_title('Per-question Human vs.\ Model Accuracy\n(colour = answer overlap)', fontsize=10)
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
save(fig, 'fig_overview_scatter.png')

# ── fig_overview_degradation: control-variant degradation curves ──────────────
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for grp in ['Human', 'VLM backbone decoder', 'VLM', 'standalone LLM', 'standalone LLM (think)']:
    if grp == 'Human':
        ys = [deg_human[deg_human['variant'] == v]['accuracy'].values[0]
              for v in VARIANT_ORDER]
    else:
        g_ = deg_model[deg_model['model_group'] == grp]
        ys = [g_[g_['variant'] == v]['accuracy'].values[0]
              for v in VARIANT_ORDER if len(g_[g_['variant'] == v]) > 0]
        if len(ys) < len(VARIANT_ORDER): continue
    marker, ls = DEG_STYLES[grp]
    ax.plot(X_DEG, ys, color=DEG_COLORS[grp], lw=2, ls=ls,
            marker=marker, markersize=8 if grp != 'Human' else 11,
            label=DEG_LABEL[grp], zorder=3)
    ax.text(X_DEG[-1] + 0.06, ys[-1], f'{ys[-1]:.2f}',
            va='center', fontsize=8, color=DEG_COLORS[grp])
ax.set_xticks(X_DEG)
ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANT_ORDER], fontsize=9)
ax.set_ylabel('Mean accuracy (blind condition)', fontsize=10)
ax.set_title('Control-Variant Degradation\n(Original → Pronominalized)', fontsize=10)
ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.14),
          ncol=3, frameon=True)
ax.set_xlim(-0.3, 2.5)
ax.set_ylim(0, 0.58)
ax.axhspan(0, 0.262, alpha=0.04, color='#1565C0')
plt.tight_layout()
save(fig, 'fig_overview_degradation.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2  fig:instruction_effect + fig:response_change
# Delegates to export_instruction_effect.py → figures/instruction_effect/
# ─────────────────────────────────────────────────────────────────────────────
print('\n── Fig 2: fig_instruction_effect + fig_response_change (delegated) ──')
result = subprocess.run(
    ['conda', 'run', '-n', 'zero', 'python',
     str(ROOT / 'analysis/export_instruction_effect.py')],
    capture_output=True, text=True
)
print(result.stdout[-500:] if result.stdout else '')
if result.returncode != 0:
    print('WARNING: instruction effect script failed:', result.stderr[-300:])



# ─────────────────────────────────────────────────────────────────────────────
# §6  fig:hm_alignment  —  Pearson r by entity type (entity-only, single panel)
# Motivation: entity groups form coherent semantic clusters (see t-SNE in appendix)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig_hm_alignment (entity-type only) ──')

vlm_i2 = df_mi[(df_mi['variant'] == 'C') & (df_mi['model_group'] == 'VLM')]
qids2  = df_h[df_h['variant'] == 'C']['question_id'].unique()
df_h2  = df_h.copy()

rows = []
for qid in qids2:
    h_ans = df_h2[(df_h2['variant'] == 'C') & (df_h2['question_id'] == qid)]['resp_norm'].tolist()
    m_ans = vlm_i2[vlm_i2['question_id'] == qid]['resp_norm'].tolist()
    hc = Counter(h_ans); mc = Counter(m_ans)
    ht = set(list(hc.keys())[:3]); mt = set(list(mc.keys())[:3])
    jacc = len(ht & mt) / len(ht | mt) if (ht | mt) else 0
    ent_ = df_h2[df_h2['question_id'] == qid]['ent'].iloc[0]
    h_acc = df_h2[(df_h2['variant'] == 'C') & (df_h2['question_id'] == qid)]['accuracy'].mean()
    m_acc = vlm_i2[vlm_i2['question_id'] == qid]['accuracy'].mean()
    rows.append({'question_id': qid, 'ent': ent_, 'jacc': jacc,
                 'h_acc': h_acc, 'm_acc': m_acc})

res = pd.DataFrame(rows)

ent_r   = res.groupby('ent').apply(
    lambda g: g['h_acc'].corr(g['m_acc']) if len(g) >= 5 else np.nan,
    include_groups=False
).dropna().sort_values(ascending=False)
ent_n   = res.groupby('ent').size()

CMAP = plt.cm.RdYlGn

# Entity type colours (consistent with t-SNE in appendix)
ENT_COLORS = {
    'object':  '#e74c3c', 'person': '#3498db', 'animal': '#f39c12',
    'food':    '#27ae60', 'other':  '#9b59b6', 'place':  '#1abc9c',
    'product': '#e67e22', 'text':   '#95a5a6', 'vehicle':'#2c3e50',
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Left panel: entity type distribution (n per entity, sorted by count) ─────
ax = axes[0]
ent_counts = ent_n.sort_values(ascending=True)
bar_colors = [ENT_COLORS.get(e, '#aaaaaa') for e in ent_counts.index]
ax.barh([e.capitalize() for e in ent_counts.index], ent_counts.values,
        color=bar_colors, edgecolor='white', height=0.65)
for i, (e, n) in enumerate(zip(ent_counts.index, ent_counts.values)):
    ax.text(n + 0.3, i, str(n), va='center', fontsize=9.5, color='#333333')
ax.set_xlabel('Number of questions (out of 113)', fontsize=10)
ax.set_title('Question Distribution by Entity Type\n(human study, variant C)', fontsize=11)
ax.set_xlim(0, 36)
ax.set_ylabel('Entity type', fontsize=10)

# ── Right panel: Pearson r by entity type ────────────────────────────────────
ax = axes[1]
ents  = ent_r.index.tolist(); vals2 = ent_r.values
ax.barh([e.capitalize() for e in ents], vals2,
        color=[ENT_COLORS.get(e, '#aaaaaa') for e in ents],
        edgecolor='white', height=0.65, alpha=0.85)
for v, e in zip(vals2, ents):
    n = ent_n.get(e, 0)
    offset = 0.03 if v >= 0 else -0.03
    ax.text(v + offset, ents.index(e),
            f'r = {v:.2f}',
            va='center', ha='left' if v >= 0 else 'right', fontsize=9.5)
ax.set_xlim(-1.0, 1.4)
ax.set_xlabel('Pearson r (human vs.\ VLM accuracy, per question)', fontsize=10)
ax.set_title('Human–Model Difficulty Correlation\nby Entity Type (VLM inst\_blind)', fontsize=11)
ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_ylabel('')

plt.tight_layout()
save(fig, 'fig_hm_alignment.png')


# ─────────────────────────────────────────────────────────────────────────────
# §6  fig:hm_quadrant  —  Per-question scatter human vs model acc
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig_hm_quadrant ──')

fig, ax = plt.subplots(figsize=(7.5, 6.5))
sc = ax.scatter(res['h_acc'], res['m_acc'], c=res['jacc'],
                cmap='RdYlGn', vmin=0, vmax=1,
                s=55, alpha=0.82, edgecolors='white', lw=0.4)
cb = plt.colorbar(sc, ax=ax, shrink=0.82)
cb.set_label('Answer overlap (Jaccard top-3)', fontsize=9)
ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.axvline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
kw = dict(fontsize=8.5, color='gray', alpha=0.7)
ax.text(0.08, 0.93, 'Both wrong\n(shared failure)',   transform=ax.transAxes, **kw)
ax.text(0.58, 0.93, 'Human right\nModel wrong',        transform=ax.transAxes, **kw)
ax.text(0.08, 0.05, 'Human wrong\nModel right',        transform=ax.transAxes, **kw)
ax.text(0.58, 0.05, 'Both correct\n(shared knowledge)', transform=ax.transAxes, **kw)

highlight = res[(res['jacc'] >= 0.9) & (res['h_acc'] < 0.5) & (res['m_acc'] < 0.5)].head(4)
for _, r in highlight.iterrows():
    qtext = df_h2[df_h2['question_id'] == r['question_id']]['question_en'].iloc[0]
    txt   = qtext[:35] + '...' if len(qtext) > 35 else qtext
    ax.annotate(txt, (r['h_acc'], r['m_acc']), xytext=(8, 8),
                textcoords='offset points', fontsize=7, color='#555555',
                arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.7))

r_all = res['h_acc'].corr(res['m_acc'])
ax.text(0.98, 0.02, f'r = {r_all:.3f}', transform=ax.transAxes,
        ha='right', fontsize=9, color='#333333')
ax.set_xlabel('Human accuracy (inst_blind)', fontsize=11)
ax.set_ylabel('VLM accuracy (inst_blind)', fontsize=11)
ax.set_title('Per-question Human vs. Model Accuracy\n'
             'colour = answer distribution overlap', fontsize=11)
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
save(fig, 'fig_hm_quadrant.png')


# ─────────────────────────────────────────────────────────────────────────────
# §6  fig:scale_alignment  —  SBERT alignment to humans vs model size
# Shows Qwen3 nothink / think / VLM families; x-axis log-scale by #params
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig_scale_alignment ──')

# Model size (B params) for Qwen3 families
SIZE_B = {
    # standalone nothink
    'Qwen3-0.6B':        0.6,
    'Qwen3-1.7B':        1.7,
    'Qwen3-4B':          4.0,
    'Qwen3-8B':          8.0,
    'Qwen3-32B':        32.0,
    # standalone think
    'Qwen3-0.6B (think)': 0.6,
    'Qwen3-1.7B (think)': 1.7,
    'Qwen3-4B (think)':   4.0,
    'Qwen3-8B (think)':   8.0,
    'Qwen3-32B (think)':  32.0,
    # VLM
    'Qwen3-VL-2B':  2.0,
    'Qwen3-VL-4B':  4.0,
    'Qwen3-VL-8B':  8.0,
    'Qwen3-VL-32B': 32.0,
}

SCALE_FAMILIES = {
    'standalone LLM':        ('Standalone LLM (nothink)', '#2E7D32', 'o', '-'),
    'standalone LLM (think)':('Standalone LLM (think)',   '#8E24AA', 's', '--'),
    'VLM':                   ('VLM (Qwen3-VL)',           '#E53935', '^', ':'),
}

hm_c = pair_df[(pair_df['variant'] == 'C') & (pair_df['pair_type'] == 'HM')]
per_model_s = (hm_c.groupby(['subject_2', 'subject_group_2'])
                    .agg(sbert=('sbert_score', 'mean'))
                    .reset_index()
                    .rename(columns={'subject_2': 'model', 'subject_group_2': 'group'}))

# Keep only Qwen3 family members with a known size
per_model_s = per_model_s[per_model_s['model'].isin(SIZE_B)].copy()
per_model_s['size_b'] = per_model_s['model'].map(SIZE_B)

fig, ax = plt.subplots(figsize=(7.5, 5.5))

for grp_key, (label, color, marker, ls) in SCALE_FAMILIES.items():
    sub_ = (per_model_s[per_model_s['group'] == grp_key]
            .sort_values('size_b'))
    if sub_.empty:
        continue
    ax.plot(sub_['size_b'], sub_['sbert'], color=color, lw=2, ls=ls,
            marker=marker, markersize=8, label=label, zorder=3,
            markeredgecolor='white', markeredgewidth=0.8)
    # annotate each point with size label
    for _, row in sub_.iterrows():
        ax.annotate(f"{row['size_b']:.0f}B" if row['size_b'] >= 1 else '0.6B',
                    xy=(row['size_b'], row['sbert']),
                    xytext=(0, 8), textcoords='offset points',
                    fontsize=7.5, ha='center', color=color)

# Human-human ceiling
ax.axhline(hh_sbert, color='#1565C0', lw=1.2, ls='--', alpha=0.55,
           label=f'Human–Human ceiling ({hh_sbert:.3f})')
ax.text(0.55 * 32, hh_sbert + 0.002, 'HH ceiling', fontsize=8,
        color='#1565C0', alpha=0.7)

ax.set_xscale('log')
ax.set_xticks([0.6, 1, 2, 4, 8, 16, 32])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Model size (billion parameters, log scale)', fontsize=11)
ax.set_ylabel('Mean SBERT cosine similarity to humans', fontsize=11)
ax.set_title('Human Alignment vs. Scale\n(Qwen3 family, variant C, inst_blind)', fontsize=11)
ax.legend(fontsize=9, loc='lower right', frameon=True)
ax.set_xlim(0.4, 50)

plt.tight_layout()
save(fig, 'fig_scale_alignment.png')


# ─────────────────────────────────────────────────────────────────────────────
# §6  fig:interrater  —  SBERT group-mean heatmap  (delegates to separate script)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig:interrater: sbert_heatmap_groups ──')
import subprocess
result = subprocess.run(
    ['conda', 'run', '-n', 'zero', 'python',
     str(ROOT / 'analysis/export_agreement_heatmaps.py')],
    capture_output=True, text=True
)
print(result.stdout[-500:] if result.stdout else '')
if result.returncode != 0:
    print('WARNING: heatmap script failed:', result.stderr[-300:])


# ─────────────────────────────────────────────────────────────────────────────
# §5  fig:dist_rates  —  Abstention figures (regenerate from notebook)
# ─────────────────────────────────────────────────────────────────────────────
# Abstention figures are produced by analysis/session2/08_char_abstention.ipynb
# Run the notebook to regenerate:
#   jupyter nbconvert --to notebook --execute analysis/session2/08_char_abstention.ipynb
# Expected outputs in figures/:  abstention_rates.png, abstention_collapse.png
print('\n── abstention figures: run 08_char_abstention.ipynb to regenerate ──')

# ─────────────────────────────────────────────────────────────────────────────
print('\nDone.')
print()
print('Main figures:')
print('  Fig 1  figures/fig_scatter_agreement.png   (§6 / intro)')
print('  Fig 2a figures/instruction_effect/fig_soft_abstention.png  (§5 — soft abstention collapse)')
print('  Fig 2b figures/instruction_effect/fig_hard_abstention.png  (§5 — hard abstention collapse)')
print('  Fig 2c figures/instruction_effect/fig_response_change.png  (§5 — response change rate)')
print('  Fig 3  figures/fig_scale_alignment.png     (§6 — new: SBERT vs model size)')
print('  Fig 4  figures/sbert_heatmap_groups.png    (§6, via heatmap script)')
print()
print('Supporting figures:')
print('  §5  fig_overview.png')
print('  §6  fig_hm_alignment.png')
print('  §6  fig_hm_quadrant.png')
print('  §5  abstention_rates.png / abstention_collapse.png  (from notebook)')
