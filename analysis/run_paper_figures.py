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
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIG_DIR    = ROOT / "latex/AnonymousSubmission/LaTeX/figures"
APPEND_DIR = ROOT / "latex/AnonymousSubmission/LaTeX/appendix/agreement"
FIG_DIR.mkdir(parents=True, exist_ok=True)
APPEND_DIR.mkdir(parents=True, exist_ok=True)

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

GROUP_COLORS = {
    'VLM backbone decoder':  '#E67E22',   # orange
    'VLM':                   '#E53935',   # red
    'standalone LLM (think)':'#8E24AA',   # purple
    'standalone LLM':        '#2E7D32',   # green
}
GROUP_ORDER = ['VLM backbone decoder', 'VLM', 'standalone LLM (think)', 'standalone LLM']

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
          loc='upper left', bbox_to_anchor=(0.01, 0.99))

plt.tight_layout()
save(fig, 'fig_scatter_agreement.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIG_MAIN  fig:overview  —  2-panel overview figure
# Left:  per-question human vs model accuracy scatter (VLM blind, coloured by Jaccard)
# Right: control-variant degradation curves (C→B→A) for all model groups + humans
# ─────────────────────────────────────────────────────────────────────────────
print('\n── Fig_main: fig_overview (2-panel) ──')

# ── Left panel data: per-question human acc vs VLM acc (variant C, blind) ───
vlm_c = df_mb[(df_mb['model_group'] == 'VLM') & (df_mb['variant'] == 'C')]
h_c   = df_h[(df_h['variant'] == 'C')]

# Human difficulty: mean accuracy per question across all participants
h_acc_q = h_c.groupby('question_id')['accuracy'].mean().reset_index()
h_acc_q.columns = ['question_id', 'h_acc']

# VLM mean accuracy per question
m_acc_q = vlm_c.groupby('question_id')['accuracy'].mean().reset_index()
m_acc_q.columns = ['question_id', 'm_acc']

# Jaccard top-3 overlap per question
jacc_rows = []
for qid in h_acc_q['question_id'].unique():
    h_ans = h_c[h_c['question_id'] == qid]['resp_norm'].tolist()
    m_ans = vlm_c[vlm_c['question_id'] == qid]['resp_norm'].tolist()
    hc_ = Counter(h_ans); mc_ = Counter(m_ans)
    ht_ = set(list(hc_.keys())[:3]); mt_ = set(list(mc_.keys())[:3])
    jacc_ = len(ht_ & mt_) / len(ht_ | mt_) if (ht_ | mt_) else 0
    jacc_rows.append({'question_id': qid, 'jacc': jacc_})
jacc_q = pd.DataFrame(jacc_rows)

quad_df = h_acc_q.merge(m_acc_q, on='question_id').merge(jacc_q, on='question_id')

# ── Right panel data: degradation curves (C→B→A), blind condition ───────────
VARIANT_LABEL = {'C': 'Original\n(C)', 'B': 'Weaker\n(B)', 'A': 'Pronoun.\n(A)'}

# Model groups
deg_model = (df_mb.groupby(['variant', 'model_group'])['accuracy']
             .mean().reset_index())
# Humans
deg_human = (df_h.groupby(['variant'])['accuracy']
             .mean().reset_index())
deg_human['model_group'] = 'Human'

VARIANT_ORDER = ['C', 'B', 'A']
X_DEG = [0, 1, 2]

# ── Build figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: scatter
ax = axes[0]
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
ax.set_xlabel('Human accuracy (inst_blind, variant C)', fontsize=10)
ax.set_ylabel('VLM accuracy (blind, variant C)', fontsize=10)
ax.set_title('Per-question Human vs.\ Model Accuracy\n(colour = answer overlap)', fontsize=10)
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

# Right: degradation curves
ax = axes[1]
DEG_COLORS = {
    'VLM backbone decoder':  '#E67E22',
    'VLM':                   '#E53935',
    'standalone LLM (think)':'#8E24AA',
    'standalone LLM':        '#2E7D32',
    'Human':                 '#1565C0',
}
DEG_STYLES = {
    'VLM backbone decoder':  ('o', '-'),
    'VLM':                   ('s', '-'),
    'standalone LLM (think)':('D', '--'),
    'standalone LLM':        ('^', '--'),
    'Human':                 ('*', '-'),
}
DEG_SIZE = {'Human': 130, 'VLM': 70, 'VLM backbone decoder': 70,
            'standalone LLM': 70, 'standalone LLM (think)': 70}
DEG_LABEL = {
    'VLM backbone decoder':  'Backbone decoder',
    'VLM':                   'VLM',
    'standalone LLM (think)':'Standalone LLM (think)',
    'standalone LLM':        'Standalone LLM',
    'Human':                 'Human',
}

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
    # annotate end-point
    ax.text(X_DEG[-1] + 0.06, ys[-1], f'{ys[-1]:.2f}',
            va='center', fontsize=8, color=DEG_COLORS[grp])

ax.set_xticks(X_DEG)
ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANT_ORDER], fontsize=9)
ax.set_ylabel('Mean accuracy (blind condition)', fontsize=10)
ax.set_title('Control-Variant Degradation\n(Original → Pronominalized)', fontsize=10)
ax.legend(fontsize=8.5, loc='lower left', frameon=True)
ax.set_xlim(-0.3, 2.5)
ax.set_ylim(0, 0.58)
ax.axhspan(0, 0.262, alpha=0.04, color='#1565C0')  # human ceiling band
ax.text(2.2, 0.268, 'Human\nceiling', fontsize=7, color='#1565C0', ha='center', alpha=0.7)

plt.tight_layout()
save(fig, 'fig_overview.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2  fig:instruction_effect
# Dual panel: soft abstention collapse + response change rate (blind → inst_blind)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── Fig 2: fig_instruction_effect ──')

SOFT_RE = re.compile(
    r'\b(none|nothing|nowhere|no one|unanswerable|cannot|can\'t|unknown|'
    r'n/a|unclear|not visible|not shown|not present|not enough)\b', re.I
)

# Exclude models with too few rows in either condition
MIN_N = 50

rows_inst = []
for model in df_mb['model'].unique():
    b = df_mb[(df_mb['model'] == model) & (df_mb['variant'] == 'C')]
    i = df_mi[(df_mi['model'] == model) & (df_mi['variant'] == 'C')]
    if len(b) < MIN_N or len(i) < MIN_N:
        continue
    grp = b['model_group'].iloc[0]
    soft_b = b['resp_norm'].apply(lambda x: bool(SOFT_RE.search(x))).mean()
    soft_i = i['resp_norm'].apply(lambda x: bool(SOFT_RE.search(x))).mean()
    # Response change rate
    merged = b[['question_id', 'resp_norm']].rename(columns={'resp_norm': 'rb'}).merge(
        i[['question_id', 'resp_norm']].rename(columns={'resp_norm': 'ri'}),
        on='question_id')
    change = (merged['rb'] != merged['ri']).mean()
    rows_inst.append(dict(model=model, group=grp,
                          soft_b=soft_b, soft_i=soft_i,
                          change=change))

df_inst = pd.DataFrame(rows_inst)
df_inst['label'] = df_inst['model'].map(LABEL_MAP).fillna(df_inst['model'])

# Sort within each group by change rate for cleaner layout
df_inst = df_inst.sort_values(['group', 'change'], ascending=[True, True])

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# ── Left panel: soft abstention collapse (slope per model, grouped) ──
ax = axes[0]
# Build y-positions grouped by model group
y_positions = {}
y = 0
group_spans = {}
for grp in GROUP_ORDER:
    sub_ = df_inst[df_inst['group'] == grp].reset_index(drop=True)
    if sub_.empty:
        continue
    start = y
    for _, row in sub_.iterrows():
        y_positions[row['model']] = y
        y += 1
    group_spans[grp] = (start, y - 1)
    y += 0.6  # gap between groups

for grp in GROUP_ORDER:
    sub_ = df_inst[df_inst['group'] == grp]
    if sub_.empty:
        continue
    c = GROUP_COLORS[grp]
    for _, row in sub_.iterrows():
        yp = y_positions[row['model']]
        ax.plot([row['soft_b'], row['soft_i']], [yp, yp],
                color=c, lw=1.5, alpha=0.7, zorder=2)
        ax.scatter([row['soft_b']], [yp], color=c, s=55, zorder=3,
                   marker='o', edgecolors='white', linewidths=0.6)
        ax.scatter([row['soft_i']], [yp], color=c, s=55, zorder=3,
                   marker='s', edgecolors='white', linewidths=0.6)
        # label on right
        ax.text(max(row['soft_b'], row['soft_i']) + 0.008, yp,
                row['label'], va='center', fontsize=7, color='#333333')

# Group labels
for grp, (s, e) in group_spans.items():
    mid = (s + e) / 2
    ax.text(-0.01, mid, grp, va='center', ha='right', fontsize=7.5,
            color=GROUP_COLORS[grp], fontweight='bold', transform=ax.get_yaxis_transform())

# legend for circle vs square
h1, = ax.plot([], [], 'o', color='gray', ms=6, label='blind', markeredgecolor='white')
h2, = ax.plot([], [], 's', color='gray', ms=6, label='inst_blind', markeredgecolor='white')
ax.legend(handles=[h1, h2], fontsize=8, loc='lower right', frameon=True)

ax.set_yticks([])
ax.set_xlabel('Soft abstention rate', fontsize=10)
ax.set_title('Soft abstention: blind → inst_blind', fontsize=11)
ax.set_xlim(-0.05, 0.65)
ax.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.4)

# ── Right panel: response change rate (horizontal bars, per model) ──
ax = axes[1]
df_inst_s = df_inst.sort_values(['group', 'change'], ascending=[True, False])
y_pos2 = {}
y2 = 0
grp_spans2 = {}
for grp in GROUP_ORDER:
    sub_ = df_inst_s[df_inst_s['group'] == grp].reset_index(drop=True)
    if sub_.empty:
        continue
    start = y2
    for _, row in sub_.iterrows():
        y_pos2[row['model']] = y2
        y2 += 1
    grp_spans2[grp] = (start, y2 - 1)
    y2 += 0.6

for grp in GROUP_ORDER:
    sub_ = df_inst_s[df_inst_s['group'] == grp]
    if sub_.empty:
        continue
    c = GROUP_COLORS[grp]
    for _, row in sub_.iterrows():
        yp = y_pos2[row['model']]
        ax.barh(yp, row['change'], color=c, alpha=0.8,
                height=0.65, edgecolor='none')
        ax.text(row['change'] + 0.01, yp,
                f"{row['change']:.0%}", va='center', fontsize=7.5, color='#333333')
        ax.text(-0.01, yp, row['label'], va='center', ha='right',
                fontsize=7, color='#333333')

for grp, (s, e) in grp_spans2.items():
    mid = (s + e) / 2
    ax.text(1.02, mid, grp, va='center', ha='left', fontsize=7.5,
            color=GROUP_COLORS[grp], fontweight='bold', transform=ax.get_yaxis_transform())

ax.set_yticks([])
ax.set_xlabel('Fraction of responses changed', fontsize=10)
ax.set_title('Response change rate: blind → inst_blind', fontsize=11)
ax.set_xlim(-0.15, 1.18)
ax.axvline(0, color='gray', lw=0.5, ls='-', alpha=0.3)

plt.tight_layout()
save(fig, 'fig_instruction_effect.png')


# ─────────────────────────────────────────────────────────────────────────────
# §4  fig:dist  —  Answer distribution biases (yes/no + count)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig_dist_answer_bias ──')

vlm_b = df_mb[(df_mb['model_group'] == 'VLM') & (df_mb['variant'] == 'C')]

# Yes/No
yn_m   = vlm_b[vlm_b['answer_type'] == 'yes/no']
yn_h   = df_h[(df_h['answer_type'] == 'yes/no') & (df_h['variant'] == 'C')]
yn_m_c = yn_m[yn_m['resp_norm'].isin(['yes', 'no'])]
yn_h_c = yn_h[yn_h['resp_norm'].isin(['yes', 'no'])]
yn_gt  = sub[sub['answer_type'] == 'yes/no']['mc_answer'].str.lower().value_counts()

m_yes  = (yn_m_c['resp_norm'] == 'yes').mean()
h_yes  = (yn_h_c['resp_norm'] == 'yes').mean()
gt_yes = yn_gt.get('yes', 0) / yn_gt.sum()

# Count
ct_m = vlm_b[vlm_b['answer_type'] == 'number'].copy()
ct_h = df_h[(df_h['answer_type'] == 'number') & (df_h['variant'] == 'C')].copy()
ct_m['n'] = ct_m['resp_norm'].apply(to_int)
ct_h['n'] = ct_h['resp_norm'].apply(to_int)
ct_m = ct_m.dropna(subset=['n']); ct_m['n'] = ct_m['n'].astype(int)
ct_h = ct_h.dropna(subset=['n']); ct_h['n'] = ct_h['n'].astype(int)

bins = [0, 1, 2, 3, 4, 5]
def count_dist(s):
    t = len(s)
    return [(s == b).sum() / t * 100 for b in bins] + [(s > 5).sum() / t * 100]

m_dist = count_dist(ct_m['n'])
h_dist = count_dist(ct_h['n'])
xlabels = ['0', '1', '2', '3', '4', '5', '>5']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Answer Distribution Biases: Yes/No and Count Questions\n'
             '(VLM blind, variant C)', fontsize=13)
w = 0.35

ax = axes[0]
x = np.arange(3)
groups_yn = ['Model\n(blind, VLM)', 'Human\n(inst_blind)', 'Ground truth']
yes_vals = [m_yes * 100, h_yes * 100, gt_yes * 100]
no_vals  = [(1 - m_yes) * 100, (1 - h_yes) * 100, (1 - gt_yes) * 100]
ax.bar(x - w/2, yes_vals, w, color='#43A047', label='Yes', alpha=0.85)
ax.bar(x + w/2, no_vals,  w, color='#E53935', label='No',  alpha=0.85)
for i, (y_, n_) in enumerate(zip(yes_vals, no_vals)):
    ax.text(i - w/2, y_ + 1, f'{y_:.0f}%', ha='center', va='bottom', fontsize=10)
    ax.text(i + w/2, n_ + 1, f'{n_:.0f}%', ha='center', va='bottom', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(groups_yn, fontsize=10)
ax.set_ylabel('% of answers', fontsize=10); ax.set_ylim(0, 100)
ax.set_title('Yes/No Questions', fontsize=11)
ax.legend(fontsize=10)

ax = axes[1]
x2 = np.arange(len(xlabels))
ax.bar(x2 - w/2, m_dist, w, color='#E53935', label='Model (blind, VLM)', alpha=0.85)
ax.bar(x2 + w/2, h_dist, w, color='#1E88E5', label='Human (inst_blind)',  alpha=0.85)
for i, (mv, hv) in enumerate(zip(m_dist, h_dist)):
    if mv > 2: ax.text(i - w/2, mv + 0.5, f'{mv:.0f}%', ha='center', va='bottom', fontsize=8)
    if hv > 2: ax.text(i + w/2, hv + 0.5, f'{hv:.0f}%', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x2); ax.set_xticklabels(xlabels, fontsize=10)
ax.set_xlabel('Answer value', fontsize=10)
ax.set_ylabel('% of count answers', fontsize=10)
ax.set_title('Count / Numerical Questions', fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout()
save(fig, 'fig_dist_answer_bias.png')


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
# §6  fig:interrater  —  SBERT group-mean heatmap  (delegates to separate script)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── fig:interrater: sbert_heatmap_groups ──')
import subprocess
result = subprocess.run(
    ['conda', 'run', '-n', 'zero', 'python',
     str(ROOT / 'analysis/session2/export_agreement_heatmaps.py')],
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
print('  Fig 2  figures/fig_instruction_effect.png  (§5)')
print('  Fig 3  figures/sbert_heatmap_groups.png    (§6, via heatmap script)')
print()
print('Supporting figures:')
print('  §4  fig_dist_answer_bias.png')
print('  §6  fig_hm_alignment.png')
print('  §6  fig_hm_quadrant.png')
print('  §5  abstention_rates.png / abstention_collapse.png  (from notebook)')
