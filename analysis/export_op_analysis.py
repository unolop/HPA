"""
Export operation-type (op) analysis figures for the paper.

Figures saved to figures/op_analysis/:
  acc_by_op_group.png             — accuracy by op × model group (+ human)
  acc_gap_by_op_group.png         — model − human accuracy gap (diverging bars)
  change_rate_by_op_group.png     — response change rate blind→inst by op × group
  sbert_gap_by_op_group.png       — HH − HM SBERT agreement gap by op × group
  sbert_hm_by_op_group.png        — mean HM SBERT with 95% CI error bars
  exist_response_dist.png         — response distribution for exist-op ("no" bias)
  acc_heatmap_op_group.png        — compact accuracy heatmap (op × group)
  + _7b variants for all bar charts

Run from repo root:
  conda run -n zero python analysis/export_op_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.constants import GROUP_COLORS, GROUP_ORDER
from config import MODELS_7B, FAMILY_SUBSETS

EXPORTS = ROOT / 'analysis/session2/exports'
OUT_DIR = ROOT / 'figures/op_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

HUMAN_COLOR = '#1565C0'
HH_COLOR    = '#1565C0'
MM_COLOR    = '#757575'

GROUP_SHORT = {
    'VLM':                    'VLM',
    'VLM backbone decoder':   'Backbone',
    'standalone LLM':         'LLM',
    'standalone LLM (think)': 'LLM (think)',
}

OP_LABELS = {
    'exist': 'Exist', 'know': 'Know', 'act': 'Action', 'attr': 'Attribute',
    'count': 'Count', 'spat': 'Spatial', 'temp': 'Temporal',
    'ident': 'Identify', 'comp': 'Compare', 'text': 'Text/OCR', 'cause': 'Cause',
}


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [op_analysis] {name}')


def op_xlabels(ax, op_order, n_q):
    ax.set_xticks(range(len(op_order)))
    ax.set_xticklabels(
        [f"{OP_LABELS[o]}\n(n={n_q.get(o,0)})" for o in op_order],
        fontsize=8.5)


# ── Load data ────────────────────────────────────────────────────────────────
print('Loading data…')
df_b = pd.read_csv(EXPORTS / 'responses_model_blind.csv')
df_i = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')
df_h = pd.read_csv(EXPORTS / 'responses_human.csv')
pair_df = pd.read_parquet(EXPORTS / 'pair_cache_raw.parquet')

df_b = df_b[df_b['variant'] == 'C'].copy()
df_i = df_i[df_i['variant'] == 'C'].copy()
df_h = df_h[df_h['variant'] == 'C'].copy()

h_acc_op = df_h.groupby('op')['accuracy'].mean()
OP_ORDER = [o for o in h_acc_op.sort_values(ascending=False).index if o in OP_LABELS]
n_q_per_op = df_b.drop_duplicates('question_id').groupby('op').size()

groups = [g for g in GROUP_ORDER if g in df_i['model_group'].unique()]
x = np.arange(len(OP_ORDER))

print(f'  {len(df_b)} model blind rows, {len(df_i)} inst rows, '
      f'{len(df_h)} human rows')
print(f'  Ops: {", ".join(f"{o}({n_q_per_op.get(o,0)})" for o in OP_ORDER)}')

# Pre-compute merged for change rate
merged = df_b[['question_id', 'op', 'model', 'model_group', 'response']].merge(
    df_i[['question_id', 'model', 'response']],
    on=['question_id', 'model'], suffixes=('_b', '_i'))
merged['b'] = merged['response_b'].fillna('').str.strip().str.lower()
merged['i'] = merged['response_i'].fillna('').str.strip().str.lower()
merged['changed'] = (merged['b'] != merged['i']).astype(float)

# Pre-compute pair subsets
hm = pair_df[(pair_df['pair_type'] == 'HM') & (pair_df['variant'] == 'C')].copy()
hh = pair_df[(pair_df['pair_type'] == 'HH') & (pair_df['variant'] == 'C')].copy()
hh_sbert_op = hh.groupby('op')['sbert_score'].mean()


# ═══════════════════════════════════════════════════════════════════════════
# Reusable plot functions (called once for full, once for 7B)
# ═══════════════════════════════════════════════════════════════════════════

def plot_acc_bars(df_inst, grp_list, h_vals, ylabel, suffix=''):
    """Fig: accuracy by op × group with human reference bars."""
    n_g = len(grp_list) + 1
    bw = 0.8 / n_g
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - bw * n_g / 2 + bw * 0.5, h_vals, bw,
           color=HUMAN_COLOR, alpha=0.85, label='Human',
           edgecolor='white', linewidth=0.5)
    for gi, grp in enumerate(grp_list):
        ga = df_inst[df_inst['model_group'] == grp].groupby('op')['accuracy'].mean()
        vals = [ga.get(op, 0) for op in OP_ORDER]
        ax.bar(x - bw * n_g / 2 + bw * (gi + 1.5), vals, bw,
               color=GROUP_COLORS[grp], alpha=0.85, label=GROUP_SHORT[grp],
               edgecolor='white', linewidth=0.5)
    op_xlabels(ax, OP_ORDER, n_q_per_op)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=n_g, frameon=True)
    plt.tight_layout()
    save(fig, f'acc_by_op_group{suffix}.png')


def plot_acc_gap(df_inst, grp_list, suffix=''):
    """Fig: model − human accuracy gap (diverging bars)."""
    bw = 0.8 / len(grp_list)
    fig, ax = plt.subplots(figsize=(12, 5))
    for gi, grp in enumerate(grp_list):
        ga = df_inst[df_inst['model_group'] == grp].groupby('op')['accuracy'].mean()
        deltas = [ga.get(op, 0) - h_acc_op.get(op, 0) for op in OP_ORDER]
        ax.bar(x - bw * len(grp_list) / 2 + bw * (gi + 0.5), deltas, bw,
               color=GROUP_COLORS[grp], alpha=0.85, label=GROUP_SHORT[grp],
               edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='gray', lw=0.8)
    op_xlabels(ax, OP_ORDER, n_q_per_op)
    ax.set_ylabel('Accuracy gap (model − human)', fontsize=10)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=len(grp_list), frameon=True)
    plt.tight_layout()
    save(fig, f'acc_gap_by_op_group{suffix}.png')


def plot_change_rate(merged_df, grp_list, ylabel, suffix=''):
    """Fig: response change rate blind→inst by op × group."""
    bw = 0.8 / len(grp_list)
    fig, ax = plt.subplots(figsize=(12, 5))
    for gi, grp in enumerate(grp_list):
        chg = merged_df[merged_df['model_group'] == grp].groupby('op')['changed'].mean()
        vals = [chg.get(op, 0) for op in OP_ORDER]
        ax.bar(x - bw * len(grp_list) / 2 + bw * (gi + 0.5), vals, bw,
               color=GROUP_COLORS[grp], alpha=0.85, label=GROUP_SHORT[grp],
               edgecolor='white', linewidth=0.5)
    op_xlabels(ax, OP_ORDER, n_q_per_op)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=len(grp_list), frameon=True)
    plt.tight_layout()
    save(fig, f'change_rate_by_op_group{suffix}.png')


def plot_sbert_gap(hm_df, grp_list, suffix=''):
    """Fig: HH − HM SBERT agreement gap by op × group."""
    bw = 0.8 / len(grp_list)
    fig, ax = plt.subplots(figsize=(12, 5))
    for gi, grp in enumerate(grp_list):
        hm_grp = hm_df[hm_df['subject_group_2'] == grp].groupby('op')['sbert_score'].mean()
        gaps = [hh_sbert_op.get(op, 0) - hm_grp.get(op, 0) for op in OP_ORDER]
        ax.bar(x - bw * len(grp_list) / 2 + bw * (gi + 0.5), gaps, bw,
               color=GROUP_COLORS[grp], alpha=0.85, label=GROUP_SHORT[grp],
               edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='gray', lw=0.8)
    op_xlabels(ax, OP_ORDER, n_q_per_op)
    ax.set_ylabel('SBERT agreement gap (HH − HM)', fontsize=10)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=len(grp_list), frameon=True)
    plt.tight_layout()
    save(fig, f'sbert_gap_by_op_group{suffix}.png')


def plot_sbert_hm_ci(hm_df, grp_list, suffix=''):
    """Fig: Mean HM SBERT by op × group with 95% CI error bars + HH ceiling."""
    bw = 0.8 / (len(grp_list) + 1)   # +1 for HH
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # HH ceiling
    hh_means = [hh_sbert_op.get(op, np.nan) for op in OP_ORDER]
    hh_ci = []
    for op in OP_ORDER:
        vals = hh[hh['op'] == op]['sbert_score'].dropna()
        if len(vals) > 2:
            se = vals.std() / np.sqrt(len(vals))
            hh_ci.append(1.96 * se)
        else:
            hh_ci.append(0)
    ax.bar(x - bw * (len(grp_list) + 1) / 2 + bw * 0.5,
           hh_means, bw, yerr=hh_ci, capsize=2,
           color=HH_COLOR, alpha=0.75, label='Human–Human',
           edgecolor='white', linewidth=0.5,
           error_kw=dict(lw=1, capthick=1, color='#333'))

    # HM per group
    for gi, grp in enumerate(grp_list):
        sub = hm_df[hm_df['subject_group_2'] == grp]
        means, cis = [], []
        for op in OP_ORDER:
            vals = sub[sub['op'] == op]['sbert_score'].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            if len(vals) > 2:
                se = vals.std() / np.sqrt(len(vals))
                cis.append(1.96 * se)
            else:
                cis.append(0)
        ax.bar(x - bw * (len(grp_list) + 1) / 2 + bw * (gi + 1.5),
               means, bw, yerr=cis, capsize=2,
               color=GROUP_COLORS[grp], alpha=0.85, label=GROUP_SHORT[grp],
               edgecolor='white', linewidth=0.5,
               error_kw=dict(lw=1, capthick=1, color='#333'))

    op_xlabels(ax, OP_ORDER, n_q_per_op)
    ax.set_ylabel('Mean SBERT cosine (HM / HH)', fontsize=10)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=len(grp_list) + 1, frameon=True)
    plt.tight_layout()
    save(fig, f'sbert_hm_by_op_group{suffix}.png')


# ═══════════════════════════════════════════════════════════════════════════
# Generate: FULL models
# ═══════════════════════════════════════════════════════════════════════════
h_vals = [h_acc_op.get(op, 0) for op in OP_ORDER]

print('\n══ Full models ══')
plot_acc_bars(df_i, groups, h_vals, 'Mean accuracy (blind, variant C)')
plot_acc_gap(df_i, groups)
plot_change_rate(merged, groups, 'Response change rate (blind → inst_blind)')
plot_sbert_gap(hm, groups)
plot_sbert_hm_ci(hm, groups)


# ═══════════════════════════════════════════════════════════════════════════
# Generate: 7B subset
# ═══════════════════════════════════════════════════════════════════════════
print('\n══ 7B subset ══')
df_i_7b = df_i[df_i['model'].isin(MODELS_7B)]
merged_7b = merged[merged['model'].isin(MODELS_7B)]
hm_7b = hm[hm['subject_2'].isin(MODELS_7B)]
groups_7b = [g for g in GROUP_ORDER if g in df_i_7b['model_group'].unique()]

plot_acc_bars(df_i_7b, groups_7b, h_vals,
             'Mean accuracy (blind, variant C, 7–8B)', suffix='_7b')
plot_acc_gap(df_i_7b, groups_7b, suffix='_7b')
plot_change_rate(merged_7b, groups_7b,
                 'Response change rate (7–8B)', suffix='_7b')
plot_sbert_gap(hm_7b, groups_7b, suffix='_7b')
plot_sbert_hm_ci(hm_7b, groups_7b, suffix='_7b')


# ═══════════════════════════════════════════════════════════════════════════
# Exist-op response distribution (with question count)
# ═══════════════════════════════════════════════════════════════════════════
print('\n── exist response distribution ──')

exist_b = df_b[df_b['op'] == 'exist'].copy()
exist_b['resp'] = exist_b['response'].fillna('').str.strip().str.lower()
n_exist_q = exist_b['question_id'].nunique()
n_exist_models = exist_b['model'].nunique()

def classify_exist(s):
    if s == 'yes' or s.startswith('yes '): return 'yes'
    if s == 'no'  or s.startswith('no '):  return 'no'
    return 'other'

exist_b['yn'] = exist_b['resp'].apply(classify_exist)

exist_h = df_h[df_h['op'] == 'exist'].copy()
exist_h['resp'] = exist_h['response'].fillna('').str.strip().str.lower()
exist_h['yn'] = exist_h['resp'].apply(classify_exist)
n_exist_humans = exist_h['participant'].nunique() if 'participant' in exist_h.columns else '?'

gt_exist = exist_b.drop_duplicates('question_id').copy()
gt_exist['yn'] = gt_exist['gt'].fillna('').str.strip().str.lower().apply(classify_exist)

sources = ['Ground Truth', 'Human'] + [GROUP_SHORT[g] for g in groups]
cats = ['yes', 'no', 'other']
colors_yn = {'yes': '#6BA292', 'no': '#C97A6A', 'other': '#B5B5B5'}

props = {}
gt_counts = gt_exist['yn'].value_counts(normalize=True)
props['Ground Truth'] = {c: gt_counts.get(c, 0) for c in cats}
h_counts = exist_h['yn'].value_counts(normalize=True)
props['Human'] = {c: h_counts.get(c, 0) for c in cats}
for grp in groups:
    sub = exist_b[exist_b['model_group'] == grp]
    c_ = sub['yn'].value_counts(normalize=True)
    props[GROUP_SHORT[grp]] = {c: c_.get(c, 0) for c in cats}

prop_df = pd.DataFrame(props).T.reindex(sources)[cats]
n_rows = len(prop_df)

fig, ax = plt.subplots(figsize=(7, 0.6 * n_rows + 1.0))
prop_df.plot(kind='barh', stacked=True, ax=ax,
             color=[colors_yn[c] for c in cats], width=0.75)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_ylabel('')
ax.set_xlabel('Proportion', fontsize=10)
ax.grid(False)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
for i, src in enumerate(prop_df.index):
    cum = 0
    for cat in cats:
        val = prop_df.loc[src, cat]
        if val > 0.04:
            ax.text(cum + val / 2, i, f'{val*100:.0f}%',
                    ha='center', va='center', color='white',
                    fontsize=10, fontweight='bold')
        cum += val
ax.legend(loc='lower right', bbox_to_anchor=(0.72, -0.22),
          ncol=3, frameon=False)
plt.tight_layout()
save(fig, f'exist_response_dist_q{n_exist_q}.png')


# ═══════════════════════════════════════════════════════════════════════════
# Accuracy heatmap (op × group)
# ═══════════════════════════════════════════════════════════════════════════
print('\n── accuracy heatmap ──')

heat_data = {}
heat_data['Human'] = {OP_LABELS[op]: h_acc_op.get(op, np.nan) for op in OP_ORDER}
for grp in groups:
    grp_acc = df_i[df_i['model_group'] == grp].groupby('op')['accuracy'].mean()
    heat_data[GROUP_SHORT[grp]] = {
        OP_LABELS[op]: grp_acc.get(op, np.nan) for op in OP_ORDER
    }

heat_df = pd.DataFrame(heat_data).T
row_order = ['Human'] + [GROUP_SHORT[g] for g in groups]
heat_df = heat_df.reindex(row_order)

fig, ax = plt.subplots(figsize=(10, 3.5))
vmin, vmax = 0, max(heat_df.max().max(), 0.95)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
im = ax.imshow(heat_df.values, cmap='RdYlGn', norm=norm, aspect='auto')

ax.set_xticks(range(len(OP_ORDER)))
ax.set_xticklabels([f"{OP_LABELS[o]}\n({n_q_per_op.get(o,0)})" for o in OP_ORDER],
                    fontsize=8)
ax.set_yticks(range(len(row_order)))
ax.set_yticklabels(row_order, fontsize=9)

tick_colors = [HUMAN_COLOR] + [GROUP_COLORS[g] for g in groups]
for tick, c in zip(ax.get_yticklabels(), tick_colors):
    tick.set_color(c)
    tick.set_fontweight('bold')

for i in range(heat_df.shape[0]):
    for j in range(heat_df.shape[1]):
        v = heat_df.values[i, j]
        if not np.isnan(v):
            text_color = 'white' if v < 0.35 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=text_color)

plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
plt.tight_layout()
save(fig, 'acc_heatmap_op_group.png')


# ═══════════════════════════════════════════════════════════════════════════
# Family-matched subsets (same backbone, different modes)
# ═══════════════════════════════════════════════════════════════════════════
for fam_key, (fam_label, fam_models) in FAMILY_SUBSETS.items():
    suffix = f'_{fam_key}'

    print(f'\n══ {fam_label} ══')
    df_i_fam    = df_i[df_i['model'].isin(fam_models)]
    merged_fam  = merged[merged['model'].isin(fam_models)]
    hm_fam      = hm[hm['subject_2'].isin(fam_models)]
    groups_fam  = [g for g in GROUP_ORDER if g in df_i_fam['model_group'].unique()]

    if not groups_fam:
        print(f'  no models found, skipping')
        continue

    plot_acc_bars(df_i_fam, groups_fam, h_vals,
                  f'Mean accuracy ({fam_label})', suffix=suffix)
    plot_acc_gap(df_i_fam, groups_fam, suffix=suffix)
    plot_change_rate(merged_fam, groups_fam,
                     f'Response change rate ({fam_label})', suffix=suffix)
    plot_sbert_gap(hm_fam, groups_fam, suffix=suffix)
    plot_sbert_hm_ci(hm_fam, groups_fam, suffix=suffix)


print(f'\nDone. All figures saved to: {OUT_DIR}')
