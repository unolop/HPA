"""
Export answer-distribution figures for the paper.

Produces 4 figures saved to:
  latex/AnonymousSubmission/LaTeX/figures/answer_dist/

  blind_yn.png          — yes/no distribution, blind condition
  blind_number.png      — number distribution, blind condition
  inst_blind_yn.png     — yes/no distribution, inst-blind condition
  inst_blind_number.png — number distribution, inst-blind condition

Run from repo root:
  conda run -n zero python analysis/export_answer_dist.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.vqa import VQAAnswerMapper, extract_number, bin_number
from export_helpers import read_response_exports
from config import MODELS_7B

OUT_DIR = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/answer_dist'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SOURCE_ORDER = [
    'Ground Truth', 'Humans',
    'VLM', 'VLM decoder', 'Standalone LLM', 'Standalone LLM (think)',
]

MODEL_GROUP_RENAME = {
    'VLM':                    'VLM',
    'VLM backbone decoder':   'VLM decoder',
    'standalone LLM':         'Standalone LLM',
    'standalone LLM (think)': 'Standalone LLM (think)',
}

CONDITIONS = ['blind', 'inst_blind']

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        False,
    'axes.labelsize':   11,
    'axes.titlesize':   13,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
})


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  [answer_dist_barplot] {name}')


# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading VQA annotations...')
mapper = VQAAnswerMapper()
mapper._load()

qid2atype = {
    int(qid): ann.get('answer_type', 'other')
    for qid, ann in mapper.annotations.items()
}
qid2gt = {
    int(qid): ann.get('multiple_choice_answer', '')
    for qid, ann in mapper.annotations.items()
}

print('Loading model outputs...')
exports = read_response_exports(ROOT)
blind_df = exports['model_blind']
inst_df  = exports['model_inst_blind']

print('Loading human answers...')
human_src = exports['human']
human_sub = human_src[human_src['variant'] == 'C'].copy()
human_qid_set = set(human_sub['question_id'].astype(int).unique())

model_rows = []
for cond, df_ in [('blind', blind_df), ('inst_blind', inst_df)]:
    sub = df_[df_['variant'] == 'C'].copy()
    sub = sub[sub['question_id'].astype(int).isin(human_qid_set)].copy()
    sub['source']      = sub['model_group'].map(MODEL_GROUP_RENAME)
    sub['condition']   = cond
    sub['qid']         = sub['question_id'].astype(int)
    sub['answer_type'] = sub['qid'].map(qid2atype).fillna('other')
    sub['output']      = sub['response'].fillna('').astype(str).str.strip().str.lower()
    model_rows.append(sub[['qid', 'answer_type', 'source', 'model', 'condition', 'output']])

model_df = pd.concat(model_rows, ignore_index=True)
human_qids = human_sub['question_id'].astype(int)
human_df = pd.DataFrame({
    'qid':         human_qids,
    'answer_type': human_qids.map(qid2atype).fillna('other'),
    'source':      'Humans',
    'condition':   'inst_blind',
    'output':      human_sub['response'].fillna('').astype(str).str.strip().str.lower(),
})

print('Building ground truth rows...')
all_qids = set(model_df['qid'].unique()) | set(human_df['qid'].unique())
gt_rows = [
    {
        'qid':         qid,
        'answer_type': qid2atype.get(qid, 'other'),
        'source':      'Ground Truth',
        'condition':   'both',
        'output':      str(qid2gt.get(qid, '')).strip().lower(),
    }
    for qid in all_qids
]
gt_df = pd.DataFrame(gt_rows)

all_df = pd.concat([gt_df, human_df, model_df], ignore_index=True)
print(f'Total rows (matched to human-study qids): {len(all_df)}')

N_HUMANS   = human_src['participant'].nunique()
human_c    = human_src[human_src['variant'] == 'C'].drop_duplicates('question_id')
human_c    = human_c.copy()
human_c['answer_type'] = human_c['question_id'].astype(int).map(qid2atype).fillna('other')
N_YN       = (human_c['answer_type'] == 'yes/no').sum()
N_NUM      = (human_c['answer_type'] == 'number').sum()
print(f'Human study: {N_YN} yes/no, {N_NUM} number questions, {N_HUMANS} participants')


# ── Classifiers & stack builders ──────────────────────────────────────────────
def classify_yn(s):
    s = str(s).strip()
    if s == 'yes' or s.startswith('yes '): return 'yes'
    if s == 'no'  or s.startswith('no '):  return 'no'
    return 'others'


def make_yn_stack(df, condition):
    sub = df[df['answer_type'] == 'yes/no'].copy()
    gt_mask = sub['source'] == 'Ground Truth'
    sub = sub[gt_mask | (sub['condition'] == condition)]

    sub['y/n'] = sub['output'].apply(classify_yn)
    sub['source'] = pd.Categorical(sub['source'], categories=SOURCE_ORDER, ordered=True)

    counts = sub.value_counts(['source', 'y/n']).reset_index(name='count')
    counts['proportion'] = counts.groupby('source')['count'].transform(
        lambda x: x / x.sum()
    )
    stack = counts.pivot(index='source', columns='y/n', values='proportion').fillna(0)
    return stack.reindex(SOURCE_ORDER).dropna(how='all')


def make_num_stack(df, condition):
    sub = df[df['answer_type'] == 'number'].copy()
    gt_mask = sub['source'] == 'Ground Truth'
    sub = sub[gt_mask | (sub['condition'] == condition)]

    sub['number_bin'] = sub['output'].apply(
        lambda x: bin_number(extract_number(x)) if extract_number(x) is not None else 'others'
    )
    sub['source'] = pd.Categorical(sub['source'], categories=SOURCE_ORDER, ordered=True)

    counts = sub.value_counts(['source', 'number_bin']).reset_index(name='count')
    counts['proportion'] = counts.groupby('source')['count'].transform(
        lambda x: x / x.sum()
    )
    stack = counts.pivot(index='source', columns='number_bin', values='proportion').fillna(0)
    return stack.reindex(SOURCE_ORDER).dropna(how='all')


# ── Plot functions ────────────────────────────────────────────────────────────
def plot_yn(stack_df, title=''):
    colors = {'yes': '#6BA292', 'no': '#C97A6A', 'others': '#B5B5B5'}
    cats   = [c for c in ['yes', 'no', 'others'] if c in stack_df.columns]
    n_rows = stack_df.shape[0]

    fig, ax = plt.subplots(figsize=(7, 0.5 * n_rows + 0.8))
    stack_df[cats].plot(kind='barh', stacked=True, ax=ax,
                        color=[colors[c] for c in cats], width=0.75)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylabel('')
    ax.grid(False)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    for i, src in enumerate(stack_df.index):
        cum = 0
        for cat in cats:
            val = stack_df.loc[src, cat]
            if val > 0.04:
                ax.text(cum + val / 2, i, f'{val*100:.0f}%',
                        ha='center', va='center', color='white',
                        fontsize=10, fontweight='bold')
            cum += val
    ax.legend(loc='lower right', bbox_to_anchor=(0.72, -0.22),
              ncol=3, frameon=False)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_num(stack_df, title=''):
    colors = {
        '0':       '#DADADA',
        '1':       '#BFD7EA',
        '2\u20133': '#9EC1A3',
        '4\u20135': '#7FB3A2',
        '6\u201310': '#5C8D89',
        '11\u201320': '#4C6A92',
        '>20':     '#8E5A5A',
        'others':  '#B5B5B5',
    }
    ORDER = ['0', '1', '2\u20133', '4\u20135', '6\u201310', '11\u201320', '>20', 'others']
    ORDER = [o for o in ORDER if o in stack_df.columns]
    n_rows = stack_df.shape[0]

    fig, ax = plt.subplots(figsize=(11, 0.5 * n_rows + 0.8))
    stack_df[ORDER].plot(kind='barh', stacked=True, ax=ax,
                         color=[colors[c] for c in ORDER], width=0.75)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylabel('')
    ax.grid(False)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    for i, src in enumerate(stack_df.index):
        cum = 0
        for cat in ORDER:
            val = stack_df.loc[src, cat] if cat in stack_df.columns else 0
            if val > 0.04:
                ax.text(cum + val / 2, i, f'{val*100:.0f}%',
                        ha='center', va='center', color='white',
                        fontsize=10, fontweight='bold')
            cum += val
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(bbox_to_anchor=(0.5, -0.18), loc='upper center',
              ncol=len(ORDER), frameon=False, fontsize=10)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Generate all figures ──────────────────────────────────────────────────────
COND_LABELS = {
    'blind':      'Blind',
    'inst_blind': 'Inst-Blind',
}

# Full set (all models)
for condition in CONDITIONS:
    label = COND_LABELS[condition]
    print(f'\n── {label} ──')
    fig = plot_yn(make_yn_stack(all_df, condition))
    save(fig, f'{condition}_vC_yn_q{N_YN}_h{N_HUMANS}.png')
    fig = plot_num(make_num_stack(all_df, condition))
    save(fig, f'{condition}_vC_number_q{N_NUM}_h{N_HUMANS}.png')

# 7–8 B matched-size subset
print('\n── 7B subset ──')
model_df_7b = model_df[model_df['model'].isin(MODELS_7B)]
all_df_7b = pd.concat([gt_df, human_df, model_df_7b], ignore_index=True)

for condition in CONDITIONS:
    label = COND_LABELS[condition]
    print(f'\n── {label} (7B) ──')
    fig = plot_yn(make_yn_stack(all_df_7b, condition))
    save(fig, f'{condition}_vC_yn_q{N_YN}_h{N_HUMANS}_7b.png')
    fig = plot_num(make_num_stack(all_df_7b, condition))
    save(fig, f'{condition}_vC_number_q{N_NUM}_h{N_HUMANS}_7b.png')

print('\nDone. All figures saved to:', OUT_DIR)
