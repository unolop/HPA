"""
Export corpus-frequency analysis figures.

Compares blind model answer distributions against VQA v2.0 training-set
answer frequencies to test whether blind biases are explained by corpus
frequency (McCoy et al., 2024).

Outputs saved to figures/corpus_frequency/:
  comparison_bar.png          — side-by-side bar chart (model vs train dist)
  group_correlation_hm.png    — group x answer_type Spearman rho heatmap
  yn_no_rate.png              — "no" rate per model vs training baseline
  number_zero_rate.png        — "0" rate per model vs training baseline
  freq_match_scatter.png      — % answers matching train-set mode per question_type

Run from repo root:
  conda run -n zero python figures/corpus_frequency.py
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from helpers import clear_output_plots
from analysis.utils.model_registry import (
    default_all_models, backbone_models, backbone_think_models, MODEL_TYPE,
)
from analysis.utils.vqa import preprocess_answer, extract_number, bin_number

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

OUT_DIR = ROOT / 'figures' / 'corpus_frequency'
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Paths ────────────────────────────────────────────────────────────────────
VQA_DIR = ROOT / 'dataset' / 'vqa'
TRAIN_ANNOT = VQA_DIR / 'v2_mscoco_train2014_annotations.json'
VAL_ANNOT = VQA_DIR / 'v2_mscoco_val2014_annotations.json'
VQA_1K = VQA_DIR / 'vqav2_1k_val.json'

GROUP_COLORS = {
    'VLM': '#E53935',
    'VLM backbone decoder': '#E67E22',
    'standalone LLM': '#2E7D32',
    'standalone LLM (think)': '#8E24AA',
}
GROUP_SHORT = {
    'VLM': 'VLM',
    'VLM backbone decoder': 'Backbone',
    'standalone LLM': 'SA-LLM',
    'standalone LLM (think)': 'SA-LLM (think)',
}
GROUP_ORDER = ['VLM', 'VLM backbone decoder', 'standalone LLM', 'standalone LLM (think)']

# ── Load annotations ─────────────────────────────────────────────────────────
print("Loading VQA annotations...")


def load_annotations(path):
    with open(path) as f:
        return json.load(f)['annotations']


train_annot = load_annotations(TRAIN_ANNOT)
val_annot = load_annotations(VAL_ANNOT)
print(f"  Train: {len(train_annot):,} | Val: {len(val_annot):,}")


def build_answer_freq(annots):
    freq = {}
    for a in annots:
        atype = a['answer_type']
        ans = a['multiple_choice_answer'].lower().strip()
        if atype not in freq:
            freq[atype] = Counter()
        freq[atype][ans] += 1
    return freq


def freq_to_prob(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}


train_freq = build_answer_freq(train_annot)
val_freq = build_answer_freq(val_annot)

# ── Load 1K question metadata ───────────────────────────────────────────────
vqa1k = json.load(open(VQA_1K))
qid_to_atype = {q['question_id']: q['answer_type'] for q in vqa1k}
qid_to_qtype = {q['question_id']: q['question_type'] for q in vqa1k}

# ── Question counts by answer type ─────────────────────────────────────────
N_YN_1K  = sum(1 for at in qid_to_atype.values() if at == 'yes/no')
N_NUM_1K = sum(1 for at in qid_to_atype.values() if at == 'number')
N_YN_TRAIN  = sum(1 for a in train_annot if a['answer_type'] == 'yes/no')
N_NUM_TRAIN = sum(1 for a in train_annot if a['answer_type'] == 'number')
print(f"  1K pool:  {N_YN_1K} yes/no, {N_NUM_1K} number questions")
print(f"  Training: {N_YN_TRAIN:,} yes/no, {N_NUM_TRAIN:,} number annotations")

# ── Load model answers (full 1K, variant C, inst_blind) ─────────────────────
print("Loading model answers...")


def load_all_model_answers(condition='_control_inst_blind'):
    """Load blind answers for all models. Returns {name: (group, {qid: answer})}."""
    models = {}
    all_m = default_all_models(ROOT)
    # Also add backbone standalone LLMs + think variants
    all_m.update(backbone_models(ROOT))
    all_m.update(backbone_think_models(ROOT))

    for name, (base_dir, subdir) in all_m.items():
        jsonl = base_dir / subdir / f'vqa_1k{condition}.jsonl'
        if not jsonl.exists():
            continue
        group = MODEL_TYPE.get(name)
        if group is None:
            continue
        answers = {}
        for line in open(jsonl):
            ex = json.loads(line)
            qid = int(ex['question_id'])
            ga = ex.get('generated_answers', {})
            if ga:
                ans = preprocess_answer(ga.get('question', ''), strip_think=True)
            else:
                ans = preprocess_answer(ex.get('generated_answer', ''), strip_think=True)
            if ans:
                answers[qid] = ans
        if answers:
            models[name] = (group, answers)
    return models


all_models = load_all_model_answers()
print(f"  Loaded {len(all_models)} models")


def model_answer_dist(answers, answer_type):
    counter = Counter()
    for qid, ans in answers.items():
        if qid_to_atype.get(qid) == answer_type:
            counter[ans] += 1
    return counter


# ── Build question_type -> most frequent training answer ─────────────────────
qtype_freq = {}
for a in train_annot:
    qt = a['question_type']
    ans = a['multiple_choice_answer'].lower().strip()
    if qt not in qtype_freq:
        qtype_freq[qt] = Counter()
    qtype_freq[qt][ans] += 1

qtype_top_answer = {}
for qt, counter in qtype_freq.items():
    top_ans, top_cnt = counter.most_common(1)[0]
    qtype_top_answer[qt] = (top_ans, top_cnt / sum(counter.values()))


# ── Correlation computation ──────────────────────────────────────────────────
def correlate_distributions(model_counter, train_counter, top_k=20):
    top_m = {a for a, _ in model_counter.most_common(top_k)}
    top_t = {a for a, _ in train_counter.most_common(top_k)}
    vocab = top_m | top_t
    if len(vocab) < 3:
        return np.nan, np.nan
    m_prob = freq_to_prob(model_counter)
    t_prob = freq_to_prob(train_counter)
    mv = [m_prob.get(a, 0) for a in sorted(vocab)]
    tv = [t_prob.get(a, 0) for a in sorted(vocab)]
    rho, p = spearmanr(mv, tv)
    return rho, p


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Stacked horizontal bars (answer_dist style) — model groups + train
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Figure 1: stacked bar comparison ──")

SOURCE_ORDER_CF = [
    'VQA v2.0 Training',
    'VLM', 'VLM decoder', 'Standalone LLM', 'Standalone LLM (think)',
]

MODEL_GROUP_RENAME_CF = {
    'VLM':                    'VLM',
    'VLM backbone decoder':   'VLM decoder',
    'standalone LLM':         'Standalone LLM',
    'standalone LLM (think)': 'Standalone LLM (think)',
}


def classify_yn(s):
    s = str(s).strip().lower()
    if s == 'yes' or s.startswith('yes '):
        return 'yes'
    if s == 'no' or s.startswith('no '):
        return 'no'
    return 'others'


def _build_train_rows(annots, answer_type, label='VQA v2.0 Training'):
    """Build per-answer rows from VQA training annotations for one answer_type."""
    rows = []
    for a in annots:
        if a['answer_type'] != answer_type:
            continue
        rows.append({
            'source': label,
            'output': a['multiple_choice_answer'].lower().strip(),
        })
    return rows


def _build_model_group_rows(all_models_dict, answer_type, qid_subset=None):
    """Build per-answer rows aggregated by model group (each model = 1 "respondent")."""
    rows = []
    for name, (group, answers) in all_models_dict.items():
        grp_label = MODEL_GROUP_RENAME_CF.get(group)
        if grp_label is None:
            continue
        for qid, ans in answers.items():
            if qid_to_atype.get(qid) != answer_type:
                continue
            if qid_subset is not None and qid not in qid_subset:
                continue
            rows.append({'source': grp_label, 'output': ans})
    return rows


def _make_stack(rows, source_order, classify_fn, col_name):
    df = pd.DataFrame(rows)
    df[col_name] = df['output'].apply(classify_fn)
    df['source'] = pd.Categorical(df['source'], categories=source_order, ordered=True)
    counts = df.value_counts(['source', col_name]).reset_index(name='count')
    counts['proportion'] = counts.groupby('source')['count'].transform(
        lambda x: x / x.sum())
    stack = counts.pivot(index='source', columns=col_name, values='proportion').fillna(0)
    return stack.reindex(source_order).dropna(how='all')


def _model_group_labels(n_q):
    """Model group labels with question count appended."""
    return {g: f'{g} (q={n_q})' for g in SOURCE_ORDER_CF[1:]}


def make_cf_yn_stack():
    train_label = f'VQA v2.0 Training (n={N_YN_TRAIN:,})'
    grp_labels = _model_group_labels(N_YN_1K)
    order = [train_label] + [grp_labels[g] for g in SOURCE_ORDER_CF[1:]]
    rows = _build_train_rows(train_annot, 'yes/no', label=train_label)
    rows += _build_model_group_rows(all_models, 'yes/no')
    # rename model group sources
    for r in rows:
        if r['source'] in grp_labels:
            r['source'] = grp_labels[r['source']]
    return _make_stack(rows, order, classify_yn, 'y/n')


def make_cf_num_stack():
    train_label = f'VQA v2.0 Training (n={N_NUM_TRAIN:,})'
    grp_labels = _model_group_labels(N_NUM_1K)
    order = [train_label] + [grp_labels[g] for g in SOURCE_ORDER_CF[1:]]
    rows = _build_train_rows(train_annot, 'number', label=train_label)
    rows += _build_model_group_rows(all_models, 'number')
    for r in rows:
        if r['source'] in grp_labels:
            r['source'] = grp_labels[r['source']]
    classify_num = lambda x: bin_number(extract_number(x)) if extract_number(x) is not None else 'others'
    return _make_stack(rows, order, classify_num, 'number_bin')


def plot_cf_yn(stack_df):
    colors = {'yes': '#6BA292', 'no': '#C97A6A', 'others': '#B5B5B5'}
    cats = [c for c in ['yes', 'no', 'others'] if c in stack_df.columns]
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
    plt.tight_layout()
    return fig


def plot_cf_num(stack_df):
    colors = {
        '0': '#DADADA', '1': '#BFD7EA', '2\u20133': '#9EC1A3',
        '4\u20135': '#7FB3A2', '6\u201310': '#5C8D89',
        '11\u201320': '#4C6A92', '>20': '#8E5A5A', 'others': '#B5B5B5',
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
    plt.tight_layout()
    return fig


fig_yn = plot_cf_yn(make_cf_yn_stack())
yn_fname = f'comparison_yn_q{N_YN_1K}.png'
fig_yn.savefig(OUT_DIR / yn_fname, dpi=220, bbox_inches='tight')
plt.close(fig_yn)
print(f"  saved: {yn_fname}")

fig_num = plot_cf_num(make_cf_num_stack())
num_fname = f'comparison_number_q{N_NUM_1K}.png'
fig_num.savefig(OUT_DIR / num_fname, dpi=220, bbox_inches='tight')
plt.close(fig_num)
print(f"  saved: {num_fname}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Group correlation heatmap
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Figure 2: group correlation heatmap ──")

corr_rows = []
for name, (group, answers) in all_models.items():
    for atype in ['yes/no', 'number', 'other']:
        dist = model_answer_dist(answers, atype)
        if sum(dist.values()) < 10:
            continue
        rho, p = correlate_distributions(dist, train_freq[atype])
        corr_rows.append({
            'model': name, 'group': GROUP_SHORT[group],
            'answer_type': atype, 'rho': rho,
        })

df_corr = pd.DataFrame(corr_rows)
group_pivot = df_corr.pivot_table(
    index='group', columns='answer_type', values='rho', aggfunc='mean')
group_pivot = group_pivot[['yes/no', 'number', 'other']]
group_pivot['mean'] = group_pivot.mean(axis=1)

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(group_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            vmin=-0.5, vmax=1.0, ax=ax, linewidths=0.5)
ax.set_title('Spearman rho: blind model dist vs VQA train dist', fontsize=11)
ax.set_ylabel('')
plt.tight_layout()
fig.savefig(OUT_DIR / 'group_correlation_hm.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  saved: group_correlation_hm.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: "No" rate per model (yes/no questions)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Figure 3: yes/no 'no' rate ──")

train_no_rate = train_freq['yes/no']['no'] / sum(train_freq['yes/no'].values())

rows = []
for name, (group, answers) in all_models.items():
    dist = model_answer_dist(answers, 'yes/no')
    total = sum(dist.values())
    if total < 10:
        continue
    rows.append({
        'model': name, 'group': group,
        'no_rate': dist.get('no', 0) / total,
    })

df_yn = pd.DataFrame(rows).sort_values('no_rate', ascending=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(df_yn) * 0.28)))
colors = [GROUP_COLORS[r['group']] for _, r in df_yn.iterrows()]
ax.barh(range(len(df_yn)), df_yn['no_rate'], color=colors, alpha=0.85)
ax.axvline(train_no_rate, color='#1565C0', ls='--', lw=2, label=f'VQA Train ({train_no_rate:.1%})')
ax.axvline(0.5, color='gray', ls=':', lw=1, alpha=0.5)
ax.set_yticks(range(len(df_yn)))
ax.set_yticklabels(df_yn['model'], fontsize=7)
ax.set_xlabel('"No" rate (yes/no questions)')
ax.set_title('Blind "no" bias vs VQA training frequency', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / 'yn_no_rate.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  saved: yn_no_rate.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: "0" rate per model (number questions)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Figure 4: number '0' rate ──")

train_zero_rate = train_freq['number']['0'] / sum(train_freq['number'].values())

rows = []
for name, (group, answers) in all_models.items():
    dist = model_answer_dist(answers, 'number')
    total = sum(dist.values())
    if total < 10:
        continue
    rows.append({
        'model': name, 'group': group,
        'zero_rate': dist.get('0', 0) / total,
    })

df_num = pd.DataFrame(rows).sort_values('zero_rate', ascending=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(df_num) * 0.28)))
colors = [GROUP_COLORS[r['group']] for _, r in df_num.iterrows()]
ax.barh(range(len(df_num)), df_num['zero_rate'], color=colors, alpha=0.85)
ax.axvline(train_zero_rate, color='#1565C0', ls='--', lw=2, label=f'VQA Train ({train_zero_rate:.1%})')
ax.set_yticks(range(len(df_num)))
ax.set_yticklabels(df_num['model'], fontsize=7)
ax.set_xlabel('"0" rate (number questions)')
ax.set_title('Blind "0" bias vs VQA training frequency', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / 'number_zero_rate.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  saved: number_zero_rate.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Frequency match rate (% answers matching training mode per qtype)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Figure 5: frequency match rate scatter ──")

rows = []
for name, (group, answers) in all_models.items():
    matches = 0
    total = 0
    for qid, ans in answers.items():
        qt = qid_to_qtype.get(qid)
        if qt and qt in qtype_top_answer:
            total += 1
            if ans == qtype_top_answer[qt][0]:
                matches += 1
    if total > 0:
        rows.append({
            'model': name, 'group': group,
            'freq_match': matches / total, 'n': total,
        })

df_match = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(8, 5))
for g in GROUP_ORDER:
    sub = df_match[df_match['group'] == g]
    if sub.empty:
        continue
    ax.scatter(sub.index, sub['freq_match'], c=GROUP_COLORS[g],
               label=GROUP_SHORT[g], s=60, alpha=0.8, edgecolors='white', lw=0.5)

ax.set_ylabel('Frequency match rate')
ax.set_title('% blind answers matching VQA training-set mode (per question_type)', fontsize=11)
ax.legend(fontsize=9)
ax.set_xticks(range(len(df_match)))
ax.set_xticklabels(df_match['model'], rotation=90, fontsize=6)
plt.tight_layout()
fig.savefig(OUT_DIR / 'freq_match_scatter.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  saved: freq_match_scatter.png")

print(f"\nAll figures saved to {OUT_DIR}")
