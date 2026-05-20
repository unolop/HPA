"""
Export figures for image-condition ablation analysis.

Compares model behaviour when shown different degraded images:
  blank (224×224 blank PNG) — standard blind condition
  gray  (uniform gray)
  noise (random pixel noise)
  white (uniform white)       [Qwen3-VL-4B only]

Models with ablation data:
  Qwen3-VL-8B  (vlm/)      — blank, gray, noise
  Qwen3-VL-4B  (ablation/) — blank, gray, noise, white

Key findings encoded in analysis:
  • Qwen3-VL-4B outputs are IDENTICAL across all four conditions
    (the model completely ignores image content).
  • Qwen3-VL-8B accuracy drops with image degradation
    (blank > gray > noise), and ~38–51% of answers change.

Figures saved to:
  latex/AnonymousSubmission/LaTeX/figures/ablation_image/

Run from repo root:
  conda run -n zero python analysis/export_ablation_image.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import sem as scipy_sem

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.vqa import preprocess_answer, VQAAnswerMapper, vqa_accuracy

OUT_DIR = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/ablation_image'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGITS = ROOT / 'evaluation/logits'

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          False,
})

COND_COLORS = {
    'blank': '#455A64',
    'gray':  '#78909C',
    'noise': '#E53935',
    'white': '#FB8C00',
}
COND_LABEL = {
    'blank': 'Blank',
    'gray':  'Gray',
    'noise': 'Noise',
    'white': 'White',
}

# ── File registry ─────────────────────────────────────────────────────────────
MODELS = {
    'Qwen3-VL-8B': {
        'answer_key': 'question',          # variant C key in generated_answers
        'conditions': {
            'blank': LOGITS / 'vlm/pretrained/Qwen3-VL-8B-Instruct/vqa_1k_control_blind.jsonl',
            'gray':  LOGITS / 'vlm/pretrained/Qwen3-VL-8B-Instruct/vqa_1k_control_blind_gray.jsonl',
            'noise': LOGITS / 'vlm/pretrained/Qwen3-VL-8B-Instruct/vqa_1k_control_blind_noise.jsonl',
        },
    },
    'Qwen3-VL-4B': {
        'answer_key': 'question',
        'conditions': {
            'blank': LOGITS / 'ablation/pretrained/Qwen3-VL-4B-Instruct/vqa_1k_blind.jsonl',
            'gray':  LOGITS / 'ablation/pretrained/Qwen3-VL-4B-Instruct/vqa_1k_blind_gray.jsonl',
            'noise': LOGITS / 'ablation/pretrained/Qwen3-VL-4B-Instruct/vqa_1k_blind_noise.jsonl',
            'white': LOGITS / 'ablation/pretrained/Qwen3-VL-4B-Instruct/vqa_1k_blind_white.jsonl',
        },
    },
}

SOFT_ABSTAIN = {'nothing', 'none', 'nowhere', 'unanswerable', 'unknown',
                'no answer', 'no image', 'no picture', "i don't know",
                "i'm sorry", 'i cannot', 'i can not', 'i am unable',
                'i am not able', 'unable to'}


def is_soft_abstain(ans: str) -> bool:
    a = ans.lower().strip()
    return any(a.startswith(tok) for tok in SOFT_ABSTAIN)


def load_condition(path: Path, answer_key: str, mapper: VQAAnswerMapper) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            ex = json.loads(line.strip())
            qid = int(ex['question_id'])
            raw = ex.get('generated_answers', {}).get(answer_key, '')
            ans = preprocess_answer(str(raw), strip_think=True)
            gt  = mapper.get_answers(qid)
            acc = vqa_accuracy(ans, gt) if gt else float('nan')
            rows.append({
                'qid':       qid,
                'answer':    ans,
                'raw':       str(raw),
                'acc':       acc,
                'atype':     ex.get('answer_type', 'other'),
                'is_abstain': is_soft_abstain(ans),
                'is_loop':   '<|im_start|>' in str(raw),
            })
    return pd.DataFrame(rows)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [ablation_image] {name}')


# ── Load all data ─────────────────────────────────────────────────────────────
print('Loading VQA annotations…')
mapper = VQAAnswerMapper()

data: dict[str, dict[str, pd.DataFrame]] = {}
for model, cfg in MODELS.items():
    print(f'\nLoading {model}…')
    data[model] = {}
    for cond, path in cfg['conditions'].items():
        df = load_condition(path, cfg['answer_key'], mapper)
        data[model][cond] = df
        print(f'  {cond}: n={len(df)}  mean_acc={df["acc"].mean():.4f}  '
              f'abstain={df["is_abstain"].mean():.3f}  loop={df["is_loop"].mean():.3f}')

# ── Fig 1: Accuracy by image condition ───────────────────────────────────────
# One grouped bar cluster per model; bars = conditions
print('\n── Fig 1: accuracy by condition ──')

models_plot = list(MODELS.keys())
all_conds   = ['blank', 'gray', 'noise', 'white']

fig, ax = plt.subplots(figsize=(8, 4.5))

n_models = len(models_plot)
n_conds  = len(all_conds)
x = np.arange(n_models)
w = 0.18

for ci, cond in enumerate(all_conds):
    accs = []
    for model in models_plot:
        df = data[model].get(cond)
        accs.append(df['acc'].mean() if df is not None else float('nan'))
    offset = (ci - (n_conds - 1) / 2) * w
    bars = ax.bar(x + offset, accs, w * 0.9,
                  color=COND_COLORS[cond], label=COND_LABEL[cond], alpha=0.85)
    for bar, val in zip(bars, accs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(models_plot, fontsize=10)
ax.set_ylabel('Mean VQA accuracy (variant C)', fontsize=10)
ax.set_title('Accuracy by Image Condition\n'
             '(blind = model sees degraded image instead of real image)', fontsize=10)
ax.legend(fontsize=9, frameon=True, loc='upper right')
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
plt.tight_layout()
save(fig, 'accuracy_by_condition.png')

# ── Fig 2: Answer change rate vs blank baseline ───────────────────────────────
print('── Fig 2: answer change rate ──')

fig, ax = plt.subplots(figsize=(7, 4))

for mi, model in enumerate(models_plot):
    blank_df = data[model]['blank'].set_index('qid')['answer']
    conds_nonblank = [c for c in all_conds if c != 'blank' and c in data[model]]
    xs, ys, colors = [], [], []
    for cond in conds_nonblank:
        other_df = data[model][cond].set_index('qid')['answer']
        common   = blank_df.index.intersection(other_df.index)
        changed  = (blank_df[common] != other_df[common]).mean()
        xs.append(f'{model}\n→ {COND_LABEL[cond]}')
        ys.append(changed)
        colors.append(COND_COLORS[cond])

    bars = ax.bar(
        [f'{mi * (len(conds_nonblank) + 1) + j}' for j in range(len(xs))],
        ys, color=colors, alpha=0.85, width=0.7,
    )
    for bar, val, label in zip(bars, ys, xs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8.5)

# Rebuild x-axis labels cleanly
tick_labels = []
tick_pos    = []
for mi, model in enumerate(models_plot):
    conds_nonblank = [c for c in all_conds if c != 'blank' and c in data[model]]
    for j, cond in enumerate(conds_nonblank):
        pos = mi * (len(conds_nonblank) + 1) + j
        tick_pos.append(str(pos))
        tick_labels.append(f'{model}\n→ {COND_LABEL[cond]}')

ax.set_xticks(range(len(tick_pos)))
ax.set_xticklabels(tick_labels, fontsize=8)
ax.set_ylabel('Fraction of answers changed vs blank', fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_title('Answer Change Rate (vs blank baseline)', fontsize=10)

legend_patches = [mpatches.Patch(color=COND_COLORS[c], label=COND_LABEL[c])
                  for c in all_conds if c != 'blank']
ax.legend(handles=legend_patches, fontsize=9, frameon=True, loc='upper right')
plt.tight_layout()
save(fig, 'answer_change_rate.png')

# ── Fig 3: Accuracy by question type per condition (8B) ──────────────────────
print('── Fig 3: accuracy by answer type (8B) ──')

model = 'Qwen3-VL-8B'
atypes = ['yes/no', 'number', 'other']
conds  = [c for c in all_conds if c in data[model]]

atype_accs = {cond: {} for cond in conds}
for cond in conds:
    df = data[model][cond]
    for at in atypes:
        sub = df[df['atype'] == at]
        atype_accs[cond][at] = sub['acc'].mean() if len(sub) else float('nan')

fig, ax = plt.subplots(figsize=(7.5, 4.5))
n_at = len(atypes)
x    = np.arange(n_at)
w    = 0.22
for ci, cond in enumerate(conds):
    vals   = [atype_accs[cond].get(at, float('nan')) for at in atypes]
    offset = (ci - (len(conds) - 1) / 2) * w
    bars = ax.bar(x + offset, vals, w * 0.9,
                  color=COND_COLORS[cond], label=COND_LABEL[cond], alpha=0.85)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(['Yes / No', 'Number', 'Other (open)'], fontsize=10)
ax.set_ylabel('Mean VQA accuracy', fontsize=10)
ax.set_title(f'Qwen3-VL-8B — Accuracy by Question Type and Image Condition', fontsize=10)
ax.legend(fontsize=9, frameon=True, loc='upper right')
ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
plt.tight_layout()
save(fig, 'qwen3vl8b_accuracy_by_qtype.png')

# ── Fig 4: 4B identical-output confirmation ───────────────────────────────────
print('── Fig 4: 4B output identity check ──')

model = 'Qwen3-VL-4B'
conds_4b = [c for c in all_conds if c in data[model]]
blank_ans = data[model]['blank'].set_index('qid')['answer']

fig, ax = plt.subplots(figsize=(6.5, 3.5))
same_rates = []
for cond in conds_4b:
    if cond == 'blank':
        same_rates.append(1.0)
        continue
    other = data[model][cond].set_index('qid')['answer']
    common = blank_ans.index.intersection(other.index)
    same_rates.append((blank_ans[common] == other[common]).mean())

colors = [COND_COLORS[c] for c in conds_4b]
bars = ax.bar(range(len(conds_4b)), same_rates, color=colors, alpha=0.85, width=0.5)
for bar, val, cond in zip(bars, same_rates, conds_4b):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(conds_4b)))
ax.set_xticklabels([COND_LABEL[c] for c in conds_4b], fontsize=10)
ax.set_ylabel('Fraction identical to blank baseline', fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_title('Qwen3-VL-4B — Image Condition Has No Effect\n'
             '(all outputs identical to blank regardless of image)', fontsize=10)
ax.axhline(1.0, color='gray', lw=1, ls='--', alpha=0.5)
plt.tight_layout()
save(fig, 'qwen3vl4b_image_invariance.png')

# ── Summary table ──────────────────────────────────────────────────────────────
print('\n── Summary ──')
rows = []
for model in models_plot:
    blank_ans = data[model]['blank'].set_index('qid')['answer']
    for cond in all_conds:
        if cond not in data[model]:
            continue
        df    = data[model][cond]
        other = df.set_index('qid')['answer']
        common = blank_ans.index.intersection(other.index)
        changed = float('nan') if cond == 'blank' else (blank_ans[common] != other[common]).mean()
        rows.append({
            'Model': model,
            'Condition': COND_LABEL[cond],
            'Mean acc': round(df['acc'].mean(), 4),
            'Abstain %': round(df['is_abstain'].mean() * 100, 1),
            'Loop %':    round(df['is_loop'].mean() * 100, 1),
            'Changed vs blank %': '' if cond == 'blank' else f'{changed*100:.1f}',
        })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

# ── Agreement across image conditions ─────────────────────────────────────────
# For each variant (C→B→A), compute pairwise agreement between image conditions.
# Metrics: exact match, token Jaccard, chrF.
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_KEYS = ['question', 'weaker_object', 'pronominalized']
VARIANT_LABELS_AGR = {
    'question':       'Original (C)',
    'weaker_object':  'Weaker (B)',
    'pronominalized': 'Pronoun. (A)',
}


def load_all_variants(path: Path, mapper: VQAAnswerMapper) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            ex  = json.loads(line.strip())
            qid = int(ex['question_id'])
            for vkey in VARIANT_KEYS:
                raw = ex.get('generated_answers', {}).get(vkey, '')
                ans = preprocess_answer(str(raw), strip_think=True)
                rows.append({'qid': qid, 'variant': vkey, 'answer': ans})
    return pd.DataFrame(rows)


def _jaccard(a: str, b: str) -> float:
    t1, t2 = set(a.split()), set(b.split())
    u = t1 | t2
    return len(t1 & t2) / len(u) if u else 1.0


def _chrf(a: str, b: str) -> float:
    try:
        from sacrebleu.metrics import CHRF
        return CHRF(beta=2).sentence_score(a, [b]).score / 100.0
    except Exception:
        return float('nan')


def compute_agreement(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    merged = df1.merge(df2, on=['qid', 'variant'], suffixes=('_1', '_2'))
    merged['exact']   = (merged['answer_1'] == merged['answer_2']).astype(float)
    merged['jaccard'] = [_jaccard(a, b) for a, b in
                         zip(merged['answer_1'], merged['answer_2'])]
    merged['chrf']    = [_chrf(a, b) for a, b in
                         zip(merged['answer_1'], merged['answer_2'])]
    return merged[['qid', 'variant', 'exact', 'jaccard', 'chrf']]


print('\n── Agreement across image conditions ──')

AGMT_METRICS = [
    ('exact',   'Exact match'),
    ('jaccard', 'Token Jaccard'),
    ('chrf',    'chrF'),
]

all_var_data: dict[str, dict[str, pd.DataFrame]] = {}
for model, cfg in MODELS.items():
    all_var_data[model] = {}
    for cond, path in cfg['conditions'].items():
        all_var_data[model][cond] = load_all_variants(path, mapper)
        print(f'  {model}/{cond}: {len(all_var_data[model][cond])} rows')

PAIR_COLORS = {
    ('blank', 'gray'):  '#78909C',
    ('blank', 'noise'): '#E53935',
    ('blank', 'white'): '#FB8C00',
    ('gray',  'noise'): '#AB47BC',
    ('gray',  'white'): '#43A047',
    ('noise', 'white'): '#1E88E5',
}

for model in MODELS:
    conds_avail = list(MODELS[model]['conditions'].keys())
    pairs = [(c1, c2) for c1, c2 in combinations(conds_avail, 2)]
    if not pairs:
        continue

    pair_agmt: dict[tuple, pd.DataFrame] = {}
    for c1, c2 in pairs:
        pair_agmt[(c1, c2)] = compute_agreement(
            all_var_data[model][c1], all_var_data[model][c2])

    for metric_key, metric_name in AGMT_METRICS:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        for c1, c2 in pairs:
            agmt  = pair_agmt[(c1, c2)]
            ys    = [agmt[agmt['variant'] == v][metric_key].mean()
                     for v in VARIANT_KEYS]
            cis   = [1.96 * float(scipy_sem(
                         agmt[agmt['variant'] == v][metric_key].dropna()))
                     for v in VARIANT_KEYS]
            color = PAIR_COLORS.get((c1, c2), '#888888')
            label = f'{COND_LABEL[c1]} vs {COND_LABEL[c2]}'
            ax.fill_between(range(3),
                            [y - c for y, c in zip(ys, cis)],
                            [y + c for y, c in zip(ys, cis)],
                            color=color, alpha=0.12)
            ax.plot(range(3), ys, color=color, lw=1.8, marker='o',
                    markersize=6, markeredgecolor='white', markeredgewidth=0.5,
                    label=label)
            ax.text(2.08, ys[-1], f'{ys[-1]:.3f}',
                    va='center', fontsize=7.5, color=color)
        ax.set_xticks(range(3))
        ax.set_xticklabels([VARIANT_LABELS_AGR[v] for v in VARIANT_KEYS], fontsize=9)
        ax.set_ylabel(f'Mean {metric_name}', fontsize=10)
        ax.set_title(f'{model} — {metric_name}\nAgreement between image conditions',
                     fontsize=10)
        ax.set_xlim(-0.3, 2.9)
        ax.legend(fontsize=8, loc='upper center',
                  bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)
        ax.grid(axis='y', color='#E0E0E0', lw=0.7)
        plt.tight_layout()
        slug = model.lower().replace('-', '').replace(' ', '_')
        save(fig, f'agreement_{slug}_{metric_key}_variants.png')

# ── Human–Model agreement vs image condition ──────────────────────────────────
# For each image condition and variant (C/B/A), compute mean agreement between
# the model's answer and each human's answer (HM pairs).
# Metrics: exact, jaccard, chrf, SBERT cosine, BERTScore F1.
# ─────────────────────────────────────────────────────────────────────────────
print('\n── Human–Model agreement vs image condition ──')

from utils.agreement import (
    _pair_exact, _pair_jaccard, _pair_chrf,
    encode_with_cache,
)

EXPORTS_DIR = ROOT / 'analysis/session2/exports'
SBERT_CACHE = EXPORTS_DIR / 'embeddings_all-mpnet-base-v2.npz'

HUMAN_VAR_TO_KEY = {'C': 'question', 'B': 'weaker_object', 'A': 'pronominalized'}
HUMAN_VARIANTS   = ['C', 'B', 'A']
VAR_X_LABELS     = {'C': 'Original (C)', 'B': 'Weaker (B)', 'A': 'Pronoun. (A)'}

print('Loading human responses…')
human_df = pd.read_csv(EXPORTS_DIR / 'responses_human.csv')
ablation_qids = set(all_var_data['Qwen3-VL-8B']['blank']['qid'].unique())
human_df = human_df[human_df['question_id'].isin(ablation_qids)].copy()
n_questions_hm = human_df['question_id'].nunique()
n_humans_hm    = human_df['participant'].nunique()
print(f'  {n_questions_hm} questions × {n_humans_hm} participants')

HM_SUFFIX = f'_q{n_questions_hm}_h{n_humans_hm}'

print('Loading SBERT model (all-mpnet-base-v2) on CPU…')
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('all-mpnet-base-v2', device='cpu')

all_answers_for_emb: set[str] = set()
for row in human_df['response'].dropna():
    all_answers_for_emb.add(preprocess_answer(str(row)))
for model_name in all_var_data:
    for cond_df in all_var_data[model_name].values():
        for ans in cond_df['answer']:
            all_answers_for_emb.add(ans)

print(f'Encoding {len(all_answers_for_emb)} unique answers…')
ans2emb = encode_with_cache(
    list(all_answers_for_emb), sbert_model, SBERT_CACHE, verbose=True)


def cosine(a: str, b: str) -> float:
    v1, v2 = ans2emb.get(a), ans2emb.get(b)
    if v1 is None or v2 is None:
        return float('nan')
    return float(np.dot(v1, v2))


def hm_agreement_for_condition(model_name: str, cond: str) -> pd.DataFrame:
    cond_var_df = all_var_data[model_name][cond]
    rows = []
    for hvar in HUMAN_VARIANTS:
        jkey = HUMAN_VAR_TO_KEY[hvar]
        model_answers = (cond_var_df[cond_var_df['variant'] == jkey]
                         .set_index('qid')['answer'])
        human_sub = human_df[human_df['variant'] == hvar]
        for qid, m_ans in model_answers.items():
            h_answers = human_sub[human_sub['question_id'] == qid]['response'].dropna()
            h_answers = [preprocess_answer(str(a)) for a in h_answers]
            if not h_answers:
                continue
            for h_ans in h_answers:
                rows.append({
                    'variant':   hvar,
                    'qid':       qid,
                    'model_ans': m_ans,
                    'human_ans': h_ans,
                    'exact':     _pair_exact(m_ans, h_ans),
                    'jaccard':   _pair_jaccard(m_ans, h_ans),
                    'chrf':      _pair_chrf(m_ans, h_ans),
                    'sbert':     cosine(m_ans, h_ans),
                })
    return pd.DataFrame(rows)


HM_AGMT_METRICS = [
    ('exact',   'Exact match'),
    ('jaccard', 'Token Jaccard'),
    ('chrf',    'chrF'),
    ('sbert',   'SBERT cosine'),
]

COND_LINE_STYLE = {
    'blank': '-',
    'gray':  '--',
    'noise': ':',
    'white': '-.',
}

for model_name in MODELS:
    conds_avail = list(MODELS[model_name]['conditions'].keys())
    print(f'\n  {model_name}…')

    hm_by_cond: dict[str, pd.DataFrame] = {}
    for cond in conds_avail:
        print(f'    {cond}…', end=' ', flush=True)
        df_pairs = hm_agreement_for_condition(model_name, cond)
        hm_by_cond[cond] = df_pairs
        print(f'n_pairs={len(df_pairs)}')

    for metric_key, metric_name in HM_AGMT_METRICS:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        for cond in conds_avail:
            df_pairs = hm_by_cond[cond]
            ys  = [df_pairs[df_pairs['variant'] == v][metric_key].mean()
                   for v in HUMAN_VARIANTS]
            cis = [1.96 * float(scipy_sem(
                       df_pairs[df_pairs['variant'] == v][metric_key].dropna()))
                   for v in HUMAN_VARIANTS]
            color = COND_COLORS[cond]
            ls    = COND_LINE_STYLE.get(cond, '-')
            ax.fill_between(range(3),
                            [y - c for y, c in zip(ys, cis)],
                            [y + c for y, c in zip(ys, cis)],
                            color=color, alpha=0.12)
            ax.plot(range(3), ys, color=color, lw=1.8, ls=ls,
                    marker='o', markersize=6,
                    markeredgecolor='white', markeredgewidth=0.5,
                    label=COND_LABEL[cond])
            ax.text(2.08, ys[-1], f'{ys[-1]:.3f}',
                    va='center', fontsize=7.5, color=color)
        ax.set_xticks(range(3))
        ax.set_xticklabels([VAR_X_LABELS[v] for v in HUMAN_VARIANTS], fontsize=9)
        ax.set_ylabel(f'Mean {metric_name} (HM)', fontsize=10)
        ax.set_title(f'{model_name} — {metric_name}\nHuman–Model agreement by image condition',
                     fontsize=10)
        ax.set_xlim(-0.3, 2.9)
        ax.legend(fontsize=9, frameon=True, loc='upper right')
        ax.grid(axis='y', color='#E0E0E0', lw=0.7)
        plt.tight_layout()
        slug = model_name.lower().replace('-', '').replace(' ', '_')
        fname = f'hm_agreement_{slug}_{metric_key}_variants{HM_SUFFIX}.png'
        save(fig, fname)

print(f'\nDone. Figures saved to: {OUT_DIR}')
