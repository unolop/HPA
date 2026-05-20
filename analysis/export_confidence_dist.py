"""
Export model confidence distribution figures for the paper.

Confidence = mean token probability (mean of exp(logprob) across generated tokens),
loaded from the canonical logits directories under evaluation/logits/.

Figures saved to figures/confidence_dist/:

  {condition}_variants.png       — mean confidence across control types per model (family colours)
  {condition}_dist.png           — KDE distribution of per-answer confidence per model family
  condition_shift.png            — blind vs inst_blind mean confidence scatter per model

  inst_blind figures also include a human accuracy reference line/curve and carry
  a _q{N}_h{N} suffix indicating the number of questions and participants.
  Human accuracy is available for variants C/B/A (→ question/weaker_object/pronominalized).

Conditions: blind, inst_blind, control

Run from repo root:
  conda run -n zero python analysis/export_confidence_dist.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.constants import MODEL_FAMILY, MODEL_FAMILY_COLORS, GROUP_COLORS, GROUP_ORDER
from utils.abstention import classify
from export_helpers import get_exports_dir, read_response_exports
from config import MODEL_LABEL_SHORT

OUT_DIR   = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/confidence_dist'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORTS   = get_exports_dir(ROOT)

HUMAN_COLOR = '#1565C0'

# ── Model name mapping (logit dir name → display name) ───────────────────────
VLM_DIR_TO_MODEL = {
    'InternVL3_5-1B':           'InternVL-1B',
    'InternVL3_5-2B':           'InternVL-2B',
    'InternVL3_5-8B':           'InternVL-8B',
    'llava-1.5-7b-hf':          'LLaVA-1.5-7B',
    'llava-v1.6-mistral-7b-hf': 'LLaVA-Mistral',
    'llava-v1.6-vicuna-13b-hf': 'LLaVA-Vicuna',
    'llava-v1.6-vicuna-7b-hf':  'LLaVA-Vicuna',
    'Qwen3-VL-2B-Instruct':     'Qwen3-VL-2B',
    'Qwen3-VL-4B-Instruct':     'Qwen3-VL-4B',
    'Qwen3-VL-8B-Instruct':     'Qwen3-VL-8B',
    'Qwen3-VL-32B-Instruct':    'Qwen3-VL-32B',
}

LM_DECODER_DIR_TO_MODEL = {
    'llava-1.5-7b-hf':          'LLaVA-1.5 (LM)',
    'llava-v1.6-mistral-7b-hf': 'LLaVA-Mistral (LM)',
    'llava-v1.6-vicuna-13b-hf': 'LLaVA-Vicuna (LM)',
    'llava-v1.6-vicuna-7b-hf':  'LLaVA-Vicuna (LM)',
}

BACKBONE_NOTHINK_DIR_TO_MODEL = {
    'Mistral-7B-Instruct-v0.2': 'Mistral-7B',
    'Phi-3.5-mini-instruct':    'Phi-3.5-mini',
    'Qwen2.5-7B':               'Qwen2.5-7B',
    'Qwen2.5-7B-Instruct':      'Qwen2.5-7B',
    'Qwen3-0.6B':               'Qwen3-0.6B',
    'Qwen3-1.7B':               'Qwen3-1.7B',
    'Qwen3-4B':                 'Qwen3-4B',
    'Qwen3-8B':                 'Qwen3-8B',
    'Qwen3-32B':                'Qwen3-32B',
    'vicuna-13b-v1.5':          'Vicuna-13B',
    'vicuna-7b-v1.5':           'Vicuna-13B',
}

BACKBONE_THINK_DIR_TO_MODEL = {
    'Qwen3-0.6B':  'Qwen3-0.6B (think)',
    'Qwen3-1.7B':  'Qwen3-1.7B (think)',
    'Qwen3-4B':    'Qwen3-4B (think)',
    'Qwen3-8B':    'Qwen3-8B (think)',
    'Qwen3-32B':   'Qwen3-32B (think)',
}

LOGIT_SOURCES = [
    (ROOT / 'evaluation/logits/vlm/pretrained', VLM_DIR_TO_MODEL),
    (ROOT / 'evaluation/logits/lm_decoder/pretrained', LM_DECODER_DIR_TO_MODEL),
    (ROOT / 'evaluation/logits/backbone_nothink/pretrained', BACKBONE_NOTHINK_DIR_TO_MODEL),
    (ROOT / 'evaluation/logits/backbone_think/pretrained', BACKBONE_THINK_DIR_TO_MODEL),
]

DATASET_TO_CONDITION = {
    'vqa_1k_control_blind':      'blind',
    'vqa_1k_control_inst_blind': 'inst_blind',
    'vqa_1k_control':            'control',
}

CONTROL_TYPES = ['question', 'deictic_removed', 'object_removed', 'weaker_object', 'pronominalized']
CT_LABELS     = ['Original', 'Deictic\nremoved', 'Object\nremoved', 'Weaker\nobject', 'Pronominalized']

# Human variant → control_type position on the x-axis
VARIANT_TO_CT_IDX = {'C': 0, 'B': 3, 'A': 4}   # C=question, B=weaker_object, A=pronominalized

CONDITIONS  = ['blind', 'inst_blind', 'control']
COND_LABELS = {'blind': 'Blind', 'inst_blind': 'Inst-Blind', 'control': 'Control (real image)'}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        False,
    'axes.labelsize':   11,
    'axes.titlesize':   13,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  9,
})


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  [confidence_dist] {name}')


def apply_axis_style(ax, grid_axis='y'):
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(length=3.5, width=0.8, color='#555555')
    if grid_axis:
        ax.grid(axis=grid_axis, color='#D9D9D9', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def spaced_label_positions(values, min_gap=0.018, lower=0.02, upper=0.98):
    """Return vertically spaced y positions for endpoint labels."""
    if not values:
        return []
    order = np.argsort(values)
    vals = np.array(values, dtype=float)[order]
    adjusted = vals.copy()
    adjusted[0] = max(adjusted[0], lower)
    for i in range(1, len(adjusted)):
        adjusted[i] = max(adjusted[i], adjusted[i - 1] + min_gap)
    overflow = adjusted[-1] - upper
    if overflow > 0:
        adjusted -= overflow
        for i in range(len(adjusted) - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1] - min_gap)
        if adjusted[0] < lower:
            adjusted += (lower - adjusted[0])
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out.tolist()


def load_jsonl_examples(path: Path):
    examples = []
    bad = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f'  WARN {path.name}: skipped {bad} malformed lines')
    return examples


# ── Load logit data ───────────────────────────────────────────────────────────
print('Loading logit files…')
rows = []
for base_dir, dir_to_model in LOGIT_SOURCES:
    for dir_name, model in dir_to_model.items():
        model_dir = base_dir / dir_name
        if not model_dir.exists():
            continue
        for ds_name, condition in DATASET_TO_CONDITION.items():
            fpath = model_dir / f'{ds_name}.jsonl'
            if not fpath.exists():
                continue
            examples = load_jsonl_examples(fpath)
            n = len(examples)
            print(f'  {model} | {condition}: {n} examples')
            for ex in examples:
                logits  = ex.get('generated_logits', {})
                answers = ex.get('generated_answers', {})
                for ct, logit_data in logits.items():
                    if ct not in CONTROL_TYPES:
                        continue
                    tokens = logit_data.get('content', [])
                    if not tokens:
                        continue
                    confidence = float(np.exp([t['logprob'] for t in tokens]).mean())
                    rows.append({
                        'model':        model,
                        'family':       MODEL_FAMILY.get(model, 'Unknown'),
                        'condition':    condition,
                        'question_id':  ex.get('question_id'),
                        'control_type': ct,
                        'output_text':  answers.get(ct, ''),
                        'confidence':   confidence,
                    })

df = pd.DataFrame(rows)
df['control_type'] = pd.Categorical(df['control_type'], categories=CONTROL_TYPES, ordered=True)
print(f'\nLoaded {len(df):,} rows | {df["model"].nunique()} models | {df["question_id"].nunique()} questions')


# ── Load human data (inst_blind, all variants) ────────────────────────────────
print('\nLoading human data…')
exports = read_response_exports(ROOT)
human_df = exports['human']
model_meta = pd.concat([
    exports['model_blind'][['question_id', 'model', 'model_group', 'gt']],
    exports['model_inst_blind'][['question_id', 'model', 'model_group', 'gt']],
    exports['model_control'][['question_id', 'model', 'model_group', 'gt']],
], ignore_index=True).drop_duplicates()

model_group_map = model_meta.drop_duplicates('model').set_index('model')['model_group'].to_dict()
gt_map = model_meta.drop_duplicates('question_id').set_index('question_id')['gt'].to_dict()
df['model_group'] = df['model'].map(model_group_map)
df['gt'] = df['question_id'].map(gt_map)
df['output_class'] = df.apply(
    lambda r: classify(r['output_text'], [r['gt']] if pd.notna(r['gt']) and str(r['gt']).strip() else None),
    axis=1,
)

N_H = human_df['participant'].nunique()
N_Q = human_df['question_id'].nunique()
print(f'  {N_H} participants × {N_Q} questions × {human_df["variant"].nunique()} variants')

# Mean accuracy per variant (for variants line)
human_var_acc = human_df.groupby('variant')['accuracy'].mean()

# Per-question mean accuracy (variant C = original question, for dist KDE)
human_q_acc = (human_df[human_df['variant'] == 'C']
               .groupby('question_id')['accuracy'].mean().values)

HUMAN_SUFFIX = f'_q{N_Q}_h{N_H}'


# ── Helpers ───────────────────────────────────────────────────────────────────
def model_color(m):
    return MODEL_FAMILY_COLORS.get(MODEL_FAMILY.get(m, 'Unknown'), '#888888')

def model_label(m):
    return MODEL_LABEL_SHORT.get(m, m)


# ── Figure 0: confidence distributions by output class and model group ───────
CLASS_ORDER = [
    'hallucinated_correct',
    'hallucinated_wrong',
    'soft_abstained',
    'hard_abstained',
]
CLASS_LABELS = {
    'hallucinated_correct': 'Correct\ncommit',
    'hallucinated_wrong':   'Wrong\ncommit',
    'soft_abstained':       'Soft\nabstain',
    'hard_abstained':       'Hard\nabstain',
}
GROUP_SHORT = {
    'VLM backbone decoder':   'Backbone',
    'VLM':                    'VLM',
    'standalone LLM (think)': 'LLM think',
    'standalone LLM':         'LLM',
}

plot_df = df[df['condition'].isin(['blind', 'inst_blind'])].copy()
plot_df = plot_df[plot_df['output_class'].isin(CLASS_ORDER) & plot_df['model_group'].isin(GROUP_ORDER)]

if not plot_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
    width = 0.16
    x = np.arange(len(CLASS_ORDER))

    for ax, condition in zip(axes, ['blind', 'inst_blind']):
        sub = plot_df[plot_df['condition'] == condition]
        for i, grp in enumerate(GROUP_ORDER):
            data = []
            positions = []
            for j, cls in enumerate(CLASS_ORDER):
                vals = sub[(sub['model_group'] == grp) & (sub['output_class'] == cls)]['confidence'].dropna().values
                if len(vals) == 0:
                    continue
                data.append(vals)
                positions.append(j + (i - 1.5) * width)
            if not data:
                continue

            vp = ax.violinplot(
                data,
                positions=positions,
                widths=width * 0.95,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in vp['bodies']:
                body.set_facecolor(GROUP_COLORS[grp])
                body.set_edgecolor('white')
                body.set_linewidth(0.6)
                body.set_alpha(0.35)
            vp['cmedians'].set_color(GROUP_COLORS[grp])
            vp['cmedians'].set_linewidth(1.5)

        apply_axis_style(ax, grid_axis='y')
        ax.set_xticks(x)
        ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_title(COND_LABELS[condition], fontsize=12, fontweight='bold')
        ax.set_xlabel('Output class', fontsize=10)

    axes[0].set_ylabel('Mean token probability', fontsize=10)
    handles = [
        mlines.Line2D([], [], color=GROUP_COLORS[g], lw=6, alpha=0.6, label=GROUP_SHORT[g])
        for g in GROUP_ORDER
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.01),
               ncol=4, frameon=True, fontsize=8)
    fig.suptitle('Confidence Distributions by Output Class and Model Group', fontsize=13, y=1.03)
    fig.tight_layout()
    save(fig, f'output_class_shift{HUMAN_SUFFIX}.png')


# ── Figures 1–3: confidence by control variant, one line per model ────────────
for condition in CONDITIONS:
    sub = df[df['condition'] == condition]
    if sub.empty:
        continue

    is_inst = (condition == 'inst_blind')
    fname_suffix = HUMAN_SUFFIX if is_inst else ''

    models = sorted(sub['model'].unique())
    fig, ax = plt.subplots(figsize=(7.6, 4.9))

    seen_families = set()
    line_endpoints = []
    for model in models:
        msub = sub[sub['model'] == model]
        ys = [msub[msub['control_type'] == ct]['confidence'].mean() for ct in CONTROL_TYPES]
        if all(np.isnan(y) for y in ys):
            continue
        fam   = MODEL_FAMILY.get(model, 'Unknown')
        color = MODEL_FAMILY_COLORS.get(fam, '#888888')
        ax.plot(range(len(CONTROL_TYPES)), ys,
                color=color, lw=1.8, marker='o', markersize=5,
                alpha=0.85, markeredgecolor='white', markeredgewidth=0.5)
        line_endpoints.append((model_label(model), ys[-1], color))
        seen_families.add(fam)

    # Human accuracy reference (inst_blind only)
    if is_inst:
        hxs = sorted(VARIANT_TO_CT_IDX.values())       # [0, 3, 4]
        hvs = ['C', 'B', 'A']
        hys = [float(human_var_acc.get(v, np.nan)) for v in hvs]
        ax.plot(hxs, hys, color=HUMAN_COLOR, lw=2, ls='--',
                marker='*', markersize=9, zorder=5,
                markeredgecolor='white', markeredgewidth=0.5)
        line_endpoints.append((f'Human acc.\n(n={N_H}, q={N_Q})', hys[-1], HUMAN_COLOR))

    apply_axis_style(ax, grid_axis='y')
    ax.set_xticks(range(len(CONTROL_TYPES)))
    ax.set_xticklabels(CT_LABELS, fontsize=9)
    ax.set_ylabel('Mean token probability / accuracy', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.3, len(CONTROL_TYPES) - 0.05)
    ax.set_title(f'Model Confidence by Control Type — {COND_LABELS[condition]}',
                 fontsize=12, fontweight='bold')

    label_ys = spaced_label_positions([y for _, y, _ in line_endpoints], min_gap=0.025, lower=0.06, upper=0.95)
    x_text = len(CONTROL_TYPES) - 0.02
    x_anchor = len(CONTROL_TYPES) - 1
    for (label, y_true, color), y_lab in zip(line_endpoints, label_ys):
        ax.plot([x_anchor, x_text - 0.02], [y_true, y_lab], color=color, lw=0.8, alpha=0.55)
        ax.text(x_text, y_lab, label, va='center', ha='left', fontsize=7.3, color=color)

    handles = [mlines.Line2D([], [], color=MODEL_FAMILY_COLORS.get(f, '#888'),
                             lw=2, marker='o', ms=5, label=f)
               for f in sorted(seen_families)]
    ax.legend(handles=handles, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True)
    fig.tight_layout()
    save(fig, f'{condition}_variants{fname_suffix}.png')


# ── Figures 4–6: KDE distribution of per-answer confidence per family ─────────
for condition in CONDITIONS:
    sub = df[(df['condition'] == condition) & (df['control_type'] == 'question')]
    if sub.empty:
        continue

    is_inst = (condition == 'inst_blind')
    fname_suffix = HUMAN_SUFFIX if is_inst else ''

    families = [f for f in MODEL_FAMILY_COLORS if f in sub['family'].unique()]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x_grid = np.linspace(0, 1, 300)

    for fam in families:
        vals = sub[sub['family'] == fam]['confidence'].dropna().values
        if len(vals) < 10:
            continue
        kde   = gaussian_kde(vals, bw_method=0.12)
        color = MODEL_FAMILY_COLORS[fam]
        ax.plot(x_grid, kde(x_grid), color=color, lw=2, label=fam)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.08, color=color)

    # Human per-question accuracy KDE (inst_blind only)
    if is_inst and len(human_q_acc) >= 5:
        kde_h = gaussian_kde(human_q_acc, bw_method=0.15)
        ax.plot(x_grid, kde_h(x_grid), color=HUMAN_COLOR, lw=2, ls='--',
                label=f'Human accuracy (q={N_Q}, h={N_H})')
        ax.fill_between(x_grid, kde_h(x_grid), alpha=0.07, color=HUMAN_COLOR)

    apply_axis_style(ax, grid_axis='y')
    ax.set_xlabel('Mean token probability / accuracy', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_title(f'Confidence Distribution — {COND_LABELS[condition]}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', frameon=True)
    fig.tight_layout()
    save(fig, f'{condition}_dist{fname_suffix}.png')


# ── Figure 7: blind vs inst_blind confidence scatter per model ────────────────
blind_means   = df[df['condition'] == 'blind'].groupby('model')['confidence'].mean()
inst_means    = df[df['condition'] == 'inst_blind'].groupby('model')['confidence'].mean()
shared_models = blind_means.index.intersection(inst_means.index)

fig, ax = plt.subplots(figsize=(5.5, 5))
plot_rows = []
for model in shared_models:
    x_val = blind_means[model]
    y_val = inst_means[model]
    color = model_color(model)
    group = model_group_map.get(model, '')
    marker = {
        'VLM': 'o',
        'VLM backbone decoder': 's',
        'standalone LLM': '^',
        'standalone LLM (think)': 'D',
    }.get(group, 'o')
    plot_rows.append((model, x_val, y_val, color, marker, group))

plot_rows = sorted(plot_rows, key=lambda row: (row[3], row[2], row[1]))
for model, x_val, y_val, color, marker, _group in plot_rows:
    ax.scatter(x_val, y_val, color=color, s=72, zorder=3,
               marker=marker, edgecolors='white', linewidths=0.8)
    arrow = FancyArrowPatch(
        (x_val, x_val), (x_val, y_val),
        arrowstyle='-|>', mutation_scale=8,
        linewidth=0.9, color=color, alpha=0.45, zorder=2,
        shrinkA=6, shrinkB=6,
    )
    ax.add_patch(arrow)

lo = min(blind_means.min(), inst_means.min()) - 0.02
hi = max(blind_means.max(), inst_means.max()) + 0.02
ax.plot([lo, hi], [lo, hi], color='#999999', lw=1, ls='--', zorder=1)

label_rows = []
for model, x_val, y_val, color, marker, group in plot_rows:
    label_rows.append((model_label(model), x_val, y_val, color))

label_ys = spaced_label_positions([row[2] for row in label_rows], min_gap=0.012, lower=lo + 0.02, upper=hi - 0.01)
label_x = hi + 0.01
for (label, x_val, y_val, color), y_lab in zip(label_rows, label_ys):
    ax.plot([x_val, label_x - 0.003], [y_val, y_lab], color=color, lw=0.75, alpha=0.5, zorder=2)
    ax.text(label_x, y_lab, label, fontsize=8, color=color, va='center', ha='left')

apply_axis_style(ax, grid_axis='both')
ax.set_xlabel('Blind — mean confidence', fontsize=10)
ax.set_ylabel('Inst-Blind — mean confidence', fontsize=10)
ax.set_title('Confidence Shift: Blind → Inst-Blind', fontsize=12, fontweight='bold')
ax.set_xlim(lo, hi + 0.12)
ax.set_ylim(lo, hi)

family_handles = [mlines.Line2D([], [], color=MODEL_FAMILY_COLORS.get(f, '#888'),
                                marker='o', ms=7, ls='', label=f)
                  for f in sorted({MODEL_FAMILY.get(m, 'Unknown') for m in shared_models})]
group_handles = [
    mlines.Line2D([], [], color='#666666', marker='o', ms=7, ls='', label='VLM'),
    mlines.Line2D([], [], color='#666666', marker='s', ms=7, ls='', label='Backbone'),
    mlines.Line2D([], [], color='#666666', marker='^', ms=7, ls='', label='Standalone LLM'),
    mlines.Line2D([], [], color='#666666', marker='D', ms=7, ls='', label='LLM (think)'),
]
leg1 = ax.legend(handles=family_handles, fontsize=8, loc='upper left', frameon=True, title='Family')
ax.add_artist(leg1)
ax.legend(handles=group_handles, fontsize=8, loc='lower right', frameon=True, title='Group')
fig.tight_layout()
save(fig, 'condition_shift.png')

print('\nDone. All figures saved to:', OUT_DIR)
