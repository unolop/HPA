"""
Export model confidence distribution figures for the paper.

Confidence = mean token probability (mean of exp(logprob) across generated tokens),
loaded from the canonical logits directories under evaluation/logits/.

Figures saved to separate top-level folders (x-axis = mean token probability):
  figures/confidence_dist_tokprob_kde/
  figures/confidence_dist_tokprob_violin/
  figures/confidence_dist_tokprob_scatter/
  figures/confidence_dist_tokprob_by_groups_kde/

  {condition}_density[_q{N}_h{N}].png  — KDE density of per-answer confidence per model family;
                                          human per-question accuracy KDE overlaid (dashed).
                                          blind generates two versions: _q1000 (all questions)
                                          and _q{N}_h{N} (human-study subset, comparable to inst_blind).
  outputclass[_q{N}_h{N}].png          — violin of confidence split by output class × model group.
  blind_instblind_scatter.png           — blind vs inst_blind mean confidence scatter per model.

Conditions: blind, inst_blind, control

Run from repo root:
  conda run -n zero python figures/confidence_dist.py
"""

import glob
import json
import sys
import argparse
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
sys.path.insert(0, str(ROOT / 'figures'))

from utils.constants import (
    MODEL_FAMILY, MODEL_FAMILY_COLORS, GROUP_COLORS, GROUP_ORDER, GROUP_HOLLOW, GROUP_MARKER,
    paired_base_model_name,
    model_scale_line_style,
)
from utils.abstention import classify
from helpers import clear_output_plots, get_exports_dir, read_response_exports
from config import MODEL_LABEL_SHORT, MODELS_7B

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true',
                    help='Delete existing plot files in each output folder before exporting.')
args = parser.parse_args()

OUT_DIR_KDE = ROOT / 'figures/confidence_dist_tokprob_kde'
OUT_DIR_VIOLIN = ROOT / 'figures/confidence_dist_tokprob_violin'
OUT_DIR_SCATTER = ROOT / 'figures/confidence_dist_tokprob_scatter'
OUT_GROUP_DIR_KDE = ROOT / 'figures/confidence_dist_tokprob_by_groups_kde'
for p in [OUT_DIR_KDE, OUT_DIR_VIOLIN, OUT_DIR_SCATTER, OUT_GROUP_DIR_KDE]:
    p.mkdir(parents=True, exist_ok=True)
    clear_output_plots(p, overwrite=args.overwrite)

EXPORTS   = get_exports_dir(ROOT)

HUMAN_COLOR = '#000000'

# ── Model name mapping (logit dir name → display name) ───────────────────────
VLM_DIR_TO_MODEL = {
    'InternVL3_5-1B':           'InternVL-1B',
    'InternVL3_5-2B':           'InternVL-2B',
    'InternVL3_5-8B':           'InternVL-8B',
    'llava-1.5-7b-hf':          'LLaVA-1.5-7B',
    'llava-v1.6-mistral-7b-hf': 'LLaVA-Mistral',
    'llava-v1.6-vicuna-13b-hf': 'LLaVA-Vicuna-13B',
    'llava-v1.6-vicuna-7b-hf':  'LLaVA-Vicuna',
    'Qwen3-VL-2B-Instruct':     'Qwen3-VL-2B',
    'Qwen3-VL-4B-Instruct':     'Qwen3-VL-4B',
    'Qwen3-VL-8B-Instruct':     'Qwen3-VL-8B',
    'Qwen3-VL-32B-Instruct':    'Qwen3-VL-32B',
}

LM_DECODER_DIR_TO_MODEL = {
    'InternVL3_5-1B':           'InternVL-1B (LM)',
    'InternVL3_5-2B':           'InternVL-2B (LM)',
    'InternVL3_5-8B':           'InternVL-8B (LM)',
    'llava-1.5-7b-hf':          'LLaVA-1.5 (LM)',
    'llava-v1.6-mistral-7b-hf': 'LLaVA-Mistral (LM)',
    'llava-v1.6-vicuna-13b-hf': 'LLaVA-Vicuna-13B (LM)',
    'llava-v1.6-vicuna-7b-hf':  'LLaVA-Vicuna (LM)',
    'Qwen3-VL-2B-Instruct':     'Qwen3-VL-2B (LM)',
    'Qwen3-VL-4B-Instruct':     'Qwen3-VL-4B (LM)',
    'Qwen3-VL-8B-Instruct':     'Qwen3-VL-8B (LM)',
    'Qwen3-VL-32B-Instruct':    'Qwen3-VL-32B (LM)',
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
    'vicuna-7b-v1.5':           'Vicuna-7B',
}

BACKBONE_THINK_DIR_TO_MODEL = {
    'Qwen3-0.6B_think':  'Qwen3-0.6B (think)',
    'Qwen3-1.7B_think':  'Qwen3-1.7B (think)',
    'Qwen3-4B_think':    'Qwen3-4B (think)',
    'Qwen3-8B_think':    'Qwen3-8B (think)',
    'Qwen3-32B_think':   'Qwen3-32B (think)',
}

LOGIT_SOURCES = [
    (ROOT / 'evaluation/logits/vlm/pretrained', VLM_DIR_TO_MODEL),
    (ROOT / 'evaluation/logits/lm_decoder/pretrained', LM_DECODER_DIR_TO_MODEL),
    (ROOT / 'evaluation/logits/backbone/pretrained', BACKBONE_NOTHINK_DIR_TO_MODEL),
    (ROOT / 'evaluation/logits/backbone/pretrained', BACKBONE_THINK_DIR_TO_MODEL),
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


def save(fig, kind, name):
    if kind == 'kde':
        path = OUT_DIR_KDE / name
    elif kind == 'violin':
        path = OUT_DIR_VIOLIN / name
    elif kind == 'scatter':
        path = OUT_DIR_SCATTER / name
    else:
        raise ValueError(f'Unknown kind: {kind}')
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  [confidence_dist_{kind}] {name}')

def save_group(fig, name):
    path = OUT_GROUP_DIR_KDE / name
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  [confidence_dist_by_groups_kde] {name}')


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

# Mean accuracy per variant (for variants line — kept for output_class figure)
human_var_acc = human_df.groupby('variant')['accuracy'].mean()

HUMAN_SUFFIX = f'_q{N_Q}_h{N_H}'
human_qids   = set(human_df['question_id'].unique())

# ── Load human confidence from raw participant JSONs ──────────────────────────
# Confidence is a 1–5 self-reported scale; normalise to [0, 1].
print('\nLoading human confidence from participant JSONs…')
human_conf_rows = []
for path in glob.glob(str(ROOT / 'evaluation/humans/by_participant/*.json')):
    with open(path) as f:
        d = json.load(f)
    for ans in d.get('answers', []):
        if ans['question_id'] in human_qids:
            human_conf_rows.append((ans['question_id'], (ans['confidence'] - 1) / 4))
human_conf_norm = np.array([c for _, c in human_conf_rows])
print(f'  {len(human_conf_norm):,} human confidence values (q={N_Q}, h={N_H}, scale→[0,1])')


# ── Helpers ───────────────────────────────────────────────────────────────────
def model_color(m):
    return MODEL_FAMILY_COLORS.get(MODEL_FAMILY.get(m, 'Unknown'), '#888888')

def model_label(m):
    return MODEL_LABEL_SHORT.get(m, m)

MODEL_SETS = [
    ('all', None),
    ('7b', set(MODELS_7B)),
]


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

for set_name, allowed_models in MODEL_SETS:
    set_suffix = '' if set_name == 'all' else f'_{set_name}'
    plot_df = df[df['condition'].isin(['blind', 'inst_blind'])].copy()
    if allowed_models is not None:
        plot_df = plot_df[plot_df['model'].isin(allowed_models)]
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
        title_suffix = '' if set_name == 'all' else ' (7B subset)'
        fig.suptitle(f'Confidence Distributions by Output Class and Model Group{title_suffix}', fontsize=13, y=1.03)
        fig.tight_layout()
        save(fig, 'violin', f'violin_outputclass{HUMAN_SUFFIX}{set_suffix}.png')


# ── Figures 1–3: KDE distribution of per-answer confidence per family ─────────
# For blind: two versions — all questions (q1000) and human-study subset (q113_h40).
# Human accuracy KDE overlaid on all versions.
DIST_PLAN = [
    ('blind',      None,       '_q1000',     False),
    ('blind',      human_qids, HUMAN_SUFFIX, False),
    ('inst_blind', human_qids, HUMAN_SUFFIX, True),
    ('control',    None,       HUMAN_SUFFIX, False),
]

GROUP_PANELS = [
    ('vlm_backbone', 'VLM + Backbone decoder', ('VLM', 'VLM backbone decoder')),
    ('llm_think', 'Standalone LLM + Think', ('standalone LLM', 'standalone LLM (think)')),
]
GROUP_STYLE = {
    'VLM':                    {'ls': '-', 'lw': 2.2},
    'VLM backbone decoder':   {'ls': ':', 'lw': 2.2},
    'standalone LLM':         {'ls': '-', 'lw': 2.2},
    'standalone LLM (think)': {'ls': ':', 'lw': 2.2},
}

# NOTE: family-level non-group KDE exports removed by request.

# ── Group-split, per-model KDEs in separate folder (by_groups) ───────────────
for set_name, allowed_models in MODEL_SETS:
    set_suffix = '' if set_name == 'all' else f'_{set_name}'
    for condition, subset_qids, fname_suffix, _show_human in DIST_PLAN:
        sub = df[(df['condition'] == condition) & (df['control_type'] == 'question')].copy()
        if allowed_models is not None:
            sub = sub[sub['model'].isin(allowed_models)]
        if subset_qids is not None:
            sub = sub[sub['question_id'].isin(subset_qids)]
        sub = sub[sub['model_group'].isin(GROUP_ORDER)]
        if sub.empty:
            continue

        x_grid = np.linspace(0, 1, 300)
        for panel_key, panel_title, groups in GROUP_PANELS:
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            any_drawn = False
            shown_base_labels = set()
            panel_models = (
                sub[sub['model_group'].isin(groups)][['model', 'model_group']]
                .drop_duplicates()
                .sort_values(['model_group', 'model'])
            )
            panel_model_names = panel_models['model'].tolist()
            for _, r in panel_models.iterrows():
                model = r['model']
                grp = r['model_group']
                vals = sub[sub['model'] == model]['confidence'].dropna().values
                if len(vals) < 10:
                    continue
                kde = gaussian_kde(vals, bw_method=0.12)
                st = GROUP_STYLE[grp]
                st_scale = model_scale_line_style(model, reference_models=panel_model_names)
                color = model_color(model)
                y = kde(x_grid)
                is_derivative = GROUP_HOLLOW.get(grp, False)
                base_label = model_label(paired_base_model_name(model, grp))
                legend_label = None
                if not is_derivative and base_label not in shown_base_labels:
                    legend_label = base_label
                    shown_base_labels.add(base_label)
                ax.plot(
                    x_grid,
                    y,
                    color=color,
                    ls=st['ls'],
                    lw=st_scale['linewidth'],
                    label=legend_label,
                    alpha=st_scale['line_alpha'],
                )
                any_drawn = True

            if not any_drawn:
                plt.close(fig)
                continue

            apply_axis_style(ax, grid_axis='y')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Confidence', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            title_suffix = '' if set_name == 'all' else ' (7B subset)'
            ax.set_title(f'{panel_title} — {COND_LABELS[condition]}{title_suffix}', fontsize=11, fontweight='bold')
            model_handles, _ = ax.get_legend_handles_labels()
            style_suffix = 'decoder' if panel_key == 'vlm_backbone' else 'think'
            style_handles = [
                mlines.Line2D([], [], color='#666666', ls='-', lw=1.8, label='base'),
                mlines.Line2D([], [], color='#666666', ls=':', lw=1.8, label=style_suffix),
            ]
            ax.legend(handles=model_handles + style_handles, fontsize=7.5,
                      loc='upper left', frameon=True, ncol=2)
            fig.tight_layout()
            save_group(fig, f'kde_{condition}_density{fname_suffix}{set_suffix}_{panel_key}.png')


# ── Figure 7: blind vs inst_blind confidence scatter per model ────────────────
SCATTER_PANELS = [
    ('vlm_backbone', 'VLM + Backbone decoder', ('VLM', 'VLM backbone decoder')),
    ('llm_think', 'Standalone LLM + Think', ('standalone LLM', 'standalone LLM (think)')),
]

for set_name, allowed_models in MODEL_SETS:
    set_suffix = '' if set_name == 'all' else f'_{set_name}'
    scatter_df = df.copy()
    if allowed_models is not None:
        scatter_df = scatter_df[scatter_df['model'].isin(allowed_models)]

    blind_means   = scatter_df[scatter_df['condition'] == 'blind'].groupby('model')['confidence'].mean()
    inst_means    = scatter_df[scatter_df['condition'] == 'inst_blind'].groupby('model')['confidence'].mean()
    shared_models = blind_means.index.intersection(inst_means.index)
    if len(shared_models) == 0:
        continue

    for panel_key, panel_title, panel_groups in SCATTER_PANELS:
        panel_rows = []
        for model in shared_models:
            group = model_group_map.get(model, '')
            if group not in panel_groups:
                continue
            panel_rows.append({
                'model': model,
                'group': group,
                'x': blind_means[model],
                'y': inst_means[model],
                'color': model_color(model),
                'marker': GROUP_MARKER.get(group, 'o'),
                'hollow': GROUP_HOLLOW.get(group, False),
            })
        if not panel_rows:
            continue

        fig, ax = plt.subplots(figsize=(5.5, 5))
        panel_rows = sorted(panel_rows, key=lambda r: (r['color'], r['y'], r['x']))
        for r in panel_rows:
            ax.scatter(
                r['x'], r['y'], s=72, zorder=3, marker=r['marker'],
                facecolors='none' if r['hollow'] else r['color'],
                edgecolors=r['color'] if r['hollow'] else 'white',
                linewidths=1.0 if r['hollow'] else 0.8,
                alpha=0.92,
            )
            arrow = FancyArrowPatch(
                (r['x'], r['x']), (r['x'], r['y']),
                arrowstyle='-|>', mutation_scale=8,
                linewidth=0.9, color=r['color'], alpha=0.45, zorder=2,
                shrinkA=6, shrinkB=6,
            )
            ax.add_patch(arrow)

        xs = np.array([r['x'] for r in panel_rows], dtype=float)
        ys = np.array([r['y'] for r in panel_rows], dtype=float)
        lo = min(xs.min(), ys.min()) - 0.02
        hi = max(xs.max(), ys.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], color='#999999', lw=1, ls='--', zorder=1)

        label_rows = [(model_label(r['model']), r['x'], r['y'], r['color']) for r in panel_rows]
        label_ys = spaced_label_positions([row[2] for row in label_rows], min_gap=0.012, lower=lo + 0.02, upper=hi - 0.01)
        label_x = hi + 0.01
        for (label, x_val, y_val, color), y_lab in zip(label_rows, label_ys):
            ax.plot([x_val, label_x - 0.003], [y_val, y_lab], color=color, lw=0.75, alpha=0.5, zorder=2)
            ax.text(label_x, y_lab, label, fontsize=8, color=color, va='center', ha='left')

        apply_axis_style(ax, grid_axis='both')
        ax.set_xlabel('Blind — mean confidence', fontsize=10)
        ax.set_ylabel('Inst-Blind — mean confidence', fontsize=10)
        title_suffix = '' if set_name == 'all' else ' (7B subset)'
        ax.set_title(f'Confidence Shift: {panel_title}{title_suffix}', fontsize=12, fontweight='bold')
        ax.set_xlim(lo, hi + 0.12)
        ax.set_ylim(lo, hi)

        fams = sorted({MODEL_FAMILY.get(r['model'], 'Unknown') for r in panel_rows})
        family_handles = [
            mlines.Line2D([], [], color=MODEL_FAMILY_COLORS.get(f, '#888'),
                          marker='o', ms=7, ls='', label=f)
            for f in fams
        ]
        style_suffix = 'decoder' if panel_key == 'vlm_backbone' else 'think'
        style_handles = [
            mlines.Line2D([], [], color='#666666', marker='o', ms=7, ls='',
                          markerfacecolor='#666666', markeredgecolor='#666666',
                          label='base'),
            mlines.Line2D([], [], color='#666666', marker='o', ms=7, ls='',
                          markerfacecolor='none', markeredgecolor='#666666',
                          label=style_suffix),
        ]
        leg1 = ax.legend(handles=family_handles, fontsize=8, loc='upper left', frameon=True, title='Family')
        ax.add_artist(leg1)
        ax.legend(handles=style_handles, fontsize=8, loc='lower right', frameon=True, title='Style')
        fig.tight_layout()
        save(fig, 'scatter', f'scatter_blind_instblind{set_suffix}_{panel_key}.png')

print('\nDone. All figures saved to separate folders:')
print('  ', OUT_DIR_KDE)
print('  ', OUT_DIR_VIOLIN)
print('  ', OUT_DIR_SCATTER)
print('  ', OUT_GROUP_DIR_KDE)
