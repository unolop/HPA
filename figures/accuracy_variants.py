"""
Export control-variant accuracy degradation figures.

Output folder:
  figures/accuracy_vABC_group/
    One figure per model group (VLM, backbone decoder, standalone LLM, think).
    Each line = one model.  Colour + marker = model family.

Filenames:
  {blind|inst_blind|control}_{slug}_q{N}_h{N}.png  (e.g. vlm.png, backbone.png)

Run from repo root:
  conda run -n zero python figures/accuracy_variants.py
"""

import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from utils.constants import (
    GROUP_COLORS, GROUP_ORDER, GROUP_MARKER, GROUP_HOLLOW, GROUP_LINESTYLE,
    ACCURACY_VABC_COND_SCHEME, ACCURACY_VABC_PLOT_SPECS, paired_base_model_name,
    MODEL_FAMILY, MODEL_FAMILY_COLORS, SCALE_STYLE_GROUPS, model_scale_style,
)
from helpers import clear_output_plots, read_response_exports
from config import MODEL_LABEL_SHORT, MODELS_7B

parser = argparse.ArgumentParser()
parser.add_argument('--only_7b', action='store_true',
                    help='Restrict plots to the matched 7–8B model subset.')
parser.add_argument('--overwrite', action='store_true',
                    help='Delete existing plot files in the output folder before exporting.')
args = parser.parse_args()

GRP_DIR = ROOT / 'figures/accuracy_vABC_group'
GRP_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(GRP_DIR, overwrite=args.overwrite)

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

VARIANT_ORDER = ['C', 'B', 'A']
VARIANT_LABEL = {'C': 'Original\n(C)', 'B': 'Weaker\n(B)', 'A': 'Pronoun.\n(A)'}
X_DEG = [0, 1, 2]

# Family marker dict (for group plots: colour=family, marker=family)
FAMILY_MARKER = {
    'Qwen3-VL':     's',
    'Qwen3':        '^',
    'LLaVA-1.5':   'o',
    'LLaVA-Mistral':'D',
    'LLaVA-Vicuna': 'P',
    'InternVL':     'X',
    'Mistral':      'v',
    'Vicuna':       'h',
    'Phi':          '*',
    'Qwen2.5':      'p',
}

def _short(model: str) -> str:
    return MODEL_LABEL_SHORT.get(model, model)

def _legend_base_label(model: str, group: str) -> str:
    """Legend label without decoder/think suffix indicators."""
    lbl = _short(model)
    lbl = re.sub(r'\s*\((LM|BB|think)\)\s*$', '', lbl, flags=re.IGNORECASE)
    return lbl


def _axis_base(ax, ylabel):
    ax.set_xticks(X_DEG)
    ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANT_ORDER], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(-0.3, 2.7)
    ax.set_ylim(0, None)


def _human_line(ax, deg_human):
    if deg_human is None:
        return
    ys = [deg_human[deg_human['variant'] == v]['accuracy'].values[0]
          for v in VARIANT_ORDER
          if len(deg_human[deg_human['variant'] == v]) > 0]
    if len(ys) == len(VARIANT_ORDER):
        ax.plot(X_DEG, ys, color='#1565C0', lw=2, ls='--',
                marker='*', markersize=10, label='Human', zorder=5)


def save(fig, name):
    path = GRP_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [accuracy_vABC_group] {name}')


# ── Paired-group figure: one plot for each logical pair ───────────────────────

def _plot_model_line(ax, df_sub, model, group, style_map=None, reference_models=None):
    sub = df_sub[(df_sub['model'] == model) & (df_sub['variant'].isin(VARIANT_ORDER))]
    ys = []
    cis = []
    for v in VARIANT_ORDER:
        sv = sub[sub['variant'] == v]['accuracy']
        if len(sv) == 0:
            return None
        y = sv.mean()
        ci = 1.96 * sv.sem() if len(sv) > 1 else 0.0
        ys.append(float(y))
        cis.append(0.0 if pd.isna(ci) else float(ci))
    base = paired_base_model_name(model, group)
    if style_map and base in style_map:
        color, marker = style_map[base]
    else:
        fam = MODEL_FAMILY.get(model, 'Unknown')
        color = MODEL_FAMILY_COLORS.get(fam, '#888888')
        marker = FAMILY_MARKER.get(fam, 'o')
    hollow = GROUP_HOLLOW.get(group, False)
    # Hollow counterpart lines should be dotted to visually pair with the filled model.
    ls = ':' if hollow else '-'
    scale_style = (
        model_scale_style(model, reference_models)
        if (not args.only_7b and group in SCALE_STYLE_GROUPS)
        else {'alpha': 0.85, 'markersize': 7.0}
    )
    ax.errorbar(
        X_DEG, ys, yerr=cis, color=color, lw=1.8, ls=ls,
        marker=marker, markersize=scale_style['markersize'], alpha=scale_style['alpha'], zorder=3,
        markerfacecolor='none' if hollow else color,
        markeredgecolor=color, markeredgewidth=1.2,
        capsize=2.5, capthick=0.9, elinewidth=0.9,
    )
    return color, marker, hollow, ls


def _build_pair_style_map(pair_models):
    """
    Ensure paired models share exact style (color + marker) by normalized base name.
    Uses the non-hollow side as the anchor when available.
    """
    style_map = {}
    pair_lookup = defaultdict(list)
    for model, group in pair_models:
        base = paired_base_model_name(model, group)
        pair_lookup[base].append((model, group))

    for base, items in pair_lookup.items():
        anchor_model = None
        for model, group in items:
            if not GROUP_HOLLOW.get(group, False):
                anchor_model = model
                break
        if anchor_model is None:
            anchor_model = items[0][0]
        fam = MODEL_FAMILY.get(anchor_model, 'Unknown')
        color = MODEL_FAMILY_COLORS.get(fam, '#888888')
        marker = FAMILY_MARKER.get(fam, 'o')
        style_map[base] = (color, marker)
    return style_map


def draw_paired_groups(ax, df_m, pair_title, pair_groups, pair_models, deg_human, ylabel):
    """One line per model across two logically paired groups."""
    style_map = _build_pair_style_map(pair_models)
    paired_names = [m for m, _ in pair_models]
    df_sub = df_m[df_m['model'].isin(paired_names)].copy()
    _human_line(ax, deg_human)

    model_handles = []
    seen_labels = set()
    reference_models = paired_names
    for model, group in pair_models:
        style = _plot_model_line(
            ax, df_sub, model, group, style_map=style_map, reference_models=reference_models
        )
        if style is None:
            continue
        color, marker, hollow, ls = style
        if hollow:
            continue
        label = _legend_base_label(model, group)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        model_handles.append(mlines.Line2D(
            [], [], color=color, marker=marker, ms=7, ls='-',
            markerfacecolor=color, markeredgecolor=color, markeredgewidth=1.2,
            label=label,
        ))

    # Intentionally no subplot title (user requested title-free figures).
    _axis_base(ax, ylabel)

    handles = []
    if deg_human is not None:
        handles.append(mlines.Line2D([], [], color='#1565C0', marker='*',
                                     ms=8, ls='--', label='Human'))
    handles.extend(model_handles)
    style_suffix = 'decoder' if 'VLM backbone decoder' in pair_groups else 'think'
    handles.extend([
        mlines.Line2D([], [], color='#666666', marker='o', ms=6, ls='-',
                      markerfacecolor='#666666', markeredgecolor='#666666',
                      label='base'),
        mlines.Line2D([], [], color='#666666', marker='o', ms=6, ls=':',
                      markerfacecolor='none', markeredgecolor='#666666',
                      label=style_suffix),
    ])
    ax.legend(handles=handles, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=True,
              handletextpad=0.5, columnspacing=0.8)


def _single_group_models(df_m, by_group, group):
    models = [m for m in by_group.get(group, []) if m in df_m['model'].unique()]
    return [(m, group) for m in models]


def _paired_models_for_groups(df_m, by_group, group_a, group_b):
    avail_a = [m for m in by_group.get(group_a, []) if m in df_m['model'].unique()]
    avail_b = [m for m in by_group.get(group_b, []) if m in df_m['model'].unique()]

    base_a = {}
    base_b = {}
    dupes = []
    for model in avail_a:
        base = paired_base_model_name(model, group_a)
        if base in base_a:
            dupes.append((group_a, base, base_a[base], model))
        base_a[base] = model
    for model in avail_b:
        base = paired_base_model_name(model, group_b)
        if base in base_b:
            dupes.append((group_b, base, base_b[base], model))
        base_b[base] = model

    common_bases = [b for b in sorted(base_a) if b in base_b]
    only_a = sorted(set(base_a) - set(base_b))
    only_b = sorted(set(base_b) - set(base_a))

    if dupes:
        print('  [flag] duplicate normalized pair bases:')
        for grp, base, m1, m2 in dupes:
            print(f'    {grp}: {base} <- {m1} / {m2}')
    if only_a:
        print(f'  [flag] {group_a} models without {group_b} counterpart: {only_a}')
    if only_b:
        print(f'  [flag] {group_b} models without {group_a} counterpart: {only_b}')

    paired = []
    for base in common_bases:
        paired.append((base_a[base], group_a))
        paired.append((base_b[base], group_b))
    return paired


def _llm_think_models(df_m, by_group):
    llm_models = [m for m in by_group.get('standalone LLM', []) if m in df_m['model'].unique()]
    think_models = [m for m in by_group.get('standalone LLM (think)', []) if m in df_m['model'].unique()]

    llm_all = {paired_base_model_name(m, 'standalone LLM'): m for m in llm_models}
    think_qwen = {paired_base_model_name(m, 'standalone LLM (think)'): m
                  for m in think_models}

    only_llm = sorted(set(llm_all) - set(think_qwen))
    only_think = sorted(set(think_qwen) - set(llm_all))
    if only_llm:
        print(f'  [flag] standalone LLM models without think counterpart: {only_llm}')
    if only_think:
        print(f'  [flag] think models without standalone LLM counterpart: {only_think}')

    plot_models = [(llm_all[base], 'standalone LLM') for base in sorted(llm_all)]
    plot_models.extend(
        (think_qwen[base], 'standalone LLM (think)')
        for base in sorted(think_qwen)
    )
    return plot_models


# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading exports…')
exports = read_response_exports(ROOT)
df_mb = exports['model_blind']
df_mi = exports['model_inst_blind']
df_mc = exports['model_control']
df_h  = exports['human']

deg_human = df_h.groupby('variant')['accuracy'].mean().reset_index()

N_QUESTIONS = df_mi['question_id'].nunique()
N_HUMANS    = df_h['participant'].nunique()
SUFFIX      = f'_q{N_QUESTIONS}_h{N_HUMANS}' + ('_7b' if args.only_7b else '')
print(f'questions={N_QUESTIONS}  humans={N_HUMANS}')

# Build group → [model] map from inst_blind data
model_group = (df_mi[['model', 'model_group']].drop_duplicates()
               .set_index('model')['model_group'].to_dict())

by_group: dict[str, list[str]] = defaultdict(list)
for model, grp in sorted(model_group.items()):
    by_group[grp].append(model)

if args.only_7b:
    allowed = set(MODELS_7B)
    df_mb = df_mb[df_mb['model'].isin(allowed)].copy()
    df_mi = df_mi[df_mi['model'].isin(allowed)].copy()
    df_mc = df_mc[df_mc['model'].isin(allowed)].copy()
    by_group = defaultdict(list, {
        grp: [m for m in models if m in allowed]
        for grp, models in by_group.items()
    })

# ── Generate figures ──────────────────────────────────────────────────────────
CONDS = [
    ('blind',      df_mb, 'Mean accuracy (blind)',      False),
    ('inst_blind', df_mi, 'Mean accuracy (inst_blind)', True),
    ('control',    df_mc, 'Mean accuracy (control)',    False),
]

for cond_name, df_m, ylabel, show_human in CONDS:
    dh = deg_human if show_human else None
    print(f'\n── {cond_name} ──')

    for slug in ACCURACY_VABC_COND_SCHEME.get(cond_name, []):
        spec = ACCURACY_VABC_PLOT_SPECS[slug]
        kind = spec['kind']
        groups = spec['groups']
        title = spec['title']

        if kind == 'paired_qwen_think':
            plot_models = _llm_think_models(df_m, by_group)
        elif kind == 'paired':
            group_a, group_b = groups
            plot_models = _paired_models_for_groups(df_m, by_group, group_a, group_b)
        elif kind == 'single_group':
            plot_models = _single_group_models(df_m, by_group, groups[0])
        else:
            raise ValueError(f'Unknown plot kind: {kind}')

        if not plot_models:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_paired_groups(ax, df_m, title, groups, plot_models, dh, ylabel)
        plt.tight_layout()
        save(fig, f'{cond_name}_{slug}{SUFFIX}.png')

print(f'\nDone.\n  Group plots → {GRP_DIR}')
