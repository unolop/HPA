"""
Export control-variant accuracy degradation figures to figures/accuracy_lineplot/.

Four versions are generated for each of the three conditions (blind / inst_blind / control):

  groups  — group-mean lines  (VLM / backbone / standalone / think / Human)
  family  — family-mean lines  (Qwen3-VL / Qwen3 / LLaVA / InternVL / …)
  models  — per-model lines, colour = family, marker = group, size ∝ params
  7b      — matched-size subset (7–8 B models only), uniform marker size

Output filenames (saved to figures/accuracy_lineplot/):
  {blind|inst_blind|control}_vABC_{groups|family|models|7b}.png

Run from repo root:
  conda run -n zero python analysis/export_accuracy_variants.py
"""

import sys
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.constants import (
    GROUP_COLORS, GROUP_ORDER, GROUP_MARKER,
    MODEL_FAMILY, MODEL_FAMILY_COLORS, MODEL_SIZE_B,
)
from config import MODELS_ALL, MODELS_7B  # centralised model lists

EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "latex/AnonymousSubmission/LaTeX/figures/accuracy_lineplot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LISTS  — edit config.py to change which models appear in all figures
# ─────────────────────────────────────────────────────────────────────────────
MODELS_INCLUDE = MODELS_ALL   # all models → _all and _family figures

# ─────────────────────────────────────────────────────────────────────────────
# Shared style constants
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_ORDER = ['C', 'B', 'A']
VARIANT_LABEL = {'C': 'Original\n(C)', 'B': 'Weaker\n(B)', 'A': 'Pronoun.\n(A)'}
X_DEG = [0, 1, 2]

# Aggregated-view: group-level colors and styles
AGG_COLORS = {**GROUP_COLORS, 'Human': '#1565C0'}
AGG_STYLES = {
    'VLM backbone decoder':   ('o', '-'),
    'VLM':                    ('s', '-'),
    'standalone LLM (think)': ('D', '--'),
    'standalone LLM':         ('^', '--'),
    'Human':                  ('*', '-'),
}
AGG_LABEL = {
    'VLM backbone decoder':   'Backbone decoder',
    'VLM':                    'VLM',
    'standalone LLM (think)': 'Standalone LLM (think)',
    'standalone LLM':         'Standalone LLM',
    'Human':                  'Human',
}

# Family-level: same color palette as MODEL_FAMILY_COLORS
# Family line style: based on most-common group membership
FAMILY_LS = {
    'Qwen3-VL':  '-',
    'Qwen3':     '--',
    'LLaVA':     '-.',
    'InternVL':  ':',
    'Mistral':   '--',
    'Vicuna':    '--',
    'Phi':       '--',
    'Qwen2.5':   '--',
    'Human':     '-',
}
FAMILY_MARKER = {
    'Qwen3-VL':  's',
    'Qwen3':     '^',
    'LLaVA':     'o',
    'InternVL':  'D',
    'Mistral':   'v',
    'Vicuna':    'P',
    'Phi':       'X',
    'Qwen2.5':   'h',
    'Human':     '*',
}


def marker_size(param_b: float) -> float:
    """Marker area (s=) scaled by sqrt(param_b), normalised so 8B→80."""
    return 80 * math.sqrt(param_b / 8.0)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [accuracy_lineplot] {name}')


def _axis_base(ax, ylabel):
    ax.set_xticks(X_DEG)
    ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANT_ORDER], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Draw functions
# ─────────────────────────────────────────────────────────────────────────────

def draw_agg(ax, deg_model, deg_human, groups, ylabel, axhspan=False):
    """Group-mean lines (VLM / backbone / standalone / think / Human)."""
    plot_order = ['Human'] + [g for g in GROUP_ORDER if g in groups]
    for grp in plot_order:
        if grp == 'Human':
            if deg_human is None:
                continue
            rows = deg_human
        else:
            rows = deg_model[deg_model['model_group'] == grp]
        ys = [rows[rows['variant'] == v]['accuracy'].values[0]
              for v in VARIANT_ORDER
              if len(rows[rows['variant'] == v]) > 0]
        if len(ys) < len(VARIANT_ORDER):
            continue
        marker, ls = AGG_STYLES[grp]
        ax.plot(X_DEG, ys, color=AGG_COLORS[grp], lw=2, ls=ls,
                marker=marker, markersize=8 if grp != 'Human' else 11,
                label=AGG_LABEL[grp], zorder=3)
        ax.text(X_DEG[-1] + 0.06, ys[-1], f'{ys[-1]:.2f}',
                va='center', fontsize=8, color=AGG_COLORS[grp])
    if axhspan:
        ax.axhspan(0, 0.262, alpha=0.04, color='#1565C0')
    _axis_base(ax, ylabel)
    ax.legend(fontsize=8.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True)


def draw_family(ax, df_models, deg_human, ylabel, axhspan=False):
    """Family-mean lines, one line per model family."""
    df_models = df_models[df_models['model'].isin(MODELS_INCLUDE)].copy()
    df_models['family'] = df_models['model'].map(MODEL_FAMILY)
    deg_fam = (df_models.dropna(subset=['family'])
               .groupby(['family', 'variant'])['accuracy'].mean().reset_index())

    families = sorted(deg_fam['family'].unique())
    # Human first
    if deg_human is not None:
        ys = [deg_human[deg_human['variant'] == v]['accuracy'].values[0]
              for v in VARIANT_ORDER
              if len(deg_human[deg_human['variant'] == v]) > 0]
        if len(ys) == len(VARIANT_ORDER):
            ax.plot(X_DEG, ys,
                    color=MODEL_FAMILY_COLORS['Human'], lw=2, ls='-',
                    marker='*', markersize=11, label='Human', zorder=3)
            ax.text(X_DEG[-1] + 0.06, ys[-1], f'{ys[-1]:.2f}',
                    va='center', fontsize=8, color=MODEL_FAMILY_COLORS['Human'])

    for fam in families:
        sub = deg_fam[deg_fam['family'] == fam]
        ys = [sub[sub['variant'] == v]['accuracy'].values[0]
              for v in VARIANT_ORDER
              if len(sub[sub['variant'] == v]) > 0]
        if len(ys) < len(VARIANT_ORDER):
            continue
        c = MODEL_FAMILY_COLORS.get(fam, '#888888')
        ax.plot(X_DEG, ys, color=c, lw=1.8,
                ls=FAMILY_LS.get(fam, '-'),
                marker=FAMILY_MARKER.get(fam, 'o'), markersize=7,
                label=fam, zorder=3)
        ax.text(X_DEG[-1] + 0.06, ys[-1], f'{ys[-1]:.2f}',
                va='center', fontsize=7.5, color=c)

    if axhspan:
        ax.axhspan(0, 0.262, alpha=0.04, color='#1565C0')
    _axis_base(ax, ylabel)
    ax.legend(fontsize=7.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True)


def draw_permodel(ax, df_models, deg_human, model_list, ylabel,
                  scale_markers=True, axhspan=False):
    """Per-model lines: colour=family, marker=group, size∝params (if scale_markers)."""
    df_sub = df_models[df_models['model'].isin(model_list)].copy()
    df_sub['family'] = df_sub['model'].map(MODEL_FAMILY)

    if deg_human is not None:
        ys = [deg_human[deg_human['variant'] == v]['accuracy'].values[0]
              for v in VARIANT_ORDER
              if len(deg_human[deg_human['variant'] == v]) > 0]
        if len(ys) == len(VARIANT_ORDER):
            ax.plot(X_DEG, ys,
                    color=MODEL_FAMILY_COLORS['Human'], lw=2, ls='-',
                    marker='*', markersize=11, label='Human', zorder=4)

    for model, grp in df_sub[['model', 'model_group']].drop_duplicates().values:
        sub = df_sub[(df_sub['model'] == model) & (df_sub['variant'].isin(VARIANT_ORDER))]
        ys = [sub[sub['variant'] == v]['accuracy'].mean()
              for v in VARIANT_ORDER
              if len(sub[sub['variant'] == v]) > 0]
        if len(ys) < len(VARIANT_ORDER):
            continue
        fam = MODEL_FAMILY.get(model, 'Unknown')
        c   = MODEL_FAMILY_COLORS.get(fam, '#888888')
        mkr = GROUP_MARKER.get(grp, 'o')
        sz  = marker_size(MODEL_SIZE_B.get(model, 7.0)) if scale_markers else 60
        ax.plot(X_DEG, ys, color=c, lw=1.2, ls='-', alpha=0.75, zorder=2)
        for xi, yi in zip(X_DEG, ys):
            ax.scatter(xi, yi, color=c, marker=mkr, s=sz,
                       edgecolors='white', linewidths=0.5, zorder=3)

    # Legend: families (color) + groups (marker shape)
    fam_handles = [
        mlines.Line2D([], [], color=MODEL_FAMILY_COLORS.get(f, '#888'),
                      marker='o', ms=6, ls='-', label=f)
        for f in sorted(set(MODEL_FAMILY.get(m, '') for m in model_list) - {''})
    ]
    grp_handles = [
        mlines.Line2D([], [], color='gray',
                      marker=GROUP_MARKER.get(g, 'o'), ms=7, ls='none',
                      label=g.replace('standalone LLM', 'SA-LLM'))
        for g in GROUP_ORDER
        if any(df_sub[df_sub['model_group'] == g]['model'].isin(model_list))
    ]
    if deg_human is not None:
        fam_handles = [mlines.Line2D([], [], color=MODEL_FAMILY_COLORS['Human'],
                                     marker='*', ms=8, ls='-', label='Human')] + fam_handles

    if scale_markers:
        size_handles = [
            mlines.Line2D([], [], color='gray', marker='o', ls='none',
                          ms=math.sqrt(marker_size(b)) * 0.6,
                          label=f'{b:g}B')
            for b in [1, 4, 8, 32]
            if any(MODEL_SIZE_B.get(m, 0) == b for m in model_list)
        ]
        all_handles = fam_handles + grp_handles + size_handles
    else:
        all_handles = fam_handles + grp_handles

    ax.legend(handles=all_handles, fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=True)
    if axhspan:
        ax.axhspan(0, 0.262, alpha=0.04, color='#1565C0')
    _axis_base(ax, ylabel)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print('Loading exports…')
df_mb = pd.read_csv(EXPORTS / 'responses_model_blind.csv')
df_mi = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')
df_mc = pd.read_csv(EXPORTS / 'responses_model_control.csv')
df_h  = pd.read_csv(EXPORTS / 'responses_human.csv')

deg_human = df_h.groupby('variant')['accuracy'].mean().reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Generate all versions for each condition
# ─────────────────────────────────────────────────────────────────────────────

CONDS = [
    ('blind',      df_mb, 'Mean accuracy (blind condition)',        False),
    ('inst_blind', df_mi, 'Mean accuracy (inst_blind condition)',   True),
    ('control',    df_mc, 'Mean accuracy (control / real image)',   False),
]

for cond_name, df_m, ylabel, show_human in CONDS:
    print(f'\n── {cond_name} ──')
    deg_model = df_m.groupby(['variant', 'model_group'])['accuracy'].mean().reset_index()
    groups    = df_m['model_group'].unique().tolist()
    dh        = deg_human if show_human else None

    # ── _agg ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    draw_agg(ax, deg_model, dh, groups, ylabel, axhspan=show_human)
    plt.tight_layout()
    save(fig, f'{cond_name}_vABC_groups.png')

    # ── _family ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    draw_family(ax, df_m, dh, ylabel, axhspan=show_human)
    plt.tight_layout()
    save(fig, f'{cond_name}_vABC_family.png')

    # ── _all ──────────────────────────────────────────────────────────────────
    all_in_data = [m for m in MODELS_INCLUDE if m in df_m['model'].unique()]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    draw_permodel(ax, df_m, dh, all_in_data, ylabel,
                  scale_markers=True, axhspan=show_human)
    plt.tight_layout()
    save(fig, f'{cond_name}_vABC_models.png')

    # ── _7b ───────────────────────────────────────────────────────────────────
    seven_in_data = [m for m in MODELS_7B if m in df_m['model'].unique()]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    draw_permodel(ax, df_m, dh, seven_in_data, ylabel,
                  scale_markers=False, axhspan=show_human)
    plt.tight_layout()
    save(fig, f'{cond_name}_vABC_7b.png')

print('\nDone. Outputs in:', OUT_DIR)
