"""
Export control-variant accuracy degradation figures.

Two output folders:

  figures/accuracy_variants_family/
    One figure per model family (Qwen3, LLaVA-Mistral, InternVL, …).
    Each line = one model.  Colour + marker = model group (VLM / backbone / LLM / think).

  figures/accuracy_variants_group/
    One figure per model group (VLM, backbone decoder, standalone LLM, think).
    Each line = one model.  Colour + marker = model family.

Filenames:
  {blind|inst_blind|control}_vABC_{family}.png   (e.g. Qwen3.png, LLaVA-Mistral.png)
  {blind|inst_blind|control}_vABC_{slug}.png     (e.g. vlm.png, backbone.png)

Run from repo root:
  conda run -n zero python analysis/export_accuracy_variants.py
"""

import sys
import math
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.constants import (
    GROUP_COLORS, GROUP_ORDER, GROUP_MARKER,
    MODEL_FAMILY, MODEL_FAMILY_COLORS,
)
from config import MODEL_LABEL_SHORT

EXPORTS = ROOT / 'analysis/session2/exports'

FAM_DIR = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/accuracy_variants_family'
GRP_DIR = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/accuracy_variants_group'
FAM_DIR.mkdir(parents=True, exist_ok=True)
GRP_DIR.mkdir(parents=True, exist_ok=True)

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
    'Qwen3-VL':    's',
    'Qwen3':       '^',
    'LLaVA-1.5':   'o',
    'LLaVA-Mistral':'D',
    'LLaVA-Vicuna': 'P',
    'InternVL':    'X',
    'Mistral':     'v',
    'Vicuna':      'h',
    'Phi':         '*',
    'Qwen2.5':     'p',
}

GROUP_LABEL = {
    'VLM':                    'VLM',
    'VLM backbone decoder':   'Backbone decoder',
    'standalone LLM':         'Standalone LLM',
    'standalone LLM (think)': 'Standalone LLM (think)',
}

GROUP_SLUG = {
    'VLM':                    'vlm',
    'VLM backbone decoder':   'backbone',
    'standalone LLM':         'standalone_llm',
    'standalone LLM (think)': 'standalone_llm_think',
}


def _short(model: str) -> str:
    return MODEL_LABEL_SHORT.get(model, model)


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
        ax.text(X_DEG[-1] + 0.08, ys[-1], f'{ys[-1]:.2f}',
                va='center', fontsize=7.5, color='#1565C0')


def save(fig, out_dir, name, folder_tag):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{folder_tag}] {name}')


# ── Per-family figure: colour+marker = group ─────────────────────────────────

def draw_by_family(ax, df_m, family, models, deg_human, ylabel):
    """One line per model in `models`; colour+marker encode model group."""
    df_sub = df_m[df_m['model'].isin(models)].copy()
    _human_line(ax, deg_human)

    plotted_groups = set()
    for model in models:
        sub = df_sub[(df_sub['model'] == model) & (df_sub['variant'].isin(VARIANT_ORDER))]
        ys = [sub[sub['variant'] == v]['accuracy'].mean()
              for v in VARIANT_ORDER if len(sub[sub['variant'] == v]) > 0]
        if len(ys) < len(VARIANT_ORDER):
            continue
        grp = df_sub[df_sub['model'] == model]['model_group'].iloc[0]
        c   = GROUP_COLORS.get(grp, '#888888')
        mkr = GROUP_MARKER.get(grp, 'o')
        ax.plot(X_DEG, ys, color=c, lw=1.8, ls='-',
                marker=mkr, markersize=7, alpha=0.85, zorder=3)
        ax.text(X_DEG[-1] + 0.08, ys[-1], _short(model),
                va='center', fontsize=7, color=c)
        plotted_groups.add(grp)

    ax.set_title(f'{family}', fontsize=11, fontweight='bold')
    _axis_base(ax, ylabel)

    # Legend: group → colour + marker
    handles = []
    if deg_human is not None:
        handles.append(mlines.Line2D([], [], color='#1565C0', marker='*',
                                     ms=8, ls='--', label='Human'))
    for grp in GROUP_ORDER:
        if grp not in plotted_groups:
            continue
        handles.append(mlines.Line2D(
            [], [], color=GROUP_COLORS.get(grp, '#888'),
            marker=GROUP_MARKER.get(grp, 'o'), ms=7, ls='-',
            label=GROUP_LABEL.get(grp, grp),
        ))
    ax.legend(handles=handles, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)


# ── Per-group figure: colour+marker = family ──────────────────────────────────

def draw_by_group(ax, df_m, group, models, deg_human, ylabel):
    """One line per model in `models`; colour+marker encode model family."""
    df_sub = df_m[df_m['model'].isin(models)].copy()
    _human_line(ax, deg_human)

    plotted_families = set()
    for model in models:
        sub = df_sub[(df_sub['model'] == model) & (df_sub['variant'].isin(VARIANT_ORDER))]
        ys = [sub[sub['variant'] == v]['accuracy'].mean()
              for v in VARIANT_ORDER if len(sub[sub['variant'] == v]) > 0]
        if len(ys) < len(VARIANT_ORDER):
            continue
        fam = MODEL_FAMILY.get(model, 'Unknown')
        c   = MODEL_FAMILY_COLORS.get(fam, '#888888')
        mkr = FAMILY_MARKER.get(fam, 'o')
        ax.plot(X_DEG, ys, color=c, lw=1.8, ls='-',
                marker=mkr, markersize=7, alpha=0.85, zorder=3)
        ax.text(X_DEG[-1] + 0.08, ys[-1], _short(model),
                va='center', fontsize=7, color=c)
        plotted_families.add(fam)

    ax.set_title(f'{GROUP_LABEL.get(group, group)}', fontsize=11, fontweight='bold')
    _axis_base(ax, ylabel)

    # Legend: family → colour + marker
    handles = []
    if deg_human is not None:
        handles.append(mlines.Line2D([], [], color='#1565C0', marker='*',
                                     ms=8, ls='--', label='Human'))
    for fam in sorted(plotted_families):
        handles.append(mlines.Line2D(
            [], [], color=MODEL_FAMILY_COLORS.get(fam, '#888'),
            marker=FAMILY_MARKER.get(fam, 'o'), ms=7, ls='-', label=fam,
        ))
    ax.legend(handles=handles, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)


# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading exports…')
df_mb = pd.read_csv(EXPORTS / 'responses_model_blind.csv')
df_mi = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')
df_mc = pd.read_csv(EXPORTS / 'responses_model_control.csv')
df_h  = pd.read_csv(EXPORTS / 'responses_human.csv')

deg_human = df_h.groupby('variant')['accuracy'].mean().reset_index()

N_QUESTIONS = df_mi['question_id'].nunique()
N_HUMANS    = df_h['participant'].nunique()
SUFFIX      = f'_q{N_QUESTIONS}_h{N_HUMANS}'
print(f'questions={N_QUESTIONS}  humans={N_HUMANS}')

# Build family → [model] and group → [model] maps from inst_blind data
model_group = (df_mi[['model', 'model_group']].drop_duplicates()
               .set_index('model')['model_group'].to_dict())

by_family: dict[str, list[str]] = defaultdict(list)
by_group:  dict[str, list[str]] = defaultdict(list)
for model, grp in sorted(model_group.items()):
    fam = MODEL_FAMILY.get(model, 'Unknown')
    by_family[fam].append(model)
    by_group[grp].append(model)

# ── Generate figures ──────────────────────────────────────────────────────────
CONDS = [
    ('blind',      df_mb, 'Mean accuracy (blind)',      False),
    ('inst_blind', df_mi, 'Mean accuracy (inst_blind)', True),
    ('control',    df_mc, 'Mean accuracy (control)',    False),
]

for cond_name, df_m, ylabel, show_human in CONDS:
    dh = deg_human if show_human else None
    print(f'\n── {cond_name} ──')

    # ── Per-family ────────────────────────────────────────────────────────────
    for family, models in sorted(by_family.items()):
        avail = [m for m in models if m in df_m['model'].unique()]
        if not avail:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_by_family(ax, df_m, family, avail, dh, ylabel)
        plt.tight_layout()
        # sanitise family name for filename
        slug = re.sub(r'[^\w\-]', '_', family)
        save(fig, FAM_DIR, f'{cond_name}_vABC_{slug}{SUFFIX}.png', 'accuracy_variants_family')

    # ── Per-group ─────────────────────────────────────────────────────────────
    for group, models in sorted(by_group.items()):
        avail = [m for m in models if m in df_m['model'].unique()]
        if not avail:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_by_group(ax, df_m, group, avail, dh, ylabel)
        plt.tight_layout()
        slug = GROUP_SLUG.get(group, re.sub(r'[^\w]', '_', group).lower())
        save(fig, GRP_DIR, f'{cond_name}_vABC_{slug}{SUFFIX}.png', 'accuracy_variants_group')

print(f'\nDone.\n  Family plots: {FAM_DIR}\n  Group plots:  {GRP_DIR}')
