"""
Export per-question accuracy scatter plots: human vs model accuracy.

For each control variant (C, B, A) produces one scatter figure where:
  x-axis = human accuracy per question (inst_blind)
  y-axis = model accuracy per question, averaged within each aggregate unit

Aggregation modes (--agg):
  model_groups  — one colour per model group (VLM / Backbone / LLM / LLM-think)
  model_family  — one colour per model family (LLaVA-1.5 / Qwen3 / InternVL / …)

Each point = one question × one aggregate unit.
Reference line y = x, quadrant labels, and Pearson r per unit.

Output:
  figures/accuracy_scatter/
  Filenames: inst_blind_v{C,B,A}_{groups,family}_q{n_questions}_h{n_humans}.png

Run from repo root:
  conda run -n zero python analysis/export_accuracy_scatter.py --agg model_groups
  conda run -n zero python analysis/export_accuracy_scatter.py --agg model_family
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.constants import (
    GROUP_COLORS, GROUP_ORDER, GROUP_MARKER,
    MODEL_FAMILY, MODEL_FAMILY_COLORS,
)
from export_helpers import load_human_subset, read_export
from config import MODELS_ALL, MODEL_GROUP, MIN_ANSWERS_DEFAULT

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--agg', choices=['model_groups', 'model_family', 'per_model'],
                    default='model_groups')
parser.add_argument('--variant', choices=['C', 'B', 'A'], default='C',
                    help='Control variant to plot (default: C).')
parser.add_argument('--min_answers', type=int, default=MIN_ANSWERS_DEFAULT)
args = parser.parse_args()

AGG = args.agg

# ─────────────────────────────────────────────────────────────────────────────
# Units to exclude from figures (constant-output or degenerate models)
# Add/remove family or group names here to control what appears in the plot.
# ─────────────────────────────────────────────────────────────────────────────
EXCLUDE_UNITS: list[str] = [
    'Phi',      # r ≈ 0, p = 0.89 — near-uniform accuracy, no correlation with human difficulty
    'Vicuna',   # constant zero — empty outputs (template bug)
    'Mistral',  # constant zero — verbose abstentions → all incorrect
]

OUT_DIR   = ROOT / 'latex/AnonymousSubmission/LaTeX/figures/accuracy_scatter'
OUT_DIR.mkdir(parents=True, exist_ok=True)
AGG_SHORT = {'model_groups': 'groups', 'model_family': 'family'}[AGG]

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

VARIANTS   = ['C', 'B', 'A']
VAR_LABELS = {'C': 'Original (C)', 'B': 'Weaker (B)', 'A': 'Pronoun. (A)'}

# ─────────────────────────────────────────────────────────────────────────────
# Load human-study subset
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nLoading human data (min_answers={args.min_answers})…')
participants, common_qids, human_df_full, _ = load_human_subset(
    ROOT, min_answers=args.min_answers, translate=False, verbose=True)
n_humans = len(participants)
print(f'  common_qids: {len(common_qids)}  |  participants: {n_humans}')

# ─────────────────────────────────────────────────────────────────────────────
# Load model responses (inst_blind) filtered to human-study subset
# ─────────────────────────────────────────────────────────────────────────────
print('\nLoading model responses (inst_blind)…')
df_mi = read_export(ROOT, 'responses_model_inst_blind.csv', subset_qids=common_qids)
df_mi = df_mi[df_mi['model'].isin(MODELS_ALL)].copy()

# Attach model_family if not already present
df_mi['model_family'] = df_mi['model'].map(MODEL_FAMILY)
# Ensure model_group column is present (may already exist as 'model_group')
if 'model_group' not in df_mi.columns:
    df_mi['model_group'] = df_mi['model'].map(MODEL_GROUP)

# Validate question counts across variants
counts = {v: df_mi[df_mi['variant'] == v]['question_id'].nunique()
          for v in VARIANTS}
if len(set(counts.values())) > 1:
    raise ValueError(f'Question count differs across variants: {counts}')

n_questions = counts['C']
SUFFIX = f'_q{n_questions}_h{n_humans}'
print(f'  rows: {len(df_mi)}  |  questions per variant: {n_questions}')

# ─────────────────────────────────────────────────────────────────────────────
# Human accuracy per question
# ─────────────────────────────────────────────────────────────────────────────
hf = human_df_full[human_df_full['question_id'].isin(common_qids)].copy()


def human_acc_per_question(variant: str) -> pd.Series:
    """Mean human accuracy per question_id for a given variant."""
    return (hf[hf['variant'] == variant]
            .groupby('question_id')['accuracy'].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation config  (not used in per_model mode)
# ─────────────────────────────────────────────────────────────────────────────
if AGG == 'per_model':
    pass  # handled entirely in the main loop below
elif AGG == 'model_groups':
    # unit_col: the column that defines the aggregate unit label
    unit_col   = 'model_group'
    unit_order = GROUP_ORDER
    unit_colors = GROUP_COLORS
    unit_markers = GROUP_MARKER

    def unit_label(u: str) -> str:
        return (u.replace('standalone LLM', 'SA-LLM')
                 .replace('VLM backbone decoder', 'Backbone'))

    def unit_color(u: str):
        return GROUP_COLORS.get(u, '#888888')

    def unit_marker(u: str):
        return GROUP_MARKER.get(u, 'o')

else:  # model_family
    unit_col   = 'model_family'
    # Build ordered list: families present in data
    all_families = sorted(set(MODEL_FAMILY.values()))
    unit_order   = all_families
    unit_colors  = MODEL_FAMILY_COLORS

    def unit_label(u: str) -> str:
        return u

    def unit_color(u: str):
        return MODEL_FAMILY_COLORS.get(u, '#888888')

    def unit_marker(u: str) -> str:
        # for model_family, derive marker from the group of the family's models
        # Use the most common group within the family
        fam_models = [m for m, f in MODEL_FAMILY.items() if f == u]
        groups = [MODEL_GROUP.get(m, '') for m in fam_models]
        from collections import Counter
        if not groups:
            return 'o'
        common_grp = Counter(groups).most_common(1)[0][0]
        return GROUP_MARKER.get(common_grp, 'o')


# ─────────────────────────────────────────────────────────────────────────────
# Quadrant annotation helper
# ─────────────────────────────────────────────────────────────────────────────
def _annotate_quadrants(ax):
    kw = dict(fontsize=7.5, color='#999999', alpha=0.75, transform=ax.transAxes)
    ax.text(0.04, 0.95, 'Both\nfail',           va='top',    **kw)
    ax.text(0.62, 0.95, 'Human ✓\nModel ✗',     va='top',    **kw)
    ax.text(0.04, 0.05, 'Human ✗\nModel ✓',     va='bottom', **kw)
    ax.text(0.62, 0.05, 'Both\ncorrect',         va='bottom', **kw)
    ax.axhline(0.5, color='gray', lw=0.6, ls='--', alpha=0.35, zorder=0)
    ax.axvline(0.5, color='gray', lw=0.6, ls='--', alpha=0.35, zorder=0)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation stats helper
# ─────────────────────────────────────────────────────────────────────────────
def corr_stats(x: pd.Series, y: pd.Series):
    """Return (r, p, ci_lo, ci_hi) with 95% CI via Fisher z-transform.

    Returns (nan, nan, nan, nan) if either series is constant.
    """
    if x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan, np.nan, np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    n = len(x)
    z  = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return r, p, lo, hi


def r_label(r, p, lo, hi) -> str:
    """Format r, 95% CI, and significance stars for legend."""
    if np.isnan(r):
        return 'r=N/A (const)'
    stars = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
    return f'r={r:.2f} [{lo:.2f},{hi:.2f}] {stars}'


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [accuracy_scatter] {name}')


# ─────────────────────────────────────────────────────────────────────────────
# per_model: multi-panel figure, one subplot per group, models as series
# ─────────────────────────────────────────────────────────────────────────────
if AGG == 'per_model':
    import matplotlib.cm as cm

    # Palette per group — 20 distinct hues, cycling if needed
    _PALETTE = [plt.cm.tab20(i / 20) for i in range(20)]

    for var in [args.variant]:
        print(f'\n── Variant {var} ({VAR_LABELS[var]}) ──')

        h_acc = human_acc_per_question(var)
        df_v  = df_mi[df_mi['variant'] == var]

        # Per-(model, question) mean accuracy
        model_q_acc = (df_v.groupby(['model', 'question_id'])['accuracy']
                       .mean()
                       .reset_index()
                       .rename(columns={'accuracy': 'model_acc'}))
        model_q_acc['human_acc']   = model_q_acc['question_id'].map(h_acc)
        model_q_acc['model_group'] = model_q_acc['model'].map(
            lambda m: df_mi.loc[df_mi['model'] == m, 'model_group'].iloc[0]
            if (df_mi['model'] == m).any() else '')
        model_q_acc['model_family'] = model_q_acc['model'].map(MODEL_FAMILY)
        model_q_acc = model_q_acc.dropna(subset=['human_acc'])

        # Exclude degenerate families
        model_q_acc = model_q_acc[
            ~model_q_acc['model_family'].isin(EXCLUDE_UNITS)
        ]

        groups_present = [g for g in GROUP_ORDER
                          if g in model_q_acc['model_group'].values]
        if not groups_present:
            print('  (no data, skipped)')
            continue

        ncols = 2
        nrows = (len(groups_present) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 5.5 * nrows),
                                 squeeze=False)

        for idx, grp in enumerate(groups_present):
            ax = axes[idx // ncols][idx % ncols]
            grp_df = model_q_acc[model_q_acc['model_group'] == grp]
            models_in_grp = sorted(grp_df['model'].unique())

            ax.plot([0, 1], [0, 1], color='#CCCCCC', lw=1, ls='-', zorder=0)
            _annotate_quadrants(ax)

            handles = []
            grp_marker = GROUP_MARKER.get(grp, 'o')

            for m_idx, model in enumerate(models_in_grp):
                sub   = grp_df[grp_df['model'] == model]
                color = _PALETTE[m_idx % len(_PALETTE)]
                # Shorten label: strip redundant suffix tokens
                mlabel = (model.replace(' (think)', '-T')
                               .replace('standalone LLM', '')
                               .strip())

                ax.scatter(sub['human_acc'], sub['model_acc'],
                           color=color, s=22, alpha=0.60, marker=grp_marker,
                           edgecolors='white', linewidths=0.3, zorder=3)

                r, p, lo, hi = corr_stats(sub['human_acc'], sub['model_acc'])
                rl = r_label(r, p, lo, hi)
                handles.append(
                    mlines.Line2D([], [], color=color, marker=grp_marker,
                                  ms=7, ls='none',
                                  label=f'{mlabel}  ({rl})'))

            ax.set_xlabel('Human accuracy (inst_blind)', fontsize=10)
            ax.set_ylabel('Model accuracy (inst_blind)', fontsize=10)
            ax.set_title(grp, fontsize=11, color=GROUP_COLORS.get(grp, '#333333'),
                         fontweight='bold')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(handles=handles, fontsize=7,
                      loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      ncol=1, frameon=True)

        # Hide unused panels
        for idx in range(len(groups_present), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(
            f'Per-question Human vs. Model Accuracy — {VAR_LABELS[var]}',
            fontsize=13, y=1.01)
        plt.tight_layout()
        save(fig, f'inst_blind_v{var}_{AGG_SHORT}{SUFFIX}.png')

    print(f'\nDone. Saved to: {OUT_DIR}')
    print(f'Suffix: {SUFFIX}')
    import sys; sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop: one figure per variant
# ─────────────────────────────────────────────────────────────────────────────
for var in [args.variant]:
    print(f'\n── Variant {var} ({VAR_LABELS[var]}) ──')

    h_acc = human_acc_per_question(var)      # Series indexed by question_id
    df_v  = df_mi[df_mi['variant'] == var]

    # Per-(unit, question) mean model accuracy
    unit_q_acc = (df_v.groupby([unit_col, 'question_id'])['accuracy']
                  .mean()
                  .reset_index()
                  .rename(columns={'accuracy': 'model_acc', unit_col: 'unit'}))
    unit_q_acc['human_acc'] = unit_q_acc['question_id'].map(h_acc)
    unit_q_acc = unit_q_acc.dropna(subset=['human_acc'])

    present_units = [u for u in unit_order
                     if u in unit_q_acc['unit'].values
                     and u not in EXCLUDE_UNITS]
    if not present_units:
        print('  (no data, skipped)')
        continue

    fig, ax = plt.subplots(figsize=(7, 6))

    # Reference line y = x
    ax.plot([0, 1], [0, 1], color='#CCCCCC', lw=1, ls='-', zorder=0)

    handles = []
    r_lines = []

    for unit in present_units:
        sub = unit_q_acc[unit_q_acc['unit'] == unit]
        color  = unit_color(unit)
        marker = unit_marker(unit)
        label  = unit_label(unit)

        ax.scatter(sub['human_acc'], sub['model_acc'],
                   color=color, s=30, alpha=0.65, marker=marker,
                   edgecolors='white', linewidths=0.4, zorder=3)

        r, p, lo, hi = corr_stats(sub['human_acc'], sub['model_acc'])
        rl = r_label(r, p, lo, hi)
        r_lines.append(f'{label}: {rl}')

        handles.append(
            mlines.Line2D([], [], color=color, marker=marker, ms=7,
                          ls='none', label=f'{label}  ({rl})'))

    _annotate_quadrants(ax)

    ax.set_xlabel('Human accuracy (inst_blind)', fontsize=11)
    ax.set_ylabel('Model accuracy (inst_blind)', fontsize=11)
    ax.set_title(
        f'Per-question Human vs. Model Accuracy\n'
        f'{VAR_LABELS[var]} — aggregated by {AGG.replace("_", " ")}',
        fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.legend(handles=handles, fontsize=8,
              loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=2, frameon=True)

    plt.tight_layout()
    save(fig, f'inst_blind_v{var}_{AGG_SHORT}{SUFFIX}.png')

print(f'\nDone. Saved to: {OUT_DIR}')
print(f'Suffix: {SUFFIX}')
