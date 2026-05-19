"""
Export model-scale (parameter count vs metric) plots.

Arguments
---------
--metric      agreement | accuracy
--agg         by_models | by_family | by_groups
--min_answers (int, default 348) participant qualification threshold

Only uses the human-study question subset (common_qids from load_human_data).
Models without a known parameter count are silently skipped.

Output
------
  figures/scale_{metric}_{agg}/
  Filenames include _q{n_questions}_n{n_humans} suffix.

  agreement → one figure per sub-metric (sbert, simcse, bertscore, chrf, rouge1, jaccard, exact)
  accuracy  → one figure per control variant (C, B, A); inst_blind condition

Aggregation modes
-----------------
  by_models  — scatter: one point per model, colour = family, marker = group,
                error bars = 95% CI across questions
  by_family  — line per (family × group), colour = family, marker = group,
                error bars at each point = 95% CI across questions
  by_groups  — scatter: colour = group, marker = group, error bars = 95% CI,
                + log-linear trend line per group (≥3 models)

Human baseline
--------------
  agreement: HH (Human–Human) mean ± 95% CI as horizontal dashed line
  accuracy:  human mean accuracy ± 95% CI as horizontal dashed line

Run from repo root:
  conda run -n zero python analysis/export_scale_plots.py --metric agreement --agg by_models
  conda run -n zero python analysis/export_scale_plots.py --metric accuracy  --agg by_family
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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
from utils.load_session import load_human_data
from config import MODELS_ALL, MODEL_GROUP, MIN_ANSWERS_DEFAULT, extend_pair_cache_with_yesno

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--metric', choices=['agreement', 'accuracy'], default='agreement')
parser.add_argument('--agg',    choices=['by_models', 'by_family', 'by_groups'],
                    default='by_models')
parser.add_argument('--min_answers', type=int, default=MIN_ANSWERS_DEFAULT)
parser.add_argument('--include_yesno', action='store_true',
                    help='Extend pair cache with yes/no question pairs.')
args = parser.parse_args()

METRIC = args.metric
AGG    = args.agg

EXPORTS   = ROOT / 'analysis/session2/exports'
OUT_DIR   = ROOT / f'latex/AnonymousSubmission/LaTeX/figures/{METRIC}_scale'
OUT_DIR.mkdir(parents=True, exist_ok=True)
AGG_SHORT = AGG.replace('by_', '')   # by_models→models, by_family→family, by_groups→groups

HH_COLOR = '#1565C0'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

# ─────────────────────────────────────────────────────────────────────────────
# Metric definitions
# ─────────────────────────────────────────────────────────────────────────────
AGREEMENT_METRICS = {
    'sbert':     ('SBERT cosine',  'sbert_score'),
    'simcse':    ('SimCSE cosine', 'simcse_score'),
    'bertscore': ('BERTScore F1',  'bertscore_f1'),
    'chrf':      ('chrF',          'chrf_score'),
    'rouge1':    ('ROUGE-1',       'rouge1_score'),
    'jaccard':   ('Token Jaccard', 'jaccard_score'),
    'exact':     ('Exact match',   'exact_score'),
}

ACCURACY_VARIANTS = ['C', 'B', 'A']
VAR_LABELS = {'C': 'Original (C)', 'B': 'Weaker (B)', 'A': 'Pronoun. (A)'}

# ─────────────────────────────────────────────────────────────────────────────
# Load human-study question subset
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nLoading human data (min_answers={args.min_answers})…')
participants, common_qids, human_df_full, _ = load_human_data(
    ROOT, min_answers=args.min_answers, translate=False, verbose=True)
n_humans  = len(participants)
print(f'  common_qids: {len(common_qids)}  |  participants: {n_humans}')

# ─────────────────────────────────────────────────────────────────────────────
# Load and filter source data
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nLoading {METRIC} data…')

pair_cache = pd.read_parquet(EXPORTS / 'pair_cache.parquet')
if args.include_yesno:
    print('  Extending pair_cache with yes/no question pairs…')
    pair_cache = extend_pair_cache_with_yesno(pair_cache, EXPORTS)
pair_cache = pair_cache[pair_cache['question_id'].isin(common_qids)]

if METRIC == 'agreement':
    raw = pair_cache[pair_cache['pair_type'] == 'HM'].copy()
else:
    raw = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')
    raw = raw[raw['question_id'].isin(common_qids)]

# ── Validate question count is consistent across variants ─────────────────────
for src, label in [(raw, 'model'), (pair_cache, 'pair_cache')]:
    counts = {v: src[src['variant'] == v]['question_id'].nunique()
              for v in ['C', 'B', 'A']}
    if len(set(counts.values())) > 1:
        raise ValueError(
            f'Question count differs across variants in {label}: {counts}')

n_questions = raw[raw['variant'] == 'C']['question_id'].nunique()
print(f'  rows after filtering: {len(raw)}  |  questions per variant: {n_questions}')

SUFFIX = f'_q{n_questions}_h{n_humans}' + ('_yesno' if args.include_yesno else '')

# ─────────────────────────────────────────────────────────────────────────────
# Models with known parameter counts
# ─────────────────────────────────────────────────────────────────────────────
MODELS_SIZED = [m for m in MODELS_ALL if m in MODEL_SIZE_B]

# ─────────────────────────────────────────────────────────────────────────────
# CI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ci95(values: np.ndarray) -> float:
    """95% CI half-width (1.96 × SEM). Returns 0 if fewer than 2 samples."""
    n = len(values)
    return 1.96 * values.std() / np.sqrt(n) if n >= 2 else 0.0


def _question_means(df: pd.DataFrame, col: str, grp_col: str) -> pd.DataFrame:
    """
    Per-(model, question) mean → rows: model, q_mean, for CI computation.
    Returns DataFrame with columns [grp_col, 'question_id', col].
    """
    return df.groupby([grp_col, 'question_id'])[col].mean().reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: per-model stats (mean + 95% CI) for a given column / variant
# ─────────────────────────────────────────────────────────────────────────────

def model_stats(col: str, variant: str | None = None) -> pd.DataFrame:
    """
    Returns DataFrame: model | mean | ci | size | family | group
    CI is 95% half-width across question-level means.
    """
    sub = raw if variant is None else raw[raw['variant'] == variant]
    grp_col = 'subject_2' if METRIC == 'agreement' else 'model'

    q_means = _question_means(sub, col, grp_col)

    rows = []
    for m in MODELS_SIZED:
        mq = q_means[q_means[grp_col] == m][col].values
        if len(mq) == 0:
            continue
        rows.append({
            'model':  m,
            'mean':   mq.mean(),
            'ci':     _ci95(mq),
            'size':   MODEL_SIZE_B[m],
            'family': MODEL_FAMILY.get(m, 'Unknown'),
            'group':  MODEL_GROUP.get(m, 'standalone LLM'),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Human baseline: mean ± 95% CI across questions
# ─────────────────────────────────────────────────────────────────────────────

def human_baseline(col: str, variant: str) -> tuple[float, float]:
    """
    agreement → Human–Human mean ± CI for col, filtered to variant.
    accuracy  → human mean accuracy ± CI for variant.
    Returns (mean, ci_half_width).
    """
    if METRIC == 'agreement':
        hh = pair_cache[(pair_cache['pair_type'] == 'HH') &
                        (pair_cache['variant'] == variant)]
        q_vals = hh.groupby('question_id')[col].mean().values
    else:
        hf = human_df_full[human_df_full['question_id'].isin(common_qids)]
        hf = hf[hf['variant'] == variant]
        q_vals = hf.groupby('question_id')['accuracy'].mean().values
    return (q_vals.mean(), _ci95(q_vals)) if len(q_vals) else (np.nan, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Shared axis helpers
# ─────────────────────────────────────────────────────────────────────────────
X_TICKS     = [0.5, 1, 2, 4, 7, 13, 32]
X_TICK_LABS = ['0.5', '1', '2', '4', '7', '13', '32']


def _configure_xaxis(ax):
    ax.set_xscale('log')
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICK_LABS, fontsize=8)
    ax.set_xlabel('Parameters (B)', fontsize=10)
    ax.set_xlim(0.4, 40)


def _draw_hh(ax, hh_mean: float, hh_ci: float):
    """Draw Human baseline as horizontal dashed line with CI band."""
    if np.isnan(hh_mean):
        return
    ax.axhline(hh_mean, color=HH_COLOR, lw=1.6, ls='--', zorder=1,
               label='Human baseline')
    if hh_ci > 0:
        ax.axhspan(hh_mean - hh_ci, hh_mean + hh_ci,
                   color=HH_COLOR, alpha=0.08, zorder=0)


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [scale_{METRIC}_{AGG}] {name}')


# ─────────────────────────────────────────────────────────────────────────────
# ── by_models ─────────────────────────────────────────────────────────────────
# Scatter + error bars; colour = family, marker = group
# ─────────────────────────────────────────────────────────────────────────────

def _plot_by_models(ax, stats: pd.DataFrame, ylabel: str, hh_mean: float, hh_ci: float):
    _draw_hh(ax, hh_mean, hh_ci)

    for _, row in stats.iterrows():
        color = MODEL_FAMILY_COLORS.get(row['family'], '#888888')
        mkr   = GROUP_MARKER.get(row['group'], 'o')
        ax.errorbar(row['size'], row['mean'], yerr=row['ci'],
                    fmt=mkr, color=color, ms=7, capsize=3, capthick=0.8,
                    elinewidth=0.8, alpha=0.88, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.5)

    ax.set_ylabel(ylabel, fontsize=10)
    _configure_xaxis(ax)

    present_fams = sorted(stats['family'].unique())
    fam_handles = [
        mlines.Line2D([], [], color=MODEL_FAMILY_COLORS.get(f, '#888'),
                      marker='o', ms=6, ls='none', label=f)
        for f in present_fams
    ]
    present_grps = stats['group'].unique()
    grp_handles = [
        mlines.Line2D([], [], color='gray', marker=GROUP_MARKER.get(g, 'o'),
                      ms=7, ls='none',
                      label=g.replace('standalone LLM', 'SA-LLM')
                             .replace('VLM backbone decoder', 'Backbone'))
        for g in GROUP_ORDER if g in present_grps
    ]
    hh_handle = mlines.Line2D([], [], color=HH_COLOR, ls='--', lw=1.6,
                               label='Human baseline')
    ax.legend(handles=fam_handles + grp_handles + [hh_handle],
              fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── by_family ─────────────────────────────────────────────────────────────────
# Line per (family × group); colour = family, marker = group shape,
# solid = no-think / dashed = think.  Error bars at each point.
# ─────────────────────────────────────────────────────────────────────────────

def _plot_by_family(ax, stats: pd.DataFrame, ylabel: str, hh_mean: float, hh_ci: float):
    _draw_hh(ax, hh_mean, hh_ci)

    # Group by (family, group) — avoids vertical connections for same-size models
    lines: dict[tuple, list] = defaultdict(list)
    for _, row in stats.iterrows():
        key = (row['family'], row['group'])
        lines[key].append(row)

    seen_fam: set[str] = set()
    all_handles: list = []

    for (fam, grp), rows in sorted(lines.items()):
        rows_s = sorted(rows, key=lambda r: r['size'])
        xs  = [r['size']  for r in rows_s]
        ys  = [r['mean']  for r in rows_s]
        cis = [r['ci']    for r in rows_s]

        color = MODEL_FAMILY_COLORS.get(fam, '#888888')
        mkr   = GROUP_MARKER.get(grp, 'o')
        think = grp == 'standalone LLM (think)'
        ls    = '--' if think else '-'
        lw    = 1.4 if think else 1.8

        ax.errorbar(xs, ys, yerr=cis, fmt=mkr, color=color,
                    ls=ls, lw=lw, ms=6, capsize=3, capthick=0.7,
                    elinewidth=0.7, markeredgecolor='white', markeredgewidth=0.4,
                    alpha=0.88, zorder=3)

        if fam not in seen_fam:
            all_handles.append(
                mlines.Line2D([], [], color=color, ls='-', lw=2,
                              marker='o', ms=5, label=fam))
            seen_fam.add(fam)

    # Group marker guide
    present_grps = stats['group'].unique()
    grp_handles = [
        mlines.Line2D([], [], color='gray', marker=GROUP_MARKER.get(g, 'o'),
                      ms=7, ls='none',
                      label=g.replace('standalone LLM', 'SA-LLM')
                             .replace('VLM backbone decoder', 'Backbone'))
        for g in GROUP_ORDER if g in present_grps
    ]
    style_handles = [
        mlines.Line2D([], [], color='gray', ls='-',  lw=2, label='no-think'),
        mlines.Line2D([], [], color='gray', ls='--', lw=2, label='think'),
    ]
    hh_handle = mlines.Line2D([], [], color=HH_COLOR, ls='--', lw=1.6,
                               label='Human baseline')

    ax.set_ylabel(ylabel, fontsize=10)
    _configure_xaxis(ax)
    ax.legend(handles=all_handles + grp_handles + style_handles + [hh_handle],
              fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── by_groups ─────────────────────────────────────────────────────────────────
# Scatter: colour = group, marker = group; error bars = 95% CI across questions
# (pooled across all models in that group); log-linear trend line (≥3 models).
# ─────────────────────────────────────────────────────────────────────────────

def _plot_by_groups(ax, stats: pd.DataFrame, ylabel: str, hh_mean: float, hh_ci: float):
    _draw_hh(ax, hh_mean, hh_ci)

    grp_col = 'subject_2' if METRIC == 'agreement' else 'model'

    for grp in GROUP_ORDER:
        gdf = stats[stats['group'] == grp]
        if gdf.empty:
            continue
        color = GROUP_COLORS.get(grp, '#888888')
        mkr   = GROUP_MARKER.get(grp, 'o')
        label = grp.replace('standalone LLM', 'SA-LLM') \
                   .replace('VLM backbone decoder', 'Backbone')

        ax.errorbar(gdf['size'].values, gdf['mean'].values, yerr=gdf['ci'].values,
                    fmt=mkr, color=color, ms=7, capsize=3, capthick=0.8,
                    elinewidth=0.8, alpha=0.85, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.5, label=label)

        # Log-linear trend line if ≥3 distinct sizes
        xs = gdf['size'].values
        ys = gdf['mean'].values
        if len(np.unique(xs)) >= 3:
            coeffs = np.polyfit(np.log(xs), ys, 1)
            x_rng  = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 50)
            y_fit  = np.polyval(coeffs, np.log(x_rng))
            ax.plot(x_rng, y_fit, color=color, ls='--', lw=1.2,
                    alpha=0.45, zorder=2)

    hh_handle = mlines.Line2D([], [], color=HH_COLOR, ls='--', lw=1.6,
                               label='Human baseline')
    ax.set_ylabel(ylabel, fontsize=10)
    _configure_xaxis(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [hh_handle],
              labels=labels + ['Human baseline'],
              fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
_PLOT_FN = {
    'by_models': _plot_by_models,
    'by_family': _plot_by_family,
    'by_groups': _plot_by_groups,
}
plot_fn = _PLOT_FN[AGG]


# ─────────────────────────────────────────────────────────────────────────────
# Generate figures
# ─────────────────────────────────────────────────────────────────────────────

if METRIC == 'agreement':
    for key, (metric_name, col) in AGREEMENT_METRICS.items():
        print(f'\n── {metric_name} ──')
        stats = model_stats(col, variant='C')
        if stats.empty:
            print('  (no data, skipped)')
            continue
        hh_mean, hh_ci = human_baseline(col, variant='C')

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_fn(ax, stats,
                f'Mean {metric_name} — HM, human-study subset, variant C',
                hh_mean, hh_ci)
        plt.tight_layout()
        _save(fig, f'inst_blind_{AGG_SHORT}_{key}{SUFFIX}.png')

else:  # accuracy
    for var in ACCURACY_VARIANTS:
        print(f'\n── Accuracy variant {var} ──')
        stats = model_stats('accuracy', variant=var)
        if stats.empty:
            print('  (no data, skipped)')
            continue
        hh_mean, hh_ci = human_baseline('accuracy', variant=var)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_fn(ax, stats,
                f'Mean accuracy — inst_blind, {VAR_LABELS[var]}',
                hh_mean, hh_ci)
        plt.tight_layout()
        _save(fig, f'inst_blind_{AGG_SHORT}_v{var}{SUFFIX}.png')

print(f'\nDone. Saved to: {OUT_DIR}')
print(f'Suffix: {SUFFIX}')
