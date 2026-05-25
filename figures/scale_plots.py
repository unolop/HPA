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
  figures/scale_scatter/
  Filenames include the metric name and _q{n_questions}_h{n_humans} suffix.

  agreement → one figure per sub-metric (sbert, simcse, bertscore, chrf, rouge1, jaccard, exact)
  accuracy  → one figure per control variant (C, B, A); inst_blind condition

Aggregation modes
-----------------
  by_models  — scatter: one point per model, colour = family, marker = group,
                error bars = 95% CI across questions
  by_groups  — scatter: colour = group, marker = group, error bars = 95% CI,
                + log-linear trend line per group (≥3 models)

Human baseline
--------------
  agreement: HH (Human–Human) mean ± 95% CI as horizontal dashed line
  accuracy:  human mean accuracy ± 95% CI as horizontal dashed line

Run from repo root:
  conda run -n zero python figures/scale_plots.py --metric agreement --agg by_models
  conda run -n zero python figures/scale_plots.py --metric accuracy  --agg by_family
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from utils.constants import (
    GROUP_COLORS, GROUP_ORDER, GROUP_MARKER, GROUP_HOLLOW, GROUP_PAIR_ORDER,
    MODEL_FAMILY, MODEL_FAMILY_COLORS, MODEL_SIZE_B,
)
from helpers import clear_output_plots, get_exports_dir, load_human_subset, load_pair_cache, read_export
from config import MODELS_ALL, MODEL_GROUP, MIN_ANSWERS_DEFAULT

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--metric', choices=['agreement', 'accuracy'], default='agreement')
parser.add_argument('--agg',    choices=['by_models', 'by_groups', 'by_group'],
                    default='by_models')
parser.add_argument('--min_answers', type=int, default=MIN_ANSWERS_DEFAULT)
parser.add_argument('--overwrite', action='store_true',
                    help='Delete existing plot files in the output folder before exporting.')
args = parser.parse_args()

METRIC = args.metric
AGG    = args.agg

EXPORTS   = get_exports_dir(ROOT)
OUT_DIR   = ROOT / 'figures/scale_scatter'
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)
AGG_SHORT = AGG.replace('by_', '')   # by_models→models, by_family→family, by_groups→groups

HH_COLOR = '#222222'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.labelsize':    11,
    'axes.titlesize':    13,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
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
participants, common_qids, human_df_full, _ = load_human_subset(
    ROOT, min_answers=args.min_answers, translate=False, verbose=True)
n_humans  = len(participants)
print(f'  common_qids: {len(common_qids)}  |  participants: {n_humans}')

# ─────────────────────────────────────────────────────────────────────────────
# Load and filter source data
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nLoading {METRIC} data…')

# Auto-compute HM pairs for any models with complete inst_blind data
# that are not yet in the cache, then load the (possibly updated) cache.
pair_cache = load_pair_cache(ROOT, include_yesno=True, verbose=True)
pair_cache = pair_cache[pair_cache['question_id'].isin(common_qids)]

if METRIC == 'agreement':
    raw = pair_cache[pair_cache['pair_type'] == 'HM'].copy()
else:
    raw = read_export(ROOT, 'responses_model_inst_blind.csv')
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

if METRIC == 'agreement':
    hh_sub = pair_cache[(pair_cache['pair_type'] == 'HH') &
                        (pair_cache['question_id'].isin(common_qids))]
    n_humans_suffix = len(set(hh_sub['subject_1']).union(set(hh_sub['subject_2'])))
else:
    n_humans_suffix = n_humans

SUFFIX = f'_q{n_questions}_h{n_humans_suffix}_yesno'

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
def _configure_xaxis(ax):
    sizes = sorted(set(MODEL_SIZE_B.values()))
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(
        [str(int(s)) if s == int(s) else str(s) for s in sizes],
        fontsize=8,
    )
    ax.set_xlabel('Parameters (B)', fontsize=10)
    margin = 0.15  # log-space padding
    ax.set_xlim(sizes[0] * (10 ** -margin), sizes[-1] * (10 ** margin))


def _draw_hh(ax, hh_mean: float, hh_ci: float):
    """Draw Human baseline as horizontal dashed line with CI band."""
    if np.isnan(hh_mean):
        return
    ax.axhline(hh_mean, color=HH_COLOR, lw=1.6, ls='--', zorder=1,
               label='Human baseline')
    if hh_ci > 0:
        ax.axhspan(hh_mean - hh_ci, hh_mean + hh_ci,
                   color=HH_COLOR, alpha=0.08, zorder=0)


def _ensure_hh_visible(ax, hh_mean: float):
    """Call after all data is plotted to guarantee the HH line stays in view."""
    if np.isnan(hh_mean):
        return
    ylo, yhi = ax.get_ylim()
    pad = (yhi - ylo) * 0.08
    if hh_mean > yhi - pad:
        ax.set_ylim(ylo, hh_mean + pad)
    elif hh_mean < ylo + pad:
        ax.set_ylim(hh_mean - pad, yhi)


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  [scale_{METRIC}_{AGG}] {name}')


# ─────────────────────────────────────────────────────────────────────────────
# ── by_models ─────────────────────────────────────────────────────────────────
# Lines connect same-family models (sorted by size); colour = family.
# Marker shape encodes base group; hollow = think / backbone derivative.
#   VLM              → filled 'o'     VLM backbone decoder → hollow 'o' + dotted
#   standalone LLM   → filled 's'     standalone LLM (think) → hollow 's' + dotted
# ─────────────────────────────────────────────────────────────────────────────

# Base marker per logical role (think/backbone share the filled group's marker)
_BASE_MARKER = {
    'VLM':                    'o',
    'VLM backbone decoder':   'o',   # same shape as VLM, hollow
    'standalone LLM':         's',
    'standalone LLM (think)': 's',   # same shape as LLM, hollow
}

def _plot_by_models(ax, stats: pd.DataFrame, ylabel: str, hh_mean: float, hh_ci: float):
    _draw_hh(ax, hh_mean, hh_ci)

    # Draw connecting lines per (family × base group), sorted by size
    for fam, fdf in stats.groupby('family'):
        color = MODEL_FAMILY_COLORS.get(fam, '#888888')
        # Solid line through base (filled) models
        for base_grp, _ in GROUP_PAIR_ORDER:
            sub = fdf[fdf['group'] == base_grp].sort_values('size')
            if len(sub) > 1:
                ax.plot(sub['size'].values, sub['mean'].values,
                        color=color, lw=1.2, ls='-', alpha=0.4, zorder=2)
        # Dotted line through derivative (hollow) models
        for _, deriv_grp in GROUP_PAIR_ORDER:
            sub = fdf[fdf['group'] == deriv_grp].sort_values('size')
            if len(sub) > 1:
                ax.plot(sub['size'].values, sub['mean'].values,
                        color=color, lw=1.2, ls=':', alpha=0.4, zorder=2)
        # Vertical dotted connector between base↔derivative at same size
        for base_grp, deriv_grp in GROUP_PAIR_ORDER:
            base_rows  = fdf[fdf['group'] == base_grp]
            deriv_rows = fdf[fdf['group'] == deriv_grp]
            for size in base_rows['size'].values:
                b = base_rows[base_rows['size'] == size]
                d = deriv_rows[deriv_rows['size'] == size]
                if b.empty or d.empty:
                    continue
                ax.plot([size, size],
                        [b['mean'].values[0], d['mean'].values[0]],
                        color=color, lw=1.0, ls=':', alpha=0.55, zorder=2)

    # Draw individual markers + error bars on top
    for _, row in stats.iterrows():
        color  = MODEL_FAMILY_COLORS.get(row['family'], '#888888')
        mkr    = _BASE_MARKER.get(row['group'], 'o')
        hollow = GROUP_HOLLOW.get(row['group'], False)
        mfc    = 'none' if hollow else color
        mec    = color
        ax.errorbar(row['size'], row['mean'], yerr=row['ci'],
                    fmt=mkr, color=color, ms=8, capsize=3, capthick=0.8,
                    elinewidth=0.8, alpha=0.88, zorder=3,
                    markerfacecolor=mfc, markeredgecolor=mec, markeredgewidth=1.2)

    ax.set_ylabel(ylabel, fontsize=10)
    _configure_xaxis(ax)
    _ensure_hh_visible(ax, hh_mean)

    present_fams = sorted(stats['family'].unique())
    fam_handles = [
        mlines.Line2D([], [], color=MODEL_FAMILY_COLORS.get(f, '#888'),
                      marker='o', ms=6, ls='-', lw=1.5, label=f)
        for f in present_fams
    ]
    present_grps = stats['group'].unique()
    grp_handles = []
    for g in GROUP_ORDER:
        if g not in present_grps:
            continue
        mkr    = _BASE_MARKER.get(g, 'o')
        hollow = GROUP_HOLLOW.get(g, False)
        label  = (g.replace('standalone LLM (think)', 'SA-LLM (think)')
                   .replace('standalone LLM', 'SA-LLM')
                   .replace('VLM backbone decoder', 'Backbone'))
        grp_handles.append(mlines.Line2D(
            [], [], color='gray', marker=mkr, ms=7, ls='none',
            markerfacecolor='none' if hollow else 'gray',
            markeredgecolor='gray', markeredgewidth=1.2,
            label=label,
        ))
    hh_handle = mlines.Line2D([], [], color=HH_COLOR, ls='--', lw=1.6,
                               label='Human baseline')
    ax.legend(handles=fam_handles + grp_handles + [hh_handle],
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
        color  = GROUP_COLORS.get(grp, '#888888')
        mkr    = GROUP_MARKER.get(grp, 'o')
        hollow = GROUP_HOLLOW.get(grp, False)
        mfc    = 'none' if hollow else color
        mec    = color if hollow else 'white'
        mew    = 1.2 if hollow else 0.5
        label  = grp.replace('standalone LLM', 'SA-LLM') \
                    .replace('VLM backbone decoder', 'Backbone')

        ax.errorbar(gdf['size'].values, gdf['mean'].values, yerr=gdf['ci'].values,
                    fmt=mkr, color=color, ms=7, capsize=3, capthick=0.8,
                    elinewidth=0.8, alpha=0.85, zorder=3,
                    markerfacecolor=mfc, markeredgecolor=mec,
                    markeredgewidth=mew, label=label)

        # Log-linear trend line if ≥3 distinct sizes
        xs = gdf['size'].values
        ys = gdf['mean'].values
        if len(np.unique(xs)) >= 3:
            coeffs = np.polyfit(np.log(xs), ys, 1)
            x_rng  = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 50)
            y_fit  = np.polyval(coeffs, np.log(x_rng))
            trend_ls = ':' if hollow else '--'
            ax.plot(x_rng, y_fit, color=color, ls=trend_ls, lw=1.2,
                    alpha=0.45, zorder=2)

    hh_handle = mlines.Line2D([], [], color=HH_COLOR, ls='--', lw=1.6,
                               label='Human baseline')
    ax.set_ylabel(ylabel, fontsize=10)
    _configure_xaxis(ax)
    _ensure_hh_visible(ax, hh_mean)
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l != 'Human baseline']
    filt_handles = [h for h, _ in filtered]
    filt_labels = [l for _, l in filtered]
    ax.legend(handles=filt_handles + [hh_handle],
              labels=filt_labels + ['Human baseline'],
              fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── by_group ──────────────────────────────────────────────────────────────────
# Generates 3 separate figures (one per group: vlm / backbone / llm).
# Each uses _plot_by_models logic filtered to that group's models.
# think and no-think LLMs are merged into one 'llm' figure.
# ─────────────────────────────────────────────────────────────────────────────

_GROUP_SUBSETS = [
    ('vlm_backbone', 'VLM + Backbone', ['VLM', 'VLM backbone decoder']),
    ('llm',          'LLM',            ['standalone LLM', 'standalone LLM (think)']),
]


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
_PLOT_FN = {
    'by_models': _plot_by_models,
    'by_groups': _plot_by_groups,
}
plot_fn = _PLOT_FN.get(AGG)  # None for 'by_group' (handled separately)


# ─────────────────────────────────────────────────────────────────────────────
# Generate figures
# ─────────────────────────────────────────────────────────────────────────────

if AGG == 'by_group':
    # One figure per group subset; reuses _plot_by_models logic on filtered stats.
    if METRIC == 'agreement':
        for key, (metric_name, col) in AGREEMENT_METRICS.items():
            print(f'\n── {metric_name} ──')
            stats_all = model_stats(col, variant='C')
            hh_mean, hh_ci = human_baseline(col, variant='C')
            for slug, title, grps in _GROUP_SUBSETS:
                sub = stats_all[stats_all['group'].isin(grps)]
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 5))
                _plot_by_models(ax, sub, f'Mean {metric_name}', hh_mean, hh_ci)
                ax.set_title(f'{metric_name} — {title}', fontsize=12)
                plt.tight_layout()
            _save(fig, f'inst_blind_agreement_models_{key}_vC_{slug}{SUFFIX}.png')
    else:
        var = 'C'
        print(f'\n── Accuracy variant {var} ──')
        stats_all = model_stats('accuracy', variant=var)
        hh_mean, hh_ci = human_baseline('accuracy', variant=var)
        for slug, title, grps in _GROUP_SUBSETS:
            sub = stats_all[stats_all['group'].isin(grps)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            _plot_by_models(ax, sub, 'Mean accuracy', hh_mean, hh_ci)
            ax.set_title(f'Accuracy — {title}', fontsize=12)
            plt.tight_layout()
            _save(fig, f'inst_blind_accuracy_models_v{var}_{slug}{SUFFIX}.png')

elif METRIC == 'agreement':
    for key, (metric_name, col) in AGREEMENT_METRICS.items():
        print(f'\n── {metric_name} ──')
        stats = model_stats(col, variant='C')
        if stats.empty:
            print('  (no data, skipped)')
            continue
        hh_mean, hh_ci = human_baseline(col, variant='C')

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_fn(ax, stats, f'Mean {metric_name}', hh_mean, hh_ci)
        plt.tight_layout()
        _save(fig, f'inst_blind_agreement_{AGG_SHORT}_{key}_vC{SUFFIX}.png')

else:  # accuracy — variant C only
    var = 'C'
    print(f'\n── Accuracy variant {var} ──')
    stats = model_stats('accuracy', variant=var)
    if not stats.empty:
        hh_mean, hh_ci = human_baseline('accuracy', variant=var)
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_fn(ax, stats, 'Mean accuracy', hh_mean, hh_ci)
        plt.tight_layout()
        _save(fig, f'inst_blind_accuracy_{AGG_SHORT}_v{var}{SUFFIX}.png')

print(f'\nDone. Saved to: {OUT_DIR}')
print(f'Suffix: {SUFFIX}')
