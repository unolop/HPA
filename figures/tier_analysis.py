"""
Tier-level comparison figures: VLM vs VLM backbone decoder vs standalone LLM.

Each function receives pre-built DataFrames (or tier_map + dirs) and saves a
PNG to the specified out_folder, printing the saved path.

Figures
-------
plot_tier_accuracy(df, out_folder, suffix='')
    Bar + scatter: per-model accuracy per tier, split by condition.

plot_tier_degradation(tier_map, out_folder, suffix='', conditions=None)
    Degradation ladder (C→B→A) per tier, mean ± std across models.

plot_tier_inst_effect(tier_map, out_folder, suffix='')
    Δacc (inst_blind − blind) per model (bar) and blind vs inst scatter.

plot_tier_abstention(df, out_folder, suffix='')
    Soft abstention rate per tier × condition grouped bar.

plot_tier_paired(tier_map, out_folder, suffix='', triples=None, conditions=None)
    Side-by-side accuracy for matched (VLM, LM-decoder, backbone) triples.

plot_all(df, tier_map, out_folder, suffix='', triples=None, conditions=None)
    Convenience wrapper that calls all five functions above.

Run from repo root (requires a pre-built `df` and `tier_map` from the notebook):
    This module is intended to be imported from analysis/session2/05_prior_lm_decoder.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from utils.constants import (
    ABSTAIN_TOKENS,
    CONDITIONS,
    CONTROL_TYPES,
    CT_LABELS,
    TIER_COLORS,
    TIER_ORDER,
    TIER_STYLE,
    TRIPLES,
)
from utils.load_session import clean_answer
from utils.vqa import VQAAnswerMapper, vqa_accuracy

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

# ── Internal data helpers ─────────────────────────────────────────────────────

_mapper: VQAAnswerMapper | None = None


def _get_mapper() -> VQAAnswerMapper:
    global _mapper
    if _mapper is None:
        _mapper = VQAAnswerMapper()
    return _mapper


def _load(path) -> list | None:
    p = Path(path)
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _score(rows, ct: str = 'question') -> float:
    if not rows:
        return np.nan
    mapper = _get_mapper()
    return float(np.mean([
        vqa_accuracy(
            clean_answer(r['generated_answers'].get(ct, '')),
            mapper.get_answers(r['question_id']),
        )
        for r in rows
    ]))


def _abstain_rate(rows, ct: str = 'question') -> float:
    if not rows:
        return np.nan
    return float(np.mean([
        any(tok in clean_answer(r['generated_answers'].get(ct, '')).lower()
            for tok in ABSTAIN_TOKENS)
        for r in rows
    ]))


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_tier_accuracy(df: pd.DataFrame, out_folder, suffix: str = '') -> None:
    """Bar + jitter scatter: per-model accuracy per tier, one panel per condition.

    Parameters
    ----------
    df : DataFrame with columns [tier, model, condition, acc]
    out_folder : directory to save figure
    suffix : appended to filename before .png (e.g. '_q88')
    """
    conditions = df['condition'].unique().tolist()
    fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond_label in zip(axes, conditions):
        sub = df[df.condition == cond_label]
        for tier in TIER_ORDER:
            if tier not in TIER_COLORS:
                continue
            t = sub[sub.tier == tier]
            if t.empty:
                continue
            x = TIER_ORDER.index(tier)
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(t))
            ax.scatter([x + j for j in jitter], t['acc'],
                       color=TIER_COLORS[tier], alpha=0.55, s=40, zorder=3)
            m, s_ = t['acc'].mean(), t['acc'].std()
            ax.bar(x, m, 0.5, color=TIER_COLORS[tier], alpha=0.25,
                   edgecolor=TIER_COLORS[tier], linewidth=1.5, zorder=2)
            ax.errorbar(x, m, yerr=s_, fmt='none',
                        color=TIER_COLORS[tier], capsize=5, lw=2, zorder=4)
            ax.plot(x, m, marker='D', markersize=9, color=TIER_COLORS[tier],
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5)
            ax.text(x, m + s_ + 0.02, f'{m:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        ax.set_xticks(range(len(TIER_ORDER)))
        ax.set_xticklabels(TIER_ORDER, fontsize=10)
        ax.set_ylabel('Accuracy (VQA)', fontsize=9)
        ax.set_ylim(0, 0.75)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(f'Condition: {cond_label}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Accuracy by Tier — blind & inst_blind\n'
                 '(bars = mean ± std,  ◆ = mean,  dots = individual models)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, Path(out_folder) / f'tier_accuracy{suffix}.png')
    plt.close(fig)


def plot_tier_degradation(
    tier_map: list,
    out_folder,
    suffix: str = '',
    conditions: list | None = None,
) -> None:
    """Degradation ladder (C→B→A) per tier, mean ± std across models.

    Parameters
    ----------
    tier_map : [(tier_name, base_dir, model_names), ...]
    out_folder : directory to save figure
    suffix : filename suffix
    conditions : list of (cond_suffix, cond_label, cond_color); defaults to constants.CONDITIONS
    """
    if conditions is None:
        conditions = CONDITIONS

    fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, (cond_suffix, cond_label, _) in zip(axes, conditions):
        for tier, tdir, models in tier_map:
            st = TIER_STYLE.get(tier, {})
            if not st:
                continue
            all_accs = []
            for model in models:
                rows_ = _load(Path(tdir) / model / f'vqa_1k{cond_suffix}.jsonl')
                if not rows_:
                    continue
                all_accs.append([_score(rows_, ct) for ct in CONTROL_TYPES])
            if not all_accs:
                continue
            mean_accs = np.nanmean(all_accs, axis=0)
            std_accs  = np.nanstd(all_accs, axis=0)
            x = range(len(CONTROL_TYPES))
            ax.plot(x, mean_accs, color=st['color'], ls=st['ls'], lw=st['lw'],
                    marker=st['marker'], markersize=6,
                    label=f"{tier} (n={len(all_accs)})", zorder=4)
            ax.fill_between(x, mean_accs - std_accs, mean_accs + std_accs,
                            color=st['color'], alpha=0.10)

        ax.set_xticks(range(len(CONTROL_TYPES)))
        ax.set_xticklabels(CT_LABELS, fontsize=9)
        ax.set_ylabel('Mean Accuracy', fontsize=9)
        ax.set_ylim(0, 0.7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(f'{cond_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Degradation Ladder by Tier (mean ± std across models)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, Path(out_folder) / f'tier_degradation{suffix}.png')
    plt.close(fig)


def plot_tier_inst_effect(tier_map: list, out_folder, suffix: str = '') -> None:
    """Δacc (inst_blind − blind) per model: horizontal bar + blind-vs-inst scatter.

    Parameters
    ----------
    tier_map : [(tier_name, base_dir, model_names), ...]
    out_folder : directory to save figure
    suffix : filename suffix
    """
    delta_rows = []
    for tier, tdir, models in tier_map:
        for model in models:
            b = _load(Path(tdir) / model / 'vqa_1k_control_blind.jsonl')
            i = _load(Path(tdir) / model / 'vqa_1k_control_inst_blind.jsonl')
            if b is None or i is None:
                continue
            delta_rows.append({
                'tier': tier, 'model': model,
                'delta': _score(i) - _score(b),
                'blind': _score(b), 'inst': _score(i),
            })

    if not delta_rows:
        print('plot_tier_inst_effect: no data found — skipping')
        return

    ddf = pd.DataFrame(delta_rows).sort_values(['tier', 'delta'])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: delta bar per model
    ax = axes[0]
    colors_ = [TIER_COLORS.get(r['tier'], '#888') for _, r in ddf.iterrows()]
    ax.barh(range(len(ddf)), ddf['delta'], color=colors_, alpha=0.75, edgecolor='white')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(range(len(ddf)))
    ax.set_yticklabels(
        [f"[{r['tier'][0]}] {r['model']}" for _, r in ddf.iterrows()], fontsize=7
    )
    ax.set_xlabel('Δ Accuracy (inst_blind − blind)', fontsize=9)
    ax.set_title('Instruction Effect per Model', fontsize=11, fontweight='bold')
    ax.legend(
        handles=[mpatches.Patch(color=TIER_COLORS.get(t, '#888'), label=t) for t in TIER_ORDER],
        fontsize=8,
    )
    ax.grid(axis='x', alpha=0.3)

    # Right: blind vs inst_blind scatter
    ax = axes[1]
    for tier in TIER_ORDER:
        t = ddf[ddf.tier == tier]
        ax.scatter(t['blind'], t['inst'], color=TIER_COLORS.get(tier, '#888'),
                   s=60, alpha=0.8, label=tier,
                   edgecolors='white', linewidths=0.5, zorder=3)
        for _, r in t.iterrows():
            ax.annotate(r['model'].split('-')[0], (r['blind'], r['inst']),
                        fontsize=5, alpha=0.6, xytext=(2, 2),
                        textcoords='offset points')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='no change')
    ax.set_xlabel('blind accuracy', fontsize=9)
    ax.set_ylabel('inst_blind accuracy', fontsize=9)
    ax.set_xlim(0.1, 0.65); ax.set_ylim(0.1, 0.65)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title('Blind vs Inst_Blind per Model', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle('Instruction Effect (Blind → Inst_Blind) by Tier',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, Path(out_folder) / f'tier_inst_effect{suffix}.png')
    plt.close(fig)

    print('\nMean Δacc by tier:')
    print(ddf.groupby('tier')['delta'].agg(['mean', 'std', 'count']).round(3))


def plot_tier_abstention(df: pd.DataFrame, out_folder, suffix: str = '') -> None:
    """Grouped bar: soft abstention rate per tier × condition.

    Parameters
    ----------
    df : DataFrame with columns [tier, condition, abstain]
    out_folder : directory to save figure
    suffix : filename suffix
    """
    cond_colors = {'blind': '#e74c3c', 'inst_blind': '#3498db'}
    conditions = df['condition'].unique().tolist()

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(TIER_ORDER))
    w = 0.35

    for ci, cond_label in enumerate(conditions):
        means, stds = [], []
        for tier in TIER_ORDER:
            t = df[(df.tier == tier) & (df.condition == cond_label)]
            means.append(t['abstain'].mean() if not t.empty else 0.0)
            stds.append(t['abstain'].std() if not t.empty else 0.0)
        offset = (ci - (len(conditions) - 1) / 2) * w
        ax.bar(x + offset, means, w, label=cond_label,
               color=cond_colors.get(cond_label, '#888'), alpha=0.75, edgecolor='white')
        ax.errorbar(x + offset, means, yerr=stds, fmt='none',
                    color=cond_colors.get(cond_label, '#888'), capsize=4, lw=1.5)
        for xi, (m, s_) in zip(x + offset, zip(means, stds)):
            ax.text(xi, m + s_ + 0.003, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(TIER_ORDER, fontsize=10)
    ax.set_ylabel('Soft abstention rate', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title('Soft Abstention by Tier & Condition', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, Path(out_folder) / f'tier_abstention{suffix}.png')
    plt.close(fig)


def plot_tier_paired(
    tier_map: list,
    out_folder,
    suffix: str = '',
    triples: list | None = None,
    conditions: list | None = None,
) -> None:
    """Side-by-side accuracy for matched (VLM, LM-decoder, backbone) triples.

    Parameters
    ----------
    tier_map : [(tier_name, base_dir, model_names), ...]
    out_folder : directory to save figure
    suffix : filename suffix
    triples : list of (vlm_model, lm_model, bb_model, label); defaults to constants.TRIPLES
    conditions : list of (cond_suffix, cond_label, cond_color); defaults to constants.CONDITIONS
    """
    if triples is None:
        triples = TRIPLES
    if conditions is None:
        conditions = CONDITIONS

    tier_dirs = {tier: Path(tdir) for tier, tdir, _ in tier_map}
    tier_keys = {'VLM': 0, 'VLM backbone decoder': 1, 'standalone LLM': 2}

    fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, (_, cond_label, _) in zip(axes, conditions):
        cond_suffix = '_control_blind' if cond_label == 'blind' else '_control_inst_blind'
        n = len(triples)
        x = np.arange(n)
        w = 0.22
        tier_offsets = {'VLM': -w, 'VLM backbone decoder': 0, 'standalone LLM': w}

        for tier, offset in tier_offsets.items():
            if tier not in tier_dirs:
                continue
            accs = []
            for vlm_m, lm_m, bb_m, label in triples:
                model = [vlm_m, lm_m, bb_m][tier_keys[tier]]
                rows_ = _load(tier_dirs[tier] / model / f'vqa_1k{cond_suffix}.jsonl')
                accs.append(_score(rows_) if rows_ else np.nan)
            ax.bar(x + offset, accs, w, label=tier,
                   color=TIER_COLORS.get(tier, '#888'), alpha=0.75, edgecolor='white')
            for xi, a in zip(x + offset, accs):
                if not np.isnan(a):
                    ax.text(xi, a + 0.005, f'{a:.2f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([t[3] for t in triples], fontsize=9)
        ax.set_ylabel('Accuracy', fontsize=9)
        ax.set_ylim(0, 0.7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(f'{cond_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Paired Comparison: Same Language Model Family Across 3 Tiers',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, Path(out_folder) / f'tier_paired{suffix}.png')
    plt.close(fig)


def plot_all(
    df: pd.DataFrame,
    tier_map: list,
    out_folder,
    suffix: str = '',
    triples: list | None = None,
    conditions: list | None = None,
) -> None:
    """Generate all five tier-comparison figures.

    Parameters
    ----------
    df : per-model per-condition summary DataFrame (columns: tier, model, condition, acc, abstain)
    tier_map : [(tier_name, base_dir, model_names), ...]
    out_folder : directory to save all figures
    suffix : filename suffix appended before .png
    triples : list of (vlm_model, lm_model, bb_model, label)
    conditions : list of (cond_suffix, cond_label, cond_color)
    """
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    figures = [
        ('tier_accuracy',    lambda: plot_tier_accuracy(df, out_folder, suffix)),
        ('tier_degradation', lambda: plot_tier_degradation(tier_map, out_folder, suffix, conditions)),
        ('tier_inst_effect', lambda: plot_tier_inst_effect(tier_map, out_folder, suffix)),
        ('tier_abstention',  lambda: plot_tier_abstention(df, out_folder, suffix)),
        ('tier_paired',      lambda: plot_tier_paired(tier_map, out_folder, suffix, triples, conditions)),
    ]

    for name, fn in figures:
        fn()
