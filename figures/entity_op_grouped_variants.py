"""
Grouped entity/op HM-SBERT plots with variant-spread error bars (A/B/C)
and HH baseline overlays.

Outputs
-------
- figures/supplementary/sbert_gap_7b/hm_sbert_entity_op_grouped_variants_matched_7b.png
- figures/supplementary/sbert_gap_7b/hm_sbert_entity_op_grouped_variants_all_models.png
- figures/supplementary/sbert_gap_7b/hm_sbert_entity_op_grouped_variants_family_*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from config import MODELS_7B, FAMILY_SUBSETS
from utils.constants import GROUP_COLORS, GROUP_ORDER, VARIANT_ORDER, MODEL_FAMILY, MODEL_SIZE_B
from helpers import load_human_subset

OUT_DIR = ROOT / 'figures/supplementary/sbert_gap_7b'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ANS = 348
ENT_MIN_Q = 10
OP_MIN_Q = 7
GROUPS = GROUP_ORDER  # include think group
EXCLUDED_MATCHED_MODELS = {'Mistral-7B', 'Phi-3.5-mini'}

# Family marker mapping (shared with other variant-plot scripts)
FAMILY_MARKER = {
    'Qwen3-VL':      '^',
    'Qwen3':         '^',
    'LLaVA-1.5':     'o',
    'LLaVA-Mistral': 'D',
    'LLaVA-Vicuna':  'P',
    'InternVL':      'X',
    'Mistral':       'v',
    'Vicuna':        'h',
    'Phi':           '*',
    'Qwen2.5':       'p',
}


def _model_point_area(model: str, reference_models: list[str] | None) -> float:
    """
    Marker area (points^2) scaled by model parameter count.
    Uses log scaling to keep large-model points visible without dominating.
    """
    lo_area, hi_area = 18.0, 64.0

    if reference_models:
        ref_sizes = [MODEL_SIZE_B[m] for m in reference_models if m in MODEL_SIZE_B]
    else:
        ref_sizes = list(MODEL_SIZE_B.values())
    if not ref_sizes:
        return 26.0

    mn, mx = min(ref_sizes), max(ref_sizes)
    size = MODEL_SIZE_B.get(model, np.median(ref_sizes))
    if mn <= 0 or mx <= 0 or abs(mx - mn) < 1e-12:
        frac = 1.0
    else:
        frac = (np.log(size) - np.log(mn)) / (np.log(mx) - np.log(mn))
        frac = float(np.clip(frac, 0.0, 1.0))
    return lo_area + frac * (hi_area - lo_area)


def _model_status(model: str) -> str:
    """Explicit status label used for visual separation in points/legend."""
    return 'inst' if 'Instruct' in str(model) else 'base'


def _family_display_name(fam: str) -> str:
    if fam in {'Qwen3-VL', 'Qwen3'}:
        return 'Qwen3-VL / Qwen3'
    return fam


def _family_display_marker(fam: str) -> str:
    if fam in {'Qwen3-VL', 'Qwen3'}:
        return '^'
    return FAMILY_MARKER.get(fam, 'o')


def _load():
    exports = ROOT / 'analysis/session2/exports'
    pair = pd.read_parquet(exports / 'pair_cache.parquet')
    resp = pd.read_csv(exports / 'responses_model_inst_blind.csv')
    _, common_qids, _, _ = load_human_subset(ROOT, min_answers=MIN_ANS, translate=False, verbose=False)

    pair = pair[(pair['variant'].isin(VARIANT_ORDER)) & (pair['question_id'].isin(common_qids))].copy()
    meta = resp[(resp['variant'].isin(VARIANT_ORDER)) & (resp['question_id'].isin(common_qids))][
        ['question_id', 'variant', 'op', 'ent']
    ].drop_duplicates()
    return pair, meta


def _group_maps(meta_c: pd.DataFrame):
    ent_counts = meta_c.groupby('ent')['question_id'].nunique().sort_values(ascending=False)
    op_counts = meta_c.groupby('op')['question_id'].nunique().sort_values(ascending=False)

    ent_map = {e: (e if n >= ENT_MIN_Q else 'other_entities') for e, n in ent_counts.items()}
    op_map = {o: (o if n >= OP_MIN_Q else 'other_ops') for o, n in op_counts.items()}

    ent_order = [e for e, n in ent_counts.items() if n >= ENT_MIN_Q]
    if any(n < ENT_MIN_Q for n in ent_counts.values):
        ent_order += ['other_entities']

    op_order = [o for o, n in op_counts.items() if n >= OP_MIN_Q]
    if any(n < OP_MIN_Q for n in op_counts.values):
        op_order += ['other_ops']

    ent_n = meta_c.assign(ent_grp=meta_c['ent'].map(ent_map)).groupby('ent_grp')['question_id'].nunique()
    op_n = meta_c.assign(op_grp=meta_c['op'].map(op_map)).groupby('op_grp')['question_id'].nunique()

    return ent_map, op_map, ent_order, op_order, ent_n, op_n


def _summaries(pair: pd.DataFrame, meta: pd.DataFrame, subset_models: set[str] | None):
    hm = pair[pair['pair_type'] == 'HM'].copy()
    hh = pair[pair['pair_type'] == 'HH'].copy()
    if subset_models is not None:
        hm = hm[hm['subject_2'].isin(subset_models)].copy()

    # pair already carries ent/op in most exports; merge metadata only as fallback
    # and avoid ent_x/ent_y style suffix collisions.
    meta_min = meta[['question_id', 'variant', 'ent', 'op']].drop_duplicates()
    meta_min = meta_min.rename(columns={'ent': 'ent_meta', 'op': 'op_meta'})
    hm = hm.merge(meta_min, on=['question_id', 'variant'], how='left')
    hh = hh.merge(meta_min, on=['question_id', 'variant'], how='left')
    for df in (hm, hh):
        if 'ent' not in df.columns:
            df['ent'] = df['ent_meta']
        else:
            df['ent'] = df['ent'].fillna(df['ent_meta'])
        if 'op' not in df.columns:
            df['op'] = df['op_meta']
        else:
            df['op'] = df['op'].fillna(df['op_meta'])
        df.drop(columns=['ent_meta', 'op_meta'], inplace=True)
    return hm, hh


def _category_stats(hm: pd.DataFrame, hh: pd.DataFrame, category: str, order: list[str]):
    # HM: average across human raters -> per question/model/category
    hm_q = (hm.groupby(['variant', 'subject_2', 'subject_group_2', 'question_id', category], dropna=False)['sbert_score']
              .mean().reset_index(name='sbert_qm'))

    # per-model per-variant per-category
    model_var = (hm_q.groupby(['variant', 'subject_2', 'subject_group_2', category], dropna=False)['sbert_qm']
                   .mean().reset_index(name='sbert'))

    # group mean per variant/category (across models)
    grp_var = (model_var.groupby(['variant', 'subject_group_2', category], dropna=False)['sbert']
                 .mean().reset_index(name='sbert'))

    grp = (grp_var.groupby(['subject_group_2', category], dropna=False)['sbert']
             .agg(center='mean', vmin='min', vmax='max').reset_index())
    grp['err_low'] = grp['center'] - grp['vmin']
    grp['err_high'] = grp['vmax'] - grp['center']

    # per-model points (averaged across variants)
    model_pts = (model_var.groupby(['subject_2', 'subject_group_2', category], dropna=False)['sbert']
                   .mean().reset_index(name='sbert'))

    # HH baseline per category, averaged across variants
    hh_q = (hh.groupby(['variant', 'question_id', category], dropna=False)['sbert_score']
              .mean().reset_index(name='hh_q'))
    hh_var = hh_q.groupby(['variant', category], dropna=False)['hh_q'].mean().reset_index(name='hh')
    hh_cat = (hh_var.groupby([category], dropna=False)['hh']
                .agg(center='mean', vmin='min', vmax='max').reset_index())

    grp[category] = pd.Categorical(grp[category], categories=order, ordered=True)
    model_pts[category] = pd.Categorical(model_pts[category], categories=order, ordered=True)
    hh_cat[category] = pd.Categorical(hh_cat[category], categories=order, ordered=True)

    grp = grp.sort_values(category)
    model_pts = model_pts.sort_values(category)
    hh_cat = hh_cat.sort_values(category)

    return grp, model_pts, hh_cat


def _plot_main(ax, grp, pts, hh, category_col, order, n_map, title, reference_models, groups_plot, show_ylabel=True):
    x = np.arange(len(order))
    w = 0.18

    for gi, g in enumerate(groups_plot):
        sub = grp[grp['subject_group_2'] == g].set_index(category_col).reindex(order)
        y = sub['center'].to_numpy()
        ylow = sub['err_low'].to_numpy()
        yhigh = sub['err_high'].to_numpy()
        xpos = x + (gi - (len(groups_plot) - 1) / 2) * w

        ax.bar(xpos, y, width=w * 0.92, color=GROUP_COLORS[g], alpha=0.75,
               edgecolor='white', linewidth=0.6, label=g if gi == 0 else None)
        ax.errorbar(xpos, y, yerr=np.vstack([ylow, yhigh]), fmt='none',
                    ecolor=GROUP_COLORS[g], capsize=3, elinewidth=1.2, zorder=4)

        # per-model scatter (mean across variants), ordered by sbert value
        spts = pts[pts['subject_group_2'] == g]
        for i, cat in enumerate(order):
            cpts = spts[spts[category_col] == cat][['subject_2', 'sbert']].sort_values('sbert')
            if cpts.empty:
                continue
            n = len(cpts)
            jitter = np.linspace(-w * 0.22, w * 0.22, n) if n > 1 else np.array([0.0])
            for j, row in enumerate(cpts.itertuples(index=False)):
                fam = MODEL_FAMILY.get(row.subject_2, '')
                marker = FAMILY_MARKER.get(fam, 'o')
                area = _model_point_area(row.subject_2, reference_models)
                status = _model_status(row.subject_2)
                is_inst = (status == 'inst')
                ax.scatter(xpos[i] + jitter[j], row.sbert,
                           s=area, marker=marker,
                           color=GROUP_COLORS[g] if is_inst else 'white',
                           edgecolors='white' if is_inst else GROUP_COLORS[g],
                           linewidths=1.0 if is_inst else 0.9,
                           alpha=0.95, zorder=5)

    # HH dotted baseline per category
    hh_idx = hh.set_index(category_col).reindex(order)
    for i, cat in enumerate(order):
        y = hh_idx.loc[cat, 'center']
        ax.hlines(y, i - 0.47, i + 0.47, colors='black', linestyles=':', linewidth=1.2, zorder=3)

    # xticks with counts
    labels = [f"{c}\n(n={int(n_map.get(c, 0))})" for c in order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    if show_ylabel:
        ax.set_ylabel('HM SBERT cosine')
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0.25, 0.70)
    ax.grid(axis='y', alpha=0.25)


def _add_legend(fig, reference_models, groups_plot):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=GROUP_COLORS[g], lw=8, label=g)
        for g in groups_plot
    ]
    handles.append(Line2D([0], [0], color='black', lw=1.5, ls=':', label='HH baseline (mean across A/B/C)'))

    fam_order = [
        'Qwen3-VL / Qwen3', 'InternVL', 'LLaVA-1.5', 'LLaVA-Mistral',
        'LLaVA-Vicuna', 'Qwen2.5', 'Mistral', 'Vicuna', 'Phi'
    ]
    fam_status_present = {}
    for m in reference_models:
        fam = MODEL_FAMILY.get(m, '')
        if not fam:
            continue
        disp = _family_display_name(fam)
        status = _model_status(m)
        fam_status_present[(disp, status)] = _family_display_marker(fam)

    for disp in fam_order:
        for status in ('base', 'inst'):
            key = (disp, status)
            if key not in fam_status_present:
                continue
            marker = fam_status_present[key]
            label = disp if status == 'base' else f'{disp} (Instruct)'
            handles.append(
                Line2D([0], [0], marker=marker,
                       color='none',
                       markerfacecolor='#777' if status == 'inst' else 'white',
                       markeredgecolor='#555', markeredgewidth=1.0,
                       lw=0, markersize=6, label=label)
            )

    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=True, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))


def plot_entity_op_grouped(pair, meta, subset_models, suffix, drop_think=False):
    hm, hh = _summaries(pair, meta, subset_models)
    if drop_think:
        hm = hm[hm['subject_group_2'] != 'standalone LLM (think)'].copy()

    meta_c = meta[meta['variant'] == 'C'].copy()
    ent_map, op_map, ent_order, op_order, _, _ = _group_maps(meta_c)

    hm = hm.assign(ent_grp=hm['ent'].map(ent_map).fillna('other_entities'),
                   op_grp=hm['op'].map(op_map).fillna('other_ops'))
    hh = hh.assign(ent_grp=hh['ent'].map(ent_map).fillna('other_entities'),
                   op_grp=hh['op'].map(op_map).fillna('other_ops'))

    # Keep only categories with HM coverage in variant C to avoid empty bars.
    hm_c = hm[hm['variant'] == 'C'].copy()
    ent_n = hm_c.groupby('ent_grp')['question_id'].nunique()
    op_n = hm_c.groupby('op_grp')['question_id'].nunique()
    ent_order = [c for c in ent_order if int(ent_n.get(c, 0)) > 0]
    op_order = [c for c in op_order if int(op_n.get(c, 0)) > 0]

    ent_grp, ent_pts, ent_hh = _category_stats(hm, hh, 'ent_grp', ent_order)
    op_grp, op_pts, op_hh = _category_stats(hm, hh, 'op_grp', op_order)
    reference_models = sorted(hm['subject_2'].dropna().unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharey=True)
    groups_plot = [g for g in GROUPS if not (drop_think and g == 'standalone LLM (think)')]
    _plot_main(axes[0], ent_grp, ent_pts, ent_hh, 'ent_grp', ent_order, ent_n, 'Entity Groups',
               reference_models, groups_plot, show_ylabel=True)
    _plot_main(axes[1], op_grp, op_pts, op_hh, 'op_grp', op_order, op_n, 'Operation Groups',
               reference_models, groups_plot, show_ylabel=False)

    if subset_models is None:
        title_scope = 'all models'
    elif len(subset_models) <= 6:
        title_scope = ', '.join(sorted(subset_models))
    else:
        title_scope = f'{len(subset_models)} matched models'
    fig.suptitle(
        f'HM SBERT by grouped entity/op categories with variant-spread error bars ({title_scope})',
        fontsize=12
    )
    _add_legend(fig, reference_models, groups_plot)
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    out = OUT_DIR / f'hm_sbert_entity_op_grouped_variants_{suffix}.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', out)

    return ent_grp, op_grp, ent_order, op_order



def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})

    matched_models = set(MODELS_7B) - EXCLUDED_MATCHED_MODELS
    pair, meta = _load()

    # 7B matched-size comparison (drop think group)
    plot_entity_op_grouped(pair, meta, matched_models, 'matched_7b', drop_think=True)

    # All models
    plot_entity_op_grouped(pair, meta, None, 'all_models', drop_think=False)

    # Family-specific subsets from config
    for suffix, (label, model_list) in FAMILY_SUBSETS.items():
        plot_entity_op_grouped(pair, meta, set(model_list), f'family_{suffix}', drop_think=False)


if __name__ == '__main__':
    main()
