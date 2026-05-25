"""
Accuracy heatmaps: per-model accuracy broken down by operation type and entity group.

Generates a grid of heatmaps (rows = conditions, columns = op-type / entity-group)
plus a delta row (inst_blind − blind).

Function
--------
plot_group_heatmaps(op_df, ent_df, out_folder, filename='group_heatmaps.png',
                    conditions=None, op_show=None, ent_show=None)

Parameters
----------
op_df : DataFrame with columns [condition, model, op, acc, n]
ent_df : DataFrame with columns [condition, model, ent, acc, n]
out_folder : directory to save the figure
filename : output filename (default: 'group_heatmaps.png')
conditions : list of condition strings to show as rows (default: ['original','blind','inst_blind'])
op_show : subset of op-type groups to display (columns); default: 7 canonical types
ent_show : subset of entity groups to display (columns); default: 9 canonical types
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

from utils.constants import COND_LABEL

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

_DEFAULT_OP_SHOW  = ['attr', 'ident', 'spat', 'act', 'count', 'cause', 'comp']
_DEFAULT_ENT_SHOW = ['object', 'person', 'animal', 'food', 'place', 'vehicle',
                     'text', 'product', 'other']
_DEFAULT_CONDITIONS = ['original', 'blind', 'inst_blind']


def _short_model(m: str) -> str:
    return (m.replace('-hf', '').replace('Instruct', '')
             .replace('Qwen3-VL-', 'Q3VL-'))


def _draw_heatmap(ax, df, group_col, groups_show, cond, title, n_dict=None,
                  cmap='RdYlGn', vmin=0.1, vmax=0.7):
    sub   = df[df['condition'] == cond]
    pivot = sub.pivot_table(index='model', columns=group_col, values='acc')
    cols  = [g for g in groups_show if g in pivot.columns]
    pivot = pivot[cols]
    labels = [_short_model(m) for m in pivot.index]

    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    xlabels = [f'{c}\n(n={n_dict[c]})' if n_dict and c in n_dict else c for c in cols]
    ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    for i in range(len(pivot.index)):
        for j in range(len(cols)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='black' if 0.2 < v < 0.55 else 'white')
    return im


def plot_group_heatmaps(
    op_df: pd.DataFrame,
    ent_df: pd.DataFrame,
    out_folder,
    filename: str = 'group_heatmaps.png',
    conditions: list | None = None,
    op_show: list | None = None,
    ent_show: list | None = None,
) -> None:
    """Save accuracy heatmaps (op-type × entity-group) across conditions.

    Parameters
    ----------
    op_df : DataFrame [condition, model, op, acc, n]
    ent_df : DataFrame [condition, model, ent, acc, n]
    out_folder : directory to save the figure
    filename : output filename
    conditions : condition rows (default: original / blind / inst_blind)
    op_show : op-type columns to display
    ent_show : entity-group columns to display
    """
    if conditions is None:
        conditions = _DEFAULT_CONDITIONS
    if op_show is None:
        op_show = _DEFAULT_OP_SHOW
    if ent_show is None:
        ent_show = _DEFAULT_ENT_SHOW

    op_n  = (op_df[op_df['condition'] == 'blind']
             .groupby('op')['n'].first().to_dict())
    ent_n = (ent_df[ent_df['condition'] == 'blind']
             .groupby('ent')['n'].first().to_dict())

    n_rows = len(conditions) + 1  # +1 for delta row
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 6 * n_rows))

    for row_i, cond in enumerate(conditions):
        label = COND_LABEL.get(cond, cond)
        im1 = _draw_heatmap(axes[row_i][0], op_df, 'op', op_show, cond,
                             f'Op-Type — {label}', n_dict=op_n)
        plt.colorbar(im1, ax=axes[row_i][0], shrink=0.8)
        im2 = _draw_heatmap(axes[row_i][1], ent_df, 'ent', ent_show, cond,
                             f'Entity Group — {label}', n_dict=ent_n)
        plt.colorbar(im2, ax=axes[row_i][1], shrink=0.8)

    # Delta row: inst_blind − blind
    for col_i, (df_grp, grp_col, groups_show, n_dict) in enumerate([
            (op_df, 'op', op_show, op_n), (ent_df, 'ent', ent_show, ent_n)]):
        b = df_grp[df_grp['condition'] == 'blind'].pivot_table(
            index='model', columns=grp_col, values='acc')
        i = df_grp[df_grp['condition'] == 'inst_blind'].pivot_table(
            index='model', columns=grp_col, values='acc')
        d = (i - b)[[c for c in groups_show if c in b.columns]]
        labels = [_short_model(m) for m in d.index]

        im = axes[n_rows - 1][col_i].imshow(
            d.values, aspect='auto', cmap='RdBu', vmin=-0.15, vmax=0.15)
        axes[n_rows - 1][col_i].set_xticks(range(len(d.columns)))
        xlabels_d = [f'{c}\n(n={n_dict[c]})' if c in n_dict else c for c in d.columns]
        axes[n_rows - 1][col_i].set_xticklabels(xlabels_d, rotation=35, ha='right', fontsize=9)
        axes[n_rows - 1][col_i].set_yticks(range(len(labels)))
        axes[n_rows - 1][col_i].set_yticklabels(labels, fontsize=8)
        label_text = 'Op-Type' if col_i == 0 else 'Entity Group'
        axes[n_rows - 1][col_i].set_title(
            f'{label_text} Delta (inst_blind − blind)', fontsize=10, fontweight='bold')
        for ii in range(len(d.index)):
            for jj in range(len(d.columns)):
                v = d.values[ii, jj]
                if not np.isnan(v):
                    axes[n_rows - 1][col_i].text(
                        jj, ii, f'{v:+.2f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=axes[n_rows - 1][col_i], shrink=0.8)

    plt.suptitle(
        'Accuracy Heatmaps: Op-Type & Entity Group\n'
        'All 1000 Qs (pretrained VLMs) — '
        + ' / '.join(conditions) + ' / delta',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    out = Path(out_folder) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)
