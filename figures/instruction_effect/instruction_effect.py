"""
Export instruction-effect figures to figures/instruction_effect/.

Generates four figures per dataset (full 1000-question and 113-question human subset):
  fig_soft_abstention_v{VARIANT}_q{N}.png
  fig_hard_abstention_v{VARIANT}_q{N}.png
  fig_response_change_v{VARIANT}_q{N}.png
  fig_merged_abstention_v{VARIANT}_q{N}.png   ← soft + hard offset dumbbells

VARIANT is always 'C' (original question). N indicates question count.

Run from repo root:
  conda run -n zero python figures/instruction_effect.py
"""

import json
import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from utils.constants import GROUP_COLORS, GROUP_ORDER
from utils.abstention import classify
from utils.load_session import clean_answer
from utils.model_registry import MODEL_TYPE, default_all_models
from helpers import clear_output_plots, read_response_exports

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true',
                    help='Delete existing plot files in the output folder before exporting.')
args = parser.parse_args()

OUT_DIR = ROOT / "figures/instruction_effect"
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)

TABLE_DIR = ROOT / "latex/AAAI2026/LaTeX/tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

VARIANT = 'C'

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

LABEL_MAP = {
    'Qwen3-VL-2B (LM)':   'Qwen3-VL-2B',
    'Qwen3-VL-4B (LM)':   'Qwen3-VL-4B',
    'Qwen3-VL-8B (LM)':   'Qwen3-VL-8B',
    'Qwen3-VL-32B (LM)':  'Qwen3-VL-32B',
    'LLaVA-Mistral (LM)': 'LLaVA-Mistral',
    'LLaVA-Vicuna (LM)':  'LLaVA-Vicuna',
    'LLaVA-1.5 (LM)':     'LLaVA-1.5',
    'InternVL-1B (LM)':   'InternVL-1B',
    'InternVL-2B (LM)':   'InternVL-2B',
    'InternVL-8B (LM)':   'InternVL-8B',
    'LLaVA-Mistral':      'LLaVA-Mistral',
    'LLaVA-Vicuna':       'LLaVA-Vicuna',
    'LLaVA-1.5-7B':       'LLaVA-1.5',
    'Qwen3-VL-2B':        'Qwen3-VL-2B',
    'Qwen3-VL-4B':        'Qwen3-VL-4B',
    'Qwen3-VL-8B':        'Qwen3-VL-8B',
    'InternVL-1B':        'InternVL-1B',
    'InternVL-2B':        'InternVL-2B',
    'InternVL-8B':        'InternVL-8B',
    'Qwen3-8B':           'Qwen3-8B',
    'Qwen3-8B (think)':   'Qwen3-8B',
    'Qwen3-32B':          'Qwen3-32B',
    'Qwen3-32B (think)':  'Qwen3-32B',
    'Qwen3-4B':           'Qwen3-4B',
    'Qwen3-4B (think)':   'Qwen3-4B',
    'Qwen3-1.7B':         'Qwen3-1.7B',
    'Qwen3-1.7B (think)': 'Qwen3-1.7B',
    'Qwen3-0.6B':         'Qwen3-0.6B',
    'Qwen3-0.6B (think)': 'Qwen3-0.6B',
    'Phi-3.5-mini':       'Phi-3.5',
    'Mistral-7B':         'Mistral-7B',
    'Vicuna-13B':         'Vicuna-13B',
}


def _family_key(model: str) -> str:
    base = (
        model.replace(' (LM)', '')
             .replace(' (think)', '')
             .replace('-7B', '')
    )
    if base.startswith('InternVL'):
        return 'InternVL'
    if base.startswith('LLaVA-1.5'):
        return 'LLaVA-1.5'
    if base.startswith('LLaVA-Mistral'):
        return 'LLaVA-Mistral'
    if base.startswith('LLaVA-Vicuna-13B'):
        return 'LLaVA-Vicuna-13B'
    if base.startswith('LLaVA-Vicuna'):
        return 'LLaVA-Vicuna'
    if base.startswith('Qwen3-VL'):
        return 'Qwen3-VL'
    if base.startswith('Qwen2.5'):
        return 'Qwen2.5'
    if base.startswith('Qwen3'):
        return 'Qwen3'
    if base.startswith('Mistral'):
        return 'Mistral'
    if base.startswith('Vicuna-13B'):
        return 'Vicuna-13B'
    if base.startswith('Vicuna-7B'):
        return 'Vicuna-7B'
    if base.startswith('Phi'):
        return 'Phi'
    return base


def _size_key(model: str) -> float:
    size_order = [
        ('32B', 32.0), ('13B', 13.0), ('8B', 8.0), ('7B', 7.0),
        ('4B', 4.0), ('3.8B', 3.8), ('2B', 2.0), ('1.7B', 1.7),
        ('1B', 1.0), ('0.6B', 0.6),
    ]
    for token, val in size_order:
        if token in model:
            return val
    return 999.0


def norm_ans(s):
    if pd.isna(s): return ''
    return str(s).strip().lower()


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [instruction_effect] {name}')


# ─── Load full 1000-question data from logit files ───────────────────────────
def load_raw_responses(condition: str) -> pd.DataFrame:
    """Load raw variant-C responses for all models from logit JSONL files."""
    all_models = default_all_models(ROOT)
    rows = []
    for label, (rdir, mdir) in all_models.items():
        path = rdir / mdir / f'{condition}.jsonl'
        if not path.exists():
            continue
        grp = MODEL_TYPE.get(label, 'unknown')
        for line in open(path):
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            ga = ex.get('generated_answers', {})
            raw = ga.get('question', '')
            rows.append({
                'question_id': int(ex['question_id']),
                'model':       label,
                'model_group': grp,
                'variant':     VARIANT,
                'response':    clean_answer(str(raw)),
            })
    return pd.DataFrame(rows)


# ─── Build datasets ───────────────────────────────────────────────────────────
print('Loading full 1000-question data from logit files…')
df_full_b = load_raw_responses('vqa_1k_control_blind')
df_full_i = load_raw_responses('vqa_1k_control_inst_blind')
N_FULL = df_full_b['question_id'].nunique()
print(f'  blind: {len(df_full_b)} rows, {N_FULL} questions, '
      f'{df_full_b["model"].nunique()} models')
print(f'  inst_blind: {len(df_full_i)} rows, '
      f'{df_full_i["question_id"].nunique()} questions, '
      f'{df_full_i["model"].nunique()} models')

print('\nLoading 113-question human-study subset from exports…')
exports = read_response_exports(ROOT, variant=VARIANT, check_completeness=False)
df_sub_b = exports['model_blind']
df_sub_i = exports['model_inst_blind']
# Re-clean exported model answers so abstention is measured on the recovered
# final answer rather than any leaked think trace present in older exports.
for _df in (df_sub_b, df_sub_i):
    if not _df.empty and 'response' in _df.columns:
        _df['response'] = _df['response'].fillna('').astype(str).apply(clean_answer)
N_SUB = df_sub_b['question_id'].nunique()
print(f'  blind: {len(df_sub_b)} rows, {N_SUB} questions, '
      f'{df_sub_b["model"].nunique()} models')

DATASETS = [
    (f'v{VARIANT}_q{N_FULL}', df_full_b, df_full_i),
    (f'v{VARIANT}_q{N_SUB}',  df_sub_b,  df_sub_i),
]


# ─── Stats computation ────────────────────────────────────────────────────────
def compute_stats(df_mb, df_mi, min_n=50, max_missing=None):
    df_mb = df_mb.copy()
    df_mi = df_mi.copy()
    df_mb['resp_norm'] = df_mb['response'].apply(norm_ans)
    df_mi['resp_norm'] = df_mi['response'].apply(norm_ans)
    df_mb['cls'] = df_mb['response'].apply(
        lambda x: classify(str(x) if not pd.isna(x) else '', None))
    df_mi['cls'] = df_mi['response'].apply(
        lambda x: classify(str(x) if not pd.isna(x) else '', None))

    rows = []
    expected_n = max(df_mb['question_id'].nunique(), df_mi['question_id'].nunique())
    for model in df_mb['model'].unique():
        b = df_mb[df_mb['model'] == model]
        i = df_mi[df_mi['model'] == model]
        if len(b) < min_n or len(i) < min_n:
            continue
        if max_missing is not None:
            if (expected_n - len(b['question_id'].unique()) > max_missing or
                expected_n - len(i['question_id'].unique()) > max_missing):
                continue
        grp = b['model_group'].iloc[0]

        soft_b = (b['cls'] == 'soft_abstained').mean()
        soft_i = (i['cls'] == 'soft_abstained').mean()
        hard_b = (b['cls'] == 'hard_abstained').mean()
        hard_i = (i['cls'] == 'hard_abstained').mean()

        merged = b[['question_id', 'resp_norm']].rename(
            columns={'resp_norm': 'rb'}).merge(
            i[['question_id', 'resp_norm']].rename(
            columns={'resp_norm': 'ri'}),
            on='question_id')
        change = (merged['rb'] != merged['ri']).mean()

        rows.append(dict(model=model, group=grp,
                         soft_b=soft_b, soft_i=soft_i,
                         hard_b=hard_b, hard_i=hard_i,
                         change=change))

    df = pd.DataFrame(rows)
    df['label'] = df['model'].map(LABEL_MAP).fillna(
        df['model'].str.replace(' (LM)', '', regex=False)
    )
    df['family'] = df['model'].map(_family_key)
    df['size_key'] = df['model'].map(_size_key)
    return df.sort_values(['group', 'family', 'size_key', 'label'],
                          ascending=[True, True, True, True])


# ─── Shared: build y-positions from change-sorted order ──────────────────────
def _build_y_positions(df, sort_col='change'):
    if {'family', 'size_key'}.issubset(df.columns):
        df_s = df.sort_values(['group', 'family', 'size_key', 'label'],
                              ascending=[True, True, True, True])
    else:
        df_s = df.sort_values(['group', sort_col], ascending=[True, True])
    y_pos, y, spans = {}, 0, {}
    for grp in GROUP_ORDER:
        sub_ = df_s[df_s['group'] == grp].reset_index(drop=True)
        if sub_.empty:
            continue
        start = y
        for _, row in sub_.iterrows():
            y_pos[row['model']] = y
            y += 1
        spans[grp] = (start, y - 1)
        y += 0.6
    return df_s, y_pos, spans


# ─── Draw helpers ─────────────────────────────────────────────────────────────
def _draw_dumbbell(ax, df_s, y_pos, spans, col_b, col_i, xlabel, title, xlim=None):
    max_val = max(df_s[col_b].max(), df_s[col_i].max())
    if xlim is None:
        xlim = max_val * 1.35
    label_thresh = xlim * 0.80
    label_pad = xlim * 0.015

    for grp in GROUP_ORDER:
        sub_ = df_s[df_s['group'] == grp]
        if sub_.empty:
            continue
        c = GROUP_COLORS[grp]
        for _, row in sub_.iterrows():
            yp = y_pos[row['model']]
            ax.plot([row[col_b], row[col_i]], [yp, yp],
                    color=c, lw=1.5, alpha=0.7, zorder=2)
            ax.scatter([row[col_b]], [yp], color=c, s=55, zorder=3,
                       marker='o', edgecolors='white', linewidths=0.6)
            ax.scatter([row[col_i]], [yp], color=c, s=55, zorder=3,
                       marker='s', edgecolors='white', linewidths=0.6)
            far  = max(row[col_b], row[col_i])
            near = min(row[col_b], row[col_i])
            if far > label_thresh:
                ax.text(near - label_pad, yp, row['label'],
                        va='center', ha='right', fontsize=7, color='#333333')
            else:
                ax.text(far + label_pad, yp, row['label'],
                        va='center', ha='left', fontsize=7, color='#333333')
    for grp, (s, e) in spans.items():
        mid = (s + e) / 2
        ax.text(-0.01, mid, grp, va='center', ha='right', fontsize=7.5,
                color=GROUP_COLORS[grp], fontweight='bold',
                transform=ax.get_yaxis_transform())
    h1, = ax.plot([], [], 'o', color='gray', ms=6, label='blind',
                  markeredgecolor='white')
    h2, = ax.plot([], [], 's', color='gray', ms=6, label='inst_blind',
                  markeredgecolor='white')
    ax.legend(handles=[h1, h2], fontsize=8, loc='lower right', frameon=True)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-xlim * 0.08, xlim)
    ax.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.4)


def _draw_response_change(ax, df_inst, y_pos_shared=None, grp_spans_shared=None,
                          show_labels=True, title='Response change rate: blind → inst_blind',
                          shared_xlim=1.18):
    """Draw a response-change horizontal bar chart.

    If y_pos_shared / grp_spans_shared are provided the caller controls row
    placement (for aligned multi-variant panels).  Otherwise positions are
    built internally from df_inst sorted by change rate within each group.

    Returns (y_pos, grp_spans) so callers can reuse them.
    """
    if y_pos_shared is None:
        df_s = df_inst.sort_values(['group', 'change'], ascending=[True, False])
        y_pos2, y2, grp_spans2 = {}, 0, {}
        for grp in GROUP_ORDER:
            sub_ = df_s[df_s['group'] == grp].reset_index(drop=True)
            if sub_.empty:
                continue
            start = y2
            for _, row in sub_.iterrows():
                y_pos2[row['model']] = y2
                y2 += 1
            grp_spans2[grp] = (start, y2 - 1)
            y2 += 0.6
    else:
        y_pos2, grp_spans2 = y_pos_shared, grp_spans_shared

    # Colored background bands instead of text group annotations
    for grp, (s, e) in grp_spans2.items():
        ax.axhspan(s - 0.35, e + 0.35, color=GROUP_COLORS[grp], alpha=0.07, zorder=0)

    for grp in GROUP_ORDER:
        sub_ = df_inst[df_inst['group'] == grp]
        if sub_.empty:
            continue
        c = GROUP_COLORS[grp]
        for _, row in sub_.iterrows():
            yp = y_pos2.get(row['model'])
            if yp is None:
                continue
            ax.barh(yp, row['change'], color=c, alpha=0.8, height=0.65,
                    edgecolor='none')
            ax.text(row['change'] + 0.01, yp,
                    f"{row['change']:.0%}", va='center', fontsize=7.5,
                    color='#333333')
            if show_labels:
                ax.text(-0.01, yp, row['label'], va='center', ha='right',
                        fontsize=7, color='#333333')

    ax.set_yticks([])
    ax.set_xlabel('Fraction of responses changed', fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.15, shared_xlim)
    ax.axvline(0, color='gray', lw=0.5, ls='-', alpha=0.3)
    return y_pos2, grp_spans2


def _draw_merged_abstention(ax, df_inst):
    """Offset-dumbbell merged plot: soft (dotted, hollow) and hard (solid, filled)
    on one shared x-axis. Soft row at y, hard row at y + HARD_OFFSET.
    Arrows (ax.annotate) show blind → inst_blind direction.
    Style after Gavrikov et al. (2024) Fig. 2.
    """
    HARD_OFFSET = 0.40

    df_s, y_pos, spans = _build_y_positions(df_inst, sort_col='soft_b')

    # x limit: cover all four columns, capped at 0.55
    max_val = max(df_s[['soft_b', 'soft_i', 'hard_b', 'hard_i']].max().max(), 0.01)
    xlim    = min(max_val * 1.40, 0.55)

    # Subtle dotted vertical reference lines (Fig. 2 style)
    for xv in [0.1, 0.2, 0.3, 0.4, 0.5]:
        if xv < xlim:
            ax.axvline(xv, color='#CCCCCC', lw=0.7, ls=':', zorder=0)

    for grp in GROUP_ORDER:
        sub_ = df_s[df_s['group'] == grp]
        if sub_.empty:
            continue
        c = GROUP_COLORS[grp]
        for _, row in sub_.iterrows():
            yp = y_pos[row['model']]
            yh = yp + HARD_OFFSET

            # ── Soft: dotted arrow, open/hollow markers ─────────────────────
            ax.annotate("",
                xy=(row['soft_i'], yp), xytext=(row['soft_b'], yp),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=1.2,
                               linestyle='dotted', mutation_scale=7, alpha=0.55),
                annotation_clip=False, zorder=2)
            ax.scatter([row['soft_b']], [yp], color=c, s=50, zorder=3,
                       marker='o', facecolors='none', edgecolors=c, linewidths=1.0)
            ax.scatter([row['soft_i']], [yp], color=c, s=50, zorder=3,
                       marker='s', facecolors='none', edgecolors=c, linewidths=1.0)

            # ── Hard: solid arrow, filled markers ───────────────────────────
            ax.annotate("",
                xy=(row['hard_i'], yh), xytext=(row['hard_b'], yh),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=1.5,
                               mutation_scale=8, alpha=0.75),
                annotation_clip=False, zorder=2)
            ax.scatter([row['hard_b']], [yh], color=c, s=38, zorder=3,
                       marker='o', edgecolors='white', linewidths=0.5)
            ax.scatter([row['hard_i']], [yh], color=c, s=38, zorder=3,
                       marker='s', edgecolors='white', linewidths=0.5)

            # Model label anchored to y-axis, centred between the two rows
            ax.text(-0.01, yp + HARD_OFFSET / 2, row['label'],
                    va='center', ha='right', fontsize=8, color='#333333',
                    transform=ax.get_yaxis_transform())

    # Colored background bands per group (replace text labels)
    _GROUP_SHORT = {
        'VLM':                    'VLM',
        'VLM backbone decoder':   'Backbone',
        'standalone LLM':         'SA-LLM',
        'standalone LLM (think)': 'Think',
    }
    for grp, (s, e) in spans.items():
        c = GROUP_COLORS[grp]
        ax.axhspan(s - 0.35, e + HARD_OFFSET + 0.35, color=c, alpha=0.07, zorder=0)

    import matplotlib.patches as mpatches
    group_handles = [
        mpatches.Patch(color=GROUP_COLORS[g], alpha=0.45, label=_GROUP_SHORT[g])
        for g in GROUP_ORDER if g in spans
    ]
    style_handles = [
        mlines.Line2D([], [], color='#888888', marker='o', ls=':',  ms=6,
                      markerfacecolor='none', markeredgecolor='#888888',
                      label='soft'),
        mlines.Line2D([], [], color='#888888', marker='s', ls=':',  ms=6,
                      markerfacecolor='none', markeredgecolor='#888888',
                      label='soft+inst'),
        mlines.Line2D([], [], color='#888888', marker='o', ls='-', ms=6,
                      markeredgecolor='white', label='hard'),
        mlines.Line2D([], [], color='#888888', marker='s', ls='-', ms=6,
                      markeredgecolor='white', label='hard+inst'),
    ]
    ax.legend(handles=group_handles + style_handles, fontsize=7.5, frameon=True,
              ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.10))

    ax.set_yticks([])
    ax.set_xlabel('Abstention rate', fontsize=10)
    ax.set_xlim(-xlim * 0.05, xlim)
    ax.axvline(0, color='gray', lw=0.5, ls='-', alpha=0.3)
    ax.invert_yaxis()


def _draw_grouped_merged_abstention(ax, df_inst):
    """Compact group-level abstention plot for the main paper."""
    HARD_OFFSET = 0.34
    group_df = (
        df_inst.groupby('group', as_index=False)[['soft_b', 'soft_i', 'hard_b', 'hard_i']]
        .mean()
    )
    group_df['group'] = pd.Categorical(group_df['group'], categories=GROUP_ORDER, ordered=True)
    group_df = group_df.sort_values('group').dropna(subset=['group']).reset_index(drop=True)

    y_base = {grp: idx for idx, grp in enumerate(group_df['group'])}
    max_val = max(group_df[['soft_b', 'soft_i', 'hard_b', 'hard_i']].max().max(), 0.01)
    xlim = min(max_val * 1.35, 0.55)

    for xv in [0.1, 0.2, 0.3, 0.4, 0.5]:
        if xv < xlim:
            ax.axvline(xv, color='#CCCCCC', lw=0.7, ls=':', zorder=0)

    for _, row in group_df.iterrows():
        grp = row['group']
        c = GROUP_COLORS[grp]
        y = y_base[grp]
        yh = y + HARD_OFFSET

        ax.annotate(
            "",
            xy=(row['soft_i'], y), xytext=(row['soft_b'], y),
            arrowprops=dict(arrowstyle='-|>', color=c, lw=1.3,
                            linestyle='dotted', mutation_scale=7, alpha=0.6),
            annotation_clip=False, zorder=2
        )
        ax.scatter([row['soft_b']], [y], color=c, s=56, zorder=3,
                   marker='o', facecolors='none', edgecolors=c, linewidths=1.1)
        ax.scatter([row['soft_i']], [y], color=c, s=56, zorder=3,
                   marker='s', facecolors='none', edgecolors=c, linewidths=1.1)

        ax.annotate(
            "",
            xy=(row['hard_i'], yh), xytext=(row['hard_b'], yh),
            arrowprops=dict(arrowstyle='-|>', color=c, lw=1.6,
                            mutation_scale=8, alpha=0.8),
            annotation_clip=False, zorder=2
        )
        ax.scatter([row['hard_b']], [yh], color=c, s=42, zorder=3,
                   marker='o', edgecolors='white', linewidths=0.5)
        ax.scatter([row['hard_i']], [yh], color=c, s=42, zorder=3,
                   marker='s', edgecolors='white', linewidths=0.5)

        ax.text(-0.01, y + HARD_OFFSET / 2, grp, va='center', ha='right',
                fontsize=9, color=c, fontweight='bold',
                transform=ax.get_yaxis_transform())

    handles = [
        mlines.Line2D([], [], color='#888888', marker='o', ls=':', ms=6,
                      markerfacecolor='none', markeredgecolor='#888888',
                      label='soft'),
        mlines.Line2D([], [], color='#888888', marker='s', ls=':', ms=6,
                      markerfacecolor='none', markeredgecolor='#888888',
                      label='soft + instruction'),
        mlines.Line2D([], [], color='#888888', marker='o', ls='-', ms=6,
                      markeredgecolor='white', label='hard'),
        mlines.Line2D([], [], color='#888888', marker='s', ls='-', ms=6,
                      markeredgecolor='white', label='hard + instruction'),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=True, ncol=4,
              loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.set_yticks([])
    ax.set_xlabel('Abstention rate', fontsize=10)
    ax.set_title('Group-level abstention: blind → inst_blind', fontsize=11)
    ax.set_xlim(-xlim * 0.05, xlim)
    ax.set_ylim(-0.2, len(group_df) - 0.15 + HARD_OFFSET)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', lw=0.5, ls='-', alpha=0.3)


# ─── Table: per-model abstention rates ───────────────────────────────────────

_GROUP_LABEL = {
    'VLM':                    r'\textit{Vision-Language Models}',
    'VLM backbone decoder':   r'\textit{VLM Backbone Decoders}',
    'standalone LLM':         r'\textit{Standalone LLMs}',
    'standalone LLM (think)': r'\textit{Standalone LLMs (thinking)}',
}


def make_abstention_table(df_inst: pd.DataFrame, suffix: str) -> None:
    """Write a LaTeX table of per-model abstention rates for a given dataset."""
    group_rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    df_s = df_inst.copy()
    df_s['_grp_idx'] = df_s['group'].map(group_rank).fillna(99)
    df_s = df_s.sort_values(['_grp_idx', 'family', 'size_key', 'label'])

    def _pct(v):
        return f"{v * 100:.1f}"

    def _delta(a, b):
        d = (b - a) * 100
        return f"$+${d:.1f}" if d >= 0 else f"${d:.1f}$"

    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\small",
        (r"\caption{Per-model soft and hard abstention rates under blind and "
         r"blind-with-instruction conditions (variant~C, 113 questions). "
         r"B\,=\,blind; I\,=\,blind+instruction; $\Delta$\,=\,I\,$-$\,B (pp). "
         r"Resp\,$\Delta$ = fraction of answers that change blind$\to$instruction.}"),
        rf"\label{{tab:abstention_inst_effect_{suffix}}}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Soft abstention (\%)} "
        r"& \multicolumn{3}{c}{Hard abstention (\%)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Model & B & I & $\Delta$ & B & I & $\Delta$ \\",
        r"\midrule",
    ]

    prev_grp = None
    for _, row in df_s.iterrows():
        grp = row['group']
        if grp != prev_grp:
            if prev_grp is not None:
                lines.append(r"\midrule")
            lines.append(f"\\multicolumn{{7}}{{l}}{{{_GROUP_LABEL.get(grp, grp)}}} \\\\")
            lines.append(r"\midrule")
            prev_grp = grp

        lines.append(
            f"{row['label']} "
            f"& {_pct(row['soft_b'])} & {_pct(row['soft_i'])} & {_delta(row['soft_b'], row['soft_i'])} "
            f"& {_pct(row['hard_b'])} & {_pct(row['hard_i'])} & {_delta(row['hard_b'], row['hard_i'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    out = TABLE_DIR / f"abstention_inst_effect_{suffix}.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  [table] {out.name}")


# ─── Generate figures for each dataset ───────────────────────────────────────
for suffix, df_mb, df_mi in DATASETS:
    print(f'\n══ {suffix} ══')
    max_missing = 1 if suffix == f'v{VARIANT}_q{N_SUB}' else 0
    df_inst = compute_stats(df_mb, df_mi, max_missing=max_missing)
    print(f'  Models with sufficient data: {len(df_inst)}')

    print('  ── fig_soft_abstention ──')
    df_s, y_pos, spans = _build_y_positions(df_inst, sort_col='soft_b')
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_dumbbell(ax, df_s, y_pos, spans,
                   col_b='soft_b', col_i='soft_i',
                   xlabel='Soft abstention rate',
                   title='Soft abstention: blind → inst_blind',
                   xlim=0.65)
    plt.tight_layout()
    save(fig, f'soft_abstention_{suffix}.png')

    print('  ── hard_abstention ──')
    df_s, y_pos, spans = _build_y_positions(df_inst, sort_col='hard_b')
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_dumbbell(ax, df_s, y_pos, spans,
                   col_b='hard_b', col_i='hard_i',
                   xlabel='Hard abstention rate',
                   title='Hard abstention: blind → inst_blind')
    plt.tight_layout()
    save(fig, f'hard_abstention_{suffix}.png')

    print('  ── response_change ──')
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_response_change(ax, df_inst)
    plt.tight_layout()
    save(fig, f'response_change_{suffix}.png')

    print('  ── merged_abstention ──')
    h = max(5.5, len(df_inst) * 0.20 + 1.8)
    fig, ax = plt.subplots(figsize=(4.8, h))
    _draw_merged_abstention(ax, df_inst)
    plt.tight_layout()
    save(fig, f'merged_abstention_{suffix}.png')

    print('  ── merged_abstention_groups ──')
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    _draw_grouped_merged_abstention(ax, df_inst)
    plt.tight_layout()
    save(fig, f'merged_abstention_groups_{suffix}.png')

    if suffix == f'v{VARIANT}_q{N_SUB}':
        print('  ── abstention table ──')
        make_abstention_table(df_inst, suffix)

# ─── 3-column variant (C/B/A) figure: merged abstention on q113 subset ────────
print('\n══ 3-column variant figure (C/B/A, q113) ══')

VARIANT_TITLES = {
    'C': 'Variant C\n(original)',
    'B': 'Variant B\n(subject ablated)',
    'A': 'Variant A\n(pronominalized)',
}
VARIANT_ORDER_3COL = ['C', 'B', 'A']

variant_stats = {}
for v in VARIANT_ORDER_3COL:
    exp = read_response_exports(ROOT, variant=v, check_completeness=False)
    db = exp['model_blind']
    di = exp['model_inst_blind']
    for _df in (db, di):
        if not _df.empty and 'response' in _df.columns:
            _df['response'] = _df['response'].fillna('').astype(str).apply(clean_answer)
    variant_stats[v] = compute_stats(db, di, max_missing=1)
    print(f'  Variant {v}: {len(variant_stats[v])} models')

# Shared x-limit across all variants
all_max = max(
    df[['soft_b', 'soft_i', 'hard_b', 'hard_i']].max().max()
    for df in variant_stats.values()
)
shared_xlim = min(all_max * 1.40, 0.55)

# Build shared y-positions from variant C so rows align across columns
_, y_pos_ref, spans_ref = _build_y_positions(variant_stats['C'], sort_col='soft_b')
HARD_OFFSET = 0.40
n_y = max(y_pos_ref.values()) + 1 + HARD_OFFSET if y_pos_ref else 12

fig, axes = plt.subplots(1, 3, figsize=(17, max(8, n_y * 0.38 + 2)), sharey=False)

for col_idx, v in enumerate(VARIANT_ORDER_3COL):
    ax = axes[col_idx]
    df_v = variant_stats[v]
    models_here = set(df_v['model'])

    for xv in [0.1, 0.2, 0.3, 0.4, 0.5]:
        if xv < shared_xlim:
            ax.axvline(xv, color='#CCCCCC', lw=0.7, ls=':', zorder=0)

    for grp in GROUP_ORDER:
        sub_ = df_v[df_v['group'] == grp]
        if sub_.empty:
            continue
        c = GROUP_COLORS[grp]
        for _, row in sub_.iterrows():
            yp = y_pos_ref.get(row['model'])
            if yp is None:
                continue
            yh = yp + HARD_OFFSET

            ax.annotate("",
                xy=(row['soft_i'], yp), xytext=(row['soft_b'], yp),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=1.2,
                               linestyle='dotted', mutation_scale=7, alpha=0.55),
                annotation_clip=False, zorder=2)
            ax.scatter([row['soft_b']], [yp], color=c, s=50, zorder=3,
                       marker='o', facecolors='none', edgecolors=c, linewidths=1.0)
            ax.scatter([row['soft_i']], [yp], color=c, s=50, zorder=3,
                       marker='s', facecolors='none', edgecolors=c, linewidths=1.0)

            ax.annotate("",
                xy=(row['hard_i'], yh), xytext=(row['hard_b'], yh),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=1.5,
                               mutation_scale=8, alpha=0.75),
                annotation_clip=False, zorder=2)
            ax.scatter([row['hard_b']], [yh], color=c, s=38, zorder=3,
                       marker='o', edgecolors='white', linewidths=0.5)
            ax.scatter([row['hard_i']], [yh], color=c, s=38, zorder=3,
                       marker='s', edgecolors='white', linewidths=0.5)

            # Model labels only on leftmost column
            if col_idx == 0:
                ax.text(-0.01, yp + HARD_OFFSET / 2, row['label'],
                        va='center', ha='right', fontsize=7, color='#333333',
                        transform=ax.get_yaxis_transform())

    # Group labels only on rightmost column
    if col_idx == 2:
        for grp, (s, e) in spans_ref.items():
            mid = (s + e) / 2 + HARD_OFFSET / 2
            ax.text(1.02, mid, grp, va='center', ha='left', fontsize=7.5,
                    color=GROUP_COLORS[grp], fontweight='bold',
                    transform=ax.get_yaxis_transform())

    ax.set_yticks([])
    ax.set_xlabel('Abstention rate', fontsize=10)
    ax.set_title(VARIANT_TITLES[v], fontsize=11)
    ax.set_xlim(-shared_xlim * 0.05, shared_xlim)
    ax.axvline(0, color='gray', lw=0.5, ls='-', alpha=0.3)
    ax.invert_yaxis()

handles = [
    mlines.Line2D([], [], color='#888888', marker='o', ls=':', ms=6,
                  markerfacecolor='none', markeredgecolor='#888888', label='soft'),
    mlines.Line2D([], [], color='#888888', marker='s', ls=':', ms=6,
                  markerfacecolor='none', markeredgecolor='#888888', label='soft + instruction'),
    mlines.Line2D([], [], color='#888888', marker='o', ls='-', ms=6,
                  markeredgecolor='white', label='hard'),
    mlines.Line2D([], [], color='#888888', marker='s', ls='-', ms=6,
                  markeredgecolor='white', label='hard + instruction'),
]
fig.legend(handles=handles, fontsize=8, frameon=True, ncol=4,
           loc='lower center', bbox_to_anchor=(0.5, 0.01))
fig.suptitle('Abstention shift: blind → inst_blind across question variants', fontsize=12, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])
save(fig, f'merged_abstention_variants_q{N_SUB}.png')

# ─── 3-column response change figure (C/B/A) — replaced by table ─────────────
# print('\n══ 3-column response change figure (C/B/A, q113) ══')
#
# vc_df = variant_stats['C'].sort_values(['group', 'change'], ascending=[True, False])
# y_pos_rc, y_rc, spans_rc = {}, 0, {}
# for grp in GROUP_ORDER:
#     sub_ = vc_df[vc_df['group'] == grp].reset_index(drop=True)
#     if sub_.empty:
#         continue
#     start = y_rc
#     for _, row in sub_.iterrows():
#         y_pos_rc[row['model']] = y_rc
#         y_rc += 1
#     spans_rc[grp] = (start, y_rc - 1)
#     y_rc += 0.6
#
# n_rows = max(y_pos_rc.values()) + 1 if y_pos_rc else 12
# fig, axes = plt.subplots(1, 3, figsize=(15, max(6, n_rows * 0.38 + 2)), sharey=False)
#
# VARIANT_TITLES_RC = {
#     'C': 'Variant C\n(original)',
#     'B': 'Variant B\n(subject ablated)',
#     'A': 'Variant A\n(pronominalized)',
# }
# for col_idx, v in enumerate(VARIANT_ORDER_3COL):
#     ax = axes[col_idx]
#     _draw_response_change(
#         ax, variant_stats[v],
#         y_pos_shared=y_pos_rc,
#         grp_spans_shared=spans_rc,
#         show_labels=(col_idx == 0),
#         title=VARIANT_TITLES_RC[v],
#         shared_xlim=1.05,
#     )
#     ax.invert_yaxis()
#
# import matplotlib.patches as mpatches
# _GROUP_SHORT_RC = {
#     'VLM':                    'VLM',
#     'VLM backbone decoder':   'Backbone',
#     'standalone LLM':         'SA-LLM',
#     'standalone LLM (think)': 'Think',
# }
# group_handles = [
#     mpatches.Patch(color=GROUP_COLORS[g], alpha=0.45, label=_GROUP_SHORT_RC[g])
#     for g in GROUP_ORDER if g in spans_rc
# ]
# fig.legend(handles=group_handles, fontsize=8, frameon=True, ncol=4,
#            loc='lower center', bbox_to_anchor=(0.5, 0.01))
# fig.suptitle('Response change rate: blind → inst_blind across question variants',
#              fontsize=12, y=1.01)
# plt.tight_layout(rect=[0, 0.05, 1, 1])
# rc_fname = f'response_change_variants_q{N_SUB}.png'
# save(fig, rc_fname)

# ─── Table: response change rate (blind → inst_blind) per model × variant ─────
print('\n══ Response change table ══')

_GROUP_LABEL_RC = {
    'VLM':                    r'\textit{Vision-Language Models}',
    'VLM backbone decoder':   r'\textit{VLM Backbone Decoders}',
    'standalone LLM':         r'\textit{Standalone LLMs}',
    'standalone LLM (think)': r'\textit{Standalone LLMs (thinking)}',
}

# Merge variant_stats into one wide dataframe
rc_frames = []
for v in VARIANT_ORDER_3COL:
    tmp = variant_stats[v][['model', 'group', 'family', 'size_key', 'label', 'change']].copy()
    tmp.rename(columns={'change': f'change_{v}'}, inplace=True)
    rc_frames.append(tmp.set_index('model'))

rc_wide = rc_frames[0].join(rc_frames[1][f'change_B']).join(rc_frames[2][f'change_A'])
rc_wide = rc_wide.reset_index()

group_rank = {g: i for i, g in enumerate(GROUP_ORDER)}
rc_wide['_grp_idx'] = rc_wide['group'].map(group_rank).fillna(99)
rc_wide = rc_wide.sort_values(['_grp_idx', 'family', 'size_key', 'label'])

def _pct_rc(v):
    return f"{v * 100:.1f}\\%" if not pd.isna(v) else "---"

def _delta_colored(base, other):
    if pd.isna(base) or pd.isna(other):
        return ""
    d = (other - base) * 100
    if abs(d) < 0.05:
        return ""
    sign = "+" if d > 0 else ""
    color = "teal" if d > 0 else "red"
    return r" \textcolor{" + color + r"}{(" + sign + f"{d:.1f}" + r")}"

lines_rc = [
    r"\begin{table}[h]",
    r"\centering",
    r"\small",
    (r"\caption{Response change rate (\% of answers changed blind"
     r"$\to$blind+instruction) per model and variant (C/B/A, 113 questions)."
     r" Colored deltas show change relative to variant~C:"
     r" \textcolor{teal}{positive} = more change, \textcolor{red}{negative} = less.}"),
    rf"\label{{tab:response_change_variants}}",
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r"Model & Orig (C) & Weaker (B) & Pronom (A) \\",
    r"\midrule",
]

prev_grp = None
for _, row in rc_wide.iterrows():
    grp = row['group']
    if grp != prev_grp:
        if prev_grp is not None:
            lines_rc.append(r"\midrule")
        lines_rc.append(
            f"\\multicolumn{{4}}{{l}}{{{_GROUP_LABEL_RC.get(grp, grp)}}} \\\\"
        )
        lines_rc.append(r"\midrule")
        prev_grp = grp

    cell_c = _pct_rc(row['change_C'])
    cell_b = _pct_rc(row['change_B']) + _delta_colored(row['change_C'], row['change_B'])
    cell_a = _pct_rc(row['change_A']) + _delta_colored(row['change_C'], row['change_A'])
    lines_rc.append(f"{row['label']} & {cell_c} & {cell_b} & {cell_a} \\\\")

lines_rc += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

rc_tex_out = TABLE_DIR / "response_change_variants.tex"
rc_tex_out.write_text("\n".join(lines_rc) + "\n")
print(f"  [table] {rc_tex_out.name}")

# Also save CSV
rc_csv_out = OUT_DIR / "response_change_variants.csv"
rc_wide[['label', 'group', 'change_C', 'change_B', 'change_A']].to_csv(
    rc_csv_out, index=False, float_format="%.4f"
)
print(f"  [csv]   {rc_csv_out.name}")

print('\nDone. Outputs in:', OUT_DIR)
