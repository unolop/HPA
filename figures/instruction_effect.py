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
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'figures'))

from utils.constants import GROUP_COLORS, GROUP_ORDER
from utils.abstention import classify
from utils.load_session import clean_answer
from utils.model_registry import MODEL_TYPE, default_all_models
from helpers import read_response_exports
OUT_DIR = ROOT / "figures/instruction_effect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT = 'C'

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

LABEL_MAP = {
    'Qwen3-VL-32B (LM)':  'Qwen3-VL-32B\n(backbone)',
    'LLaVA-Mistral (LM)': 'LLaVA-M\n(backbone)',
    'LLaVA-Vicuna (LM)':  'LLaVA-V\n(backbone)',
    'LLaVA-1.5 (LM)':     'LLaVA-1.5\n(backbone)',
    'LLaVA-Mistral':      'LLaVA-M',
    'LLaVA-Vicuna':       'LLaVA-V',
    'LLaVA-1.5-7B':       'LLaVA-1.5',
    'Qwen3-VL-8B':        'Qwen3-VL-8B',
    'InternVL-1B':        'IVL-1B',
    'InternVL-2B':        'IVL-2B',
    'InternVL-8B':        'IVL-8B',
    'Qwen3-8B':           'Qwen3-8B',
    'Qwen3-8B (think)':   'Qwen3-8B-T',
    'Qwen3-32B':          'Qwen3-32B',
    'Qwen3-32B (think)':  'Qwen3-32B-T',
    'Qwen3-4B':           'Qwen3-4B',
    'Qwen3-4B (think)':   'Qwen3-4B-T',
    'Qwen3-1.7B':         'Qwen3-1.7B',
    'Qwen3-1.7B (think)': 'Qwen3-1.7B-T',
    'Qwen3-0.6B':         'Qwen3-0.6B',
    'Qwen3-0.6B (think)': 'Qwen3-0.6B-T',
    'Qwen2.5-7B':         'Qwen2.5-7B',
    'Phi-3.5-mini':       'Phi-3.5',
    'Mistral-7B':         'Mistral-7B',
    'Vicuna-13B':         'Vicuna-13B',
}


def norm_ans(s):
    if pd.isna(s): return ''
    return str(s).strip().lower()


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [instruction_effect] {name}')


# ─── Remove old figures ───────────────────────────────────────────────────────
for old in OUT_DIR.glob('*.png'):
    old.unlink()
print(f'Cleared old figures from {OUT_DIR}')


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
exports = read_response_exports(ROOT, variant=VARIANT)
df_sub_b = exports['model_blind']
df_sub_i = exports['model_inst_blind']
N_SUB = df_sub_b['question_id'].nunique()
print(f'  blind: {len(df_sub_b)} rows, {N_SUB} questions, '
      f'{df_sub_b["model"].nunique()} models')

DATASETS = [
    (f'v{VARIANT}_q{N_FULL}', df_full_b, df_full_i),
    (f'v{VARIANT}_q{N_SUB}',  df_sub_b,  df_sub_i),
]


# ─── Stats computation ────────────────────────────────────────────────────────
def compute_stats(df_mb, df_mi, min_n=50):
    df_mb = df_mb.copy()
    df_mi = df_mi.copy()
    df_mb['resp_norm'] = df_mb['response'].apply(norm_ans)
    df_mi['resp_norm'] = df_mi['response'].apply(norm_ans)
    df_mb['cls'] = df_mb['response'].apply(
        lambda x: classify(str(x) if not pd.isna(x) else '', None))
    df_mi['cls'] = df_mi['response'].apply(
        lambda x: classify(str(x) if not pd.isna(x) else '', None))

    rows = []
    for model in df_mb['model'].unique():
        b = df_mb[df_mb['model'] == model]
        i = df_mi[df_mi['model'] == model]
        if len(b) < min_n or len(i) < min_n:
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
    df['label'] = df['model'].map(LABEL_MAP).fillna(df['model'])
    return df.sort_values(['group', 'change'], ascending=[True, True])


# ─── Shared: build y-positions from change-sorted order ──────────────────────
def _build_y_positions(df, sort_col='change'):
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


def _draw_response_change(ax, df_inst):
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
    for grp in GROUP_ORDER:
        sub_ = df_s[df_s['group'] == grp]
        if sub_.empty:
            continue
        c = GROUP_COLORS[grp]
        for _, row in sub_.iterrows():
            yp = y_pos2[row['model']]
            ax.barh(yp, row['change'], color=c, alpha=0.8, height=0.65,
                    edgecolor='none')
            ax.text(row['change'] + 0.01, yp,
                    f"{row['change']:.0%}", va='center', fontsize=7.5,
                    color='#333333')
            ax.text(-0.01, yp, row['label'], va='center', ha='right',
                    fontsize=7, color='#333333')
    for grp, (s, e) in grp_spans2.items():
        mid = (s + e) / 2
        ax.text(1.02, mid, grp, va='center', ha='left', fontsize=7.5,
                color=GROUP_COLORS[grp], fontweight='bold',
                transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_xlabel('Fraction of responses changed', fontsize=10)
    ax.set_title('Response change rate: blind → inst_blind', fontsize=11)
    ax.set_xlim(-0.15, 1.18)
    ax.axvline(0, color='gray', lw=0.5, ls='-', alpha=0.3)


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
                    va='center', ha='right', fontsize=7, color='#333333',
                    transform=ax.get_yaxis_transform())

    # Group labels on right
    for grp, (s, e) in spans.items():
        mid = (s + e) / 2 + HARD_OFFSET / 2
        ax.text(1.02, mid, grp, va='center', ha='left', fontsize=7.5,
                color=GROUP_COLORS[grp], fontweight='bold',
                transform=ax.get_yaxis_transform())

    # Legend below — all 4 entries in one horizontal row
    handles = [
        mlines.Line2D([], [], color='#888888', marker='o', ls=':',  ms=6,
                      markerfacecolor='none', markeredgecolor='#888888',
                      label='soft / blind'),
        mlines.Line2D([], [], color='#888888', marker='s', ls=':',  ms=6,
                      markerfacecolor='none', markeredgecolor='#888888',
                      label='soft / inst_blind'),
        mlines.Line2D([], [], color='#888888', marker='o', ls='-', ms=6,
                      markeredgecolor='white', label='hard / blind'),
        mlines.Line2D([], [], color='#888888', marker='s', ls='-', ms=6,
                      markeredgecolor='white', label='hard / inst_blind'),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=True, ncol=4,
              loc='upper center', bbox_to_anchor=(0.5, -0.08))

    ax.set_yticks([])
    ax.set_xlabel('Abstention rate', fontsize=10)
    ax.set_title('Soft & hard abstention: blind → inst_blind', fontsize=11)
    ax.set_xlim(-xlim * 0.05, xlim)
    ax.axvline(0, color='gray', lw=0.5, ls='-', alpha=0.3)


# ─── Generate figures for each dataset ───────────────────────────────────────
for suffix, df_mb, df_mi in DATASETS:
    print(f'\n══ {suffix} ══')
    df_inst = compute_stats(df_mb, df_mi)
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
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_merged_abstention(ax, df_inst)
    plt.tight_layout()
    save(fig, f'merged_abstention_{suffix}.png')

print('\nDone. Outputs in:', OUT_DIR)
