#!/usr/bin/env python3
"""
Generate one figure per sampled question:
  Left col (3 rows): confidence line plots per condition, hued by model
  Right col (3 rows): compact answer tables per condition
Filename: Q{qid}_{tiers}.png
"""

import json, os, sys, shutil, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import pandas as pd

sys.path.insert(0, '/home/david/Desktop/yuna/HPA/analysis')
from utils.vqa import VQAAnswerMapper, vqa_accuracy

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = '/home/david/Desktop/yuna/HPA/evaluation/logits/pretrained'
CONTROL_PATH = '/home/david/Desktop/yuna/HPA/dataset/vqa/vqa1k_control.jsonl'
IMAGES_DIR  = '/home/david/Desktop/yuna/data/val2014'
SAMPLE_CSV  = '/home/david/Desktop/yuna/HPA/analysis/csv/sampled_control_questions.csv'
HUMAN_CSV   = '/home/david/Desktop/yuna/HPA/analysis/csv/human_study_v2_blind.csv'
OUT_DIR     = '/home/david/Desktop/yuna/HPA/analysis/figures/tier_examples'

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

# ── Model sets ────────────────────────────────────────────────────────────────
ORIG_MODELS  = {'llava-v1.6-vicuna-7b-hf': 'Vc7B',  'llava-v1.6-vicuna-13b-hf': 'Vc13B',
                'Qwen3-VL-2B-Instruct':    'Q2B',    'Qwen3-VL-4B-Instruct':    'Q4B'}
BLIND_MODELS = {'llava-v1.6-vicuna-7b-hf': 'Vc7B',  'llava-1.5-7b-hf':         'L1.5',
                'llava-v1.6-mistral-7b-hf':'LMst',   'Qwen3-VL-8B-Instruct':    'Q8B'}
INST_MODELS  = {'llava-v1.6-vicuna-7b-hf': 'Vc7B',  'Qwen3-VL-2B-Instruct':    'Q2B',
                'llava-1.5-7b-hf':         'L1.5',   'llava-v1.6-mistral-7b-hf':'LMst'}

CONDITIONS = [
    ('vqa_1k_control',            'With Image', ORIG_MODELS,  '#2e7d32'),
    ('vqa_1k_control_blind',      'Blind',      BLIND_MODELS, '#b71c1c'),
    ('vqa_1k_control_inst_blind', 'Blind+Inst', INST_MODELS,  '#4a148c'),
]

CONTROL_TYPES = ['question', 'deictic_removed', 'object_removed', 'weaker_object', 'pronominalized']
CT_LABELS     = ['Orig.Q', 'Deict.', 'Obj.Rm', 'Wk.Obj', 'Pron.']

# Consistent colors per model short name
MODEL_COLORS = {
    'Vc7B': '#1565c0', 'Vc13B': '#e65100', 'Q2B':  '#2e7d32',
    'Q4B':  '#c62828', 'L1.5':  '#6a1b9a', 'LMst': '#00695c', 'Q8B': '#f57f17',
}

mapper = VQAAnswerMapper()

# ── Load control questions ────────────────────────────────────────────────────
print("Loading control questions...")
qid_to_ctrl = {}
with open(CONTROL_PATH) as f:
    for line in f:
        ex = json.loads(line)
        qid_to_ctrl[ex['question_id']] = ex

# ── Load all model outputs upfront ───────────────────────────────────────────
print("Loading model outputs...")
ans_data, conf_data = {}, {}
all_model_ids = set(list(ORIG_MODELS) + list(BLIND_MODELS) + list(INST_MODELS))

for ds, _, mdict, _ in CONDITIONS:
    for model in mdict:
        path = f'{RESULTS_DIR}/{model}/{ds}.jsonl'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                ex      = json.loads(line)
                qid     = ex['question_id']
                answers = ex.get('generated_answers') or {}
                logits  = ex.get('generated_logits')  or {}
                for ct in CONTROL_TYPES:
                    if ct in answers and answers[ct]:
                        ans_data[(model, ds, qid, ct)] = answers[ct]
                    if ct in logits:
                        probs = np.exp([t['logprob'] for t in logits[ct]['content']])
                        conf_data[(model, ds, qid, ct)] = float(probs.mean())
print("Done.")

# ── Previous human study question ids ────────────────────────────────────────
human_qids = set(pd.read_csv(HUMAN_CSV)['question_id']) if os.path.exists(HUMAN_CSV) else set()
print(f"Human study questions: {len(human_qids)}")

# ── Tier assignments ──────────────────────────────────────────────────────────
qdf = pd.read_csv(SAMPLE_CSV)
np.random.seed(42)

def strat_sample(mask, n=30, seed=42):
    pool = qdf[mask].copy()
    if len(pool) == 0: return set()
    counts = pool['ent'].value_counts(normalize=True)
    out = []
    for ent, frac in counts.items():
        k = max(1, round(n * frac))
        sub = pool[pool['ent'] == ent]
        out.extend(sub.sample(min(k, len(sub)), random_state=seed)['question_id'].tolist())
    return set(list(set(out))[:n])

tier_sets = {
    'A': strat_sample(qdf['overconfident_wrong']),
    'B': strat_sample(qdf['mean_output_changes'] > 2),
    'C': strat_sample(qdf['consensus_wrong']),
    'D': strat_sample(qdf['acc_drop'] > 0.3),
    'E': strat_sample((qdf['conf_drop'] > 0.05) & (qdf['acc_baseline'] > 0.5)),
}
tier_descs  = {'A': 'Overconfident & Wrong', 'B': 'Answer Flip',
               'C': 'Consensus Wrong', 'D': 'High Degradation', 'E': 'Confidence Collapse'}
tier_colors = {'A': '#c62828', 'B': '#1565c0', 'C': '#2e7d32',
               'D': '#e65100', 'E': '#6a1b9a'}

all_sampled = set.union(*tier_sets.values())
qid_tiers   = {''.join(t for t in 'ABCDE' if q in tier_sets[t]): None
               for q in all_sampled}  # just to show structure
qid_tiers   = {q: ''.join(t for t in 'ABCDE' if q in tier_sets[t]) for q in all_sampled}
print(f"Total sampled: {len(all_sampled)}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def cell_bg(answer, gt):
    if not answer or answer == '—':
        return '#DEDEDE'
    acc = vqa_accuracy(answer, gt)
    if acc is None: return '#DEDEDE'
    return '#A8D8A8' if acc > 0.5 else '#F4AAAA'

def draw_table(ax, qid, model_dict, dataset, ds_label, header_color):
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    gt   = mapper.get_answers(qid)
    ctrl = qid_to_ctrl.get(qid, {})

    n_m  = len(model_dict)
    # Row heights — question row taller to fit wrapped full text
    rh_h = 0.09
    rh_q = 0.26
    rh_m = (1.0 - rh_h - rh_q) / n_m   # ~0.1625 for 4 models

    # Column widths
    w0  = 0.085
    wct = (1.0 - w0) / len(CONTROL_TYPES)

    col_x = [0.0] + [w0 + i * wct for i in range(len(CONTROL_TYPES))]

    # y coords = bottom edge of each row (matplotlib y increases upward)
    y_h = 1.0 - rh_h          # bottom of header row
    y_q = y_h - rh_q          # bottom of question row
    y_m = [y_q - (i + 1) * rh_m for i in range(n_m)]  # bottom of each model row

    def rect(x, y, w, h, fc, ec='#999', lw=0.4):
        ax.add_patch(plt.Rectangle(
            (x + 0.001, y + 0.002), w - 0.002, h - 0.004,
            facecolor=fc, edgecolor=ec, linewidth=lw,
            transform=ax.transAxes, clip_on=False))

    def txt(x, y, w, h, s, fs=9, bold=False, color='#111'):
        ax.text(x + w / 2, y + h / 2, s,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=fs, color=color,
                fontweight='bold' if bold else 'normal',
                multialignment='center', clip_on=False,
                linespacing=1.3)

    # Header row
    rect(0, y_h, 1.0, rh_h, header_color, ec=header_color)
    txt(col_x[0], y_h, w0, rh_h, ds_label, fs=9, bold=True, color='white')
    for ci, lbl in enumerate(CT_LABELS):
        txt(col_x[ci+1], y_h, wct, rh_h, lbl, fs=9, bold=True, color='white')

    # Question text row — full text, wrapped at ~28 chars per line
    rect(0, y_q, 1.0, rh_q, '#F2F2F2', ec='#bbb')
    txt(col_x[0], y_q, w0, rh_q, 'Q:', fs=8, bold=True, color='#555')
    for ci, ct in enumerate(CONTROL_TYPES):
        q       = ctrl.get(ct, '—')
        wrapped = '\n'.join(textwrap.wrap(q, width=28))
        txt(col_x[ci+1], y_q, wct, rh_q, wrapped, fs=8, color='#222')

    # Model answer rows
    for ri, (model, short) in enumerate(model_dict.items()):
        ry   = y_m[ri]
        bg_l = '#F0F0F0' if ri % 2 == 0 else '#E8E8E8'
        rect(col_x[0], ry, w0, rh_m, bg_l, ec='#bbb')
        txt(col_x[0], ry, w0, rh_m, short, fs=9, bold=True,
            color=MODEL_COLORS.get(short, '#333'))
        for ci, ct in enumerate(CONTROL_TYPES):
            ans = ans_data.get((model, dataset, qid, ct), '—')
            bg  = cell_bg(ans, gt)
            rect(col_x[ci+1], ry, wct, rh_m, bg)
            txt(col_x[ci+1], ry, wct, rh_m, ans, fs=9)

def draw_conf_plot(ax, qid, model_dict, dataset, ds_label, header_color):
    """Confidence degradation curves, one line per model, hued by model."""
    all_vals = []
    for model, short in model_dict.items():
        vals = [conf_data.get((model, dataset, qid, ct), np.nan) for ct in CONTROL_TYPES]
        valid = [v for v in vals if not np.isnan(v)]
        if not valid:
            continue
        all_vals.extend(valid)
        color = MODEL_COLORS.get(short, '#888')
        ax.plot(range(len(CONTROL_TYPES)), vals,
                color=color, lw=1.6, marker='o', ms=3.5,
                label=short, zorder=3)

    ax.set_xticks(range(len(CONTROL_TYPES)))
    ax.set_xticklabels(CT_LABELS, fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylabel('Conf.', fontsize=7.5, labelpad=2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25, lw=0.5)

    if all_vals:
        lo = max(0.0, min(all_vals) - 0.04)
        ax.set_ylim(lo, min(1.0, max(all_vals) + 0.04))

    ax.set_title(ds_label, fontsize=8, fontweight='bold',
                 color='white', pad=3,
                 bbox=dict(facecolor=header_color, edgecolor='none',
                           boxstyle='round,pad=0.25'))
    ax.legend(fontsize=6, loc='lower left', framealpha=0.7,
              handlelength=1.2, borderpad=0.3, labelspacing=0.15,
              handletextpad=0.4, ncol=2)

# ── Generate figures ──────────────────────────────────────────────────────────
print(f"Generating {len(all_sampled)} figures...")
for i, qid in enumerate(sorted(all_sampled)):
    tiers    = qid_tiers[qid]
    ctrl     = qid_to_ctrl.get(qid, {})
    gt       = mapper.get_answers(qid)
    gt_str   = ', '.join(sorted(set(gt))) if gt else '?'
    base_q   = ctrl.get('question', '?')
    image_id = ctrl.get('image_id')
    is_human = qid in human_qids

    # ── Title (1 line) ────────────────────────────────────────────────────────
    if len(tiers) == 1:
        tier_part = f'Tier {tiers} — {tier_descs[tiers]}'
    else:
        tier_part = f'Tiers {", ".join(tiers)}'
    star      = '  ★ prev.study' if is_human else ''
    q_short   = (base_q[:65] + '…') if len(base_q) > 65 else base_q
    title     = f'{tier_part}{star}  ·  "{q_short}"  (GT: {gt_str})'
    t_color   = tier_colors.get(tiers[0], '#333')

    # ── Figure ────────────────────────────────────────────────────────────────
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(18, 9.5))
    fig.patch.set_facecolor('#F8F8F8')

    # Outer grid: left col (plots+image) | right col (tables)
    gs = GridSpec(1, 2, figure=fig,
                  width_ratios=[0.22, 0.78],
                  wspace=0.03,
                  left=0.01, right=0.99, top=0.93, bottom=0.02)

    # Left: image on top + 3 confidence plots
    gs_left = GridSpecFromSubplotSpec(
        4, 1, subplot_spec=gs[0, 0],
        height_ratios=[1.1, 1, 1, 1], hspace=0.18)
    ax_img = fig.add_subplot(gs_left[0])
    ax_p   = [fig.add_subplot(gs_left[r + 1]) for r in range(3)]

    # Right: 3 condition tables
    gs_right = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs[0, 1], hspace=0.06)
    ax_t = [fig.add_subplot(gs_right[r]) for r in range(3)]

    fig.text(0.5, 0.975, title, ha='center', va='top',
             fontsize=8.5, fontweight='bold', color=t_color)

    # Image
    ax_img.axis('off')
    img_path = os.path.join(IMAGES_DIR, f'COCO_val2014_{image_id:012d}.jpg')
    if os.path.exists(img_path):
        ax_img.imshow(Image.open(img_path))
    else:
        ax_img.set_facecolor('#CCC')
        ax_img.text(0.5, 0.5, 'no img', ha='center', va='center',
                    transform=ax_img.transAxes, fontsize=7)

    # Left col: confidence plots per condition
    for ax, (ds, label, mdict, hcolor) in zip(ax_p, CONDITIONS):
        draw_conf_plot(ax, qid, mdict, ds, label, hcolor)

    # Right col: answer tables per condition
    for ax, (ds, label, mdict, hcolor) in zip(ax_t, CONDITIONS):
        draw_table(ax, qid, mdict, ds, label, hcolor)

    # Legend
    fig.legend(
        handles=[mpatches.Patch(color='#A8D8A8', label='Correct'),
                 mpatches.Patch(color='#F4AAAA', label='Wrong'),
                 mpatches.Patch(color='#DEDEDE', label='N/A')],
        loc='lower right', fontsize=7, ncol=3, framealpha=0.8,
        bbox_to_anchor=(0.99, 0.005), borderpad=0.4
    )

    out_path = os.path.join(OUT_DIR, f'Q{qid}_{tiers}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#F8F8F8')
    plt.close(fig)

    if (i + 1) % 20 == 0 or (i + 1) == len(all_sampled):
        print(f'  {i+1}/{len(all_sampled)}')

print(f'\nDone. {len(os.listdir(OUT_DIR))} figures in {OUT_DIR}')
