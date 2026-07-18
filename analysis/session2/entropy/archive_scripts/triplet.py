"""
Generate matched-triplet figure: VLM → Backbone → Base LLM.

For each VLM architecture, shows HM SBERT alignment at three derivation
levels (base LLM, backbone decoder, full VLM) with variant A/B/C spread
as error bars. This controls for architecture: the only variable is the
model derivation pathway.

Triplets (7/8B):
  Qwen3-VL-8B      → Qwen3-VL-8B (LM)      → Qwen3-8B
  InternVL-8B       → InternVL-8B (LM)       → Qwen3-8B  (shared base)
  LLaVA-1.5-7B      → LLaVA-1.5 (LM)        → Vicuna-7B
  LLaVA-Mistral     → LLaVA-Mistral (LM)     → Mistral-7B
  LLaVA-Vicuna      → LLaVA-Vicuna (LM)      → Vicuna-7B (shared base)

Run from repo root:
  conda run -n zero python analysis/session2/entropy/archive_scripts/triplet.py
"""

import sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy as sp_entropy

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from figures.helpers import save_fig
from utils.constants import GROUP_COLORS, VARIANT_ORDER
from config import MODELS_7B, VLM_BASE_LLM

EXPORTS = ROOT / 'analysis/session2/exports'
OUT_DIR = ROOT / 'analysis/session2/entropy/exports/figures'
LATEX_FIG = ROOT / 'latex/AAAI2026/LaTeX/figures'

# ── Matched triplets (7/8B only) ─────────────────────────────────────────────
TRIPLETS = [
    {'label': 'Qwen3-VL-8B',
     'VLM': 'Qwen3-VL-8B', 'Backbone': 'Qwen3-VL-8B (LM)', 'Base LLM': 'Qwen3-8B'},
    {'label': 'InternVL-8B',
     'VLM': 'InternVL-8B', 'Backbone': 'InternVL-8B (LM)', 'Base LLM': 'Qwen3-8B'},
    {'label': 'LLaVA-1.5',
     'VLM': 'LLaVA-1.5-7B', 'Backbone': 'LLaVA-1.5 (LM)', 'Base LLM': 'Vicuna-7B'},
    {'label': 'LLaVA-Mistral',
     'VLM': 'LLaVA-Mistral', 'Backbone': 'LLaVA-Mistral (LM)', 'Base LLM': 'Mistral-7B'},
    {'label': 'LLaVA-Vicuna',
     'VLM': 'LLaVA-Vicuna', 'Backbone': 'LLaVA-Vicuna (LM)', 'Base LLM': 'Vicuna-7B'},
]

STAGE_ORDER = ['Base LLM', 'Backbone', 'VLM']
STAGE_COLORS = {
    'VLM': GROUP_COLORS['VLM'],
    'Backbone': GROUP_COLORS['VLM backbone decoder'],
    'Base LLM': '#2E7D32',
}

TRIPLET_COLORS = {
    'Qwen3-VL-8B': '#E53935',
    'InternVL-8B': '#9C27B0',
    'LLaVA-1.5': '#FF9800',
    'LLaVA-Mistral': '#2196F3',
    'LLaVA-Vicuna': '#4CAF50',
}

# ── Load data ─────────────────────────────────────────────────────────────────
pc = pd.read_parquet(EXPORTS / 'pair_cache_raw.parquet')
human = pd.read_csv(EXPORTS / 'responses_human.csv')

hm_pairs = pc[pc['pair_type'] == 'HM']
hh_pairs = pc[pc['pair_type'] == 'HH']

# ── Per-model, per-variant, per-question HM SBERT ────────────────────────────
model_hm = hm_pairs.groupby(
    ['question_id', 'variant', 'subject_2']
)['sbert_score'].mean().reset_index()
model_hm.columns = ['question_id', 'variant', 'model', 'hm_sbert']

# ── HH baseline per variant ──────────────────────────────────────────────────
hh_by_variant = hh_pairs.groupby('variant')['sbert_score'].mean()
hh_mean_all = hh_pairs['sbert_score'].mean()

# ── Build triplet data: per-triplet, per-stage, per-variant mean HM SBERT ────
triplet_rows = []
for tri in TRIPLETS:
    for stage in STAGE_ORDER:
        model_name = tri[stage]
        for v in VARIANT_ORDER:
            sub = model_hm[(model_hm['model'] == model_name) & (model_hm['variant'] == v)]
            if len(sub) > 0:
                triplet_rows.append({
                    'triplet': tri['label'],
                    'stage': stage,
                    'model': model_name,
                    'variant': v,
                    'hm_sbert': sub['hm_sbert'].mean(),
                })

tri_df = pd.DataFrame(triplet_rows)

print(f'Triplet data: {len(tri_df)} rows')
print(tri_df.groupby(['triplet', 'stage', 'variant'])['hm_sbert'].mean().unstack('variant').round(3))

# ── Human entropy (for panel 2) ──────────────────────────────────────────────
def answer_entropy(responses):
    counts = responses['response'].value_counts()
    probs = counts / counts.sum()
    return sp_entropy(probs, base=2)

h_ent = {}
for v in VARIANT_ORDER:
    h_ent[v] = human[human['variant'] == v].groupby('question_id').apply(
        answer_entropy, include_groups=False
    )

# Per-model per-variant per-question HM SBERT (for panel 2 scatter)
model_hm_q = model_hm.copy()

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ═════════════════════════════════════════════════════════════════════════════
# Figure: 2-panel matched triplet analysis
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Panel 1: HM SBERT trajectory per triplet, variant error bars ─────────────
ax = axes[0]
stage_x = {s: i for i, s in enumerate(STAGE_ORDER)}

for tri_label in [t['label'] for t in TRIPLETS]:
    means = []
    lows = []
    highs = []
    for stage in STAGE_ORDER:
        sub = tri_df[(tri_df['triplet'] == tri_label) & (tri_df['stage'] == stage)]
        variant_means = sub.groupby('variant')['hm_sbert'].mean()
        m = variant_means.mean()
        means.append(m)
        lows.append(m - variant_means.min())
        highs.append(variant_means.max() - m)

    color = TRIPLET_COLORS[tri_label]
    xs = [stage_x[s] for s in STAGE_ORDER]
    ax.errorbar(xs, means, yerr=[lows, highs], fmt='-o', color=color,
                lw=2, markersize=7, alpha=0.85, capsize=4, capthick=1.5,
                label=tri_label)

# HH baseline with variant spread
hh_vals = [hh_by_variant.get(v, np.nan) for v in VARIANT_ORDER]
hh_m = np.mean(hh_vals)
ax.axhline(hh_m, color='black', ls='--', alpha=0.4, lw=1)
ax.axhspan(min(hh_vals), max(hh_vals), alpha=0.06, color='black')
ax.text(2.15, hh_m, 'HH', fontsize=8, va='center', alpha=0.5)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(STAGE_ORDER, fontsize=10)
ax.set_ylabel('Mean HM SBERT', fontsize=10)
ax.set_title('Alignment Trajectory\n(error bars = variant A/B/C spread)', fontsize=10)
ax.legend(fontsize=7.5, loc='lower right')

# ── Panel 2: Human entropy vs HM SBERT, per stage (variant C) ───────────────
ax = axes[1]

# For each stage, pool all triplet models at that stage and plot per-question scatter
for stage in STAGE_ORDER:
    stage_models = [tri[stage] for tri in TRIPLETS]
    # Get unique models (some share base LLM)
    unique_models = list(set(stage_models))
    sub = model_hm_q[(model_hm_q['model'].isin(unique_models)) & (model_hm_q['variant'] == 'C')]
    # Mean HM SBERT per question across models at this stage
    q_mean = sub.groupby('question_id')['hm_sbert'].mean()
    # Join with human entropy
    h_ent_c = h_ent['C']
    common = q_mean.index.intersection(h_ent_c.index)
    x = h_ent_c.loc[common]
    y = q_mean.loc[common]
    r, p = stats.pearsonr(x, y)
    color = STAGE_COLORS[stage]
    label = stage
    ax.scatter(x, y, s=20, alpha=0.45, color=color, edgecolors='white', lw=0.3,
               label='{} (r={:.2f})'.format(label, r))

ax.set_xlabel('Human answer entropy (bits)', fontsize=10)
ax.set_ylabel('HM SBERT (variant C)', fontsize=10)
ax.set_title('Human Uncertainty vs Alignment\nby Derivation Stage', fontsize=10)
ax.legend(fontsize=8, loc='upper right')

plt.suptitle('Matched-Triplet Analysis: Base LLM → Backbone → VLM (7/8B)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = OUT_DIR / 'entropy_alignment_triplet.png'
save_fig(fig, OUT_DIR, 'entropy_alignment_triplet.png')
plt.close(fig)

# Copy to latex figures
latex_dest = LATEX_FIG / 'entropy_alignment_triplet.png'
shutil.copy2(out_path, latex_dest)
print(f'Copied to: {latex_dest}')

# ── Summary stats ─────────────────────────────────────────────────────────────
print('\n=== Mean HM SBERT by stage per triplet (mean across variants) ===')
for tri_label in [t['label'] for t in TRIPLETS]:
    parts = []
    for stage in STAGE_ORDER:
        sub = tri_df[(tri_df['triplet'] == tri_label) & (tri_df['stage'] == stage)]
        variant_means = sub.groupby('variant')['hm_sbert'].mean()
        m = variant_means.mean()
        lo, hi = variant_means.min(), variant_means.max()
        parts.append('{}: {:.3f} [{:.3f}, {:.3f}]'.format(stage, m, lo, hi))
    print('  {} — {}'.format(tri_label, ', '.join(parts)))

print('\n=== Per-stage Pearson r (human entropy vs HM SBERT, variant C) ===')
for stage in STAGE_ORDER:
    stage_models = list(set([tri[stage] for tri in TRIPLETS]))
    sub = model_hm_q[(model_hm_q['model'].isin(stage_models)) & (model_hm_q['variant'] == 'C')]
    q_mean = sub.groupby('question_id')['hm_sbert'].mean()
    h_ent_c = h_ent['C']
    common = q_mean.index.intersection(h_ent_c.index)
    r, p = stats.pearsonr(h_ent_c.loc[common], q_mean.loc[common])
    print('  {}: r={:.3f}, p={:.1e}, N={}'.format(stage, r, p, len(common)))

print(f'\nDone. Output → {out_path}')
