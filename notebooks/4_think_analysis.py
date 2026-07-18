"""
4_think_analysis.py — Analysis of Qwen3 think-mode inference vs. human difficulty / HM alignment.
"""

import matplotlib
matplotlib.use('Agg')

import json
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path('/home/david/Desktop/yuna/HPA')
LOGITS_BASE = ROOT / 'evaluation/logits/backbone/pretrained'
PAIR_CACHE  = ROOT / 'analysis/session2/exports/pair_cache_cleaned.parquet'
RESP_CSV    = ROOT / 'analysis/session2/exports/responses_model_inst_blind.csv'
OUT_DIR     = ROOT / 'figures/think_eda'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────────────
MODEL_DIRS = {
    'Qwen3-0.6B_think': 'Qwen3-0.6B (think)',
    'Qwen3-1.7B_think': 'Qwen3-1.7B (think)',
    'Qwen3-4B_think':   'Qwen3-4B (think)',
    'Qwen3-8B_think':   'Qwen3-8B (think)',
    'Qwen3-32B_think':  'Qwen3-32B (think)',
}
VARIANT_MAP = {
    'question':       'C',
    'weaker_object':  'B',
    'pronominalized': 'A',
}
VARIANT_ORDER = ['C', 'B', 'A']
COLORS = {'C': '#2196F3', 'B': '#FF9800', 'A': '#4CAF50'}

THINK_RE = re.compile(r'<think>(.*?)</think>(.*)', re.DOTALL)

DPI = 220

# ── Helper ──────────────────────────────────────────────────────────────────
def parse_record(record, model_name):
    rows = []
    for vkey, vlabel in VARIANT_MAP.items():
        ans_text = record.get('generated_answers', {}).get(vkey, '')
        logits_obj = record.get('generated_logits', {}).get(vkey, {})
        logit_content = logits_obj.get('content', []) if isinstance(logits_obj, dict) else []

        m = THINK_RE.search(ans_text)
        if m:
            think_text  = m.group(1)
            final_text  = m.group(2).strip()
        else:
            think_text  = ''
            final_text  = ans_text.strip()

        think_words    = len(think_text.split()) if think_text.strip() else 0
        is_empty_think = think_text.strip() == ''

        # Split logprobs into think vs final portions
        # Find </think> token boundary in logit_content
        think_lps, final_lps = [], []
        in_think = True
        assembled = ''
        for tok in logit_content:
            lp = tok.get('logprob', None)
            assembled += tok.get('token', '')
            if '</think>' in assembled:
                in_think = False
            if in_think and lp is not None:
                think_lps.append(lp)
            elif not in_think and lp is not None:
                final_lps.append(lp)

        mean_think_lp = float(np.mean(think_lps)) if think_lps else np.nan
        mean_final_lp = float(np.mean(final_lps)) if final_lps else np.nan

        rows.append({
            'model_name':         model_name,
            'question_id':        record['question_id'],
            'variant':            vlabel,
            'think_words':        think_words,
            'is_empty_think':     is_empty_think,
            'final_answer':       final_text,
            'mean_think_logprob': mean_think_lp,
            'mean_final_logprob': mean_final_lp,
        })
    return rows


# ════════════════════════════════════════════════════════════════════════════
# 1. Build think_df
# ════════════════════════════════════════════════════════════════════════════
print("=== 1. Parsing think JSONL files ===")
all_rows = []
for model_dir, model_name in MODEL_DIRS.items():
    fpath = LOGITS_BASE / model_dir / 'vqa_1k_control_inst_blind.jsonl'
    if not fpath.exists():
        print(f"  WARNING: {fpath} not found, skipping.")
        continue
    count = 0
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            all_rows.extend(parse_record(rec, model_name))
            count += 1
    print(f"  {model_name}: {count} records parsed")

think_df = pd.DataFrame(all_rows)
print(f"\nthink_df shape: {think_df.shape}")
print(think_df.head(3).to_string())


# ════════════════════════════════════════════════════════════════════════════
# 2. Load pair_cache → HH difficulty + HM SBERT
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Loading pair_cache ===")
pc = pd.read_parquet(PAIR_CACHE)

# HH difficulty: mean sbert_score per (question_id, variant)
hh = (
    pc[pc.pair_type == 'HH']
    .groupby(['question_id', 'variant'])['sbert_score']
    .mean()
    .reset_index()
    .rename(columns={'sbert_score': 'hh_sbert'})
)
print(f"HH difficulty rows: {len(hh)}")

# HM SBERT for think models specifically
THINK_MODEL_NAMES = list(MODEL_DIRS.values())
hm_think = (
    pc[(pc.pair_type == 'HM') & (pc.subject_2.isin(THINK_MODEL_NAMES))]
    .groupby(['subject_2', 'question_id', 'variant'])['sbert_score']
    .mean()
    .reset_index()
    .rename(columns={'subject_2': 'model_name', 'sbert_score': 'hm_sbert'})
)
print(f"HM-think rows: {len(hm_think)}")

# HM SBERT for no-think counterparts (e.g., "Qwen3-4B")
NOTHINK_MAP = {
    'Qwen3-0.6B (think)': 'Qwen3-0.6B',
    'Qwen3-1.7B (think)': 'Qwen3-1.7B',
    'Qwen3-4B (think)':   'Qwen3-4B',
    'Qwen3-8B (think)':   'Qwen3-8B',
    'Qwen3-32B (think)':  'Qwen3-32B',
}
NOTHINK_NAMES = list(NOTHINK_MAP.values())
hm_nothink = (
    pc[(pc.pair_type == 'HM') & (pc.subject_2.isin(NOTHINK_NAMES))]
    .groupby(['subject_2', 'question_id', 'variant'])['sbert_score']
    .mean()
    .reset_index()
    .rename(columns={'subject_2': 'model_name', 'sbert_score': 'hm_sbert_nothink'})
)
print(f"HM-nothink rows: {len(hm_nothink)}")

# Merge think_df with hh and hm_think
df = think_df.merge(hh, on=['question_id', 'variant'], how='left')
df = df.merge(hm_think, on=['model_name', 'question_id', 'variant'], how='left')

# op / ent from responses CSV
resp = pd.read_csv(RESP_CSV)
op_ent = (
    resp[['question_id', 'variant', 'op', 'ent']]
    .drop_duplicates(['question_id', 'variant'])
)
df = df.merge(op_ent, on=['question_id', 'variant'], how='left')

print(f"\nMerged df shape: {df.shape}")
print(df.dtypes)
print(df.head(3).to_string())


# ════════════════════════════════════════════════════════════════════════════
# 3. Key analyses
# ════════════════════════════════════════════════════════════════════════════

def pearson_spearman(x, y, label=''):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return dict(n=n, r=np.nan, r_p=np.nan, rho=np.nan, rho_p=np.nan)
    r, r_p = stats.pearsonr(x, y)
    rho, rho_p = stats.spearmanr(x, y)
    return dict(n=n, r=r, r_p=r_p, rho=rho, rho_p=rho_p)

summary_rows = []

# ── A. Think length vs HH difficulty ────────────────────────────────────────
print("\n=== A. Think length vs HH difficulty (Pearson r / Spearman ρ) ===")
print(f"{'Model':<25} {'Variant':<8} {'n':>6} {'r':>7} {'p(r)':>8} {'ρ':>7} {'p(ρ)':>8}")
sub_nonempty = df[~df.is_empty_think].copy()
for mname in THINK_MODEL_NAMES:
    for v in VARIANT_ORDER:
        subset = sub_nonempty[(sub_nonempty.model_name == mname) & (sub_nonempty.variant == v)]
        res = pearson_spearman(subset.think_words.values.astype(float),
                               subset.hh_sbert.values.astype(float))
        print(f"  {mname:<25} {v:<8} {res['n']:>6} {res['r']:>7.3f} {res['r_p']:>8.4f} "
              f"{res['rho']:>7.3f} {res['rho_p']:>8.4f}")
        summary_rows.append({'analysis': 'A_think_vs_hh', 'model': mname, 'variant': v, **res})

# ── B. Think length vs HM SBERT ─────────────────────────────────────────────
print("\n=== B. Think length vs HM SBERT (Pearson r) ===")
print(f"{'Model':<25} {'Variant':<8} {'n':>6} {'r':>7} {'p(r)':>8}")
for mname in THINK_MODEL_NAMES:
    for v in VARIANT_ORDER:
        subset = sub_nonempty[(sub_nonempty.model_name == mname) & (sub_nonempty.variant == v)]
        res = pearson_spearman(subset.think_words.values.astype(float),
                               subset.hm_sbert.values.astype(float))
        print(f"  {mname:<25} {v:<8} {res['n']:>6} {res['r']:>7.3f} {res['r_p']:>8.4f}")
        summary_rows.append({'analysis': 'B_think_vs_hm', 'model': mname, 'variant': v, **res})

# ── C. Think vs no-think HM SBERT (paired t-test) ───────────────────────────
print("\n=== C. Think vs No-think HM SBERT (paired t-test) ===")
print(f"{'Model':<25} {'Variant':<8} {'n':>6} {'mean_think':>11} {'mean_nothink':>13} {'delta':>8} {'p':>8}")
paired_rows = []
for think_name, nothink_name in NOTHINK_MAP.items():
    for v in VARIANT_ORDER:
        think_hm  = hm_think[(hm_think.model_name == think_name) & (hm_think.variant == v)]
        nothink_hm = hm_nothink[(hm_nothink.model_name == nothink_name) & (hm_nothink.variant == v)]
        # Merge on question_id
        merged = think_hm.merge(nothink_hm[['question_id','hm_sbert_nothink']], on='question_id')
        n = len(merged)
        if n < 5:
            print(f"  {think_name:<25} {v:<8} {n:>6}  (insufficient data)")
            continue
        t, p = stats.ttest_rel(merged.hm_sbert.values, merged.hm_sbert_nothink.values)
        mt  = merged.hm_sbert.mean()
        mnt = merged.hm_sbert_nothink.mean()
        delta = mt - mnt
        print(f"  {think_name:<25} {v:<8} {n:>6} {mt:>11.4f} {mnt:>13.4f} {delta:>8.4f} {p:>8.4f}")
        paired_rows.append({'think_model': think_name, 'nothink_model': nothink_name,
                            'variant': v, 'n': n, 'mean_think': mt, 'mean_nothink': mnt,
                            'delta': delta, 't': t, 'p': p})
        summary_rows.append({'analysis': 'C_think_vs_nothink', 'model': think_name,
                             'variant': v, 'n': n, 'r': delta, 'r_p': p})

paired_df = pd.DataFrame(paired_rows)

# ── D. Think length by operation type ────────────────────────────────────────
print("\n=== D. Think length by op type (Spearman ρ with mean HH sbert) ===")
for mname in THINK_MODEL_NAMES:
    sub = df[df.model_name == mname].copy()
    op_tw  = sub.groupby('op')['think_words'].mean()
    op_hh  = sub.groupby('op')['hh_sbert'].mean()
    ops    = op_tw.index.intersection(op_hh.index)
    if len(ops) < 3:
        print(f"  {mname}: insufficient ops")
        continue
    rho, p = stats.spearmanr(op_tw[ops].values, op_hh[ops].values)
    print(f"  {mname:<25}  ρ={rho:.3f}  p={p:.4f}  (n_ops={len(ops)})")
    # Print per-op table
    op_tbl = pd.DataFrame({'mean_think_words': op_tw, 'mean_hh_sbert': op_hh}).loc[ops]
    op_tbl = op_tbl.sort_values('mean_hh_sbert', ascending=False)
    print(op_tbl.to_string())
    summary_rows.append({'analysis': 'D_op_corr', 'model': mname, 'variant': 'all',
                         'n': len(ops), 'rho': rho, 'rho_p': p})

# ── E. Empty think rate by variant ───────────────────────────────────────────
print("\n=== E. Empty think rate by model × variant ===")
empty_rate = (
    df.groupby(['model_name', 'variant'])['is_empty_think']
    .mean()
    .mul(100)
    .reset_index()
    .rename(columns={'is_empty_think': 'empty_pct'})
)
print(empty_rate.pivot(index='model_name', columns='variant', values='empty_pct').to_string())


# ════════════════════════════════════════════════════════════════════════════
# 4. Figures
# ════════════════════════════════════════════════════════════════════════════

MODEL_SHORT = {
    'Qwen3-0.6B (think)': '0.6B',
    'Qwen3-1.7B (think)': '1.7B',
    'Qwen3-4B (think)':   '4B',
    'Qwen3-8B (think)':   '8B',
    'Qwen3-32B (think)':  '32B',
}

# ── Figure B: think_hm_vs_nothink_hm ─────────────────────────────────────────
print("=== Saving Figure B ===")
# Compute mean HM sbert per model×variant for think and no-think
think_mean = hm_think.groupby(['model_name', 'variant'])['hm_sbert'].mean().reset_index()
nothink_mean = hm_nothink.groupby(['model_name', 'variant'])['hm_sbert_nothink'].mean().reset_index()

fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
x = np.arange(len(VARIANT_ORDER))
width = 0.35
for ax, (think_name, nothink_name) in zip(axes, NOTHINK_MAP.items()):
    th_vals  = [think_mean[(think_mean.model_name == think_name) &
                             (think_mean.variant == v)]['hm_sbert'].values[0]
                if len(think_mean[(think_mean.model_name == think_name) &
                                  (think_mean.variant == v)]) > 0 else np.nan
                for v in VARIANT_ORDER]
    nth_vals = [nothink_mean[(nothink_mean.model_name == nothink_name) &
                              (nothink_mean.variant == v)]['hm_sbert_nothink'].values[0]
                if len(nothink_mean[(nothink_mean.model_name == nothink_name) &
                                    (nothink_mean.variant == v)]) > 0 else np.nan
                for v in VARIANT_ORDER]
    b1 = ax.bar(x - width/2, th_vals,  width, label='Think',    color='#5C85D6', alpha=0.85)
    b2 = ax.bar(x + width/2, nth_vals, width, label='No-think',  color='#D6855C', alpha=0.85)
    # Delta labels
    for xi, (t, nt) in enumerate(zip(th_vals, nth_vals)):
        if not (np.isnan(t) or np.isnan(nt)):
            delta = t - nt
            ymax = max(t, nt)
            ax.text(xi, ymax + 0.005, f'{delta:+.3f}', ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANT_ORDER)
    ax.set_title(MODEL_SHORT[think_name], fontsize=9)
    ax.set_xlabel('Variant', fontsize=8)
    ax.set_ylabel('Mean HM SBERT', fontsize=8)
    ax.tick_params(labelsize=7)

axes[0].legend(fontsize=8)
fig.suptitle('HM SBERT: Think vs No-Think', fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT_DIR / 'think_hm_vs_nothink_hm.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  Saved think_hm_vs_nothink_hm.png")



# ════════════════════════════════════════════════════════════════════════════
# 5. Summary CSV
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 5. Summary stats ===")

# Overall think word stats per model
overall_stats = (
    df.groupby('model_name')
    .agg(
        total_qs=('question_id', 'nunique'),
        mean_think_words=('think_words', 'mean'),
        median_think_words=('think_words', 'median'),
        empty_think_pct=('is_empty_think', lambda x: x.mean()*100),
        mean_hm_sbert=('hm_sbert', 'mean'),
        mean_hh_sbert=('hh_sbert', 'mean'),
    )
    .reset_index()
)
print("\nOverall stats per model:")
print(overall_stats.to_string(index=False))

# Persist
summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / 'think_analysis_results.csv'
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary CSV saved to {summary_path}")

# Also save overall stats
overall_path = OUT_DIR / 'think_overall_stats.csv'
overall_stats.to_csv(overall_path, index=False)
print(f"Overall stats CSV saved to {overall_path}")

print("\n=== Done ===")
