"""
EDA: Qwen3 backbone think model analysis.

Analyzes thinking token lengths, final answer characteristics,
confidence (logprob) distributions, and CoT behavior patterns.

Run from repo root:
  conda run -n zero python analysis/think_model_eda.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'analysis'))

THINK_RE = re.compile(r'<think>(.*?)</think>\s*', re.DOTALL)
VARIANT_MAP = {'question': 'C', 'weaker_object': 'B', 'pronominalized': 'A'}

OUT_DIR = ROOT / 'figures' / 'think_eda'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    'Qwen3-0.6B (think)': 'evaluation/logits/backbone/pretrained/Qwen3-0.6B_think/vqa_1k_control_inst_blind.jsonl',
    'Qwen3-1.7B (think)': 'evaluation/logits/backbone/pretrained/Qwen3-1.7B_think/vqa_1k_control_inst_blind.jsonl',
    'Qwen3-4B (think)':   'evaluation/logits/backbone/pretrained/Qwen3-4B_think/vqa_1k_control_inst_blind.jsonl',
    'Qwen3-8B (think)':   'evaluation/logits/backbone/pretrained/Qwen3-8B_think/vqa_1k_control_inst_blind.jsonl',
    'Qwen3-32B (think)':  'evaluation/logits/backbone/pretrained/Qwen3-32B_think/vqa_1k_control_inst_blind.jsonl',
}

NOTHINK_MODELS = {
    'Qwen3-0.6B': 'evaluation/logits/backbone/pretrained/Qwen3-0.6B/vqa_1k_control_inst_blind.jsonl',
    'Qwen3-1.7B': 'evaluation/logits/backbone/pretrained/Qwen3-1.7B/vqa_1k_control_inst_blind.jsonl',
    'Qwen3-4B':   'evaluation/logits/backbone/pretrained/Qwen3-4B/vqa_1k_control_inst_blind.jsonl',
    'Qwen3-8B':   'evaluation/logits/backbone/pretrained/Qwen3-8B/vqa_1k_control_inst_blind.jsonl',
    'Qwen3-32B':  'evaluation/logits/backbone/pretrained/Qwen3-32B/vqa_1k_control_inst_blind.jsonl',
}

# Color ramp by size
import matplotlib.cm as cm
_sizes = [0.6, 1.7, 4.0, 8.0, 32.0]
_cmap = cm.get_cmap('plasma', len(_sizes) + 2)
COLORS = {f'Qwen3-{s}B (think)': _cmap(i + 1) for i, s in enumerate([0.6, 1.7, 4, 8, 32])}


def display_model_name(model: str) -> str:
    return model.replace(' (think)', '')


def strip_think(text: str) -> str:
    return THINK_RE.sub('', text).strip()


def get_think(text: str) -> str:
    m = THINK_RE.search(text)
    return m.group(1) if m else ''


def load_model_data(path: str) -> pd.DataFrame:
    rows = []
    with open(ROOT / path) as f:
        for line in f:
            row = json.loads(line)
            qid = row['question_id']
            answers = row.get('generated_answers', {})
            logits = row.get('generated_logits', {})
            for ct, ans in answers.items():
                variant = VARIANT_MAP.get(ct, ct)
                think_content = get_think(ans)
                final = strip_think(ans)
                has_closed_think = '</think>' in ans
                # Logprob: use first logit for the first generated token
                lp_dict = logits.get(ct, {})
                # generated_logits stores per-token dicts; take mean of all top-1 logprobs
                lp_values = []
                # Format: {'content': [{'token': ..., 'logprob': ..., 'top_logprobs': [...]}, ...]}
                token_list = lp_dict.get('content', []) if isinstance(lp_dict, dict) else []
                for tok in token_list:
                    if isinstance(tok, dict) and 'logprob' in tok:
                        lp_values.append(tok['logprob'])
                think_words = len(think_content.split()) if think_content.strip() else 0
                final_words = len(final.split()) if final.strip() else 0
                # Separate think-phase vs final-answer logprobs
                # Approximate by splitting token list at </think> token boundary
                think_lps, final_lps = [], []
                if lp_values and has_closed_think:
                    # Reconstruct token text to find boundary
                    token_list2 = lp_dict.get('content', []) if isinstance(lp_dict, dict) else []
                    in_think = True
                    for tok in token_list2:
                        tok_text = tok.get('token', '')
                        lp = tok.get('logprob', np.nan)
                        if in_think:
                            think_lps.append(lp)
                            if '</think>' in tok_text:
                                in_think = False
                        else:
                            final_lps.append(lp)
                rows.append({
                    'question_id': qid,
                    'variant': variant,
                    'think_words': think_words,
                    'final_words': final_words,
                    'has_closed_think': has_closed_think,
                    'empty_think': think_content.strip() == '',
                    'final_ans': final,
                    'raw': ans,
                    'mean_logprob': np.mean(lp_values) if lp_values else np.nan,
                    'mean_think_logprob': np.mean(think_lps) if think_lps else np.nan,
                    'mean_final_logprob': np.mean(final_lps) if final_lps else np.nan,
                    'n_logprob_tokens': len(lp_values),
                })
    return pd.DataFrame(rows)


# ── 1. Load data ─────────────────────────────────────────────────────────────
print('Loading model data...')
dfs = {}
for model, path in MODELS.items():
    df = load_model_data(path)
    df['model'] = model
    dfs[model] = df
    print(f'  {model}: {len(df)} rows, {df["question_id"].nunique()} questions')

all_df = pd.concat(dfs.values(), ignore_index=True)

# Load non-think counterparts for comparison
nothink_dfs = {}
for model, path in NOTHINK_MODELS.items():
    fpath = ROOT / path
    if not fpath.exists():
        print(f'  [skip] {model} no-think file not found')
        continue
    rows = []
    with open(fpath) as f:
        for line in f:
            row = json.loads(line)
            qid = row['question_id']
            for ct, ans in row.get('generated_answers', {}).items():
                var = VARIANT_MAP.get(ct, ct)
                lp_dict = row.get('generated_logits', {}).get(ct, {})
                lp_values = []
                token_list2 = lp_dict.get('content', []) if isinstance(lp_dict, dict) else []
                for tok in token_list2:
                    if isinstance(tok, dict) and 'logprob' in tok:
                        lp_values.append(tok['logprob'])
                rows.append({'question_id': qid, 'variant': var, 'final_ans': ans.strip(),
                             'mean_logprob': np.mean(lp_values) if lp_values else np.nan,
                             'model': f'{model} (no-think)'})
    nothink_dfs[model] = pd.DataFrame(rows)
    print(f'  {model} (no-think): {len(nothink_dfs[model])} rows')


# ── 2. Print EDA summary ─────────────────────────────────────────────────────
print('\n=== THINK LENGTH STATS ===')
for model, df in dfs.items():
    tw = df['think_words']
    print(f'\n{model}:')
    print(f'  think_words: mean={tw.mean():.0f}  median={tw.median():.0f}  '
          f'p5={tw.quantile(0.05):.0f}  p95={tw.quantile(0.95):.0f}  max={tw.max():.0f}')
    print(f'  empty_think (no reasoning): {df["empty_think"].sum()} ({df["empty_think"].mean():.1%})')
    print(f'  unclosed </think>: {(~df["has_closed_think"]).sum()} ({(~df["has_closed_think"]).mean():.1%})')
    print(f'  final_words: mean={df["final_words"].mean():.1f}  median={df["final_words"].median():.0f}')
    print(f'  By variant:')
    for var, g in df.groupby('variant'):
        print(f'    {var}: think_words mean={g["think_words"].mean():.0f}  '
              f'final_words mean={g["final_words"].mean():.1f}')

print('\n=== OP GROUP THINK LENGTH ===')
# Need op labels — load from pair_cache or responses
try:
    pc = pd.read_parquet(ROOT / 'analysis/session2/exports/pair_cache_cleaned.parquet')
    qid_op = pc[['question_id', 'op']].drop_duplicates().set_index('question_id')['op'].to_dict()
    qid_ent = pc[['question_id', 'ent']].drop_duplicates().set_index('question_id')['ent'].to_dict()
    all_df['op'] = all_df['question_id'].map(qid_op)
    all_df['ent'] = all_df['question_id'].map(qid_ent)
    print('Op groups loaded from pair_cache.')
    for model, df in dfs.items():
        df['op'] = df['question_id'].map(qid_op)
except Exception as e:
    print(f'  [warn] could not load op labels: {e}')
    qid_op = {}


# ── 3. Plot: Think length distribution ───────────────────────────────────────
n_models = len(dfs)
ncols = min(3, n_models)
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes_flat = np.array(axes).flatten()
bins = np.linspace(0, 3000, 61)
for ax, (model, df) in zip(axes_flat, dfs.items()):
    tw = df['think_words']
    ax.hist(tw, bins=bins, color=COLORS[model], alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.axvline(tw.mean(), color='red', linestyle='--', linewidth=1.2, label=f'mean={tw.mean():.0f}')
    ax.axvline(tw.median(), color='orange', linestyle='--', linewidth=1.2, label=f'median={tw.median():.0f}')
    ax.set_title(model, fontsize=10)
    ax.set_xlabel('Think token word count')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    empty_pct = df['empty_think'].mean() * 100
    ax.text(0.98, 0.95, f'Empty think: {empty_pct:.1f}%', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='gray')
for ax in axes_flat[n_models:]:
    ax.set_visible(False)
fig.suptitle('Distribution of thinking token lengths (word count)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / 'think_length_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: think_length_dist.png')


# ── 4. Plot: Think length by operation group ─────────────────────────────────
if qid_op:
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    axes_flat4 = np.array(axes).flatten()
    for ax, (model, df) in zip(axes_flat4, dfs.items()):
        df = df.copy()
        df['op'] = df['question_id'].map(qid_op)
        op_means = df.groupby('op')['think_words'].mean().sort_values(ascending=True)
        bars = ax.barh(op_means.index, op_means.values, color=COLORS[model], alpha=0.8)
        ax.set_xlabel('Mean think words')
        ax.set_title(f'{model}\nThink length by operation group', fontsize=10)
        for bar, val in zip(bars, op_means.values):
            ax.text(val + 5, bar.get_y() + bar.get_height() / 2, f'{val:.0f}',
                    va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'think_length_by_op.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: think_length_by_op.png')


# ── 5. Plot: Think length vs final answer length ─────────────────────────────
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
axes_flat5 = np.array(axes).flatten()
for ax, (model, df) in zip(axes_flat5, dfs.items()):
    # Remove outlier unclosed thinks for scatter
    mask = df['has_closed_think'] & (df['think_words'] < 3000) & (df['final_words'] < 50)
    x = df.loc[mask, 'think_words']
    y = df.loc[mask, 'final_words']
    ax.scatter(x, y, alpha=0.1, s=5, color=COLORS[model], rasterized=True)
    # Bin means
    bins_x = np.percentile(x, np.linspace(0, 100, 21))
    bin_means = []
    bin_centers = []
    for i in range(len(bins_x) - 1):
        m_mask = (x >= bins_x[i]) & (x < bins_x[i + 1])
        if m_mask.sum() > 0:
            bin_centers.append((bins_x[i] + bins_x[i + 1]) / 2)
            bin_means.append(y[m_mask].mean())
    ax.plot(bin_centers, bin_means, 'r-', linewidth=2, label='bin mean')
    r, p = stats.pearsonr(x, y)
    ax.set_xlabel('Think words')
    ax.set_ylabel('Final answer words')
    ax.set_title(f'{model}\nr={r:.3f} (p={p:.2e})', fontsize=10)
    ax.legend(fontsize=8)
plt.suptitle('Think length vs final answer length', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'think_vs_final_length.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: think_vs_final_length.png')


# ── 6. Plot: Think length by variant ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
variant_order = ['C', 'B', 'A']
x = np.arange(len(variant_order))
width = 0.35
for i, (model, df) in enumerate(dfs.items()):
    means = [df[df['variant'] == v]['think_words'].mean() for v in variant_order]
    sems = [df[df['variant'] == v]['think_words'].sem() for v in variant_order]
    ax.bar(x + (i - 0.5) * width, means, width, yerr=sems,
           label=model, color=COLORS[model], alpha=0.85, capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(['C (original)', 'B (weaker_obj)', 'A (pronominalized)'])
ax.set_ylabel('Mean think words')
ax.set_title('Think length by question variant')
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / 'think_length_by_variant.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: think_length_by_variant.png')


# ── 7. Plot: Logprob analysis ────────────────────────────────────────────────
# a) Think-phase vs final-answer logprob comparison (all models)
fig, axes = plt.subplots(
    2,
    n_models,
    figsize=(2.85 * n_models, 4.9),
    sharex='col',
    sharey='row',
)
for col_i, (model, df) in enumerate(dfs.items()):
    ax = axes[0, col_i]
    think_lp = df['mean_think_logprob'].dropna()
    final_lp = df['mean_final_logprob'].dropna()
    if len(think_lp) > 0:
        ax.hist(think_lp, bins=40, alpha=0.6, color='steelblue', label=f'think (n={len(think_lp)})')
    if len(final_lp) > 0:
        ax.hist(final_lp, bins=40, alpha=0.6, color='darkorange', label=f'answer (n={len(final_lp)})')
    ax.set_title(display_model_name(model), fontsize=9)
    ax.legend(fontsize=7)
    ax2 = axes[1, col_i]
    both = df[df['mean_think_logprob'].notna() & df['mean_final_logprob'].notna()]
    ax2.scatter(both['mean_think_logprob'], both['mean_final_logprob'],
                alpha=0.15, s=5, color=COLORS[model], rasterized=True)
    if col_i == 0:
        ax.set_ylabel('Count')
        ax2.set_ylabel('Answer span mean logprob')
    else:
        ax.set_ylabel('')
        ax2.set_ylabel('')
    ax2.set_title('', fontsize=9)
    if len(both) > 10:
        r, p = stats.pearsonr(both['mean_think_logprob'], both['mean_final_logprob'])
        ax2.text(0.05, 0.95, f'r={r:.3f}', transform=ax2.transAxes, fontsize=8, va='top')
    ax2.set_xlabel('Think span mean logprob')
for ax in axes[0, :]:
    ax.set_xlabel('')
plt.suptitle('Logprob (confidence): thinking vs final answer tokens', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.5, h_pad=0.7)
plt.savefig(OUT_DIR / 'confidence_think_vs_final.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: confidence_think_vs_final.png')

# b) Think vs no-think overall confidence comparison (all scales)
size_pairs = [
    ('0.6B', 'Qwen3-0.6B (think)', 'Qwen3-0.6B'),
    ('1.7B', 'Qwen3-1.7B (think)', 'Qwen3-1.7B'),
    ('4B',   'Qwen3-4B (think)',   'Qwen3-4B'),
    ('8B',   'Qwen3-8B (think)',   'Qwen3-8B'),
    ('32B',  'Qwen3-32B (think)',  'Qwen3-32B'),
]
fig, axes = plt.subplots(1, len(size_pairs), figsize=(4 * len(size_pairs), 4))
for ax, (size, think_model, base_model) in zip(np.array(axes).flatten(), size_pairs):
    think_lp = dfs.get(think_model, pd.DataFrame(columns=['mean_logprob']))['mean_logprob'].dropna()
    nothink_lp = nothink_dfs.get(base_model, pd.DataFrame(columns=['mean_logprob']))['mean_logprob'].dropna()
    if len(think_lp) > 0:
        ax.hist(think_lp, bins=40, alpha=0.65, color='steelblue', label=f'think {think_lp.mean():.3f}')
    if len(nothink_lp) > 0:
        ax.hist(nothink_lp, bins=40, alpha=0.65, color='coral', label=f'no-think {nothink_lp.mean():.3f}')
    ax.set_xlabel('Mean logprob (all tokens)')
    ax.set_title(f'Qwen3-{size}', fontsize=10)
    ax.legend(fontsize=7)
plt.suptitle('Think vs no-think: token logprob distribution by scale', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'confidence_think_vs_nothink.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: confidence_think_vs_nothink.png')


# ── 8. Print: Empty think and truncation stats ───────────────────────────────
print('\n=== TRUNCATION & EMPTY THINK SUMMARY ===')
for model, df in dfs.items():
    print(f'\n{model}:')
    print(f'  Total answers: {len(df)}')
    print(f'  Empty think (no reasoning): {df["empty_think"].sum()} ({df["empty_think"].mean():.1%})')
    print(f'  Truncated (no </think>): {(~df["has_closed_think"]).sum()} ({(~df["has_closed_think"]).mean():.1%})')
    # Truncated not empty = hit 4096 token limit mid-think
    trunc_nonempty = (~df['has_closed_think']) & (~df['empty_think'])
    print(f'  Truncated mid-think: {trunc_nonempty.sum()} ({trunc_nonempty.mean():.1%})')
    print(f'  Closed think (complete): {df["has_closed_think"].sum()} ({df["has_closed_think"].mean():.1%})')
    by_var = df.groupby('variant')['has_closed_think'].mean()
    print(f'  Completion rate by variant: {by_var.to_dict()}')


# ── 9. Export summary CSV ─────────────────────────────────────────────────────
summary_rows = []
for model, df in dfs.items():
    for var in ['C', 'B', 'A']:
        g = df[df['variant'] == var]
        summary_rows.append({
            'model': model, 'variant': var,
            'n': len(g),
            'think_mean': round(g['think_words'].mean(), 1),
            'think_median': round(g['think_words'].median(), 1),
            'think_p95': round(g['think_words'].quantile(0.95), 1),
            'think_max': int(g['think_words'].max()),
            'empty_think_pct': round(g['empty_think'].mean() * 100, 1),
            'truncated_pct': round((~g['has_closed_think']).mean() * 100, 1),
            'final_mean_words': round(g['final_words'].mean(), 2),
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / 'think_eda_summary.csv', index=False)
print('\nSaved: think_eda_summary.csv')
print(summary_df.to_string(index=False))


# ── 10. Plot: Think length distribution for op groups (box plot) ─────────────
if qid_op and 'op' in all_df.columns:
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    axes_flat10 = np.array(axes).flatten()
    for ax, (model, df) in zip(axes_flat10, dfs.items()):
        df = df.copy()
        df['op'] = df['question_id'].map(qid_op)
        df = df[df['has_closed_think']]  # only complete think traces
        op_order = df.groupby('op')['think_words'].median().sort_values(ascending=True).index
        data = [df[df['op'] == op]['think_words'].values for op in op_order]
        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2),
                        flierprops=dict(marker='.', markersize=2, alpha=0.3))
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.6)
        ax.set_yticks(range(1, len(op_order) + 1))
        ax.set_yticklabels(op_order)
        ax.set_xlabel('Think words')
        ax.set_title(f'{model}\nThink length by op (complete traces only)', fontsize=10)
        ax.set_xlim(0, 1500)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'think_length_boxplot_op.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: think_length_boxplot_op.png')

print(f'\nAll EDA outputs saved to: {OUT_DIR}')
