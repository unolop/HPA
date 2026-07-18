"""
Generate aggregated entropy statistics tables by entity group.

Run from repo root:
  conda run -n zero python analysis/session2/entropy/archive_scripts/entropy_tables.py
"""

import argparse
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import entropy as sp_entropy

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from config import MODELS_7B, extend_pair_cache_with_yesno
from utils.constants import VARIANT_ORDER

EXPORTS = ROOT / 'analysis/session2/exports'
_7b = set(MODELS_7B)

parser = argparse.ArgumentParser()
parser.add_argument('--free_text_only', action='store_true')
args = parser.parse_args()

pc = pd.read_parquet(EXPORTS / 'pair_cache_raw.parquet')
if not args.free_text_only:
    pc = extend_pair_cache_with_yesno(pc, EXPORTS)
human = pd.read_csv(EXPORTS / 'responses_human.csv')
model_ib = pd.read_csv(EXPORTS / 'responses_model_inst_blind.csv')

# Build per-question table
rows = []
for v in VARIANT_ORDER:
    sub = pc[pc['variant'] == v]
    hh = sub[sub['pair_type'] == 'HH'].groupby('question_id')['sbert_score'].mean()
    hm = sub[(sub['pair_type'] == 'HM') & (sub['subject_2'].isin(_7b))].groupby('question_id')['sbert_score'].mean()
    for qid in hh.index:
        rows.append({'question_id': qid, 'variant': v,
                     'hh_sbert': hh.get(qid, np.nan),
                     'hm_sbert': hm.get(qid, np.nan)})
qv = pd.DataFrame(rows)
meta = human[human['variant'] == 'C'].drop_duplicates('question_id')[
    ['question_id', 'ent', 'op']
].set_index('question_id')
qwide = qv.pivot(index='question_id', columns='variant', values=['hh_sbert', 'hm_sbert'])
qwide.columns = ['_'.join(c) for c in qwide.columns]
df = qwide.join(meta)
df['hm_drop_CA'] = df['hm_sbert_C'] - df['hm_sbert_A']

# Entropy
def answer_entropy(responses):
    counts = responses['response'].value_counts()
    probs = counts / counts.sum()
    return sp_entropy(probs, base=2)

h_ent = human[human['variant'] == 'C'].groupby('question_id').apply(
    answer_entropy, include_groups=False
).rename('human_entropy')
m_ent = model_ib[(model_ib['variant'] == 'C') & (model_ib['model'].isin(_7b))].groupby('question_id').apply(
    answer_entropy, include_groups=False
).rename('model_entropy')

ent_df = pd.DataFrame({'human_entropy': h_ent, 'model_entropy': m_ent}).join(
    df[['hm_sbert_C', 'hm_drop_CA', 'ent', 'op']]
)
for c in ['human_entropy', 'model_entropy', 'hm_sbert_C', 'hm_drop_CA']:
    ent_df = ent_df[ent_df[c].notna() & np.isfinite(ent_df[c])]
ent_df['entropy_gap'] = ent_df['model_entropy'] - ent_df['human_entropy']

# Merge entities
ENT_MERGE = {
    'person': 'person', 'animal': 'animal', 'object': 'object', 'food': 'food',
    'other': 'other', 'product': 'other', 'place': 'other',
    'vehicle': 'other', 'text': 'other',
}
ent_df['ent_group'] = ent_df['ent'].map(ENT_MERGE).fillna('other')

# ── TABLE 1: Per-Entity-Group Summary ────────────────────────────────────────
print("=" * 80)
print("TABLE 1: Per-Entity-Group Entropy & Alignment Statistics (variant C, 7/8B)")
print("=" * 80)
header = "{:>10s} {:>4s} {:>8s} {:>8s} {:>8s} {:>9s} {:>8s}".format(
    "Group", "N", "H_human", "H_model", "Gap", "HM_SBERT", "C->A_deg")
print(header)
print("-" * 65)
for g in ['person', 'animal', 'object', 'food', 'other']:
    sub = ent_df[ent_df['ent_group'] == g]
    print("{:>10s} {:>4d} {:>8.3f} {:>8.3f} {:>8.3f} {:>9.3f} {:>8.3f}".format(
        g, len(sub),
        sub.human_entropy.mean(), sub.model_entropy.mean(),
        sub.entropy_gap.mean(), sub.hm_sbert_C.mean(),
        sub.hm_drop_CA.mean()))
print("-" * 65)
sub = ent_df
print("{:>10s} {:>4d} {:>8.3f} {:>8.3f} {:>8.3f} {:>9.3f} {:>8.3f}".format(
    "ALL", len(sub),
    sub.human_entropy.mean(), sub.model_entropy.mean(),
    sub.entropy_gap.mean(), sub.hm_sbert_C.mean(),
    sub.hm_drop_CA.mean()))

# ── TABLE 2: Correlation Summary ────────────────────────────────────────────
print("\n" + "=" * 80)
print("TABLE 2: Global Correlations")
print("=" * 80)
r1, p1 = stats.pearsonr(ent_df['human_entropy'], ent_df['model_entropy'])
r2, p2 = stats.pearsonr(ent_df['human_entropy'], ent_df['hm_sbert_C'])
valid = ent_df[ent_df['entropy_gap'].notna() & ent_df['hm_drop_CA'].notna()]
r3, p3 = stats.pearsonr(valid['entropy_gap'], valid['hm_drop_CA'])
print("  Human ent. vs Model ent.:    r = {:.3f}, p = {:.1e}, N = {}".format(r1, p1, len(ent_df)))
print("  Human ent. vs HM SBERT:      r = {:.3f}, p = {:.1e}, N = {}".format(r2, p2, len(ent_df)))
print("  Entropy gap vs C->A deg.:    r = {:.3f}, p = {:.1e}, N = {}".format(r3, p3, len(valid)))

# ── TABLE 3: Paired t-test (Human vs Model Entropy) ────────────────────────
print("\n" + "=" * 80)
print("TABLE 3: Paired t-test: Human Entropy vs Model Entropy (by entity group)")
print("=" * 80)
for g in ['person', 'animal', 'object', 'food', 'other', 'ALL']:
    sub = ent_df if g == 'ALL' else ent_df[ent_df['ent_group'] == g]
    t, p = stats.ttest_rel(sub['human_entropy'], sub['model_entropy'])
    n_higher = (sub['entropy_gap'] < 0).sum()
    print("  {:>10s}: t = {:>7.3f}, p = {:.1e}, mean_gap = {:>+.3f}, human > model: {}/{}".format(
        g, t, p, sub.entropy_gap.mean(), n_higher, len(sub)))

# ── TABLE 4: Per-Entity Correlation (Entropy vs HM SBERT) ──────────────────
print("\n" + "=" * 80)
print("TABLE 4: Per-Entity-Group Correlation: Human Entropy vs HM SBERT")
print("=" * 80)
for g in ['person', 'animal', 'object', 'food', 'other']:
    sub = ent_df[ent_df['ent_group'] == g]
    if len(sub) > 3:
        r, p = stats.pearsonr(sub['human_entropy'], sub['hm_sbert_C'])
        print("  {:>10s}: r = {:>+.3f}, p = {:.1e}, N = {}".format(g, r, p, len(sub)))
    else:
        print("  {:>10s}: N = {} (too few)".format(g, len(sub)))

# ── TABLE 5: Per-Entity Correlation (Entropy Gap vs C→A) ───────────────────
print("\n" + "=" * 80)
print("TABLE 5: Per-Entity-Group Correlation: Entropy Gap vs C->A Degradation")
print("=" * 80)
for g in ['person', 'animal', 'object', 'food', 'other']:
    sub = ent_df[ent_df['ent_group'] == g]
    sub = sub[sub['entropy_gap'].notna() & sub['hm_drop_CA'].notna()]
    if len(sub) > 3:
        r, p = stats.pearsonr(sub['entropy_gap'], sub['hm_drop_CA'])
        print("  {:>10s}: r = {:>+.3f}, p = {:.1e}, N = {}".format(g, r, p, len(sub)))
    else:
        print("  {:>10s}: N = {} (too few)".format(g, len(sub)))

# ── Also do by operation type ───────────────────────────────────────────────
print("\n" + "=" * 80)
print("TABLE 6: Per-Operation-Type Entropy & Alignment Statistics")
print("=" * 80)
header = "{:>10s} {:>4s} {:>8s} {:>8s} {:>8s} {:>9s} {:>8s}".format(
    "Op", "N", "H_human", "H_model", "Gap", "HM_SBERT", "C->A_deg")
print(header)
print("-" * 65)
for op in sorted(ent_df['op'].unique()):
    sub = ent_df[ent_df['op'] == op]
    print("{:>10s} {:>4d} {:>8.3f} {:>8.3f} {:>8.3f} {:>9.3f} {:>8.3f}".format(
        op, len(sub),
        sub.human_entropy.mean(), sub.model_entropy.mean(),
        sub.entropy_gap.mean(), sub.hm_sbert_C.mean(),
        sub.hm_drop_CA.mean()))

print("\nDone.")
