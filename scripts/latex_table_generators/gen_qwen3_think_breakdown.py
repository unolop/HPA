"""
gen_qwen3_think_breakdown.py

Computes per-op SBERT score breakdown for:
  - HH (human-human baseline)
  - HM Qwen3-8B
  - HM Qwen3-8B (think)
  - Delta = HM - HH for each model

Averages across all variants (C, B, A) combined.
Also computes overall means (all ops combined).

Output: printed summary table.
"""

import pandas as pd
import numpy as np

PARQUET = "/home/david/Desktop/yuna/HPA/analysis/session2/exports/pair_cache_cleaned.parquet"

MODEL_BASE  = "Qwen3-8B"
MODEL_THINK = "Qwen3-8B (think)"

OP_ORDER = ["act", "attr", "cause", "comp", "exist", "ident", "know", "spat", "temp", "text", "yesno", "count"]

def main():
    print(f"Loading {PARQUET} ...")
    df = pd.read_parquet(PARQUET)
    print(f"  Loaded {len(df):,} rows\n")

    # Verify models exist
    for m in [MODEL_BASE, MODEL_THINK]:
        n = (df['subject_2'] == m).sum()
        print(f"  Rows with subject_2=={m!r}: {n:,}")

    # ── HH subset (pair_type == 'HH') ───────────────────────────────────────
    hh = df[df['pair_type'] == 'HH'].copy()

    # ── HM subsets ──────────────────────────────────────────────────────────
    hm_base  = df[(df['pair_type'] == 'HM') & (df['subject_2'] == MODEL_BASE)].copy()
    hm_think = df[(df['pair_type'] == 'HM') & (df['subject_2'] == MODEL_THINK)].copy()

    print(f"\n  HH rows: {len(hh):,}")
    print(f"  HM {MODEL_BASE} rows: {len(hm_base):,}")
    print(f"  HM {MODEL_THINK} rows: {len(hm_think):,}")

    # Check which ops actually appear
    all_ops = sorted(df['op'].dropna().unique())
    print(f"\n  All ops in data: {all_ops}")

    # Build per-op means
    def mean_by_op(subset, col='sbert_score'):
        return subset.groupby('op')[col].mean()

    hh_means    = mean_by_op(hh)
    base_means  = mean_by_op(hm_base)
    think_means = mean_by_op(hm_think)

    # Overall means
    hh_overall    = hh['sbert_score'].mean()
    base_overall  = hm_base['sbert_score'].mean()
    think_overall = hm_think['sbert_score'].mean()

    # Build ops present in at least one of the three groups
    ops_present = sorted(set(hh_means.index) | set(base_means.index) | set(think_means.index))

    # Pretty-print
    col_w = 14
    header = (
        f"{'Op':<8} "
        f"{'HH':>{col_w}} "
        f"{'Qwen3-8B':>{col_w}} "
        f"{'Δ(base-HH)':>{col_w}} "
        f"{'Qwen3-8B(think)':>{col_w}} "
        f"{'Δ(think-HH)':>{col_w}}"
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("SBERT Score Breakdown: Qwen3-8B vs Qwen3-8B (think)")
    print("Averaged across all variants (C, B, A)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    rows = {}
    for op in ops_present:
        hh_v    = hh_means.get(op, np.nan)
        base_v  = base_means.get(op, np.nan)
        think_v = think_means.get(op, np.nan)
        d_base  = base_v  - hh_v if not (np.isnan(base_v)  or np.isnan(hh_v)) else np.nan
        d_think = think_v - hh_v if not (np.isnan(think_v) or np.isnan(hh_v)) else np.nan
        rows[op] = dict(hh=hh_v, base=base_v, think=think_v, d_base=d_base, d_think=d_think)
        def fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else "  --  "
        print(
            f"{op:<8} "
            f"{fmt(hh_v):>{col_w}} "
            f"{fmt(base_v):>{col_w}} "
            f"{fmt(d_base):>{col_w}} "
            f"{fmt(think_v):>{col_w}} "
            f"{fmt(d_think):>{col_w}}"
        )

    print(sep)
    d_base_overall  = base_overall  - hh_overall
    d_think_overall = think_overall - hh_overall
    def fmt(v):
        return f"{v:.4f}" if not np.isnan(v) else "  --  "
    print(
        f"{'OVERALL':<8} "
        f"{fmt(hh_overall):>{col_w}} "
        f"{fmt(base_overall):>{col_w}} "
        f"{fmt(d_base_overall):>{col_w}} "
        f"{fmt(think_overall):>{col_w}} "
        f"{fmt(d_think_overall):>{col_w}}"
    )
    print(f"{'='*len(header)}\n")

    # ── Also break down by variant ───────────────────────────────────────────
    print("Per-variant overall means:")
    var_header = f"{'Variant':<10} {'HH':>10} {'Qwen3-8B':>12} {'Δbase':>10} {'Qwen3-8B(think)':>17} {'Δthink':>10}"
    print(var_header)
    print("-" * len(var_header))
    for v in ['C', 'B', 'A']:
        hh_v    = hh[hh['variant']==v]['sbert_score'].mean()
        base_v  = hm_base[hm_base['variant']==v]['sbert_score'].mean()
        think_v = hm_think[hm_think['variant']==v]['sbert_score'].mean()
        d_b = base_v  - hh_v
        d_t = think_v - hh_v
        n_hh    = (hh['variant']==v).sum()
        n_base  = (hm_base['variant']==v).sum()
        n_think = (hm_think['variant']==v).sum()
        print(f"{'  ' + v:<10} {hh_v:>10.4f} {base_v:>12.4f} {d_b:>10.4f} {think_v:>17.4f} {d_t:>10.4f}   (n_HH={n_hh:,}, n_base={n_base:,}, n_think={n_think:,})")

    # ── Variant × Op breakdown for MODEL_BASE ───────────────────────────────
    print("\n\nQwen3-8B SBERT by variant × op:")
    pivot_base = hm_base.groupby(['op','variant'])['sbert_score'].mean().unstack('variant')
    print(pivot_base.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n\nQwen3-8B (think) SBERT by variant × op:")
    pivot_think = hm_think.groupby(['op','variant'])['sbert_score'].mean().unstack('variant')
    print(pivot_think.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n\nHH SBERT by variant × op:")
    pivot_hh = hh.groupby(['op','variant'])['sbert_score'].mean().unstack('variant')
    print(pivot_hh.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Row counts sanity check ──────────────────────────────────────────────
    print("\n\nRow counts by op:")
    cnt_hh    = hh.groupby('op').size().rename('n_HH')
    cnt_base  = hm_base.groupby('op').size().rename('n_base')
    cnt_think = hm_think.groupby('op').size().rename('n_think')
    counts = pd.concat([cnt_hh, cnt_base, cnt_think], axis=1)
    print(counts.to_string())


if __name__ == "__main__":
    main()
