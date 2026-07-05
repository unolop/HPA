"""Group-level HH vs HM analysis: Spearman ρ, profile cosine, rmcorr, + scatter plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "analysis/session2/exports"
OUTDIR = Path(__file__).resolve().parent

pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")

# --- HH aggregate per question x variant ---
hh = pc[pc["pair_type"] == "HH"]
hh_q = (
    hh.groupby(["question_id", "variant"])
    .agg(hh_sbert=("sbert_score", "mean"), op=("op", "first"), ent=("ent", "first"))
    .reset_index()
)

# --- HM aggregate per question x variant x model ---
hm = pc[pc["pair_type"] == "HM"]
hm_q = (
    hm.groupby(["question_id", "variant", "subject_2"])
    .agg(hm_sbert=("sbert_score", "mean"))
    .reset_index()
)

# 7B-scale models — grouped by: VLM, Backbone decoder, Standalone LLM
models_7b = [
    # VLM
    "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
    # Backbone decoder
    "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    # Standalone LLM
    "Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B",
]
models_7b = [m for m in models_7b if m in hm_q["subject_2"].unique()]

merged = hm_q[hm_q["subject_2"].isin(models_7b)].merge(
    hh_q[["question_id", "variant", "hh_sbert", "op", "ent"]],
    on=["question_id", "variant"],
)

# ═══════════════════════════════════════════════════════════════════
# Group-level aggregates (op groups)
# ═══════════════════════════════════════════════════════════════════
hh_grp = hh_q.groupby(["op", "variant"]).agg(hh_sbert=("hh_sbert", "mean")).reset_index()
hm_grp = (
    merged.groupby(["op", "variant", "subject_2"])
    .agg(hm_sbert=("hm_sbert", "mean"))
    .reset_index()
)
grp = hm_grp.merge(hh_grp, on=["op", "variant"])

# Also make variant-collapsed version for Spearman + cosine
hh_grp_all = hh_q.groupby("op").agg(hh_sbert=("hh_sbert", "mean")).reset_index()
hm_grp_all = (
    merged.groupby(["op", "subject_2"])
    .agg(hm_sbert=("hm_sbert", "mean"))
    .reset_index()
)
grp_all = hm_grp_all.merge(hh_grp_all, on="op")

# ═══════════════════════════════════════════════════════════════════
# 1. Spearman ρ on group means (variant-collapsed)
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. SPEARMAN ρ ON OP-GROUP MEANS (variant-collapsed)")
print("=" * 70)
n_ops = hh_grp_all["op"].nunique()
print(f"   Number of op groups: {n_ops}")
print(f"{'Model':<25} {'ρ':>7} {'p':>10}")
spearman_results = {}
for model in models_7b:
    sub = grp_all[grp_all["subject_2"] == model].sort_values("op")
    rho, p = stats.spearmanr(sub["hh_sbert"], sub["hm_sbert"])
    spearman_results[model] = (rho, p)
    print(f"{model:<25} {rho:>7.3f} {p:>10.3f}")

# Also per-variant Spearman
print(f"\n{'Model':<25} {'Orig ρ':>8} {'Weak ρ':>8} {'Pron ρ':>8}")
spearman_variant = {}
for model in models_7b:
    rs = {}
    for v in ["C", "B", "A"]:
        sub = grp[(grp["subject_2"] == model) & (grp["variant"] == v)].sort_values("op")
        if len(sub) > 2:
            rho, _ = stats.spearmanr(sub["hh_sbert"], sub["hm_sbert"])
            rs[v] = rho
    spearman_variant[model] = rs
    print(f"{model:<25} {rs.get('C', float('nan')):>8.3f} {rs.get('B', float('nan')):>8.3f} {rs.get('A', float('nan')):>8.3f}")

# ═══════════════════════════════════════════════════════════════════
# 2. Profile cosine similarity (variant-collapsed)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. PROFILE COSINE SIMILARITY (variant-collapsed op-group means)")
print("=" * 70)
hh_vec = hh_grp_all.sort_values("op")["hh_sbert"].values
print(f"   HH profile ({', '.join(hh_grp_all.sort_values('op')['op'])}): {np.round(hh_vec, 3)}")
print(f"{'Model':<25} {'Cosine Sim':>10} {'Profile vector'}")
cosine_results = {}
for model in models_7b:
    sub = grp_all[grp_all["subject_2"] == model].sort_values("op")
    # Align on shared ops
    shared = sub.merge(hh_grp_all, on="op", suffixes=("", "_hh")).sort_values("op")
    hh_v = shared["hh_sbert"].values
    hm_v = shared["hm_sbert"].values
    sim = 1 - cosine_dist(hh_v, hm_v)
    cosine_results[model] = sim
    print(f"{model:<25} {sim:>10.4f}   {np.round(hm_v, 3)}")

# Per-variant cosine
print(f"\n{'Model':<25} {'Orig cos':>9} {'Weak cos':>9} {'Pron cos':>9}")
cosine_variant = {}
for model in models_7b:
    cs = {}
    for v in ["C", "B", "A"]:
        hh_v = hh_grp[hh_grp["variant"] == v].sort_values("op")["hh_sbert"].values
        hm_v = grp[(grp["subject_2"] == model) & (grp["variant"] == v)].sort_values("op")["hm_sbert"].values
        if len(hh_v) == len(hm_v) and len(hh_v) > 0:
            cs[v] = 1 - cosine_dist(hh_v, hm_v)
    cosine_variant[model] = cs
    print(f"{model:<25} {cs.get('C', float('nan')):>9.4f} {cs.get('B', float('nan')):>9.4f} {cs.get('A', float('nan')):>9.4f}")

# ═══════════════════════════════════════════════════════════════════
# 3. Repeated-measures correlation (variant as repeated measure)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. REPEATED-MEASURES CORRELATION (op group × variant)")
print("=" * 70)
print(f"{'Model':<25} {'rmcorr':>7} {'p':>10} {'CI_low':>8} {'CI_high':>8} {'df':>4}")
rmcorr_results = {}
for model in models_7b:
    sub = grp[grp["subject_2"] == model].copy()
    sub = sub.dropna(subset=["hh_sbert", "hm_sbert"])
    if sub["op"].nunique() < 3 or sub["variant"].nunique() < 2:
        print(f"{model:<25} — insufficient data")
        continue
    try:
        rm = pg.rm_corr(data=sub, x="hh_sbert", y="hm_sbert", subject="op")
        r = rm["r"].values[0]
        p = rm["pval"].values[0]
        ci = rm["CI95%"].values[0]
        dof = rm["dof"].values[0]
        rmcorr_results[model] = (r, p, ci, dof)
        print(f"{model:<25} {r:>7.3f} {p:>10.3f} {ci[0]:>8.3f} {ci[1]:>8.3f} {dof:>4}")
    except Exception as e:
        print(f"{model:<25} — error: {e}")

# ═══════════════════════════════════════════════════════════════════
# SCATTER PLOTS: group-level HH vs HM
# ═══════════════════════════════════════════════════════════════════
variant_colors = {"C": "#1f77b4", "B": "#ff7f0e", "A": "#2ca02c"}
variant_labels = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}

ncols = 5
nrows = int(np.ceil(len(models_7b) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)

for idx, model in enumerate(models_7b):
    ax = axes[idx // ncols][idx % ncols]
    sub = grp[grp["subject_2"] == model]

    for v in ["C", "B", "A"]:
        vs = sub[sub["variant"] == v]
        ax.scatter(
            vs["hh_sbert"], vs["hm_sbert"],
            c=variant_colors[v], label=variant_labels[v],
            alpha=0.7, s=50, edgecolors="white", linewidth=0.5,
        )
        # Label each point with op name
        for _, row in vs.iterrows():
            ax.annotate(
                row["op"], (row["hh_sbert"], row["hm_sbert"]),
                fontsize=6, alpha=0.7,
                xytext=(3, 3), textcoords="offset points",
            )

    # Annotation box with all 3 metrics
    rho, rho_p = spearman_results.get(model, (float("nan"), float("nan")))
    cos_sim = cosine_results.get(model, float("nan"))
    rm_info = rmcorr_results.get(model, None)
    rm_str = f"rmcorr={rm_info[0]:.3f}" if rm_info else "rmcorr=—"

    txt = f"Spearman ρ={rho:.3f}\nCosine={cos_sim:.4f}\n{rm_str}"
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes,
        fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    ax.plot([0.2, 0.8], [0.2, 0.8], "--", color="gray", alpha=0.4, lw=1)
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_title(model, fontsize=11, fontweight="bold")

    if idx % ncols == 0:
        ax.set_ylabel("HM SBERT (group mean)", fontsize=10)
    if idx // ncols == nrows - 1:
        ax.set_xlabel("HH SBERT (group mean)", fontsize=10)

for idx in range(len(models_7b), nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

# Row labels
row_labels = ["VLM", "Backbone Decoder", "Standalone LLM"]
for row_idx, label in enumerate(row_labels):
    if row_idx < nrows:
        axes[row_idx][0].annotate(
            label, xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=12, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

handles, labels_leg = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="upper center", ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Group-Level: HH vs HM SBERT by Operation (7B scale)", fontsize=14, y=1.05)
plt.tight_layout()
fig.subplots_adjust(left=0.07)

out = OUTDIR / "hh_vs_hm_group_scatter_7b.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nsaved {out}")

# ═══════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Model':<25} {'Spearman ρ':>10} {'Cosine':>8} {'rmcorr':>8}")
for model in models_7b:
    rho = spearman_results.get(model, (float("nan"),))[0]
    cos_sim = cosine_results.get(model, float("nan"))
    rm = rmcorr_results.get(model, None)
    rm_r = rm[0] if rm else float("nan")
    print(f"{model:<25} {rho:>10.3f} {cos_sim:>8.4f} {rm_r:>8.3f}")
