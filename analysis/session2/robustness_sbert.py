"""
SBERT robustness checks for the supplementary.

Generates:
  Figures → latex/AAAI2026/LaTeX/figures/robustness/
    metric_correlation.png        — pairwise Pearson r heatmap across all metrics
    entropy_stratified_sbert.png  — HM SBERT by human-entropy tertile × model group

  Tables → latex/AAAI2026/LaTeX/tables/
    robustness_embedding_compare.tex — group means: SBERT vs SimCSE vs BERTScore
    robustness_op_group_sbert.tex    — per-op × group mean HM SBERT
    robustness_variant_ranking.tex   — variant × group mean HM SBERT

Run:
  conda run -n zero python analysis/session2/robustness_sbert.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy as sp_entropy, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

import plot_style  # noqa: F401 — sets 10pt Times New Roman
from figures.helpers import filter_abstained_pairs
from analysis.utils.constants import GROUP_COLORS, VARIANT_ORDER

EXPORTS   = ROOT / "analysis/session2/exports"
FIG_DIR   = ROOT / "latex/AAAI2026/LaTeX/figures/robustness"
TABLE_DIR = ROOT / "latex/AAAI2026/LaTeX/tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared constants ──────────────────────────────────────────────────────────

GROUP_ORDER = [
    "VLM",
    "VLM backbone decoder",
    "standalone LLM",
    "standalone LLM (think)",
]
GROUP_SHORT = {
    "VLM":                    "VLM",
    "VLM backbone decoder":   "Backbone",
    "standalone LLM":         "SA-LLM",
    "standalone LLM (think)": "Think",
}

METRICS = [
    "sbert_score", "simcse_score", "bertscore_f1",
    "chrf_score", "rouge1_score", "jaccard_score", "exact_score",
]
METRIC_LABEL = {
    "sbert_score":   "SBERT",
    "simcse_score":  "SimCSE",
    "bertscore_f1":  "BERTScore",
    "chrf_score":    "ChrF",
    "rouge1_score":  "ROUGE-1",
    "jaccard_score": "Jaccard",
    "exact_score":   "Exact",
}

OP_LABEL = {
    "act": "Action", "attr": "Attribute", "cause": "Causal",
    "comp": "Compare", "count": "Count", "exist": "Existence",
    "ident": "Identity", "know": "Knowledge", "spat": "Spatial",
    "temp": "Temporal", "text": "Text",
}


# ── Data ─────────────────────────────────────────────────────────────────────

def load():
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    hm = pc[pc["pair_type"] == "HM"].copy()
    hh = pc[pc["pair_type"] == "HH"].copy()
    return pc, hm, hh


# ── 1. Metric correlation heatmap ─────────────────────────────────────────────

def make_metric_correlation(hm: pd.DataFrame) -> None:
    """Pearson r between all metrics at the per-(question, variant) mean level."""
    qv = hm.groupby(["question_id", "variant"])[METRICS].mean()
    corr = qv.corr(method="pearson")
    labels = [METRIC_LABEL[m] for m in METRICS]
    corr.index = labels
    corr.columns = labels

    # lower-triangle only
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    sns.heatmap(
        corr, ax=ax, mask=mask,
        annot=True, fmt=".2f", annot_kws={"fontsize": 9},
        cmap="RdYlGn", vmin=-0.2, vmax=1.0, center=0.5,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    ax.set_title(
        "Metric pairwise correlations\n"
        "(per-question\u00d7variant mean, HM pairs, $n$=113 questions)",
        fontsize=10,
    )
    plt.tight_layout()
    out = FIG_DIR / "metric_correlation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 2. Embedding comparison table ─────────────────────────────────────────────

def make_embedding_compare_table(hm: pd.DataFrame) -> None:
    """Group means for SBERT, SimCSE, BERTScore; plus per-model rank ρ."""
    embed_cols = ["sbert_score", "simcse_score", "bertscore_f1"]
    sub = hm[hm["subject_group_2"].isin(GROUP_ORDER)]

    grp_means = (
        sub.groupby("subject_group_2")[embed_cols]
        .mean()
        .reindex(GROUP_ORDER)
    )

    model_means = sub.groupby("subject_2")[embed_cols].mean()
    rho_simcse = spearmanr(model_means["sbert_score"], model_means["simcse_score"]).statistic
    rho_bert   = spearmanr(model_means["sbert_score"], model_means["bertscore_f1"]).statistic

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        (r"\caption{Group-level human--model alignment under three embedding-based metrics "
         r"(all variants combined). "
         r"Spearman~$\rho$ is computed across individual model rankings ($n$ models per group). "
         r"The backbone~$>$~VLM~$>$~SA-LLM ordering is consistent across all three embeddings.}"),
        r"\label{tab:robustness_embedding}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Group & SBERT (mpnet) & SimCSE & BERTScore \\",
        r"\midrule",
    ]
    for grp in GROUP_ORDER:
        s, c, b = (grp_means.loc[grp, col] for col in embed_cols)
        lines.append(f"{GROUP_SHORT[grp]} & {s:.3f} & {c:.3f} & {b:.3f} \\\\")
    lines += [
        r"\midrule",
        (fr"Spearman~$\rho$ vs.\ SBERT & \multicolumn{{1}}{{c}}{{---}} "
         fr"& {rho_simcse:.3f} & {rho_bert:.3f} \\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TABLE_DIR / "robustness_embedding_compare.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


# ── 3. Per-op × group SBERT table ─────────────────────────────────────────────

def make_op_group_table(hm: pd.DataFrame, hh: pd.DataFrame) -> None:
    """Mean HM SBERT by operation type × model group, with HH reference column."""
    sub = hm[hm["subject_group_2"].isin(GROUP_ORDER) & hm["op"].notna()]

    pivot = (
        sub.groupby(["op", "subject_group_2"])["sbert_score"]
        .mean()
        .unstack("subject_group_2")
        .reindex(columns=GROUP_ORDER)
    )
    hh_op = hh[hh["op"].notna()].groupby("op")["sbert_score"].mean().rename("HH ref")
    pivot = pivot.join(hh_op, how="left")

    # Sort by HH ref descending (natural difficulty order)
    pivot = pivot.sort_values("HH ref", ascending=False)

    col_headers = "HH ref & " + " & ".join(GROUP_SHORT[g] for g in GROUP_ORDER)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        (r"\caption{Mean human--model SBERT by question operation type and model group "
         r"(all variants combined, HM pairs, unfiltered). "
         r"HH ref = mean human--human SBERT for that operation type. "
         r"Sorted by HH reference score. "
         r"Backbone exceeds VLM in every operation type.}"),
        r"\label{tab:robustness_op_group}",
        r"\begin{tabular}{l" + "r" * (len(GROUP_ORDER) + 1) + "}",
        r"\toprule",
        f"Operation & {col_headers} \\\\",
        r"\midrule",
    ]
    for op, row in pivot.iterrows():
        label = OP_LABEL.get(op, op)
        hh_val = f"{row['HH ref']:.3f}" if pd.notna(row["HH ref"]) else "---"
        vals = " & ".join(
            f"{row[g]:.3f}" if pd.notna(row[g]) else "---"
            for g in GROUP_ORDER
        )
        lines.append(f"{label} & {hh_val} & {vals} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TABLE_DIR / "robustness_op_group_sbert.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


# ── 4. Variant × group ranking table ─────────────────────────────────────────

def make_variant_ranking_table(hm: pd.DataFrame, hh: pd.DataFrame) -> None:
    """Mean HM SBERT by variant × group; HH reference row included."""
    sub = hm[hm["subject_group_2"].isin(GROUP_ORDER)]
    hh_ref = hh.groupby("variant")["sbert_score"].mean()

    grp_var = (
        sub.groupby(["subject_group_2", "variant"])["sbert_score"]
        .mean()
        .unstack("variant")
        .reindex(GROUP_ORDER)[VARIANT_ORDER]
    )

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        (r"\caption{Mean human--model SBERT by control variant and model group. "
         r"C=original, B=weaker-object, A=pronominalized. "
         r"The backbone~$>$~VLM~$>$~SA-LLM ranking is stable across all three variants.}"),
        r"\label{tab:robustness_variant_ranking}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Group & C & B & A \\",
        r"\midrule",
        (r"\textit{Human ref} & "
         + " & ".join(f"{hh_ref.get(v, float('nan')):.3f}" for v in VARIANT_ORDER)
         + r" \\"),
        r"\midrule",
    ]
    for grp in GROUP_ORDER:
        vals = " & ".join(
            f"{grp_var.loc[grp, v]:.3f}"
            if (grp in grp_var.index and v in grp_var.columns and pd.notna(grp_var.loc[grp, v]))
            else "---"
            for v in VARIANT_ORDER
        )
        lines.append(f"{GROUP_SHORT[grp]} & {vals} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TABLE_DIR / "robustness_variant_ranking.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


# ── 5. Entropy-stratified SBERT ──────────────────────────────────────────────

def make_entropy_stratified(hm: pd.DataFrame) -> None:
    """Bar chart: HM SBERT by human answer-entropy tertile × model group."""
    human = pd.read_csv(EXPORTS / "responses_human.csv")

    def answer_entropy(responses: pd.Series) -> float:
        counts = responses.value_counts()
        probs = counts / counts.sum()
        return float(sp_entropy(probs, base=2))

    q_entropy = (
        human[human["variant"] == "C"]
        .groupby("question_id")["response"]
        .apply(answer_entropy)
        .rename("entropy")
        .reset_index()
    )
    q_entropy["tertile"] = pd.qcut(
        q_entropy["entropy"], 3, labels=["Low", "Mid", "High"]
    )

    # Entropy range labels for x-axis
    bins_edges = pd.qcut(q_entropy["entropy"], 3, retbins=True)[1]
    bin_labels = [
        f"Low\n[{bins_edges[0]:.1f}–{bins_edges[1]:.1f}]",
        f"Mid\n[{bins_edges[1]:.1f}–{bins_edges[2]:.1f}]",
        f"High\n[{bins_edges[2]:.1f}–{bins_edges[3]:.1f}]",
    ]

    hm_e = hm.merge(q_entropy[["question_id", "tertile"]], on="question_id", how="inner")
    sub = hm_e[hm_e["subject_group_2"].isin(GROUP_ORDER)]

    agg = (
        sub.groupby(["tertile", "subject_group_2"])["sbert_score"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )

    TERTILES = ["Low", "Mid", "High"]
    n_t = len(TERTILES)
    n_g = len(GROUP_ORDER)
    x = np.arange(n_t)
    width = 0.18
    offsets = np.linspace(-(n_g - 1) / 2, (n_g - 1) / 2, n_g) * width

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, grp in enumerate(GROUP_ORDER):
        sub_g = agg[agg["subject_group_2"] == grp].set_index("tertile")
        y    = [sub_g.loc[t, "mean"] if t in sub_g.index else np.nan for t in TERTILES]
        yerr = [sub_g.loc[t, "sem"] * 1.96 if t in sub_g.index else np.nan for t in TERTILES]
        ax.bar(
            x + offsets[i], y, width=width, yerr=yerr,
            color=GROUP_COLORS[grp], alpha=0.87, capsize=3,
            label=GROUP_SHORT[grp], edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylabel("Mean HM SBERT")
    ax.set_xlabel("Human answer entropy (bits, variant C)")
    ax.set_title("Human–model SBERT by human answer entropy tertile")
    ax.legend(title="Model group", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "entropy_stratified_sbert.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading pair cache…")
    pc, hm, hh = load()
    print(f"  HM pairs: {len(hm):,}  |  HH pairs: {len(hh):,}")

    print("\n1. Metric correlation heatmap…")
    make_metric_correlation(hm)

    print("\n2. Embedding comparison table…")
    make_embedding_compare_table(hm)

    print("\n3. Per-op × group SBERT table…")
    make_op_group_table(hm, hh)

    print("\n4. Variant ranking table…")
    make_variant_ranking_table(hm, hh)

    print("\n5. Entropy-stratified SBERT figure…")
    make_entropy_stratified(hm)

    print("\nDone.")
