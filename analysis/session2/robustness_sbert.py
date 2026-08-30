"""
SBERT robustness checks for the supplementary.

Generates:
  Figures → latex/AAAI2026/LaTeX/figures/robustness/
    metric_correlation.png        — pairwise Pearson r heatmap across all metrics

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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

import plot_style  # noqa: F401 — sets 10pt Times New Roman
from figures.helpers import filter_abstained_pairs
from analysis.utils.constants import GROUP_COLORS, VARIANT_ORDER

EXPORTS   = ROOT / "analysis/session2/exports"
FIG_DIR   = ROOT / "latex/AAAI2026/LaTeX/figures/appQ_sbert_robustness"
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
    pc = filter_abstained_pairs(pc)
    hm = pc[pc["pair_type"] == "HM"].copy()
    hh = pc[pc["pair_type"] == "HH"].copy()
    return pc, hm, hh


# ── 1. Metric correlation heatmap ─────────────────────────────────────────────

def _corr_matrix(hm_sub: pd.DataFrame):
    """Compute per-(question, variant) mean correlation matrix."""
    qv = hm_sub.groupby(["question_id", "variant"])[METRICS].mean()
    corr = qv.corr(method="pearson")
    labels = [METRIC_LABEL[m] for m in METRICS]
    corr.index = labels
    corr.columns = labels
    return corr


def _plot_single_corr(corr, mask, out_path: Path, *, cbar: bool = True) -> None:
    figw = 6.0 if cbar else 5.0
    fig, ax = plt.subplots(figsize=(figw, 5.0))
    sns.heatmap(
        corr, ax=ax, mask=mask,
        annot=True, fmt=".2f", annot_kws={"fontsize": 9},
        cmap="RdYlGn", vmin=-0.2, vmax=1.0, center=0.5,
        square=True, linewidths=0.5,
        cbar=cbar,
        **({"cbar_kws": {"shrink": 0.75, "label": "Pearson r"}} if cbar else {}),
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_metric_correlation(hm: pd.DataFrame) -> None:
    _at = pd.read_csv(ROOT.parent / "human-prior-alignment" / "data" / "vqa_answer_types.csv")
    ft_qids = set(_at[_at["answer_type"] == "other"]["question_id"].astype(int))
    hm_ft = hm[hm["question_id"].astype(int).isin(ft_qids)]

    corr_all = _corr_matrix(hm)
    corr_ft  = _corr_matrix(hm_ft)
    mask = np.triu(np.ones_like(corr_all, dtype=bool), k=1)

    _plot_single_corr(corr_ft,  mask, FIG_DIR / "metric_correlation_per_question_hm_183.png", cbar=False)
    _plot_single_corr(corr_all, mask, FIG_DIR / "metric_correlation_per_question_hm_339.png", cbar=True)


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


# ── 2b. Per-model embedding comparison table (7/8B) ──────────────────────────

# Models to include, in display order: (group_label, model_id, display_name)
PER_MODEL_ROWS = [
    ("VLM",      "InternVL-8B",       "InternVL-8B"),
    ("VLM",      "Qwen3-VL-8B",       "Qwen3-VL-8B"),
    ("VLM",      "LLaVA-1.5-7B",      "LLaVA-1.5"),
    ("VLM",      "LLaVA-Mistral",      "LLaVA-Mistral"),
    ("VLM",      "LLaVA-Vicuna",       "LLaVA-Vicuna"),
    ("Backbone", "InternVL-8B (LM)",   "InternVL-8B"),
    ("Backbone", "Qwen3-VL-8B (LM)",   "Qwen3-VL-8B"),
    ("Backbone", "LLaVA-1.5 (LM)",     "LLaVA-1.5"),
    ("Backbone", "LLaVA-Mistral (LM)", "LLaVA-Mistral"),
    ("Backbone", "LLaVA-Vicuna (LM)",  "LLaVA-Vicuna"),
    ("SA-LLM",   "Qwen3-8B",           "Qwen3-8B"),
    ("SA-LLM",   "Qwen3-8B (think)",   "Qwen3-8B (think)"),
    ("SA-LLM",   "Qwen2.5-7B-Instruct","Qwen2.5-7B"),
    ("SA-LLM",   "Vicuna-7B",          "Vicuna-7B"),
    ("SA-LLM",   "Mistral-7B",         "Mistral-7B"),
]


def make_embedding_compare_per_model_table(hm: pd.DataFrame) -> None:
    """Per-model (7/8B) alignment under SBERT, SimCSE, BERTScore + Spearman ρ footer."""
    embed_cols = ["sbert_score", "simcse_score", "bertscore_f1"]
    model_means = hm.groupby("subject_2")[embed_cols].mean()

    rho_simcse = spearmanr(
        model_means["sbert_score"], model_means["simcse_score"]
    ).statistic
    rho_bert = spearmanr(
        model_means["sbert_score"], model_means["bertscore_f1"]
    ).statistic

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        (r"\caption{Per-model human--model alignment under three embedding metrics "
         r"(all variants combined, 7/8B models). "
         r"SBERT = \texttt{all-mpnet-base-v2}; SimCSE = \texttt{sup-simcse-roberta-large}; "
         r"BERTScore = \texttt{roberta-large}. "
         r"Spearman~$\rho$ in footer compares per-model rankings against SBERT.}"),
        r"\label{tab:robustness_embedding_per_model}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{SBERT} & \textbf{SimCSE} & \textbf{BERTScore} \\",
        r"\midrule",
    ]

    GROUP_DISPLAY = {
        "VLM": "VLM",
        "Backbone": "Backbone Decoder",
        "SA-LLM": "Standalone LLM",
    }

    prev_group = None
    for group, mid, name in PER_MODEL_ROWS:
        if group != prev_group:
            if prev_group is not None:
                lines.append(r"\midrule")
            lines.append(
                r"\multicolumn{4}{l}{\textit{" + GROUP_DISPLAY[group] + r"}} \\"
            )
            prev_group = group
        if mid in model_means.index:
            s, c, b = (model_means.loc[mid, col] for col in embed_cols)
            lines.append(rf"\quad {name} & {s:.3f} & {c:.3f} & {b:.3f} \\")
        else:
            lines.append(rf"\quad {name} & --- & --- & --- \\")

    lines += [
        r"\midrule",
        (fr"Spearman~$\rho$ vs.\ SBERT & --- "
         fr"& {rho_simcse:.3f} & {rho_bert:.3f} \\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TABLE_DIR / "robustness_embedding_per_model.tex"
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

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading pair cache…")
    pc, hm, hh = load()
    print(f"  HM pairs: {len(hm):,}  |  HH pairs: {len(hh):,}")

    print("\n1. Metric correlation heatmap…")
    make_metric_correlation(hm)

    print("\n2. Embedding comparison table (group-level)…")
    make_embedding_compare_table(hm)

    print("\n2b. Embedding comparison table (per-model 7/8B)…")
    make_embedding_compare_per_model_table(hm)

    print("\n3. Per-op × group SBERT table…")
    make_op_group_table(hm, hh)

    print("\n4. Variant ranking table…")
    make_variant_ranking_table(hm, hh)

    print("\nDone.")
