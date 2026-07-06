"""Scatter: HH vs HM SBERT — generates both free-text-only and all-questions versions."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import plot_style  # noqa: F401 — sets 10pt Times New Roman

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "analysis/session2/exports"
OUTDIR = Path(__file__).resolve().parent

# 7B-scale models — grouped by row: VLM, Backbone decoder, Standalone LLM
MODELS_7B = [
    # VLM
    "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
    # Backbone decoder
    "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    # Standalone LLM
    "Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B",
]

VARIANT_COLORS = {"C": "#1f77b4", "B": "#ff7f0e", "A": "#2ca02c"}
VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}


def get_yn_qids(exports: Path) -> set:
    """Return question_ids where majority of human responses are yes/no."""
    human = pd.read_csv(exports / "responses_human.csv")
    vC = human[human["variant"] == "C"]
    q_yn = vC.groupby("question_id").apply(
        lambda g: (g["response"].str.lower().str.strip().isin(["yes", "no"])).mean() > 0.5,
        include_groups=False,
    )
    return set(q_yn[q_yn].index)


def make_scatter(pc: pd.DataFrame, models: list[str], suffix: str, title_extra: str):
    hh = pc[pc["pair_type"] == "HH"]
    hh_agg = (
        hh.groupby(["question_id", "variant"])
        .agg(hh_sbert=("sbert_score", "mean"))
        .reset_index()
    )

    hm = pc[pc["pair_type"] == "HM"]
    hm_agg = (
        hm[hm["subject_2"].isin(models)]
        .groupby(["question_id", "variant", "subject_2"])
        .agg(hm_sbert=("sbert_score", "mean"))
        .reset_index()
    )

    merged = hm_agg.merge(hh_agg, on=["question_id", "variant"])

    ncols = 5
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = merged[merged["subject_2"] == model]

        for v in ["C", "B", "A"]:
            vs = sub[sub["variant"] == v]
            ax.scatter(
                vs["hh_sbert"], vs["hm_sbert"],
                c=VARIANT_COLORS[v], label=VARIANT_LABELS[v],
                alpha=0.5, s=20, edgecolors="none",
            )

        r_all, p_all = stats.pearsonr(sub["hh_sbert"], sub["hm_sbert"])
        corr_text = f"r={r_all:.3f} (p={p_all:.1e})\n"
        for v, short in [("C", "Orig"), ("B", "Weak"), ("A", "Pron")]:
            vs = sub[sub["variant"] == v]
            if len(vs) > 2:
                r_v, _ = stats.pearsonr(vs["hh_sbert"], vs["hm_sbert"])
                corr_text += f"{short}: r={r_v:.3f}\n"

        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.text(
            0.02, 0.98, corr_text.strip(), transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.plot([0.1, 0.95], [0.1, 0.95], "--", color="gray", alpha=0.4, lw=1)
        ax.set_xlim(0.1, 0.95)
        ax.set_ylim(0.1, 0.95)

        if idx % ncols == 0:
            ax.set_ylabel("HM SBERT")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("HH SBERT")

    for idx in range(len(models), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    row_labels = ["VLM", "Backbone Decoder", "Standalone LLM"]
    for row_idx, label in enumerate(row_labels):
        if row_idx < nrows:
            axes[row_idx][0].annotate(
                label, xy=(-0.35, 0.5), xycoords="axes fraction",
                fontsize=10, fontweight="bold", rotation=90,
                ha="center", va="center",
            )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02))
    n_q = pc[pc["pair_type"] == "HH"]["question_id"].nunique()
    fig.suptitle(f"HH vs HM SBERT (7B scale, n={n_q}){title_extra}", fontsize=11, y=1.05)
    plt.tight_layout()
    fig.subplots_adjust(left=0.07)

    out = OUTDIR / f"hh_vs_hm_scatter_7b{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # Correlation summary
    print(f"\n{'Model':<25} {'All r':>7} {'All p':>10} {'Orig r':>8} {'Weak r':>8} {'Pron r':>8}")
    for model in models:
        sub = merged[merged["subject_2"] == model]
        if len(sub) < 3:
            continue
        r_all, p_all = stats.pearsonr(sub["hh_sbert"], sub["hm_sbert"])
        rs = {}
        for v in ["C", "B", "A"]:
            vs = sub[sub["variant"] == v]
            if len(vs) > 2:
                rs[v], _ = stats.pearsonr(vs["hh_sbert"], vs["hm_sbert"])
        print(f"{model:<25} {r_all:>7.3f} {p_all:>10.1e} {rs.get('C', float('nan')):>8.3f} {rs.get('B', float('nan')):>8.3f} {rs.get('A', float('nan')):>8.3f}")


if __name__ == "__main__":
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    yn_qids = get_yn_qids(EXPORTS)
    available = [m for m in MODELS_7B if m in pc[pc["pair_type"] == "HM"]["subject_2"].unique()]

    # 1. All questions (including yes/no)
    print("=" * 60)
    print("ALL QUESTIONS (including yes/no)")
    print("=" * 60)
    make_scatter(pc, available, "_all", " — all questions")

    # 2. Free-text only (excluding yes/no)
    print("\n" + "=" * 60)
    print("FREE-TEXT ONLY (excluding yes/no)")
    print("=" * 60)
    pc_ft = pc[~pc["question_id"].isin(yn_qids)]
    make_scatter(pc_ft, available, "_freetext", " — free-text only")
