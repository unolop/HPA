"""Heatmap: models × op groups, cell = HM SBERT, with HH reference row."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plot_style  # noqa: F401 — sets 10pt Times New Roman

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "analysis/session2/exports"
OUTDIR = Path(__file__).resolve().parent


def make_heatmap(pc: pd.DataFrame, suffix: str, title_extra: str = ""):

    # HH group means
    hh = pc[pc["pair_type"] == "HH"]
    hh_grp = (
        hh.groupby("op")
        .agg(sbert=("sbert_score", "mean"))
        .reset_index()
    )

    # HM group means per model
    hm = pc[pc["pair_type"] == "HM"]
    models_7b = [
        # VLM
        "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
        # Backbone decoder
        "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
        # Standalone LLM
        "Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B",
    ]
    models_7b = [m for m in models_7b if m in hm["subject_2"].unique()]

    hm_grp = (
        hm[hm["subject_2"].isin(models_7b)]
        .groupby(["subject_2", "op"])
        .agg(sbert=("sbert_score", "mean"))
        .reset_index()
    )

    # Pivot to matrix
    hm_piv = hm_grp.pivot(index="subject_2", columns="op", values="sbert")
    hm_piv = hm_piv.reindex(models_7b)

    # Sort ops by HH value (descending)
    op_order = hh_grp.sort_values("sbert", ascending=False)["op"].tolist()
    # Only keep ops present in both
    op_order = [o for o in op_order if o in hm_piv.columns]
    hm_piv = hm_piv[op_order]

    # Add HH row at top
    hh_row = hh_grp.set_index("op")["sbert"].reindex(op_order)
    full = pd.concat([
        pd.DataFrame([hh_row.values], columns=op_order, index=["Human–Human"]),
        hm_piv,
    ])

    # Group separators
    group_labels = {}
    idx = 1  # skip HH row
    for label, models in [
        ("VLM", [m for m in models_7b if "(LM)" not in m and m not in
                 ["Qwen3-8B", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B"]]),
        ("Backbone\nDecoder", [m for m in models_7b if "(LM)" in m]),
        ("Standalone\nLLM", [m for m in models_7b if m in
                             ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B"]]),
    ]:
        n = len(models)
        if n > 0:
            group_labels[idx + n / 2 - 0.5] = label
            idx += n

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        full.astype(float), annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Mean SBERT", "shrink": 0.8},
        ax=ax, vmin=0.15, vmax=0.70,
    )

    # Bold the HH row
    ax.axhline(1, color="black", linewidth=2)

    # Group separator lines
    vlm_count = len([m for m in models_7b if "(LM)" not in m and m not in
                     ["Qwen3-8B", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B"]])
    dec_count = len([m for m in models_7b if "(LM)" in m])
    ax.axhline(1 + vlm_count, color="black", linewidth=1.5, linestyle="--")
    ax.axhline(1 + vlm_count + dec_count, color="black", linewidth=1.5, linestyle="--")

    # Group labels on right
    for y_pos, label in group_labels.items():
        ax.text(
            len(op_order) + 0.6, y_pos + 0.5, label,
            fontsize=10, fontweight="bold", va="center", ha="left",
        )

    ax.set_xlabel("Operation Type")
    ax.set_ylabel("")
    ax.set_title(f"HM SBERT by Operation Group (7B scale){title_extra}")
    plt.tight_layout()

    out = OUTDIR / f"hh_vs_hm_heatmap_7b{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    pc_cleaned = EXPORTS / "pair_cache_cleaned.parquet"
    if not pc_cleaned.exists():
        print("pair_cache_cleaned.parquet not found")
        sys.exit(1)

    pc = pd.read_parquet(pc_cleaned)

    # Detect yes/no question IDs
    human = pd.read_csv(EXPORTS / "responses_human.csv")
    vC = human[human["variant"] == "C"]
    q_yn = vC.groupby("question_id").apply(
        lambda g: (g["response"].str.lower().str.strip().isin(["yes", "no"])).mean() > 0.5,
        include_groups=False,
    )
    yn_qids = set(q_yn[q_yn].index)

    # 1. All questions
    make_heatmap(pc, "_all", " — all questions")

    # 2. Free-text only
    pc_ft = pc[~pc["question_id"].isin(yn_qids)]
    make_heatmap(pc_ft, "_freetext", " — free-text only")
