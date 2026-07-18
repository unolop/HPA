from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "think_eda"

MODEL_PAIRS = [
    ("Qwen3-0.6B", "Qwen3-0.6B (think)", 0.6),
    ("Qwen3-1.7B", "Qwen3-1.7B (think)", 1.7),
    ("Qwen3-4B", "Qwen3-4B (think)", 4.0),
    ("Qwen3-8B", "Qwen3-8B (think)", 8.0),
    ("Qwen3-32B", "Qwen3-32B (think)", 32.0),
]
VARIANT_ORDER = ["C", "B", "A"]
VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
VARIANT_COLORS = {"C": "#4C78A8", "B": "#F58518", "A": "#54A24B"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    think_summary = pd.read_csv(OUT_DIR / "think_eda_summary.csv")
    pair_cache = pd.read_parquet(ROOT / "analysis/session2/exports/pair_cache_cleaned.parquet")

    hh = (
        pair_cache[pair_cache["pair_type"] == "HH"]
        .groupby(["question_id", "variant"], as_index=False)["sbert_score"]
        .mean()
        .rename(columns={"sbert_score": "hh_sbert"})
    )
    hm = (
        pair_cache[
            (pair_cache["pair_type"] == "HM")
            & (pair_cache["subject_2"].isin([m for pair in MODEL_PAIRS for m in pair[:2]]))
        ]
        .groupby(["subject_2", "question_id", "variant", "op"], as_index=False)["sbert_score"]
        .mean()
        .rename(columns={"subject_2": "model", "sbert_score": "hm_sbert"})
    )

    think_len = think_summary.rename(columns={"model": "think_model", "think_mean": "mean_think_words"})[
        ["think_model", "variant", "mean_think_words"]
    ]

    r_rows = []
    op_rows = []
    for base_model, think_model, size in MODEL_PAIRS:
        for variant in VARIANT_ORDER:
            base_q = hm[(hm["model"] == base_model) & (hm["variant"] == variant)][["question_id", "hm_sbert"]]
            think_q = hm[(hm["model"] == think_model) & (hm["variant"] == variant)][["question_id", "hm_sbert"]]
            hh_q = hh[hh["variant"] == variant][["question_id", "hh_sbert"]]

            base_merged = hh_q.merge(base_q, on="question_id", how="inner")
            think_merged = hh_q.merge(think_q, on="question_id", how="inner")

            base_r = pearsonr(base_merged["hh_sbert"], base_merged["hm_sbert"]).statistic
            think_r = pearsonr(think_merged["hh_sbert"], think_merged["hm_sbert"]).statistic
            mean_think_words = float(
                think_len[
                    (think_len["think_model"] == think_model) & (think_len["variant"] == variant)
                ]["mean_think_words"].iloc[0]
            )

            r_rows.append(
                {
                    "size_b": size,
                    "size_label": f"{size:g}B",
                    "base_model": base_model,
                    "think_model": think_model,
                    "variant": variant,
                    "mean_think_words": mean_think_words,
                    "base_r": base_r,
                    "think_r": think_r,
                    "delta_r": think_r - base_r,
                }
            )

        base_op = hm[hm["model"] == base_model].groupby(["op", "variant"], as_index=False)["hm_sbert"].mean()
        think_op = hm[hm["model"] == think_model].groupby(["op", "variant"], as_index=False)["hm_sbert"].mean()
        merged_op = think_op.merge(base_op, on=["op", "variant"], suffixes=("_think", "_base"))
        merged_op["delta_hm_sbert"] = merged_op["hm_sbert_think"] - merged_op["hm_sbert_base"]
        merged_op["size_b"] = size
        merged_op["size_label"] = f"{size:g}B"
        op_rows.append(merged_op)

    r_df = pd.DataFrame(r_rows).sort_values(["size_b", "variant"])
    op_df = pd.concat(op_rows, ignore_index=True)
    op_avg = (
        op_df.groupby(["size_label", "op"], as_index=False)["delta_hm_sbert"]
        .mean()
    )

    r_df.to_csv(OUT_DIR / "think_delta_r_vs_length.csv", index=False)
    op_df.to_csv(OUT_DIR / "think_delta_by_op_variant.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.9, 4.2))
    for size, g in r_df.groupby("size_b"):
        g = g.sort_values("variant", key=lambda s: s.map({v: i for i, v in enumerate(VARIANT_ORDER)}))
        ax.plot(g["mean_think_words"], g["delta_r"], color="#C7C7C7", linewidth=1.0, zorder=1)
    for _, row in r_df.iterrows():
        ax.scatter(
            row["mean_think_words"],
            row["delta_r"],
            s=58,
            color=VARIANT_COLORS[row["variant"]],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            row["size_label"],
            (row["mean_think_words"], row["delta_r"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )
    ax.axhline(0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean think words")
    ax.set_ylabel(r"$\Delta r$ (think $-$ no-think)")
    ax.set_title("Question-level structure tracking")
    ax.grid(axis="y", alpha=0.18)
    ax.grid(axis="x", alpha=0.08)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=VARIANT_COLORS[v], markeredgecolor="white",
               markeredgewidth=0.7, markersize=7, label=VARIANT_LABELS[v])
        for v in VARIANT_ORDER
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "think_delta_r_vs_length.png", dpi=180, bbox_inches="tight")

    heatmap = op_avg.pivot(index="size_label", columns="op", values="delta_hm_sbert")
    heatmap = heatmap.reindex([f"{s:g}B" for _, _, s in MODEL_PAIRS])
    fig2, ax2 = plt.subplots(figsize=(10.5, 2.9))
    vmax = float(np.nanmax(np.abs(heatmap.values))) if np.isfinite(heatmap.values).any() else 0.1
    im = ax2.imshow(heatmap.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(len(heatmap.columns)))
    ax2.set_xticklabels(heatmap.columns, rotation=35, ha="right")
    ax2.set_yticks(range(len(heatmap.index)))
    ax2.set_yticklabels(heatmap.index)
    ax2.set_xlabel("Operation")
    ax2.set_ylabel("Qwen3 size")
    ax2.set_title(r"Mean $\Delta$ HM SBERT by operation (averaged over C/B/A)")
    for i in range(len(heatmap.index)):
        for j in range(len(heatmap.columns)):
            val = heatmap.iloc[i, j]
            if pd.notna(val):
                ax2.text(j, i, f"{val:.02f}", ha="center", va="center", fontsize=7, color="#222222")
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\Delta$ HM SBERT")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "think_delta_by_op_heatmap.png", dpi=180, bbox_inches="tight")

    print(r_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(op_avg.sort_values(["op", "size_label"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {OUT_DIR / 'think_delta_r_vs_length.png'}")
    print(f"Saved: {OUT_DIR / 'think_delta_by_op_heatmap.png'}")


if __name__ == "__main__":
    main()
