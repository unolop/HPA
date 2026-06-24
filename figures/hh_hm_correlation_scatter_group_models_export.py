from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT
if REPO.name == "figures":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from figures.helpers import save_fig
from utils.constants import VARIANT_LABELS, VARIANT_ORDER

OUT_DIR = REPO / "figures/hh_hm_correlation_scatter"
GROUP_DIR = OUT_DIR / "by_group_models"
GROUP_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS = REPO / "analysis/session2/exports"

VARIANT_COLORS = {
    "C": "#1f77b4",
    "B": "#ff7f0e",
    "A": "#2ca02c",
}

GROUP_TITLE = {
    "VLM": "VLM",
    "VLM backbone decoder": "Backbone decoder",
    "standalone LLM": "SA-LLM",
    "standalone LLM (think)": "SA-LLM (think)",
}

MODEL_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*", "8"]


def group_slug(group: str) -> str:
    return (
        group.lower()
        .replace(".", "")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def attach_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        pd.read_csv(EXPORTS / "responses_human.csv")
        .query("variant == 'C'")
        .drop_duplicates("question_id")[["question_id", "ent", "op"]]
    )
    return df.merge(meta, on="question_id", how="left")


def aggregate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    return (
        df.groupby(["model", by, "variant"], dropna=False)[["hm_sbert", "hh_sbert"]]
        .mean()
        .reset_index()
        .dropna(subset=["model", by, "hm_sbert", "hh_sbert"])
    )


def draw_panel(ax, agg: pd.DataFrame, label_col: str, title: str) -> None:
    lim_min = float(min(agg["hm_sbert"].min(), agg["hh_sbert"].min())) - 0.02
    lim_max = float(max(agg["hm_sbert"].max(), agg["hh_sbert"].max())) + 0.02
    lim_min = max(0.0, lim_min)
    lim_max = min(1.0, lim_max)

    models = sorted(agg["model"].unique().tolist())
    marker_map = {m: MODEL_MARKERS[i % len(MODEL_MARKERS)] for i, m in enumerate(models)}
    text_lines = []

    for variant in VARIANT_ORDER:
        vdf = agg[agg["variant"] == variant].copy()
        if vdf.empty:
            continue
        color = VARIANT_COLORS.get(variant, "#777777")
        for model, mdf in vdf.groupby("model", dropna=False):
            ax.scatter(
                mdf["hm_sbert"],
                mdf["hh_sbert"],
                s=44,
                color=color,
                marker=marker_map[model],
                edgecolors="white",
                linewidths=0.6,
                alpha=0.88,
            )
            cx = float(mdf["hm_sbert"].mean())
            cy = float(mdf["hh_sbert"].mean())
            ax.text(cx + 0.004, cy + 0.004, str(model), fontsize=7, color=color, alpha=0.85)
            if len(mdf) >= 3:
                r = mdf["hm_sbert"].corr(mdf["hh_sbert"], method="pearson")
                short = str(model).replace("Qwen2.5-", "Q2.5-").replace("Qwen3-VL-", "Q3VL-").replace("Qwen3-", "Q3-")
                text_lines.append(f"{VARIANT_LABELS.get(variant, variant)} {short} {r:.2f}")

    if text_lines:
        ax.text(
            0.97,
            0.03,
            "\n".join(text_lines[:10]),
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#dddddd", alpha=0.9),
        )

    ax.plot([lim_min, lim_max], [lim_min, lim_max], ls="--", lw=1.0, color="#bdbdbd", alpha=0.8)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=11)

    return marker_map


def plot_group(df: pd.DataFrame, group: str) -> None:
    subdf = df[df["model_group"] == group].copy()
    if subdf.empty:
        return
    subdf = attach_meta(subdf)
    ent_agg = aggregate(subdf, "ent")
    op_agg = aggregate(subdf, "op")

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.0), sharex=False, sharey=False)
    marker_map = draw_panel(axes[0], ent_agg, "ent", "Aggregated by entity")
    draw_panel(axes[1], op_agg, "op", "Aggregated by operation")
    axes[0].set_ylabel("HH SBERT", fontsize=11)
    axes[0].set_xlabel("HM SBERT", fontsize=11)
    axes[1].set_ylabel("HH SBERT", fontsize=11)
    axes[1].set_xlabel("HM SBERT", fontsize=11)

    variant_handles = [
        mlines.Line2D([], [], color=VARIANT_COLORS[v], marker="o", ls="none", ms=6, label=VARIANT_LABELS[v])
        for v in VARIANT_ORDER if v in subdf["variant"].unique()
    ]
    model_handles = [
        mlines.Line2D([], [], color="#666666", marker=marker_map[m], ls="none", ms=6, label=m)
        for m in sorted(subdf["model"].unique().tolist())
    ]
    if variant_handles:
        fig.legend(
            handles=variant_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(variant_handles),
            fontsize=8,
            title="Variant",
            title_fontsize=8,
            frameon=True,
        )
    if model_handles:
        fig.legend(
            handles=model_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(4, len(model_handles)),
            fontsize=7,
            title="Model",
            title_fontsize=8,
            frameon=True,
        )
    fig.suptitle(f"{GROUP_TITLE.get(group, group)}: per-model aggregated HH vs HM SBERT across all scales (q113_yesno)", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    save_fig(fig, GROUP_DIR, f"hh_hm_sbert_correlation_scatter_{group_slug(group)}_all_models_q113_yesno.png")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(OUT_DIR / "hh_hm_sbert_correlation_points_all_models_q113_yesno.csv")
    for group in sorted(df["model_group"].dropna().unique().tolist()):
        plot_group(df, group)
        print(f"Saved group figure: {group}")


if __name__ == "__main__":
    main()
