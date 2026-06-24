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

from config import MODELS_7B
from figures.helpers import save_fig

OUT_DIR = REPO / "figures/hh_hm_correlation_scatter"
GROUP_DIR = OUT_DIR / "by_group_models_7b_modelcolor"
GROUP_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS = REPO / "analysis/session2/exports"

GROUP_TITLE = {
    "VLM": "VLM",
    "VLM backbone decoder": "Backbone decoder",
    "standalone LLM": "SA-LLM",
    "standalone LLM (think)": "SA-LLM (think)",
}

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
MIN_CATEGORY_POINTS = 5
ENTITY_MARKERS = {
    "animal": "o",
    "food": "s",
    "object": "^",
    "other": "D",
    "person": "P",
    "place": "X",
    "product": "v",
    "text": "<",
    "vehicle": ">",
}
OP_MARKERS = {
    "act": "o",
    "attr": "s",
    "cause": "^",
    "comp": "D",
    "count": "P",
    "exist": "X",
    "ident": "v",
    "know": "<",
    "spat": ">",
    "temp": "h",
    "text": "*",
}


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
    question_level = (
        df.groupby(["model", "question_id", by], dropna=False)[["hm_sbert", "hh_sbert"]]
        .mean()
        .reset_index()
    )
    return (
        question_level.groupby(["model", by], dropna=False)[["hm_sbert", "hh_sbert"]]
        .agg(["mean", "sem", "count"])
        .reset_index()
    ).pipe(_flatten_agg_columns).dropna(subset=["model", by, "hm_mean", "hh_mean"])


def _flatten_agg_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        col if isinstance(col, str) else "_".join(str(c) for c in col if c)
        for col in df.columns
    ]
    return df.rename(
        columns={
            "hm_sbert_mean": "hm_mean",
            "hm_sbert_sem": "hm_sem",
            "hm_sbert_count": "n_questions",
            "hh_sbert_mean": "hh_mean",
            "hh_sbert_sem": "hh_sem",
        }
    )


def draw_panel(ax, agg: pd.DataFrame, label_col: str, title: str, color_map: dict[str, str]):
    lim_min = float(min(agg["hm_mean"].min(), agg["hh_mean"].min())) - 0.04
    lim_max = float(max(agg["hm_mean"].max(), agg["hh_mean"].max())) + 0.04
    lim_min = max(0.0, lim_min)
    lim_max = min(1.0, lim_max)
    marker_map = ENTITY_MARKERS if label_col == "ent" else OP_MARKERS

    text_lines = []
    for model, mdf in agg.groupby("model", dropna=False):
        color = color_map[model]
        x = mdf["hm_mean"].to_numpy()
        y = mdf["hh_mean"].to_numpy()
        for _, row in mdf.iterrows():
            ax.scatter(
                row["hm_mean"],
                row["hh_mean"],
                s=58,
                color=color,
                marker=marker_map.get(str(row[label_col]), "o"),
                edgecolors="white",
                linewidths=0.6,
                alpha=0.9,
                zorder=3,
            )
        if len(mdf) >= 3:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ys = slope * xs + intercept
            ax.plot(xs, ys, color=color, lw=1.8, alpha=0.9)
        if len(mdf) >= 3:
            r = mdf["hm_mean"].corr(mdf["hh_mean"], method="pearson")
            short = str(model).replace("Qwen2.5-", "Q2.5-").replace("Qwen3-VL-", "Q3VL-").replace("Qwen3-", "Q3-")
            text_lines.append(f"{short} {r:.2f}")

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
    subdf = df[(df["model_group"] == group) & (df["model"].isin(MODELS_7B))].copy()
    if subdf.empty:
        return
    subdf = attach_meta(subdf)
    ent_agg = aggregate(subdf, "ent")
    op_agg = aggregate(subdf, "op")
    ent_valid = set(ent_agg.groupby("model").size().loc[lambda s: s >= MIN_CATEGORY_POINTS].index.tolist())
    op_valid = set(op_agg.groupby("model").size().loc[lambda s: s >= MIN_CATEGORY_POINTS].index.tolist())
    valid_models = sorted(ent_valid & op_valid)
    ent_agg = ent_agg[ent_agg["model"].isin(valid_models)].copy()
    op_agg = op_agg[op_agg["model"].isin(valid_models)].copy()
    models = valid_models
    if not models:
        return
    color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.0), sharex=False, sharey=False)
    ent_markers = draw_panel(axes[0], ent_agg, "ent", "Aggregated by entity", color_map)
    op_markers = draw_panel(axes[1], op_agg, "op", "Aggregated by operation", color_map)
    axes[0].set_ylabel("HH SBERT", fontsize=11)
    axes[0].set_xlabel("HM SBERT", fontsize=11)
    axes[1].set_ylabel("HH SBERT", fontsize=11)
    axes[1].set_xlabel("HM SBERT", fontsize=11)

    model_handles = [
        mlines.Line2D([], [], color=color_map[m], marker="o", ls="none", ms=6, label=m)
        for m in models
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=max(6, min(8, len(model_handles))),
        fontsize=7,
        title="7/8B model",
        title_fontsize=8,
        frameon=True,
    )
    ent_handles = [
        mlines.Line2D([], [], color="#666666", marker=ent_markers[k], ls="none", ms=6, label=k)
        for k in sorted(ent_agg["ent"].dropna().unique().tolist())
    ]
    op_handles = [
        mlines.Line2D([], [], color="#666666", marker=op_markers[k], ls="none", ms=6, label=k)
        for k in sorted(op_agg["op"].dropna().unique().tolist())
    ]
    axes[0].legend(
        handles=ent_handles, title="Entity", fontsize=7, title_fontsize=8,
        frameon=True, loc="lower right",
        bbox_to_anchor=(1.0, 0.13),
    )
    axes[1].legend(
        handles=op_handles, title="Op", fontsize=7, title_fontsize=8,
        frameon=True, loc="lower right",
        bbox_to_anchor=(1.0, 0.13),
    )
    fig.suptitle(f"{GROUP_TITLE.get(group, group)}: per-model aggregated HH vs HM SBERT, variants averaged, 7/8B only", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    save_fig(fig, GROUP_DIR, f"hh_hm_sbert_correlation_scatter_{group_slug(group)}_7b_avgvariants_modelcolor_q113_yesno.png")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(OUT_DIR / "hh_hm_sbert_correlation_points_all_models_q113_yesno.csv")
    for group in sorted(df["model_group"].dropna().unique().tolist()):
        plot_group(df, group)
        print(f"Saved group figure: {group}")


if __name__ == "__main__":
    main()
