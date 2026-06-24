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
from utils.constants import (
    GROUP_HOLLOW,
    GROUP_MARKER,
    MODEL_FAMILY_COLORS,
    VARIANT_LABELS,
    VARIANT_ORDER,
)

OUT_DIR = REPO / "figures/hh_hm_correlation_scatter"
FAMILY_DIR = OUT_DIR / "by_family"
FAMILY_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS = REPO / "analysis/session2/exports"

GROUP_LINE_COLORS = {
    "VLM": "#1f77b4",
    "VLM backbone decoder": "#2ca02c",
    "standalone LLM": "#ff7f0e",
    "standalone LLM (think)": "#d62728",
}
GROUP_SHORT_LABELS = {
    "VLM": "VLM",
    "VLM backbone decoder": "Backbone",
    "standalone LLM": "SA-LLM",
    "standalone LLM (think)": "Think",
}
VARIANT_COLORS = {
    "C": "#1f77b4",
    "B": "#ff7f0e",
    "A": "#2ca02c",
}


def size_to_marker_area(size_b: float) -> float:
    return 18 + 6 * np.sqrt(float(size_b))


def family_slug(family: str) -> str:
    return (
        family.lower()
        .replace(".", "")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _attach_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        pd.read_csv(EXPORTS / "responses_human.csv")
        .query("variant == 'C'")
        .drop_duplicates("question_id")[["question_id", "ent", "op"]]
    )
    return df.merge(meta, on="question_id", how="left")


def _aggregate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    return (
        df.groupby([by, "variant"], dropna=False)[["hm_sbert", "hh_sbert"]]
        .mean()
        .reset_index()
        .dropna(subset=[by, "hm_sbert", "hh_sbert"])
    )


def _draw_agg_scatter(ax, agg: pd.DataFrame, label_col: str, title: str) -> None:
    lim_min = float(min(agg["hm_sbert"].min(), agg["hh_sbert"].min())) - 0.02
    lim_max = float(max(agg["hm_sbert"].max(), agg["hh_sbert"].max())) + 0.02
    lim_min = max(0.0, lim_min)
    lim_max = min(1.0, lim_max)

    lines = []
    for variant in VARIANT_ORDER:
        vdf = agg[agg["variant"] == variant].copy()
        if vdf.empty:
            continue
        color = VARIANT_COLORS.get(variant, "#777777")
        ax.scatter(
            vdf["hm_sbert"],
            vdf["hh_sbert"],
            s=58,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
            label=VARIANT_LABELS.get(variant, variant),
        )
        for _, row in vdf.iterrows():
            ax.text(
                row["hm_sbert"] + 0.004,
                row["hh_sbert"] + 0.004,
                str(row[label_col]),
                fontsize=8,
                color=color,
                alpha=0.95,
            )
        if len(vdf) >= 3:
            x = vdf["hm_sbert"].to_numpy()
            y = vdf["hh_sbert"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ys = slope * xs + intercept
            ax.plot(xs, ys, color=color, lw=2.0, alpha=0.9)
            r = vdf["hm_sbert"].corr(vdf["hh_sbert"], method="pearson")
            lines.append(f"{VARIANT_LABELS.get(variant, variant)} {r:.3f}")

    if lines:
        ax.text(
            0.97,
            0.03,
            "\n".join(lines),
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.9),
        )

    ax.plot([lim_min, lim_max], [lim_min, lim_max], ls="--", lw=1.0, color="#bdbdbd", alpha=0.8)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=11)


def plot_family(df: pd.DataFrame, family: str) -> None:
    subdf = df[df["model_family"] == family].copy()
    if subdf.empty:
        return
    subdf = _attach_meta(subdf)
    ent_agg = _aggregate(subdf, "ent")
    op_agg = _aggregate(subdf, "op")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), sharex=False, sharey=False)
    _draw_agg_scatter(axes[0], ent_agg, "ent", "Aggregated by entity")
    _draw_agg_scatter(axes[1], op_agg, "op", "Aggregated by operation")
    axes[0].set_ylabel("HH SBERT", fontsize=11)
    axes[0].set_xlabel("HM SBERT", fontsize=11)
    axes[1].set_ylabel("HH SBERT", fontsize=11)
    axes[1].set_xlabel("HM SBERT", fontsize=11)

    variant_handles = [
        mlines.Line2D([], [], color=VARIANT_COLORS[v], marker="o", ls="-", ms=6, label=VARIANT_LABELS[v])
        for v in VARIANT_ORDER if v in subdf["variant"].unique()
    ]
    fig.legend(handles=variant_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(variant_handles), fontsize=8, title="Variant", title_fontsize=8, frameon=True)
    fig.suptitle(f"{family}: aggregated HH vs HM SBERT across all models (q113_yesno)", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    save_fig(fig, FAMILY_DIR, f"hh_hm_sbert_correlation_scatter_{family_slug(family)}_all_models_q113_yesno.png")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(OUT_DIR / "hh_hm_sbert_correlation_points_all_models_q113_yesno.csv")
    for family in sorted(df["model_family"].dropna().unique().tolist()):
        plot_family(df, family)
        print(f"Saved family figure: {family}")


if __name__ == "__main__":
    main()
