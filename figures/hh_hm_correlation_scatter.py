"""
Scatterplot of question-level HH vs HM agreement across all models.

Each point is a (question, model) pair.
  x-axis: HM SBERT
  y-axis: HH SBERT
  size:   model parameter scale (billions)
  color:  model family
  marker: model group

Default uses the yes/no-inclusive q113 set and facets by variant (C/B/A).

Run from repo root:
  conda run -n zero python figures/hh_hm_correlation_scatter.py
  conda run -n zero python figures/hh_hm_correlation_scatter.py --free_text_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from config import MODEL_GROUP, MODELS_7B, MODELS_ALL, extend_pair_cache_with_yesno
from figures.helpers import save_fig
from utils.constants import (
    GROUP_HOLLOW,
    GROUP_MARKER,
    MODEL_FAMILY,
    MODEL_FAMILY_COLORS,
    MODEL_SIZE_B,
    VARIANT_LABELS,
    VARIANT_ORDER,
)

EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "figures/hh_hm_correlation_scatter"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FAMILY_DIR = OUT_DIR / "by_family"
FAMILY_DIR.mkdir(parents=True, exist_ok=True)
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


def load_pair_df(include_yesno: bool) -> pd.DataFrame:
    pair_df = pd.read_parquet(EXPORTS / "pair_cache.parquet")
    if include_yesno:
        pair_df = extend_pair_cache_with_yesno(pair_df, EXPORTS)
    return pair_df


def build_plot_df(pair_df: pd.DataFrame, allowed_models: list[str]) -> pd.DataFrame:
    hh = (
        pair_df[pair_df["pair_type"] == "HH"]
        .groupby(["question_id", "variant"], dropna=False)["sbert_score"]
        .mean()
        .rename("hh_sbert")
        .reset_index()
    )
    hm = (
        pair_df[pair_df["pair_type"] == "HM"]
        .groupby(["question_id", "variant", "subject_2"], dropna=False)["sbert_score"]
        .mean()
        .rename("hm_sbert")
        .reset_index()
        .rename(columns={"subject_2": "model"})
    )
    df = hm.merge(hh, on=["question_id", "variant"], how="inner")
    df = df[df["model"].isin(allowed_models)].copy()
    df["model_group"] = df["model"].map(MODEL_GROUP).fillna("unknown")
    df["model_family"] = df["model"].map(MODEL_FAMILY).fillna("Unknown")
    df["size_b"] = df["model"].map(MODEL_SIZE_B).fillna(df["model"].map(MODEL_SIZE_B).median())
    return df


def size_to_marker_area(size_b: float) -> float:
    return 18 + 6 * np.sqrt(float(size_b))


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANT_ORDER:
        sub = df[df["variant"] == variant].dropna(subset=["hm_sbert", "hh_sbert"])
        if len(sub) >= 3:
            pearson = sub["hm_sbert"].corr(sub["hh_sbert"], method="pearson")
            spearman = sub["hm_sbert"].corr(sub["hh_sbert"], method="spearman")
            pearson_p = stats.pearsonr(sub["hm_sbert"], sub["hh_sbert"]).pvalue
            spearman_p = stats.spearmanr(sub["hm_sbert"], sub["hh_sbert"]).pvalue
        else:
            pearson = np.nan
            spearman = np.nan
            pearson_p = np.nan
            spearman_p = np.nan
        rows.append(
            {
                "variant": variant,
                "scope_type": "overall",
                "scope": "all_models",
                "n_points": len(sub),
                "pearson_r": pearson,
                "spearman_r": spearman,
                "pearson_p": pearson_p,
                "spearman_p": spearman_p,
            }
        )
        for group, gsub in sub.groupby("model_group", dropna=False):
            if len(gsub) >= 3:
                pearson_g = gsub["hm_sbert"].corr(gsub["hh_sbert"], method="pearson")
                spearman_g = gsub["hm_sbert"].corr(gsub["hh_sbert"], method="spearman")
                pearson_gp = stats.pearsonr(gsub["hm_sbert"], gsub["hh_sbert"]).pvalue
                spearman_gp = stats.spearmanr(gsub["hm_sbert"], gsub["hh_sbert"]).pvalue
            else:
                pearson_g = np.nan
                spearman_g = np.nan
                pearson_gp = np.nan
                spearman_gp = np.nan
            rows.append(
                {
                    "variant": variant,
                    "scope_type": "group",
                    "scope": str(group),
                    "n_points": len(gsub),
                    "pearson_r": pearson_g,
                    "spearman_r": spearman_g,
                    "pearson_p": pearson_gp,
                    "spearman_p": spearman_gp,
                }
            )
        for family, fsub in sub.groupby("model_family", dropna=False):
            if len(fsub) >= 3:
                pearson_f = fsub["hm_sbert"].corr(fsub["hh_sbert"], method="pearson")
                spearman_f = fsub["hm_sbert"].corr(fsub["hh_sbert"], method="spearman")
                pearson_fp = stats.pearsonr(fsub["hm_sbert"], fsub["hh_sbert"]).pvalue
                spearman_fp = stats.spearmanr(fsub["hm_sbert"], fsub["hh_sbert"]).pvalue
            else:
                pearson_f = np.nan
                spearman_f = np.nan
                pearson_fp = np.nan
                spearman_fp = np.nan
            rows.append(
                {
                    "variant": variant,
                    "scope_type": "family",
                    "scope": str(family),
                    "n_points": len(fsub),
                    "pearson_r": pearson_f,
                    "spearman_r": spearman_f,
                    "pearson_p": pearson_fp,
                    "spearman_p": spearman_fp,
                }
            )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, suffix: str, q_label: str, title_scope: str) -> None:
    fig, axes = plt.subplots(1, len(VARIANT_ORDER), figsize=(17.5, 5.6), sharex=True, sharey=True)
    lim_min = float(min(df["hm_sbert"].min(), df["hh_sbert"].min())) - 0.02
    lim_max = float(max(df["hm_sbert"].max(), df["hh_sbert"].max())) + 0.02
    lim_min = max(0.0, lim_min)
    lim_max = min(1.0, lim_max)

    for ax, variant in zip(axes, VARIANT_ORDER):
        sub = df[df["variant"] == variant].dropna(subset=["hm_sbert", "hh_sbert"]).copy()
        for _, row in sub.iterrows():
            family = row["model_family"]
            group = row["model_group"]
            face = MODEL_FAMILY_COLORS.get(family, "#777777")
            marker = GROUP_MARKER.get(group, "o")
            hollow = GROUP_HOLLOW.get(group, False)
            ax.scatter(
                row["hm_sbert"],
                row["hh_sbert"],
                s=size_to_marker_area(row["size_b"]),
                marker=marker,
                facecolors="none" if hollow else face,
                edgecolors=face,
                linewidths=1.0,
                alpha=0.75,
            )

        for group, gsub in sub.groupby("model_group", dropna=False):
            gsub = gsub.dropna(subset=["hm_sbert", "hh_sbert"])
            if len(gsub) < 3:
                continue
            x = gsub["hm_sbert"].to_numpy()
            y = gsub["hh_sbert"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ys = slope * xs + intercept
            ax.plot(
                xs,
                ys,
                color=GROUP_LINE_COLORS.get(str(group), "#555555"),
                lw=2.0,
                alpha=0.95,
            )

        if len(sub) >= 3:
            lines = [f"All {sub['hm_sbert'].corr(sub['hh_sbert'], method='pearson'):.3f}"]
            for group, gsub in sub.groupby("model_group", dropna=False):
                gsub = gsub.dropna(subset=["hm_sbert", "hh_sbert"])
                if len(gsub) < 3:
                    continue
                r = gsub["hm_sbert"].corr(gsub["hh_sbert"], method="pearson")
                label = GROUP_SHORT_LABELS.get(str(group), str(group))
                lines.append(f"{label} {r:.3f}")
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
        ax.set_title(VARIANT_LABELS[variant], fontsize=11)
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("HH SBERT", fontsize=11)
    for ax in axes:
        ax.set_xlabel("HM SBERT", fontsize=11)

    family_handles = [
        mlines.Line2D([], [], color=MODEL_FAMILY_COLORS.get(fam, "#666666"), marker="o", ls="none", ms=7, label=fam)
        for fam in sorted(df["model_family"].dropna().unique().tolist())
    ]
    group_handles = []
    for grp in sorted(df["model_group"].dropna().unique().tolist()):
        group_handles.append(
            mlines.Line2D(
                [],
                [],
                color=GROUP_LINE_COLORS.get(grp, "#555555"),
                marker=GROUP_MARKER.get(grp, "o"),
                markerfacecolor="none" if GROUP_HOLLOW.get(grp, False) else "#555555",
                ls="-",
                ms=7,
                label=grp,
            )
        )
    size_handles = [
        plt.scatter([], [], s=size_to_marker_area(s), color="#777777", alpha=0.5, label=f"{s:g}B")
        for s in [1, 2, 4, 8, 13, 32]
    ]

    fig.legend(handles=family_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(6, len(family_handles)), fontsize=8, title="Model family", title_fontsize=8, frameon=True)
    fig.legend(handles=group_handles, loc="lower left", bbox_to_anchor=(0.01, -0.02), ncol=2, fontsize=8, title="Model group", title_fontsize=8, frameon=True)
    fig.legend(handles=size_handles, loc="lower right", bbox_to_anchor=(0.99, -0.02), ncol=len(size_handles), fontsize=8, title="Model size", title_fontsize=8, frameon=True)
    fig.suptitle(f"Question-level HH vs HM SBERT across {title_scope} ({q_label})", fontsize=13)
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    save_fig(fig, OUT_DIR, f"hh_hm_sbert_correlation_scatter{suffix}.png")
    plt.close(fig)


def family_slug(family: str) -> str:
    return (
        family.lower()
        .replace(".", "")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def plot_family(df: pd.DataFrame, family: str, suffix: str, q_label: str, title_scope: str) -> None:
    subdf = df[df["model_family"] == family].copy()
    if subdf.empty:
        return

    fig, axes = plt.subplots(1, len(VARIANT_ORDER), figsize=(16.5, 5.2), sharex=True, sharey=True)
    lim_min = float(min(subdf["hm_sbert"].min(), subdf["hh_sbert"].min())) - 0.02
    lim_max = float(max(subdf["hm_sbert"].max(), subdf["hh_sbert"].max())) + 0.02
    lim_min = max(0.0, lim_min)
    lim_max = min(1.0, lim_max)
    fam_color = MODEL_FAMILY_COLORS.get(family, "#777777")

    for ax, variant in zip(axes, VARIANT_ORDER):
        vdf = subdf[subdf["variant"] == variant].dropna(subset=["hm_sbert", "hh_sbert"]).copy()
        for _, row in vdf.iterrows():
            group = row["model_group"]
            marker = GROUP_MARKER.get(group, "o")
            hollow = GROUP_HOLLOW.get(group, False)
            ax.scatter(
                row["hm_sbert"],
                row["hh_sbert"],
                s=size_to_marker_area(row["size_b"]),
                marker=marker,
                facecolors="none" if hollow else fam_color,
                edgecolors=fam_color,
                linewidths=1.0,
                alpha=0.8,
            )

        lines = []
        if len(vdf) >= 3:
            lines.append(f"All {vdf['hm_sbert'].corr(vdf['hh_sbert'], method='pearson'):.3f}")
        for group, gsub in vdf.groupby("model_group", dropna=False):
            gsub = gsub.dropna(subset=["hm_sbert", "hh_sbert"])
            if len(gsub) < 3:
                continue
            x = gsub["hm_sbert"].to_numpy()
            y = gsub["hh_sbert"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ys = slope * xs + intercept
            ax.plot(
                xs,
                ys,
                color=GROUP_LINE_COLORS.get(str(group), "#555555"),
                lw=2.0,
                alpha=0.95,
            )
            r = gsub["hm_sbert"].corr(gsub["hh_sbert"], method="pearson")
            lines.append(f"{GROUP_SHORT_LABELS.get(str(group), str(group))} {r:.3f}")

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
        ax.set_title(VARIANT_LABELS[variant], fontsize=11)
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("HH SBERT", fontsize=11)
    for ax in axes:
        ax.set_xlabel("HM SBERT", fontsize=11)

    group_handles = []
    for grp in sorted(subdf["model_group"].dropna().unique().tolist()):
        group_handles.append(
            mlines.Line2D(
                [],
                [],
                color=GROUP_LINE_COLORS.get(grp, "#555555"),
                marker=GROUP_MARKER.get(grp, "o"),
                markerfacecolor="none" if GROUP_HOLLOW.get(grp, False) else fam_color,
                markeredgecolor=GROUP_LINE_COLORS.get(grp, "#555555"),
                ls="-",
                ms=7,
                label=grp,
            )
        )
    size_handles = [
        plt.scatter([], [], s=size_to_marker_area(s), color="#777777", alpha=0.5, label=f"{s:g}B")
        for s in [1, 2, 4, 8, 13, 32]
    ]

    if group_handles:
        fig.legend(handles=group_handles, loc="lower left", bbox_to_anchor=(0.01, -0.02), ncol=min(3, len(group_handles)), fontsize=8, title="Model group", title_fontsize=8, frameon=True)
    fig.legend(handles=size_handles, loc="lower right", bbox_to_anchor=(0.99, -0.02), ncol=len(size_handles), fontsize=8, title="Model size", title_fontsize=8, frameon=True)
    fig.suptitle(f"{family}: question-level HH vs HM SBERT across {title_scope} ({q_label})", fontsize=13)
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    save_fig(fig, FAMILY_DIR, f"hh_hm_sbert_correlation_scatter_{family_slug(family)}{suffix}.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free_text_only", action="store_true")
    parser.add_argument("--models", choices=["all", "7b"], default="all")
    args = parser.parse_args()

    include_yesno = not args.free_text_only
    pair_df = load_pair_df(include_yesno=include_yesno)
    allowed_models = MODELS_7B if args.models == "7b" else MODELS_ALL
    plot_df = build_plot_df(pair_df, allowed_models=allowed_models)
    q_label = f"q{plot_df['question_id'].nunique()}" + ("_yesno" if include_yesno else "")
    scope_label = "7b" if args.models == "7b" else "all_models"
    suffix = f"_{scope_label}_{q_label}"

    corr = correlation_table(plot_df)
    corr_path = OUT_DIR / f"hh_hm_sbert_correlation_stats{suffix}.csv"
    corr.to_csv(corr_path, index=False)
    print(f"Saved: {corr_path}")
    plot_df_path = OUT_DIR / f"hh_hm_sbert_correlation_points{suffix}.csv"
    plot_df.to_csv(plot_df_path, index=False)
    print(f"Saved: {plot_df_path}")

    title_scope = "matched 7/8B models" if args.models == "7b" else "all models"
    plot(plot_df, suffix, q_label, title_scope=title_scope)
    for family in sorted(plot_df["model_family"].dropna().unique().tolist()):
        plot_family(plot_df, family, suffix, q_label, title_scope=title_scope)


if __name__ == "__main__":
    main()
