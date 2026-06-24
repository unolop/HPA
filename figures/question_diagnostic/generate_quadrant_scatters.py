"""
Generate question-level diagnostic quadrant scatter variants.

Outputs:
  1. diagnostic_quadrant_scatter_by_group.png
  2. diagnostic_quadrant_scatter_vlm_backbone.png
  3. diagnostic_quadrant_scatter_by_entity.png

Run from repo root:
  conda run -n zero python figures/question_diagnostic/generate_quadrant_scatters.py
"""

import argparse

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from figures.helpers import save_fig
from utils.constants import GROUP_COLORS, GROUP_ORDER, VARIANT_ORDER
from config import MODELS_7B, extend_pair_cache_with_yesno

EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "figures/question_diagnostic"
LATEX_DIR = ROOT / "latex/AAAI2026/LaTeX/figures/question_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

_7B = set(MODELS_7B)

OP_COLORS = {
    "yesno": "#2196F3",
    "count": "#E53935",
    "attr": "#4CAF50",
    "act": "#FF9800",
    "ident": "#9C27B0",
    "spat": "#795548",
    "wk": "#607D8B",
    "comp": "#00BCD4",
    "temp": "#CDDC39",
    "text": "#F44336",
    "caus": "#3F51B5",
}


def _build_question_diagnostic_df(include_yesno: bool = False) -> pd.DataFrame:
    pc = pd.read_parquet(EXPORTS / "pair_cache.parquet")
    if include_yesno:
        pc = extend_pair_cache_with_yesno(pc, EXPORTS)
    human = pd.read_csv(EXPORTS / "responses_human.csv")

    rows = []
    for variant in VARIANT_ORDER:
        sub = pc[pc["variant"] == variant]
        hh = sub[sub["pair_type"] == "HH"].groupby("question_id")["sbert_score"].mean()
        hm = sub[
            (sub["pair_type"] == "HM") & (sub["subject_2"].isin(_7B))
        ].groupby("question_id")["sbert_score"].mean()
        hm_grp = sub[
            (sub["pair_type"] == "HM") & (sub["subject_2"].isin(_7B))
        ].groupby(["question_id", "subject_group_2"])["sbert_score"].mean().unstack()

        for qid in hh.index:
            row = {
                "question_id": qid,
                "variant": variant,
                "hh_sbert": hh.get(qid, np.nan),
                "hm_sbert": hm.get(qid, np.nan),
            }
            for group in GROUP_ORDER:
                if group in hm_grp.columns:
                    row[f"hm_{group}"] = hm_grp[group].get(qid, np.nan)
            rows.append(row)

    qv = pd.DataFrame(rows)
    meta = human[human["variant"] == "C"].drop_duplicates("question_id")[
        ["question_id", "question_en", "ent", "op", "gt"]
    ].set_index("question_id")

    qwide = qv.pivot(index="question_id", columns="variant", values=["hh_sbert", "hm_sbert"])
    qwide.columns = [f"{metric}_{variant}" for metric, variant in qwide.columns]

    for group in GROUP_ORDER:
        col = f"hm_{group}"
        if col in qv.columns:
            gpiv = qv.pivot(index="question_id", columns="variant", values=col)
            gpiv.columns = [f"{col}_{variant}" for variant in gpiv.columns]
            qwide = qwide.join(gpiv)

    df = qwide.join(meta)
    df["hm_drop_CA"] = df["hm_sbert_C"] - df["hm_sbert_A"]
    return df


def _apply_common_axis_style(ax, x_med: float, y_med: float, xlim, ylim, title: str) -> None:
    ax.plot(xlim, xlim, "k--", alpha=0.25, lw=1, zorder=1)
    ax.axvline(x_med, color="gray", ls=":", alpha=0.4, lw=1)
    ax.axhline(y_med, color="gray", ls=":", alpha=0.4, lw=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_group_facets(df: pd.DataFrame, suffix: str, q_label: str) -> None:
    x_med = float(df["hh_sbert_C"].median())
    xlim = (
        max(0.10, float(df["hh_sbert_C"].min()) - 0.03),
        min(0.90, float(df["hh_sbert_C"].max()) + 0.03),
    )
    ylim = (
        max(0.10, float(min(df[f"hm_{g}_C"].min() for g in GROUP_ORDER if f"hm_{g}_C" in df.columns)) - 0.03),
        min(0.90, float(max(df[f"hm_{g}_C"].max() for g in GROUP_ORDER if f"hm_{g}_C" in df.columns)) + 0.03),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    legend_map = {}
    for ax, group in zip(axes, GROUP_ORDER):
        col = f"hm_{group}_C"
        if col not in df.columns:
            ax.axis("off")
            continue

        sub = df[["hh_sbert_C", col, "op", "hm_drop_CA"]].dropna().copy()
        for op in sorted(sub["op"].unique()):
            op_sub = sub[sub["op"] == op]
            sizes = 30 + 200 * op_sub["hm_drop_CA"].clip(0, 0.4)
            ax.scatter(
                op_sub["hh_sbert_C"],
                op_sub[col],
                s=sizes,
                c=OP_COLORS.get(op, "#888"),
                alpha=0.65,
                edgecolors="white",
                lw=0.5,
                label=op,
                zorder=3,
            )
            if op not in legend_map:
                legend_map[op] = plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    markerfacecolor=OP_COLORS.get(op, "#888"),
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    alpha=0.8,
                    label=op,
                )

        y_med = float(sub[col].median())
        nice_title = (
            group.replace("standalone LLM (think)", "SA-LLM (think)")
            .replace("standalone LLM", "SA-LLM")
            .replace("VLM backbone decoder", "Backbone decoder")
        )
        _apply_common_axis_style(ax, x_med, y_med, xlim, ylim, nice_title)

        high_hh = sub["hh_sbert_C"] >= x_med
        high_hm = sub[col] >= y_med
        ax.text(
            0.03,
            0.03,
            f"UR {int((high_hh & high_hm).sum())} | LR {int((high_hh & ~high_hm).sum())}\n"
            f"UL {int((~high_hh & high_hm).sum())} | LL {int((~high_hh & ~high_hm).sum())}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.9),
        )

    axes[0].set_ylabel("Human-Model SBERT", fontsize=11)
    axes[2].set_ylabel("Human-Model SBERT", fontsize=11)
    axes[2].set_xlabel("Human-Human SBERT", fontsize=11)
    axes[3].set_xlabel("Human-Human SBERT", fontsize=11)

    fig.suptitle(
        "Question-Level Diagnostic by Model Group\n"
        f"(variant C, inst_blind, q={q_label}; bubble size ∝ C→A degradation)",
        fontsize=12,
    )
    if legend_map:
        ordered_ops = [op for op in OP_COLORS if op in legend_map]
        fig.legend(
            [legend_map[op] for op in ordered_ops],
            ordered_ops,
            loc="lower center",
            ncol=max(1, len(ordered_ops)),
            fontsize=7.5,
            frameon=True,
            title="Operation type",
            title_fontsize=8,
            bbox_to_anchor=(0.5, -0.01),
            columnspacing=0.9,
            handletextpad=0.4,
            borderpad=0.4,
        )
    plt.tight_layout(rect=[0, 0.045, 1, 0.95])
    out = save_fig(fig, OUT_DIR, f"diagnostic_quadrant_scatter_by_group{suffix}.png")
    fig.savefig(LATEX_DIR / out.name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_vlm_backbone_focus(df: pd.DataFrame, suffix: str, q_label: str) -> None:
    groups = ["VLM", "VLM backbone decoder"]
    x_med = float(df["hh_sbert_C"].median())
    xlim = (
        max(0.10, float(df["hh_sbert_C"].min()) - 0.03),
        min(0.90, float(df["hh_sbert_C"].max()) + 0.03),
    )
    ylim = (
        max(0.10, float(min(df[f"hm_{g}_C"].min() for g in groups)) - 0.03),
        min(0.90, float(max(df[f"hm_{g}_C"].max() for g in groups)) + 0.03),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, group in zip(axes, groups):
        col = f"hm_{group}_C"
        sub = df[["hh_sbert_C", col, "op", "hm_drop_CA"]].dropna().copy()
        for op in sorted(sub["op"].unique()):
            op_sub = sub[sub["op"] == op]
            sizes = 30 + 200 * op_sub["hm_drop_CA"].clip(0, 0.4)
            ax.scatter(
                op_sub["hh_sbert_C"],
                op_sub[col],
                s=sizes,
                c=OP_COLORS.get(op, "#888"),
                alpha=0.65,
                edgecolors="white",
                lw=0.5,
                zorder=3,
            )
        y_med = float(sub[col].median())
        _apply_common_axis_style(
            ax,
            x_med,
            y_med,
            xlim,
            ylim,
            group.replace("VLM backbone decoder", "Backbone decoder"),
        )
    axes[0].set_ylabel("Human-Model SBERT", fontsize=11)
    axes[0].set_xlabel("Human-Human SBERT", fontsize=11)
    axes[1].set_xlabel("Human-Human SBERT", fontsize=11)
    fig.suptitle(
        "Question-Level Diagnostic: VLM vs Backbone Decoder\n"
        f"(variant C, inst_blind, q={q_label}; bubble size ∝ C→A degradation)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = save_fig(fig, OUT_DIR, f"diagnostic_quadrant_scatter_vlm_backbone{suffix}.png")
    fig.savefig(LATEX_DIR / out.name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_entity_scatter(df: pd.DataFrame, suffix: str, q_label: str) -> None:
    ent_counts = df["ent"].value_counts()
    ent_order = ent_counts.index.tolist()
    cmap = plt.get_cmap("tab20")
    ent_colors = {ent: cmap(i % 20) for i, ent in enumerate(ent_order)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for ent in ent_order:
        sub = df[df["ent"] == ent]
        sizes = 30 + 200 * sub["hm_drop_CA"].clip(0, 0.4)
        ax.scatter(
            sub["hh_sbert_C"],
            sub["hm_sbert_C"],
            s=sizes,
            c=[ent_colors[ent]],
            alpha=0.68,
            edgecolors="white",
            lw=0.5,
            label=f"{ent} (n={len(sub)})",
            zorder=3,
        )

    x_med = float(df["hh_sbert_C"].median())
    y_med = float(df["hm_sbert_C"].median())
    xlim = (
        max(0.10, float(df["hh_sbert_C"].min()) - 0.03),
        min(0.90, float(df["hh_sbert_C"].max()) + 0.03),
    )
    ylim = (
        max(0.10, float(df["hm_sbert_C"].min()) - 0.03),
        min(0.90, float(df["hm_sbert_C"].max()) + 0.03),
    )
    _apply_common_axis_style(ax, x_med, y_med, xlim, ylim, "Overall pooled view, colored by entity")
    ax.set_xlabel("Human-Human SBERT", fontsize=11)
    ax.set_ylabel("Human-Model SBERT", fontsize=11)
    ax.set_title(
        "Question-Level Diagnostic by Entity Type\n"
        f"(variant C, inst_blind, 7/8B pooled, q={q_label}; bubble size ∝ C→A degradation)",
        fontsize=11,
    )
    ax.legend(fontsize=7, ncol=2, loc="lower right", frameon=True, title="Entity type")
    plt.tight_layout()
    out = save_fig(fig, OUT_DIR, f"diagnostic_quadrant_scatter_by_entity{suffix}.png")
    fig.savefig(LATEX_DIR / out.name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_quadrant_summary(df: pd.DataFrame) -> None:
    print(f"Questions: {len(df)}")
    print(f"Entity types: {sorted(df['ent'].dropna().unique().tolist())}")
    x_med = float(df["hh_sbert_C"].median())
    pooled_y_med = float(df["hm_sbert_C"].median())
    pooled = df[["hh_sbert_C", "hm_sbert_C"]].dropna()
    high_hh = pooled["hh_sbert_C"] >= x_med
    high_hm = pooled["hm_sbert_C"] >= pooled_y_med
    print(
        "Pooled quadrants: "
        f"UR={(high_hh & high_hm).sum()} "
        f"LR={(high_hh & ~high_hm).sum()} "
        f"UL={(~high_hh & high_hm).sum()} "
        f"LL={(~high_hh & ~high_hm).sum()}"
    )

    for group in GROUP_ORDER:
        col = f"hm_{group}_C"
        if col not in df.columns:
            continue
        sub = df[["hh_sbert_C", col, "hm_drop_CA"]].dropna()
        y_med = float(sub[col].median())
        high_hh = sub["hh_sbert_C"] >= x_med
        high_hm = sub[col] >= y_med
        corr = sub["hh_sbert_C"].corr(sub[col])
        print(
            f"{group}: r(HH,HM)={corr:.3f}; "
            f"UR={(high_hh & high_hm).sum()} "
            f"LR={(high_hh & ~high_hm).sum()} "
            f"UL={(~high_hh & high_hm).sum()} "
            f"LL={(~high_hh & ~high_hm).sum()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_yesno", action="store_true")
    parser.add_argument("--free_text_only", action="store_true")
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    include_yesno = True
    if args.free_text_only:
        include_yesno = False
    elif args.include_yesno:
        include_yesno = True
    df = _build_question_diagnostic_df(include_yesno=include_yesno)
    suffix = "_q113_yesno" if include_yesno else ""
    q_label = str(len(df))
    _print_quadrant_summary(df)
    _plot_group_facets(df, suffix, q_label)
    _plot_vlm_backbone_focus(df, suffix, q_label)
    _plot_entity_scatter(df, suffix, q_label)
    print(f"\nDone. Outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
