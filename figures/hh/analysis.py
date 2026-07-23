"""
Human-only agreement/diversity analysis across variants.

Single entry point for:
  - CI barplots
  - boxplots
  - summary CSVs

Outputs are saved directly into `figures/hh/` with descriptive
filenames rather than subfolders. Entropy-specific outputs are stored under
`analysis/session2/entropy/exports/`.

Examples:
  conda run -n zero python figures/hh/analysis.py
  conda run -n zero python figures/hh/analysis.py --style box
  conda run -n zero python figures/hh/analysis.py --style bar --free_text_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy as sp_entropy
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from config import extend_pair_cache_with_yesno
from figures.helpers import load_cleaned_pair_cache, save_fig
from utils.constants import VARIANT_COLORS, VARIANT_LABELS, VARIANT_ORDER

EXPORTS = ROOT / "analysis" / "session2" / "exports"
OUT_DIR = ROOT / "figures" / "hh"
CSV_DIR = ROOT / "notebooks" / "exports" / "hh"
LATEX_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures"
ENTROPY_EXPORTS = ROOT / "analysis" / "session2" / "entropy" / "exports"
ENTROPY_CSV_DIR = ENTROPY_EXPORTS / "csv"
ENTROPY_FIG_DIR = ENTROPY_EXPORTS / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)
ENTROPY_CSV_DIR.mkdir(parents=True, exist_ok=True)
ENTROPY_FIG_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_NAME = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
VARIANT_ORDER_LONG = ["Original", "Weaker", "Pronominalized"]
VARIANT_PALETTE = {
    "Original": VARIANT_COLORS["C"],
    "Weaker": VARIANT_COLORS["B"],
    "Pronominalized": VARIANT_COLORS["A"],
}

OP_MERGE = {
    "attr": "attr",
    "count": "count",
    "ident": "ident",
    "spat": "spat",
    "exist": "exist",
    "act": "act",
    "know": "other",
    "text": "other",
    "temp": "other",
    "comp": "other",
    "other": "other",
    "cause": "other",
}
OP_GROUP_ORDER = ["attr", "count", "spat", "ident", "act", "exist", "other"]
OP_GROUP_LABELS = {
    "attr": "Attribute",
    "count": "Counting",
    "spat": "Spatial Relation",
    "ident": "Identity",
    "act": "Action",
    "exist": "Existence",
    "other": "Other",
}
ENT_MERGE = {
    "person": "person",
    "animal": "animal",
    "object": "object",
    "food": "food",
    "other": "other",
    "product": "other",
    "place": "other",
    "vehicle": "other",
    "text": "other",
}
ENT_ORDER = ["person", "animal", "object", "food", "other"]
ENT_LABELS = {
    "person": "Person",
    "animal": "Animal",
    "object": "Object",
    "food": "Food",
    "other": "Other",
}


def load_hh_pairs(include_yesno: bool) -> pd.DataFrame:
    pair_df = load_cleaned_pair_cache(
        ROOT,
        condition="inst_blind",
        include_yesno=include_yesno,
        verbose=True,
    )
    return pair_df[pair_df["pair_type"] == "HH"].copy()


def build_long_df(hh: pd.DataFrame) -> pd.DataFrame:
    human = pd.read_csv(EXPORTS / "responses_human.csv")
    meta = (
        human[human["variant"] == "C"]
        .drop_duplicates("question_id")[["question_id", "ent", "op"]]
        .copy()
    )
    meta["ent_group"] = meta["ent"].map(ENT_MERGE).fillna("other")

    rows = []
    for v_code, v_name in VARIANT_NAME.items():
        hh_v = (
            hh[hh["variant"] == v_code]
            .groupby("question_id", as_index=False)["sbert_score"]
            .mean()
            .rename(columns={"sbert_score": "hh_sbert"})
        )
        tmp = hh_v.merge(meta, on="question_id", how="left")
        tmp["variant_name"] = v_name
        rows.append(tmp)

    long_df = pd.concat(rows, ignore_index=True)
    long_df["op_group"] = long_df["op"].map(OP_MERGE).fillna("other")
    long_df = long_df[
        long_df["op_group"].isin(OP_GROUP_ORDER) & long_df["ent_group"].isin(ENT_ORDER)
    ].copy()
    long_df["op_label"] = long_df["op_group"].map(OP_GROUP_LABELS)
    long_df["ent_label"] = long_df["ent_group"].map(ENT_LABELS)
    return long_df


def summarize_pair_metric(hh: pd.DataFrame, group_col: str, metric_col: str, out_col: str) -> pd.DataFrame:
    question_means = (
        hh.groupby(["question_id", group_col, "variant"], dropna=False)[metric_col]
        .mean()
        .rename(out_col)
        .reset_index()
    )
    return (
        question_means.groupby([group_col, "variant"], dropna=False)[out_col]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "hh_mean", "sem": "hh_sem", "count": "n_questions"})
    )


def summarize_entropy(group_col: str, include_yesno: bool) -> pd.DataFrame:
    human = pd.read_csv(EXPORTS / "responses_human.csv")
    if not include_yesno:
        text_qids = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")["question_id"].drop_duplicates().tolist()
        human = human[human["question_id"].isin(text_qids)].copy()

    def answer_entropy(responses: pd.Series) -> float:
        counts = responses.value_counts()
        probs = counts / counts.sum()
        return float(sp_entropy(probs, base=2))

    q_level = (
        human.groupby(["question_id", group_col, "variant"], dropna=False)["response"]
        .apply(answer_entropy)
        .rename("entropy")
        .reset_index()
    )
    return (
        q_level.groupby([group_col, "variant"], dropna=False)["entropy"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "hh_mean", "sem": "hh_sem", "count": "n_questions"})
    )


def save_summary_csv(df: pd.DataFrame, name: str) -> None:
    path = CSV_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_entropy_csv(df: pd.DataFrame, name: str) -> None:
    path = ENTROPY_CSV_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_both(fig: plt.Figure, name: str) -> None:
    save_fig(fig, OUT_DIR, name)
    fig.savefig(LATEX_DIR / name, dpi=180, bbox_inches="tight")
    print(f"Saved: {LATEX_DIR / name}")


def save_entropy_figure(fig: plt.Figure, name: str) -> None:
    save_fig(fig, ENTROPY_FIG_DIR, name)


def _ordered_labels(summary: pd.DataFrame, group_col: str) -> list[str]:
    sub_c = summary[summary["variant"] == "C"].sort_values("hh_mean", ascending=False)
    ordered = sub_c[group_col].dropna().astype(str).tolist()
    seen = set(ordered)
    for label in summary[group_col].dropna().astype(str).tolist():
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def _label_counts(summary: pd.DataFrame, group_col: str) -> dict[str, int]:
    counts = (
        summary[summary["variant"] == "C"]
        .groupby(group_col)["n_questions"]
        .max()
        .to_dict()
    )
    return {str(k): int(v) for k, v in counts.items()}


def draw_barplot(ax: plt.Axes, summary: pd.DataFrame, group_col: str, title: str, ylabel: str) -> None:
    labels = _ordered_labels(summary, group_col)
    counts = _label_counts(summary, group_col)
    x = np.arange(len(labels))
    width = 0.24

    for i, variant in enumerate(VARIANT_ORDER):
        sub = summary[summary["variant"] == variant].set_index(group_col).reindex(labels)
        y = sub["hh_mean"].to_numpy(dtype=float)
        sem = sub["hh_sem"].fillna(0).to_numpy(dtype=float)
        ax.bar(
            x + (i - 1) * width,
            y,
            width=width,
            yerr=1.96 * sem,
            color=VARIANT_COLORS[variant],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.7,
            capsize=2.6,
            label=VARIANT_LABELS[variant],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={counts.get(lab, 0)})" for lab in labels], rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0.0, 1.0 if "entropy" not in ylabel.lower() else None)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title=None, ncol=1, fontsize=8, frameon=True, loc="best")


def draw_boxplot(
    ax: plt.Axes,
    long_df: pd.DataFrame,
    x_col: str,
    order: list[str],
    counts: dict[str, int],
    title: str,
    *,
    show_legend: bool,
) -> None:
    sns.boxplot(
        data=long_df,
        x=x_col,
        y="hh_sbert",
        hue="variant_name",
        order=order,
        hue_order=VARIANT_ORDER_LONG,
        palette=VARIANT_PALETTE,
        width=0.56,
        fliersize=2.2,
        linewidth=0.9,
        ax=ax,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticklabels(order, rotation=0, fontsize=12.5)
    ax.tick_params(axis="y", labelsize=11)
    leg = ax.get_legend()
    if not show_legend:
        if leg is not None:
            leg.remove()
    else:
        ax.legend(
            title=None,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            frameon=False,
            fontsize=10.5,
        )


def _paired_ca_pvalues(long_df: pd.DataFrame, group_col: str, order: list[str]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    sub = long_df[long_df["variant_name"].isin(["Original", "Pronominalized"])].copy()
    for label in order:
        grp = sub[sub[group_col] == label]
        wide = (
            grp.pivot_table(
                index="question_id",
                columns="variant_name",
                values="hh_sbert",
                aggfunc="mean",
            )
            .dropna(subset=["Original", "Pronominalized"])
        )
        if len(wide) < 3:
            continue
        _, p = wilcoxon(wide["Original"], wide["Pronominalized"])
        p = float(p) if np.isfinite(p) else np.nan
        out.append((label, p))
    return out


def _sig_text(p: float) -> str | None:
    if not np.isfinite(p):
        return None
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None


def _annotate_boxplot_significance(ax: plt.Axes, long_df: pd.DataFrame, x_col: str, order: list[str]) -> None:
    pairs = _paired_ca_pvalues(long_df, x_col, order)
    sig_pairs = [(label, p) for label, p in pairs if _sig_text(p) is not None]
    if not sig_pairs:
        return

    sig_color = "black"
    base = 0.90
    step = 0.055
    half_width = 0.21

    for i, (label, p) in enumerate(sig_pairs):
        idx = order.index(label)
        x0 = idx - half_width
        x1 = idx + half_width
        y = base
        txt = _sig_text(p)
        ax.plot(
            [x0, x0, x1, x1],
            [y - 0.014, y, y, y - 0.014],
            color=sig_color,
            lw=1.6,
            clip_on=False,
        )
        ax.text(
            idx,
            y + 0.01,
            txt,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=sig_color,
        )

    cur_lo, cur_hi = ax.get_ylim()
    ax.set_ylim(cur_lo, max(cur_hi, base + 0.07))


def export_metric_bar_figure(entity_summary: pd.DataFrame, op_summary: pd.DataFrame, q_tag: str, metric_label: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), sharey=False)
    draw_barplot(axes[0], entity_summary, "ent", "By entity type", metric_label)
    draw_barplot(axes[1], op_summary, "op", "By operation type", metric_label)
    fig.suptitle(f"Human-only {metric_label} across variants ({q_tag})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_both(fig, f"{stem}_{q_tag}.png")
    plt.close(fig)


def export_hh_boxplots(long_df: pd.DataFrame, q_tag: str) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    op_df = long_df[long_df["op_group"].isin(OP_GROUP_ORDER)].copy()
    op_counts = (
        op_df[op_df["variant_name"] == "Original"]
        .groupby("op_label")["question_id"]
        .nunique()
        .to_dict()
    )
    op_order = [OP_GROUP_LABELS[k] for k in OP_GROUP_ORDER if OP_GROUP_LABELS[k] in op_counts]
    ent_df = long_df[long_df["ent_group"].isin(ENT_ORDER)].copy()
    ent_counts = ent_df[ent_df["variant_name"] == "Original"].groupby("ent_label")["question_id"].nunique().to_dict()
    ent_order = (
        ent_df[ent_df["variant_name"] == "Original"]
        .groupby("ent_label")["hh_sbert"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # ── Separate op-only figure (legend on top) ──────────────────────────────
    fig_op, ax_op = plt.subplots(figsize=(7.3, 3.6))
    draw_boxplot(ax_op, op_df, "op_label", op_order, op_counts, "By operation type", show_legend=False)
    _annotate_boxplot_significance(ax_op, op_df, "op_label", op_order)
    ax_op.set_ylim(None, 1.0)
    ax_op.set_ylabel("HH SBERT", fontsize=11)
    ax_op.tick_params(axis='x', labelsize=9)
    plt.setp(ax_op.get_xticklabels(), rotation=20, ha='right')
    handles, labels = ax_op.get_legend_handles_labels()
    if handles:
        fig_op.legend(
            handles, labels, title=None, ncol=3,
            loc="upper center", bbox_to_anchor=(0.5, 1.02),
            frameon=False, fontsize=10.5,
            columnspacing=1.2, handletextpad=0.5,
        )
    fig_op.tight_layout(rect=[0, 0, 1, 0.91])
    save_both(fig_op, f"sbert_by_op_variant_box_{q_tag}.png")
    plt.close(fig_op)

    # ── Separate ent-only figure (no legend — shared with op figure above) ───
    fig_ent, ax_ent = plt.subplots(figsize=(6.5, 2.1))
    draw_boxplot(ax_ent, ent_df, "ent_label", ent_order, ent_counts, "By entity type", show_legend=False)
    _annotate_boxplot_significance(ax_ent, ent_df, "ent_label", ent_order)
    ax_ent.set_ylim(None, 1.0)
    ax_ent.set_ylabel("HH SBERT", fontsize=11)
    ax_ent.tick_params(axis='x', labelsize=9)
    plt.setp(ax_ent.get_xticklabels(), rotation=20, ha='right')
    ax_ent.get_legend() and ax_ent.get_legend().remove()
    fig_ent.tight_layout(rect=[0, 0, 1, 1])
    save_both(fig_ent, f"sbert_by_entity_variant_box_{q_tag}.png")
    plt.close(fig_ent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free_text_only", action="store_true")
    parser.add_argument("--style", choices=["bar", "box", "all"], default="all")
    args = parser.parse_args()

    include_yesno = not args.free_text_only
    hh = load_hh_pairs(include_yesno=include_yesno)
    q_tag = f"q{hh['question_id'].nunique()}" + ("_yesno" if include_yesno else "")

    if args.style in {"bar", "all"}:
        sbert_ent = summarize_pair_metric(hh, "ent", "sbert_score", "sbert")
        sbert_op = summarize_pair_metric(hh, "op", "sbert_score", "sbert")
        save_summary_csv(sbert_ent, f"sbert_by_entity_{q_tag}.csv")
        save_summary_csv(sbert_op, f"sbert_by_op_{q_tag}.csv")

        chrf_ent = summarize_pair_metric(hh, "ent", "chrf_score", "chrf")
        chrf_op = summarize_pair_metric(hh, "op", "chrf_score", "chrf")
        save_summary_csv(chrf_ent, f"chrf_by_entity_{q_tag}.csv")
        save_summary_csv(chrf_op, f"human_hh_chrf_by_op_{q_tag}.csv")

        ent_ent = summarize_entropy("ent", include_yesno=include_yesno)
        ent_op = summarize_entropy("op", include_yesno=include_yesno)
        save_entropy_csv(ent_ent, f"human_entropy_by_entity_{q_tag}.csv")
        save_entropy_csv(ent_op, f"human_entropy_by_op_{q_tag}.csv")

        export_metric_bar_figure(chrf_ent, chrf_op, q_tag, "HH chrF", "chrf_entity_op_bar")
        fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), sharey=False)
        draw_barplot(axes[0], ent_ent, "ent", "By entity type", "Answer entropy (bits)")
        draw_barplot(axes[1], ent_op, "op", "By operation type", "Answer entropy (bits)")
        fig.suptitle(f"Human-only Answer entropy (bits) across variants ({q_tag})", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_entropy_figure(fig, f"human_entropy_entity_op_bar_{q_tag}.png")
        plt.close(fig)

    if args.style in {"box", "all"}:
        long_df = build_long_df(hh)
        export_hh_boxplots(long_df, q_tag)


if __name__ == "__main__":
    main()
