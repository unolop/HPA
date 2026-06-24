"""
Human-only agreement/diversity analysis across variants.

Exports a 2-panel figure:
  - by entity type across C/B/A
  - by operation type across C/B/A

Default uses the yes/no-inclusive q113 pair set.

Run from repo root:
  conda run -n zero python figures/human_hh_analysis.py
  conda run -n zero python figures/human_hh_analysis.py --free_text_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from config import extend_pair_cache_with_yesno
from figures.helpers import save_fig
from utils.constants import VARIANT_COLORS, VARIANT_LABELS, VARIANT_ORDER

EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "figures/human_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITY_COLOR_MAP = {
    "animal": "#D95F02",
    "food": "#1B9E77",
    "object": "#7570B3",
    "other": "#66A61E",
    "person": "#E7298A",
    "place": "#E6AB02",
    "product": "#A6761D",
    "text": "#666666",
    "vehicle": "#1F78B4",
}

OP_COLOR_MAP = {
    "act": "#FF9800",
    "attr": "#4CAF50",
    "cause": "#3F51B5",
    "comp": "#00BCD4",
    "count": "#E53935",
    "exist": "#00897B",
    "ident": "#9C27B0",
    "know": "#6D4C41",
    "spat": "#795548",
    "temp": "#CDDC39",
    "text": "#F44336",
}


def load_hh_pairs(include_yesno: bool) -> pd.DataFrame:
    pair_df = pd.read_parquet(EXPORTS / "pair_cache.parquet")
    if include_yesno:
        pair_df = extend_pair_cache_with_yesno(pair_df, EXPORTS)
    hh = pair_df[pair_df["pair_type"] == "HH"].copy()
    return hh


def summarize_hh(hh: pd.DataFrame, group_col: str) -> pd.DataFrame:
    question_means = (
        hh.groupby(["question_id", group_col, "variant"], dropna=False)["sbert_score"]
        .mean()
        .rename("hh_sbert")
        .reset_index()
    )
    summary = (
        question_means.groupby([group_col, "variant"], dropna=False)["hh_sbert"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "hh_mean", "sem": "hh_sem", "count": "n_questions"})
    )
    return summary


def save_summary_csv(df: pd.DataFrame, name: str) -> None:
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


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


def _draw_bars(ax, summary: pd.DataFrame, group_col: str, title: str) -> None:
    labels = _ordered_labels(summary, group_col)
    counts = _label_counts(summary, group_col)
    x = np.arange(len(labels))
    width = 0.24

    for i, variant in enumerate(VARIANT_ORDER):
        sub = (
            summary[summary["variant"] == variant]
            .set_index(group_col)
            .reindex(labels)
        )
        y = sub["hh_mean"].to_numpy(dtype=float)
        sem = sub["hh_sem"].fillna(0).to_numpy(dtype=float)
        ax.bar(
            x + (i - 1) * width,
            y,
            width=width,
            yerr=1.96 * sem,
            color=VARIANT_COLORS[variant],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            capsize=2.5,
            label=VARIANT_LABELS[variant],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={counts.get(lab, 0)})" for lab in labels], rotation=35, ha="right", fontsize=9)
    ax.set_xlabel(group_col, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title=None, ncol=1, fontsize=8, frameon=True, loc="best")


def summarize_pair_metric(hh: pd.DataFrame, group_col: str, metric_col: str, out_col: str) -> pd.DataFrame:
    question_means = (
        hh.groupby(["question_id", group_col, "variant"], dropna=False)[metric_col]
        .mean()
        .rename(out_col)
        .reset_index()
    )
    summary = (
        question_means.groupby([group_col, "variant"], dropna=False)[out_col]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "hh_mean", "sem": "hh_sem", "count": "n_questions"})
    )
    return summary


def summarize_entropy(group_col: str, include_yesno: bool) -> pd.DataFrame:
    human = pd.read_csv(EXPORTS / "responses_human.csv")
    if not include_yesno:
        text_qids = pd.read_parquet(EXPORTS / "pair_cache.parquet")["question_id"].drop_duplicates().tolist()
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
    summary = (
        q_level.groupby([group_col, "variant"], dropna=False)["entropy"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "hh_mean", "sem": "hh_sem", "count": "n_questions"})
    )
    return summary


def export_metric_figure(entity_summary: pd.DataFrame, op_summary: pd.DataFrame, q_tag: str, metric_label: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), sharey=True)
    _draw_bars(
        axes[0],
        entity_summary,
        "ent",
        "By entity type",
    )
    _draw_bars(
        axes[1],
        op_summary,
        "op",
        "By operation type",
    )
    axes[0].set_ylabel(metric_label, fontsize=11)
    axes[1].set_ylabel("")
    fig.suptitle(f"Human-only {metric_label} across variants ({q_tag})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, OUT_DIR, f"{stem}_{q_tag}.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free_text_only", action="store_true")
    args = parser.parse_args()

    include_yesno = not args.free_text_only
    hh = load_hh_pairs(include_yesno=include_yesno)
    q_tag = f"q{hh['question_id'].nunique()}" + ("_yesno" if include_yesno else "")

    sbert_ent = summarize_pair_metric(hh, "ent", "sbert_score", "sbert")
    sbert_op = summarize_pair_metric(hh, "op", "sbert_score", "sbert")
    save_summary_csv(sbert_ent, f"human_hh_sbert_by_entity_{q_tag}.csv")
    save_summary_csv(sbert_op, f"human_hh_sbert_by_op_{q_tag}.csv")
    export_metric_figure(sbert_ent, sbert_op, q_tag, "HH SBERT", "human_hh_sbert_entity_op")

    chrf_ent = summarize_pair_metric(hh, "ent", "chrf_score", "chrf")
    chrf_op = summarize_pair_metric(hh, "op", "chrf_score", "chrf")
    save_summary_csv(chrf_ent, f"human_hh_chrf_by_entity_{q_tag}.csv")
    save_summary_csv(chrf_op, f"human_hh_chrf_by_op_{q_tag}.csv")
    export_metric_figure(chrf_ent, chrf_op, q_tag, "HH chrF", "human_hh_chrf_entity_op")

    ent_ent = summarize_entropy("ent", include_yesno=include_yesno)
    ent_op = summarize_entropy("op", include_yesno=include_yesno)
    save_summary_csv(ent_ent, f"human_entropy_by_entity_{q_tag}.csv")
    save_summary_csv(ent_op, f"human_entropy_by_op_{q_tag}.csv")
    export_metric_figure(ent_ent, ent_op, q_tag, "Answer entropy (bits)", "human_entropy_entity_op")


if __name__ == "__main__":
    main()
