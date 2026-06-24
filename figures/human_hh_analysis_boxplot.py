"""
Human-only agreement/diversity boxplots across variants.

Exports separate 2-panel boxplot figures for:
  - HH SBERT
  - HH chrF
  - answer entropy

Each figure has:
  - left: grouped by entity type
  - right: grouped by operation type

Default uses the yes/no-inclusive q113 set.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from figures.helpers import save_fig
from utils.constants import VARIANT_COLORS, VARIANT_LABELS, VARIANT_ORDER
from utils.vqa import preprocess_answer

EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "figures/human_hh_analysis_boxplot"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_hh_pairs(include_yesno: bool) -> pd.DataFrame:
    pair_df = pd.read_parquet(EXPORTS / "pair_cache.parquet")
    if not include_yesno:
        return pair_df[pair_df["pair_type"] == "HH"].copy()

    yesno = pd.read_csv(EXPORTS / "answer_pairs_yesno.csv")

    cache = np.load(EXPORTS / "embeddings_all-mpnet-base-v2.npz", allow_pickle=True)
    strings = cache["strings"].tolist()
    vectors = cache["vectors"]
    lookup = {s: vectors[i] for i, s in enumerate(strings)}

    a1 = yesno["answer_1"].fillna("").apply(preprocess_answer)
    a2 = yesno["answer_2"].fillna("").apply(preprocess_answer)

    sbert_scores = []
    for s1, s2 in zip(a1, a2):
        v1, v2 = lookup.get(s1), lookup.get(s2)
        if v1 is None or v2 is None:
            sbert_scores.append(np.nan)
        else:
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            sbert_scores.append(float(np.dot(v1, v2) / (n1 * n2)) if n1 and n2 else 0.0)
    yesno["sbert_score"] = sbert_scores

    try:
        from sacrebleu.metrics import CHRF
        _chrf = CHRF(beta=2)
        yesno["chrf_score"] = [
            _chrf.sentence_score(s1, [s2]).score / 100.0
            for s1, s2 in zip(a1, a2)
        ]
    except Exception:
        yesno["chrf_score"] = np.nan

    cols = ["question_id", "variant", "ent", "op", "pair_type", "sbert_score", "chrf_score"]
    hh = pd.concat(
        [
            pair_df[pair_df["pair_type"] == "HH"][cols],
            yesno[yesno["pair_type"] == "HH"][cols],
        ],
        ignore_index=True,
    )
    return hh


def build_pair_metric_df(hh: pd.DataFrame, group_col: str, metric_col: str, out_col: str) -> pd.DataFrame:
    return (
        hh.groupby(["question_id", group_col, "variant"], dropna=False)[metric_col]
        .mean()
        .rename(out_col)
        .reset_index()
    )


def build_entropy_df(group_col: str, include_yesno: bool) -> pd.DataFrame:
    human = pd.read_csv(EXPORTS / "responses_human.csv")
    if not include_yesno:
        text_qids = pd.read_parquet(EXPORTS / "pair_cache.parquet")["question_id"].drop_duplicates().tolist()
        human = human[human["question_id"].isin(text_qids)].copy()

    def answer_entropy(responses: pd.Series) -> float:
        counts = responses.value_counts()
        probs = counts / counts.sum()
        return float(sp_entropy(probs, base=2))

    return (
        human.groupby(["question_id", group_col, "variant"], dropna=False)["response"]
        .apply(answer_entropy)
        .rename("entropy")
        .reset_index()
    )


def _ordered_labels(df: pd.DataFrame, group_col: str, value_col: str) -> list[str]:
    sub_c = df[df["variant"] == "C"].groupby(group_col)[value_col].mean().sort_values(ascending=False)
    return sub_c.index.astype(str).tolist()


def _label_counts(df: pd.DataFrame, group_col: str) -> dict[str, int]:
    counts = df[df["variant"] == "C"].groupby(group_col)["question_id"].nunique().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def draw_boxplot(ax, df: pd.DataFrame, group_col: str, value_col: str, title: str) -> None:
    order = _ordered_labels(df, group_col, value_col)
    counts = _label_counts(df, group_col)
    sns.boxplot(
        data=df,
        x=group_col,
        y=value_col,
        hue="variant",
        order=order,
        hue_order=VARIANT_ORDER,
        palette=[VARIANT_COLORS[v] for v in VARIANT_ORDER],
        ax=ax,
        width=0.75,
        fliersize=2,
        linewidth=0.9,
    )
    ax.set_xlabel(group_col, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticklabels([f"{lab}\n(n={counts.get(lab, 0)})" for lab in order], rotation=35, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title=None, fontsize=8, frameon=True, loc="best")


def export_metric(df_ent: pd.DataFrame, df_op: pd.DataFrame, value_col: str, y_label: str, q_tag: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4), sharey=False)
    draw_boxplot(axes[0], df_ent, "ent", value_col, "By entity type")
    draw_boxplot(axes[1], df_op, "op", value_col, "By operation type")
    axes[0].set_ylabel(y_label, fontsize=11)
    axes[1].set_ylabel("")
    fig.suptitle(f"Human-only {y_label} across variants ({q_tag})", fontsize=13)
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

    sbert_ent = build_pair_metric_df(hh, "ent", "sbert_score", "sbert")
    sbert_op = build_pair_metric_df(hh, "op", "sbert_score", "sbert")
    export_metric(sbert_ent, sbert_op, "sbert", "HH SBERT", q_tag, "human_hh_sbert_entity_op_boxplot")

    chrf_ent = build_pair_metric_df(hh, "ent", "chrf_score", "chrf")
    chrf_op = build_pair_metric_df(hh, "op", "chrf_score", "chrf")
    export_metric(chrf_ent, chrf_op, "chrf", "HH chrF", q_tag, "human_hh_chrf_entity_op_boxplot")

    ent_ent = build_entropy_df("ent", include_yesno=include_yesno)
    ent_op = build_entropy_df("op", include_yesno=include_yesno)
    export_metric(ent_ent, ent_op, "entropy", "Answer entropy (bits)", q_tag, "human_entropy_entity_op_boxplot")


if __name__ == "__main__":
    main()
