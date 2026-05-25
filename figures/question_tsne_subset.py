"""
Export t-SNE plots of the 1k VQA question pool with the 113-question human-study
subset highlighted.

Outputs:
  figures/question_tsne/
    subset_tsne_entities.png
    subset_tsne_operations.png

Behavior:
  - Fit t-SNE once on all 1k question embeddings (all-mpnet-base-v2).
  - Show all 1k points, colored by entity/op group.
  - Highlight the 113-question study subset with larger, outlined markers.
  - Annotate example questions nearest each highlighted group's centroid.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from utils.constants import (
    TSNE_ENTITY_COLORS,
    TSNE_OP_COLORS,
    TSNE_POOL_POINT_STYLE,
    TSNE_SUBSET_POINT_STYLE,
)
from helpers import clear_output_plots

parser = argparse.ArgumentParser()
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete existing plot files in the output folder before exporting.",
)
args = parser.parse_args()

OUT_DIR = ROOT / "figures/question_tsne"
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)
EXPORTS = ROOT / "analysis/session2/exports"
SEM_PATH = ROOT / "dataset/vqa/vqa1k_semantics.jsonl"
EMB_CACHE = EXPORTS / "question_embeddings_all-mpnet-base-v2_vqa1k.npz"
TSNE_CACHE = EXPORTS / "question_tsne_all-mpnet-base-v2_vqa1k.csv"
HF_CACHE = ROOT.parent / ".cache/hf"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})


def load_semantics() -> pd.DataFrame:
    sem = pd.read_json(SEM_PATH, lines=True)[["question_id", "question", "ent", "op", "w"]].copy()
    subset_qids = set(pd.read_csv(EXPORTS / "responses_human.csv")["question_id"].unique().tolist())
    sem["is_subset"] = sem["question_id"].isin(subset_qids)
    return sem


def load_question_embeddings(questions: list[str]) -> np.ndarray:
    if EMB_CACHE.exists():
        cache = np.load(EMB_CACHE, allow_pickle=True)
        cached_qs = cache["questions"].tolist()
        if cached_qs == questions:
            return cache["embeddings"]

    print("Embedding 1k questions...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", cache_folder=str(HF_CACHE))
    embs = model.encode(
        questions,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    np.savez_compressed(EMB_CACHE, questions=np.array(questions, dtype=object), embeddings=embs)
    print(f"Saved embedding cache → {EMB_CACHE}")
    return embs


def load_tsne(df: pd.DataFrame) -> pd.DataFrame:
    if TSNE_CACHE.exists():
        cached = pd.read_csv(TSNE_CACHE)
        if set(cached["question_id"]) == set(df["question_id"]):
            return df.merge(cached[["question_id", "x", "y"]], on="question_id", how="left")

    embs = load_question_embeddings(df["question"].tolist())
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, init="pca", max_iter=1000)
    coords = tsne.fit_transform(embs)
    out = df.copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    out[["question_id", "x", "y"]].to_csv(TSNE_CACHE, index=False)
    print(f"Saved t-SNE cache → {TSNE_CACHE}")
    return out


def _label_offsets(n: int) -> list[tuple[float, float]]:
    base = [
        (0, 16), (18, 6), (18, -10), (0, -18),
        (-18, -10), (-18, 6), (28, 16), (-28, 16),
    ]
    return base[:n]


def annotate_cluster_examples(ax, sub: pd.DataFrame, color: str, k: int = 2) -> None:
    if sub.empty:
        return
    center = sub[["x", "y"]].mean().to_numpy()
    pts = sub[["x", "y"]].to_numpy()
    dists = np.linalg.norm(pts - center, axis=1)
    ann = sub.iloc[np.argsort(dists)[: min(k, len(sub))]].copy()
    offsets = _label_offsets(len(ann))
    for (_, row), (dx, dy) in zip(ann.iterrows(), offsets):
        text = row["question"]
        text = text if len(text) <= 52 else text[:49].rstrip() + "..."
        ax.annotate(
            text,
            (row["x"], row["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            color=color,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=0.8, alpha=0.92),
            arrowprops=dict(arrowstyle="-", lw=0.7, color=color, alpha=0.8),
            zorder=6,
        )


def plot_subset(df: pd.DataFrame, group_col: str, colors: dict[str, str], title: str, out_name: str) -> None:
    pool = df[df[group_col].notna()].copy()
    subset = pool[pool["is_subset"]].copy()
    order = subset[group_col].value_counts().index.tolist()

    fig, ax = plt.subplots(figsize=(9.8, 7.2))

    for grp in order:
        color = colors.get(grp, "#888888")
        pool_grp = pool[pool[group_col] == grp]
        sub_grp = subset[subset[group_col] == grp]

        ax.scatter(
            pool_grp["x"], pool_grp["y"],
            c=color,
            s=TSNE_POOL_POINT_STYLE["size"],
            alpha=TSNE_POOL_POINT_STYLE["alpha"],
            marker=TSNE_POOL_POINT_STYLE["marker"],
            edgecolors=TSNE_POOL_POINT_STYLE["edgecolor"],
            linewidths=TSNE_POOL_POINT_STYLE["linewidth"],
            zorder=1,
        )
        ax.scatter(
            sub_grp["x"], sub_grp["y"],
            c=color,
            s=TSNE_SUBSET_POINT_STYLE["size"],
            alpha=TSNE_SUBSET_POINT_STYLE["alpha"],
            marker=TSNE_SUBSET_POINT_STYLE["marker"],
            edgecolors=TSNE_SUBSET_POINT_STYLE["edgecolor"],
            linewidths=TSNE_SUBSET_POINT_STYLE["linewidth"],
            zorder=4,
        )
        annotate_cluster_examples(ax, sub_grp, color=color, k=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)

    marker_handles = [
        mlines.Line2D(
            [], [], color="#666666", marker=TSNE_POOL_POINT_STYLE["marker"],
            linestyle="None", markersize=4, alpha=0.4, label="1k pool"
        ),
        mlines.Line2D(
            [], [], color="#666666", marker=TSNE_SUBSET_POINT_STYLE["marker"],
            linestyle="None", markersize=7,
            markeredgecolor=TSNE_SUBSET_POINT_STYLE["edgecolor"],
            markeredgewidth=TSNE_SUBSET_POINT_STYLE["linewidth"],
            label=f"study subset (N={len(subset)})"
        ),
    ]
    group_handles = [
        mlines.Line2D([], [], color=colors.get(grp, "#888888"), marker="o",
                      linestyle="None", markersize=6, label=grp)
        for grp in order
    ]
    ax.legend(
        handles=marker_handles + group_handles,
        fontsize=7.3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=5,
        frameon=True,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [question_tsne] {out_name}")


def main() -> None:
    print("Loading semantics and 113-question study subset...")
    df = load_semantics()
    print(f"Pool: {len(df)} questions | subset: {int(df['is_subset'].sum())}")
    df = load_tsne(df)

    plot_subset(
        df,
        group_col="ent",
        colors=TSNE_ENTITY_COLORS,
        title="Question Embedding Space — 113-Question Study Subset by Entity",
        out_name="subset_tsne_entities.png",
    )
    plot_subset(
        df,
        group_col="op",
        colors=TSNE_OP_COLORS,
        title="Question Embedding Space — 113-Question Study Subset by Operation",
        out_name="subset_tsne_operations.png",
    )
    print(f"\nDone. Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
