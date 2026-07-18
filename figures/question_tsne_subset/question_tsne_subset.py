"""
Export t-SNE plots of the 1k VQA question pool with the 113-question
human-study subset highlighted.

Outputs:
  figures/question_tsne_subset/subset_tsne_entities.png
  figures/question_tsne_subset/subset_tsne_operations.png

Behavior:
  - Fit t-SNE once on all 1k question embeddings (all-mpnet-base-v2).
  - Show all 1k points, colored by entity/op group.
  - Highlight the 113-question study subset with larger markers.
  - Annotate one central question per cluster (nearest to centroid).
"""

from __future__ import annotations

import sys
import argparse
import textwrap
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from utils.constants import (
    TSNE_ENTITY_COLORS,
    TSNE_OP_COLORS,
    TSNE_POOL_POINT_STYLE,
    TSNE_SUBSET_POINT_STYLE,
    TSNE_LABEL_STYLE,
    TSNE_LEGEND_STYLE,
    TSNE_FIGURE_STYLE,
)
from helpers import clear_output_plots

parser = argparse.ArgumentParser()
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete existing plot files in the output folder before exporting.",
)
args = parser.parse_args()

OUT_DIR = ROOT / "figures/question_tsne_subset"
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

# Use a moderately smaller canvas than the default style so labels render
# larger in the appendix, but keep enough width for the legend to wrap cleanly.
TSNE_FIGURE_STYLE["single_figsize"] = (8.1, 4.9)
TSNE_LEGEND_STYLE["ncol_max"] = 6
TSNE_LEGEND_STYLE["bbox_y"] = -0.08
TSNE_LABEL_STYLE["fontweight"] = "bold"
TSNE_LABEL_STYLE["offset_radius_points"] = (72, 100, 130, 162, 196, 234)
TSNE_LABEL_STYLE["offset_angle_jitter_deg"] = (-75, -55, -38, -22, -8, 8, 22, 38, 55, 75, 92, -92, 110, -110)
TSNE_LABEL_STYLE["min_box_gap_px"] = 32.0

ENTITY_LABEL_ADJUSTMENTS = {
    # Centroid (data units): animal=(20.8,-7.5), food=(-11.3,-29.1), object=(-5.5,2.4),
    # other=(5.8,6.2), person=(1.2,-7.8), place=(-10.5,21.0), product=(2.8,-5.3),
    # text=(15.8,21.1), vehicle=(-11.8,2.8)
    "animal":  {"absolute": (185.0,  -15.0)},   # far right (rightmost centroid)
    "food":    {"absolute": (120.0,   80.0)},   # upper right (centroid at bottom, push up+right)
    "object":  {"absolute": (-170.0,  55.0)},   # far left-up (separate from vehicle)
    "other":   {"absolute": ( 95.0,   95.0)},   # upper right
    "person":  {"absolute": (120.0,  -50.0)},   # right (not down, avoids legend)
    "place":   {"absolute": (-155.0,  80.0)},   # upper left (centroid top-left)
    "product": {"absolute": (160.0,   45.0)},   # right
    "text":    {"absolute": (130.0,  110.0)},   # upper right (centroid top-right)
    "vehicle": {"absolute": (-150.0, -80.0)},   # lower left (separate from object)
}

OP_LABEL_ADJUSTMENTS = {
    # Centroid (data units): act=(12.1,-9.8), attr=(5.6,-4.0), cause=(2.2,4.0),
    # comp=(-1.1,-2.8), count=(-10.6,-4.8), exist=(-13.6,-19.1), ident=(8.2,-18.1),
    # know=(-2.8,1.9), spat=(-2.6,-1.3), temp=(0.3,13.7), text=(13.8,20.8)
    "exist":  {"absolute": (-155.0,  90.0)},    # upper-left (centroid far left-below, push up to stay on-screen)
    "ident":  {"absolute": ( 175.0,  -25.0)},   # far right lower (below attr)
    "count":  {"absolute": (-170.0,  -20.0)},   # left (centroid left-below)
    "cause":  {"absolute": ( -75.0,  145.0)},   # upper-left (central cluster)
    "attr":   {"absolute": ( 155.0,   90.0)},   # right-upper (above ident)
    "know":   {"absolute": (-145.0,   65.0)},   # upper-left (central cluster)
    "text":   {"absolute": ( 130.0,  135.0)},   # upper-right (centroid top-right)
    "temp":   {"absolute": (  40.0,  160.0)},   # upper (centroid top-center)
    "spat":   {"absolute": (-155.0,  -25.0)},   # left at mid-height (separate from comp below)
    "act":    {"absolute": ( 175.0,  -60.0)},   # far right-lower (centroid right-below)
    "comp":   {"absolute": ( -45.0, -115.0)},   # lower-center (below spat)
}

TSNE_POOL_POINT_STYLE["size"] = 13
TSNE_POOL_POINT_STYLE["alpha"] = 0.18


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
    from sentence_transformers import SentenceTransformer

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


def wrap_question_text(question: str) -> str:
    words = question.split()
    if not words:
        return question
    if len(words) <= TSNE_LABEL_STYLE["long_question_words"]:
        width = min(
            TSNE_LABEL_STYLE["max_wrap_width"],
            max(TSNE_LABEL_STYLE["min_wrap_width"], len(question)),
        )
        return "\n".join(textwrap.wrap(question, width=width))

    half_words = max(1, len(words) // 2)
    lines = [" ".join(words[:half_words]), " ".join(words[half_words:])]
    wrapped_lines = []
    for line in lines:
        width = min(
            TSNE_LABEL_STYLE["max_wrap_width"],
            max(TSNE_LABEL_STYLE["min_wrap_width"], len(line)),
        )
        wrapped_lines.extend(textwrap.wrap(line, width=width) or [line])
    return "\n".join(wrapped_lines)


def estimate_text_box_px(text: str, dpi: float) -> tuple[float, float]:
    lines = text.split("\n")
    max_chars = max((len(line) for line in lines), default=1)
    fontsize = TSNE_LABEL_STYLE["fontsize"]
    px_per_pt = dpi / 72.0
    width = max_chars * fontsize * 0.58 * px_per_pt + 24
    height = len(lines) * fontsize * 1.55 * px_per_pt + 18
    return width, height


def boxes_overlap(box_a, box_b, gap_px: float) -> bool:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    return not (
        ax1 + gap_px < bx0 or
        bx1 + gap_px < ax0 or
        ay1 + gap_px < by0 or
        by1 + gap_px < ay0
    )


def choose_annotation_offset(
    ax,
    anchor_x: float,
    anchor_y: float,
    text: str,
    map_center: tuple[float, float],
    used_boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    cx, cy = map_center
    base_angle = np.degrees(np.arctan2(anchor_y - cy, anchor_x - cx))
    radii = TSNE_LABEL_STYLE["offset_radius_points"]
    jitters = TSNE_LABEL_STYLE["offset_angle_jitter_deg"]
    gap_px = TSNE_LABEL_STYLE["min_box_gap_px"]
    dpi = ax.figure.dpi
    width_px, height_px = estimate_text_box_px(text, dpi)
    anchor_disp = ax.transData.transform((anchor_x, anchor_y))
    px_per_pt = dpi / 72.0

    candidates = []
    for radius in radii:
        for jitter in jitters:
            angle = np.radians(base_angle + jitter)
            dx = np.cos(angle) * radius
            dy = np.sin(angle) * radius
            candidates.append((dx, dy))

    for dx, dy in candidates:
        center_x = anchor_disp[0] + dx * px_per_pt
        center_y = anchor_disp[1] + dy * px_per_pt
        box = (
            center_x - width_px / 2.0,
            center_y - height_px / 2.0,
            center_x + width_px / 2.0,
            center_y + height_px / 2.0,
        )
        if all(not boxes_overlap(box, used_box, gap_px) for used_box in used_boxes):
            used_boxes.append(box)
            return dx, dy

    dx, dy = candidates[len(used_boxes) % len(candidates)]
    center_x = anchor_disp[0] + dx * px_per_pt
    center_y = anchor_disp[1] + dy * px_per_pt
    used_boxes.append((
        center_x - width_px / 2.0,
        center_y - height_px / 2.0,
        center_x + width_px / 2.0,
        center_y + height_px / 2.0,
    ))
    return dx, dy


def annotate_centroid(
    ax,
    sub: pd.DataFrame,
    color: str,
    used_boxes: list[tuple[float, float, float, float]],
    map_center: tuple[float, float],
    *,
    group_col: str,
    group_name: str,
) -> None:
    """Annotate only the single question nearest to the cluster centroid."""
    if sub.empty:
        return
    center = sub[["x", "y"]].mean().to_numpy()
    pts = sub[["x", "y"]].to_numpy()
    dists = np.linalg.norm(pts - center, axis=1)
    row = sub.iloc[np.argmin(dists)]
    text = wrap_question_text(row["question"])
    dx, dy = choose_annotation_offset(
        ax,
        float(row["x"]),
        float(row["y"]),
        text,
        map_center,
        used_boxes,
    )
    adjust = {}
    if group_col == "ent":
        adjust = ENTITY_LABEL_ADJUSTMENTS.get(group_name, {})
    elif group_col == "op":
        adjust = OP_LABEL_ADJUSTMENTS.get(group_name, {})
    if "absolute" in adjust:
        dx, dy = adjust["absolute"]
    else:
        scale = float(adjust.get("scale", 1.0))
        dx *= scale
        dy *= scale
        dx += float(adjust.get("dx", 0.0))
        dy += float(adjust.get("dy", 0.0))
    ax.annotate(
        text,
        (row["x"], row["y"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=TSNE_LABEL_STYLE["fontsize"],
        fontweight=TSNE_LABEL_STYLE["fontweight"],
        color=TSNE_LABEL_STYLE["color"],
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc=mcolors.to_rgba(color, TSNE_LABEL_STYLE["bbox_alpha"]),
            ec="none",
        ),
        arrowprops=dict(
            arrowstyle="-",
            lw=TSNE_LABEL_STYLE["arrow_lw"],
            color=color,
            alpha=TSNE_LABEL_STYLE["arrow_alpha"],
        ),
        zorder=6,
        clip_on=False,
    )


def plot_panel(ax, df: pd.DataFrame, group_col: str, colors: dict[str, str]) -> None:
    pool = df[df[group_col].notna()].copy()
    subset = pool[pool["is_subset"]].copy()
    order = sorted(
        pool[group_col].unique(),
        key=lambda g: np.arctan2(
            pool[pool[group_col] == g]["y"].mean() - pool["y"].mean(),
            pool[pool[group_col] == g]["x"].mean() - pool["x"].mean(),
        ),
    )
    used_boxes: list[tuple[float, float, float, float]] = []
    map_center = (float(pool["x"].mean()), float(pool["y"].mean()))

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
            edgecolors="none",
            linewidths=0,
            zorder=4,
        )
        annotate_centroid(
            ax,
            sub_grp,
            color=color,
            used_boxes=used_boxes,
            map_center=map_center,
            group_col=group_col,
            group_name=grp,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend: only group colors (no pool/subset markers)
    subset_groups = [g for g in order if g in subset[group_col].values]
    group_handles = [
        mlines.Line2D([], [], color=colors.get(grp, "#888888"), marker="o",
                      linestyle="None", markersize=TSNE_LEGEND_STYLE["markersize"], label=grp)
        for grp in subset_groups
    ]
    ax.legend(
        handles=group_handles,
        fontsize=TSNE_LEGEND_STYLE["fontsize"],
        loc="upper center",
        bbox_to_anchor=(0.5, TSNE_LEGEND_STYLE["bbox_y"]),
        ncol=min(len(group_handles), TSNE_LEGEND_STYLE["ncol_max"]),
        frameon=True,
        handletextpad=TSNE_LEGEND_STYLE["handletextpad"],
        columnspacing=TSNE_LEGEND_STYLE["columnspacing"],
    )


def save_single_panel(df: pd.DataFrame, group_col: str, colors: dict[str, str], filename: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=TSNE_FIGURE_STYLE["single_figsize"])
    plot_panel(ax, df, group_col=group_col, colors=colors)
    fig.tight_layout(pad=0.6)
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=TSNE_FIGURE_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [question_tsne] {filename}")


def main() -> None:
    print("Loading semantics and 113-question study subset...")
    df = load_semantics()
    print(f"Pool: {len(df)} questions | subset: {int(df['is_subset'].sum())}")
    df = load_tsne(df)

    save_single_panel(df, group_col="ent", colors=TSNE_ENTITY_COLORS, filename="subset_tsne_entities.png")
    save_single_panel(df, group_col="op", colors=TSNE_OP_COLORS, filename="subset_tsne_operations.png")
    print(f"\nDone. Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
