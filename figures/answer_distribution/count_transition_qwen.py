from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from analysis.utils.vqa import VQAAnswerMapper, extract_number, preprocess_answer


EXPORTS = ROOT / "analysis" / "session2" / "exports"
OUT_DIR = ROOT / "figures" / "answer_distribution" / "transition"
LATEX_OUT_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "answer_distribution" / "transition"

QWEN_MODELS = [
    "Qwen3-VL-8B",
    "Qwen3-VL-8B (LM)",
    "Qwen3-8B",
    "Qwen3-8B (think)",
]
INTERNVL_MODELS = [
    "InternVL-8B",
    "InternVL-8B (LM)",
    "Qwen3-8B",
    "Qwen3-8B (think)",
]
LLAVA_MODELS = [
    "LLaVA-Mistral",
    "LLaVA-Mistral (LM)",
    "Qwen3-8B",
    "Qwen3-8B (think)",
]
QWEN_VL_MODELS = [
    "Qwen3-VL-8B",
    "Qwen3-VL-8B (LM)",
    "Qwen3-8B",
    "Qwen3-8B (think)",
]
ALL_MODELS = [
    "Qwen3-VL-8B",
    "LLaVA-1.5-7B",
    "LLaVA-Mistral",
    "LLaVA-Vicuna",
    "InternVL-8B",
    "Qwen3-VL-8B (LM)",
    "LLaVA-1.5 (LM)",
    "LLaVA-Mistral (LM)",
    "LLaVA-Vicuna (LM)",
    "InternVL-8B (LM)",
    "Qwen3-8B",
    "Qwen2.5-7B-Instruct",
    "Mistral-7B",
    "Vicuna-7B",
    "Qwen3-8B (think)",
]
GROUP_COLORS = {
    "VLM": "#9E4A3A",
    "VLM decoder": "#C0843D",
    "LLM": "#5E8C61",
    "LLM (think)": "#5B84B8",
}
GROUP_MARKERS = {
    "VLM": "o",
    "VLM decoder": "o",
    "LLM": "o",
    "LLM (think)": "o",
}
MODEL_TO_GROUP = {
    "Qwen3-VL-8B": "VLM",
    "LLaVA-1.5-7B": "VLM",
    "LLaVA-Mistral": "VLM",
    "LLaVA-Vicuna": "VLM",
    "InternVL-8B": "VLM",
    "Qwen3-VL-8B (LM)": "VLM decoder",
    "LLaVA-1.5 (LM)": "VLM decoder",
    "LLaVA-Mistral (LM)": "VLM decoder",
    "LLaVA-Vicuna (LM)": "VLM decoder",
    "InternVL-8B (LM)": "VLM decoder",
    "Qwen3-8B": "LLM",
    "Qwen2.5-7B-Instruct": "LLM",
    "Mistral-7B": "LLM",
    "Vicuna-7B": "LLM",
    "Qwen3-8B (think)": "LLM (think)",
}
MODEL_SHORT = {
    "Qwen3-VL-8B": "Qwen3-VL-8B",
    "LLaVA-1.5-7B": "LLaVA-1.5-7B",
    "LLaVA-Mistral": "LLaVA-Mistral",
    "LLaVA-Vicuna": "LLaVA-Vicuna",
    "InternVL-8B": "InternVL-8B",
    "Qwen3-VL-8B (LM)": "Qwen3-VL-8B (LM)",
    "LLaVA-1.5 (LM)": "LLaVA-1.5 (LM)",
    "LLaVA-Mistral (LM)": "LLaVA-Mistral (LM)",
    "LLaVA-Vicuna (LM)": "LLaVA-Vicuna (LM)",
    "InternVL-8B (LM)": "InternVL-8B (LM)",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Mistral-7B": "Mistral-7B",
    "Vicuna-7B": "Vicuna-7B",
    "Qwen3-8B (think)": "Qwen3-8B (think)",
}
ALL_MODEL_ROWS = [
    sorted([m for m in ALL_MODELS if MODEL_TO_GROUP[m] == "VLM"]),
    sorted([m for m in ALL_MODELS if MODEL_TO_GROUP[m] == "VLM decoder"]),
    sorted([m for m in ALL_MODELS if MODEL_TO_GROUP[m] in {"LLM", "LLM (think)"}]),
]
X_LABELS = [str(i) for i in range(11)] + ["11+"]
BOOT_N = 400
RNG = np.random.default_rng(7)


def load_qid_to_answer_type() -> dict[int, str]:
    mapper = VQAAnswerMapper()
    mapper._load()
    return {
        int(qid): ann.get("answer_type", "other")
        for qid, ann in mapper.annotations.items()
    }


def _filter_variants(df: pd.DataFrame, variants: tuple[str, ...] | None) -> pd.DataFrame:
    if variants is None:
        return df.copy()
    return df[df["variant"].isin(variants)].copy()


def load_model_condition(
    condition: str,
    qid2atype: dict[int, str],
    variants: tuple[str, ...] | None = ("C",),
) -> pd.DataFrame:
    df = pd.read_csv(EXPORTS / f"responses_model_{condition}.csv")
    df = _filter_variants(df, variants)
    df = df[df["model"].isin(ALL_MODELS)].copy()
    df["qid"] = df["question_id"].astype(int)
    df["answer_type"] = df["qid"].map(qid2atype).fillna("other")
    df = df[df["answer_type"] == "number"].copy()
    df["clean"] = df["response"].apply(lambda x: preprocess_answer(x, strip_think=True).lower())
    df["number"] = df["clean"].apply(extract_number)
    df["group"] = df["model"].map(MODEL_TO_GROUP)
    return df


def load_human(
    qid2atype: dict[int, str],
    variants: tuple[str, ...] | None = ("C",),
) -> pd.DataFrame:
    df = pd.read_csv(EXPORTS / "responses_human.csv")
    df = _filter_variants(df, variants)
    df["qid"] = df["question_id"].astype(int)
    df["answer_type"] = df["qid"].map(qid2atype).fillna("other")
    df = df[df["answer_type"] == "number"].copy()
    response_col = "response" if "response" in df.columns else "answer_en"
    df["clean"] = df[response_col].apply(lambda x: preprocess_answer(x, strip_think=True).lower())
    df["number"] = df["clean"].apply(extract_number)
    return df


def integer_bin(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        v = int(round(float(value)))
    except Exception:
        return None
    if v < 0:
        return None
    if v >= 11:
        return "11+"
    return str(v)


def count_proportions(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    work = df.copy()
    work["bin"] = work["number"].apply(integer_bin)
    work = work.dropna(subset=["bin"]).copy()
    counts = work.groupby([group_col, "bin"]).size().rename("n").reset_index()
    counts["prop"] = counts["n"] / counts.groupby(group_col)["n"].transform("sum")
    out = (
        counts.pivot(index=group_col, columns="bin", values="prop")
        .reindex(columns=X_LABELS, fill_value=0.0)
        .fillna(0.0)
    )
    return out


def bin_distribution(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    work["bin"] = work["number"].apply(integer_bin)
    work = work.dropna(subset=["bin"]).copy()
    if work.empty:
        return pd.Series(0.0, index=X_LABELS, dtype=float)
    counts = work["bin"].value_counts(normalize=True)
    return counts.reindex(X_LABELS, fill_value=0.0).astype(float)


def bootstrap_bin_ci(df: pd.DataFrame, n_boot: int = BOOT_N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = df.copy()
    work["bin"] = work["number"].apply(integer_bin)
    work = work.dropna(subset=["bin"]).copy()
    if work.empty:
        zeros = np.zeros(len(X_LABELS), dtype=float)
        return zeros, zeros, zeros
    qids = work["qid"].dropna().astype(int).unique()
    if len(qids) == 0:
        base = bin_distribution(work).to_numpy(dtype=float)
        return base, base, base
    by_qid = {qid: grp.copy() for qid, grp in work.groupby("qid", sort=False)}
    boot = np.zeros((n_boot, len(X_LABELS)), dtype=float)
    for idx in range(n_boot):
        sampled_qids = RNG.choice(qids, size=len(qids), replace=True)
        sampled = pd.concat([by_qid[int(qid)] for qid in sampled_qids], ignore_index=True)
        boot[idx] = bin_distribution(sampled).to_numpy(dtype=float)
    center = bin_distribution(work).to_numpy(dtype=float)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return center, lo, hi


def bootstrap_bin_ci_clustered(
    df: pd.DataFrame,
    cluster_cols: list[str],
    n_boot: int = BOOT_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = df.copy()
    work["bin"] = work["number"].apply(integer_bin)
    work = work.dropna(subset=["bin"]).copy()
    if work.empty:
        zeros = np.zeros(len(X_LABELS), dtype=float)
        return zeros, zeros, zeros
    cluster_df = work[cluster_cols].drop_duplicates().reset_index(drop=True)
    if cluster_df.empty:
        base = bin_distribution(work).to_numpy(dtype=float)
        return base, base, base
    groups = {(tuple(key) if isinstance(key, tuple) else (key,)): grp.copy() for key, grp in work.groupby(cluster_cols, sort=False)}
    cluster_keys = [tuple(row) for row in cluster_df.itertuples(index=False, name=None)]
    boot = np.zeros((n_boot, len(X_LABELS)), dtype=float)
    for idx in range(n_boot):
        sampled_keys = RNG.choice(len(cluster_keys), size=len(cluster_keys), replace=True)
        sampled = pd.concat([groups[cluster_keys[i]] for i in sampled_keys], ignore_index=True)
        boot[idx] = bin_distribution(sampled).to_numpy(dtype=float)
    center = bin_distribution(work).to_numpy(dtype=float)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return center, lo, hi


def draw_condition_line(
    ax: plt.Axes,
    x: list[int],
    y: np.ndarray | list[float],
    *,
    color: str,
    marker: str,
    filled: bool,
    linewidth: float,
    linestyle: str,
    alpha: float = 0.95,
    markersize: float = 4.5,
    zorder: float | None = None,
    label: str | None = None,
) -> None:
    kwargs: dict[str, object] = {
        "color": color,
        "linestyle": linestyle,
        "linewidth": linewidth,
        "marker": marker,
        "markersize": markersize,
        "alpha": alpha,
    }
    if zorder is not None:
        kwargs["zorder"] = zorder
    if not filled:
        kwargs["markerfacecolor"] = "white"
        kwargs["markeredgewidth"] = 1.4
    if label is not None:
        kwargs["label"] = label
    ax.plot(x, y, **kwargs)


def draw_condition_errorbars(
    ax: plt.Axes,
    x: list[int],
    center: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    color: str,
    marker: str,
    filled: bool,
    linewidth: float,
    linestyle: str,
    markersize: float,
    zorder: float,
) -> None:
    ax.errorbar(
        x,
        center,
        yerr=None,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        markerfacecolor=(color if filled else "white"),
        markeredgecolor=color,
        markeredgewidth=(1.3 if not filled else 1.0),
        zorder=zorder,
        alpha=0.95,
    )


def add_compact_legends(ax: plt.Axes, include_human: bool = True) -> None:
    group_handles: list[Line2D] = []
    if include_human:
        group_handles.append(
            Line2D([0], [0], color="#444444", linewidth=2.2, marker="D", markersize=5.5, label="Human")
        )
    for group_name in ["VLM", "VLM decoder", "LLM", "LLM (think)"]:
        group_handles.append(
            Line2D(
                [0],
                [0],
                color=GROUP_COLORS[group_name],
                linewidth=2.0,
                marker=GROUP_MARKERS[group_name],
                markersize=5.0,
                label=group_name,
            )
        )
    ax.legend(
        handles=group_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        fontsize=8,
        ncol=1,
        handlelength=1.6,
        columnspacing=0.8,
        labelspacing=0.3,
    )


def build_integer_plot(
    blind: pd.DataFrame,
    inst: pd.DataFrame,
    human: pd.DataFrame,
    out_name: str = "vC_qwen_integer.png",
) -> Path:
    blind_qwen = blind[blind["model"].isin(QWEN_MODELS)].copy()
    inst_qwen = inst[inst["model"].isin(QWEN_MODELS)].copy()
    blind_props = count_proportions(blind_qwen, "model")
    inst_props = count_proportions(inst_qwen, "model")
    human_props = count_proportions(human.assign(source="Human"), "source")

    x = list(range(len(X_LABELS)))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))

    human_y = human_props.loc["Human", X_LABELS].tolist()
    ax.plot(x, human_y, color="#444444", linewidth=2.2, marker="D", markersize=5.5, label="Human")

    for model in QWEN_MODELS:
        if model not in blind_props.index or model not in inst_props.index:
            continue
        group_name = MODEL_TO_GROUP[model]
        color = GROUP_COLORS[group_name]
        marker = GROUP_MARKERS[group_name]
        y_blind = blind_props.loc[model, X_LABELS].tolist()
        y_inst = inst_props.loc[model, X_LABELS].tolist()
        draw_condition_line(ax, x, y_inst, color=color, marker=marker, filled=True, linewidth=2.0, linestyle="-", markersize=4.3, zorder=3)
        draw_condition_line(ax, x, y_blind, color=color, marker=marker, filled=False, linewidth=1.8, linestyle=":", markersize=5.4, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=9)
    ax.set_xlabel("Count answer", fontsize=10)
    ax.set_ylabel("Proportion", fontsize=10)
    ax.grid(True, alpha=0.22)
    ax.set_ylim(0, max(0.9, ax.get_ylim()[1] * 1.05))
    add_compact_legends(ax, include_human=True)
    fig.tight_layout()

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def build_integer_plot_by_model(blind: pd.DataFrame, inst: pd.DataFrame, human: pd.DataFrame) -> Path:
    x = list(range(len(X_LABELS)))
    human_y = bin_distribution(human).to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    ymax = human_y.max()

    for ax, model in zip(axes_flat, QWEN_MODELS):
        group_name = MODEL_TO_GROUP[model]
        blind_y = bin_distribution(blind[blind["model"] == model]).to_numpy(dtype=float)
        inst_y = bin_distribution(inst[inst["model"] == model]).to_numpy(dtype=float)
        ymax = max(ymax, blind_y.max(), inst_y.max())
        ax.plot(x, human_y, color="#444444", linewidth=2.0, marker="D", markersize=4.8)
        draw_condition_line(ax, x, blind_y, color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=False, linewidth=1.8, linestyle=":")
        draw_condition_line(ax, x, inst_y, color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=True, linewidth=2.0, linestyle="-")
        ax.set_title(group_name, fontsize=11)
        ax.grid(True, alpha=0.22)

    for idx, ax in enumerate(axes_flat):
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, fontsize=8)
        if idx // 2 == 1:
            ax.set_xlabel("Count answer", fontsize=10)
        if idx % 2 == 0:
            ax.set_ylabel("Proportion", fontsize=10)
        ax.set_ylim(0, max(0.72, ymax * 1.12))

    instruction_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.0, marker="D", markersize=4.8, label="Human"),
        Line2D([0], [0], color="#666666", linewidth=2.0, marker="o", markersize=5.0, markerfacecolor="#666666", label="w/ instruction"),
        Line2D([0], [0], color="#666666", linewidth=1.8, marker="o", markersize=5.0, markerfacecolor="white", markeredgewidth=1.4, label="w/o instruction"),
    ]
    fig.legend(
        handles=instruction_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fontsize=9,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "vC_qwen_integer_by_model.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def build_integer_plot_ci(
    blind: pd.DataFrame,
    inst: pd.DataFrame,
    human: pd.DataFrame,
    out_name: str = "vC_qwen_integer_ci_b400.png",
) -> Path:
    blind_qwen = blind[blind["model"].isin(QWEN_MODELS)].copy()
    inst_qwen = inst[inst["model"].isin(QWEN_MODELS)].copy()
    x = list(range(len(X_LABELS)))
    fig, ax = plt.subplots(figsize=(6.1, 3.8))

    human_center, human_lo, human_hi = bootstrap_bin_ci(human)
    ax.fill_between(x, human_lo, human_hi, color="#444444", alpha=0.12, zorder=1)
    ax.plot(x, human_center, color="#444444", linewidth=2.3, marker="D", markersize=5.2, zorder=3)

    ymax = float(human_hi.max())
    for model in QWEN_MODELS:
        group_name = MODEL_TO_GROUP[model]
        blind_center, blind_lo, blind_hi = bootstrap_bin_ci(blind_qwen[blind_qwen["model"] == model])
        inst_center, inst_lo, inst_hi = bootstrap_bin_ci(inst_qwen[inst_qwen["model"] == model])
        ymax = max(ymax, float(blind_hi.max()), float(inst_hi.max()))
        ax.fill_between(x, blind_lo, blind_hi, color=GROUP_COLORS[group_name], alpha=0.08, zorder=1)
        ax.fill_between(x, inst_lo, inst_hi, color=GROUP_COLORS[group_name], alpha=0.16, zorder=1)
        draw_condition_line(ax, x, inst_center, color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=True, linewidth=2.0, linestyle="-", markersize=4.3, zorder=3)
        draw_condition_line(ax, x, blind_center, color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=False, linewidth=1.7, linestyle=":", markersize=5.4, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=9)
    ax.set_xlabel("Count answer", fontsize=10)
    ax.set_ylabel("Proportion", fontsize=10)
    ax.set_ylim(0, max(0.9, ymax * 1.08))
    ax.grid(True, alpha=0.22)
    add_compact_legends(ax, include_human=True)
    fig.tight_layout()

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def build_integer_plot_all_models(
    blind: pd.DataFrame,
    inst: pd.DataFrame,
    human: pd.DataFrame,
    out_name: str = "vC_all_models.png",
) -> Path:
    x = list(range(len(X_LABELS)))
    human_center, human_lo, human_hi = bootstrap_bin_ci(human)
    nrows = len(ALL_MODEL_ROWS)
    ncols = max(len(row) for row in ALL_MODEL_ROWS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 5.8), sharex=True, sharey=True)
    ymax = float(human_hi.max())

    for r, row_models in enumerate(ALL_MODEL_ROWS):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(row_models):
                ax.axis("off")
                continue
            model = row_models[c]
            group_name = MODEL_TO_GROUP[model]
            blind_y = bin_distribution(blind[blind["model"] == model]).to_numpy(dtype=float)
            inst_y = bin_distribution(inst[inst["model"] == model]).to_numpy(dtype=float)
            ymax = max(ymax, blind_y.max(), inst_y.max())
            ax.fill_between(x, human_lo, human_hi, color="#444444", alpha=0.12, zorder=1)
            ax.plot(x, human_center, color="#444444", linewidth=1.9, marker="D", markersize=4.2)
            draw_condition_line(ax, x, blind_y, color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=False, linewidth=1.6, linestyle=":")
            draw_condition_line(ax, x, inst_y, color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=True, linewidth=1.9, linestyle="-")
            ax.set_title(MODEL_SHORT[model], fontsize=10)
            ax.grid(True, alpha=0.22)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if not ax.axison:
                continue
            ax.set_xticks(x)
            ax.set_xticklabels(X_LABELS, fontsize=8)
            if r == nrows - 1:
                ax.set_xlabel("Count answer", fontsize=10)
            if c == 0:
                ax.set_ylabel("Proportion", fontsize=10)
            ax.set_ylim(0, max(0.86, ymax * 1.08))

    fig.legend(
        handles=[
            Line2D([0], [0], color="#444444", linewidth=2.0, marker="D", markersize=4.6, label="Human"),
            Line2D([0], [0], color="#666666", linewidth=1.8, marker="o", markersize=5.0, markerfacecolor="#666666", label="w/ instruction"),
            Line2D([0], [0], color="#666666", linewidth=1.8, marker="o", markersize=5.0, markerfacecolor="white", markeredgewidth=1.4, label="w/o instruction"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=True,
        fontsize=9,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def build_integer_plot_grouped_ci(
    blind: pd.DataFrame,
    inst: pd.DataFrame,
    human: pd.DataFrame,
    out_name: str = "vC_grouped_all_models_ci_b400.png",
) -> Path:
    x = list(range(len(X_LABELS)))
    fig, ax = plt.subplots(figsize=(6.3, 3.9))

    human_center, human_lo, human_hi = bootstrap_bin_ci(human)
    ax.fill_between(x, human_lo, human_hi, color="#444444", alpha=0.12, zorder=1)
    ax.plot(x, human_center, color="#444444", linewidth=2.3, marker="D", markersize=5.2, zorder=3)

    ymax = float(human_hi.max())
    for group_name in ["VLM", "VLM decoder", "LLM", "LLM (think)"]:
        blind_group = blind[blind["group"] == group_name].copy()
        inst_group = inst[inst["group"] == group_name].copy()
        blind_center, blind_lo, blind_hi = bootstrap_bin_ci_clustered(blind_group, ["model", "qid"])
        inst_center, inst_lo, inst_hi = bootstrap_bin_ci_clustered(inst_group, ["model", "qid"])
        ymax = max(ymax, float(blind_hi.max()), float(inst_hi.max()))
        ax.fill_between(x, blind_lo, blind_hi, color=GROUP_COLORS[group_name], alpha=0.08, zorder=1)
        ax.fill_between(x, inst_lo, inst_hi, color=GROUP_COLORS[group_name], alpha=0.16, zorder=1)
        draw_condition_line(ax, x, blind_center, color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=False, linewidth=1.7, linestyle=":")
        draw_condition_line(ax, x, inst_center, color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=True, linewidth=2.0, linestyle="-")

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=9)
    ax.set_xlabel("Count answer", fontsize=10)
    ax.set_ylabel("Proportion", fontsize=10)
    ax.set_ylim(0, max(0.9, ymax * 1.08))
    ax.grid(True, alpha=0.22)
    add_compact_legends(ax, include_human=True)
    fig.tight_layout()

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def build_integer_plot_ci_errorbars(
    blind: pd.DataFrame,
    inst: pd.DataFrame,
    human: pd.DataFrame,
    out_name: str = "vC_qwen_integer_err_b400.png",
    model_list: list[str] | None = None,
) -> Path:
    if model_list is None:
        model_list = QWEN_MODELS
    blind_qwen = blind[blind["model"].isin(model_list)].copy()
    inst_qwen = inst[inst["model"].isin(model_list)].copy()
    x = list(range(len(X_LABELS)))
    fig, ax = plt.subplots(figsize=(4.8, 3.0))

    human_center, human_lo, human_hi = bootstrap_bin_ci(human)
    ax.fill_between(x, human_lo, human_hi, color="#444444", alpha=0.12, zorder=1, label="_nolegend_")
    draw_condition_errorbars(
        ax, x, human_center, human_lo, human_hi,
        color="#444444", marker="D", filled=True,
        linewidth=2.0, linestyle="-", markersize=4.6, zorder=4,
    )

    ymax = float(human_center.max())
    for model in model_list:
        group_name = MODEL_TO_GROUP[model]
        inst_center, inst_lo, inst_hi = bootstrap_bin_ci(inst_qwen[inst_qwen["model"] == model])
        ymax = max(ymax, float(inst_center.max()))
        draw_condition_errorbars(
            ax, x, inst_center, inst_lo, inst_hi,
            color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=True,
            linewidth=1.7, linestyle="-", markersize=3.9, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=9)
    ax.set_xlabel("Count answer", fontsize=10)
    ax.set_ylabel("Proportion", fontsize=10)
    ax.grid(True, alpha=0.22)
    add_compact_legends(ax, include_human=True)
    fig.tight_layout()
    ax.set_ylim(bottom=0, top=ymax * 1.12)
    fig.canvas.draw()

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def build_integer_plot_grouped_ci_errorbars(
    blind: pd.DataFrame,
    inst: pd.DataFrame,
    human: pd.DataFrame,
    out_name: str = "vC_grouped_all_models_err_b400.png",
) -> Path:
    x = list(range(len(X_LABELS)))
    fig, ax = plt.subplots(figsize=(6.3, 3.9))

    human_center, human_lo, human_hi = bootstrap_bin_ci(human)
    ax.fill_between(x, human_lo, human_hi, color="#444444", alpha=0.12, zorder=1, label="_nolegend_")
    draw_condition_errorbars(
        ax, x, human_center, human_lo, human_hi,
        color="#444444", marker="D", filled=True,
        linewidth=2.0, linestyle="-", markersize=4.6, zorder=4,
    )

    ymax = float(human_hi.max())
    for group_name in ["VLM", "VLM decoder", "LLM", "LLM (think)"]:
        blind_group = blind[blind["group"] == group_name].copy()
        inst_group = inst[inst["group"] == group_name].copy()
        blind_center, blind_lo, blind_hi = bootstrap_bin_ci_clustered(blind_group, ["model", "qid"])
        inst_center, inst_lo, inst_hi = bootstrap_bin_ci_clustered(inst_group, ["model", "qid"])
        ymax = max(ymax, float(blind_hi.max()), float(inst_hi.max()))
        draw_condition_errorbars(
            ax, x, inst_center, inst_lo, inst_hi,
            color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=True,
            linewidth=1.7, linestyle="-", markersize=3.9, zorder=3,
        )
        draw_condition_errorbars(
            ax, x, blind_center, blind_lo, blind_hi,
            color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name], filled=False,
            linewidth=1.4, linestyle=":", markersize=4.6, zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=9)
    ax.set_xlabel("Count answer", fontsize=10)
    ax.set_ylabel("Proportion", fontsize=10)
    ax.set_ylim(0, max(0.9, ymax * 1.08))
    ax.grid(True, alpha=0.22)
    add_compact_legends(ax, include_human=True)
    fig.tight_layout()

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def small_positive_share(df: pd.DataFrame) -> tuple[float, float]:
    vals = [extract_number(x) for x in df["clean"].tolist()]
    vals = [int(round(float(v))) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return (0.0, 0.0)
    total = len(vals)
    zero = sum(v == 0 for v in vals) / total
    small_pos = sum(1 <= v <= 3 for v in vals) / total
    return zero, small_pos


def build_arrow_plot(
    blind: pd.DataFrame,
    inst: pd.DataFrame,
    human: pd.DataFrame,
    out_name: str = "vC_qwen_arrow.png",
) -> Path:
    blind_qwen = blind[blind["model"].isin(QWEN_MODELS)].copy()
    inst_qwen = inst[inst["model"].isin(QWEN_MODELS)].copy()
    fig, ax = plt.subplots(figsize=(5.4, 4.2))

    points_x: list[float] = []
    points_y: list[float] = []

    hx, hy = small_positive_share(human)
    points_x.append(hx)
    points_y.append(hy)
    ax.scatter([hx], [hy], s=72, marker="D", color="#444444", label="Human reference", zorder=4)
    ax.text(hx + 0.006, hy + 0.006, "Human", color="#444444", fontsize=11)

    for model in QWEN_MODELS:
        b = blind_qwen[blind_qwen["model"] == model]
        i = inst_qwen[inst_qwen["model"] == model]
        if b.empty or i.empty:
            continue
        bx, by = small_positive_share(b)
        ix, iy = small_positive_share(i)
        points_x.extend([bx, ix])
        points_y.extend([by, iy])
        group_name = MODEL_TO_GROUP[model]
        color = GROUP_COLORS[group_name]
        marker = GROUP_MARKERS[group_name]

        ax.annotate(
            "",
            xy=(ix, iy),
            xytext=(bx, by),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=0.85),
        )
        ax.scatter(
            [bx],
            [by],
            s=90,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            marker=marker,
            zorder=5,
        )
        ax.scatter(
            [ix],
            [iy],
            s=90,
            color=color,
            marker=marker,
            zorder=6,
        )
        ax.text(ix + 0.006, iy + 0.003, model.replace("Qwen3-", "Q3-"), color=color, fontsize=10)

    xmin = max(0.0, min(points_x) - 0.03)
    xmax = min(1.0, max(points_x) + 0.08)
    ymin = max(0.0, min(points_y) - 0.02)
    ymax = min(1.0, max(points_y) + 0.08)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Share of "0" answers')
    ax.set_ylabel("Share of small positive counts (1-3)")
    ax.grid(True, alpha=0.22)

    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


FAMILY_PANELS = [
    ("InternVL",     ["InternVL-8B",   "InternVL-8B (LM)",   "Qwen3-8B",  "Qwen3-8B (think)"]),
    ("LLaVA-1.5",   ["LLaVA-1.5-7B",  "LLaVA-1.5 (LM)",    "Vicuna-7B"]),
    ("LLaVA-Mistral",["LLaVA-Mistral", "LLaVA-Mistral (LM)", "Mistral-7B"]),
    ("LLaVA-Vicuna", ["LLaVA-Vicuna",  "LLaVA-Vicuna (LM)",  "Vicuna-7B"]),
    ("Qwen3-VL",     ["Qwen3-VL-8B",   "Qwen3-VL-8B (LM)",   "Qwen3-8B",  "Qwen3-8B (think)"]),
]


def build_family_panels(
    blind: pd.DataFrame,
    inst: pd.DataFrame,
    human: pd.DataFrame,
    out_name: str = "vABC_family_panels_err_b400.png",
) -> Path:
    x = list(range(len(X_LABELS)))
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 6.0), sharex=True, sharey=False,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.38})
    axes_flat = axes.flatten()

    human_center, human_lo, human_hi = bootstrap_bin_ci(human)

    for idx, (family_name, model_list) in enumerate(FAMILY_PANELS):
        ax = axes_flat[idx]
        ax.fill_between(x, human_lo, human_hi, color="#444444", alpha=0.12, zorder=1, label="_nolegend_")
        draw_condition_errorbars(
            ax, x, human_center, human_lo, human_hi,
            color="#444444", marker="D", filled=True,
            linewidth=1.8, linestyle="-", markersize=4.0, zorder=4,
        )
        ymax = float(human_hi.max())
        for model in model_list:
            if model not in blind["model"].values:
                continue
            group_name = MODEL_TO_GROUP.get(model)
            if group_name is None:
                continue
            b = blind[blind["model"] == model]
            i = inst[inst["model"] == model]
            bc, blo, bhi = bootstrap_bin_ci(b)
            ic, ilo, ihi = bootstrap_bin_ci(i)
            ymax = max(ymax, float(bhi.max()), float(ihi.max()))
            draw_condition_errorbars(ax, x, ic, ilo, ihi,
                color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name],
                filled=True, linewidth=1.5, linestyle="-", markersize=3.5, zorder=3)
            draw_condition_errorbars(ax, x, bc, blo, bhi,
                color=GROUP_COLORS[group_name], marker=GROUP_MARKERS[group_name],
                filled=False, linewidth=1.2, linestyle=":", markersize=4.2, zorder=5)
        ax.set_ylim(0, max(0.9, ymax * 1.10))
        ax.set_xticks(x[::2])
        ax.set_xticklabels(X_LABELS[::2], fontsize=7.5)
        ax.set_title(family_name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.20)
        if idx % ncols == 0:
            ax.set_ylabel("Proportion", fontsize=9)
        if idx // ncols == nrows - 1 or idx >= len(FAMILY_PANELS) - ncols:
            ax.set_xlabel("Count answer", fontsize=9)

    # Hide unused panel
    for idx in range(len(FAMILY_PANELS), nrows * ncols):
        axes_flat[idx].axis("off")

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color="#444444", linewidth=1.8, marker="D", markersize=4.5, label="Human"),
    ]
    for g in ["VLM", "VLM decoder", "LLM", "LLM (think)"]:
        legend_handles.append(Line2D([0], [0], color=GROUP_COLORS[g], linewidth=1.8,
                                     marker="o", markersize=4.5, label=g))
    legend_handles += [
        Line2D([0], [0], color="#666", linewidth=1.6, marker="o", markersize=4.5,
               markerfacecolor="#666", label="w/ instruction"),
        Line2D([0], [0], color="#666", linewidth=1.6, marker="o", markersize=4.5,
               markerfacecolor="white", markeredgewidth=1.3, label="w/o instruction"),
    ]
    axes_flat[-1].axis("off")
    axes_flat[-1].legend(handles=legend_handles, loc="center", fontsize=8.5,
                         frameon=True, handlelength=1.6, labelspacing=0.45)

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def copy_to_latex(path: Path) -> None:
    LATEX_OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, LATEX_OUT_DIR / path.name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qid2atype = load_qid_to_answer_type()
    blind = load_model_condition("blind", qid2atype)
    inst = load_model_condition("inst_blind", qid2atype)
    human = load_human(qid2atype)
    blind_vabc = load_model_condition("blind", qid2atype, variants=("A", "B", "C"))
    inst_vabc = load_model_condition("inst_blind", qid2atype, variants=("A", "B", "C"))
    human_vabc = load_human(qid2atype, variants=("A", "B", "C"))

    outputs = [
        build_integer_plot(blind, inst, human, out_name="vC_qwen_integer.png"),
        build_integer_plot(blind_vabc, inst_vabc, human_vabc, out_name="vABC_qwen_integer.png"),
        build_integer_plot_by_model(blind, inst, human),
        build_integer_plot_ci(blind, inst, human),
        build_integer_plot_ci(blind_vabc, inst_vabc, human_vabc, out_name="vABC_qwen_integer_ci_b400.png"),
        build_integer_plot_ci_errorbars(blind, inst, human, out_name="vC_qwen_integer_err_b400.png"),
        build_integer_plot_ci_errorbars(blind_vabc, inst_vabc, human_vabc, out_name="vABC_qwen_integer_err_b400.png"),
        build_integer_plot_ci_errorbars(blind_vabc, inst_vabc, human_vabc, out_name="vABC_internvl_integer_err_b400.png", model_list=INTERNVL_MODELS),
        build_integer_plot_ci_errorbars(blind_vabc, inst_vabc, human_vabc, out_name="vABC_llava_integer_err_b400.png", model_list=LLAVA_MODELS),
        build_integer_plot_ci_errorbars(blind_vabc, inst_vabc, human_vabc, out_name="vABC_qwen_vl_integer_err_b400.png", model_list=QWEN_VL_MODELS),
        build_family_panels(blind_vabc, inst_vabc, human_vabc, out_name="vABC_family_panels_err_b400.png"),
        build_integer_plot_all_models(blind, inst, human, out_name="vC_all_models.png"),
        build_integer_plot_all_models(blind_vabc, inst_vabc, human_vabc, out_name="vABC_all_models.png"),
        build_integer_plot_grouped_ci(blind, inst, human, out_name="vC_grouped_all_models_ci_b400.png"),
        build_integer_plot_grouped_ci(blind_vabc, inst_vabc, human_vabc, out_name="vABC_grouped_all_models_ci_b400.png"),
        build_integer_plot_grouped_ci_errorbars(blind, inst, human, out_name="vC_grouped_all_models_err_b400.png"),
        build_integer_plot_grouped_ci_errorbars(blind_vabc, inst_vabc, human_vabc, out_name="vABC_grouped_all_models_err_b400.png"),
        build_arrow_plot(blind, inst, human, out_name="vC_qwen_arrow.png"),
    ]
    for path in outputs:
        copy_to_latex(path)
        print(path)


if __name__ == "__main__":
    main()
