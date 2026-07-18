from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from config import MODELS_ALL, MODEL_GROUP, MIN_ANSWERS_DEFAULT
from helpers import load_cleaned_pair_cache, load_human_subset, filter_abstained_pairs, read_export
from utils.constants import GROUP_COLORS, GROUP_HOLLOW, MODEL_SIZE_B
from analysis.utils.abstention import classify, is_abstained


OUT_DIR = ROOT / "figures" / "scale_alignment_variant_by_family"
LATEX_OUT = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "scale_alignment_variant_by_family"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["C", "B", "A"]
EXCLUDED_MODELS = {"Phi-3.5-mini"}
MODELS_SIZED = [m for m in MODELS_ALL if m in MODEL_SIZE_B and m not in EXCLUDED_MODELS]
HH_COLOR = "#222222"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
})

PANEL_SPECS = [
    ("Qwen3", ["Qwen3-VL", "Qwen3-VL (LM)", "Qwen3", "Qwen3 (think)"]),
    ("InternVL", ["InternVL", "InternVL (LM)"]),
    ("LLaVA-Vicuna / Vicuna", ["LLaVA-Vicuna", "LLaVA-Vicuna (LM)", "Vicuna"]),
]


def family_name(model: str) -> str:
    if "InternVL" in model and "(LM)" in model:
        return "InternVL (LM)"
    if "InternVL" in model:
        return "InternVL"
    if "Qwen3-VL" in model and "(LM)" in model:
        return "Qwen3-VL (LM)"
    if "Qwen3-VL" in model:
        return "Qwen3-VL"
    if "Qwen3" in model and "think" in model.lower():
        return "Qwen3 (think)"
    if model.startswith("Qwen3-"):
        return "Qwen3"
    if "Qwen2.5" in model:
        return "Qwen2.5"
    if "LLaVA-1.5" in model and "(LM)" in model:
        return "LLaVA-1.5 (LM)"
    if "LLaVA-1.5" in model:
        return "LLaVA-1.5"
    if "LLaVA-Mistral" in model and "(LM)" in model:
        return "LLaVA-Mistral (LM)"
    if "LLaVA-Mistral" in model:
        return "LLaVA-Mistral"
    if "LLaVA-Vicuna" in model and "(LM)" in model:
        return "LLaVA-Vicuna (LM)"
    if "LLaVA-Vicuna" in model:
        return "LLaVA-Vicuna"
    if model.startswith("Vicuna"):
        return "Vicuna"
    if model.startswith("Mistral"):
        return "Mistral"
    return model


def slugify(name: str) -> str:
    return (
        name.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def ci95(values: np.ndarray) -> float:
    n = len(values)
    return 1.96 * values.std() / np.sqrt(n) if n >= 2 else 0.0


def human_sbert_baseline(pair_cache: pd.DataFrame) -> tuple[float, float]:
    hh = pair_cache[pair_cache["pair_type"] == "HH"].copy()
    q_vals = hh.groupby("question_id")["sbert_score"].mean().dropna().values
    return float(q_vals.mean()), float(ci95(q_vals))


def model_sbert_stats(pair_cache: pd.DataFrame) -> pd.DataFrame:
    hm = pair_cache[pair_cache["pair_type"] == "HM"].copy()
    q_means = hm.groupby(["subject_2", "question_id"])["sbert_score"].mean().reset_index()
    rows = []
    for model in MODELS_SIZED:
        vals = q_means[q_means["subject_2"] == model]["sbert_score"].dropna().values
        if len(vals) == 0:
            continue
        rows.append(
            {
                "model": model,
                "family": family_name(model),
                "mean": float(vals.mean()),
                "ci": float(ci95(vals)),
                "size": float(MODEL_SIZE_B[model]),
                "group": MODEL_GROUP.get(model, "standalone LLM"),
            }
    )
    return pd.DataFrame(rows)


def human_accuracy_baseline(human_df: pd.DataFrame, common_qids: list[int]) -> tuple[float, float]:
    q_vals = (
        human_df[
            human_df["question_id"].isin(common_qids) &
            human_df["variant"].isin(VARIANTS)
        ]
        .groupby("question_id")["accuracy"]
        .mean()
        .to_numpy()
    )
    return float(q_vals.mean()), float(ci95(q_vals))


def model_accuracy_stats(raw_acc: pd.DataFrame) -> pd.DataFrame:
    q_means = raw_acc.groupby(["model", "question_id"])["accuracy"].mean().reset_index()
    rows = []
    for model in MODELS_SIZED:
        vals = q_means[q_means["model"] == model]["accuracy"].dropna().values
        if len(vals) == 0:
            continue
        rows.append(
            {
                "model": model,
                "family": family_name(model),
                "mean": float(vals.mean()),
                "ci": float(ci95(vals)),
                "size": float(MODEL_SIZE_B[model]),
                "group": MODEL_GROUP.get(model, "standalone LLM"),
            }
        )
    return pd.DataFrame(rows)


def configure_xaxis(ax, sizes: list[float]) -> None:
    xs = sorted(set(sizes))
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) if x == int(x) else str(x) for x in xs])
    margin = 0.12
    ax.set_xlim(xs[0] * (10 ** -margin), xs[-1] * (10 ** margin))


def save(fig, name: str) -> None:
    out = OUT_DIR / name
    fig.savefig(out, dpi=240, bbox_inches="tight")
    shutil.copy2(out, LATEX_OUT / name)
    print(out)
    plt.close(fig)


def panel_marker(family: str) -> str:
    if "InternVL" in family:
        return "o"
    if "Qwen3" in family:
        return "X"
    if "1.5" in family:
        return "D"
    if "Mistral" in family:
        return "^"
    if "Vicuna" in family:
        return "v"
    return "s"


def plot_panel(
    ax,
    title: str,
    families: list[str],
    df: pd.DataFrame,
    baseline_mean: float,
    baseline_ci: float,
    ylabel: str,
    show_ylabel: bool,
) -> None:
    sub = df[df["family"].isin(families)].sort_values(["family", "size"]).copy()
    if sub.empty:
        ax.axis("off")
        return

    ax.axhline(baseline_mean, color=HH_COLOR, lw=1.5, ls="--", zorder=1)
    if baseline_ci > 0:
        ax.axhspan(baseline_mean - baseline_ci, baseline_mean + baseline_ci, color=HH_COLOR, alpha=0.08, zorder=0)

    ylo_vals = [baseline_mean - baseline_ci - 0.02]
    yhi_vals = [baseline_mean + baseline_ci + 0.02]
    sizes = []

    for family in families:
        fam_sub = sub[sub["family"] == family].sort_values("size").copy()
        if fam_sub.empty:
            continue

        grp = fam_sub["group"].iloc[0]
        color = GROUP_COLORS.get(grp, "#666666")
        hollow = GROUP_HOLLOW.get(grp, False)
        marker = panel_marker(family)

        x = fam_sub["size"].to_numpy(dtype=float)
        y = fam_sub["mean"].to_numpy(dtype=float)
        e = fam_sub["ci"].to_numpy(dtype=float)
        ms = [4.0 + 5.0 * (np.log10(max(v, 0.3)) / np.log10(32)) for v in x]

        sizes.extend(x.tolist())
        ylo_vals.append(float(y.min() - max(e.max(), 0.01) - 0.02))
        yhi_vals.append(float(y.max() + max(e.max(), 0.01) + 0.02))

        face = "none" if hollow else color
        edge = color if hollow else "white"
        edgew = 1.3 if hollow else 0.45

        for xi, yi, ei, msi in zip(x, y, e, ms):
            ax.errorbar(
                xi,
                yi,
                yerr=ei,
                fmt=marker,
                color=color,
                ms=msi,
                capsize=2.5,
                capthick=0.8,
                elinewidth=0.8,
                alpha=0.9,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=edgew,
                zorder=3,
            )

        if len(np.unique(x)) >= 2:
            coeffs = np.polyfit(np.log(x), y, 1)
            x_rng = np.logspace(np.log10(x.min()), np.log10(x.max()), 40)
            ax.plot(
                x_rng,
                np.polyval(coeffs, np.log(x_rng)),
                color=color,
                lw=1.1,
                ls=":" if hollow else "--",
                alpha=0.45,
            )

    configure_xaxis(ax, sizes)
    ax.set_title(title)
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel(ylabel if show_ylabel else "")
    ax.grid(True, color="#d7d7d7", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_ylim(min(ylo_vals), max(yhi_vals))


def plot_grid(
    df: pd.DataFrame,
    baseline_mean: float,
    baseline_ci: float,
    suffix: str,
    ylabel: str,
    prefix: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for idx, ((title, families), ax) in enumerate(zip(PANEL_SPECS, axes)):
        plot_panel(
            ax=ax,
            title=title,
            families=families,
            df=df,
            baseline_mean=baseline_mean,
            baseline_ci=baseline_ci,
            ylabel=ylabel,
            show_ylabel=(idx == 0),
        )

    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color=HH_COLOR, ls="--", lw=1.5, label="Human baseline"),
        mlines.Line2D([], [], color=GROUP_COLORS["VLM"], marker="o", lw=0, markersize=6,
                      markerfacecolor=GROUP_COLORS["VLM"], markeredgecolor="white", label="VLM"),
        mlines.Line2D([], [], color=GROUP_COLORS["VLM backbone decoder"], marker="o", lw=0, markersize=6,
                      markerfacecolor="none", markeredgecolor=GROUP_COLORS["VLM backbone decoder"], label="Backbone decoder"),
        mlines.Line2D([], [], color=GROUP_COLORS["standalone LLM"], marker="o", lw=0, markersize=6,
                      markerfacecolor=GROUP_COLORS["standalone LLM"], markeredgecolor="white", label="SA-LLM"),
        mlines.Line2D([], [], color=GROUP_COLORS["standalone LLM (think)"], marker="o", lw=0, markersize=6,
                      markerfacecolor="none", markeredgecolor=GROUP_COLORS["standalone LLM (think)"], label="Think"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.2)
    save(fig, f"{prefix}_subplots{suffix}.png")


def main() -> None:
    participants, common_qids, human_df, _ = load_human_subset(
        ROOT, min_answers=MIN_ANSWERS_DEFAULT, translate=False, verbose=False
    )
    pair_cache = load_cleaned_pair_cache(ROOT, condition="inst_blind", include_yesno=True, verbose=False)
    pair_cache = pair_cache[
        pair_cache["question_id"].isin(common_qids) &
        pair_cache["variant"].isin(VARIANTS)
    ].copy()

    raw_acc = read_export(ROOT, "responses_model_inst_blind.csv")
    raw_acc = raw_acc[
        raw_acc["question_id"].isin(common_qids) &
        raw_acc["variant"].isin(VARIANTS)
    ].copy()

    pair_cache_filt = filter_abstained_pairs(pair_cache)
    raw_acc_filt = raw_acc[
        ~raw_acc["response"].fillna("").astype(str).apply(lambda x: is_abstained(classify(x, None)))
    ].copy()

    sbert_mean, sbert_ci = human_sbert_baseline(pair_cache)
    sbert_mean_filt, sbert_ci_filt = human_sbert_baseline(pair_cache_filt)
    acc_mean, acc_ci = human_accuracy_baseline(human_df, common_qids)

    raw_sbert_df = model_sbert_stats(pair_cache)
    filt_sbert_df = model_sbert_stats(pair_cache_filt)
    raw_acc_df = model_accuracy_stats(raw_acc)
    filt_acc_df = model_accuracy_stats(raw_acc_filt)

    plot_grid(raw_sbert_df, sbert_mean, sbert_ci, "_vabc_raw", "Mean SBERT", "sbert")
    plot_grid(filt_sbert_df, sbert_mean_filt, sbert_ci_filt, "_vabc_abstfiltered", "Mean SBERT", "sbert")
    plot_grid(raw_acc_df, acc_mean, acc_ci, "_vabc_raw", "Mean accuracy", "accuracy")
    plot_grid(filt_acc_df, acc_mean, acc_ci, "_vabc_abstfiltered", "Mean accuracy", "accuracy")


if __name__ == "__main__":
    main()
