"""
Export grouped 3-step variant ladder plots for accuracy and HM SBERT.

Default output:
  figures/variant_ladder_groups_inst_blind/
    accuracy_groups_q88_h40.png
    sbert_groups_q88_h40.png

With --include_yesno:
  figures/variant_ladder_groups_inst_blind/
    accuracy_groups_q113_h40_yesno.png
    sbert_groups_q113_h40_yesno.png

Each figure contains 3 subplots with shared y-limits:
  VLM | VLM backbone decoder | standalone LLM

Included:
  - matched 7/8B subset only

Excluded:
  - standalone LLM (think)
  - Phi-3.5-mini
  - Mistral-7B
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from config import MODELS_7B, COLLAPSED_VARIANT_QIDS
from helpers import clear_output_plots, get_exports_dir, load_cleaned_pair_cache
from utils.abstention import classify, is_abstained
from utils.load_session import clean_answer
from utils.constants import MODEL_FAMILY, MODEL_FAMILY_COLORS, MODEL_SIZE_B


parser = argparse.ArgumentParser()
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete existing plot files in the output folder before exporting.",
)
parser.add_argument(
    "--include_yesno",
    action="store_true",
    help="Use the q113 yes/no-inclusive pair cache and response exports.",
)
parser.add_argument(
    "--filter_abstentions",
    action="store_true",
    help="Restrict model-level statistics to substantive (non-abstaining) answers only.",
)
args = parser.parse_args()

OUT_DIR = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)

LATEX_FIG = ROOT / "latex/AAAI2026/LaTeX/figures"
LATEX_LADDER = LATEX_FIG / "variant_ladder_groups_inst_blind"
LATEX_LADDER.mkdir(parents=True, exist_ok=True)

EXPORTS = get_exports_dir(ROOT)
pair_df = load_cleaned_pair_cache(ROOT, include_yesno=args.include_yesno, verbose=True)
resp_df = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")
human_df = pd.read_csv(EXPORTS / "responses_human.csv")

VARIANTS = ["C", "B", "A"]
VAR_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
GROUPS = ["VLM", "VLM backbone decoder", "standalone LLM"]
GROUP_TITLE = {
    "VLM": "VLM",
    "VLM backbone decoder": "Backbone Decoder",
    "standalone LLM": "Standalone LLM",
}
EXCLUDED_MODELS = {"Phi-3.5-mini", "Mistral-7B"}
INCLUDED_MODELS = {
    m for m in MODELS_7B
    if "think" not in m
} - EXCLUDED_MODELS

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
    }
)

FIGSIZE = (11.5, 4.1)
CI_Z = 1.96
REF_COLOR = "#37474F"
REF_BAND_COLOR = "#B0B7BE"
MODEL_COLOR_OVERRIDE = {
    "Qwen2.5-7B-Instruct": "#2E8B57",
}


def hh_participant_count(df: pd.DataFrame) -> int:
    hh = df[df["pair_type"] == "HH"]
    return len(set(hh["subject_1"]).union(set(hh["subject_2"])))


def model_sort_key(model: str):
    return (
        GROUPS.index(model_group(model)),
        MODEL_FAMILY.get(model, model).lower(),
        float(MODEL_SIZE_B.get(model, 999.0)),
        model.lower(),
    )


def model_group(model: str) -> str:
    if model.endswith("(LM)"):
        return "VLM backbone decoder"
    if model in resp_df["model"].unique():
        group = resp_df.loc[resp_df["model"] == model, "model_group"].iloc[0]
        return str(group)
    if model in pair_df.get("subject_2", pd.Series(dtype=str)).unique():
        group = pair_df.loc[pair_df["subject_2"] == model, "subject_group_2"].iloc[0]
        return str(group)
    return "standalone LLM"


def display_label(model: str) -> str:
    label = model
    label = label.replace("LLaVA-Mistral", "LLaVA-M")
    label = label.replace("LLaVA-Vicuna", "LLaVA-V")
    label = label.replace("LLaVA-1.5-7B", "LLaVA-1.5")
    label = label.replace("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Inst")
    label = label.replace(" (LM)", " (bb)")
    return label


def mean_and_ci(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return np.nan, np.nan
    mean = clean.mean()
    if len(clean) < 2:
        return mean, 0.0
    ci = CI_Z * clean.std(ddof=1) / np.sqrt(len(clean))
    return mean, ci


resp_df = resp_df[
    resp_df["model_group"].isin(GROUPS) & resp_df["model"].isin(INCLUDED_MODELS)
].copy()
resp_df["response"] = resp_df["response"].fillna("").astype(str).apply(clean_answer)
resp_df["is_substantive"] = resp_df["response"].apply(lambda x: not is_abstained(classify(x, None)))
pair_df = pair_df[
    pair_df["pair_type"].isin(["HM", "HH"])
            & (
        (pair_df["pair_type"] == "HH")
        | (
            pair_df["subject_group_2"].isin(GROUPS)
            & pair_df["subject_2"].isin(INCLUDED_MODELS)
        )
    )
].copy()

if not args.include_yesno:
    free_text_qids = sorted(pair_df[pair_df["pair_type"] == "HH"]["question_id"].unique())
    resp_df = resp_df[resp_df["question_id"].isin(free_text_qids)].copy()
    human_df = human_df[human_df["question_id"].isin(free_text_qids)].copy()

# Exclude collapsed-variant questions from degradation analysis
pair_df = pair_df[~pair_df["question_id"].isin(COLLAPSED_VARIANT_QIDS)].copy()
resp_df = resp_df[~resp_df["question_id"].isin(COLLAPSED_VARIANT_QIDS)].copy()
human_df = human_df[~human_df["question_id"].isin(COLLAPSED_VARIANT_QIDS)].copy()

n_questions = pair_df[pair_df["pair_type"] == "HH"]["question_id"].nunique()
n_humans = hh_participant_count(pair_df)
suffix = f"_q{n_questions}_h{n_humans}" + ("_yesno" if args.include_yesno else "")

MODEL_ORDER = sorted(INCLUDED_MODELS, key=model_sort_key)

SUBSTANTIVE_QIDS: dict[tuple[str, str], set[int]] = {}
for (model, variant), sub in resp_df.groupby(["model", "variant"]):
    SUBSTANTIVE_QIDS[(str(model), str(variant))] = set(
        sub.loc[sub["is_substantive"], "question_id"].astype(int).tolist()
    )


def filtered_qvals_hm(df: pd.DataFrame, model: str, variant: str, substantive_only: bool) -> pd.Series:
    sub = df[(df["subject_2"] == model) & (df["variant"] == variant)]
    if substantive_only:
        keep = SUBSTANTIVE_QIDS.get((model, variant), set())
        sub = sub[sub["question_id"].isin(keep)]
    return sub.groupby("question_id")["sbert_score"].mean() * 100


def build_accuracy_stats() -> tuple[
    dict[str, dict[str, dict[str, list[float]]]],
    dict[str, list[float]],
]:
    stats: dict[str, dict[str, dict[str, list[float]]]] = {}
    for group in GROUPS:
        models = [m for m in MODEL_ORDER if model_group(m) == group and m in set(resp_df["model"])]
        stats[group] = {}
        for model in models:
            means = []
            cis = []
            for var in VARIANTS:
                sub = resp_df[(resp_df["model"] == model) & (resp_df["variant"] == var)]
                if args.filter_abstentions:
                    sub = sub[sub["is_substantive"]]
                q_vals = sub.groupby("question_id")["accuracy"].mean() * 100
                mean, ci = mean_and_ci(q_vals)
                means.append(mean)
                cis.append(ci)
            stats[group][model] = {"mean": means, "ci": cis}
    human_acc_mean = []
    human_acc_ci = []
    for var in VARIANTS:
        sub = human_df[human_df["variant"] == var]
        q_vals = sub.groupby("question_id")["accuracy"].mean() * 100
        mean, ci = mean_and_ci(q_vals)
        human_acc_mean.append(mean)
        human_acc_ci.append(ci)
    return stats, {"mean": human_acc_mean, "ci": human_acc_ci}


def build_sbert_stats() -> tuple[
    dict[str, dict[str, dict[str, list[float]]]],
    dict[str, list[float]],
]:
    hm = pair_df[pair_df["pair_type"] == "HM"].copy()
    hh = pair_df[pair_df["pair_type"] == "HH"].copy()
    stats: dict[str, dict[str, dict[str, list[float]]]] = {}
    for group in GROUPS:
        models = [m for m in MODEL_ORDER if model_group(m) == group and m in set(hm["subject_2"])]
        stats[group] = {}
        for model in models:
            means = []
            cis = []
            for var in VARIANTS:
                q_vals = filtered_qvals_hm(hm, model, var, args.filter_abstentions)
                mean, ci = mean_and_ci(q_vals)
                means.append(mean)
                cis.append(ci)
            stats[group][model] = {"mean": means, "ci": cis}
    hh_mean = []
    hh_ci = []
    for var in VARIANTS:
        q_vals = hh[hh["variant"] == var].groupby("question_id")["sbert_score"].mean() * 100
        mean, ci = mean_and_ci(q_vals)
        hh_mean.append(mean)
        hh_ci.append(ci)
    return stats, {"mean": hh_mean, "ci": hh_ci}


def axis_limits(
    series_dict: dict[str, dict[str, dict[str, list[float]]]],
    ref_stats: dict[str, list[float]],
) -> tuple[float, float]:
    vals = []
    for group_data in series_dict.values():
        for stat in group_data.values():
            vals.extend([v for v in stat["mean"] if pd.notna(v)])
            vals.extend(
                [m + c for m, c in zip(stat["mean"], stat["ci"]) if pd.notna(m) and pd.notna(c)]
            )
            vals.extend(
                [m - c for m, c in zip(stat["mean"], stat["ci"]) if pd.notna(m) and pd.notna(c)]
            )
    vals.extend([v for v in ref_stats["mean"] if pd.notna(v)])
    vals.extend(
        [m + c for m, c in zip(ref_stats["mean"], ref_stats["ci"]) if pd.notna(m) and pd.notna(c)]
    )
    vals.extend(
        [m - c for m, c in zip(ref_stats["mean"], ref_stats["ci"]) if pd.notna(m) and pd.notna(c)]
    )
    lo = min(vals)
    hi = max(vals)
    pad = max(2.0, 0.08 * (hi - lo if hi > lo else 10.0))
    return lo - pad, hi + pad


def save_combined_plot(
    metric_slug: str,
    stats: dict[str, dict[str, dict[str, list[float]]]],
    ref_stats: dict[str, list[float]],
    ylabel: str,
):
    ymin, ymax = axis_limits(stats, ref_stats)
    x = np.arange(len(VARIANTS))
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
    legend_handles = []
    legend_labels = []

    ref_mean = np.array(ref_stats["mean"], dtype=float)
    ref_ci = np.array(ref_stats["ci"], dtype=float)
    ref_lo = np.clip(ref_mean - ref_ci, 0.0, 100.0)
    ref_hi = np.clip(ref_mean + ref_ci, 0.0, 100.0)

    for ax, group in zip(axes, GROUPS):
        for model, stat in stats[group].items():
            family = MODEL_FAMILY.get(model, model)
            color = MODEL_COLOR_OVERRIDE.get(model, MODEL_FAMILY_COLORS.get(family, "#666666"))
            means = np.array(stat["mean"], dtype=float)
            cis = np.array(stat["ci"], dtype=float)
            lower = np.minimum(cis, means)
            upper = np.minimum(cis, 100.0 - means)
            container = ax.errorbar(
                x,
                means,
                yerr=np.vstack([lower, upper]),
                color=color,
                lw=1.25 if args.filter_abstentions else 1.8,
                ls=":" if args.filter_abstentions else "-",
                alpha=0.9 if args.filter_abstentions else 0.95,
                marker="o",
                markersize=5.0 if args.filter_abstentions else 5.6,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.0,
                elinewidth=0.9 if args.filter_abstentions else 1.0,
                capsize=2.8,
                capthick=1.0,
            )
            if model not in legend_labels:
                legend_handles.append(container.lines[0])
                legend_labels.append(model)

        ax.fill_between(
            x,
            ref_lo,
            ref_hi,
            color=REF_BAND_COLOR,
            alpha=0.35,
            zorder=0,
            linewidth=0,
        )
        ax.plot(
            x,
            ref_mean,
            color=REF_COLOR,
            lw=1.8,
            ls="--",
            marker="D",
            markersize=4.8,
            zorder=1,
        )
        ax.set_title(GROUP_TITLE[group], fontsize=11, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS])
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(length=3.5, width=0.8, color="#555555")

    axes[0].set_ylabel(ylabel)

    # Legend: VLMs first, then standalone LLMs; exclude backbone decoders
    vlm_order = [m for m in MODEL_ORDER if m in legend_labels and "(LM)" not in m and model_group(m) == "VLM"]
    llm_order = [m for m in MODEL_ORDER if m in legend_labels and "(LM)" not in m and model_group(m) == "standalone LLM"]
    order = vlm_order + llm_order
    label_map = {m: display_label(m) for m in legend_labels}
    handle_map = {m: h for m, h in zip(legend_labels, legend_handles)}
    ordered_handles = [handle_map[m] for m in order]
    ordered_labels = [label_map[m] for m in order]

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(order),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.22, wspace=0.10)
    extra = "_abstfiltered" if args.filter_abstentions else "_raw"
    local_name = f"{metric_slug}{suffix}{extra}.png"
    latex_name = f"{metric_slug}_groups{suffix}{extra}.png"
    path = OUT_DIR / local_name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    shutil.copy(path, LATEX_LADDER / latex_name)
    plt.close(fig)
    print(f"[variant_ladders] {path.name}")


def save_overlay_plot(
    metric_slug: str,
    base_stats: dict[str, dict[str, dict[str, list[float]]]],
    filt_stats: dict[str, dict[str, dict[str, list[float]]]],
    ref_stats: dict[str, list[float]],
    ylabel: str,
    *,
    filename_suffix: str = "_abstfiltered",
):
    ymin, ymax = axis_limits(base_stats, ref_stats)
    x = np.arange(len(VARIANTS))
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
    legend_handles = []
    legend_labels = []

    ref_mean = np.array(ref_stats["mean"], dtype=float)
    ref_ci = np.array(ref_stats["ci"], dtype=float)
    ref_lo = np.clip(ref_mean - ref_ci, 0.0, 100.0)
    ref_hi = np.clip(ref_mean + ref_ci, 0.0, 100.0)

    for ax, group in zip(axes, GROUPS):
        for model, stat in base_stats[group].items():
            family = MODEL_FAMILY.get(model, model)
            color = MODEL_COLOR_OVERRIDE.get(model, MODEL_FAMILY_COLORS.get(family, "#666666"))
            means = np.array(stat["mean"], dtype=float)
            cis = np.array(stat["ci"], dtype=float)
            lower = np.minimum(cis, means)
            upper = np.minimum(cis, 100.0 - means)
            container = ax.errorbar(
                x, means, yerr=np.vstack([lower, upper]),
                color=color, lw=1.8, ls="-", alpha=0.95,
                marker="o", markersize=5.6, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=1.0,
                elinewidth=1.0, capsize=2.8, capthick=1.0,
            )
            if model not in legend_labels:
                legend_handles.append(container.lines[0])
                legend_labels.append(model)

        for model, stat in filt_stats[group].items():
            family = MODEL_FAMILY.get(model, model)
            color = MODEL_COLOR_OVERRIDE.get(model, MODEL_FAMILY_COLORS.get(family, "#666666"))
            means = np.array(stat["mean"], dtype=float)
            cis = np.array(stat["ci"], dtype=float)
            lower = np.minimum(cis, means)
            upper = np.minimum(cis, 100.0 - means)
            ax.errorbar(
                x, means, yerr=np.vstack([lower, upper]),
                color=color, lw=1.15, ls=":", alpha=0.88,
                marker="o", markersize=4.8, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=0.9,
                elinewidth=0.8, capsize=2.4, capthick=0.8,
            )

        ax.fill_between(x, ref_lo, ref_hi, color=REF_BAND_COLOR, alpha=0.35, zorder=0, linewidth=0)
        ax.plot(x, ref_mean, color=REF_COLOR, lw=1.8, ls="--", marker="D", markersize=4.8, zorder=1)
        ax.set_title(GROUP_TITLE[group], fontsize=11, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS])
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(length=3.5, width=0.8, color="#555555")

    axes[0].set_ylabel(ylabel)
    vlm_order = [m for m in MODEL_ORDER if m in legend_labels and "(LM)" not in m and model_group(m) == "VLM"]
    llm_order = [m for m in MODEL_ORDER if m in legend_labels and "(LM)" not in m and model_group(m) == "standalone LLM"]
    order = vlm_order + llm_order
    label_map = {m: display_label(m) for m in legend_labels}
    handle_map = {m: h for m, h in zip(legend_labels, legend_handles)}
    ordered_handles = [handle_map[m] for m in order]
    ordered_labels = [label_map[m] for m in order]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(order),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.22, wspace=0.10)
    local_name = f"{metric_slug}{suffix}{filename_suffix}.png"
    latex_name = f"{metric_slug}_groups{suffix}{filename_suffix}.png"
    path = OUT_DIR / local_name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    shutil.copy(path, LATEX_LADDER / latex_name)
    plt.close(fig)
    print(f"[variant_ladders] {path.name}")


if args.filter_abstentions:
    # overlay original solid lines with substantive-only dotted lines
    args.filter_abstentions = False
    acc_base, human_acc = build_accuracy_stats()
    sbert_base, hh_sbert = build_sbert_stats()
    args.filter_abstentions = True
    acc_filt, _ = build_accuracy_stats()
    sbert_filt, _ = build_sbert_stats()
    save_combined_plot("accuracy", acc_filt, human_acc, "Mean accuracy (%)")
    save_combined_plot("sbert", sbert_filt, hh_sbert, "Mean HM SBERT (%)")
    save_overlay_plot(
        "accuracy", acc_base, acc_filt, human_acc, "Mean accuracy (%)",
        filename_suffix="_abstfiltered",
    )
    save_overlay_plot(
        "sbert", sbert_base, sbert_filt, hh_sbert, "Mean HM SBERT (%)",
        filename_suffix="_abstfiltered",
    )
else:
    acc_stats, human_acc = build_accuracy_stats()
    save_combined_plot("accuracy", acc_stats, human_acc, "Mean accuracy (%)")

    sbert_stats, hh_sbert = build_sbert_stats()
    save_combined_plot("sbert", sbert_stats, hh_sbert, "Mean HM SBERT (%)")
