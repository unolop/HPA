from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from config import MODELS_ALL, MODEL_GROUP, MIN_ANSWERS_DEFAULT, extend_pair_cache_with_yesno
from helpers import clear_output_plots, get_exports_dir, load_human_subset, read_export
from utils.constants import GROUP_COLORS, GROUP_MARKER, GROUP_HOLLOW, MODEL_SIZE_B


OUT_DIR = ROOT / "figures/agreement/scale_main"
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=True)

EXPORTS = get_exports_dir(ROOT)
VARIANT = "C"
HH_COLOR = "#222222"
EXCLUDED_MODELS = {"Mistral-7B", "Phi-3.5-mini"}
MODELS_SIZED = [m for m in MODELS_ALL if m in MODEL_SIZE_B and m not in EXCLUDED_MODELS]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def _ci95(values: np.ndarray) -> float:
    n = len(values)
    return 1.96 * values.std() / np.sqrt(n) if n >= 2 else 0.0


def _question_means(df: pd.DataFrame, col: str, grp_col: str) -> pd.DataFrame:
    return df.groupby([grp_col, "question_id"])[col].mean().reset_index()


def _x_with_shift(size: float, group: str) -> float:
    shifts = {
        "VLM": -0.015,
        "VLM backbone decoder": 0.015,
        "standalone LLM": -0.015,
        "standalone LLM (think)": 0.015,
    }
    return float(size) * (1.0 + float(shifts.get(group, 0.0)))


def _configure_xaxis(ax):
    sizes = sorted(set(MODEL_SIZE_B[m] for m in MODELS_SIZED))
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(int(s)) if s == int(s) else str(s) for s in sizes], fontsize=9)
    ax.set_xlabel("Parameters (B)", fontsize=11)
    margin = 0.15
    ax.set_xlim(sizes[0] * (10 ** -margin), sizes[-1] * (10 ** margin))


def _draw_hh(ax, hh_mean: float, hh_ci: float):
    ax.axhline(hh_mean, color=HH_COLOR, lw=1.6, ls="--", zorder=1, label="Human baseline")
    if hh_ci > 0:
        ax.axhspan(hh_mean - hh_ci, hh_mean + hh_ci, color=HH_COLOR, alpha=0.08, zorder=0)


def _ensure_hh_visible(ax, hh_mean: float):
    ylo, yhi = ax.get_ylim()
    pad = (yhi - ylo) * 0.08
    if hh_mean > yhi - pad:
        ax.set_ylim(ylo, hh_mean + pad)
    elif hh_mean < ylo + pad:
        ax.set_ylim(hh_mean - pad, yhi)


def _plot_by_groups(ax, stats: pd.DataFrame, ylabel: str, hh_mean: float, hh_ci: float):
    _draw_hh(ax, hh_mean, hh_ci)
    group_order = ["VLM", "VLM backbone decoder", "standalone LLM (think)", "standalone LLM"]

    for grp in group_order:
        gdf = stats[stats["group"] == grp]
        if gdf.empty:
            continue
        color = GROUP_COLORS.get(grp, "#888888")
        mkr = GROUP_MARKER.get(grp, "o")
        hollow = GROUP_HOLLOW.get(grp, False)
        mfc = "none" if hollow else color
        mec = color if hollow else "white"
        mew = 1.2 if hollow else 0.5
        alpha = 0.50 if grp == "standalone LLM (think)" else 0.88
        point_z = 2 if grp == "standalone LLM (think)" else 3
        line_z = 1 if grp == "standalone LLM (think)" else 2
        label = grp.replace("standalone LLM", "SA-LLM").replace("VLM backbone decoder", "Backbone")

        x = np.array([_x_with_shift(s, grp) for s in gdf["size"].values], dtype=float)
        ax.errorbar(
            x, gdf["mean"].values, yerr=gdf["ci"].values,
            fmt=mkr, color=color, ms=7, capsize=3, capthick=0.8,
            elinewidth=0.8, alpha=alpha, zorder=point_z,
            markerfacecolor=mfc, markeredgecolor=mec, markeredgewidth=mew,
            label=label,
        )
        if len(np.unique(x)) >= 3:
            coeffs = np.polyfit(np.log(x), gdf["mean"].values, 1)
            x_rng = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
            y_fit = np.polyval(coeffs, np.log(x_rng))
            ax.plot(x_rng, y_fit, color=color, ls=":" if hollow else "--", lw=1.2, alpha=0.35 if grp == "standalone LLM (think)" else 0.45, zorder=line_z)

    hh_handle = mlines.Line2D([], [], color=HH_COLOR, ls="--", lw=1.6, label="Human baseline")
    ax.set_ylabel(ylabel, fontsize=11)
    _configure_xaxis(ax)
    _ensure_hh_visible(ax, hh_mean)
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l != "Human baseline"]
    ax.legend(
        handles=[h for h, _ in filtered] + [hh_handle],
        labels=[l for _, l in filtered] + ["Human baseline"],
        fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=True,
    )


print(f"\nLoading human data (min_answers={MIN_ANSWERS_DEFAULT})…")
participants, common_qids, human_df_full, _ = load_human_subset(
    ROOT, min_answers=MIN_ANSWERS_DEFAULT, translate=False, verbose=True
)
print(f"  common_qids: {len(common_qids)}  |  participants: {len(participants)}")

pair_cache = pd.read_parquet(EXPORTS / "pair_cache.parquet")
pair_cache = extend_pair_cache_with_yesno(pair_cache, EXPORTS)
pair_cache = pair_cache[pair_cache["question_id"].isin(common_qids)].copy()
pair_cache = pair_cache[pair_cache["variant"] == VARIANT].copy()

raw_acc = read_export(ROOT, "responses_model_inst_blind.csv")
raw_acc = raw_acc[(raw_acc["question_id"].isin(common_qids)) & (raw_acc["variant"] == VARIANT)].copy()


def model_stats_agreement(col: str) -> pd.DataFrame:
    sub = pair_cache[pair_cache["pair_type"] == "HM"].copy()
    q_means = _question_means(sub, col, "subject_2")
    rows = []
    for m in MODELS_SIZED:
        mq = q_means[q_means["subject_2"] == m][col].dropna().values
        if len(mq) == 0:
            continue
        rows.append({"model": m, "mean": mq.mean(), "ci": _ci95(mq), "size": MODEL_SIZE_B[m], "group": MODEL_GROUP.get(m, "standalone LLM")})
    return pd.DataFrame(rows)


def model_stats_accuracy() -> pd.DataFrame:
    q_means = _question_means(raw_acc, "accuracy", "model")
    rows = []
    for m in MODELS_SIZED:
        mq = q_means[q_means["model"] == m]["accuracy"].values
        if len(mq) == 0:
            continue
        rows.append({"model": m, "mean": mq.mean(), "ci": _ci95(mq), "size": MODEL_SIZE_B[m], "group": MODEL_GROUP.get(m, "standalone LLM")})
    return pd.DataFrame(rows)


def human_baseline_agreement(col: str) -> tuple[float, float]:
    hh = pair_cache[pair_cache["pair_type"] == "HH"].copy()
    q_vals = hh.groupby("question_id")[col].mean().dropna().values
    return q_vals.mean(), _ci95(q_vals)


def human_baseline_accuracy() -> tuple[float, float]:
    hf = human_df_full[human_df_full["question_id"].isin(common_qids)]
    hf = hf[hf["variant"] == VARIANT]
    q_vals = hf.groupby("question_id")["accuracy"].mean().values
    return q_vals.mean(), _ci95(q_vals)


stats_acc = model_stats_accuracy()
hh_acc, hh_acc_ci = human_baseline_accuracy()
fig, ax = plt.subplots(figsize=(8, 5))
_plot_by_groups(ax, stats_acc, "Mean accuracy", hh_acc, hh_acc_ci)
plt.tight_layout()
acc_path = OUT_DIR / "inst_blind_accuracy_groups_vC_q113_h40_yesno.png"
fig.savefig(acc_path, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {acc_path}")

stats_sbert = model_stats_agreement("sbert_score")
hh_sbert, hh_sbert_ci = human_baseline_agreement("sbert_score")
fig, ax = plt.subplots(figsize=(8, 5))
_plot_by_groups(ax, stats_sbert, "Mean SBERT cosine", hh_sbert, hh_sbert_ci)
plt.tight_layout()
sbert_path = OUT_DIR / "inst_blind_agreement_groups_sbert_vC_q113_h40_yesno.png"
fig.savefig(sbert_path, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {sbert_path}")

print(f"\nDone. Saved to: {OUT_DIR}")
