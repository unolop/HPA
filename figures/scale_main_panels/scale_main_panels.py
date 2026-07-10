from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from config import MODELS_ALL, MODEL_GROUP, MIN_ANSWERS_DEFAULT
from helpers import clear_output_plots, get_exports_dir, load_cleaned_pair_cache, load_human_subset, read_export, filter_abstained_pairs
from utils.constants import GROUP_COLORS, GROUP_MARKER, GROUP_HOLLOW, MODEL_SIZE_B


OUT_DIR = ROOT / "figures/scale_main_panels"
LATEX_OUT = ROOT / "latex/AAAI2026/LaTeX/figures/scale_metrics"
LATEX_OUT.mkdir(parents=True, exist_ok=True)

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


# ── Family/model-variant definitions ─────────────────────────────────────────
# Markers match the reference scatter figure conventions.
# Think models share SA-LLM color but get a distinct "+" marker.
FAMILY_MARKER = {
    "InternVL":    "o",   # circle
    "Qwen3":       "s",   # square  (no-think)
    "LLaVA-1.5":  "D",   # diamond
    "Mistral":     "^",   # triangle up
    "Vicuna":      "v",   # triangle down
    "Qwen3 (think)": "P", # plus-filled
    "Qwen2.5":     "X",   # x-mark
    "Other":       "*",
}

# Legend order matching reference figure
FAMILY_ORDER = ["InternVL", "Qwen3", "LLaVA-1.5", "Mistral", "Vicuna",
                "Qwen3 (think)", "Qwen2.5"]

def _get_family(model: str) -> str:
    m = model.replace(" (LM)", "").replace("-7B", "").replace("-13B", "").replace("-2B", "").replace("-8B", "").replace("-4B", "").replace("-32B", "").replace("-1B", "").strip()
    if "InternVL" in model:
        return "InternVL"
    if "Qwen3-VL" in model:
        return "Qwen3"          # VLM Qwen3 family uses same Qwen3 square
    if "Qwen3" in model and "think" in model.lower():
        return "Qwen3 (think)"
    if "Qwen3" in model:
        return "Qwen3"
    if "Qwen2.5" in model:
        return "Qwen2.5"
    if "LLaVA-1.5" in model or "llava-1.5" in model.lower():
        return "LLaVA-1.5"
    if "Mistral" in model:
        return "Mistral"
    if "Vicuna" in model or "vicuna" in model.lower():
        return "Vicuna"
    return "Other"


def _x_with_shift(size: float, family: str) -> float:
    # Small jitter per family to avoid overlap at same size
    fam_list = list(FAMILY_MARKER.keys())
    idx = fam_list.index(family) if family in fam_list else 0
    n = len(fam_list)
    shift = (idx / (n - 1) - 0.5) * 0.06  # spread across ±3%
    return float(size) * (1.0 + shift)


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

    # Add family column
    stats = stats.copy()
    stats["family"] = stats["model"].apply(_get_family)

    group_order = ["VLM", "VLM backbone decoder", "standalone LLM (think)", "standalone LLM"]
    group_alpha  = {"standalone LLM (think)": 0.55}
    group_zorder = {"standalone LLM (think)": 2}

    # Collect all plotted families for ordered legend
    plotted_families = set()

    for grp in group_order:
        gdf = stats[stats["group"] == grp]
        if gdf.empty:
            continue
        color  = GROUP_COLORS.get(grp, "#888888")
        hollow = GROUP_HOLLOW.get(grp, False)
        alpha  = group_alpha.get(grp, 0.88)
        pz     = group_zorder.get(grp, 3)
        lz     = pz - 1

        for fam, fdf in gdf.groupby("family"):
            mkr = FAMILY_MARKER.get(fam, "*")
            mfc = "none" if hollow else color
            mec = color  if hollow else "white"
            mew = 1.4    if hollow else 0.5

            x = np.array([_x_with_shift(s, fam) for s in fdf["size"].values], dtype=float)
            # Marker size scales with log(params)
            ms_arr = [4 + 8 * (np.log10(max(s, 0.3)) / np.log10(32))
                      for s in fdf["size"].values]
            for xi, yi, ei, msi in zip(x, fdf["mean"].values, fdf["ci"].values, ms_arr):
                ax.errorbar(xi, yi, yerr=ei,
                    fmt=mkr, color=color, ms=msi, capsize=3, capthick=0.8,
                    elinewidth=0.8, alpha=alpha, zorder=pz,
                    markerfacecolor=mfc, markeredgecolor=mec, markeredgewidth=mew,
                    label=None)
            if len(np.unique(x)) >= 2:
                coeffs = np.polyfit(np.log(x), fdf["mean"].values, 1)
                x_rng  = np.logspace(np.log10(x.min()), np.log10(x.max()), 40)
                ax.plot(x_rng, np.polyval(coeffs, np.log(x_rng)),
                        color=color, ls=":" if hollow else "--", lw=1.1,
                        alpha=0.35 if hollow else 0.45, zorder=lz)
            plotted_families.add(fam)

    # Legend: model families in reference order, then human baseline
    legend_handles = []
    for fam in FAMILY_ORDER:
        if fam not in plotted_families:
            continue
        mkr = FAMILY_MARKER.get(fam, "*")
        legend_handles.append(
            mlines.Line2D([], [], marker=mkr, color="#555555",
                          markerfacecolor="#555555", markersize=7,
                          linestyle="none", label=fam)
        )
    legend_handles.append(
        mlines.Line2D([], [], color=HH_COLOR, ls="--", lw=1.6, label="Human baseline")
    )

    ax.set_ylabel(ylabel, fontsize=11)
    _configure_xaxis(ax)
    _ensure_hh_visible(ax, hh_mean)
    ax.legend(handles=legend_handles, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=True)


print(f"\nLoading human data (min_answers={MIN_ANSWERS_DEFAULT})…")
participants, common_qids, human_df_full, _ = load_human_subset(
    ROOT, min_answers=MIN_ANSWERS_DEFAULT, translate=False, verbose=True
)
print(f"  common_qids: {len(common_qids)}  |  participants: {len(participants)}")

pair_cache = load_cleaned_pair_cache(
    ROOT,
    condition="inst_blind",
    include_yesno=True,
    verbose=True,
)
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
        rows.append({"model": m, "mean": mq.mean(), "ci": _ci95(mq), "size": MODEL_SIZE_B[m], "group": MODEL_GROUP.get(m, "standalone LLM"), "model": m})
    return pd.DataFrame(rows)


def model_stats_accuracy() -> pd.DataFrame:
    q_means = _question_means(raw_acc, "accuracy", "model")
    rows = []
    for m in MODELS_SIZED:
        mq = q_means[q_means["model"] == m]["accuracy"].values
        if len(mq) == 0:
            continue
        rows.append({"model": m, "mean": mq.mean(), "ci": _ci95(mq), "size": MODEL_SIZE_B[m], "group": MODEL_GROUP.get(m, "standalone LLM"), "model": m})
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


def _save(fig, fname, suffix=''):
    name = fname.replace('.png', f'{suffix}.png')
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight")
    fig.savefig(LATEX_OUT / name, dpi=220, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / name}")


# ── Raw (unfiltered) ─────────────────────────────────────────────────────────
stats_acc = model_stats_accuracy()
hh_acc, hh_acc_ci = human_baseline_accuracy()
fig, ax = plt.subplots(figsize=(8, 5))
_plot_by_groups(ax, stats_acc, "Mean accuracy", hh_acc, hh_acc_ci)
plt.tight_layout()
_save(fig, "inst_blind_accuracy_groups_vC_q113_h40_yesno.png")
plt.close(fig)

stats_sbert = model_stats_agreement("sbert_score")
hh_sbert, hh_sbert_ci = human_baseline_agreement("sbert_score")
fig, ax = plt.subplots(figsize=(8, 5))
_plot_by_groups(ax, stats_sbert, "Mean SBERT cosine", hh_sbert, hh_sbert_ci)
plt.tight_layout()
_save(fig, "inst_blind_agreement_groups_sbert_vC_q113_h40_yesno.png")
plt.close(fig)

# ── Abstention-filtered ───────────────────────────────────────────────────────
print("\nGenerating abstention-filtered versions…")
pair_cache_filt = filter_abstained_pairs(pair_cache)

# For accuracy: exclude rows where model answer is abstained
from analysis.utils.abstention import classify, is_abstained
raw_acc_filt = raw_acc[~raw_acc['response'].fillna('').astype(str).apply(
    lambda x: is_abstained(classify(x, None)))].copy()


def model_stats_agreement_filt(col: str) -> pd.DataFrame:
    sub = pair_cache_filt[pair_cache_filt["pair_type"] == "HM"].copy()
    q_means = _question_means(sub, col, "subject_2")
    rows = []
    for m in MODELS_SIZED:
        mq = q_means[q_means["subject_2"] == m][col].dropna().values
        if len(mq) == 0:
            continue
        rows.append({"model": m, "mean": mq.mean(), "ci": _ci95(mq),
                     "size": MODEL_SIZE_B[m], "group": MODEL_GROUP.get(m, "standalone LLM")})
    return pd.DataFrame(rows)


def model_stats_accuracy_filt() -> pd.DataFrame:
    q_means = _question_means(raw_acc_filt, "accuracy", "model")
    rows = []
    for m in MODELS_SIZED:
        mq = q_means[q_means["model"] == m]["accuracy"].values
        if len(mq) == 0:
            continue
        rows.append({"model": m, "mean": mq.mean(), "ci": _ci95(mq),
                     "size": MODEL_SIZE_B[m], "group": MODEL_GROUP.get(m, "standalone LLM")})
    return pd.DataFrame(rows)


def human_baseline_agreement_filt(col: str) -> tuple[float, float]:
    hh = pair_cache_filt[pair_cache_filt["pair_type"] == "HH"].copy()
    q_vals = hh.groupby("question_id")[col].mean().dropna().values
    return q_vals.mean(), _ci95(q_vals)


stats_acc_filt = model_stats_accuracy_filt()
fig, ax = plt.subplots(figsize=(8, 5))
_plot_by_groups(ax, stats_acc_filt, "Mean accuracy (abstention filtered)", hh_acc, hh_acc_ci)
plt.tight_layout()
_save(fig, "inst_blind_accuracy_groups_vC_q113_h40_yesno.png", "_abst_filtered")
plt.close(fig)

stats_sbert_filt = model_stats_agreement_filt("sbert_score")
hh_sbert_filt, hh_sbert_filt_ci = human_baseline_agreement_filt("sbert_score")
fig, ax = plt.subplots(figsize=(8, 5))
_plot_by_groups(ax, stats_sbert_filt, "Mean SBERT cosine (abstention filtered)",
                hh_sbert_filt, hh_sbert_filt_ci)
plt.tight_layout()
_save(fig, "inst_blind_agreement_groups_sbert_vC_q113_h40_yesno.png", "_abst_filtered")
plt.close(fig)

print(f"\nDone. Saved to: {OUT_DIR} and {LATEX_OUT}")
