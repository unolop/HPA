from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from notebooks.helpers import (
    distribution_base_table_variants,
    _answer_type_subset,
    _grouped_slice_metrics,
    _mark_abstentions,
)

OUT_DIR = ROOT / "figures" / "grouped_distribution_categorical_inst_blind_filtered"
LATEX_OUT = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "grouped_distribution_categorical_inst_blind_filtered"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["C", "B", "A"]
VAR_LABELS = ["Orig", "Weak", "Pron"]
VAR_X = np.arange(3)

GROUP_ORDER = ["Human baseline", "VLM", "Backbone Decoder", "Standalone LLM"]
GROUP_COLORS = {
    "Human baseline": "#424242",
    "VLM": "#9E4A3A",
    "Backbone Decoder": "#C0843D",
    "Standalone LLM": "#5E8C61",
}
GROUP_LABELS = {
    "Human baseline": "Human",
    "VLM": "VLM",
    "Backbone Decoder": "Backbone",
    "Standalone LLM": "SA-LLM",
}

OP_ORDER = ["attr", "comp", "cause", "count", "ident", "spat", "exist", "act", "know", "temp", "text", "other"]
ENT_ORDER = ["object", "person", "animal", "food", "place", "vehicle", "product", "text", "other"]

ANSWER_TYPE_LABEL = {"yes/no": "yesno", "number": "number"}
METRIC_SPECS = {
    "js": {"col": "js", "ylabel": "Mean JS", "ylim": None},
    "tv": {"col": "tv", "ylabel": "Mean TV", "ylim": None},
    "cramers_v": {"col": "cramers_v", "ylabel": "Mean Cramer's V", "ylim": (0.0, 1.0)},
    "sig_rate": {"col": "sig_rate", "ylabel": "% sig slices", "ylim": (0.0, 100.0)},
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def _variant_name(v: str) -> str:
    return {"C": "Original", "B": "Weaker-object", "A": "Pronominalized"}[v]


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _group_ordered_slices(values: list[str], dimension: str) -> list[str]:
    order = OP_ORDER if dimension == "op" else ENT_ORDER
    seen = set(values)
    ordered = [v for v in order if v in seen]
    ordered.extend([v for v in values if v not in set(ordered)])
    return ordered


def _shared_nonabstaining(human_sub: pd.DataFrame, model_sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_sub = model_sub[~model_sub["is_abstained"]].copy()
    kept_qvs = set(zip(model_sub["question_id"].astype(int), model_sub["variant"].astype(str)))
    human_sub = human_sub[
        [qv in kept_qvs for qv in zip(human_sub["question_id"].astype(int), human_sub["variant"].astype(str))]
    ].copy()
    human_sub = human_sub[~human_sub["is_abstained"]].copy()
    return human_sub, model_sub


def build_slice_df(answer_type: str, dimension: str) -> pd.DataFrame:
    human_all, model_all = distribution_base_table_variants(condition="inst_blind", variants=tuple(VARIANTS))
    human_all = _mark_abstentions(human_all)
    model_all = _mark_abstentions(model_all)
    key = "op" if dimension == "op" else "ent"
    rows: list[dict] = []

    # Human leave-one-out baseline
    for v in VARIANTS:
        hv = human_all[human_all["variant"] == v].copy()
        hv = _answer_type_subset(hv, answer_type)
        hv = hv[~hv["is_abstained"]].copy()
        for participant, self_df in hv.groupby("participant"):
            other = hv[hv["participant"] != participant].copy()
            slice_df = _grouped_slice_metrics(other, self_df.copy(), answer_type, dimension, min_questions_per_slice=3)
            if slice_df.empty:
                continue
            for _, row in slice_df.iterrows():
                rows.append(
                    {
                        "group": "Human baseline",
                        "variant": v,
                        "slice": row["slice"],
                        "js": float(row["js"]),
                        "tv": float(row["tv"]),
                        "cramers_v": float(row["cramers_v"]),
                        "sig_rate": float(100.0 * row["sig"]),
                        "n_questions": int(row["n_questions"]),
                    }
                )

    # Model groups
    for v in VARIANTS:
        hv = human_all[human_all["variant"] == v].copy()
        mv = model_all[model_all["variant"] == v].copy()
        hv = _answer_type_subset(hv, answer_type)
        mv = _answer_type_subset(mv, answer_type)
        for model_name, msub in mv.groupby("model"):
            group = {
                "InternVL-8B": "VLM",
                "Qwen3-VL-8B": "VLM",
                "LLaVA-1.5-7B": "VLM",
                "LLaVA-Mistral": "VLM",
                "LLaVA-Vicuna": "VLM",
                "InternVL-8B (LM)": "Backbone Decoder",
                "Qwen3-VL-8B (LM)": "Backbone Decoder",
                "LLaVA-1.5 (LM)": "Backbone Decoder",
                "LLaVA-Mistral (LM)": "Backbone Decoder",
                "LLaVA-Vicuna (LM)": "Backbone Decoder",
                "Qwen3-8B": "Standalone LLM",
                "Qwen2.5-7B-Instruct": "Standalone LLM",
                "Vicuna-7B": "Standalone LLM",
            }.get(model_name, "Other")
            if group == "Other":
                continue
            h_use, m_use = _shared_nonabstaining(hv.copy(), msub.copy())
            slice_df = _grouped_slice_metrics(h_use, m_use, answer_type, dimension, min_questions_per_slice=3)
            if slice_df.empty:
                continue
            for _, row in slice_df.iterrows():
                rows.append(
                    {
                        "group": group,
                        "variant": v,
                        "slice": row["slice"],
                        "js": float(row["js"]),
                        "tv": float(row["tv"]),
                        "cramers_v": float(row["cramers_v"]),
                        "sig_rate": float(100.0 * row["sig"]),
                        "n_questions": int(row["n_questions"]),
                    }
                )

    return pd.DataFrame(rows)


def aggregate_for_plot(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    metric_col = METRIC_SPECS[metric_key]["col"]
    out = []
    for (group, variant, slice_name), sdf in df.groupby(["group", "variant", "slice"]):
        mean, lo, hi = _bootstrap_ci(sdf[metric_col].values)
        out.append(
            {
                "group": group,
                "variant": variant,
                "slice": slice_name,
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_questions": int(np.median(sdf["n_questions"])) if len(sdf) else 0,
            }
        )
    return pd.DataFrame(out)


def plot_answer_type(answer_type: str, metric_key: str):
    metric = METRIC_SPECS[metric_key]
    op_df = aggregate_for_plot(build_slice_df(answer_type, "op"), metric_key)
    ent_df = aggregate_for_plot(build_slice_df(answer_type, "ent"), metric_key)
    op_slices = _group_ordered_slices(sorted(op_df["slice"].unique().tolist()), "op")
    ent_slices = _group_ordered_slices(sorted(ent_df["slice"].unique().tolist()), "ent")
    ncols = max(len(op_slices), len(ent_slices))

    vals = pd.concat([op_df["mean"], ent_df["mean"]]).dropna().astype(float)
    if metric["ylim"] is None:
        lo = float(vals.min()) if len(vals) else 0.0
        hi = float(vals.max()) if len(vals) else 1.0
        pad = max(0.015, 0.12 * (hi - lo))
        ylim = (max(0.0, lo - pad), hi + pad)
    else:
        ylim = metric["ylim"]

    fig = plt.figure(figsize=(ncols * 1.7, 4.2))
    gs = fig.add_gridspec(2, ncols, hspace=0.38, wspace=0.22)

    for row_idx, (data, slices, row_label) in enumerate([(op_df, op_slices, "Operation"), (ent_df, ent_slices, "Entity")]):
        for col_idx, slice_name in enumerate(slices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            sub = data[data["slice"] == slice_name]
            for group in GROUP_ORDER:
                gsub = sub[sub["group"] == group]
                if gsub.empty:
                    continue
                ys = []
                los = []
                his = []
                ns = []
                for v in VARIANTS:
                    vsub = gsub[gsub["variant"] == v]
                    ys.append(float(vsub["mean"].mean()) if not vsub.empty else np.nan)
                    los.append(float(vsub["ci_lo"].mean()) if not vsub.empty else np.nan)
                    his.append(float(vsub["ci_hi"].mean()) if not vsub.empty else np.nan)
                    if not vsub.empty:
                        ns.extend(vsub["n_questions"].tolist())
                ys = np.asarray(ys, dtype=float)
                los = np.asarray(los, dtype=float)
                his = np.asarray(his, dtype=float)
                valid = np.isfinite(ys)
                if not np.any(valid):
                    continue
                style = dict(
                    color=GROUP_COLORS[group],
                    lw=1.8 if group != "Human baseline" else 1.6,
                    marker="o" if group != "Human baseline" else "D",
                    markersize=4.8 if group != "Human baseline" else 4.2,
                    markerfacecolor=GROUP_COLORS[group] if group != "Human baseline" else "white",
                    markeredgecolor="white" if group != "Human baseline" else GROUP_COLORS[group],
                    markeredgewidth=0.5,
                    ls="-" if group != "Human baseline" else "--",
                    zorder=3,
                )
                ax.plot(VAR_X[valid], ys[valid], **style)
                ci_valid = valid & np.isfinite(los) & np.isfinite(his)
                if np.any(ci_valid):
                    x_ci = VAR_X[ci_valid]
                    y_ci = ys[ci_valid]
                    yerr = np.vstack([y_ci - los[ci_valid], his[ci_valid] - y_ci])
                    ax.errorbar(
                        x_ci, y_ci, yerr=yerr,
                        color=GROUP_COLORS[group], lw=1.0, ls="none",
                        capsize=2.3, capthick=0.9, elinewidth=0.9, zorder=2,
                    )
                median_n = int(np.median(ns)) if ns else 0

            ax.set_title(slice_name, fontsize=8.5, fontweight="bold")
            ax.set_xticks(VAR_X)
            ax.set_xticklabels(VAR_LABELS, fontsize=7)
            ax.set_xlim(-0.3, 2.3)
            ax.set_ylim(*ylim)
            ax.tick_params(axis="y", labelsize=7)
            if col_idx == 0:
                ax.set_ylabel(metric["ylabel"], fontsize=7.5)
                ax.annotate(row_label, xy=(-0.45, 0.5), xycoords="axes fraction",
                            fontsize=8, fontweight="bold", rotation=90, va="center", ha="center")
            else:
                ax.set_yticklabels([])
            ns = sub["n_questions"].unique()
            if len(ns) > 0:
                ax.annotate(
                    f"n≈{int(np.median(ns))}q", xy=(0.97, 0.04), xycoords="axes fraction",
                    fontsize=6, ha="right", color="#666"
                )

        for col_idx in range(len(slices), ncols):
            fig.add_subplot(gs[row_idx, col_idx]).set_visible(False)

    handles = [
        mlines.Line2D([], [], color=GROUP_COLORS[g], lw=1.8 if g != "Human baseline" else 1.6,
                      ls="-" if g != "Human baseline" else "--",
                      marker="o" if g != "Human baseline" else "D",
                      markersize=5 if g != "Human baseline" else 4.2,
                      markerfacecolor=GROUP_COLORS[g] if g != "Human baseline" else "white",
                      markeredgecolor="white" if g != "Human baseline" else GROUP_COLORS[g],
                      markeredgewidth=0.5, label=GROUP_LABELS[g])
        for g in GROUP_ORDER
    ]
    fig.legend(handles=handles, fontsize=7, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=True, handletextpad=0.4, columnspacing=0.8)
    fig.subplots_adjust(bottom=0.10)

    stem = f"{ANSWER_TYPE_LABEL[answer_type]}_{metric_key}.png"
    out = OUT_DIR / stem
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_OUT / stem)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    for answer_type in ("yes/no", "number"):
        for metric_key in ("js", "tv", "cramers_v", "sig_rate"):
            plot_answer_type(answer_type, metric_key)
    print(f"Done. Saved to: {OUT_DIR}")
