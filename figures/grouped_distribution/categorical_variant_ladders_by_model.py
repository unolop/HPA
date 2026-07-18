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
    load_human_responses,
    _clean_answer_series,
    _answer_type_subset,
    _grouped_slice_metrics,
    _mark_abstentions,
)

OUT_DIR = ROOT / "figures" / "grouped_distribution_categorical_by_model_inst_blind_filtered"
LATEX_OUT = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "grouped_distribution_categorical_by_model_inst_blind_filtered"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["C", "B", "A"]
VAR_LABELS = ["Orig", "Weak", "Pron"]
VAR_X = np.arange(3)

MODELS_7B = {
    "VLM": [
        "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna", "InternVL-8B",
    ],
    "VLM backbone decoder": [
        "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)",
        "LLaVA-Vicuna (LM)", "InternVL-8B (LM)",
    ],
    "standalone LLM": [
        "Qwen3-8B", "Qwen3-8B (think)", "Mistral-7B", "Vicuna-7B", "Qwen2.5-7B-Instruct",
    ],
}
ROW_ORDER = ["VLM", "VLM backbone decoder", "standalone LLM"]
ROW_LABEL = {
    "VLM": "VLM",
    "VLM backbone decoder": "Backbone\nDecoder",
    "standalone LLM": "Standalone\nLLM",
}
SHORT_LABEL = {
    "Qwen3-VL-8B": "Qwen3-VL-8B",
    "LLaVA-1.5-7B": "LLaVA-1.5-7B",
    "LLaVA-Mistral": "LLaVA-Mistral",
    "LLaVA-Vicuna": "LLaVA-Vicuna",
    "InternVL-8B": "InternVL-8B",
    "Qwen3-VL-8B (LM)": "Q3-VL-8B",
    "LLaVA-1.5 (LM)": "LLaVA-1.5",
    "LLaVA-Mistral (LM)": "Mistral",
    "LLaVA-Vicuna (LM)": "Vicuna",
    "InternVL-8B (LM)": "InternVL",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-8B (think)": "Qwen3-8B (T)",
    "Mistral-7B": "Mistral-7B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Vicuna-7B": "Vicuna-7B",
}

DIM_COLORS = {"Operation": "#8A4F2A", "Entity": "#5B84B8"}
DIM_LABELS = {"Operation": "Operation", "Entity": "Entity"}
METRIC_SPECS = {
    "js": {"col": "js", "ylabel": "Mean JS", "ylim": None},
    "tv": {"col": "tv", "ylabel": "Mean TV", "ylim": None},
    "cramers_v": {"col": "cramers_v", "ylabel": "Mean Cramer's V", "ylim": (0.0, 1.0)},
    "sig_rate": {"col": "sig_rate", "ylabel": "% sig slices", "ylim": (0.0, 100.0)},
}
ANSWER_TYPE_LABEL = {"yes/no": "yesno", "number": "number"}
MATCHED_MODELS_WITH_THINK = sorted({m for models in MODELS_7B.values() for m in models})

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


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


def _distribution_base_table_variants_with_think(
    condition: str = "inst_blind",
    variants: tuple[str, ...] = ("C", "B", "A"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from analysis.utils.vqa import VQAAnswerMapper

    human = load_human_responses()
    model = pd.read_csv(
        ROOT / "analysis" / "session2" / "exports" / f"responses_model_{condition}.csv"
    )
    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {
        int(qid): ann.get("answer_type", "other")
        for qid, ann in mapper.annotations.items()
    }

    human = human[human["variant"].isin(variants)].copy()
    model = model[(model["variant"].isin(variants)) & (model["model"].isin(MATCHED_MODELS_WITH_THINK))].copy()
    qvs = set(zip(human["question_id"].astype(int), human["variant"].astype(str)))
    human = human[[qv in qvs for qv in zip(human["question_id"].astype(int), human["variant"].astype(str))]].copy()
    model = model[[qv in qvs for qv in zip(model["question_id"].astype(int), model["variant"].astype(str))]].copy()

    human["answer_type"] = human["question_id"].astype(int).map(qid2atype).fillna("other")
    model["answer_type"] = model["question_id"].astype(int).map(qid2atype).fillna("other")
    human["clean_response"] = _clean_answer_series(human["response"])
    model["clean_response"] = _clean_answer_series(model["response"])
    return human, model


def _shared_nonabstaining(human_sub: pd.DataFrame, model_sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_sub = model_sub[~model_sub["is_abstained"]].copy()
    kept_qvs = set(zip(model_sub["question_id"].astype(int), model_sub["variant"].astype(str)))
    human_sub = human_sub[
        [qv in kept_qvs for qv in zip(human_sub["question_id"].astype(int), human_sub["variant"].astype(str))]
    ].copy()
    human_sub = human_sub[~human_sub["is_abstained"]].copy()
    return human_sub, model_sub


def build_long_df(answer_type: str, metric_key: str) -> pd.DataFrame:
    metric_col = METRIC_SPECS[metric_key]["col"]
    human_all, model_all = _distribution_base_table_variants_with_think(condition="inst_blind", variants=tuple(VARIANTS))
    human_all = _mark_abstentions(human_all)
    model_all = _mark_abstentions(model_all)
    rows: list[dict] = []

    for dimension, key in [("Operation", "op"), ("Entity", "ent")]:
        # Human baseline by variant
        for v in VARIANTS:
            hv = _answer_type_subset(human_all[human_all["variant"] == v].copy(), answer_type)
            hv = hv[~hv["is_abstained"]].copy()
            participant_metrics = []
            for participant, self_df in hv.groupby("participant"):
                other = hv[hv["participant"] != participant].copy()
                slice_df = _grouped_slice_metrics(other, self_df.copy(), answer_type, "op" if dimension == "Operation" else "ent", min_questions_per_slice=3)
                if slice_df.empty:
                    continue
                participant_metrics.append(float(slice_df[metric_col].mean()) if metric_col != "sig_rate" else float(100.0 * slice_df["sig"].mean()))
            mean, lo, hi = _bootstrap_ci(np.asarray(participant_metrics, dtype=float))
            rows.append({"model": "Human LOO", "group": "Human baseline", "dimension": dimension, "variant": v, "mean": mean, "ci_lo": lo, "ci_hi": hi})

        for group_key, models in MODELS_7B.items():
            for model_name in models:
                m_all = model_all[model_all["model"] == model_name].copy()
                if m_all.empty:
                    continue
                for v in VARIANTS:
                    hv = _answer_type_subset(human_all[human_all["variant"] == v].copy(), answer_type)
                    mv = _answer_type_subset(m_all[m_all["variant"] == v].copy(), answer_type)
                    h_use, m_use = _shared_nonabstaining(hv, mv)
                    slice_df = _grouped_slice_metrics(h_use, m_use, answer_type, "op" if dimension == "Operation" else "ent", min_questions_per_slice=3)
                    if slice_df.empty:
                        continue
                    vals = slice_df[metric_col].values if metric_col != "sig_rate" else (100.0 * slice_df["sig"].values)
                    mean, lo, hi = _bootstrap_ci(vals)
                    rows.append({"model": model_name, "group": group_key, "dimension": dimension, "variant": v, "mean": mean, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def make_grouped_rows_by_model(df: pd.DataFrame, answer_type: str, metric_key: str):
    metric = METRIC_SPECS[metric_key]
    all_vals = df["mean"].dropna().astype(float)
    if metric["ylim"] is None:
        lo = float(all_vals.min()) if len(all_vals) else 0.0
        hi = float(all_vals.max()) if len(all_vals) else 1.0
        pad = max(0.015, 0.12 * (hi - lo))
        ylim = (max(0.0, lo - pad), hi + pad)
    else:
        ylim = metric["ylim"]

    NCOLS = 5
    NROWS = len(ROW_ORDER)
    panel_w, panel_h = 1.35, 1.20
    left_margin = 0.80
    right_margin = 0.05
    top_margin = 0.15
    bottom_margin = 0.58
    fig_w = left_margin + NCOLS * panel_w + right_margin
    fig_h = top_margin + NROWS * panel_h + bottom_margin

    fig, axes = plt.subplots(
        NROWS, NCOLS, figsize=(fig_w, fig_h), squeeze=False,
        sharex=True, sharey=True,
        gridspec_kw={
            "hspace": 0.40, "wspace": 0.15,
            "left": left_margin / fig_w,
            "right": 1.0 - right_margin / fig_w,
            "top": 1.0 - top_margin / fig_h,
            "bottom": bottom_margin / fig_h,
        },
    )

    human = df[df["model"] == "Human LOO"].copy()

    for row_idx, group_key in enumerate(ROW_ORDER):
        models_in_row = [m for m in MODELS_7B[group_key] if m in set(df["model"])]
        for col_idx in range(NCOLS):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(models_in_row):
                ax.set_visible(False)
                continue
            label = models_in_row[col_idx]
            sub = df[df["model"] == label].copy()

            for dim in ["Operation", "Entity"]:
                dsub = sub[sub["dimension"] == dim]
                hsub = human[human["dimension"] == dim]
                ys = []
                los = []
                his = []
                hys = []
                hlos = []
                hhis = []
                for v in VARIANTS:
                    vsub = dsub[dsub["variant"] == v]
                    ys.append(float(vsub["mean"].mean()) if not vsub.empty else np.nan)
                    los.append(float(vsub["ci_lo"].mean()) if not vsub.empty else np.nan)
                    his.append(float(vsub["ci_hi"].mean()) if not vsub.empty else np.nan)
                    vh = hsub[hsub["variant"] == v]
                    hys.append(float(vh["mean"].mean()) if not vh.empty else np.nan)
                    hlos.append(float(vh["ci_lo"].mean()) if not vh.empty else np.nan)
                    hhis.append(float(vh["ci_hi"].mean()) if not vh.empty else np.nan)
                ys = np.asarray(ys, dtype=float)
                los = np.asarray(los, dtype=float)
                his = np.asarray(his, dtype=float)
                hys = np.asarray(hys, dtype=float)
                hlos = np.asarray(hlos, dtype=float)
                hhis = np.asarray(hhis, dtype=float)
                color = DIM_COLORS[dim]
                valid = np.isfinite(ys)
                if np.any(valid):
                    ax.plot(VAR_X[valid], ys[valid], color=color, lw=1.8, marker="o", markersize=4.2, zorder=3)
                    ci_valid = valid & np.isfinite(los) & np.isfinite(his)
                    if np.any(ci_valid):
                        x_ci = VAR_X[ci_valid]
                        y_ci = ys[ci_valid]
                        yerr = np.vstack([y_ci - los[ci_valid], his[ci_valid] - y_ci])
                        ax.errorbar(x_ci, y_ci, yerr=yerr, color=color, lw=1.0, ls="none",
                                    capsize=2.0, capthick=0.8, elinewidth=0.8, zorder=2)
                hvalid = np.isfinite(hys)
                if np.any(hvalid):
                    ax.plot(VAR_X[hvalid], hys[hvalid], color=color, lw=1.3, ls="--", marker=None, alpha=0.85, zorder=1)
                    hci_valid = hvalid & np.isfinite(hlos) & np.isfinite(hhis)
                    if np.any(hci_valid):
                        x_ci = VAR_X[hci_valid]
                        y_ci = hys[hci_valid]
                        yerr = np.vstack([y_ci - hlos[hci_valid], hhis[hci_valid] - y_ci])
                        ax.errorbar(x_ci, y_ci, yerr=yerr, color=color, lw=0.9, ls="none",
                                    capsize=1.8, capthick=0.7, elinewidth=0.7, alpha=0.75, zorder=1)

            ax.set_title(SHORT_LABEL.get(label, label), fontsize=10, pad=3, fontweight="normal")
            ax.set_xticks(VAR_X)
            ax.set_xticklabels(VAR_LABELS, fontsize=8)
            ax.set_ylim(*ylim)
            ax.grid(True, axis="both", alpha=0.18, linewidth=0.6)
            if row_idx == NROWS - 1:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel(metric["ylabel"], fontsize=9)
            else:
                ax.set_ylabel("")

        axes[row_idx][0].annotate(
            ROW_LABEL[group_key],
            xy=(-0.56, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="normal", ha="center", va="center",
            rotation=90, annotation_clip=False,
        )

    handles = [
        mlines.Line2D([], [], color=DIM_COLORS["Operation"], lw=1.8, marker="o", markersize=4.2, label="Operation"),
        mlines.Line2D([], [], color=DIM_COLORS["Operation"], lw=1.3, ls="--", label="Human baseline (Operation)"),
        mlines.Line2D([], [], color=DIM_COLORS["Entity"], lw=1.8, marker="o", markersize=4.2, label="Entity"),
        mlines.Line2D([], [], color=DIM_COLORS["Entity"], lw=1.3, ls="--", label="Human baseline (Entity)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))

    stem = f"{ANSWER_TYPE_LABEL[answer_type]}_{metric_key}_by_model.png"
    out = OUT_DIR / stem
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_OUT / stem)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    for answer_type in ("yes/no", "number"):
        for metric_key in ("js", "tv", "cramers_v", "sig_rate"):
            df = build_long_df(answer_type, metric_key)
            make_grouped_rows_by_model(df, answer_type, metric_key)
    print(f"Done. Saved to: {OUT_DIR}")
