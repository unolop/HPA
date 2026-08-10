from __future__ import annotations

import json
import glob
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))
import plot_style  # noqa: F401
from helpers import filter_abstained_pairs
from analysis.utils.abstention import classify, is_abstained
from analysis.utils.vqa import VQAAnswerMapper

OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR = ROOT / "latex/AAAI2026/LaTeX/figures/hh_hm_grouped_scatter_7b_all_variants"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

EXPORTS = ROOT / "analysis/session2/exports"

MODEL_FAMILIES = [
    ("o", "InternVL-8B", "InternVL-8B (LM)", None, "InternVL"),
    ("X", "Qwen3-VL-8B", "Qwen3-VL-8B (LM)", "Qwen3-8B", "Qwen3"),
    ("D", "LLaVA-1.5-7B", "LLaVA-1.5 (LM)", None, "LLaVA-1.5"),
    ("^", "LLaVA-Mistral", "LLaVA-Mistral (LM)", "Mistral-7B", "Mistral"),
    ("v", "LLaVA-Vicuna", "LLaVA-Vicuna (LM)", "Vicuna-7B", "Vicuna"),
    ("P", None, None, "Qwen3-8B (think)", "Qwen3 (think)"),
    ("s", None, None, "Qwen2.5-7B-Instruct", "Qwen2.5-Instruct"),
]

MODEL_STYLE = {}
for marker, vlm, dec, llm, _ in MODEL_FAMILIES:
    if vlm:
        MODEL_STYLE[vlm] = marker
    if dec:
        MODEL_STYLE[dec] = marker
    if llm:
        MODEL_STYLE[llm] = marker

MODEL_GROUPS = {
    "VLM": ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone Decoder": ["InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)",
                         "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"],
    "Standalone LLM": ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct",
                       "Vicuna-7B", "Mistral-7B"],
}
ROW_ORDER = ["VLM", "Backbone Decoder", "Standalone LLM"]
ROW_LABELS = {"VLM": "VLM", "Backbone Decoder": "Backbone", "Standalone LLM": "SA-LLM"}

_tab20 = plt.colormaps["tab20"]
_tab10 = plt.colormaps["tab10"]

# Use a colorblind-safe base palette (Matplotlib's Tableau colors) and extend it
# only where needed for the larger operation inventory.
_CB10 = [_tab10(i) for i in range(10)]
OP_PALETTE = _CB10 + [_tab20(18)]
ENT_PALETTE = _CB10[:9]

OP_FULL_NAMES = {
    "act": "Action", "attr": "Attribute", "cause": "Causality",
    "comp": "Comparison", "count": "Count", "exist": "Existence",
    "ident": "Identity", "know": "World Know.", "spat": "Spatial",
    "temp": "Temporal", "text": "Text Reading", "other": "Other",
}
ENT_FULL_NAMES = {
    "animal": "Animal", "food": "Food", "object": "Object",
    "other": "Other", "person": "Person", "place": "Place",
    "product": "Product", "text": "Text", "vehicle": "Vehicle",
}

CT_TO_VARIANT = {"question": "C", "weaker_object": "B", "pronominalized": "A"}
EOS_TOKENS = {"<|im_end|>", "</s>", "<eos>", "<|endoftext|>", "<|end|>"}

VLM_DIR_TO_MODEL = {
    "InternVL3_5-8B": "InternVL-8B",
    "llava-1.5-7b-hf": "LLaVA-1.5-7B",
    "llava-v1.6-mistral-7b-hf": "LLaVA-Mistral",
    "llava-v1.6-vicuna-7b-hf": "LLaVA-Vicuna",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
}
LM_DECODER_DIR_TO_MODEL = {
    "InternVL3_5-8B": "InternVL-8B (LM)",
    "llava-1.5-7b-hf": "LLaVA-1.5 (LM)",
    "llava-v1.6-mistral-7b-hf": "LLaVA-Mistral (LM)",
    "llava-v1.6-vicuna-7b-hf": "LLaVA-Vicuna (LM)",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B (LM)",
}
BACKBONE_DIR_TO_MODEL = {
    "Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen3-8B": "Qwen3-8B",
    "vicuna-7b-v1.5": "Vicuna-7B",
}
LOGIT_SOURCES = [
    (ROOT / "evaluation/logits/vlm/pretrained", VLM_DIR_TO_MODEL),
    (ROOT / "evaluation/logits/lm_decoder/pretrained", LM_DECODER_DIR_TO_MODEL),
    (ROOT / "evaluation/logits/backbone/pretrained", BACKBONE_DIR_TO_MODEL),
]


def model_class_map() -> dict[str, str]:
    out = {}
    for grp, models in MODEL_GROUPS.items():
        for m in models:
            out[m] = grp
    return out


def family_handles():
    handles = [
        Line2D([0], [0], marker=marker, color="#666", markerfacecolor="#666",
               linestyle="None", markersize=5, label=label)
        for marker, _, _, _, label in MODEL_FAMILIES
    ]
    return sorted(handles, key=lambda h: h.get_label().lower())


def qgroup_handles(groups_sorted, color_map):
    return [
        Line2D([0], [0], marker="o", color=color_map[g], linestyle="None",
               markersize=4.8, label=g)
        for g in groups_sorted
    ]


def build_palette(n: int, base: list):
    if n <= len(base):
        return base[:n]
    extra = [_tab20(i % 20) for i in range(n)]
    return extra


def save(fig, name: str):
    path = OUTDIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    shutil.copy(path, LATEX_DIR / name)
    plt.close(fig)
    print(f"saved {path}")


def sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def load_accuracy_frames(*, abstfiltered: bool = False):
    human = pd.read_csv(EXPORTS / "responses_human.csv")
    model = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")
    keep_models = {m for ms in MODEL_GROUPS.values() for m in ms}
    model = model[model["model"].isin(keep_models)].copy()
    if abstfiltered:
        model = model[~model["response"].fillna("").astype(str).apply(
            lambda x: is_abstained(classify(x, None))
        )].copy()
    return human, model


def load_confidence_frames(*, abstfiltered: bool = False):
    sem_df = pd.read_json(ROOT / "dataset/vqa/vqa1k_semantics.jsonl", lines=True)[["question_id", "ent", "op"]]
    human_files = sorted(glob.glob(str(ROOT / "evaluation/humans/by_participant/*.json")))
    human_rows = []
    for fp in human_files:
        with open(fp) as f:
            data = json.load(f)
        pid = data.get("code", Path(fp).stem)
        for ans in data.get("answers", []):
            raw_conf = ans.get("confidence", 3)
            human_rows.append({
                "question_id": ans["question_id"],
                "participant": pid,
                "variant": ans.get("variant", "C"),
                "confidence": (raw_conf - 1) / 4.0,
            })
    human_df = pd.DataFrame(human_rows).merge(sem_df, on="question_id", how="left")
    human_qids = set(human_df["question_id"].unique())
    keep_rows = None
    if abstfiltered:
        resp = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")
        keep_models = {m for ms in MODEL_GROUPS.values() for m in ms}
        resp = resp[resp["model"].isin(keep_models)].copy()
        resp = resp[~resp["response"].fillna("").astype(str).apply(
            lambda x: is_abstained(classify(x, None))
        )].copy()
        keep_rows = set(
            zip(
                resp["question_id"].astype(int),
                resp["model"].astype(str),
                resp["variant"].astype(str),
            )
        )

    rows = []
    for base_dir, mapping in LOGIT_SOURCES:
        for dir_name, display in mapping.items():
            fpath = base_dir / dir_name / "vqa_1k_control_inst_blind.jsonl"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    qid = ex["question_id"]
                    if qid not in human_qids:
                        continue
                    logits = ex.get("generated_logits", {})
                    for ct, variant in CT_TO_VARIANT.items():
                        logit_data = logits.get(ct)
                        if not logit_data:
                            continue
                        probs = [np.exp(t["logprob"]) for t in logit_data.get("content", [])
                                 if t.get("token") not in EOS_TOKENS]
                        if not probs:
                            continue
                        if keep_rows is not None and (qid, display, variant) not in keep_rows:
                            continue
                        rows.append({
                            "question_id": qid,
                            "model": display,
                            "variant": variant,
                            "confidence": float(np.mean(probs)),
                        })
    model_df = pd.DataFrame(rows).merge(sem_df, on="question_id", how="left")
    return human_df, model_df


def build_group_metric_frames(metric: str, *, abstfiltered: bool = False):
    cls_map = model_class_map()
    if metric == "accuracy":
        human_df, model_df = load_accuracy_frames(abstfiltered=abstfiltered)
        value_col = "accuracy"
    elif metric == "confidence":
        human_df, model_df = load_confidence_frames(abstfiltered=abstfiltered)
        value_col = "confidence"
    elif metric == "sbert":
        pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
        if abstfiltered:
            pc = filter_abstained_pairs(pc)
        # Filter to free-form (answer_type == "other") questions only
        mapper = VQAAnswerMapper()
        mapper._load()
        qid2atype = {int(qid): ann.get("answer_type", "other")
                     for qid, ann in mapper.annotations.items()}
        text_qids = {qid for qid, atype in qid2atype.items() if atype == "other"}
        pc = pc[pc["question_id"].astype(int).isin(text_qids)].copy()
        hh = pc[pc["pair_type"] == "HH"].copy()
        hm = pc[pc["pair_type"] == "HM"].copy()
        keep_models = {m for ms in MODEL_GROUPS.values() for m in ms}
        hm = hm[hm["subject_2"].isin(keep_models)].copy()
        human_op = hh.groupby("op")["sbert_score"].mean().reset_index().rename(columns={"sbert_score": "human_value"})
        human_ent = hh.groupby("ent")["sbert_score"].mean().reset_index().rename(columns={"sbert_score": "human_value"})
        hm["model_class"] = hm["subject_2"].map(cls_map)
        hm = hm.dropna(subset=["model_class"]).copy()
        op = (hm.groupby(["op", "subject_2", "model_class"])["sbert_score"]
              .mean().reset_index().rename(columns={"subject_2": "model", "sbert_score": "sbert"})
              .merge(human_op, on="op"))
        ent = (hm.groupby(["ent", "subject_2", "model_class"])["sbert_score"]
               .mean().reset_index().rename(columns={"subject_2": "model", "sbert_score": "sbert"})
               .merge(human_ent, on="ent"))
        return op, ent
    else:
        raise ValueError(metric)

    human_op = human_df.groupby("op")[value_col].mean().reset_index().rename(columns={value_col: "human_value"})
    human_ent = human_df.groupby("ent")[value_col].mean().reset_index().rename(columns={value_col: "human_value"})

    model_df["model_class"] = model_df["model"].map(cls_map)
    model_df = model_df.dropna(subset=["model_class"]).copy()

    op = (model_df.groupby(["op", "model", "model_class"])[value_col]
          .mean().reset_index().merge(human_op, on="op"))
    ent = (model_df.groupby(["ent", "model", "model_class"])[value_col]
           .mean().reset_index().merge(human_ent, on="ent"))
    return op, ent


def compute_text_loocv(hh: pd.DataFrame) -> dict:
    """For text-answer-type questions only, compute per-group LOOCV (p5, p95)
    across participants' mean SBERT. Returns {(dim, grp): (p5, p95)}."""
    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other")
                 for qid, ann in mapper.annotations.items()}
    hh = hh.copy()
    hh["answer_type"] = hh["question_id"].astype(int).map(qid2atype).fillna("other")
    text_hh = hh[hh["answer_type"] == "other"].copy()  # VQA "other" = free-form text

    result = {}
    for dim in ("op", "ent"):
        for grp, grp_df in text_hh.groupby(dim):
            per_part = [pdata["sbert_score"].mean()
                        for _, pdata in grp_df.groupby("subject_1")
                        if len(pdata) >= 2]
            if len(per_part) >= 5:
                arr = np.array(per_part)
                result[(dim, grp)] = (float(np.percentile(arr, 5)),
                                      float(np.percentile(arr, 95)))
    return result


def plot_metric_combined(metric: str, *, abstfiltered: bool = False):
    op_df, ent_df = build_group_metric_frames(metric, abstfiltered=abstfiltered)
    op_df["op"] = op_df["op"].map(OP_FULL_NAMES).fillna(op_df["op"])
    ent_df["ent"] = ent_df["ent"].map(ENT_FULL_NAMES).fillna(ent_df["ent"])
    specs = [("op", op_df, "Operation", OP_PALETTE, "op"),
             ("ent", ent_df, "Entity", ENT_PALETTE, "ent")]
    if metric == "accuracy":
        xlab = "Human acc."
        ylab = "Model accuracy"
        lims = [0.0, 1.0]
        metric_col = "accuracy"
    elif metric == "sbert":
        xlab = "Human\u2013Human SBERT"
        ylab = "Human\u2013Model SBERT"
        lims = [0.28, 0.65]
        metric_col = "sbert"
    else:
        xlab = "Human conf."
        ylab = "Model confidence"
        lims = [0.0, 1.0]
        metric_col = "confidence"

    color_maps = {}
    for group_type, df, _, palette, _ in specs:
        groups_sorted = sorted(df[group_type].unique())
        palette_use = build_palette(len(groups_sorted), palette)
        color_maps[group_type] = (groups_sorted, {g: palette_use[i] for i, g in enumerate(groups_sorted)})

    # 2 rows (op / ent) × 3 cols (VLM / Backbone / SA-LLM)
    fig, axes = plt.subplots(
        2, 3, figsize=(8.5, 3.8), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.14, "wspace": 0.10,
                     "left": 0.07, "right": 0.66, "top": 0.88, "bottom": 0.14},
    )
    fig.supylabel(ylab, fontsize=8.5, x=0.02)
    fig.supxlabel(xlab, fontsize=8.5, y=0.05, x=0.32)

    # ── Pre-compute human LOOCV bands (text answer type only) ────────────────
    text_loocv = {}  # (dim, grp) → (p5, p95) — only for free-form groups
    if metric == "sbert":
        pc_raw = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
        if abstfiltered:
            pc_raw = filter_abstained_pairs(pc_raw)
        hh_raw = pc_raw[pc_raw["pair_type"] == "HH"].copy()
        text_loocv = compute_text_loocv(hh_raw)

    # ── Pass 1: collect subplot data + raw Spearman ρ/p ─────────────────────
    subplot_data = {}
    for row_idx, (group_type, df, row_label, _, _) in enumerate(specs):
        groups_sorted, color_map = color_maps[group_type]
        for col_idx, col_title in enumerate(ROW_ORDER):
            sub = df[df["model_class"] == col_title]
            mean_df = (
                sub.groupby(group_type)
                .agg(
                    human_value=("human_value", "mean"),
                    model_mean=(metric_col, "mean"),
                    model_ci=(metric_col, lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
                )
                .reset_index()
            )
            rho, p = (stats.pearsonr(mean_df["human_value"], mean_df["model_mean"])
                      if len(mean_df) >= 3 else (np.nan, np.nan))
            subplot_data[(row_idx, col_idx)] = {
                "group_type": group_type, "col_title": col_title,
                "sub": sub, "mean_df": mean_df, "rho": rho, "p_raw": p,
            }

    # ── Holm-Bonferroni correction across all 6 subplots ─────────────────────
    keys = sorted(subplot_data)
    raw_ps = [subplot_data[k]["p_raw"] for k in keys]
    valid_mask = [not np.isnan(p) for p in raw_ps]
    valid_ps = [p for p, v in zip(raw_ps, valid_mask) if v]
    if valid_ps:
        _, p_corr_vals, _, _ = multipletests(valid_ps, method="holm")
        p_corr_iter = iter(p_corr_vals)
        for k in keys:
            subplot_data[k]["p_corr"] = next(p_corr_iter) if valid_mask[keys.index(k)] else np.nan
    else:
        for k in keys:
            subplot_data[k]["p_corr"] = np.nan

    # ── Pass 2: draw ─────────────────────────────────────────────────────────
    for row_idx, (group_type, df, row_label, _, _) in enumerate(specs):
        groups_sorted, color_map = color_maps[group_type]
        for col_idx, col_title in enumerate(ROW_ORDER):
            ax = axes[row_idx, col_idx]
            sd = subplot_data[(row_idx, col_idx)]
            sub = sd["sub"]
            mean_df = sd["mean_df"]

            # ── Human LOOCV bands for free-text groups ───────────────────
            dim_key = "op" if group_type == "op" else "ent"
            band_hw = 0.012
            for _, mrow in mean_df.iterrows():
                grp_raw = mrow[group_type]  # full name (already mapped)
                # reverse-map to original code for loocv lookup
                rev_op = {v: k for k, v in OP_FULL_NAMES.items()}
                rev_ent = {v: k for k, v in ENT_FULL_NAMES.items()}
                rev_map = rev_op if dim_key == "op" else rev_ent
                grp_code = rev_map.get(grp_raw, grp_raw)
                key = (dim_key, grp_code)
                if key in text_loocv:
                    p5, p95 = text_loocv[key]
                    x = mrow["human_value"]
                    ax.fill_between(
                        [x - band_hw, x + band_hw], p5, p95,
                        alpha=0.18, color="#424242", linewidth=0, zorder=0,
                    )

            for model, msub in sub.groupby("model"):
                marker = MODEL_STYLE.get(model, "o")
                for _, row in msub.iterrows():
                    c = color_map[row[group_type]]
                    ax.scatter(
                        row["human_value"], row[metric_col],
                        s=22, alpha=1.0, color=c, marker=marker,
                        edgecolors="white", linewidth=0.45, zorder=3,
                    )

            for _, row in mean_df.iterrows():
                c = color_map[row[group_type]]
                ax.errorbar(
                    row["human_value"], row["model_mean"], yerr=row["model_ci"],
                    fmt="none", ecolor=c, elinewidth=1.0, capsize=2, zorder=3, alpha=0.95,
                )
                ax.scatter(
                    row["human_value"], row["model_mean"],
                    s=44, alpha=0.95, color=c, edgecolors="black", linewidth=0.45, zorder=4,
                )

            rho, p_corr = sd["rho"], sd["p_corr"]
            if not np.isnan(rho):
                stars = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else ""
                ax.text(
                    0.03, 0.97, f"$r$={rho:.2f}{stars}",
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=9.0, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.82, linewidth=0.0),
                )

            ax.plot(lims, lims, "--", color="gray", alpha=0.35, lw=1, zorder=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_facecolor("#f7f7f7")
            ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8.0)

            if row_idx == 0:
                ax.set_title(ROW_LABELS[col_title], fontsize=9.5, fontweight="bold")

    op_groups, op_color_map = color_maps["op"]
    ent_groups, ent_color_map = color_maps["ent"]

    # Family handles — one row, just above the plots
    leg = fig.legend(
        handles=family_handles(), loc="upper center",
        bbox_to_anchor=(0.40, 1.02), ncol=7,
        fontsize=8.0, frameon=True, edgecolor="none", fancybox=False,
        handletextpad=0.25, borderpad=0.32, labelspacing=0.20, columnspacing=0.6,
    )
    leg.get_frame().set_facecolor("#f4f4f4")

    # Operation legend — right side, upper half
    op_leg = fig.legend(
        handles=qgroup_handles(op_groups, op_color_map),
        loc="upper left", bbox_to_anchor=(0.65, 0.93),
        ncol=1, title="Operation", title_fontsize=8.5,
        fontsize=8.0, frameon=False,
        handletextpad=0.25, borderpad=0.35, labelspacing=0.20,
    )
    op_leg.get_title().set_fontweight("bold")
    fig.add_artist(op_leg)

    # Entity legend — right side, lower half (close to Operation)
    ent_leg = fig.legend(
        handles=qgroup_handles(ent_groups, ent_color_map),
        loc="upper left", bbox_to_anchor=(0.65, 0.46),
        ncol=1, title="Entity", title_fontsize=8.5,
        fontsize=8.0, frameon=False,
        handletextpad=0.25, borderpad=0.35, labelspacing=0.20,
    )
    ent_leg.get_title().set_fontweight("bold")

    suffix = "_abstfiltered" if abstfiltered else ""
    if metric == "sbert":
        out_name = f"7b_sbert_op_ent{suffix}.png"
    elif metric == "accuracy":
        out_name = f"7b_accuracy_op_ent{suffix}.png"
    else:
        out_name = f"7b_confidence_op_ent{suffix}.png"
    save(fig, out_name)
if __name__ == "__main__":
    # sbert scatter now generated by generate_variant_scatter.py
    plot_metric_combined("accuracy", abstfiltered=False)
    plot_metric_combined("accuracy", abstfiltered=True)
    plot_metric_combined("confidence", abstfiltered=False)
    plot_metric_combined("confidence", abstfiltered=True)
