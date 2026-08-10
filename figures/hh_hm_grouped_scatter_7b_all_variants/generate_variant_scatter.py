"""
Variant-separated SBERT scatter: same visual style as generate_aux.py but each
group appears 3× (Original / Weaker / Pronominalized) so trajectory C→B→A is
visible within each group cluster.

Encoding (same as main scatter):
  color  = operation / entity group  (right-side legend, two blocks)
  marker = model family              (top legend bar)
  alpha  = variant (C=0.95, B=0.65, A=0.35)

Additional elements not in main scatter:
  trajectory line  = connects group-mean points across C→B→A
  r / p            = Pearson r over group×variant means (n=33 op, n=27 ent)

Saved as: 7b_sbert_variant_op_ent_abstfiltered.png
"""
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))
import plot_style  # noqa: F401
from helpers import filter_abstained_pairs
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.utils.vqa import VQAAnswerMapper

OUTDIR    = Path(__file__).resolve().parent
LATEX_DIR = ROOT / "latex/AAAI2026/LaTeX/figures/hh_hm_grouped_scatter_7b_all_variants"
LATEX_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS   = ROOT / "analysis/session2/exports"

# ── model style (identical to generate_aux) ───────────────────────────────────
MODEL_FAMILIES = [
    ("o", "InternVL-8B",       "InternVL-8B (LM)",    None,                 "InternVL"),
    ("X", "Qwen3-VL-8B",       "Qwen3-VL-8B (LM)",    "Qwen3-8B",           "Qwen3"),
    ("D", "LLaVA-1.5-7B",      "LLaVA-1.5 (LM)",      None,                 "LLaVA-1.5"),
    ("^", "LLaVA-Mistral",      "LLaVA-Mistral (LM)",  "Mistral-7B",         "Mistral"),
    ("v", "LLaVA-Vicuna",       "LLaVA-Vicuna (LM)",   "Vicuna-7B",          "Vicuna"),
    ("P", None,                 None,                   "Qwen3-8B (think)",   "Qwen3 (think)"),
    ("s", None,                 None,                   "Qwen2.5-7B-Instruct","Qwen2.5-Instruct"),
]
MODEL_STYLE = {}
for marker, vlm, dec, llm, _ in MODEL_FAMILIES:
    if vlm: MODEL_STYLE[vlm] = marker
    if dec: MODEL_STYLE[dec] = marker
    if llm: MODEL_STYLE[llm] = marker

MODEL_GROUPS = {
    "VLM":             ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone Decoder":["InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)",
                        "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"],
    "Standalone LLM":  ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct",
                        "Vicuna-7B"],
}
ROW_ORDER  = ["VLM", "Backbone Decoder", "Standalone LLM"]
ROW_LABELS = {"VLM": "VLM", "Backbone Decoder": "Backbone", "Standalone LLM": "SA-LLM"}

_tab10  = plt.colormaps["tab10"]
_tab20  = plt.colormaps["tab20"]
_CB10   = [_tab10(i) for i in range(10)]
_PALETTE_OP  = _CB10 + [_tab20(18)]   # tab10, matches generate_aux.py
_PALETTE_ENT = [                       # Okabe-Ito, distinct from tab10
    '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
    '#D55E00', '#CC79A7', '#999999', '#44BB99',
]

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

VARIANT_ALPHA = {"C": 0.95, "B": 0.65, "A": 0.35}
VARIANT_LABEL = {"C": "Orig.", "B": "Weak.", "A": "Pron."}



def family_handles():
    handles = [
        mlines.Line2D([0], [0], marker=mk, color="#666", markerfacecolor="#666",
                      linestyle="None", markersize=5, label=lbl)
        for mk, _, _, _, lbl in MODEL_FAMILIES
    ]
    return sorted(handles, key=lambda h: h.get_label().lower())


class _TitleHandler(HandlerBase):
    """Renders no marker so the title label aligns with entry labels."""
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        handlebox.set_width(0)
        handlebox.set_height(0)
        import matplotlib.patches as mpatches
        return mpatches.Rectangle((0, 0), 0, 0, visible=False,
                                   transform=handlebox.get_transform())


def qgroup_handles(title_str, groups_sorted, color_map):
    title_h = mlines.Line2D([], [], color="none", marker="none",
                             linewidth=0, label=title_str)
    entries = [
        mlines.Line2D([0], [0], marker="o", color=color_map[g],
                      linestyle="None", markersize=4.8, label=g)
        for g in groups_sorted
    ]
    return [title_h] + entries


_TITLE_HANDLER_MAP = lambda title_h: {title_h: _TitleHandler()}


def _is_yesno(df):
    yn = {"yes", "no"}
    return df["answer_1"].str.lower().isin(yn) & df["answer_2"].str.lower().isin(yn)


def load_data(ft_only: bool = True):
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)
    pc = pc[~pc["subject_2"].str.contains("32B|32b|Mistral-7B", na=False)]
    cls_map = {m: grp for grp, ms in MODEL_GROUPS.items() for m in ms}
    keep_models = set(cls_map)

    hh = pc[pc["pair_type"] == "HH"].copy()
    hm = pc[(pc["pair_type"] == "HM") & pc["subject_2"].isin(keep_models)].copy()

    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other")
                 for qid, ann in mapper.annotations.items()}
    if ft_only:
        keep_qids = {qid for qid, atype in qid2atype.items() if atype == "other"}
        hh = hh[hh["question_id"].astype(int).isin(keep_qids)].copy()
        hm = hm[hm["question_id"].astype(int).isin(keep_qids)].copy()

    hm["model_class"] = hm["subject_2"].map(cls_map)
    return hh, hm


def plot_variant_scatter(color_by: str = "mean", ft_only: bool = True):
    """
    color_by: "mean"  → gradient by mean HH SBERT (high=bright)
              "delta" → gradient by ΔHH SBERT (orig C − pron A; high delta=bright)
    ft_only:  True  → free-text questions only (answer_type == "other")
              False → all question types
    """
    hh, hm = load_data(ft_only=ft_only)

    fig, axes = plt.subplots(
        2, 3, figsize=(8.5, 4.3), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.14, "wspace": 0.10,
                     "left": 0.07, "right": 0.66, "top": 0.78, "bottom": 0.12},
    )
    fig.supylabel("Human–Model SBERT", fontsize=8.5, x=0.02)
    fig.supxlabel("Human–Human SBERT", fontsize=8.5, y=0.03, x=0.32)

    xlims = [0.30, 0.60]
    ylims = [0.10, 0.66]

    color_maps = {}
    hh_group_order = {}
    hh_sbert_order = {}  # always sorted by mean sHH desc (for legends)
    for dim in ("op", "ent"):
        palette = _PALETTE_OP if dim == "op" else _PALETTE_ENT
        hh_mean = hh.groupby(dim)["sbert_score"].mean().sort_values(ascending=False)
        if color_by == "delta":
            # ΔHH = mean HH SBERT (variant C) − mean HH SBERT (variant A)
            hh_by_var = hh.groupby([dim, "variant"])["sbert_score"].mean().unstack("variant")
            delta = (hh_by_var.get("C", 0) - hh_by_var.get("A", 0)).dropna()
            sort_vals = delta.sort_values(ascending=False)  # high Δ → bright
        else:
            sort_vals = hh_mean
        groups_sorted = sort_vals.index.tolist()
        n = len(groups_sorted)
        colors = [palette[i % len(palette)] for i in range(n)]
        color_maps[dim] = {g: colors[i] for i, g in enumerate(groups_sorted)}
        hh_group_order[dim] = groups_sorted
        hh_sbert_order[dim] = hh_mean.index.tolist()

    dims_cfg = [
        ("op",  OP_FULL_NAMES),
        ("ent", ENT_FULL_NAMES),
    ]

    # ── Pre-compute human LOOCV range per (dim, group) ───────────────────────
    # For each participant, mean SBERT against all others within that group
    human_loocv = {}  # (dim, grp) → (p5, mean, p95)
    for dim, _ in dims_cfg:
        for grp, grp_df in hh.groupby(dim):
            per_part = [pdata["sbert_score"].mean()
                        for _, pdata in grp_df.groupby("subject_1")
                        if len(pdata) >= 2]
            if len(per_part) >= 5:
                arr = np.array(per_part)
                human_loocv[(dim, grp)] = (
                    float(np.percentile(arr, 5)),
                    float(arr.mean()),
                    float(np.percentile(arr, 95)),
                )

    # ── Pass 1: collect all subplot data ─────────────────────────────────────
    subplot_data = {}  # (row_idx, col_idx) → {"points", "group_means", "human_refs", "r_xs", "r_ys"}
    for row_idx, (dim, full_names) in enumerate(dims_cfg):
        hh_grp = hh.groupby([dim, "variant"])["sbert_score"].mean().reset_index()
        groups = hh_group_order[dim]  # ordered low→high HH SBERT
        for col_idx, cls in enumerate(ROW_ORDER):
            sub_hm = hm[hm["model_class"] == cls]
            points = []       # (model_name, x, y_val, c)
            group_means = []  # (x_avg, mean_y, ci_y, c)
            human_refs = []   # (x_avg, p5, p95)
            r_xs = []
            r_ys = []

            for grp in groups:
                grp_hh = hh_grp[hh_grp[dim] == grp]
                grp_hm = sub_hm[sub_hm[dim] == grp]
                c = color_maps[dim][grp]

                model_means = grp_hm.groupby("subject_2")["sbert_score"].mean()
                x_avg = grp_hh["sbert_score"].mean()
                if model_means.empty:
                    continue

                for model_name, y_val in model_means.items():
                    points.append((model_name, x_avg, y_val, c))

                y_vals = model_means.values
                mean_y = float(np.mean(y_vals))
                ci_y = (1.96 * float(np.std(y_vals, ddof=1)) / np.sqrt(len(y_vals))
                        if len(y_vals) > 1 else 0.0)
                group_means.append((x_avg, mean_y, ci_y, c))

                r_xs.append(x_avg)
                r_ys.append(mean_y)

                if (dim, grp) in human_loocv:
                    p5, _, p95 = human_loocv[(dim, grp)]
                    human_refs.append((x_avg, p5, p95))

            subplot_data[(row_idx, col_idx)] = {
                "dim": dim, "points": points, "group_means": group_means,
                "human_refs": human_refs, "r_xs": r_xs, "r_ys": r_ys,
            }

    # ── Pass 2: draw ─────────────────────────────────────────────────────────
    for row_idx, (dim, full_names) in enumerate(dims_cfg):
        for col_idx, cls in enumerate(ROW_ORDER):
            ax = axes[row_idx, col_idx]
            sd = subplot_data[(row_idx, col_idx)]

            # Individual model markers (small)
            for model_name, x, y_val, c in sd["points"]:
                mkr = MODEL_STYLE.get(model_name, "o")
                ax.scatter(x, y_val, s=22, color=c, marker=mkr,
                           edgecolors="white", linewidth=0.45,
                           alpha=0.85, zorder=3)

            # Group mean + 95% CI errorbar (large circle)
            for x_avg, mean_y, ci_y, c in sd["group_means"]:
                ax.errorbar(x_avg, mean_y, yerr=ci_y,
                            fmt="none", ecolor=c, elinewidth=1.0, capsize=2,
                            zorder=4, alpha=0.95)
                ax.scatter(x_avg, mean_y, s=44, alpha=0.95, color=c, marker="o",
                           edgecolors="black", linewidth=0.45, zorder=5)

            # Human LOOCV range: per-group vertical shaded band (p5–p95)
            band_hw = 0.012
            for x_avg, p5, p95 in sd["human_refs"]:
                ax.fill_between([x_avg - band_hw, x_avg + band_hw],
                                p5, p95,
                                alpha=0.18, color="#424242",
                                linewidth=0, zorder=0)

            ax.plot(xlims, xlims, "--", color="gray", alpha=0.35, lw=1, zorder=1)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_facecolor("#f7f7f7")
            ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8.0)

            if row_idx == 0:
                ax.set_title(ROW_LABELS[cls], fontsize=9.5, fontweight="bold")

    # ── Family legend (top, above plot grid) ─────────────────────────────────
    leg = fig.legend(
        handles=family_handles(), loc="lower center",
        bbox_to_anchor=(0.40, 0.85), ncol=7,
        fontsize=8.0, frameon=True, edgecolor="none", fancybox=False,
        handletextpad=0.25, borderpad=0.32, labelspacing=0.20, columnspacing=0.6,
    )
    leg.get_frame().set_facecolor("#f4f4f4")

    # ── Operation color legend (right, upper) ────────────────────────────────
    op_groups = hh_sbert_order["op"]
    _op_title  = r"$\mathbf{Operation}$"
    op_handles = qgroup_handles(
        _op_title,
        [OP_FULL_NAMES.get(g, g) for g in op_groups],
        {OP_FULL_NAMES.get(g, g): color_maps["op"][g] for g in op_groups},
    )
    op_leg = fig.legend(
        handles=op_handles,
        handler_map=_TITLE_HANDLER_MAP(op_handles[0]),
        loc="upper left", bbox_to_anchor=(0.66, 0.82),
        ncol=1, fontsize=8.0, frameon=False,
        handletextpad=0.25, borderpad=0.20, labelspacing=0.20,
    )
    fig.add_artist(op_leg)

    # ── Entity color legend (right, lower) ───────────────────────────────────
    ent_groups = hh_sbert_order["ent"]
    _ent_title  = r"$\mathbf{Entity}$"
    ent_handles = qgroup_handles(
        _ent_title,
        [ENT_FULL_NAMES.get(g, g) for g in ent_groups],
        {ENT_FULL_NAMES.get(g, g): color_maps["ent"][g] for g in ent_groups},
    )
    ent_leg = fig.legend(
        handles=ent_handles,
        handler_map=_TITLE_HANDLER_MAP(ent_handles[0]),
        loc="upper left", bbox_to_anchor=(0.66, 0.46),
        ncol=1, fontsize=8.0, frameon=False,
        handletextpad=0.25, borderpad=0.20, labelspacing=0.20,
    )

    q_tag    = "" if ft_only else "_allq"
    col_tag  = f"_{color_by}" if color_by != "mean" else ""
    out = OUTDIR / f"7b_sbert_op_ent_abstfiltered{q_tag}{col_tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_DIR / out.name)
    plt.close()
    print(f"Saved: {out}")


def plot_variant_scatter_vertical(color_by: str = "mean", ft_only: bool = True):
    """
    Vertical layout: 3 rows (VLM / Backbone / SA-LLM) × 2 cols (Operation / Entity).
    Model family legend at top; operation and entity color legends on the right.
    """
    hh, hm = load_data(ft_only=ft_only)

    # Plot area: 2 cols + right legend panel
    fig, axes = plt.subplots(
        3, 2, figsize=(7.5, 6.5), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.12, "wspace": 0.10,
                     "left": 0.10, "right": 0.62, "top": 0.84, "bottom": 0.09},
    )
    fig.supylabel("Human–Model SBERT", fontsize=8.5, x=0.02)
    fig.supxlabel("Human–Human SBERT", fontsize=8.5, y=0.02, x=0.36)

    xlims = [0.28, 0.60]
    ylims = [0.10, 0.72]

    color_maps = {}
    hh_group_order = {}
    for dim in ("op", "ent"):
        palette = _PALETTE_OP if dim == "op" else _PALETTE_ENT
        if color_by == "delta":
            hh_by_var = hh.groupby([dim, "variant"])["sbert_score"].mean().unstack("variant")
            delta = (hh_by_var.get("C", 0) - hh_by_var.get("A", 0)).dropna()
            sort_vals = delta.sort_values(ascending=False)
        else:
            sort_vals = hh.groupby(dim)["sbert_score"].mean().sort_values(ascending=False)
        groups_sorted = sort_vals.index.tolist()
        n = len(groups_sorted)
        colors = [palette[i % len(palette)] for i in range(n)]
        color_maps[dim] = {g: colors[i] for i, g in enumerate(groups_sorted)}
        hh_group_order[dim] = groups_sorted

    dims_cfg = [("op", OP_FULL_NAMES), ("ent", ENT_FULL_NAMES)]

    # ── Pre-compute human LOOCV range per (dim, group) ───────────────────────
    human_loocv = {}
    for dim, _ in dims_cfg:
        for grp, grp_df in hh.groupby(dim):
            per_part = [pdata["sbert_score"].mean()
                        for _, pdata in grp_df.groupby("subject_1")
                        if len(pdata) >= 2]
            if len(per_part) >= 5:
                arr = np.array(per_part)
                human_loocv[(dim, grp)] = (
                    float(np.percentile(arr, 5)),
                    float(arr.mean()),
                    float(np.percentile(arr, 95)),
                )

    # ── Pass 1: collect subplot data (row=cls, col=dim) ──────────────────────
    subplot_data = {}
    for row_idx, cls in enumerate(ROW_ORDER):
        sub_hm = hm[hm["model_class"] == cls]
        for col_idx, (dim, _) in enumerate(dims_cfg):
            hh_grp = hh.groupby([dim, "variant"])["sbert_score"].mean().reset_index()
            groups = hh_group_order[dim]
            points = []
            human_refs = []
            for grp in groups:
                grp_hh = hh_grp[hh_grp[dim] == grp]
                grp_hm = sub_hm[sub_hm[dim] == grp]
                c = color_maps[dim][grp]
                model_means = grp_hm.groupby("subject_2")["sbert_score"].mean()
                x_avg = grp_hh["sbert_score"].mean()
                if model_means.empty:
                    continue
                for model_name, y_val in model_means.items():
                    points.append((model_name, x_avg, y_val, 0.75, c))
                if (dim, grp) in human_loocv:
                    p5, _, p95 = human_loocv[(dim, grp)]
                    human_refs.append((x_avg, p5, p95))
            subplot_data[(row_idx, col_idx)] = {"dim": dim, "points": points, "human_refs": human_refs}

    # ── Pass 2: draw ─────────────────────────────────────────────────────────
    DIM_TITLES = {"op": "Operation", "ent": "Entity"}
    band_hw = 0.012
    for row_idx, cls in enumerate(ROW_ORDER):
        for col_idx, (dim, _) in enumerate(dims_cfg):
            ax = axes[row_idx, col_idx]
            sd = subplot_data[(row_idx, col_idx)]

            for model_name, x, y_val, alpha, c in sd["points"]:
                mkr = MODEL_STYLE.get(model_name, "o")
                ax.scatter(x, y_val, s=52, color=c, marker=mkr,
                           edgecolors="#333333", linewidth=0.4,
                           alpha=alpha, zorder=3)

            for x_avg, p5, p95 in sd["human_refs"]:
                ax.fill_between([x_avg - band_hw, x_avg + band_hw],
                                p5, p95, alpha=0.18, color="#424242",
                                linewidth=0, zorder=0)

            ax.plot(xlims, xlims, "--", color="gray", alpha=0.35, lw=1, zorder=1)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_facecolor("#f7f7f7")
            ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8.0)

            if row_idx == 0:
                ax.set_title(DIM_TITLES[dim], fontsize=9.5, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(ROW_LABELS[cls], fontsize=9.0, fontweight="bold", labelpad=4)

    # ── Family legend (top, above plot grid) ─────────────────────────────────
    fig.legend(
        handles=family_handles(), loc="lower center",
        bbox_to_anchor=(0.36, 0.86), ncol=4,
        fontsize=8.0, frameon=False,
        handletextpad=0.25, borderpad=0.32, labelspacing=0.20, columnspacing=0.6,
    )

    # ── Operation color legend (right, upper) ────────────────────────────────
    op_groups = hh_group_order["op"]
    _op_title  = r"$\mathbf{Operation}$"
    op_handles = qgroup_handles(
        _op_title,
        [OP_FULL_NAMES.get(g, g) for g in op_groups],
        {OP_FULL_NAMES.get(g, g): color_maps["op"][g] for g in op_groups},
    )
    op_leg = fig.legend(
        handles=op_handles,
        handler_map=_TITLE_HANDLER_MAP(op_handles[0]),
        loc="upper left", bbox_to_anchor=(0.63, 0.84),
        ncol=1, fontsize=8.0, frameon=False,
        handletextpad=0.25, borderpad=0.20, labelspacing=0.20,
    )
    fig.add_artist(op_leg)

    # ── Entity color legend (right, lower) ───────────────────────────────────
    ent_groups = hh_group_order["ent"]
    _ent_title  = r"$\mathbf{Entity}$"
    ent_handles = qgroup_handles(
        _ent_title,
        [ENT_FULL_NAMES.get(g, g) for g in ent_groups],
        {ENT_FULL_NAMES.get(g, g): color_maps["ent"][g] for g in ent_groups},
    )
    fig.legend(
        handles=ent_handles,
        handler_map=_TITLE_HANDLER_MAP(ent_handles[0]),
        loc="upper left", bbox_to_anchor=(0.63, 0.40),
        ncol=1, fontsize=8.0, frameon=False,
        handletextpad=0.25, borderpad=0.20, labelspacing=0.20,
    )

    q_tag   = "" if ft_only else "_allq"
    col_tag = f"_{color_by}" if color_by != "mean" else ""
    out = OUTDIR / f"7b_sbert_op_ent_abstfiltered{q_tag}{col_tag}_vert.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_DIR / out.name)
    plt.close()
    print(f"Saved: {out}")


plot_variant_scatter("mean",  ft_only=True)
plot_variant_scatter("delta", ft_only=True)
plot_variant_scatter("mean",  ft_only=False)
plot_variant_scatter("delta", ft_only=False)
