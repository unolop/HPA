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
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))
import plot_style  # noqa: F401
from helpers import filter_abstained_pairs

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
                        "Vicuna-7B", "Mistral-7B"],
}
ROW_ORDER  = ["VLM", "Backbone Decoder", "Standalone LLM"]
ROW_LABELS = {"VLM": "VLM", "Backbone Decoder": "Backbone", "Standalone LLM": "SA-LLM"}

_tab20 = plt.colormaps["tab20"]
OP_PALETTE  = [_tab20(i) for i in range(0, 22, 2)][:12]
ENT_PALETTE = [_tab20(i) for i in range(1, 19, 2)][:9]

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


def sig_stars(p):
    if np.isnan(p): return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return ""


def build_palette(n, base):
    return base[:n] if n <= len(base) else [_tab20(i % 20) for i in range(n)]


def family_handles():
    handles = [
        mlines.Line2D([0], [0], marker=mk, color="#666", markerfacecolor="#666",
                      linestyle="None", markersize=5, label=lbl)
        for mk, _, _, _, lbl in MODEL_FAMILIES
    ]
    return sorted(handles, key=lambda h: h.get_label().lower())


def qgroup_handles(groups_sorted, color_map):
    return [
        mlines.Line2D([0], [0], marker="o", color=color_map[g],
                      linestyle="None", markersize=4.8, label=g)
        for g in groups_sorted
    ]


def load_data():
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)
    pc = pc[~pc["subject_2"].str.contains("32B|32b", na=False)]
    cls_map = {m: grp for grp, ms in MODEL_GROUPS.items() for m in ms}
    keep_models = set(cls_map)

    hh = pc[pc["pair_type"] == "HH"].copy()
    hm = pc[(pc["pair_type"] == "HM") & pc["subject_2"].isin(keep_models)].copy()
    hm["model_class"] = hm["subject_2"].map(cls_map)
    return hh, hm


def plot_variant_scatter():
    hh, hm = load_data()

    fig, axes = plt.subplots(
        2, 3, figsize=(8.5, 4.3), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.14, "wspace": 0.10,
                     "left": 0.07, "right": 0.66, "top": 0.78, "bottom": 0.12},
    )
    fig.supylabel("Human–Model SBERT", fontsize=8.5, x=0.02)
    fig.supxlabel("Human–Human SBERT", fontsize=8.5, y=0.03, x=0.32)

    xlims = [0.25, 0.72]
    ylims = [0.10, 0.72]

    color_maps = {}
    for dim, full_names, palette_base in [("op", OP_FULL_NAMES, OP_PALETTE),
                                           ("ent", ENT_FULL_NAMES, ENT_PALETTE)]:
        hh_grp = hh.groupby([dim, "variant"])["sbert_score"].mean().reset_index()
        all_groups = sorted(hh_grp[dim].unique())
        palette = build_palette(len(all_groups), palette_base)
        color_maps[dim] = {g: palette[i] for i, g in enumerate(all_groups)}

    dims_cfg = [
        ("op",  OP_FULL_NAMES,  OP_PALETTE),
        ("ent", ENT_FULL_NAMES, ENT_PALETTE),
    ]

    # ── Pass 1: collect all subplot data and raw Spearman ρ/p ────────────────
    subplot_data = {}  # (row_idx, col_idx) → {"rho_xs", "rho_ys", "points"}
    for row_idx, (dim, full_names, _) in enumerate(dims_cfg):
        hh_grp = hh.groupby([dim, "variant"])["sbert_score"].mean().reset_index()
        groups = sorted(hh_grp[dim].unique())
        for col_idx, cls in enumerate(ROW_ORDER):
            sub_hm = hm[hm["model_class"] == cls]
            rho_xs, rho_ys = [], []
            points = []  # (grp, v, model_name, x, y_val, alpha, x_avg, y_avg, y_ci, c)
            group_means = []  # (x_avg, y_avg, y_ci, c) per group

            for grp in groups:
                grp_hh = hh_grp[hh_grp[dim] == grp]
                grp_hm = sub_hm[sub_hm[dim] == grp]
                c = color_maps[dim][grp]

                # Per-model mean across all variants
                model_means = grp_hm.groupby("subject_2")["sbert_score"].mean()
                x_avg = grp_hh["sbert_score"].mean()
                if model_means.empty:
                    continue

                for model_name, y_val in model_means.items():
                    points.append((model_name, x_avg, y_val, 0.75, c))

                y_avg = model_means.mean()
                y_ci  = (1.96 * model_means.std(ddof=1) / np.sqrt(len(model_means))
                         if len(model_means) > 1 else 0.0)
                group_means.append((x_avg, y_avg, y_ci, c))
                rho_xs.append(x_avg)
                rho_ys.append(y_avg)

            rho, p = stats.spearmanr(rho_xs, rho_ys) if len(rho_xs) >= 4 else (np.nan, np.nan)
            subplot_data[(row_idx, col_idx)] = {
                "dim": dim, "points": points, "group_means": group_means,
                "rho": rho, "p_raw": p,
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
    for row_idx, (dim, full_names, _) in enumerate(dims_cfg):
        for col_idx, cls in enumerate(ROW_ORDER):
            ax = axes[row_idx, col_idx]
            sd = subplot_data[(row_idx, col_idx)]

            for model_name, x, y_val, alpha, c in sd["points"]:
                mkr = MODEL_STYLE.get(model_name, "o")
                ax.scatter(x, y_val, s=22, color=c, marker=mkr,
                           edgecolors="white", linewidth=0.45,
                           alpha=alpha, zorder=3)

            for x_avg, y_avg, y_ci, c in sd["group_means"]:
                ax.errorbar(x_avg, y_avg, yerr=y_ci, fmt="none",
                            ecolor=c, elinewidth=1.0, capsize=2,
                            alpha=0.95, zorder=4)
                ax.scatter(x_avg, y_avg, s=52, color=c,
                           edgecolors="black", linewidth=0.55,
                           alpha=1.0, zorder=6)

            # Spearman ρ with Holm-corrected p
            rho, p_corr = sd["rho"], sd["p_corr"]
            if not np.isnan(rho):
                stars = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else ""
                ax.text(0.04, 0.96, f"$\\rho$={rho:.2f}{stars}",
                        transform=ax.transAxes, fontsize=7.5,
                        va="top", ha="left", color="#333")

            ax.plot(xlims, ylims, "--", color="gray", alpha=0.35, lw=1, zorder=1)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_facecolor("#f7f7f7")
            ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8.0)

            if row_idx == 0:
                ax.set_title(ROW_LABELS[cls], fontsize=9.5, fontweight="bold")

    # ── Family legend (top, below variant) ───────────────────────────────────
    fig.legend(
        handles=family_handles(), loc="lower center",
        bbox_to_anchor=(0.35, 0.86), ncol=7,
        fontsize=8.0, frameon=False,
        handletextpad=0.25, borderpad=0.32, labelspacing=0.20, columnspacing=0.6,
    )

    # ── Operation color legend (right, upper) ─────────────────────────────────
    op_groups = sorted(color_maps["op"].keys())
    op_leg = fig.legend(
        handles=qgroup_handles(
            [OP_FULL_NAMES.get(g, g) for g in op_groups],
            {OP_FULL_NAMES.get(g, g): color_maps["op"][g] for g in op_groups},
        ),
        loc="upper left", bbox_to_anchor=(0.65, 0.82),
        ncol=1, title="Operation", title_fontsize=8.5,
        fontsize=8.0, frameon=False,
        handletextpad=0.25, borderpad=0.35, labelspacing=0.20,
    )
    op_leg.get_title().set_fontweight("bold")
    fig.add_artist(op_leg)

    # ── Entity color legend (right, lower) ───────────────────────────────────
    ent_groups = sorted(color_maps["ent"].keys())
    ent_leg = fig.legend(
        handles=qgroup_handles(
            [ENT_FULL_NAMES.get(g, g) for g in ent_groups],
            {ENT_FULL_NAMES.get(g, g): color_maps["ent"][g] for g in ent_groups},
        ),
        loc="upper left", bbox_to_anchor=(0.65, 0.42),
        ncol=1, title="Entity", title_fontsize=8.5,
        fontsize=8.0, frameon=False,
        handletextpad=0.25, borderpad=0.35, labelspacing=0.20,
    )
    ent_leg.get_title().set_fontweight("bold")

    out = OUTDIR / "7b_sbert_variant_op_ent_abstfiltered.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_DIR / "7b_sbert_variant_op_ent_abstfiltered.png")
    plt.close()
    print(f"Saved: {out}")


plot_variant_scatter()
