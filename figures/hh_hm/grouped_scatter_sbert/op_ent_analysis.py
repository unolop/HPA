"""Group-level HH vs HM analysis for both operation and entity groupings.
Computes Spearman ρ, profile cosine, rmcorr. Generates scatter plots + LaTeX tables.

Outputs are saved as `op_...` and `ent_...` files in the consolidated
`grouped_scatter_sbert` folder.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist

import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "figures"))
import plot_style  # noqa: F401
EXPORTS = ROOT / "analysis/session2/exports"
OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = ROOT / "latex/AAAI2026/LaTeX/tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")

# --- HH aggregate per question x variant ---
hh = pc[pc["pair_type"] == "HH"]
hh_q = (
    hh.groupby(["question_id", "variant"])
    .agg(hh_sbert=("sbert_score", "mean"), op=("op", "first"), ent=("ent", "first"))
    .reset_index()
)

# --- HM aggregate per question x variant x model ---
hm = pc[pc["pair_type"] == "HM"]
hm_q = (
    hm.groupby(["question_id", "variant", "subject_2"])
    .agg(hm_sbert=("sbert_score", "mean"))
    .reset_index()
)

# 7B-scale models
models_7b = [
    "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
    "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    "Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B",
]
models_7b = [m for m in models_7b if m in hm_q["subject_2"].unique()]

MODEL_GROUPS = {
    "VLM": [m for m in models_7b if "(LM)" not in m and m not in
            ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B"]],
    "Backbone Decoder": [m for m in models_7b if "(LM)" in m],
    "Standalone LLM": [m for m in models_7b if m in
                       ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B"]],
}

merged = hm_q[hm_q["subject_2"].isin(models_7b)].merge(
    hh_q[["question_id", "variant", "hh_sbert", "op", "ent"]],
    on=["question_id", "variant"],
)

variant_colors = {"C": "#1f77b4", "B": "#ff7f0e", "A": "#2ca02c"}
variant_labels = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}


def run_analysis(group_col: str, group_label: str):
    """Run full analysis for a grouping dimension (op or ent)."""

    # Group-level aggregates with variant
    hh_grp = hh_q.groupby([group_col, "variant"]).agg(hh_sbert=("hh_sbert", "mean")).reset_index()
    hm_grp = (
        merged.groupby([group_col, "variant", "subject_2"])
        .agg(hm_sbert=("hm_sbert", "mean"))
        .reset_index()
    )
    grp = hm_grp.merge(hh_grp, on=[group_col, "variant"])

    # Variant-collapsed
    hh_grp_all = hh_q.groupby(group_col).agg(hh_sbert=("hh_sbert", "mean")).reset_index()
    hm_grp_all = (
        merged.groupby([group_col, "subject_2"])
        .agg(hm_sbert=("hm_sbert", "mean"))
        .reset_index()
    )
    grp_all = hm_grp_all.merge(hh_grp_all, on=group_col)

    n_groups = hh_grp_all[group_col].nunique()

    # ── Spearman ρ ──
    print(f"\n{'='*70}")
    print(f"SPEARMAN ρ ON {group_label.upper()} GROUP MEANS (variant-collapsed, n={n_groups})")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'ρ':>7} {'p':>10}")
    spearman_results = {}
    for model in models_7b:
        sub = grp_all[grp_all["subject_2"] == model].sort_values(group_col)
        if len(sub) < 3:
            continue
        rho, p = stats.spearmanr(sub["hh_sbert"], sub["hm_sbert"])
        spearman_results[model] = (rho, p)
        print(f"{model:<25} {rho:>7.3f} {p:>10.3f}")

    # ── Cosine similarity ──
    print(f"\n{'Model':<25} {'Cosine Sim':>10}")
    cosine_results = {}
    for model in models_7b:
        sub = grp_all[grp_all["subject_2"] == model].sort_values(group_col)
        shared = sub.merge(hh_grp_all, on=group_col, suffixes=("", "_hh")).sort_values(group_col)
        if len(shared) < 2:
            continue
        hh_v = shared["hh_sbert"].values
        hm_v = shared["hm_sbert"].values
        sim = 1 - cosine_dist(hh_v, hm_v)
        cosine_results[model] = sim
        print(f"{model:<25} {sim:>10.4f}")

    # ── rmcorr ──
    print(f"\n{'Model':<25} {'rmcorr':>7} {'p':>10}")
    rmcorr_results = {}
    for model in models_7b:
        sub = grp[grp["subject_2"] == model].copy()
        sub = sub.dropna(subset=["hh_sbert", "hm_sbert"])
        if sub[group_col].nunique() < 3 or sub["variant"].nunique() < 2:
            continue
        try:
            rm = pg.rm_corr(data=sub, x="hh_sbert", y="hm_sbert", subject=group_col)
            r = rm["r"].values[0]
            p = rm["pval"].values[0]
            ci = rm["CI95%"].values[0]
            dof = rm["dof"].values[0]
            rmcorr_results[model] = (r, p, ci, dof)
            print(f"{model:<25} {r:>7.3f} {p:>10.3f}")
        except Exception as e:
            print(f"{model:<25} — error: {e}")

    # ── MAD ──
    mad_results = {}
    for model in models_7b:
        sub = grp_all[grp_all["subject_2"] == model]
        if len(sub) < 2:
            continue
        mad_results[model] = np.mean(np.abs(sub["hm_sbert"].values - sub["hh_sbert"].values))

    # ══════════════════════════════════════════════════════════════
    # SCATTER PLOT
    # ══════════════════════════════════════════════════════════════
    ncols = 5
    nrows = int(np.ceil(len(models_7b) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, model in enumerate(models_7b):
        ax = axes[idx // ncols][idx % ncols]
        sub = grp[grp["subject_2"] == model]

        for v in ["C", "B", "A"]:
            vs = sub[sub["variant"] == v]
            ax.scatter(
                vs["hh_sbert"], vs["hm_sbert"],
                c=variant_colors[v], label=variant_labels[v],
                alpha=0.7, s=50, edgecolors="white", linewidth=0.5,
            )
            for _, row in vs.iterrows():
                ax.annotate(
                    row[group_col], (row["hh_sbert"], row["hm_sbert"]),
                    fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points",
                )

        rho = spearman_results.get(model, (float("nan"),))[0]
        cos_sim = cosine_results.get(model, float("nan"))
        rm_info = rmcorr_results.get(model, None)
        rm_str = f"rmcorr={rm_info[0]:.3f}" if rm_info else "rmcorr=\u2014"

        txt = f"\u03c1={rho:.3f}  cos={cos_sim:.3f}\n{rm_str}"
        ax.text(
            0.02, 0.98, txt, transform=ax.transAxes,
            fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

        ax.plot([0.15, 0.8], [0.15, 0.8], "--", color="gray", alpha=0.4, lw=1)
        ax.set_xlim(0.15, 0.8)
        ax.set_ylim(0.15, 0.8)
        ax.set_title(model, fontweight="bold")

        if idx % ncols == 0:
            ax.set_ylabel("HM SBERT (group mean)")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("HH SBERT (group mean)")

    for idx in range(len(models_7b), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    row_labels_list = ["VLM", "Backbone Decoder", "Standalone LLM"]
    for row_idx, label in enumerate(row_labels_list):
        if row_idx < nrows:
            axes[row_idx][0].annotate(
                label, xy=(-0.3, 0.5), xycoords="axes fraction",
                fontweight="bold", rotation=90,
                ha="center", va="center",
            )

    handles, labels_leg = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    fig.subplots_adjust(left=0.07)

    out = OUTDIR / f"{group_col}_7b.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")

    # ══════════════════════════════════════════════════════════════
    # LaTeX TABLE
    # ══════════════════════════════════════════════════════════════
    def fmt_p(p):
        return "$<$.001" if p < 0.001 else f"{p:.3f}"

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Group-level correlation between HH and HM SBERT agreement, "
        f"aggregated by \\textbf{{{group_label.lower()}}} type (n={n_groups} groups). "
        r"$\rho$: Spearman; $r$: Pearson; Cos: cosine similarity; "
        r"MAD: mean absolute deviation from diagonal; "
        r"rmcorr: repeated-measures correlation across variants.}"
    )
    lines.append(r"\label{tab:group_corr_" + group_col + "}")
    lines.append(r"\begin{tabular}{ll ccc cc}")
    lines.append(r"\toprule")
    lines.append(r"Group & Model & $\rho$ & Cos & MAD & rmcorr & $p$ \\")
    lines.append(r"\midrule")

    prev_mg = None
    for mg_name, mg_models in MODEL_GROUPS.items():
        if prev_mg is not None:
            lines.append(r"\cdashline{2-7}")
        prev_mg = mg_name

        for model in mg_models:
            rho = spearman_results.get(model, (float("nan"),))[0]
            cos_sim = cosine_results.get(model, float("nan"))
            mad = mad_results.get(model, float("nan"))
            rm_info = rmcorr_results.get(model, None)
            rm_r = rm_info[0] if rm_info else float("nan")
            rm_p = rm_info[1] if rm_info else float("nan")

            model_disp = model.replace("(LM)", r"{\scriptsize(LM)}")
            model_disp = model_disp.replace("(think)", r"{\scriptsize(think)}")

            mg_col = ""  # group label only on first row
            if model == mg_models[0]:
                mg_col = mg_name

            rm_str = f"{rm_r:.3f}" if not np.isnan(rm_r) else r"\textemdash"
            rm_p_str = fmt_p(rm_p) if not np.isnan(rm_p) else r"\textemdash"

            lines.append(
                f"{mg_col} & {model_disp} & "
                f"{rho:.3f} & {cos_sim:.4f} & {mad:.3f} & "
                f"{rm_str} & {rm_p_str} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    tex_path = TABLES_DIR / f"group_corr_{group_col}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved: {tex_path}")

    # Summary
    print(f"\n--- {group_label} group averages by model type ---")
    rows = []
    for model in models_7b:
        mg = next((k for k, v in MODEL_GROUPS.items() if model in v), "?")
        rows.append({
            "Model Group": mg,
            "Spearman ρ": spearman_results.get(model, (float("nan"),))[0],
            "Cosine": cosine_results.get(model, float("nan")),
            "MAD": mad_results.get(model, float("nan")),
        })
    summary = pd.DataFrame(rows)
    print(summary.groupby("Model Group")[["Spearman ρ", "Cosine", "MAD"]].mean()
          .reindex(["VLM", "Backbone Decoder", "Standalone LLM"]).round(3).to_string())

    return spearman_results, cosine_results, rmcorr_results


# ═══════════════════════════════════════════════════════════════════
# Run for both groupings
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OPERATION GROUPS")
print("=" * 70)
run_analysis("op", "Operation")

print("\n\n" + "=" * 70)
print("ENTITY GROUPS")
print("=" * 70)
run_analysis("ent", "Entity")
