"""
Forest-plot style dot chart: per-model HH–HM SBERT Pearson r,
split by variant (C/B/A), grouped by architecture, with a leave-one-out
human reference row.

Outputs:
  - figures/hh_hm_corr/hh_hm_corr_by_variant.png
  - latex/AAAI2026/LaTeX/figures/hh_hm_corr_by_variant.png
  - latex/AAAI2026/LaTeX/tables/hh_hm_scatter_correlations_table.tex
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from utils.constants import VARIANT_COLORS  # noqa: E402
sys.path.insert(0, str(ROOT / "figures"))
from helpers import filter_abstained_pairs  # noqa: E402

EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_FIG_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures"
LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
LATEX_TABLE = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables" / "hh_hm_scatter_correlations_table.tex"
LATEX_TABLE_ABST = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables" / "hh_hm_scatter_correlations_table_abstfiltered.tex"

VARIANTS = ["C", "B", "A"]
VARIANT_LABEL = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
JITTER = {"C": -0.18, "B": 0.0, "A": 0.18}
MARKER = {"C": "o", "B": "s", "A": "^"}

MODEL_GROUPS = {
    "VLM": [
        "InternVL-8B", "Qwen3-VL-8B", "LLaVA-Mistral", "LLaVA-Vicuna", "LLaVA-1.5-7B",
    ],
    "Backbone": [
        "LLaVA-Mistral (LM)", "InternVL-8B (LM)", "LLaVA-Vicuna (LM)", "LLaVA-1.5 (LM)", "Qwen3-VL-8B (LM)",
    ],
    "Standalone LLM": [
        "Qwen2.5-7B-Instruct", "Vicuna-7B", "Qwen3-8B",
    ],
}
GROUP_ORDER = ["Human", "VLM", "Backbone", "Standalone LLM"]
GROUP_LABELS = {
    "Human": "Human (LOO)",
    "VLM": "Full VLMs",
    "Backbone": "Backbone Decoders",
    "Standalone LLM": "Standalone LLMs",
}
GROUP_COLORS = {
    "Human": "#5A5A5A",
    "VLM": "#9E4A3A",
    "Backbone": "#C0843D",
    "Standalone LLM": "#5E8C61",
}


def fisher_pool(rs: list[float]) -> float:
    vals = np.array([r for r in rs if np.isfinite(r) and abs(r) < 1], dtype=float)
    if len(vals) == 0:
        return np.nan
    vals = np.clip(vals, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(vals))))


def pearson_summary(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    n = len(x)
    if n < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x, y)
    return float(r), float(p), int(n)


def load_aggregates(filtered: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    if filtered:
        pc = filter_abstained_pairs(pc)
    hh = pc[pc["pair_type"] == "HH"].copy()
    hm = pc[pc["pair_type"] == "HM"].copy()

    hh_agg = (
        hh.groupby(["question_id", "variant"], as_index=False)
        .agg(hh_sbert=("sbert_score", "mean"))
    )
    hm_agg = (
        hm.groupby(["question_id", "variant", "subject_2"], as_index=False)
        .agg(hm_sbert=("sbert_score", "mean"))
    )
    return hh_agg, hm_agg, hh


def compute_model_rows(hh_agg: pd.DataFrame, hm_agg: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for group, models in MODEL_GROUPS.items():
        for model in models:
            sub = hm_agg[hm_agg["subject_2"] == model].merge(
                hh_agg, on=["question_id", "variant"], how="inner"
            )
            r_all, p_all, n_all = pearson_summary(sub["hh_sbert"], sub["hm_sbert"])
            row = {
                "name": model,
                "group": group,
                "all_r": r_all,
                "p": p_all,
                "n": n_all,
            }
            for v in VARIANTS:
                vs = sub[sub["variant"] == v]
                row[v], _, _ = pearson_summary(vs["hh_sbert"], vs["hm_sbert"])
            rows.append(row)
    return rows


def compute_human_loo_row(hh_pairs: pd.DataFrame) -> dict:
    participants = sorted(set(hh_pairs["subject_1"]).union(hh_pairs["subject_2"]))
    per_participant = []
    for pid in participants:
        lhs = hh_pairs[(hh_pairs["subject_1"] == pid) | (hh_pairs["subject_2"] == pid)]
        rhs = hh_pairs[(hh_pairs["subject_1"] != pid) & (hh_pairs["subject_2"] != pid)]
        if lhs.empty or rhs.empty:
            continue

        lhs_q = lhs.groupby(["question_id", "variant"], as_index=False).agg(loo_hm=("sbert_score", "mean"))
        rhs_q = rhs.groupby(["question_id", "variant"], as_index=False).agg(loo_hh=("sbert_score", "mean"))
        merged = lhs_q.merge(rhs_q, on=["question_id", "variant"], how="inner")

        r_all, _, n_all = pearson_summary(merged["loo_hh"], merged["loo_hm"])
        row = {"participant": pid, "all_r": r_all, "n": n_all}
        for v in VARIANTS:
            vs = merged[merged["variant"] == v]
            row[v], _, _ = pearson_summary(vs["loo_hh"], vs["loo_hm"])
        per_participant.append(row)

    if not per_participant:
        raise RuntimeError("No participant leave-one-out rows computed.")

    df = pd.DataFrame(per_participant)
    return {
        "name": "Human (LOO)",
        "group": "Human",
        "all_r": fisher_pool(df["all_r"].tolist()),
        "p": np.nan,
        "n": int(df["n"].median()),
        "C": fisher_pool(df["C"].tolist()),
        "B": fisher_pool(df["B"].tolist()),
        "A": fisher_pool(df["A"].tolist()),
        "participant_n": int(len(df)),
        "participant_rows": df,
    }


def write_table(rows: list[dict], *, filtered: bool = False) -> None:
    def fmt_r(x: float) -> str:
        return "--" if not np.isfinite(x) else f"{x:.3f}"

    def fmt_p(x: float) -> str:
        return "--" if not np.isfinite(x) else f"{x:.1e}"

    caption_suffix = (
        " Abstention-filtered version: rows where either answer was classified as a hard or soft abstention are removed before aggregation."
        if filtered else
        ""
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Per-model correlation summary for Figure~\ref{{fig:scatter_sbert}}. Pearson correlation is computed between per-question HH SBERT and HM SBERT across the matched 7/8B models. The human row is a leave-one-out pooled reference: for each participant, per-question agreement to the rest of the humans is correlated with the corresponding HH-rest agreement, then pooled with Fisher $z$-averaging across participants. Variant columns report the same correlation within each control formulation. The misprocessed think variant is excluded.{caption_suffix}}}",
        r"\label{tab:hh_hm_scatter_corr}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & All $r$ & $p$ & Original & Weaker & Pronominalized & $N$ \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Human Reference}} \\",
        r"\midrule",
    ]
    human = next(r for r in rows if r["group"] == "Human")
    lines.append(
        f'{human["name"]} & {fmt_r(human["all_r"])} & {fmt_p(human["p"])} & {fmt_r(human["C"])} & {fmt_r(human["B"])} & {fmt_r(human["A"])} & {human["n"]} \\\\'
    )

    sections = [
        ("VLM", r"\multicolumn{7}{l}{\textit{Vision-Language Models}} \\"),
        ("Backbone", r"\multicolumn{7}{l}{\textit{VLM Backbone Decoders}} \\"),
        ("Standalone LLM", r"\multicolumn{7}{l}{\textit{Standalone LLMs}} \\"),
    ]
    for group, header in sections:
        lines.extend([r"\midrule", header, r"\midrule"])
        for row in [r for r in rows if r["group"] == group]:
            lines.append(
                f'{row["name"]} & {fmt_r(row["all_r"])} & {fmt_p(row["p"])} & {fmt_r(row["C"])} & {fmt_r(row["B"])} & {fmt_r(row["A"])} & {row["n"]} \\\\'
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    table_path = LATEX_TABLE_ABST if filtered else LATEX_TABLE
    table_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {table_path}")


def build_plot(rows: list[dict], *, filtered: bool = False) -> None:
    data = [r for grp in GROUP_ORDER for r in rows if r["group"] == grp]

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(5.2, 4.8))

    y_positions = {}
    y = 0.0
    group_label_y = {}
    for g in GROUP_ORDER:
        rows_in_group = [r["name"] for r in data if r["group"] == g]
        group_label_y[g] = y + (len(rows_in_group) - 1) / 2 if rows_in_group else y
        for name in rows_in_group:
            y_positions[name] = y
            y += 1
        y += 0.8
    n_rows = y - 0.8

    y_cur = 0.0
    for g in GROUP_ORDER:
        rows_in_group = [r["name"] for r in data if r["group"] == g]
        if not rows_in_group:
            continue
        n = len(rows_in_group)
        ax.axhspan(y_cur - 0.5, y_cur + n - 0.5, color=GROUP_COLORS[g], alpha=0.06, zorder=0)
        y_cur += n + 0.8

    for row in data:
        ybase = y_positions[row["name"]]
        vals = {"C": row["C"], "B": row["B"], "A": row["A"]}
        finite = [v for v in vals.values() if np.isfinite(v)]
        if finite:
            ax.plot([min(finite), max(finite)], [ybase, ybase], color="#aaa", lw=0.8, zorder=2, alpha=0.6)
        for vk in VARIANTS:
            r = row[vk]
            if not np.isfinite(r):
                continue
            ax.scatter(
                r, ybase + JITTER[vk],
                marker=MARKER[vk], color=VARIANT_COLORS[vk], s=28, zorder=4, alpha=0.92,
                edgecolors="white", linewidths=0.3,
            )

    ax.set_yticks([y_positions[r["name"]] for r in data])
    ax.set_yticklabels([r["name"] for r in data], fontsize=8)
    ax.invert_yaxis()

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([group_label_y[g] for g in GROUP_ORDER])
    ax2.set_yticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], fontsize=8, fontweight="bold")
    for tick, g in zip(ax2.get_yticklabels(), GROUP_ORDER):
        tick.set_color(GROUP_COLORS[g])
    ax2.tick_params(axis="y", length=0, pad=4)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)

    ax.axvline(1, color="#bbb", lw=0.5, ls=":", zorder=1)
    ax.set_xlabel("Pearson r  (HH SBERT vs. HM SBERT)", fontsize=8.5)
    ax.set_xlim(0.50, 0.97)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.grid(axis="x", lw=0.4, alpha=0.3, zorder=0)
    ax.set_ylim(n_rows + 0.3, -0.7)

    handles = [
        mlines.Line2D([], [], marker=MARKER[vk], color=VARIANT_COLORS[vk],
                      markersize=6, linestyle="None", label=VARIANT_LABEL[vk])
        for vk in VARIANTS
    ]
    ax.legend(handles=handles, ncol=3, loc="lower right",
              frameon=True, fontsize=7.5, handletextpad=0.3, columnspacing=0.6)

    fig.tight_layout(pad=0.6)
    suffix = "_abstfiltered" if filtered else ""
    out = OUT_DIR / f"hh_hm_corr_by_variant{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    latex_out = LATEX_FIG_DIR / f"hh_hm_corr_by_variant{suffix}.png"
    fig.savefig(latex_out, dpi=300, bbox_inches="tight")
    print(f"Saved: {latex_out}")
    plt.close(fig)


def run_one(filtered: bool = False) -> None:
    hh_agg, hm_agg, hh_pairs = load_aggregates(filtered=filtered)
    human_row = compute_human_loo_row(hh_pairs)
    model_rows = compute_model_rows(hh_agg, hm_agg)
    rows = [human_row] + model_rows
    write_table(rows, filtered=filtered)
    build_plot(rows, filtered=filtered)
    tag = "abstention-filtered" if filtered else "unfiltered"
    print(
        f"Human LOO pooled r ({tag}):",
        f"all={human_row['all_r']:.3f}, C={human_row['C']:.3f}, B={human_row['B']:.3f}, A={human_row['A']:.3f}",
        f"(participants={human_row['participant_n']})",
    )


def main() -> None:
    run_one(filtered=False)
    run_one(filtered=True)


if __name__ == "__main__":
    main()
