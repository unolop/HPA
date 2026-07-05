"""
Export grouped-pattern correlation diagnostics and identity-focused examples.

Outputs
-------
- figures/question_diagnostic/group_pattern_correlation_matched_7b.png
- figures/question_diagnostic/table_group_pattern_correlation.tex
- figures/question_diagnostic/table_group_pattern_winners_by_op.tex
- figures/question_diagnostic/table_group_pattern_winners_by_entity.tex
- figures/question_diagnostic/table_identity_focus_examples.tex

Run from repo root:
  conda run -n zero python figures/question_diagnostic/generate_pattern_correlation_and_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from config import MODELS_7B, MODEL_GROUP
from helpers import load_human_subset
from utils.constants import GROUP_COLORS
from figures.entity_op_grouped_variants import (
    _group_maps,
    _summaries,
    _load,
)

OUT_DIR = ROOT / "figures" / "question_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ANS = 348
EXCLUDED_MATCHED_MODELS = {"Mistral-7B", "Phi-3.5-mini", "Qwen3-8B (think)"}


def short_label(model: str) -> str:
    label = str(model)
    label = label.replace("Qwen3-VL-8B (LM)", "Qwen3-VL-8B bb")
    label = label.replace("InternVL-8B (LM)", "InternVL-8B bb")
    label = label.replace("LLaVA-1.5 (LM)", "LLaVA-1.5 bb")
    label = label.replace("LLaVA-Mistral (LM)", "LLaVA-M bb")
    label = label.replace("LLaVA-Vicuna (LM)", "LLaVA-V bb")
    label = label.replace("LLaVA-Mistral", "LLaVA-M")
    label = label.replace("LLaVA-Vicuna", "LLaVA-V")
    label = label.replace("LLaVA-1.5-7B", "LLaVA-1.5")
    return label


def export_latex(df: pd.DataFrame, filename: str, caption: str, label: str) -> None:
    path = OUT_DIR / filename
    latex = df.to_latex(
        escape=False,
        index=False,
        float_format=lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x),
    )
    with open(path, "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\small\n")
        f.write(latex)
        f.write("\\end{table*}\n")
    print(f"Exported: {path}")


def build_category_model_matrix(hm: pd.DataFrame, hh: pd.DataFrame, category: str) -> tuple[pd.DataFrame, pd.Series]:
    hm_q = (
        hm.groupby(["variant", "subject_2", "question_id", category], dropna=False)["sbert_score"]
        .mean()
        .reset_index(name="hm_qm")
    )
    model_cat = (
        hm_q.groupby(["subject_2", category], dropna=False)["hm_qm"]
        .mean()
        .reset_index(name="hm")
    )
    matrix = model_cat.pivot(index="subject_2", columns=category, values="hm")

    hh_q = (
        hh.groupby(["variant", "question_id", category], dropna=False)["sbert_score"]
        .mean()
        .reset_index(name="hh_q")
    )
    hh_cat = hh_q.groupby(category, dropna=False)["hh_q"].mean()
    return matrix, hh_cat


def compute_correlations(matrix: pd.DataFrame, hh_cat: pd.Series, dimension: str) -> pd.DataFrame:
    cats = [c for c in hh_cat.index if c in matrix.columns]
    rows = []
    for model, row in matrix.iterrows():
        sub = pd.DataFrame({"model": row[cats], "human": hh_cat.loc[cats]}).dropna()
        if len(sub) < 3:
            continue
        pearson = float(sub["model"].corr(sub["human"], method="pearson"))
        spearman = float(sub["model"].corr(sub["human"], method="spearman"))
        rows.append(
            {
                "Model": short_label(model),
                "model_raw": model,
                "Group": MODEL_GROUP.get(model, "other"),
                "Dimension": dimension,
                "Pearson r": pearson,
                "Spearman $\\rho$": spearman,
                "N groups": len(sub),
            }
        )
    return pd.DataFrame(rows)


def winners_table(matrix: pd.DataFrame, dimension_label: str) -> pd.DataFrame:
    rows = []
    for cat in matrix.columns:
        col = matrix[cat].dropna().sort_values(ascending=False)
        if col.empty:
            continue
        winner = col.index[0]
        winner_val = float(col.iloc[0])
        second_val = float(col.iloc[1]) if len(col) > 1 else np.nan
        rows.append(
            {
                dimension_label: cat,
                "Winner": short_label(winner),
                "Group": MODEL_GROUP.get(winner, "other"),
                "HM SBERT": winner_val,
                "Gap to #2": winner_val - second_val if np.isfinite(second_val) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def top_answers_str(df: pd.DataFrame, top_n: int = 3) -> str:
    if df.empty:
        return "--"
    vc = df["response"].astype(str).value_counts().head(top_n)
    return "; ".join(f"{ans} ({cnt})" for ans, cnt in vc.items())


def build_identity_examples(common_qids: set[int]) -> pd.DataFrame:
    human = pd.read_csv(ROOT / "analysis/session2/exports/responses_human.csv")
    model_ib = pd.read_csv(ROOT / "analysis/session2/exports/responses_model_inst_blind.csv")
    pc = pd.read_parquet(ROOT / "analysis/session2/exports/pair_cache.parquet")

    human = human[(human["question_id"].isin(common_qids)) & (human["variant"].isin(["C", "A"]))].copy()
    model_ib = model_ib[
        (model_ib["question_id"].isin(common_qids))
        & (model_ib["variant"].isin(["C", "A"]))
        & (model_ib["model"].isin(set(MODELS_7B) - EXCLUDED_MATCHED_MODELS))
    ].copy()
    pc = pc[
        (pc["question_id"].isin(common_qids))
        & (pc["variant"].isin(["C", "A"]))
        & (pc["pair_type"].isin(["HH", "HM"]))
        & (pc["op"] == "ident")
    ].copy()

    meta = (
        human[human["variant"] == "C"][["question_id", "question_en", "ent", "op", "gt"]]
        .drop_duplicates("question_id")
        .set_index("question_id")
    )
    hh_c = pc[(pc["pair_type"] == "HH") & (pc["variant"] == "C")].groupby("question_id")["sbert_score"].mean()
    hh_a = pc[(pc["pair_type"] == "HH") & (pc["variant"] == "A")].groupby("question_id")["sbert_score"].mean()
    hm_c = pc[(pc["pair_type"] == "HM") & (pc["variant"] == "C")].groupby("question_id")["sbert_score"].mean()
    hm_a = pc[(pc["pair_type"] == "HM") & (pc["variant"] == "A")].groupby("question_id")["sbert_score"].mean()

    qdf = meta.join(hh_c.rename("HH_C")).join(hh_a.rename("HH_A")).join(hm_c.rename("HM_C")).join(hm_a.rename("HM_A"))
    qdf = qdf.dropna().copy()
    qdf["HH C→A"] = qdf["HH_C"] - qdf["HH_A"]
    qdf["HM C→A"] = qdf["HM_C"] - qdf["HM_A"]
    qdf["HH-HM"] = qdf["HH_C"] - qdf["HM_C"]

    top = qdf.sort_values(["HH C→A", "HH-HM"], ascending=False).head(5)
    rows = []
    for qid, row in top.iterrows():
        h_sub = human[(human["question_id"] == qid) & (human["variant"] == "C")]
        m_sub = model_ib[(model_ib["question_id"] == qid) & (model_ib["variant"] == "C")]
        rows.append(
            {
                "Question": str(row["question_en"])[:72],
                "Entity": row["ent"],
                "HH$_C$": row["HH_C"],
                "HH$_A$": row["HH_A"],
                "$\\Delta_{C\\to A}^{HH}$": row["HH C→A"],
                "HH-HM": row["HH-HM"],
                "Human top": top_answers_str(h_sub),
                "Model top": top_answers_str(m_sub),
            }
        )
    return pd.DataFrame(rows)


def plot_correlations(df: pd.DataFrame) -> None:
    dims = ["Operation groups", "Entity groups"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    for ax, dim in zip(axes, dims):
        sub = df[df["Dimension"] == dim].sort_values("Pearson r", ascending=True)
        y = np.arange(len(sub))
        colors = [GROUP_COLORS.get(g, "#777777") for g in sub["Group"]]
        ax.barh(y, sub["Pearson r"], color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["Model"], fontsize=8)
        ax.set_title(dim, fontsize=11)
        ax.set_xlabel("Pearson correlation with HH grouped pattern")
        ax.axvline(0, color="#999999", lw=0.8)
        ax.grid(axis="x", alpha=0.2)
        winner_idx = int(sub["Pearson r"].idxmax())
        # annotate winner
        winner_row = sub.loc[winner_idx]
        ax.text(
            winner_row["Pearson r"] + 0.01,
            list(sub.index).index(winner_idx),
            "winner",
            va="center",
            fontsize=8,
            color="#333333",
        )
    fig.suptitle("How closely each model follows the human grouped-agreement pattern", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "group_pattern_correlation_matched_7b.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    matched_models = set(MODELS_7B) - EXCLUDED_MATCHED_MODELS
    pair, meta = _load()
    hm, hh = _summaries(pair, meta, matched_models)

    meta_c = meta[meta["variant"] == "C"].copy()
    ent_map, op_map, ent_order, op_order, _, _ = _group_maps(meta_c)
    hm = hm[hm["subject_group_2"] != "standalone LLM (think)"].copy()
    hm = hm.assign(
        ent_grp=hm["ent"].map(ent_map).fillna("other_entities"),
        op_grp=hm["op"].map(op_map).fillna("other_ops"),
    )
    hh = hh.assign(
        ent_grp=hh["ent"].map(ent_map).fillna("other_entities"),
        op_grp=hh["op"].map(op_map).fillna("other_ops"),
    )

    hm_c = hm[hm["variant"] == "C"].copy()
    ent_n = hm_c.groupby("ent_grp")["question_id"].nunique()
    op_n = hm_c.groupby("op_grp")["question_id"].nunique()
    ent_order = [c for c in ent_order if int(ent_n.get(c, 0)) > 0]
    op_order = [c for c in op_order if int(op_n.get(c, 0)) > 0]

    ent_matrix, ent_hh = build_category_model_matrix(
        hm[hm["ent_grp"].isin(ent_order)].copy(),
        hh[hh["ent_grp"].isin(ent_order)].copy(),
        "ent_grp",
    )
    op_matrix, op_hh = build_category_model_matrix(
        hm[hm["op_grp"].isin(op_order)].copy(),
        hh[hh["op_grp"].isin(op_order)].copy(),
        "op_grp",
    )

    corr_df = pd.concat(
        [
            compute_correlations(op_matrix.reindex(sorted(op_matrix.index)), op_hh, "Operation groups"),
            compute_correlations(ent_matrix.reindex(sorted(ent_matrix.index)), ent_hh, "Entity groups"),
        ],
        ignore_index=True,
    )
    corr_df = corr_df.sort_values(["Dimension", "Pearson r"], ascending=[True, False]).reset_index(drop=True)
    plot_correlations(corr_df)
    export_latex(
        corr_df.drop(columns=["model_raw"]),
        "table_group_pattern_correlation.tex",
        r"Correlation between each model's grouped HM-SBERT profile and the human grouped HH-SBERT profile (matched 7/8B subset, variants averaged). Higher values indicate that the model preserves the same category-level patterning as humans, regardless of absolute score level.",
        "tab:group_pattern_correlation",
    )

    op_win = winners_table(op_matrix.reindex(sorted(op_matrix.index)), "Operation")
    ent_win = winners_table(ent_matrix.reindex(sorted(ent_matrix.index)), "Entity")
    export_latex(
        op_win,
        "table_group_pattern_winners_by_op.tex",
        r"Winners by operation group: model with the highest HM-SBERT average in each operation category (matched 7/8B subset, variants averaged).",
        "tab:group_winners_op",
    )
    export_latex(
        ent_win,
        "table_group_pattern_winners_by_entity.tex",
        r"Winners by entity group: model with the highest HM-SBERT average in each entity category (matched 7/8B subset, variants averaged).",
        "tab:group_winners_entity",
    )

    _, common_qids, _, _ = load_human_subset(ROOT, min_answers=MIN_ANS, translate=False, verbose=False)
    ident_df = build_identity_examples(set(common_qids))
    export_latex(
        ident_df,
        "table_identity_focus_examples.tex",
        r"Identity-focused human examples with the largest human degradation under pronominalization. These cases highlight where human agreement itself collapses most strongly, making identity the most fragile operation under referential underspecification.",
        "tab:identity_focus_examples",
    )


if __name__ == "__main__":
    main()
