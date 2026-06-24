"""
Generate qualitative question-diagnostic tables.

Exports:
  - top C->A degradation by operation/entity
  - top HH-HM gap by operation/entity
  - each table includes compact top-answer summaries for humans and models

Run from repo root:
  conda run -n zero python figures/question_diagnostic/generate_qualitative_tables.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from config import MODELS_7B
from config import extend_pair_cache_with_yesno

EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "figures/question_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_7B = set(MODELS_7B)


def export_latex(df: pd.DataFrame, filename: str, caption: str, label: str) -> None:
    path = OUT_DIR / filename
    latex = df.to_latex(
        escape=False,
        float_format=lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x),
        column_format="p{4.0cm}llrrrp{2.6cm}p{2.6cm}",
    )
    with open(path, "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write(latex)
        f.write("\\end{table*}\n")
    print(f"Exported: {path}")


def top_answers_str(df: pd.DataFrame, group_col: str | None = None, top_n: int = 2) -> str:
    if df.empty:
        return "--"
    vc = df["response"].astype(str).value_counts().head(top_n)
    parts = [f"{ans} ({cnt})" for ans, cnt in vc.items()]
    return "; ".join(parts)


def build_qdf(include_yesno: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pc = pd.read_parquet(EXPORTS / "pair_cache.parquet")
    if include_yesno:
        pc = extend_pair_cache_with_yesno(pc, EXPORTS)
    human = pd.read_csv(EXPORTS / "responses_human.csv")
    model_ib = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")

    meta_full = human[human["variant"] == "C"].drop_duplicates("question_id")[
        ["question_id", "question_en", "ent", "op", "gt"]
    ].set_index("question_id")

    hm_vc = pc[
        (pc["variant"] == "C")
        & (pc["pair_type"] == "HM")
        & (pc["subject_2"].isin(_7B))
    ].groupby("question_id")["sbert_score"].mean()
    hm_va = pc[
        (pc["variant"] == "A")
        & (pc["pair_type"] == "HM")
        & (pc["subject_2"].isin(_7B))
    ].groupby("question_id")["sbert_score"].mean()
    hh_vc = pc[(pc["variant"] == "C") & (pc["pair_type"] == "HH")].groupby("question_id")["sbert_score"].mean()

    qdf = meta_full.join(hm_vc.rename("HM")).join(hm_va.rename("HM_A")).join(hh_vc.rename("HH"))
    qdf = qdf[qdf["HM"].notna()].copy()
    qdf[r"C$\to$A"] = qdf["HM"] - qdf["HM_A"]
    qdf["HH-HM"] = qdf["HH"] - qdf["HM"]

    return qdf, human, model_ib


def attach_answer_summaries(
    sub: pd.DataFrame,
    human: pd.DataFrame,
    model_ib: pd.DataFrame,
    secondary_col: str,
    secondary_name: str,
    score_col: str,
) -> pd.DataFrame:
    rows = []
    for qid, row in sub.iterrows():
        h_sub = human[(human["question_id"] == qid) & (human["variant"] == "C")]
        m_sub = model_ib[
            (model_ib["question_id"] == qid)
            & (model_ib["variant"] == "C")
            & (model_ib["model"].isin(_7B))
        ]
        rows.append(
            {
                "Question": str(row["question_en"])[:52],
                secondary_name: row[secondary_col],
                "Op" if secondary_name == "Ent" else "Ent": row["op"] if secondary_name == "Ent" else row["ent"],
                "HM": row["HM"],
                "HH": row["HH"],
                score_col: row[score_col],
                "Human top": top_answers_str(h_sub, top_n=2),
                "Model top": top_answers_str(m_sub, top_n=2),
            }
        )
    return pd.DataFrame(rows).set_index("Question")


def export_ranked_tables(qdf: pd.DataFrame, human: pd.DataFrame, model_ib: pd.DataFrame) -> None:
    op_focus = ["ident", "count", "attr", "act", "spat"]
    ent_focus = ["person", "animal", "object", "food"]

    op_deg_rows = []
    for op in op_focus:
        sub = qdf[qdf["op"] == op].sort_values(r"C$\to$A", ascending=False).head(3)
        op_deg_rows.append(
            attach_answer_summaries(sub, human, model_ib, "op", "Op", r"C$\to$A")
        )
    op_deg_df = pd.concat(op_deg_rows)
    export_latex(
        op_deg_df,
        "table_qualitative_by_op_answers.tex",
        r"Top questions by C$\to$A degradation per operation type (variant~C, 7/8B). "
        r"These are the most entity-anchor-sensitive examples within each operation, "
        r"with compact human/model answer summaries.",
        "tab:qual_op_answers",
    )

    ent_deg_rows = []
    for ent in ent_focus:
        sub = qdf[qdf["ent"] == ent].sort_values(r"C$\to$A", ascending=False).head(3)
        ent_deg_rows.append(
            attach_answer_summaries(sub, human, model_ib, "ent", "Ent", r"C$\to$A")
        )
    ent_deg_df = pd.concat(ent_deg_rows)
    export_latex(
        ent_deg_df,
        "table_qualitative_by_entity_answers.tex",
        r"Top questions by C$\to$A degradation per entity type (variant~C, 7/8B). "
        r"These are the strongest entity-anchor-dependence cases within each entity class, "
        r"with compact human/model answer summaries.",
        "tab:qual_entity_answers",
    )

    op_gap_rows = []
    for op in op_focus:
        sub = qdf[qdf["op"] == op].sort_values("HH-HM", ascending=False).head(3)
        op_gap_rows.append(
            attach_answer_summaries(sub, human, model_ib, "op", "Op", "HH-HM")
        )
    op_gap_df = pd.concat(op_gap_rows)
    export_latex(
        op_gap_df,
        "table_qualitative_gap_by_op_answers.tex",
        r"Top questions by HH$-$HM gap per operation type (variant~C, 7/8B). "
        r"These are the clearest human-consensus / model-mismatch examples within each operation, "
        r"with compact human/model answer summaries.",
        "tab:qual_gap_op_answers",
    )

    ent_gap_rows = []
    for ent in ent_focus:
        sub = qdf[qdf["ent"] == ent].sort_values("HH-HM", ascending=False).head(3)
        ent_gap_rows.append(
            attach_answer_summaries(sub, human, model_ib, "ent", "Ent", "HH-HM")
        )
    ent_gap_df = pd.concat(ent_gap_rows)
    export_latex(
        ent_gap_df,
        "table_qualitative_gap_by_entity_answers.tex",
        r"Top questions by HH$-$HM gap per entity type (variant~C, 7/8B). "
        r"These are the clearest human-consensus / model-mismatch examples within each entity class, "
        r"with compact human/model answer summaries.",
        "tab:qual_gap_entity_answers",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free_text_only", action="store_true")
    args = parser.parse_args()

    qdf, human, model_ib = build_qdf(include_yesno=not args.free_text_only)
    export_ranked_tables(qdf, human, model_ib)
    print(f"\nDone. Outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
