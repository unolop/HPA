from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import json


ROOT = Path(__file__).resolve().parents[1]
PAIR_CACHE = ROOT / "analysis" / "session2" / "exports" / "pair_cache_cleaned.parquet"
MODEL_RESPONSES = ROOT / "analysis" / "session2" / "exports" / "responses_model_inst_blind.csv"
LATEX_TABLES = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables"

VARIANT_ORDER = ["C", "B", "A"]
VARIANT_NAME = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
MATCHED_7B_MODELS = [
    "InternVL-8B",
    "Qwen3-VL-8B",
    "LLaVA-1.5-7B",
    "LLaVA-Mistral",
    "LLaVA-Vicuna",
    "InternVL-8B (LM)",
    "Qwen3-VL-8B (LM)",
    "LLaVA-1.5 (LM)",
    "LLaVA-Mistral (LM)",
    "LLaVA-Vicuna (LM)",
    "Qwen3-8B",
    "Qwen2.5-7B-Instruct",
    "Vicuna-7B",
]
MATCHED_7B_MODEL_GROUP = {
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
}
DIST_YN_ORDER = ["yes", "no", "others"]
DIST_NUM_ORDER = ["0", "1", "2–3", "4–5", "6–10", "11–20", ">20", "others"]
DIST_TEXT_TOP_K = 10


def load_pair_cache() -> pd.DataFrame:
    return pd.read_parquet(PAIR_CACHE)


def load_model_responses() -> pd.DataFrame:
    return pd.read_csv(MODEL_RESPONSES)


def load_human_responses() -> pd.DataFrame:
    return pd.read_csv(ROOT / "analysis" / "session2" / "exports" / "responses_human.csv")


def hh_question_means() -> pd.DataFrame:
    df = load_pair_cache()
    hh = df[df["pair_type"] == "HH"].copy()
    return (
        hh.groupby(["question_id", "variant", "op", "ent"], as_index=False)["sbert_score"]
        .mean()
        .rename(columns={"sbert_score": "hh_sbert"})
    )


def hm_question_means(matched_only: bool = False) -> pd.DataFrame:
    df = load_pair_cache()
    hm = df[df["pair_type"] == "HM"].copy()
    hm = (
        hm.groupby(["question_id", "variant", "op", "ent", "subject_2"], as_index=False)["sbert_score"]
        .mean()
        .rename(columns={"subject_2": "model", "sbert_score": "hm_sbert"})
    )
    if matched_only:
        hm = hm[hm["model"].isin(MATCHED_7B_MODELS)].copy()
    return hm


def pearson_with_p(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    sub = pd.concat([x, y], axis=1).dropna()
    r, p = stats.pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
    return float(r), float(p)


def to_latex_table(
    df: pd.DataFrame,
    out_path: Path,
    caption: str,
    label: str,
    column_format: str | None = None,
    float_formatters: dict[str, str] | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = {}
    if float_formatters:
        for col, spec in float_formatters.items():
            if col in df.columns:
                fmt[col] = lambda x, s=spec: format(x, s)
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        formatters=fmt if fmt else None,
    )
    lines = latex.splitlines()
    lines.insert(1, rf"\caption{{{caption}}}")
    lines.insert(2, rf"\label{{{label}}}")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def pretty_print_path(path: Path) -> str:
    return str(path.resolve())


def variant_summary_table(hh_q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in VARIANT_ORDER:
        sub = hh_q[hh_q["variant"] == v]
        rows.append(
            {
                "Variant": VARIANT_NAME[v],
                "Code": v,
                "N questions": int(sub["question_id"].nunique()),
                "Mean HH SBERT": float(sub["hh_sbert"].mean()),
                "Std HH SBERT": float(sub["hh_sbert"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def pairwise_variant_correlation_table(hh_q: pd.DataFrame) -> pd.DataFrame:
    piv = hh_q.pivot(index="question_id", columns="variant", values="hh_sbert")
    rows = []
    for a, b in [("C", "B"), ("C", "A"), ("B", "A")]:
        sub = piv[[a, b]].dropna()
        r, p = stats.pearsonr(sub[a], sub[b])
        rows.append(
            {
                "Pair": f"{VARIANT_NAME[a]} vs {VARIANT_NAME[b]}",
                "Codes": f"{a} vs {b}",
                "Pearson r": float(r),
                "p": float(p),
                "N questions": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def grouped_pattern_table(dimension: str) -> pd.DataFrame:
    hh_q = hh_question_means()
    hm_q = hm_question_means(matched_only=True)
    key = "op" if dimension == "op" else "ent"

    hh_profile = hh_q.groupby(key, as_index=True)["hh_sbert"].mean()
    hm_profile = hm_q.groupby(["model", key], as_index=False)["hm_sbert"].mean()
    matrix = hm_profile.pivot(index="model", columns=key, values="hm_sbert")

    rows = []
    for model, row in matrix.iterrows():
        aligned = pd.concat([hh_profile.rename("hh"), row.rename("hm")], axis=1).dropna()
        r, p = stats.pearsonr(aligned["hh"], aligned["hm"])
        rho, rho_p = stats.spearmanr(aligned["hh"], aligned["hm"])
        rows.append(
            {
                "Model": model,
                "Dimension": "Operation groups" if dimension == "op" else "Entity groups",
                "Pearson r": float(r),
                "Pearson p": float(p),
                "Pearson sig": "yes" if p < 0.05 else "no",
                "Spearman rho": float(rho),
                "Spearman p": float(rho_p),
                "Spearman sig": "yes" if rho_p < 0.05 else "no",
                "N groups": int(len(aligned)),
            }
        )
    return pd.DataFrame(rows).sort_values("Pearson r", ascending=False).reset_index(drop=True)


def scatter_correlation_table() -> pd.DataFrame:
    hh_q = hh_question_means()[["question_id", "variant", "hh_sbert"]]
    hm_q = hm_question_means(matched_only=True)
    merged = hm_q.merge(hh_q, on=["question_id", "variant"], how="inner")

    rows = []
    for model, sub in merged.groupby("model"):
        r, p = stats.pearsonr(sub["hh_sbert"], sub["hm_sbert"])
        row = {
            "Model": model,
            "All r": float(r),
            "p": float(p),
            "Significant": "yes" if p < 0.05 else "no",
            "N": int(len(sub)),
        }
        for v in VARIANT_ORDER:
            sv = sub[sub["variant"] == v]
            row[VARIANT_NAME[v]] = float(stats.pearsonr(sv["hh_sbert"], sv["hm_sbert"])[0])
            row[f"N {VARIANT_NAME[v]}"] = int(len(sv))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("All r", ascending=False).reset_index(drop=True)


def blind_accuracy_summary(variant: str = "C") -> pd.DataFrame:
    df = load_model_responses()
    sub = df[df["variant"] == variant].copy()
    return (
        sub.groupby("model", as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "Blind accuracy"})
        .sort_values("Blind accuracy", ascending=False)
        .reset_index(drop=True)
    )


def qualitative_qdf(include_yesno: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pc = pd.read_parquet(ROOT / "analysis" / "session2" / "exports" / "pair_cache_cleaned.parquet")
    if include_yesno:
        from analysis.config import extend_pair_cache_with_yesno
        pc = extend_pair_cache_with_yesno(pc, ROOT / "analysis" / "session2" / "exports")
    human = pd.read_csv(ROOT / "analysis" / "session2" / "exports" / "responses_human.csv")
    model_ib = pd.read_csv(ROOT / "analysis" / "session2" / "exports" / "responses_model_inst_blind.csv")

    meta_full = human[human["variant"] == "C"].drop_duplicates("question_id")[
        ["question_id", "question_en", "ent", "op", "gt"]
    ].set_index("question_id")

    hm_vc = pc[
        (pc["variant"] == "C")
        & (pc["pair_type"] == "HM")
        & (pc["subject_2"].isin(MATCHED_7B_MODELS))
    ].groupby("question_id")["sbert_score"].mean()
    hm_va = pc[
        (pc["variant"] == "A")
        & (pc["pair_type"] == "HM")
        & (pc["subject_2"].isin(MATCHED_7B_MODELS))
    ].groupby("question_id")["sbert_score"].mean()
    hh_vc = pc[(pc["variant"] == "C") & (pc["pair_type"] == "HH")].groupby("question_id")["sbert_score"].mean()

    qdf = meta_full.join(hm_vc.rename("HM")).join(hm_va.rename("HM_A")).join(hh_vc.rename("HH"))
    qdf = qdf[qdf["HM"].notna()].copy()
    qdf[r"C$\to$A"] = qdf["HM"] - qdf["HM_A"]
    qdf["HH-HM"] = qdf["HH"] - qdf["HM"]
    return qdf, human, model_ib


def top_answers_str(df: pd.DataFrame, top_n: int = 2) -> str:
    if df.empty:
        return "--"
    vc = df["response"].astype(str).value_counts().head(top_n)
    return "; ".join(f"{ans} ({cnt})" for ans, cnt in vc.items())


def sample_composition_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qmeta = pd.DataFrame(json.load(open(ROOT / "experiment" / "s2_v4" / "s4_question.json")))
    human = pd.read_csv(ROOT / "analysis" / "session2" / "exports" / "answers_human_q113.csv")
    qids = set(human[human["variant"] == "C"]["question_id"].unique())
    sub = qmeta[qmeta["question_id"].isin(qids)].copy()

    sub["bucket"] = sub["answer_type"].map({"yesno": "Yes/No", "text": "Free-text"})
    sub.loc[sub["op"] == "count", "bucket"] = "Count"

    order = ["Yes/No", "Count", "Free-text"]

    def _block(df: pd.DataFrame, key: str | None = None) -> pd.DataFrame:
        if key is None:
            counts = df["bucket"].value_counts().to_dict()
            row = {"Group": "All questions"}
            for col in order:
                row[col] = int(counts.get(col, 0))
            row["Total"] = int(len(df))
            return pd.DataFrame([row])
        ct = pd.crosstab(df[key], df["bucket"])
        for col in order:
            if col not in ct.columns:
                ct[col] = 0
        ct = ct[order].copy()
        ct["Total"] = ct.sum(axis=1)
        ct = ct.reset_index().rename(columns={key: "Group"})
        return ct

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

    overall = _block(sub, None)
    by_op = _block(sub, "op")
    by_op["Group"] = by_op["Group"].map(OP_FULL_NAMES).fillna(by_op["Group"])
    by_ent = _block(sub, "ent")
    by_ent["Group"] = by_ent["Group"].map(ENT_FULL_NAMES).fillna(by_ent["Group"])
    return overall, by_op, by_ent


def write_sample_composition_table(out_path: Path | None = None) -> Path:
    if out_path is None:
        out_path = LATEX_TABLES / "sample_q113_composition.tex"

    overall, by_op, by_ent = sample_composition_tables()

    def _rows(df: pd.DataFrame) -> list[str]:
        lines = []
        for _, row in df.iterrows():
            lines.append(
                f"{row['Group']} & {int(row['Yes/No'])} & {int(row['Count'])} & "
                f"{int(row['Free-text'])} & {int(row['Total'])} \\\\"
            )
        return lines

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Composition of the $113$-question human-study subset by answer format and semantic grouping. Free-text denotes non-yes/no, non-count questions. Counts are based on the original-formulation question set and therefore do not vary across control variants.}",
        r"\label{tab:q113_sample_composition}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Group & Yes/No & Count & Free-text & Total \\",
        r"\midrule",
        r"\multicolumn{5}{c}{\textit{Overall}} \\",
        r"\midrule",
        *_rows(overall),
        r"\midrule",
        r"\multicolumn{5}{c}{\textit{By operation}} \\",
        r"\midrule",
        *_rows(by_op),
        r"\midrule",
        r"\multicolumn{5}{c}{\textit{By entity}} \\",
        r"\midrule",
        *_rows(by_ent),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    return out_path


def write_sample_composition_horizontal(out_path: Path | None = None) -> Path:
    """Horizontal composition table: answer types as rows, group categories as columns.

    Uses VQA v2 answer_type (yes/no, number, other) for correct 26/26/61 split.
    """
    if out_path is None:
        out_path = LATEX_TABLES / "sample_q113_composition_horizontal.tex"

    from analysis.utils.vqa import VQAAnswerMapper

    qmeta = pd.DataFrame(json.load(open(ROOT / "experiment" / "s2_v4" / "s4_question.json")))
    human = pd.read_csv(ROOT / "analysis" / "session2" / "exports" / "answers_human_q113.csv")
    qids = set(human[human["variant"] == "C"]["question_id"].unique())
    sub = qmeta[qmeta["question_id"].isin(qids)].copy()

    mapper = VQAAnswerMapper()
    mapper._load()
    qid2at = {int(qid): ann.get("answer_type", "other") for qid, ann in mapper.annotations.items()}
    sub["bucket"] = sub["question_id"].map(qid2at).map(
        {"yes/no": "Yes/No", "number": "Number", "other": "Free-text"}
    )

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
    sub["op_label"] = sub["op"].map(OP_FULL_NAMES).fillna(sub["op"])
    sub["ent_label"] = sub["ent"].map(ENT_FULL_NAMES).fillna(sub["ent"])

    answer_types = ["Yes/No", "Number", "Free-text"]

    # Build crosstabs
    by_op = pd.crosstab(sub["op_label"], sub["bucket"])
    for col in answer_types:
        if col not in by_op.columns:
            by_op[col] = 0
    by_op = by_op[answer_types]
    by_op["Total"] = by_op.sum(axis=1)
    by_op = by_op.sort_values("Total", ascending=False)
    op_names = by_op.index.tolist()

    by_ent = pd.crosstab(sub["ent_label"], sub["bucket"])
    for col in answer_types:
        if col not in by_ent.columns:
            by_ent[col] = 0
    by_ent = by_ent[answer_types]
    by_ent["Total"] = by_ent.sum(axis=1)
    by_ent = by_ent.sort_values("Total", ascending=False)
    ent_names = by_ent.index.tolist()

    n_op = len(op_names)
    n_ent = len(ent_names)

    # Column spec: Answer Type | op cols | ent cols | Total
    col_spec = "l" + "c" * n_op + "|" + "c" * n_ent + "|c"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row 1: spans
    lines.append(
        r"& \multicolumn{" + str(n_op) + r"}{c|}{\textbf{Operation}}"
        r" & \multicolumn{" + str(n_ent) + r"}{c|}{\textbf{Entity}}"
        r" & \\"
    )

    # cmidrules
    op_start, op_end = 2, 1 + n_op
    ent_start, ent_end = op_end + 1, op_end + n_ent
    total_col = ent_end + 1
    lines.append(rf"\cmidrule(lr){{{op_start}-{op_end}}} \cmidrule(lr){{{ent_start}-{ent_end}}}")

    # Header row 2: individual group names
    header = [r"\textbf{Answer Type}"]
    for name in op_names:
        # Abbreviate long names
        short = name.replace("World Know.", "Know.").replace("Text Reading", "Text")
        header.append(rf"\rotatebox{{60}}{{\small {short}}}")
    for name in ent_names:
        header.append(rf"\rotatebox{{60}}{{\small {name}}}")
    header.append(r"\textbf{Total}")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Data rows: one per answer type
    for at in answer_types:
        parts = [at]
        row_total = 0
        for name in op_names:
            v = int(by_op.loc[name, at])
            parts.append(str(v) if v > 0 else "--")
            row_total += v
        for name in ent_names:
            v = int(by_ent.loc[name, at])
            parts.append(str(v) if v > 0 else "--")
        parts.append(str(row_total))
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\midrule")

    # Total row
    parts = [r"\textbf{Total}"]
    for name in op_names:
        parts.append(rf"\textbf{{{int(by_op.loc[name, 'Total'])}}}")
    for name in ent_names:
        parts.append(rf"\textbf{{{int(by_ent.loc[name, 'Total'])}}}")
    parts.append(rf"\textbf{{{len(sub)}}}")
    lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Composition of the $113$-question human-study subset by answer format "
        r"and semantic grouping. Columns show operation type and entity type; "
        r"rows show answer format (yes/no, count, free-text). "
        r"Counts are based on original-formulation questions and do not vary across control variants.}"
    )
    lines.append(r"\label{tab:q113_sample_composition_h}")
    lines.append(r"\end{table*}")
    lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path


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
            & (model_ib["model"].isin(MATCHED_7B_MODELS))
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
    return pd.DataFrame(rows)


def hh_ranked_examples() -> tuple[pd.DataFrame, pd.DataFrame]:
    pc = pd.read_parquet(ROOT / "analysis" / "session2" / "exports" / "pair_cache_cleaned.parquet")
    human = pd.read_csv(ROOT / "analysis" / "session2" / "exports" / "responses_human.csv")

    hh = (
        pc[pc["pair_type"] == "HH"]
        .groupby(["question_id", "variant"], as_index=False)["sbert_score"]
        .mean()
        .rename(columns={"sbert_score": "hh_sbert"})
    )

    meta = human.groupby(["question_id", "variant"], as_index=False).agg(
        question_en=("question_en", "first"),
        op=("op", "first"),
        ent=("ent", "first"),
        accuracy=("accuracy", "mean"),
    )
    ranked = meta.merge(hh, on=["question_id", "variant"])
    return ranked, human


def top_answers(df: pd.DataFrame, n: int = 5) -> str:
    counts = df["response"].astype(str).str.strip()
    counts = counts[counts != ""].value_counts().head(n)
    return "; ".join(f"{answer} ({count})" for answer, count in counts.items())


def variant_top_bottom_table(ranked: pd.DataFrame, human: pd.DataFrame, variant: str, top_n: int = 10) -> pd.DataFrame:
    sub = ranked[ranked["variant"] == variant].sort_values("hh_sbert", ascending=False).reset_index(drop=True)
    rows = []
    for split, frame in [("Top", sub.head(top_n)), ("Bottom", sub.tail(top_n))]:
        for _, row in frame.iterrows():
            h_sub = human[(human["question_id"] == row["question_id"]) & (human["variant"] == variant)]
            rows.append(
                {
                    "Split": split,
                    "Question ID": int(row["question_id"]),
                    "HH SBERT": float(row["hh_sbert"]),
                    "Op": row["op"],
                    "Ent": row["ent"],
                    "Question": row["question_en"],
                    "Top answers": top_answers(h_sub),
                }
            )
    return pd.DataFrame(rows)


def hh_degradation_table(ranked: pd.DataFrame, human: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    pivot = ranked.pivot(index="question_id", columns="variant", values="hh_sbert").reset_index()
    meta_c = ranked[ranked["variant"] == "C"][["question_id", "question_en", "op", "ent"]]
    deg = meta_c.merge(pivot, on="question_id")
    deg["Delta C->A"] = deg["C"] - deg["A"]
    rows = []
    for _, row in deg.sort_values("Delta C->A", ascending=False).head(top_n).iterrows():
        h_c = human[(human["question_id"] == row["question_id"]) & (human["variant"] == "C")]
        h_a = human[(human["question_id"] == row["question_id"]) & (human["variant"] == "A")]
        rows.append(
            {
                "Question ID": int(row["question_id"]),
                "Delta C->A": float(row["Delta C->A"]),
                "Op": row["op"],
                "Ent": row["ent"],
                "Question": row["question_en"],
                "Top answers C": top_answers(h_c),
                "Top answers A": top_answers(h_a),
            }
        )
    return pd.DataFrame(rows)


def top_questions_tables(include_yesno: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    pc = pd.read_parquet(ROOT / "analysis" / "session2" / "exports" / "pair_cache_cleaned.parquet")
    hh = pc[pc["pair_type"] == "HH"]
    hm = pc[(pc["pair_type"] == "HM") & (pc["subject_2"].isin(MATCHED_7B_MODELS))]

    hh_q = hh.groupby(["question_id", "question_en", "variant", "op", "ent"], as_index=False).agg(
        hh_sbert=("sbert_score", "mean")
    )
    hm_q = hm.groupby(["question_id", "variant"], as_index=False).agg(hm_sbert=("sbert_score", "mean"))

    merged = hh_q.merge(hm_q, on=["question_id", "variant"])
    vC = merged[merged["variant"] == "C"].copy()
    vA = merged[merged["variant"] == "A"][["question_id", "hh_sbert", "hm_sbert"]].rename(
        columns={"hh_sbert": "hh_A", "hm_sbert": "hm_A"}
    )
    vC = vC.merge(vA, on="question_id", how="left")
    vC["hh_drop"] = vC["hh_sbert"] - vC["hh_A"]
    vC["hm_drop"] = vC["hm_sbert"] - vC["hm_A"]

    human_resp = pd.read_csv(ROOT / "analysis" / "session2" / "exports" / "responses_human.csv")

    def _top_answers(qid: int, variant: str = "C", n: int = 3) -> str:
        sub = human_resp[(human_resp["question_id"] == qid) & (human_resp["variant"] == variant)]
        counts = sub["response"].value_counts().head(n)
        return ", ".join([f"{a} ({c})" for a, c in counts.items()])

    def build(group_col: str, n_per_group: int = 2) -> pd.DataFrame:
        rows = []
        groups = sorted(vC[group_col].dropna().unique())
        for grp in groups:
            sub = vC[vC[group_col] == grp].nlargest(n_per_group, "hh_sbert")
            for _, row in sub.iterrows():
                rows.append(
                    {
                        group_col.capitalize(): grp,
                        "Question": row["question_en"][:55] + ("..." if len(row["question_en"]) > 55 else ""),
                        "HH": float(row["hh_sbert"]),
                        "HM": float(row["hm_sbert"]),
                        r"$\Delta_{C \to A}^{HH}$": float(row["hh_drop"]),
                        r"$\Delta_{C \to A}^{HM}$": float(row["hm_drop"]),
                        "Human top answers": _top_answers(int(row["question_id"])),
                    }
                )
        return pd.DataFrame(rows)

    return build("op"), build("ent")


def _top_counts_str(series: pd.Series, n: int = 5) -> str:
    vc = series.astype(str).str.strip()
    vc = vc[vc != ""].value_counts().head(n)
    return "; ".join(f"{ans} ({cnt})" for ans, cnt in vc.items()) if len(vc) else "--"


def structural_alignment_examples(
    *,
    condition: str = "inst_blind",
    variants: tuple[str, ...] = ("C", "B", "A"),
    filter_abstention: bool = False,
    answer_types: tuple[str, ...] = ("yes/no", "number", "text"),
    dimensions: tuple[str, ...] = ("op", "ent"),
    top_n: int = 2,
) -> pd.DataFrame:
    """Representative examples tied to grouped distributional winners.

    For each answer type × grouping dimension, identify the best model by
    grouped-distribution Mean JS, find its best-matching slice, and surface a
    few questions with compact human/model answer summaries.
    """
    human_all, model_all = distribution_base_table_variants(condition=condition, variants=variants)
    human_all = _mark_abstentions(human_all)
    model_all = _mark_abstentions(model_all)

    rows = []
    for dimension in dimensions:
        key = "op" if dimension == "op" else "ent"
        grouping_label = "Operation" if dimension == "op" else "Entity"
        long_df = grouped_distribution_long_table(
            dimension,
            condition=condition,
            variants=variants,
            filter_abstention=filter_abstention,
            answer_types=answer_types,
            include_human_baseline=False,
        )
        for answer_type in answer_types:
            winners = long_df[long_df["Answer type"] == answer_type].sort_values(["Mean JS", "Mean TV"])
            if winners.empty:
                continue
            winner = winners.iloc[0]
            model_name = winner["Model"]

            human_sub = _answer_type_subset(human_all, answer_type)
            model_sub = _answer_type_subset(model_all[model_all["model"] == model_name], answer_type)
            if filter_abstention:
                model_sub = model_sub[~model_sub["is_abstained"]].copy()
                kept_qvs = set(_qv_pairs(model_sub))
                human_sub = human_sub[[qv in kept_qvs for qv in _qv_pairs(human_sub)]].copy()
                human_sub = human_sub[~human_sub["is_abstained"]].copy()

            human_sub, model_sub, category_order = _prepare_distribution_category(human_sub, model_sub, answer_type)
            slice_rows = []
            for slice_name in sorted(set(human_sub[key].dropna()) & set(model_sub[key].dropna())):
                h_slice = human_sub[human_sub[key] == slice_name]
                m_slice = model_sub[model_sub[key] == slice_name]
                if h_slice.empty or m_slice.empty:
                    continue
                human_counts = h_slice["category"].value_counts().reindex(category_order, fill_value=0)
                model_counts = m_slice["category"].value_counts().reindex(category_order, fill_value=0)
                if human_counts.sum() == 0 or model_counts.sum() == 0:
                    continue
                chi2, p, js, tv, cv = _distribution_stats_from_counts(human_counts, model_counts)
                slice_rows.append((slice_name, js, tv, cv))
            if not slice_rows:
                continue
            best_slice, js_best, tv_best, cv_best = sorted(slice_rows, key=lambda x: (x[1], x[2]))[0]
            h_best = human_sub[human_sub[key] == best_slice].copy()
            m_best = model_sub[model_sub[key] == best_slice].copy()
            qmeta = (
                h_best[["question_id", "question_en", "variant"]]
                .drop_duplicates()
                .sort_values(["variant", "question_id"])
                .head(top_n)
            )
            for _, qrow in qmeta.iterrows():
                qid = int(qrow["question_id"])
                var = str(qrow["variant"])
                hq = h_best[(h_best["question_id"] == qid) & (h_best["variant"] == var)]
                mq = m_best[(m_best["question_id"] == qid) & (m_best["variant"] == var)]
                rows.append(
                    {
                        "Grouping": grouping_label,
                        "Answer type": answer_type,
                        "Filtered": "yes" if filter_abstention else "no",
                        "Winner model": model_name,
                        "Winner group": winner["Group"],
                        "Best slice": best_slice,
                        "Slice JS": js_best,
                        "Slice TV": tv_best,
                        "Question": qrow["question_en"],
                        "Variant": qrow["variant"],
                        "Human top": _top_counts_str(hq["clean_response"], n=5),
                        "Model top": _top_counts_str(mq["clean_response"], n=5),
                    }
                )
    return pd.DataFrame(rows)


def hm_degradation_table(top_n: int = 10, matched_only: bool = True) -> pd.DataFrame:
    """Top questions by HM SBERT drop C→A, with per-variant question text and model answers."""
    pc = pd.read_parquet(PAIR_CACHE)
    model_resp = pd.read_csv(MODEL_RESPONSES)

    hh = pc[pc["pair_type"] == "HH"]
    hm = pc[pc["pair_type"] == "HM"]
    if matched_only:
        hm = hm[hm["subject_2"].isin(MATCHED_7B_MODELS)]

    # per-question per-variant means
    hh_q = hh.groupby(["question_id", "variant", "op", "ent"], as_index=False)["sbert_score"].mean().rename(columns={"sbert_score": "hh_sbert"})
    hm_q = hm.groupby(["question_id", "variant"], as_index=False)["sbert_score"].mean().rename(columns={"sbert_score": "hm_sbert"})

    # question text per variant (C, B, A)
    q_text = pc[["question_id", "variant", "question_en"]].drop_duplicates()
    q_text_wide = q_text.pivot(index="question_id", columns="variant", values="question_en").rename(
        columns={"C": "Question (Original)", "B": "Question (Weaker)", "A": "Question (Pronominalized)"}
    ).reset_index()

    # compute drops
    hm_wide = hm_q.pivot(index="question_id", columns="variant", values="hm_sbert").rename(
        columns={"C": "HM C", "B": "HM B", "A": "HM A"}
    ).reset_index()
    hh_wide = hh_q.pivot(index="question_id", columns="variant", values="hh_sbert").rename(
        columns={"C": "HH C", "B": "HH B", "A": "HH A"}
    ).reset_index()
    meta = hh_q[hh_q["variant"] == "C"][["question_id", "op", "ent"]]

    df = meta.merge(hm_wide, on="question_id").merge(hh_wide, on="question_id").merge(q_text_wide, on="question_id")
    df["HM drop C→A"] = df["HM C"] - df["HM A"]
    df["HH drop C→A"] = df["HH C"] - df["HH A"]

    top = df.nlargest(top_n, "HM drop C→A").reset_index(drop=True)

    # top model answers per variant
    def _model_top(qid, variant, n=3):
        sub = model_resp[(model_resp["question_id"] == qid) & (model_resp["variant"] == variant)]
        counts = sub["response"].value_counts().head(n)
        return ", ".join(f"{a} ({c})" for a, c in counts.items())

    top["Model answers C"] = top.apply(lambda r: _model_top(r["question_id"], "C"), axis=1)
    top["Model answers B"] = top.apply(lambda r: _model_top(r["question_id"], "B"), axis=1)
    top["Model answers A"] = top.apply(lambda r: _model_top(r["question_id"], "A"), axis=1)

    cols = [
        "op", "ent",
        "Question (Original)", "Question (Weaker)", "Question (Pronominalized)",
        "HH C", "HH B", "HH A", "HH drop C→A",
        "HM C", "HM B", "HM A", "HM drop C→A",
        "Model answers C", "Model answers B", "Model answers A",
    ]
    return top[cols]


def _clean_answer_series(series: pd.Series) -> pd.Series:
    from analysis.utils.vqa import preprocess_answer

    return (
        series.fillna("")
        .astype(str)
        .apply(lambda x: preprocess_answer(x, strip_think=True))
        .str.lower()
        .str.strip()
    )


def _classify_yesno(text: str) -> str:
    txt = str(text).strip()
    if txt == "yes" or txt.startswith("yes "):
        return "yes"
    if txt == "no" or txt.startswith("no "):
        return "no"
    return "others"


def _classify_number(text: str) -> str:
    from analysis.utils.vqa import extract_number, bin_number

    value = extract_number(text)
    return bin_number(value) if value is not None else "others"


def distribution_base_table(condition: str = "inst_blind", variant: str = "C") -> tuple[pd.DataFrame, pd.DataFrame]:
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
    human = human[human["variant"] == variant].copy()
    model = model[(model["variant"] == variant) & (model["model"].isin(MATCHED_7B_MODELS))].copy()
    qids = set(human["question_id"].astype(int).unique())
    human = human[human["question_id"].astype(int).isin(qids)].copy()
    model = model[model["question_id"].astype(int).isin(qids)].copy()
    human["answer_type"] = human["question_id"].astype(int).map(qid2atype).fillna("other")
    model["answer_type"] = model["question_id"].astype(int).map(qid2atype).fillna("other")
    human["clean_response"] = _clean_answer_series(human["response"])
    model["clean_response"] = _clean_answer_series(model["response"])
    return human, model


def distribution_base_table_variants(
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
    model = model[(model["variant"].isin(variants)) & (model["model"].isin(MATCHED_7B_MODELS))].copy()
    qvs = set(zip(human["question_id"].astype(int), human["variant"].astype(str)))
    human = human[[qv in qvs for qv in zip(human["question_id"].astype(int), human["variant"].astype(str))]].copy()
    model = model[[qv in qvs for qv in zip(model["question_id"].astype(int), model["variant"].astype(str))]].copy()

    human["answer_type"] = human["question_id"].astype(int).map(qid2atype).fillna("other")
    model["answer_type"] = model["question_id"].astype(int).map(qid2atype).fillna("other")
    human["clean_response"] = _clean_answer_series(human["response"])
    model["clean_response"] = _clean_answer_series(model["response"])
    return human, model


def _distribution_profile(df: pd.DataFrame, category_col: str, category_order: list[str]) -> pd.Series:
    counts = df[category_col].value_counts().reindex(category_order, fill_value=0).astype(float)
    total = counts.sum()
    if total <= 0:
        return counts * 0.0
    return counts / total


def _qv_pairs(df: pd.DataFrame) -> list[tuple[int, str]]:
    return list(zip(df["question_id"].astype(int), df["variant"].astype(str)))


def _mark_abstentions(df: pd.DataFrame) -> pd.DataFrame:
    from analysis.utils.abstention import classify, is_abstained

    out = df.copy()
    out["abstention_class"] = out["clean_response"].apply(lambda x: classify(x, None))
    out["is_abstained"] = out["abstention_class"].apply(is_abstained)
    return out


def _distribution_stats_from_counts(
    human_counts: pd.Series,
    model_counts: pd.Series,
) -> tuple[float, float, float, float, float]:
    contingency = pd.concat(
        [human_counts.rename("human"), model_counts.rename("model")],
        axis=1,
    ).fillna(0.0)
    human_probs = contingency["human"] / contingency["human"].sum()
    model_probs = contingency["model"] / contingency["model"].sum()
    js = float(jensenshannon(human_probs, model_probs, base=2) ** 2)
    tv = float(0.5 * (human_probs - model_probs).abs().sum())
    # Chi-square can fail on degenerate slices; drop empty categories and
    # return NaN when the effective table has fewer than two categories.
    eff = contingency.loc[contingency.sum(axis=1) > 0].copy()
    if len(eff) >= 2 and eff["human"].sum() > 0 and eff["model"].sum() > 0:
        try:
            chi2, p, _, _ = stats.chi2_contingency(eff.values)
            n = float(eff.values.sum())
            k = float(min(eff.shape[0] - 1, eff.shape[1] - 1))
            cramers_v = float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else float("nan")
        except ValueError:
            chi2, p, cramers_v = float("nan"), float("nan"), float("nan")
    else:
        chi2, p, cramers_v = float("nan"), float("nan"), float("nan")
    return float(chi2), float(p), js, tv, cramers_v


def yesno_distribution_table(condition: str = "inst_blind", variant: str = "C") -> tuple[pd.DataFrame, pd.DataFrame]:
    human, model = distribution_base_table(condition=condition, variant=variant)
    human["category"] = human["clean_response"].apply(_classify_yesno)
    model["category"] = model["clean_response"].apply(_classify_yesno)

    human_counts = human["category"].value_counts().reindex(DIST_YN_ORDER, fill_value=0)
    human_profile = (
        _distribution_profile(human, "category", DIST_YN_ORDER)
        .rename("Proportion")
        .reset_index()
        .rename(columns={"index": "Category"})
    )

    rows = []
    for model_name, sub in model.groupby("model"):
        model_counts = sub["category"].value_counts().reindex(DIST_YN_ORDER, fill_value=0)
        chi2, p, js, tv, cramers_v = _distribution_stats_from_counts(human_counts, model_counts)
        rows.append(
            {
                "Model": model_name,
                "Group": MATCHED_7B_MODEL_GROUP.get(model_name, "Other"),
                "N": int(len(sub)),
                "JS divergence": js,
                "TV distance": tv,
                "Chi-square": chi2,
                "p": p,
                "Cramer's V": cramers_v,
                "Significant": "yes" if p < 0.05 else "no",
                "Yes": float(model_counts.get("yes", 0) / max(len(sub), 1)),
                "No": float(model_counts.get("no", 0) / max(len(sub), 1)),
                "Others": float(model_counts.get("others", 0) / max(len(sub), 1)),
            }
        )
    result = pd.DataFrame(rows).sort_values(["JS divergence", "TV distance", "Model"]).reset_index(drop=True)
    return human_profile, result


def number_distribution_table(condition: str = "inst_blind", variant: str = "C") -> tuple[pd.DataFrame, pd.DataFrame]:
    human, model = distribution_base_table(condition=condition, variant=variant)
    human = human[human["question_id"].astype(int).isin(human.loc[human["question_id"].astype(int).isin(human["question_id"]), "question_id"])].copy()
    human["category"] = human["clean_response"].apply(_classify_number)
    model["category"] = model["clean_response"].apply(_classify_number)

    human_counts = human["category"].value_counts().reindex(DIST_NUM_ORDER, fill_value=0)
    human_profile = (
        _distribution_profile(human, "category", DIST_NUM_ORDER)
        .rename("Proportion")
        .reset_index()
        .rename(columns={"index": "Category"})
    )

    rows = []
    for model_name, sub in model.groupby("model"):
        model_counts = sub["category"].value_counts().reindex(DIST_NUM_ORDER, fill_value=0)
        chi2, p, js, tv, cramers_v = _distribution_stats_from_counts(human_counts, model_counts)
        row = {
            "Model": model_name,
            "Group": MATCHED_7B_MODEL_GROUP.get(model_name, "Other"),
            "N": int(len(sub)),
            "JS divergence": js,
            "TV distance": tv,
            "Chi-square": chi2,
            "p": p,
            "Cramer's V": cramers_v,
            "Significant": "yes" if p < 0.05 else "no",
        }
        for cat in DIST_NUM_ORDER:
            row[cat] = float(model_counts.get(cat, 0) / max(len(sub), 1))
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["JS divergence", "TV distance", "Model"]).reset_index(drop=True)
    return human_profile, result


def _answer_type_subset(df: pd.DataFrame, answer_type: str) -> pd.DataFrame:
    if answer_type == "yes/no":
        return df[df["answer_type"] == "yes/no"].copy()
    if answer_type == "number":
        return df[df["answer_type"] == "number"].copy()
    if answer_type == "text":
        return df[~df["answer_type"].isin(["yes/no", "number"])].copy()
    raise ValueError(f"Unsupported answer_type: {answer_type}")


def _prepare_distribution_category(
    human: pd.DataFrame,
    model: pd.DataFrame,
    answer_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    human = _answer_type_subset(human, answer_type)
    model = _answer_type_subset(model, answer_type)

    if answer_type == "yes/no":
        order = DIST_YN_ORDER
        human["category"] = human["clean_response"].apply(_classify_yesno)
        model["category"] = model["clean_response"].apply(_classify_yesno)
        return human, model, order

    if answer_type == "number":
        order = DIST_NUM_ORDER
        human["category"] = human["clean_response"].apply(_classify_number)
        model["category"] = model["clean_response"].apply(_classify_number)
        return human, model, order

    # free text: build a compact shared space from human-dominant answers
    top_answers = (
        human["clean_response"]
        .value_counts()
        .head(DIST_TEXT_TOP_K)
        .index
        .tolist()
    )
    order = top_answers + ["others"]
    human["category"] = human["clean_response"].where(human["clean_response"].isin(top_answers), "others")
    model["category"] = model["clean_response"].where(model["clean_response"].isin(top_answers), "others")
    return human, model, order


def grouped_distribution_summary_table(
    answer_type: str,
    dimension: str,
    *,
    condition: str = "inst_blind",
    variant: str = "C",
    min_questions_per_slice: int = 3,
) -> pd.DataFrame:
    key = "op" if dimension == "op" else "ent"
    human, model = distribution_base_table(condition=condition, variant=variant)
    human, model, category_order = _prepare_distribution_category(human, model, answer_type)

    # restrict to shared question set within the selected answer type
    qids = set(human["question_id"].astype(int).unique())
    human = human[human["question_id"].astype(int).isin(qids)].copy()
    model = model[model["question_id"].astype(int).isin(qids)].copy()

    human_slice_n = human.groupby(key)["question_id"].nunique()
    valid_slices = human_slice_n[human_slice_n >= min_questions_per_slice].index.tolist()
    human = human[human[key].isin(valid_slices)].copy()
    model = model[model[key].isin(valid_slices)].copy()

    rows = []
    for model_name, sub_model in model.groupby("model"):
        slice_rows = []
        for slice_name in valid_slices:
            h_slice = human[human[key] == slice_name]
            m_slice = sub_model[sub_model[key] == slice_name]
            if h_slice.empty or m_slice.empty:
                continue

            human_counts = h_slice["category"].value_counts().reindex(category_order, fill_value=0)
            model_counts = m_slice["category"].value_counts().reindex(category_order, fill_value=0)
            if human_counts.sum() == 0 or model_counts.sum() == 0:
                continue

            chi2, p, js, tv, cramers_v = _distribution_stats_from_counts(human_counts, model_counts)
            slice_rows.append(
                {
                    "slice": slice_name,
                    "n_questions": int(h_slice["question_id"].nunique()),
                    "n_human": int(len(h_slice)),
                    "n_model": int(len(m_slice)),
                    "js": js,
                    "tv": tv,
                    "chi2": chi2,
                    "p": p,
                    "cramers_v": cramers_v,
                    "sig": float(p < 0.05),
                }
            )

        if not slice_rows:
            continue

        sdf = pd.DataFrame(slice_rows)
        rows.append(
            {
                "Model": model_name,
                "Group": MATCHED_7B_MODEL_GROUP.get(model_name, "Other"),
                "Variant": VARIANT_NAME.get(variant, variant),
                "Answer type": answer_type,
                "Grouping": "Operation" if dimension == "op" else "Entity",
                "Mean JS": float(sdf["js"].mean()),
                "Mean TV": float(sdf["tv"].mean()),
                "Mean Chi-square": float(sdf["chi2"].mean()),
                "Mean p": float(sdf["p"].mean()),
                "Mean Cramer's V": float(sdf["cramers_v"].mean()),
                "% sig slices": float(100.0 * sdf["sig"].mean()),
                "N slices": int(len(sdf)),
                "Questions covered": int(sdf["n_questions"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["Mean JS", "Mean TV", "Model"]).reset_index(drop=True)


def _grouped_slice_metrics(
    human: pd.DataFrame,
    model: pd.DataFrame,
    answer_type: str,
    dimension: str,
    *,
    min_questions_per_slice: int = 3,
) -> pd.DataFrame:
    key = "op" if dimension == "op" else "ent"
    human, model, category_order = _prepare_distribution_category(human, model, answer_type)

    human_slice_n = human.groupby(key)["question_id"].nunique()
    valid_slices = human_slice_n[human_slice_n >= min_questions_per_slice].index.tolist()
    human = human[human[key].isin(valid_slices)].copy()
    model = model[model[key].isin(valid_slices)].copy()

    slice_rows = []
    for slice_name in valid_slices:
        h_slice = human[human[key] == slice_name]
        m_slice = model[model[key] == slice_name]
        if h_slice.empty or m_slice.empty:
            continue
        human_counts = h_slice["category"].value_counts().reindex(category_order, fill_value=0)
        model_counts = m_slice["category"].value_counts().reindex(category_order, fill_value=0)
        if human_counts.sum() == 0 or model_counts.sum() == 0:
            continue
        chi2, p, js, tv, cramers_v = _distribution_stats_from_counts(human_counts, model_counts)
        slice_rows.append(
            {
                "slice": slice_name,
                "n_questions": int(h_slice[["question_id", "variant"]].drop_duplicates().shape[0]),
                "n_human": int(len(h_slice)),
                "n_model": int(len(m_slice)),
                "js": js,
                "tv": tv,
                "chi2": chi2,
                "p": p,
                "cramers_v": cramers_v,
                "sig": float((not np.isnan(p)) and (p < 0.05)),
            }
        )
    return pd.DataFrame(slice_rows)


def grouped_distribution_long_table(
    dimension: str,
    *,
    condition: str = "inst_blind",
    variants: tuple[str, ...] = ("C", "B", "A"),
    filter_abstention: bool = False,
    answer_types: tuple[str, ...] = ("yes/no", "number", "text"),
    min_questions_per_slice: int = 3,
    include_human_baseline: bool = True,
) -> pd.DataFrame:
    human_all, model_all = distribution_base_table_variants(condition=condition, variants=variants)
    human_all = _mark_abstentions(human_all)
    model_all = _mark_abstentions(model_all)

    rows = []
    for model_name, model_sub_all in model_all.groupby("model"):
        for answer_type in answer_types:
            model_sub = _answer_type_subset(model_sub_all, answer_type)
            human_sub = _answer_type_subset(human_all, answer_type)
            abst_rate = float(100.0 * model_sub["is_abstained"].mean()) if len(model_sub) else float("nan")

            if filter_abstention:
                model_sub = model_sub[~model_sub["is_abstained"]].copy()
                kept_qvs = set(_qv_pairs(model_sub))
                human_sub = human_sub[[qv in kept_qvs for qv in _qv_pairs(human_sub)]].copy()
                human_sub = human_sub[~human_sub["is_abstained"]].copy()

            slice_df = _grouped_slice_metrics(
                human_sub,
                model_sub,
                answer_type,
                dimension,
                min_questions_per_slice=min_questions_per_slice,
            )
            if slice_df.empty:
                continue

            rows.append(
                {
                    "Model": model_name,
                    "Group": MATCHED_7B_MODEL_GROUP.get(model_name, "Other"),
                    "Answer type": answer_type,
                    "Grouping": "Operation" if dimension == "op" else "Entity",
                    "Variants": "C+B+A",
                    "Filtered": "yes" if filter_abstention else "no",
                    "Mean JS": float(slice_df["js"].mean()),
                    "Mean TV": float(slice_df["tv"].mean()),
                    "Mean Chi-square": float(slice_df["chi2"].mean()),
                    "Mean p": float(slice_df["p"].mean()),
                    "Mean Cramer's V": float(slice_df["cramers_v"].mean()),
                    "% sig slices": float(100.0 * slice_df["sig"].mean()),
                    "N slices": int(len(slice_df)),
                    "Questions covered": int(slice_df["n_questions"].sum()),
                    "Model abstention %": abst_rate,
                }
            )

    if include_human_baseline:
        for answer_type in answer_types:
            human_type = _answer_type_subset(human_all, answer_type)
            if filter_abstention:
                human_type = human_type[~human_type["is_abstained"]].copy()
            base_rows = []
            for participant, sub_all in human_type.groupby("participant"):
                other = human_type[human_type["participant"] != participant].copy()
                self_df = sub_all.copy()
                slice_df = _grouped_slice_metrics(
                    other,
                    self_df,
                    answer_type,
                    dimension,
                    min_questions_per_slice=min_questions_per_slice,
                )
                if slice_df.empty:
                    continue
                base_rows.append(
                    {
                        "Mean JS": float(slice_df["js"].mean()),
                        "Mean TV": float(slice_df["tv"].mean()),
                        "Mean Chi-square": float(slice_df["chi2"].mean()),
                        "Mean p": float(slice_df["p"].mean()),
                        "Mean Cramer's V": float(slice_df["cramers_v"].mean()),
                        "% sig slices": float(100.0 * slice_df["sig"].mean()),
                        "N slices": int(len(slice_df)),
                        "Questions covered": int(slice_df["n_questions"].sum()),
                    }
                )
            if base_rows:
                bdf = pd.DataFrame(base_rows)
                rows.append(
                    {
                        "Model": "Human LOO",
                        "Group": "Human baseline",
                        "Answer type": answer_type,
                        "Grouping": "Operation" if dimension == "op" else "Entity",
                        "Variants": "C+B+A",
                        "Filtered": "yes" if filter_abstention else "no",
                        "Mean JS": float(bdf["Mean JS"].mean()),
                        "Mean TV": float(bdf["Mean TV"].mean()),
                        "Mean Chi-square": float(bdf["Mean Chi-square"].mean()),
                        "Mean p": float(bdf["Mean p"].mean()),
                        "Mean Cramer's V": float(bdf["Mean Cramer's V"].mean()),
                        "% sig slices": float(bdf["% sig slices"].mean()),
                        "N slices": int(round(bdf["N slices"].mean())),
                        "Questions covered": int(round(bdf["Questions covered"].mean())),
                        "Model abstention %": float(100.0 * human_type["is_abstained"].mean()) if len(human_type) else 0.0,
                    }
                )

    return pd.DataFrame(rows).sort_values(["Answer type", "Mean JS", "Mean TV", "Model"]).reset_index(drop=True)


def grouped_distribution_wide_table(
    dimension: str,
    *,
    condition: str = "inst_blind",
    variants: tuple[str, ...] = ("C", "B", "A"),
    filter_abstention: bool = False,
    answer_types: tuple[str, ...] = ("yes/no", "number", "text"),
    min_questions_per_slice: int = 3,
    include_human_baseline: bool = True,
) -> pd.DataFrame:
    long_df = grouped_distribution_long_table(
        dimension,
        condition=condition,
        variants=variants,
        filter_abstention=filter_abstention,
        answer_types=answer_types,
        min_questions_per_slice=min_questions_per_slice,
        include_human_baseline=include_human_baseline,
    )
    metrics = ["Mean JS", "Mean TV", "Mean Cramer's V", "% sig slices", "Model abstention %"]
    wide = (
        long_df.pivot_table(
            index=["Model", "Group"],
            columns="Answer type",
            values=metrics,
            aggfunc="first",
        )
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1, level=[0, 1])
    )
    wide.columns = [f"{atype} | {metric}" for atype, metric in wide.columns]
    wide = wide.reset_index()
    return wide


def grouped_distribution_js_heatmap(
    *,
    condition: str = "inst_blind",
    variants: tuple[str, ...] = ("C", "B", "A"),
    filter_abstention: bool = False,
    answer_types: tuple[str, ...] = ("yes/no", "number", "text"),
    min_questions_per_slice: int = 3,
    include_human_baseline: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact JS-divergence matrices for operation/entity heatmaps."""
    outputs: list[pd.DataFrame] = []
    group_rank = {"Human baseline": 0, "Backbone Decoder": 1, "VLM": 2, "Standalone LLM": 3, "Other": 9}

    for dimension in ("op", "ent"):
        long_df = grouped_distribution_long_table(
            dimension,
            condition=condition,
            variants=variants,
            filter_abstention=filter_abstention,
            answer_types=answer_types,
            min_questions_per_slice=min_questions_per_slice,
            include_human_baseline=include_human_baseline,
        )
        mat = (
            long_df.pivot_table(
                index=["Model", "Group"],
                columns="Answer type",
                values="Mean JS",
                aggfunc="first",
            )
            .reindex(columns=[a for a in answer_types if a in long_df["Answer type"].unique()])
        )
        summary = mat.mean(axis=1).rename("Mean JS overall").reset_index()
        summary["group_rank"] = summary["Group"].map(group_rank).fillna(8)
        summary = summary.sort_values(["group_rank", "Mean JS overall", "Model"])
        order = pd.MultiIndex.from_frame(summary[["Model", "Group"]])
        mat = mat.reindex(order)
        mat.index = [f"{m} [{g}]" for m, g in mat.index]
        outputs.append(mat)

    return outputs[0], outputs[1]
