"""
Generate appendix example tables showing human answer distributions
and per-class model answers for representative questions from each
operation group and entity group.

Two tables produced:
  - tables/examples_op_groups.tex    (one row per op group)
  - tables/examples_ent_groups.tex   (one row per entity group)
"""
from pathlib import Path
import sys
import textwrap
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "latex/AAAI2026/LaTeX/tables"

EXPORTS = ROOT / "analysis/session2/exports"

VLM = ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"]
BD  = ["InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)",
       "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"]
SA  = ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B"]

# ── question registry ──────────────────────────────────────────────────────────
OP_QUESTIONS = [
    dict(qid=288202001,  group="Identity",     label=r"\textit{Identity}"),
    dict(qid=96618004,   group="Attribute",    label=r"\textit{Attribute}"),
    dict(qid=271934000,  group="World Know.",  label=r"\textit{World K.}"),
    dict(qid=33994009,   group="Spatial",      label=r"\textit{Spatial}"),
    dict(qid=140713012,  group="Action",       label=r"\textit{Action}"),
    dict(qid=537206001,  group="Causality",    label=r"\textit{Causality}"),
    dict(qid=229391001,  group="Comparison",   label=r"\textit{Comparison}"),
    dict(qid=267690002,  group="Temporal",     label=r"\textit{Temporal}"),
    dict(qid=226154000,  group="Text Reading", label=r"\textit{Text R.}"),
]

ENT_QUESTIONS = [
    dict(qid=394002002,  group="Product",  label=r"\textit{Product}"),
    dict(qid=315905008,  group="Animal",   label=r"\textit{Animal}"),
    dict(qid=209856000,  group="Food",     label=r"\textit{Food}"),
    dict(qid=187196001,  group="Object",   label=r"\textit{Object}"),
    dict(qid=169996009,  group="Place",    label=r"\textit{Place}"),
    dict(qid=213359003,  group="Person",   label=r"\textit{Person}"),
    dict(qid=91236000,   group="Vehicle",  label=r"\textit{Vehicle}"),
    dict(qid=1342072,    group="Other",    label=r"\textit{Other}"),
    dict(qid=226154000,  group="Text",     label=r"\textit{Text}"),
]


def latex_escape(s):
    s = str(s)
    for ch, rep in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                    ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                    ("^", r"\textasciicircum{}"), ("\\", r"\textbackslash{}")]:
        s = s.replace(ch, rep)
    return s


def truncate(s, n=52):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def human_cell(hh_sub, qid, top_n=3):
    """Return LaTeX cell with top-N human answers and their % (sorted desc)."""
    rows = hh_sub[hh_sub["question_id"] == qid]
    ans = pd.concat([rows["answer_1"], rows["answer_2"]]).str.lower().str.strip()
    vc = ans.value_counts()
    total = len(ans)
    parts = []
    for val, cnt in vc.head(top_n).items():
        pct = round(100 * cnt / total)
        parts.append(rf"\textit{{{latex_escape(val)}}} ({pct}\%)")
    return r"\makecell[l]{" + r" \\ ".join(parts) + "}"


def model_cell(hm_sub, qid, models, top_n=2):
    """Return LaTeX cell with top-N model answers (most common across class)."""
    rows = hm_sub[(hm_sub["question_id"] == qid) & hm_sub["subject_2"].isin(models)]
    if rows.empty:
        return "---"
    vc = rows["answer_2"].str.lower().str.strip().value_counts()
    parts = []
    for val, _ in vc.head(top_n).items():
        val_clean = latex_escape(truncate(val, 30))
        parts.append(rf"\textit{{{val_clean}}}")
    return r"\makecell[l]{" + r" \\ ".join(parts) + "}"


def make_table(questions, pc, label, caption, filename, dim_label):
    hh_c = pc[(pc["pair_type"] == "HH") & (pc["variant"] == "C")]
    hm_c = pc[(pc["pair_type"] == "HM") & (pc["variant"] == "C")]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.0cm}"
                 r">{\raggedright\arraybackslash}p{4.4cm}"
                 r">{\raggedright\arraybackslash}p{3.6cm}"
                 r">{\raggedright\arraybackslash}p{2.2cm}"
                 r">{\raggedright\arraybackslash}p{2.2cm}"
                 r">{\raggedright\arraybackslash}p{2.2cm}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Group} & \textbf{Question} & "
                 r"\textbf{Human answers} & \textbf{VLM} & \textbf{Backbone} & \textbf{SA-LLM} \\")
    lines.append(r"\midrule")

    for i, q in enumerate(questions):
        qid = q["qid"]
        grp_cell = q["label"]
        # question text
        q_text = hh_c[hh_c["question_id"] == qid]["question_en"].iloc[0]
        q_cell = latex_escape(q_text)

        h_cell   = human_cell(hh_c, qid, top_n=3)
        vlm_cell = model_cell(hm_c, qid, VLM)
        bd_cell  = model_cell(hm_c, qid, BD)
        sa_cell  = model_cell(hm_c, qid, SA)

        lines.append(rf"{grp_cell} & {q_cell} & {h_cell} & {vlm_cell} & {bd_cell} & {sa_cell} \\")
        if i < len(questions) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = TABLE_DIR / filename
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


def main():
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")

    make_table(
        OP_QUESTIONS, pc,
        label="tab:examples_op_groups",
        caption=(
            r"Representative questions for each operation group with human answer "
            r"distributions (top 3, sorted by frequency) and the most common responses "
            r"per architecture class (VLM / Backbone decoder / SA-LLM). "
            r"Human answers reflect the \emph{original} (non-pronominalized) variant; "
            r"model answers are from the same variant under the instruction-blind condition."
        ),
        filename="examples_op_groups.tex",
        dim_label="op",
    )

    make_table(
        ENT_QUESTIONS, pc,
        label="tab:examples_ent_groups",
        caption=(
            r"Representative questions for each entity group with human answer "
            r"distributions (top 3, sorted by frequency) and the most common responses "
            r"per architecture class. Same format as Table~\ref{tab:examples_op_groups}."
        ),
        filename="examples_ent_groups.tex",
        dim_label="ent",
    )


if __name__ == "__main__":
    main()
