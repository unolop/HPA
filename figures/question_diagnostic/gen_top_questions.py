"""Generate LaTeX tables: top representative questions per operation and entity group."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "analysis/session2/exports"
TABLES_DIR = ROOT / "latex/AAAI2026/LaTeX/tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
hh = pc[pc["pair_type"] == "HH"]
hm = pc[pc["pair_type"] == "HM"]

models_7b = [
    "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
    "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
]

hh_q = hh.groupby(["question_id", "question_en", "variant", "op", "ent"]).agg(
    hh_sbert=("sbert_score", "mean")).reset_index()

hm_sub = hm[hm["subject_2"].isin(models_7b)]
hm_q = hm_sub.groupby(["question_id", "variant"]).agg(
    hm_sbert=("sbert_score", "mean")).reset_index()

merged = hh_q.merge(hm_q, on=["question_id", "variant"])
vC = merged[merged["variant"] == "C"].copy()
vA = merged[merged["variant"] == "A"][["question_id", "hh_sbert", "hm_sbert"]].rename(
    columns={"hh_sbert": "hh_A", "hm_sbert": "hm_A"})
vC = vC.merge(vA, on="question_id", how="left")
vC["hh_drop"] = vC["hh_sbert"] - vC["hh_A"]
vC["hm_drop"] = vC["hm_sbert"] - vC["hm_A"]

human_resp = pd.read_csv(EXPORTS / "responses_human.csv")


def get_top_answers(qid, variant="C", n=3):
    sub = human_resp[(human_resp["question_id"] == qid) & (human_resp["variant"] == variant)]
    counts = sub["response"].value_counts().head(n)
    return ", ".join([f"{a} ({c})" for a, c in counts.items()])


def escape_tex(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def gen_table(group_col, group_label, n_per_group=2):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Representative high-agreement questions by " + group_label.lower() +
        r" type (original formulation, top " + str(n_per_group) +
        r" per group by HH SBERT). $\Delta_{C \to A}$: degradation from original to pronominalized."
        r" Human top answers show the most frequent blind responses with counts in parentheses.}")
    lines.append(r"\label{tab:top_questions_" + group_col + "}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrrrp{5.5cm}}")
    lines.append(r"\toprule")
    lines.append(
        group_label + r" & Question & HH & HM & $\Delta_{C \to A}^{HH}$ & $\Delta_{C \to A}^{HM}$ & Human top answers \\")
    lines.append(r"\midrule")

    groups = sorted(vC[group_col].unique())
    for i, grp in enumerate(groups):
        sub = vC[vC[group_col] == grp].nlargest(n_per_group, "hh_sbert")
        if i > 0:
            lines.append(r"\cdashline{1-7}")
        first = True
        for _, row in sub.iterrows():
            q = row["question_en"]
            if len(q) > 55:
                q = q[:52] + "..."
            q = escape_tex(q)
            top_ans = get_top_answers(row["question_id"])
            top_ans = escape_tex(top_ans)
            if len(top_ans) > 65:
                top_ans = top_ans[:62] + "..."
            grp_disp = grp if first else ""
            first = False
            lines.append(
                f"{grp_disp} & {q} & {row['hh_sbert']:.3f} & {row['hm_sbert']:.3f} & "
                f"{row['hh_drop']:+.3f} & {row['hm_drop']:+.3f} & {top_ans} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


op_tex = gen_table("op", "Operation")
ent_tex = gen_table("ent", "Entity")

(TABLES_DIR / "top_questions_op.tex").write_text(op_tex)
(TABLES_DIR / "top_questions_ent.tex").write_text(ent_tex)
print(f"Saved {TABLES_DIR / 'top_questions_op.tex'}")
print(f"Saved {TABLES_DIR / 'top_questions_ent.tex'}")

# Also print summary
print(f"\nOperation groups: {sorted(vC['op'].unique())}")
print(f"Entity groups: {sorted(vC['ent'].unique())}")
for col, label in [("op", "Operation"), ("ent", "Entity")]:
    print(f"\n--- {label} ---")
    for grp in sorted(vC[col].unique()):
        sub = vC[vC[col] == grp]
        print(f"  {grp:>8}: n={len(sub):>2}, HH={sub['hh_sbert'].mean():.3f}, HM={sub['hm_sbert'].mean():.3f}")
