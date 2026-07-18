from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def write_nb(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2))


COMMON_IMPORT = """from pathlib import Path
import sys
import pandas as pd
from IPython.display import display

_NOTEBOOK_DIR = Path.cwd()
if (_NOTEBOOK_DIR / "helpers.py").exists():
    sys.path.insert(0, str(_NOTEBOOK_DIR.parent))
elif (_NOTEBOOK_DIR / "notebooks" / "helpers.py").exists():
    sys.path.insert(0, str(_NOTEBOOK_DIR))

from notebooks.helpers import (
    ROOT,
    LATEX_TABLES,
    pretty_print_path,
    hh_question_means,
    hm_question_means,
    variant_summary_table,
    pairwise_variant_correlation_table,
    grouped_pattern_table,
    scatter_correlation_table,
    blind_accuracy_summary,
    qualitative_qdf,
    attach_answer_summaries,
    hh_ranked_examples,
    variant_top_bottom_table,
    hh_degradation_table,
    top_questions_tables,
    to_latex_table,
)
"""


def build() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    write_nb(
        NOTEBOOK_DIR / "1_HH.ipynb",
        [
            md_cell("# 1_HH\n\nHuman-only blind-prior analysis. This notebook computes per-variant HH SBERT summaries, pairwise cross-variant HH correlations, and exports LaTeX tables for the paper."),
            code_cell(COMMON_IMPORT),
            code_cell(
                """hh_q = hh_question_means()
hh_summary = variant_summary_table(hh_q)
hh_summary"""
            ),
            code_cell(
                """display(hh_summary)
out = LATEX_TABLES / "hh_variant_summary.tex"
to_latex_table(
    hh_summary,
    out,
    "Per-variant HH SBERT summary over the 113-question human subset.",
    "tab:hh_variant_summary",
    float_formatters={"Mean HH SBERT": ".3f", "Std HH SBERT": ".3f"},
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """pairwise = pairwise_variant_correlation_table(hh_q)
pairwise"""
            ),
            code_cell(
                """display(pairwise)
out = LATEX_TABLES / "hh_variant_pairwise_correlations.tex"
to_latex_table(
    pairwise,
    out,
    "Pairwise question-level correlations of HH SBERT across control variants.",
    "tab:hh_variant_pairwise_corr",
    float_formatters={"Pearson r": ".3f", "p": ".1e"},
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """ranked, human = hh_ranked_examples()
ranked.head()"""
            ),
            code_cell(
                """top_bottom_c = variant_top_bottom_table(ranked, human, "C", top_n=10)
top_bottom_c.head()"""
            ),
            code_cell(
                """display(top_bottom_c)
out = LATEX_TABLES / "hh_variant_C_top_bottom_examples.tex"
to_latex_table(
    top_bottom_c,
    out,
    "Top and bottom HH SBERT questions for the Original variant, with compact human answer summaries.",
    "tab:hh_variant_c_examples",
    float_formatters={"HH SBERT": ".3f"},
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """deg = hh_degradation_table(ranked, human, top_n=10)
deg"""
            ),
            code_cell(
                """display(deg)
out = LATEX_TABLES / "hh_degradation_examples.tex"
to_latex_table(
    deg,
    out,
    "Top question-level HH SBERT degradation cases from Original to Pronominalized, with compact human answer summaries.",
    "tab:hh_degradation_examples",
    float_formatters={"Delta C->A": ".3f"},
)
print(pretty_print_path(out))"""
            ),
        ],
    )

    write_nb(
        NOTEBOOK_DIR / "2_HM.ipynb",
        [
            md_cell("# 2_HM\n\nHuman--model aggregate summaries. This notebook reports blind accuracy and HM SBERT summaries and exports compact LaTeX tables."),
            code_cell(COMMON_IMPORT),
            code_cell(
                """acc_c = blind_accuracy_summary("C")
acc_c.head(10)"""
            ),
            code_cell(
                """display(acc_c)
out = LATEX_TABLES / "hm_blind_accuracy_variant_C.tex"
to_latex_table(
    acc_c,
    out,
    "Instruction-only blind accuracy by model on the Original variant.",
    "tab:hm_blind_accuracy_c",
    float_formatters={"Blind accuracy": ".3f"},
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """hm_q = hm_question_means(matched_only=False)
hm_summary = (
    hm_q[hm_q["variant"] == "C"]
    .groupby("model", as_index=False)["hm_sbert"]
    .mean()
    .sort_values("hm_sbert", ascending=False)
    .rename(columns={"hm_sbert": "HM SBERT"})
)
hm_summary.head(10)"""
            ),
            code_cell(
                """display(hm_summary)
out = LATEX_TABLES / "hm_sbert_variant_C.tex"
to_latex_table(
    hm_summary,
    out,
    "Mean HM SBERT by model on the Original variant.",
    "tab:hm_sbert_c",
    float_formatters={"HM SBERT": ".3f"},
)
print(pretty_print_path(out))"""
            ),
        ],
    )

    write_nb(
        NOTEBOOK_DIR / "2b_HM_answer_distribution.ipynb",
        [
            md_cell("# 2b_HM_yesno_distribution_stats\n\nYes/no distribution statistics for the matched HM comparison set across all three control variants. Plot generation lives in `figures/answer_distribution/answer_distribution.py`; this notebook is statistical-only."),
            code_cell(COMMON_IMPORT + "\nfrom notebooks.helpers import yesno_distribution_table\n"),
            code_cell(
                """VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}

human_yesno_profiles = {}
yesno_inst_tables = []
for variant in ["C", "B", "A"]:
    human_profile, table = yesno_distribution_table(condition="inst_blind", variant=variant)
    human_yesno_profiles[variant] = human_profile
    yesno_inst_tables.append(table.assign(Variant=VARIANT_LABELS[variant]))

yesno_inst = pd.concat(yesno_inst_tables, ignore_index=True)
for variant in ["C", "B", "A"]:
    print(VARIANT_LABELS[variant])
    display(human_yesno_profiles[variant])
    display(yesno_inst[yesno_inst["Variant"] == VARIANT_LABELS[variant]])"""
            ),
            code_cell(
                """out = LATEX_TABLES / "hm_yesno_distribution_inst_blind.tex"
to_latex_table(
    yesno_inst,
    out,
    "Yes/no answer-distribution statistics against the human reference across control variants (instruction-aware blind condition).",
    "tab:hm_yesno_distribution_inst_blind",
    float_formatters={"JS divergence": ".3f", "TV distance": ".3f", "Chi-square": ".2f", "p": ".1e", "Yes": ".3f", "No": ".3f", "Others": ".3f"},
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """yesno_blind_tables = []
for variant in ["C", "B", "A"]:
    _, table = yesno_distribution_table(condition="blind", variant=variant)
    yesno_blind_tables.append(table.assign(Variant=VARIANT_LABELS[variant]))

yesno_blind = pd.concat(yesno_blind_tables, ignore_index=True)
display(yesno_blind)"""
            ),
            code_cell(
                """ranked_yesno = yesno_inst.sort_values(["Variant", "JS divergence", "TV distance", "Model"]).reset_index(drop=True)
for variant in ["Original", "Weaker", "Pronominalized"]:
    sub = ranked_yesno[ranked_yesno["Variant"] == variant]
    best_js = sub.iloc[0][["Variant", "Model", "Group", "JS divergence", "TV distance", "Chi-square", "p", "Significant"]]
    worst_js = sub.iloc[-1][["Variant", "Model", "Group", "JS divergence", "TV distance", "Chi-square", "p", "Significant"]]
    print(f"Closest yes/no distribution to humans ({variant}, inst_blind):")
    display(best_js.to_frame().T)
    print(f"Farthest yes/no distribution from humans ({variant}, inst_blind):")
    display(worst_js.to_frame().T)

print("Interpretation: models are ranked by closeness to the human yes/no distribution using JS divergence and TV distance. The chi-square test is also reported as a significance test for whether a model's aggregate category counts differ from the human distribution.")"""
            ),
        ],
    )

    write_nb(
        NOTEBOOK_DIR / "2c_HM_number_distribution_stats.ipynb",
        [
            md_cell("# 2c_HM_number_distribution_stats\n\nCount-answer distribution statistics for the matched HM comparison set across all three control variants. Plot generation lives in `figures/answer_distribution/answer_distribution.py`; this notebook is statistical-only."),
            code_cell(COMMON_IMPORT + "\nfrom notebooks.helpers import number_distribution_table\n"),
            code_cell(
                """VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}

human_number_profiles = {}
number_inst_tables = []
for variant in ["C", "B", "A"]:
    human_profile, table = number_distribution_table(condition="inst_blind", variant=variant)
    human_number_profiles[variant] = human_profile
    number_inst_tables.append(table.assign(Variant=VARIANT_LABELS[variant]))

number_inst = pd.concat(number_inst_tables, ignore_index=True)
for variant in ["C", "B", "A"]:
    print(VARIANT_LABELS[variant])
    display(human_number_profiles[variant])
    display(number_inst[number_inst["Variant"] == VARIANT_LABELS[variant]])"""
            ),
            code_cell(
                """out = LATEX_TABLES / "hm_number_distribution_inst_blind.tex"
to_latex_table(
    number_inst,
    out,
    "Count-answer distribution statistics against the human reference across control variants (instruction-aware blind condition).",
    "tab:hm_number_distribution_inst_blind",
    float_formatters={"JS divergence": ".3f", "TV distance": ".3f", "Chi-square": ".2f", "p": ".1e", "0": ".3f", "1": ".3f", "2–3": ".3f", "4–5": ".3f", "6–10": ".3f", "11–20": ".3f", ">20": ".3f", "others": ".3f"},
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """number_blind_tables = []
for variant in ["C", "B", "A"]:
    _, table = number_distribution_table(condition="blind", variant=variant)
    number_blind_tables.append(table.assign(Variant=VARIANT_LABELS[variant]))

number_blind = pd.concat(number_blind_tables, ignore_index=True)
display(number_blind)"""
            ),
            code_cell(
                """ranked_number = number_inst.sort_values(["Variant", "JS divergence", "TV distance", "Model"]).reset_index(drop=True)
for variant in ["Original", "Weaker", "Pronominalized"]:
    sub = ranked_number[ranked_number["Variant"] == variant]
    best_num = sub.iloc[0][["Variant", "Model", "Group", "JS divergence", "TV distance", "Chi-square", "p", "Significant"]]
    worst_num = sub.iloc[-1][["Variant", "Model", "Group", "JS divergence", "TV distance", "Chi-square", "p", "Significant"]]
    print(f"Closest count distribution to humans ({variant}, inst_blind):")
    display(best_num.to_frame().T)
    print(f"Farthest count distribution from humans ({variant}, inst_blind):")
    display(worst_num.to_frame().T)

print("Interpretation: models are ranked by closeness to the human count distribution using JS divergence and TV distance, with chi-square reported as the significance test on aggregate category counts.")"""
            ),
        ],
    )

    write_nb(
        NOTEBOOK_DIR / "2d_HM_grouped_distribution_7b.ipynb",
        [
            md_cell("# 2d_HM_grouped_distribution_7b\n\nGrouped pooled-distribution summaries for the matched 7/8B model set, pooled across all three control variants. Each row is a model; within each operation/entity slice we compare the model's pooled answer distribution against the human pooled distribution, then average those slice-level metrics. Both inclusive and abstention-filtered versions are exported."),
            code_cell(COMMON_IMPORT + "\nfrom notebooks.helpers import grouped_distribution_long_table, grouped_distribution_wide_table\n"),
            code_cell(
                """op_wide = grouped_distribution_wide_table(
    "op",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=False,
)
op_wide_filt = grouped_distribution_wide_table(
    "op",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=True,
)
ent_wide = grouped_distribution_wide_table(
    "ent",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=False,
)
ent_wide_filt = grouped_distribution_wide_table(
    "ent",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=True,
)

display(op_wide)
display(op_wide_filt)
display(ent_wide)
display(ent_wide_filt)"""
            ),
            code_cell(
                """out = LATEX_TABLES / "hm_grouped_distribution_7b_op_all_variants.tex"
to_latex_table(
    op_wide,
    out,
    "Matched 7/8B operation-group pooled-distribution alignment to humans across all control variants (inclusive version). Lower JS/TV indicate closer alignment; Cramer's V summarizes residual distribution mismatch; abstention rate is measured before any filtering.",
    "tab:hm_grouped_distribution_7b_op",
    float_formatters={col: ".3f" for col in op_wide.columns if "|" in col},
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """out = LATEX_TABLES / "hm_grouped_distribution_7b_op_all_variants_filtered.tex"
to_latex_table(
    op_wide_filt,
    out,
    "Matched 7/8B operation-group pooled-distribution alignment to humans across all control variants (shared substantive-response version). Lower JS/TV indicate closer alignment after removing abstentions and comparing on the shared non-abstaining subset.",
    "tab:hm_grouped_distribution_7b_op_filtered",
    float_formatters={col: ".3f" for col in op_wide_filt.columns if "|" in col},
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """out = LATEX_TABLES / "hm_grouped_distribution_7b_ent_all_variants.tex"
to_latex_table(
    ent_wide,
    out,
    "Matched 7/8B entity-group pooled-distribution alignment to humans across all control variants (inclusive version). Lower JS/TV indicate closer alignment; Cramer's V summarizes residual distribution mismatch; abstention rate is measured before any filtering.",
    "tab:hm_grouped_distribution_7b_ent",
    float_formatters={col: ".3f" for col in ent_wide.columns if "|" in col},
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """out = LATEX_TABLES / "hm_grouped_distribution_7b_ent_all_variants_filtered.tex"
to_latex_table(
    ent_wide_filt,
    out,
    "Matched 7/8B entity-group pooled-distribution alignment to humans across all control variants (shared substantive-response version). Lower JS/TV indicate closer alignment after removing abstentions and comparing on the shared non-abstaining subset.",
    "tab:hm_grouped_distribution_7b_ent_filtered",
    float_formatters={col: ".3f" for col in ent_wide_filt.columns if "|" in col},
)
print(pretty_print_path(out))"""
            ),
            md_cell("## Relation to HH--HM Scatter\n\nThe HH--HM scatterplots ask whether a model preserves the *question-level* human agreement structure. The grouped pooled-distribution tables ask a different but complementary question: within broad operation/entity slices, does the model place answer mass on the same categories that humans do? A model can be close to the HH--HM diagonal yet still show a non-human pooled distribution within count or yes/no slices, so the two views should be read together."),
            code_cell(
                """for label, table in [
    ("Operation, inclusive", op_wide),
    ("Operation, filtered", op_wide_filt),
    ("Entity, inclusive", ent_wide),
    ("Entity, filtered", ent_wide_filt),
]:
    print(label)
    winners = table.iloc[0][["Model", "Group"] + [c for c in table.columns if c.endswith("Mean JS")][:3]]
    display(winners.to_frame().T)"""
            ),
        ],
    )

    write_nb(
        NOTEBOOK_DIR / "2e_alignment_tables.ipynb",
        [
            md_cell("# 2e_alignment_tables\n\nConsolidated alignment tables for the matched 7/8B set. This notebook brings together the per-question HH--HM correlation table, grouped HH--HM profile correlations, and grouped pooled-distribution alignment tables (inclusive and abstention-filtered) in one place."),
            code_cell(COMMON_IMPORT + "\nfrom notebooks.helpers import scatter_correlation_table, grouped_pattern_table, grouped_distribution_wide_table, grouped_distribution_long_table, MATCHED_7B_MODEL_GROUP\n"),
            md_cell("## Per-question alignment\n\nThese tables summarize question-level alignment between human--human agreement and human--model agreement."),
            code_cell(
                """scatter = scatter_correlation_table()
display(scatter)"""
            ),
            code_cell(
                """ALIGN_OUT = NOTEBOOK_DIR / "exports" / "alignment_tables"
ALIGN_OUT.mkdir(parents=True, exist_ok=True)

scatter.to_csv(ALIGN_OUT / "per_question_scatter_correlation_7b.csv", index=False)
print(pretty_print_path(ALIGN_OUT / "per_question_scatter_correlation_7b.csv"))"""
            ),
            md_cell("## Grouped profile correlations\n\nThese tables ask whether models preserve the same operation-level or entity-level ordering that humans show."),
            code_cell(
                """op_corr = grouped_pattern_table("op")
ent_corr = grouped_pattern_table("ent")
display(op_corr)
display(ent_corr)"""
            ),
            code_cell(
                """op_corr.to_csv(ALIGN_OUT / "group_profile_correlation_op_7b.csv", index=False)
ent_corr.to_csv(ALIGN_OUT / "group_profile_correlation_ent_7b.csv", index=False)
print(pretty_print_path(ALIGN_OUT / "group_profile_correlation_op_7b.csv"))
print(pretty_print_path(ALIGN_OUT / "group_profile_correlation_ent_7b.csv"))"""
            ),
            md_cell("## Grouped pooled-distribution alignment\n\nThese tables compare pooled answer distributions to the human grouped distributions across all three variants. The inclusive version keeps all answers; the filtered version removes abstentions and compares on the shared substantive subset."),
            code_cell(
                """op_wide = grouped_distribution_wide_table(
    "op",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=False,
)
op_wide_filt = grouped_distribution_wide_table(
    "op",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=True,
)
ent_wide = grouped_distribution_wide_table(
    "ent",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=False,
)
ent_wide_filt = grouped_distribution_wide_table(
    "ent",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=True,
)

drop_cols = [c for c in ["Variants", "Questions covered"] if c in op_wide.columns]
display(op_wide.drop(columns=drop_cols, errors="ignore"))
display(op_wide_filt.drop(columns=drop_cols, errors="ignore"))
display(ent_wide.drop(columns=drop_cols, errors="ignore"))
display(ent_wide_filt.drop(columns=drop_cols, errors="ignore"))"""
            ),
            code_cell(
                """op_wide.to_csv(ALIGN_OUT / "grouped_distribution_op_7b_all_variants.csv", index=False)
op_wide_filt.to_csv(ALIGN_OUT / "grouped_distribution_op_7b_all_variants_filtered.csv", index=False)
ent_wide.to_csv(ALIGN_OUT / "grouped_distribution_ent_7b_all_variants.csv", index=False)
ent_wide_filt.to_csv(ALIGN_OUT / "grouped_distribution_ent_7b_all_variants_filtered.csv", index=False)
print(pretty_print_path(ALIGN_OUT / "grouped_distribution_op_7b_all_variants.csv"))
print(pretty_print_path(ALIGN_OUT / "grouped_distribution_op_7b_all_variants_filtered.csv"))
print(pretty_print_path(ALIGN_OUT / "grouped_distribution_ent_7b_all_variants.csv"))
print(pretty_print_path(ALIGN_OUT / "grouped_distribution_ent_7b_all_variants_filtered.csv"))"""
            ),
            md_cell("## Long-form grouped-distribution tables\n\nThese keep one row per model and answer type, with the full structural-alignment metrics (`JS`, `TV`, `chi-square`, `p`, `Cramer's V`, `% significant slices`)."),
            code_cell(
                """op_long = grouped_distribution_long_table(
    "op",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=False,
)
op_long_filt = grouped_distribution_long_table(
    "op",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=True,
)
ent_long = grouped_distribution_long_table(
    "ent",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=False,
)
ent_long_filt = grouped_distribution_long_table(
    "ent",
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=True,
)

drop_cols = [c for c in ["Variants", "Questions covered"] if c in op_long.columns]
display(op_long.drop(columns=drop_cols, errors="ignore"))
display(op_long_filt.drop(columns=drop_cols, errors="ignore"))
display(ent_long.drop(columns=drop_cols, errors="ignore"))
display(ent_long_filt.drop(columns=drop_cols, errors="ignore"))"""
            ),
            code_cell(
                """op_long.to_csv(ALIGN_OUT / "grouped_distribution_op_7b_all_variants_long.csv", index=False)
op_long_filt.to_csv(ALIGN_OUT / "grouped_distribution_op_7b_all_variants_filtered_long.csv", index=False)
ent_long.to_csv(ALIGN_OUT / "grouped_distribution_ent_7b_all_variants_long.csv", index=False)
ent_long_filt.to_csv(ALIGN_OUT / "grouped_distribution_ent_7b_all_variants_filtered_long.csv", index=False)
print(pretty_print_path(ALIGN_OUT / "grouped_distribution_op_7b_all_variants_long.csv"))
print(pretty_print_path(ALIGN_OUT / "grouped_distribution_op_7b_all_variants_filtered_long.csv"))
print(pretty_print_path(ALIGN_OUT / "grouped_distribution_ent_7b_all_variants_long.csv"))
print(pretty_print_path(ALIGN_OUT / "grouped_distribution_ent_7b_all_variants_filtered_long.csv"))"""
            ),
            md_cell("## Quick winners\n\nA compact sanity check for the strongest models under the main correlation and structural-alignment views."),
            code_cell(
                """summary = pd.DataFrame([
    {
        "Claim": "Best per-question correlation",
        "Model": scatter.iloc[0]["Model"],
        "Value": round(float(scatter.iloc[0]["All r"]), 3),
    },
    {
        "Claim": "Best operation-group rho",
        "Model": op_corr.sort_values("Spearman rho", ascending=False).iloc[0]["Model"],
        "Value": round(float(op_corr["Spearman rho"].max()), 3),
    },
    {
        "Claim": "Best entity-group rho",
        "Model": ent_corr.sort_values("Spearman rho", ascending=False).iloc[0]["Model"],
        "Value": round(float(ent_corr["Spearman rho"].max()), 3),
    },
    {
        "Claim": "Best decoder per-question correlation",
        "Model": scatter.loc[
            scatter["Model"].map(MATCHED_7B_MODEL_GROUP) == "Backbone Decoder"
        ].sort_values("All r", ascending=False).iloc[0]["Model"],
        "Value": round(
            float(
                scatter.loc[
                    scatter["Model"].map(MATCHED_7B_MODEL_GROUP) == "Backbone Decoder",
                    "All r",
                ].max()
            ),
            3,
        ),
    },
])

display(summary)"""
            ),
        ],
    )

    write_nb(
        NOTEBOOK_DIR / "3_HH_HM_group_patterns.ipynb",
        [
            md_cell("# 3_HH_HM_group_patterns\n\nCombined operation-group and entity-group HH/HM structural comparison for the matched 7/8B model set."),
            code_cell(COMMON_IMPORT + "\nfrom notebooks.helpers import MATCHED_7B_MODEL_GROUP\n"),
            code_cell(
                """op_corr = grouped_pattern_table("op")
display(op_corr)

out = LATEX_TABLES / "group_pattern_correlation_operation.tex"
to_latex_table(
    op_corr,
    out,
    "Operation-group correlation between grouped HH and grouped HM SBERT (matched 7/8B models).",
    "tab:group_pattern_corr_operation",
    float_formatters={"Pearson r": ".3f", "Pearson p": ".1e", "Spearman rho": ".3f", "Spearman p": ".1e"},
)
print(pretty_print_path(out))"""
            ),
            md_cell("## Operation Conclusions\n\nThis table is most useful for comparing how strongly each model preserves the human ordering across operation types."),
            code_cell(
                """op_group_summary = (
    op_corr.assign(Group=op_corr["Model"].map(MATCHED_7B_MODEL_GROUP))
    .groupby("Group", as_index=False)[["Pearson r", "Spearman rho"]]
    .mean()
    .sort_values("Pearson r", ascending=False)
)
display(op_group_summary)

best_op_group = op_group_summary.iloc[0]["Group"]
worst_op_group = op_group_summary.iloc[-1]["Group"]
print(f"Operation-level conclusion: {best_op_group} preserves the human pattern most strongly on average, while {worst_op_group} is weakest.")
print("Interpretation: operation-type structure is where architecture-level differences are easiest to see, especially for the decoder vs standalone contrast.")"""
            ),
            code_cell(
                """entity_corr = grouped_pattern_table("ent")
display(entity_corr)

out = LATEX_TABLES / "group_pattern_correlation_entity.tex"
to_latex_table(
    entity_corr,
    out,
    "Entity-group correlation between grouped HH and grouped HM SBERT (matched 7/8B models).",
    "tab:group_pattern_corr_entity",
    float_formatters={"Pearson r": ".3f", "Pearson p": ".1e", "Spearman rho": ".3f", "Spearman p": ".1e"},
)
print(pretty_print_path(out))"""
            ),
            md_cell("## Entity Conclusions\n\nEntity-group correlations provide the parallel view for semantic fields rather than reasoning operations."),
            code_cell(
                """entity_group_summary = (
    entity_corr.assign(Group=entity_corr["Model"].map(MATCHED_7B_MODEL_GROUP))
    .groupby("Group", as_index=False)[["Pearson r", "Spearman rho"]]
    .mean()
    .sort_values("Pearson r", ascending=False)
)
display(entity_group_summary)

best_ent_group = entity_group_summary.iloc[0]["Group"]
worst_ent_group = entity_group_summary.iloc[-1]["Group"]
print(f"Entity-level conclusion: {best_ent_group} aligns best with the human ordering on average, while {worst_ent_group} is weakest.")
print("Interpretation: entity-group structure is still informative, but the operation-group view is usually the cleaner summary of human–model pattern alignment.")"""
            ),
        ],
    )

    write_nb(
        NOTEBOOK_DIR / "4_HH_HM_scatter.ipynb",
        [
            md_cell("# 4_HH_HM_scatter\n\nPer-model question-level HH vs HM SBERT correlations for the matched 7/8B model set, followed by grouped operation/entity pattern correlations."),
            code_cell(COMMON_IMPORT),
            code_cell(
                """scatter_corr = scatter_correlation_table()
scatter_corr"""
            ),
            code_cell(
                """display(scatter_corr)
out = LATEX_TABLES / "hh_hm_scatter_correlations_7b.tex"
to_latex_table(
    scatter_corr,
    out,
    "Per-model question-level HH vs HM SBERT correlations for the matched 7/8B model set.",
    "tab:hh_hm_scatter_corr",
    float_formatters={"All r": ".3f", "p": ".1e", "Original": ".3f", "Weaker": ".3f", "Pronominalized": ".3f"},
)
print(pretty_print_path(out))"""
            ),
            md_cell("## Operation-aggregated correlations\n\nThis view collapses HM SBERT by operation type first, then correlates each model's operation profile against the human HH operation profile."),
            code_cell(
                """op_corr_nb4 = grouped_pattern_table("op")
display(op_corr_nb4)"""
            ),
            md_cell("## Entity-aggregated correlations\n\nThis parallel view uses entity-type profiles instead of operation profiles."),
            code_cell(
                """entity_corr_nb4 = grouped_pattern_table("ent")
display(entity_corr_nb4)"""
            ),
            md_cell("## Conclusions\n\nThese summaries help distinguish local question-level agreement from broader profile alignment."),
            code_cell(
                """best_scatter = scatter_corr.iloc[0][["Model", "All r", "Significant", "N"]]
best_op = op_corr_nb4.iloc[0][["Model", "Pearson r", "Pearson sig", "N groups"]]
best_ent = entity_corr_nb4.iloc[0][["Model", "Pearson r", "Pearson sig", "N groups"]]

print("Best per-question scatter alignment:")
display(best_scatter.to_frame().T)

print("Best operation-profile alignment:")
display(best_op.to_frame().T)

print("Best entity-profile alignment:")
display(best_ent.to_frame().T)

print("Interpretation: the per-question scatter asks whether a model tracks the question-by-question human pattern, while the grouped tables ask whether it preserves the broader operation/entity ordering. The two views are related but not identical.")"""
            ),
        ],
    )

    write_nb(
        NOTEBOOK_DIR / "5_qualitative_tables.ipynb",
        [
            md_cell("# 5_qualitative_tables\n\nQualitative table exports for operation/entity degradation and HH-HM gap examples."),
            code_cell(COMMON_IMPORT + "\nfrom notebooks.helpers import structural_alignment_examples\n"),
            code_cell(
                """qdf, human, model_ib = qualitative_qdf(include_yesno=True)
qdf.head()"""
            ),
            code_cell(
                """op_focus = ["ident", "count", "attr", "act", "spat"]
rows = []
for op in op_focus:
    sub = qdf[qdf["op"] == op].sort_values(r"C$\\to$A", ascending=False).head(3)
    rows.append(attach_answer_summaries(sub, human, model_ib, "op", "Op", r"C$\\to$A"))
op_deg = pd.concat(rows, ignore_index=True)
op_deg"""
            ),
            code_cell(
                """display(op_deg)
out = LATEX_TABLES / "table_qualitative_by_op_answers.tex"
to_latex_table(
    op_deg,
    out,
    r"Top questions by C$\\to$A degradation per operation type (variant~C, 7/8B).",
    "tab:qual_op_answers",
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """ent_focus = ["person", "animal", "object", "food"]
rows = []
for ent in ent_focus:
    sub = qdf[qdf["ent"] == ent].sort_values(r"C$\\to$A", ascending=False).head(3)
    rows.append(attach_answer_summaries(sub, human, model_ib, "ent", "Ent", r"C$\\to$A"))
ent_deg = pd.concat(rows, ignore_index=True)
ent_deg"""
            ),
            code_cell(
                """display(ent_deg)
out = LATEX_TABLES / "table_qualitative_by_entity_answers.tex"
to_latex_table(
    ent_deg,
    out,
    r"Top questions by C$\\to$A degradation per entity type (variant~C, 7/8B).",
    "tab:qual_entity_answers",
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """rows = []
for op in op_focus:
    sub = qdf[qdf["op"] == op].sort_values("HH-HM", ascending=False).head(3)
    rows.append(attach_answer_summaries(sub, human, model_ib, "op", "Op", "HH-HM"))
op_gap = pd.concat(rows, ignore_index=True)
op_gap"""
            ),
            code_cell(
                """display(op_gap)
out = LATEX_TABLES / "table_qualitative_gap_by_op_answers.tex"
to_latex_table(
    op_gap,
    out,
    r"Top questions by HH$-$HM gap per operation type (variant~C, 7/8B).",
    "tab:qual_gap_op_answers",
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """rows = []
for ent in ent_focus:
    sub = qdf[qdf["ent"] == ent].sort_values("HH-HM", ascending=False).head(3)
    rows.append(attach_answer_summaries(sub, human, model_ib, "ent", "Ent", "HH-HM"))
ent_gap = pd.concat(rows, ignore_index=True)
ent_gap"""
            ),
            code_cell(
                """display(ent_gap)
out = LATEX_TABLES / "table_qualitative_gap_by_entity_answers.tex"
to_latex_table(
    ent_gap,
    out,
    r"Top questions by HH$-$HM gap per entity type (variant~C, 7/8B).",
    "tab:qual_gap_entity_answers",
)
print(pretty_print_path(out))"""
            ),
            md_cell("## Structural Alignment Examples\n\nThese examples are tied to the grouped distributional winners rather than to question-level SBERT alone. They help interpret which question slices the top structural models match most closely, while staying anchored to the same human answer patterns discussed earlier in the results."),
            code_cell(
                """struct_examples = structural_alignment_examples(
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=False,
    answer_types=("yes/no", "number", "text"),
    dimensions=("op", "ent"),
    top_n=2,
)
display(struct_examples)"""
            ),
            code_cell(
                """struct_examples_filt = structural_alignment_examples(
    condition="inst_blind",
    variants=("C", "B", "A"),
    filter_abstention=True,
    answer_types=("yes/no", "number", "text"),
    dimensions=("op", "ent"),
    top_n=2,
)
display(struct_examples_filt)"""
            ),
        ],
    )

    write_nb(
        NOTEBOOK_DIR / "6_top_questions.ipynb",
        [
            md_cell("# 6_top_questions\n\nRepresentative high-agreement questions by operation and entity group for the supplementary appendix."),
            code_cell(COMMON_IMPORT),
            code_cell(
                """top_op, top_ent = top_questions_tables(include_yesno=True)
top_op.head()"""
            ),
            code_cell(
                """display(top_op)
out = LATEX_TABLES / "top_questions_op.tex"
to_latex_table(
    top_op,
    out,
    r"Representative high-agreement questions by operation type (original formulation, top 2 per group by HH SBERT). $\Delta_{C \\to A}$: degradation from original to pronominalized. Human top answers show the most frequent blind responses with counts in parentheses.",
    "tab:top_questions_op",
    float_formatters={"HH": ".3f", "HM": ".3f", r"$\\Delta_{C \\to A}^{HH}$": ".3f", r"$\\Delta_{C \\to A}^{HM}$": ".3f"},
)
print(pretty_print_path(out))"""
            ),
            code_cell(
                """top_ent.head()"""
            ),
            code_cell(
                """display(top_ent)
out = LATEX_TABLES / "top_questions_ent.tex"
to_latex_table(
    top_ent,
    out,
    r"Representative high-agreement questions by entity type (original formulation, top 2 per group by HH SBERT). $\Delta_{C \\to A}$: degradation from original to pronominalized. Human top answers show the most frequent blind responses with counts in parentheses.",
    "tab:top_questions_ent",
    float_formatters={"HH": ".3f", "HM": ".3f", r"$\\Delta_{C \\to A}^{HH}$": ".3f", r"$\\Delta_{C \\to A}^{HM}$": ".3f"},
)
print(pretty_print_path(out))"""
            ),
        ],
    )


if __name__ == "__main__":
    build()
