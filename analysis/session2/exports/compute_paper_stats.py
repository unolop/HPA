from pathlib import Path
"""
compute_paper_stats.py
Computes every statistical value cited in the paper and saves to paper_stats.csv.
Also writes paper_stats_ops_table.csv and paper_stats_entity_table.csv.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import sys

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
EXPORTS = str(Path(__file__).resolve().parents[1])

print("Loading data...", file=sys.stderr)
pair_ib    = pd.read_parquet(f"{EXPORTS}/pair_cache_cleaned.parquet")
pair_blind = pd.read_parquet(f"{EXPORTS}/pair_cache_cleaned_blind.parquet")
resp       = pd.read_csv(f"{EXPORTS}/responses_model_inst_blind.csv")

# Convenience subsets
hh     = pair_ib[pair_ib["pair_type"] == "HH"]
hm     = pair_ib[pair_ib["pair_type"] == "HM"]
hm_bl  = pair_blind[pair_blind["pair_type"] == "HM"]

# ---------------------------------------------------------------------------
# Helper: append a row
# ---------------------------------------------------------------------------
rows = []

def add(key, value, description, location):
    rows.append(dict(
        key=key,
        value=round(float(value), 4),
        description=description,
        location=location,
    ))

# ---------------------------------------------------------------------------
# 1. Group-level SBERT (inst_blind, all variants pooled, all models)
# ---------------------------------------------------------------------------
print("Computing group-level SBERT...", file=sys.stderr)

add("hh_sbert_ceiling",
    hh["sbert_score"].mean(),
    "Mean HH SBERT across all questions/variants (inst_blind)",
    "§4")

for grp_key, grp_label, grp_name in [
    ("vlm_sbert_mean",                "VLM",                    "VLM"),
    ("backbone_sbert_mean",           "VLM backbone decoder",   "VLM backbone decoder"),
    ("standalone_sbert_mean",         "standalone LLM",         "standalone LLM (no think)"),
    ("standalone_think_sbert_mean",   "standalone LLM (think)", "standalone LLM (think)"),
]:
    add(grp_key,
        hm[hm["subject_group_2"] == grp_label]["sbert_score"].mean(),
        f"Mean HM SBERT for {grp_name} group (inst_blind, all variants)",
        "§4 / Table 1")

# ---------------------------------------------------------------------------
# 2. Individual model SBERT
# ---------------------------------------------------------------------------
print("Computing individual model SBERT...", file=sys.stderr)

add("internvl_8b_hm_sbert",
    hm[hm["subject_2"] == "InternVL-8B"]["sbert_score"].mean(),
    "InternVL-8B mean HM SBERT (inst_blind, all variants)",
    "§4")

backbone_by_model = (
    hm[hm["subject_group_2"] == "VLM backbone decoder"]
    .groupby("subject_2")["sbert_score"].mean()
)
best_bb  = backbone_by_model.idxmax()
worst_bb = backbone_by_model.idxmin()

add("backbone_best_sbert",
    backbone_by_model.max(),
    f"Highest backbone decoder HM SBERT — {best_bb}",
    "§4")

add("backbone_worst_sbert",
    backbone_by_model.min(),
    f"Lowest backbone decoder HM SBERT — {worst_bb}",
    "§4")

standalone_by_model = (
    hm[hm["subject_group_2"] == "standalone LLM"]
    .groupby("subject_2")["sbert_score"].mean()
)
best_sa  = standalone_by_model.idxmax()
worst_sa = standalone_by_model.idxmin()

add("standalone_best_sbert",
    standalone_by_model.max(),
    f"Best standalone LLM (no think) HM SBERT — {best_sa}",
    "§4")

add("standalone_worst_sbert",
    standalone_by_model.min(),
    f"Worst standalone LLM (no think) HM SBERT — {worst_sa}",
    "§4")

# ---------------------------------------------------------------------------
# 3. Scale SBERT
# ---------------------------------------------------------------------------
print("Computing scale SBERT...", file=sys.stderr)

add("backbone_2b_sbert",
    hm[hm["subject_2"].isin(["InternVL-2B (LM)", "Qwen3-VL-2B (LM)"])]["sbert_score"].mean(),
    "Mean HM SBERT for ~2B backbone decoders (InternVL-2B LM + Qwen3-VL-2B LM)",
    "§4")

backbone_8b_models = [
    "InternVL-8B (LM)", "Qwen3-VL-8B (LM)",
    "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)", "LLaVA-1.5 (LM)",
]
add("backbone_8b_sbert",
    hm[hm["subject_2"].isin(backbone_8b_models)]["sbert_score"].mean(),
    "Mean HM SBERT for ~7-8B backbone decoders (InternVL-8B/Qwen3-8B/LLaVA LM variants)",
    "§4")

add("standalone_06b_sbert",
    hm[hm["subject_2"] == "Qwen3-0.6B"]["sbert_score"].mean(),
    "Qwen3-0.6B standalone LLM mean HM SBERT",
    "§4")

add("standalone_32b_sbert",
    hm[hm["subject_2"] == "Qwen3-32B"]["sbert_score"].mean(),
    "Qwen3-32B standalone LLM mean HM SBERT (no think)",
    "§4")

add("standalone_32b_think_sbert",
    hm[hm["subject_2"] == "Qwen3-32B (think)"]["sbert_score"].mean(),
    "Qwen3-32B (think) mean HM SBERT",
    "§4")

# ---------------------------------------------------------------------------
# 4. Accuracy
# ---------------------------------------------------------------------------
print("Computing accuracy stats...", file=sys.stderr)

# human_avg_acc — one value per question+variant pair, then mean
q_var_human = resp.groupby(["question_id", "variant"])["human_avg_acc"].first()
human_avg   = q_var_human.mean()

def grp_acc(group_name):
    return resp[resp["model_group"] == group_name]["accuracy"].mean()

vlm_avg        = grp_acc("VLM")
backbone_avg   = grp_acc("VLM backbone decoder")
standalone_avg = grp_acc("standalone LLM")

add("human_avg_acc",      human_avg,    "Mean human accuracy across all questions/variants",      "Table 1")
add("vlm_avg_acc",        vlm_avg,      "Mean VLM accuracy (all variants)",                       "Table 1")
add("backbone_avg_acc",   backbone_avg, "Mean backbone decoder accuracy (all variants)",          "Table 1")
add("standalone_avg_acc", standalone_avg, "Mean standalone LLM accuracy (no think, all variants)", "Table 1")

add("vlm_acc_advantage_pp",
    (vlm_avg - human_avg) * 100,
    "VLM accuracy advantage over humans in percentage points",
    "§4")

add("backbone_acc_advantage_pp",
    (backbone_avg - human_avg) * 100,
    "Backbone decoder accuracy advantage over humans in percentage points",
    "§4")

add("standalone_acc_gap_pp",
    (standalone_avg - human_avg) * 100,
    "Standalone LLM accuracy gap vs humans in percentage points (negative = below human)",
    "§4")

vlm_vC_by_model = (
    resp[(resp["model_group"] == "VLM") & (resp["variant"] == "C")]
    .groupby("model")["accuracy"].mean()
)
add("vlm_acc_max_vC",
    vlm_vC_by_model.max(),
    f"Max VLM accuracy on variant C — {vlm_vC_by_model.idxmax()}",
    "Table 1")

add("vlm_acc_min_vC",
    vlm_vC_by_model.min(),
    f"Min VLM accuracy on variant C — {vlm_vC_by_model.idxmin()}",
    "Table 1")

# ---------------------------------------------------------------------------
# 5. Accuracy drop C→A
# ---------------------------------------------------------------------------
print("Computing accuracy drop C->A...", file=sys.stderr)

for key_prefix, grp_name in [
    ("vlm",        "VLM"),
    ("backbone",   "VLM backbone decoder"),
    ("standalone", "standalone LLM"),
]:
    grp_resp = resp[resp["model_group"] == grp_name]
    acc_C    = grp_resp[grp_resp["variant"] == "C"]["accuracy"].mean()
    acc_A    = grp_resp[grp_resp["variant"] == "A"]["accuracy"].mean()
    add(f"{key_prefix}_acc_drop_pp",
        (acc_C - acc_A) * 100,
        f"{grp_name} accuracy drop from variant C to A in percentage points",
        "§4 / Table 1")

# ---------------------------------------------------------------------------
# 6. HH stats per operation (variant C and A, with t-test)
# ---------------------------------------------------------------------------
print("Computing per-operation HH stats...", file=sys.stderr)

ops_paper = ["ident", "count", "attr", "act", "spat", "exist", "text", "know", "temp"]

ops_table_rows = []

for op in pair_ib["op"].unique():
    hh_op = hh[hh["op"] == op]
    q_variant = hh_op.groupby(["question_id", "variant"])["sbert_score"].mean().unstack()
    n_q = len(q_variant)

    means = {}
    for v in ["C", "B", "A"]:
        means[v] = q_variant[v].mean() if v in q_variant.columns else np.nan

    valid_CA = q_variant.dropna(subset=["C", "A"]) if ("C" in q_variant.columns and "A" in q_variant.columns) else pd.DataFrame()
    if len(valid_CA) >= 2:
        _, p_CA = stats.ttest_rel(valid_CA["C"], valid_CA["A"])
        _, p_wilcox = stats.wilcoxon(valid_CA["C"], valid_CA["A"])
    else:
        p_CA = np.nan
        p_wilcox = np.nan

    ops_table_rows.append(dict(
        op=op,
        n_questions=n_q,
        hh_C_mean=round(means["C"], 4) if not np.isnan(means["C"]) else np.nan,
        hh_B_mean=round(means["B"], 4) if not np.isnan(means["B"]) else np.nan,
        hh_A_mean=round(means["A"], 4) if not np.isnan(means["A"]) else np.nan,
        delta_CA=round(means["C"] - means["A"], 4) if not (np.isnan(means["C"]) or np.isnan(means["A"])) else np.nan,
        ttest_p_CA=round(p_CA, 4) if not np.isnan(p_CA) else np.nan,
        wilcoxon_p_CA=round(p_wilcox, 4) if not np.isnan(p_wilcox) else np.nan,
    ))

    if op in ops_paper:
        add(f"hh_{op}_C",
            means["C"],
            f"Mean HH SBERT for op={op}, variant C",
            "§4 / Table 2")
        add(f"hh_{op}_A",
            means["A"],
            f"Mean HH SBERT for op={op}, variant A",
            "§4 / Table 2")
        add(f"hh_{op}_wilcoxon_p",
            p_wilcox,
            f"Wilcoxon signed-rank p-value HH SBERT variant C vs A for op={op}",
            "§4 / Table 2")

# ---------------------------------------------------------------------------
# 7. ANOVA (HH SBERT by operation type)
# ---------------------------------------------------------------------------
print("Computing operation ANOVA...", file=sys.stderr)

for v in ["C", "B", "A"]:
    hh_v = hh[hh["variant"] == v]
    q_op = hh_v.groupby(["question_id", "op"])["sbert_score"].mean().reset_index()
    groups = [g["sbert_score"].values for _, g in q_op.groupby("op")]
    F, p = stats.f_oneway(*groups)
    H, p_kw = stats.kruskal(*groups)

    add(f"anova_op_F_v{v}", F,
        f"One-way ANOVA F-statistic for HH SBERT by op type, variant {v}",
        "§4")
    add(f"anova_op_p_v{v}", p,
        f"One-way ANOVA p-value for HH SBERT by op type, variant {v}",
        "§4")

    if v in ["C", "A"]:
        add(f"kruskal_op_H_v{v}", H,
            f"Kruskal-Wallis H-statistic for HH SBERT by op type, variant {v}",
            "§4")
        add(f"kruskal_op_p_v{v}", p_kw,
            f"Kruskal-Wallis p-value for HH SBERT by op type, variant {v}",
            "§4")

# ---------------------------------------------------------------------------
# 8. Entity ANOVA
# ---------------------------------------------------------------------------
print("Computing entity ANOVA...", file=sys.stderr)

for v in ["C", "A"]:
    hh_v = hh[hh["variant"] == v]
    q_ent = hh_v.groupby(["question_id", "ent"])["sbert_score"].mean().reset_index()
    groups = [g["sbert_score"].values for _, g in q_ent.groupby("ent")]
    F, p = stats.f_oneway(*groups)
    add(f"anova_ent_F_v{v}", F,
        f"One-way ANOVA F-statistic for HH SBERT by entity type, variant {v}",
        "§4")
    add(f"anova_ent_p_v{v}", p,
        f"One-way ANOVA p-value for HH SBERT by entity type, variant {v}",
        "§4")

# ---------------------------------------------------------------------------
# 9. Entity group stats (HH SBERT, variant C and A, t-test)
# ---------------------------------------------------------------------------
print("Computing entity group stats...", file=sys.stderr)

entities_paper = ["food", "animal", "person", "object"]
ent_table_rows = []

for ent in pair_ib["ent"].unique():
    hh_ent = hh[hh["ent"] == ent]
    q_v = hh_ent.groupby(["question_id", "variant"])["sbert_score"].mean().unstack()
    n_q = len(q_v)

    c_mean = q_v["C"].mean() if "C" in q_v.columns else np.nan
    a_mean = q_v["A"].mean() if "A" in q_v.columns else np.nan

    valid_CA = q_v.dropna(subset=["C", "A"]) if ("C" in q_v.columns and "A" in q_v.columns) else pd.DataFrame()
    if len(valid_CA) >= 2:
        _, p_CA = stats.wilcoxon(valid_CA["C"], valid_CA["A"])
    else:
        p_CA = np.nan

    ent_table_rows.append(dict(
        entity=ent,
        n_questions=n_q,
        hh_C_mean=round(c_mean, 4) if not np.isnan(c_mean) else np.nan,
        hh_A_mean=round(a_mean, 4) if not np.isnan(a_mean) else np.nan,
        delta_CA=round(c_mean - a_mean, 4) if not (np.isnan(c_mean) or np.isnan(a_mean)) else np.nan,
        wilcoxon_p_CA=round(p_CA, 4) if not np.isnan(p_CA) else np.nan,
    ))

    if ent in entities_paper:
        add(f"hh_{ent}_C",
            c_mean,
            f"Mean HH SBERT for entity={ent}, variant C",
            "§4 / Table 3")
        add(f"hh_{ent}_A",
            a_mean,
            f"Mean HH SBERT for entity={ent}, variant A",
            "§4 / Table 3")
        add(f"hh_{ent}_wilcoxon_p",
            p_CA,
            f"Wilcoxon signed-rank p-value HH SBERT variant C vs A for entity={ent}",
            "§4 / Table 3")

# ---------------------------------------------------------------------------
# 10. HH cross-variant correlations
# ---------------------------------------------------------------------------
print("Computing cross-variant correlations...", file=sys.stderr)

hh_qv = hh.groupby(["question_id", "variant"])["sbert_score"].mean().unstack()
hh_qv.columns.name = None

add("hh_corr_CA",
    hh_qv["C"].corr(hh_qv["A"]),
    "Pearson r between per-question mean HH SBERT variant C and A",
    "§4")

add("hh_corr_CB",
    hh_qv["C"].corr(hh_qv["B"]),
    "Pearson r between per-question mean HH SBERT variant C and B",
    "§4")

add("hh_corr_BA",
    hh_qv["B"].corr(hh_qv["A"]),
    "Pearson r between per-question mean HH SBERT variant B and A",
    "§4")

# ---------------------------------------------------------------------------
# 11. Instruction effect (blind -> inst_blind delta in HM SBERT)
# ---------------------------------------------------------------------------
print("Computing instruction effect...", file=sys.stderr)

# Per question+variant+model mean SBERT in each condition
ib_qvm = (
    hm
    .groupby(["question_id", "variant", "subject_2", "subject_group_2"])["sbert_score"]
    .mean()
    .reset_index()
    .rename(columns={"sbert_score": "sbert_ib"})
)

bl_qvm = (
    hm_bl
    .groupby(["question_id", "variant", "subject_2", "subject_group_2"])["sbert_score"]
    .mean()
    .reset_index()
    .rename(columns={"sbert_score": "sbert_blind"})
)

merged = ib_qvm.merge(
    bl_qvm,
    on=["question_id", "variant", "subject_2", "subject_group_2"],
    how="inner",
)
merged["delta"] = merged["sbert_ib"] - merged["sbert_blind"]

add("inst_effect_overall",
    merged["delta"].mean(),
    "Mean SBERT delta (inst_blind minus blind), all models",
    "§5")

for grp_key, grp_label in [
    ("vlm",        "VLM"),
    ("backbone",   "VLM backbone decoder"),
    ("standalone", "standalone LLM"),
]:
    add(f"inst_effect_{grp_key}",
        merged[merged["subject_group_2"] == grp_label]["delta"].mean(),
        f"Mean SBERT delta (inst_blind minus blind) for {grp_label}",
        "§5")

# ---------------------------------------------------------------------------
# 12. Top model accuracy (variant C)
# ---------------------------------------------------------------------------
print("Computing top model accuracy stats...", file=sys.stderr)

def model_acc_vC(model_name):
    sub = resp[(resp["model"] == model_name) & (resp["variant"] == "C")]
    return sub["accuracy"].mean() if len(sub) > 0 else np.nan

add("top_acc_lavamistral_vC",
    model_acc_vC("LLaVA-Mistral"),
    "LLaVA-Mistral (VLM) accuracy on variant C",
    "Table 1")

add("top_acc_lavamistral_lm_vC",
    model_acc_vC("LLaVA-Mistral (LM)"),
    "LLaVA-Mistral (LM backbone decoder) accuracy on variant C",
    "Table 1")

add("top_acc_lava15_lm_vC",
    model_acc_vC("LLaVA-1.5 (LM)"),
    "LLaVA-1.5 (LM backbone decoder) accuracy on variant C",
    "Table 1")

add("internvl8b_hm_sbert_vC_top",
    hm[(hm["subject_2"] == "InternVL-8B") & (hm["variant"] == "C")]["sbert_score"].mean(),
    "InternVL-8B HM SBERT on variant C",
    "§4")

# ---------------------------------------------------------------------------
# Save paper_stats.csv
# ---------------------------------------------------------------------------
print("Saving paper_stats.csv...", file=sys.stderr)

df_stats = pd.DataFrame(rows, columns=["key", "value", "description", "location"])
out_stats = f"{EXPORTS}/paper_stats.csv"
df_stats.to_csv(out_stats, index=False)

# ---------------------------------------------------------------------------
# Save paper_stats_ops_table.csv
# ---------------------------------------------------------------------------
print("Saving paper_stats_ops_table.csv...", file=sys.stderr)

df_ops = (
    pd.DataFrame(ops_table_rows)
    .sort_values("hh_C_mean", ascending=False)
    .reset_index(drop=True)
)
df_ops.to_csv(f"{EXPORTS}/paper_stats_ops_table.csv", index=False)

# ---------------------------------------------------------------------------
# Save paper_stats_entity_table.csv
# ---------------------------------------------------------------------------
print("Saving paper_stats_entity_table.csv...", file=sys.stderr)

df_ent = (
    pd.DataFrame(ent_table_rows)
    .sort_values("hh_C_mean", ascending=False)
    .reset_index(drop=True)
)
df_ent.to_csv(f"{EXPORTS}/paper_stats_entity_table.csv", index=False)

# ---------------------------------------------------------------------------
# Print paper_stats.csv to stdout
# ---------------------------------------------------------------------------
print(df_stats.to_csv(index=False))

print(f"\nDone. Wrote {len(df_stats)} stats rows.", file=sys.stderr)
print(f"Ops table: {len(df_ops)} rows", file=sys.stderr)
print(f"Entity table: {len(df_ent)} rows", file=sys.stderr)
