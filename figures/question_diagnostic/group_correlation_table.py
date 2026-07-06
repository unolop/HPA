"""Generate LaTeX table: group-level HH vs HM correlation stats for supplementary.
Per model: Spearman ρ, Pearson r, cosine similarity — for op and ent groupings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "analysis/session2/exports"
OUTDIR = Path(__file__).resolve().parent

MODEL_GROUPS = {
    "VLM": [
        "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
    ],
    "Backbone Decoder": [
        "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    ],
    "Standalone LLM": [
        "Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B",
    ],
}


def compute_stats(hh_vec, hm_vec):
    """Compute Spearman ρ, Pearson r, cosine similarity between two vectors."""
    if len(hh_vec) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    rho, rho_p = stats.spearmanr(hh_vec, hm_vec)
    r, r_p = stats.pearsonr(hh_vec, hm_vec)
    cos_sim = 1 - cosine_dist(hh_vec, hm_vec)
    return rho, rho_p, r, r_p, cos_sim


def generate_table(pc: pd.DataFrame, suffix: str = ""):
    hh = pc[pc["pair_type"] == "HH"]
    hm = pc[pc["pair_type"] == "HM"]

    all_models = [m for models in MODEL_GROUPS.values() for m in models]

    rows = []
    for group_type, col in [("Operation", "op"), ("Entity", "ent")]:
        hh_grp = hh.groupby(col).agg(hh_sbert=("sbert_score", "mean")).reset_index()
        hh_grp.rename(columns={col: "group"}, inplace=True)

        hm_grp = (
            hm.groupby([col, "subject_2"])
            .agg(hm_sbert=("sbert_score", "mean"))
            .reset_index()
        )
        hm_grp.rename(columns={col: "group"}, inplace=True)

        merged = hm_grp.merge(hh_grp, on="group")
        n_groups = hh_grp["group"].nunique()

        for model_group_name, models in MODEL_GROUPS.items():
            for model in models:
                sub = merged[merged["subject_2"] == model].sort_values("group")
                if len(sub) < 3:
                    continue
                rho, rho_p, r, r_p, cos_sim = compute_stats(
                    sub["hh_sbert"].values, sub["hm_sbert"].values
                )
                # Mean absolute deviation from diagonal
                mad = np.mean(np.abs(sub["hm_sbert"].values - sub["hh_sbert"].values))
                rows.append({
                    "Grouping": group_type,
                    "Model Group": model_group_name,
                    "Model": model,
                    "n": n_groups,
                    "Spearman ρ": rho,
                    "ρ p-val": rho_p,
                    "Pearson r": r,
                    "r p-val": r_p,
                    "Cosine Sim": cos_sim,
                    "MAD": mad,
                })

    df = pd.DataFrame(rows)

    # Print console summary
    print(f"\n{'='*90}")
    print(f"GROUP-LEVEL CORRELATION TABLE{suffix}")
    print(f"{'='*90}")
    for gt in ["Operation", "Entity"]:
        print(f"\n--- {gt} grouping ---")
        sub = df[df["Grouping"] == gt]
        print(f"{'Model':<25} {'ρ':>6} {'p':>8} {'r':>6} {'p':>8} {'cos':>6} {'MAD':>6}")
        for _, row in sub.iterrows():
            print(f"{row['Model']:<25} {row['Spearman ρ']:>6.3f} {row['ρ p-val']:>8.3f} "
                  f"{row['Pearson r']:>6.3f} {row['r p-val']:>8.3f} "
                  f"{row['Cosine Sim']:>6.4f} {row['MAD']:>6.3f}")

    # Generate LaTeX
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\caption{Group-level correlation between human--human (HH) and human--model (HM) "
                       r"SBERT agreement, aggregated by operation and entity type. "
                       r"$\rho$: Spearman rank correlation; $r$: Pearson correlation; "
                       r"Cos: cosine similarity of group-mean profiles; "
                       r"MAD: mean absolute deviation from diagonal (lower = closer to human pattern).}")
    latex_lines.append(r"\label{tab:group_pattern_correlation}")
    latex_lines.append(r"\begin{tabular}{ll cc cc c c}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r" & & \multicolumn{2}{c}{Spearman} & \multicolumn{2}{c}{Pearson} & & \\")
    latex_lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    latex_lines.append(r"Grouping & Model & $\rho$ & $p$ & $r$ & $p$ & Cos & MAD \\")
    latex_lines.append(r"\midrule")

    for gt in ["Operation", "Entity"]:
        sub = df[df["Grouping"] == gt]
        first_in_group = True
        prev_mg = None
        for _, row in sub.iterrows():
            mg = row["Model Group"]
            model_disp = row["Model"].replace("(LM)", "{\\scriptsize(LM)}")
            model_disp = model_disp.replace("(think)", "{\\scriptsize(think)}")

            # Add separator between model groups
            if prev_mg is not None and mg != prev_mg:
                latex_lines.append(r"\cdashline{2-8}")
            prev_mg = mg

            grouping_col = f"\\multirow{{{len(sub)}}}{{*}}{{{gt}}}" if first_in_group else ""
            first_in_group = False

            # Format p-values
            def fmt_p(p):
                if p < 0.001:
                    return f"$<$.001"
                elif p < 0.01:
                    return f"{p:.3f}"
                elif p < 0.05:
                    return f"{p:.3f}"
                else:
                    return f"{p:.3f}"

            line = (f"{grouping_col} & {model_disp} & "
                    f"{row['Spearman ρ']:.3f} & {fmt_p(row['ρ p-val'])} & "
                    f"{row['Pearson r']:.3f} & {fmt_p(row['r p-val'])} & "
                    f"{row['Cosine Sim']:.4f} & {row['MAD']:.3f} \\\\")
            latex_lines.append(line)

        if gt == "Operation":
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table*}")

    latex = "\n".join(latex_lines)

    out_path = OUTDIR / f"group_correlation_table{suffix}.tex"
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX saved to {out_path}")

    # Also compute model-group averages
    print(f"\n--- Model group averages ---")
    for gt in ["Operation", "Entity"]:
        print(f"\n{gt}:")
        sub = df[df["Grouping"] == gt]
        avg = sub.groupby("Model Group")[["Spearman ρ", "Pearson r", "Cosine Sim", "MAD"]].mean()
        avg = avg.reindex(["VLM", "Backbone Decoder", "Standalone LLM"])
        print(avg.round(3).to_string())

    return df


if __name__ == "__main__":
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    generate_table(pc, "_all")
