"""
Lightweight summary of existing think-length analyses.

This script does not reload the raw token-logit JSONLs. It summarizes the
already-generated EDA CSVs under figures/think_eda/ and answers:
what is think length positively correlated with, and what is it not helping?

Outputs:
  - figures/think_eda/think_length_correlation_digest.csv

Run from repo root:
  conda run -n zero python analysis/think_length_correlations.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = ROOT / "figures" / "think_eda"


def pearson(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y, method="pearson"))


def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y, method="spearman"))


def main() -> None:
    analysis_df = pd.read_csv(EDA_DIR / "think_analysis_results.csv")
    summary_df = pd.read_csv(EDA_DIR / "think_eda_summary.csv")
    overall_df = pd.read_csv(EDA_DIR / "think_overall_stats.csv")

    rows = []

    # Existing per-question correlations from think_model_eda.py
    for analysis_name in ["A_think_vs_hh", "B_think_vs_hm", "D_op_corr"]:
        sub = analysis_df[analysis_df["analysis"] == analysis_name].copy()
        score_col = "rho" if analysis_name == "D_op_corr" else "r"
        p_col = "rho_p" if analysis_name == "D_op_corr" else "r_p"
        pos_sig = sub[(sub[score_col] > 0) & (sub[p_col] < 0.05)].sort_values(score_col, ascending=False)
        for _, row in pos_sig.iterrows():
            rows.append(
                {
                    "source": analysis_name,
                    "model": row["model"],
                    "scope": row["variant"],
                    "target": "hh_sbert" if analysis_name != "B_think_vs_hm" else "hm_sbert",
                    "stat": score_col,
                    "value": row[score_col],
                    "p_value": row[p_col],
                    "n": row["n"],
                }
            )

    # Variant-level summary relationships across the 15 model-variant rows.
    think_mean = summary_df["think_mean"]
    for target in ["final_mean_words", "empty_think_pct", "truncated_pct"]:
        rows.append(
            {
                "source": "variant_summary",
                "model": "all models",
                "scope": "15 model-variant rows",
                "target": target,
                "stat": "pearson",
                "value": pearson(think_mean, summary_df[target]),
                "p_value": None,
                "n": len(summary_df),
            }
        )
        rows.append(
            {
                "source": "variant_summary",
                "model": "all models",
                "scope": "15 model-variant rows",
                "target": target,
                "stat": "spearman",
                "value": spearman(think_mean, summary_df[target]),
                "p_value": None,
                "n": len(summary_df),
            }
        )

    # Across-scale summary relationships across the 5 think models.
    mean_think = overall_df["mean_think_words"]
    for target in ["mean_hm_sbert", "empty_think_pct"]:
        rows.append(
            {
                "source": "overall_model_summary",
                "model": "all think models",
                "scope": "5 models",
                "target": target,
                "stat": "pearson",
                "value": pearson(mean_think, overall_df[target]),
                "p_value": None,
                "n": len(overall_df),
            }
        )
        rows.append(
            {
                "source": "overall_model_summary",
                "model": "all think models",
                "scope": "5 models",
                "target": target,
                "stat": "spearman",
                "value": spearman(mean_think, overall_df[target]),
                "p_value": None,
                "n": len(overall_df),
            }
        )

    digest = pd.DataFrame(rows)
    out_path = EDA_DIR / "think_length_correlation_digest.csv"
    digest.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print("\nPositive significant results already present in the EDA:")
    existing = digest[digest["source"].isin(["A_think_vs_hh", "B_think_vs_hm", "D_op_corr"])]
    if existing.empty:
        print("  none")
    else:
        print(existing.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nCross-summary relationships:")
    cross = digest[digest["source"].isin(["variant_summary", "overall_model_summary"])]
    print(cross.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
