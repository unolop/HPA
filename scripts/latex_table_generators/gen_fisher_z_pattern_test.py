"""
Fisher's z test: do model groups differ in how closely they track the human
difficulty pattern across operation types?

Approach:
  For each model, compute Pearson r between:
    - per-op mean HM SBERT  (model vs. humans, blind condition)
    - per-op mean HH SBERT  (human-human agreement, same questions)
  across the 10 operation types.

  Fisher-z transform each r, then compare group distributions
  (backbone decoders vs VLMs, backbone vs standalone LLMs)
  using two-sample t-tests on the z scores.

Outputs:
  - printed summary table
  - latex/AAAI2026/LaTeX/tables/fisher_z_pattern_test.tex
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables"

MATCHED_MODELS = [
    "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
    "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    "Qwen2.5-7B-Instruct", "Qwen3-8B", "Vicuna-7B",
]

MODEL_GROUPS = {
    "VLM":             ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone Decoder": ["InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"],
    "Standalone LLM":  ["Qwen2.5-7B-Instruct", "Qwen3-8B", "Vicuna-7B"],
}

OP_LABELS = {
    "act": "Action", "attr": "Attribute", "cause": "Causality",
    "comp": "Comparison", "count": "Count", "ident": "Identity",
    "know": "World Knowledge", "spat": "Spatial", "temp": "Temporal", "text": "Text Reading",
}


def fisher_z(r):
    """Fisher z-transformation, clipped to avoid inf at ±1."""
    r = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r)


def compare_groups(z1, z2, label1, label2):
    """Two-sample Welch t-test on Fisher-z scores."""
    t, p = stats.ttest_ind(z1, z2, equal_var=False)
    n1, n2 = len(z1), len(z2)
    # Convert pooled mean z-difference back to r-difference for reporting
    mean_r1 = np.tanh(np.mean(z1))
    mean_r2 = np.tanh(np.mean(z2))
    return {
        "group1": label1, "group2": label2,
        "n1": n1, "n2": n2,
        "mean_r1": mean_r1, "mean_r2": mean_r2,
        "delta_r": mean_r1 - mean_r2,
        "t": t, "p": p,
    }


def main():
    exports = ROOT / "analysis" / "session2" / "exports"
    df = pd.read_parquet(exports / "pair_cache_cleaned_blind.parquet")

    hm = df[df["pair_type"] == "HM"].copy()
    hh = df[df["pair_type"] == "HH"].copy()

    # Per-op mean HH SBERT (human reference pattern, pooled across all questions)
    hh_op = hh.groupby("op")["sbert_score"].mean()

    # Per-model, per-op mean HM SBERT
    hm_matched = hm[hm["subject_2"].isin(MATCHED_MODELS)]
    hm_op = hm_matched.groupby(["subject_2", "op"])["sbert_score"].mean().unstack("op")

    # Only keep ops present in both
    common_ops = sorted(set(hh_op.index) & set(hm_op.columns))
    hh_vec = hh_op[common_ops].values  # shape (n_ops,)
    hm_op = hm_op[common_ops]           # shape (n_models, n_ops)

    print(f"Operations used (n={len(common_ops)}): {common_ops}")
    print(f"Models: {list(hm_op.index)}\n")

    # Per-model Pearson r and Fisher z
    results = []
    for model in hm_op.index:
        model_vec = hm_op.loc[model].values
        mask = ~np.isnan(model_vec)
        if mask.sum() < 4:
            print(f"  Skipping {model}: too few ops ({mask.sum()})")
            continue
        r, p_r = stats.pearsonr(hh_vec[mask], model_vec[mask])
        z = fisher_z(r)
        group = next((g for g, ms in MODEL_GROUPS.items() if model in ms), "Other")
        results.append({"model": model, "group": group, "r": r, "z": z,
                        "p_r": p_r, "n_ops": mask.sum()})

    res_df = pd.DataFrame(results)
    print("Per-model pattern correlations (HM vs HH SBERT across operation types):")
    print(res_df[["model", "group", "r", "p_r", "n_ops"]].to_string(index=False))
    print()

    # Group-level Fisher's z comparisons
    group_z = {g: res_df[res_df["group"] == g]["z"].values for g in MODEL_GROUPS}
    group_r_mean = {g: np.tanh(np.mean(group_z[g])) for g in MODEL_GROUPS}

    print("Group mean r (back-transformed from Fisher z):")
    for g, r in group_r_mean.items():
        print(f"  {g}: r = {r:.3f}  (n={len(group_z[g])})")
    print()

    comparisons = [
        ("Backbone Decoder", "VLM"),
        ("Backbone Decoder", "Standalone LLM"),
        ("VLM", "Standalone LLM"),
    ]
    comp_results = []
    for g1, g2 in comparisons:
        comp = compare_groups(group_z[g1], group_z[g2], g1, g2)
        comp_results.append(comp)
        print(f"{g1} vs {g2}: Δr={comp['delta_r']:+.3f}  "
              f"(r1={comp['mean_r1']:.3f}, r2={comp['mean_r2']:.3f})  "
              f"t({comp['n1']+comp['n2']-2})={comp['t']:.2f}  p={comp['p']:.4f}")

    # Write LaTeX table
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Comparison & $\bar{r}_1$ & $\bar{r}_2$ & $\Delta r$ & $p$ \\",
        r"\midrule",
    ]
    for c in comp_results:
        p_str = f"{c['p']:.3f}" if c['p'] >= 0.001 else r"${<}.001$"
        lines.append(
            f"{c['group1']} vs.\\ {c['group2']} & "
            f"{c['mean_r1']:.3f} & {c['mean_r2']:.3f} & "
            f"{c['delta_r']:+.3f} & {p_str} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Fisher's $z$ comparison of per-model pattern correlation "
        r"($r$ between model and human HH SBERT across operation types, $n{=}"
        + str(len(common_ops))
        + r"$ operation groups). "
        r"$\bar{r}$ is the group mean back-transformed from Fisher $z$; "
        r"$p$-values from two-sample Welch $t$-test on $z$ scores.}",
        r"\label{tab:fisher_z_pattern}",
        r"\end{table}",
    ]

    out = OUT_DIR / "fisher_z_pattern_test.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
