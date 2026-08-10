"""Generate per-model Spearman ρ table: concordance on degradation patterns.

Computes Spearman ρ between human HH Δ and model HM Δ across operation and entity
groups, where Δ = Pronominalized - Original SBERT (degradation from anchor removal).
This is what §4.3 of the paper reports at the architecture-class level (0.69/0.48/0.25).
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
    "Qwen2.5-7B-Instruct", "Qwen3-8B", "Qwen3-8B (think)", "Vicuna-7B", "Mistral-7B",
]

GROUP_ORDER = ["VLM", "Backbone Decoder", "Standalone LLM"]
MODEL_GROUPS = {
    "VLM":             ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone Decoder": ["InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"],
    "Standalone LLM":  ["Qwen2.5-7B-Instruct", "Qwen3-8B", "Qwen3-8B (think)", "Vicuna-7B", "Mistral-7B"],
}

FREE_TEXT_OPS = ["act", "attr", "cause", "comp", "ident", "know", "spat", "temp", "text"]


def display_name(model: str) -> str:
    if "(think)" in model:
        return "Qwen3 (think)"
    name = model.replace(" (LM)", "")
    for s in ["-8B", "-7B", "-Instruct"]:
        name = name.replace(s, "")
    return name.strip()


def compute_delta_rho(df: pd.DataFrame, group_col: str) -> dict[str, tuple[float, float]]:
    """Compute Spearman ρ between human HH Δ and model HM Δ across groups.

    Δ = mean SBERT on Pronominalized (A) - mean SBERT on Original (C), per group.
    """
    # Free-text only
    ft = df[df["op"].isin(FREE_TEXT_OPS)].copy()

    # Human HH Δ per group
    hh = ft[ft["pair_type"] == "HH"].groupby([group_col, "variant"])["sbert_score"].mean().unstack("variant")
    if "C" not in hh.columns or "A" not in hh.columns:
        return {}
    hh_delta = (hh["A"] - hh["C"]).dropna()

    results = {}
    hm = ft[ft["pair_type"] == "HM"]

    for model in MATCHED_MODELS:
        sub = hm[hm["subject_2"] == model].groupby([group_col, "variant"])["sbert_score"].mean().unstack("variant")
        if "C" not in sub.columns or "A" not in sub.columns:
            results[model] = (float("nan"), float("nan"))
            continue
        model_delta = (sub["A"] - sub["C"]).dropna()

        # Align on common groups
        common = hh_delta.index.intersection(model_delta.index)
        if len(common) < 5:
            results[model] = (float("nan"), float("nan"))
            continue
        rho, p = stats.spearmanr(hh_delta.loc[common].values, model_delta.loc[common].values)
        results[model] = (rho, p)

    return results


def fmt_rho(rho: float, p: float, bold: bool = False, underline: bool = False) -> str:
    if np.isnan(rho):
        return "--"
    sig = "" if p < 0.05 else r"$^{\dagger}$"
    s = f"{rho:+.3f}".replace("+0.", ".").replace("-0.", "$-$.")
    if rho >= 0 and not s.startswith("."):
        s = "." + s.split(".", 1)[1] if "." in s else s
    # simpler: just format as .3f with sign
    s = f"{rho:.3f}".lstrip("0") or "0"
    if rho < 0:
        s = f"$-${abs(rho):.3f}".replace("0.", ".")
    else:
        s = f"{rho:.3f}".replace("0.", ".")
    s = s + sig
    if bold:
        return rf"\textbf{{{s}}}"
    if underline:
        return rf"\underline{{{s}}}"
    return s


def main():
    exports = ROOT / "analysis/session2/exports"
    df = pd.read_parquet(exports / "pair_cache_cleaned.parquet")

    op_rho = compute_delta_rho(df, "op")
    ent_rho = compute_delta_rho(df, "ent")

    # Find best/second per column
    def rank_col(rho_dict: dict) -> tuple[float, float]:
        vals = [v[0] for v in rho_dict.values() if not np.isnan(v[0])]
        vals_sorted = sorted(vals, reverse=True)
        best = vals_sorted[0] if vals_sorted else float("nan")
        second = vals_sorted[1] if len(vals_sorted) > 1 else float("nan")
        return best, second

    op_best, op_second = rank_col(op_rho)
    ent_best, ent_second = rank_col(ent_rho)

    lines = [
        r"\begin{table}[ht]",
        r"\small\centering",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Per-model Spearman $\rho$ between human HH $\Delta$ and model HM $\Delta$ "
        r"across operation and entity groups ($\Delta$ = Pronominalized $-$ Original SBERT, "
        r"free-text questions only). "
        r"This measures concordance on \emph{degradation patterns}: whether models lose alignment "
        r"on the same groups that humans do. "
        r"$\dagger$ = not significant ($p{>}.05$). "
        r"\textbf{Bold}: highest; \underline{underline}: second highest per column.}",
        r"\label{tab:per_model_r}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Op} $\rho$ & \textbf{Ent} $\rho$ \\",
        r"\midrule",
    ]

    for grp in GROUP_ORDER:
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{grp}s}}}} \\")
        for model in MODEL_GROUPS[grp]:
            op_r, op_p = op_rho.get(model, (float("nan"), float("nan")))
            ent_r, ent_p = ent_rho.get(model, (float("nan"), float("nan")))
            op_bold = not np.isnan(op_r) and np.isclose(op_r, op_best)
            op_under = not np.isnan(op_r) and not op_bold and not np.isnan(op_second) and np.isclose(op_r, op_second)
            ent_bold = not np.isnan(ent_r) and np.isclose(ent_r, ent_best)
            ent_under = not np.isnan(ent_r) and not ent_bold and not np.isnan(ent_second) and np.isclose(ent_r, ent_second)
            lines.append(
                f"{display_name(model)} & {fmt_rho(op_r, op_p, op_bold, op_under)} "
                f"& {fmt_rho(ent_r, ent_p, ent_bold, ent_under)} \\\\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = OUT_DIR / "per_model_r_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")

    # Print values for inspection
    print("\nPer-model Δ-ρ (Op, Ent):")
    for grp in GROUP_ORDER:
        print(f"\n  {grp}:")
        for model in MODEL_GROUPS[grp]:
            op_r, op_p = op_rho.get(model, (float("nan"), float("nan")))
            ent_r, ent_p = ent_rho.get(model, (float("nan"), float("nan")))
            sig_op = "" if np.isnan(op_p) or op_p < 0.05 else " (ns)"
            sig_ent = "" if np.isnan(ent_p) or ent_p < 0.05 else " (ns)"
            print(f"    {display_name(model):<22} Op={op_r:+.3f}{sig_op}  Ent={ent_r:+.3f}{sig_ent}")


if __name__ == "__main__":
    main()
