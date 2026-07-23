"""Generate per-model HM SBERT instruction effect table (blind vs inst_blind, all 3 variants)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables"

MATCHED_MODELS = [
    "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
    "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    "Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B",
]

GROUP_ORDER = ["VLM", "Backbone Decoder", "Standalone LLM", "Think"]
MODEL_GROUPS = {
    "VLM":            ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone Decoder":["InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"],
    "Standalone LLM": ["Qwen2.5-7B-Instruct", "Qwen3-8B", "Vicuna-7B"],
    "Think":          ["Qwen3-8B (think)"],
}
SECTION_TITLES = {
    "VLM": "VLMs", "Backbone Decoder": "Backbone decoders",
    "Standalone LLM": "Standalone LLMs", "Think": "Think models",
}
VARIANTS = ["C", "B", "A"]
VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronom."}


def display_name(model: str) -> str:
    if "(think)" in model:
        return "Qwen3 (think)"
    name = model.replace(" (LM)", "")
    for s in ["-8B", "-7B", "-Instruct"]:
        name = name.replace(s, "")
    return name.replace("LLaVA-1.5", "LLaVA-1.5").strip()


def load_hm_sbert(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    hm = df[df["pair_type"] == "HM"].copy()
    hm = hm[hm["subject_2"].isin(MATCHED_MODELS)]
    return hm.groupby(["subject_2", "variant"])["sbert_score"].mean().reset_index()


def fmt(v: float, bold: bool = False, underline: bool = False) -> str:
    if pd.isna(v):
        return "--"
    s = f"{v:.3f}".lstrip("0") or "0"
    if bold:
        return rf"\textbf{{{s}}}"
    if underline:
        return rf"\underline{{{s}}}"
    return s


def fmt_delta_inline(delta: float) -> str:
    if pd.isna(delta) or abs(delta) < 0.0005:
        return ""
    s = f"{delta:+.3f}"
    if s.startswith("+0."):
        s = "+" + s[2:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    color = "teal" if delta > 0 else "red"
    return rf" {{\scriptsize (\textcolor{{{color}}}{{{s}}})}}"


def main():
    exports = ROOT / "analysis/session2/exports"
    blind = load_hm_sbert(exports / "pair_cache_cleaned_blind.parquet")
    inst  = load_hm_sbert(exports / "pair_cache_cleaned.parquet")

    blind = blind.rename(columns={"sbert_score": "blind"})
    inst  = inst.rename(columns={"sbert_score": "inst"})
    merged = blind.merge(inst, on=["subject_2", "variant"])
    merged["delta"] = merged["inst"] - merged["blind"]
    merged = merged.rename(columns={"subject_2": "model"})

    # Rankings over blind SBERT per variant (higher = better)
    rankings = {}
    for v in VARIANTS:
        sub = merged[merged["variant"] == v]
        vals = sub["blind"].dropna().tolist()
        uniq = sorted(set(vals), reverse=True)
        rankings[v] = (uniq[0] if uniq else float("nan"),
                       uniq[1] if len(uniq) > 1 else float("nan"))

    # Build table
    n_col_spans = len(VARIANTS)
    col_spec = "l" + "c" * n_col_spans
    header_spans = " & ".join(
        rf"\multicolumn{{1}}{{c}}{{\textbf{{{VARIANT_LABELS[v]}}}}}" for v in VARIANTS
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Model & {header_spans} \\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}",
        rf"& \small blind ($\Delta$) & \small blind ($\Delta$) & \small blind ($\Delta$) \\",
        r"\midrule",
    ]

    def get_row(model: str) -> dict:
        return {v: merged[(merged["model"] == model) & (merged["variant"] == v)].iloc[0]
                if not merged[(merged["model"] == model) & (merged["variant"] == v)].empty
                else None
                for v in VARIANTS}

    for grp in GROUP_ORDER:
        lines.append(rf"\multicolumn{{{n_col_spans + 1}}}{{c}}{{\textit{{{SECTION_TITLES[grp]}}}}} \\")
        lines.append(r"\midrule")
        for model in MODEL_GROUPS[grp]:
            row_data = get_row(model)
            cells = []
            for v in VARIANTS:
                r = row_data[v]
                if r is None:
                    cells.append("--")
                    continue
                best, second = rankings[v]
                bold = np.isclose(r["blind"], best)
                under = (not np.isnan(second)) and np.isclose(r["blind"], second)
                s = fmt(r["blind"], bold=bold, underline=under)
                cells.append(s + fmt_delta_inline(r["delta"]))
            lines.append(f"{display_name(model)} & {' & '.join(cells)} \\\\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-model HM SBERT under the blind condition with instruction effect $\Delta$ in parentheses "
        r"(\textcolor{teal}{teal} = instruction improves alignment). "
        r"Pooled over all question--human pairs per variant. "
        r"\textbf{Bold}: best per column; \underline{underline}: second best.}",
        r"\label{tab:sbert_inst_effect}",
        r"\end{table}",
    ]

    out = OUT_DIR / "sbert_inst_effect.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")

    # Print group means for reference
    print("\nGroup means (Δ blind→inst, pooled across variants):")
    for grp, models in MODEL_GROUPS.items():
        sub = merged[merged["model"].isin(models)]
        print(f"  {grp}: blind={sub['blind'].mean():.3f}  inst={sub['inst'].mean():.3f}  Δ={sub['delta'].mean():.3f}")


if __name__ == "__main__":
    main()
