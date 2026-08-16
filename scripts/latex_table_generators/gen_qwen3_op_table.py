"""Generate per-op SBERT breakdown table for Qwen3-8B vs Qwen3-8B (think).

Shows mean SBERT per operation type averaged across all variants (C/B/A),
with deltas relative to the human HH baseline.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables" / "supp"

PARQUET = ROOT / "analysis/session2/exports/pair_cache_cleaned.parquet"
MODEL_BASE  = "Qwen3-8B"
MODEL_THINK = "Qwen3-8B (think)"

# Display names for ops
OP_DISPLAY = {
    "act":   "Action",
    "attr":  "Attribute",
    "cause": "Causality",
    "comp":  "Comparison",
    "count": "Count",
    "exist": "Existence",
    "ident": "Identity",
    "know":  "World Knowledge",
    "spat":  "Spatial",
    "temp":  "Temporal",
    "text":  "Text Reading",
}

# Group: structured (yes/no + count) vs free-text
STRUCTURED_OPS = ["exist", "count"]
FREETEXT_OPS   = ["act", "attr", "cause", "comp", "ident", "know", "spat", "temp", "text"]


def fmt_cell(val: float, delta: float) -> str:
    """Format as 'val (±Δ)'."""
    if np.isnan(val):
        return r"---"
    d_str = f"{delta:+.3f}".replace("+0.", "+.").replace("-0.", "$-$.")
    # handle exact zero
    if delta == 0.0:
        d_str = ".000"
    v_str = f"{val:.3f}".replace("0.", ".")
    return rf"{v_str} \small({d_str})"


def fmt_hh(val: float) -> str:
    if np.isnan(val):
        return "---"
    return f"{val:.3f}".replace("0.", ".")


def main():
    df = pd.read_parquet(PARQUET)

    hh      = df[df["pair_type"] == "HH"]
    hm_base = df[(df["pair_type"] == "HM") & (df["subject_2"] == MODEL_BASE)]
    hm_think= df[(df["pair_type"] == "HM") & (df["subject_2"] == MODEL_THINK)]

    hh_means    = hh.groupby("op")["sbert_score"].mean()
    base_means  = hm_base.groupby("op")["sbert_score"].mean()
    think_means = hm_think.groupby("op")["sbert_score"].mean()

    def get_row(op):
        hh_v    = hh_means.get(op, np.nan)
        base_v  = base_means.get(op, np.nan)
        think_v = think_means.get(op, np.nan)
        d_base  = base_v  - hh_v if not np.isnan(base_v)  else np.nan
        d_think = think_v - hh_v if not np.isnan(think_v) else np.nan
        return hh_v, base_v, d_base, think_v, d_think

    # Overall means
    hh_all    = hh["sbert_score"].mean()
    base_all  = hm_base["sbert_score"].mean()
    think_all = hm_think["sbert_score"].mean()

    lines = [
        r"\begin{table}[t]",
        r"\small\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Per-operation SBERT breakdown for Qwen3-8B vs.\ Qwen3-8B (think), "
        r"averaged across all variants (Orig./Weak./Pron.). "
        r"Each cell shows mean SBERT with delta relative to the human HH baseline in parentheses. "
        r"Think mode gains most on \textit{Count} ($\Delta{=}{+}.125$ vs.\ no-think) "
        r"but loses on \textit{World Knowledge} ($\Delta{=}{-}.077$) and \textit{Action} ($\Delta{=}{-}.090$), "
        r"explaining its lower all-question Pearson~$r$ despite higher structured-question alignment.}",
        r"\label{tab:qwen3_op_breakdown}",
        r"\begin{tabular}{lcrr}",
        r"\toprule",
        r"\textbf{Operation} & \textbf{Human (HH)} & \textbf{Qwen3-8B ($\Delta$)} & \textbf{Qwen3-8B\,(think) ($\Delta$)} \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Structured}} \\",
    ]

    for op in STRUCTURED_OPS:
        if op not in OP_DISPLAY:
            continue
        hh_v, base_v, d_base, think_v, d_think = get_row(op)
        lines.append(
            rf"\quad {OP_DISPLAY[op]} & {fmt_hh(hh_v)} & {fmt_cell(base_v, d_base)} & {fmt_cell(think_v, d_think)} \\"
        )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Free-text}} \\")

    for op in FREETEXT_OPS:
        if op not in OP_DISPLAY:
            continue
        hh_v, base_v, d_base, think_v, d_think = get_row(op)
        lines.append(
            rf"\quad {OP_DISPLAY[op]} & {fmt_hh(hh_v)} & {fmt_cell(base_v, d_base)} & {fmt_cell(think_v, d_think)} \\"
        )

    lines += [
        r"\midrule",
        rf"Overall & {fmt_hh(hh_all)} & {fmt_cell(base_all, base_all - hh_all)} & {fmt_cell(think_all, think_all - hh_all)} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = OUT_DIR / "qwen3_op_breakdown.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")
    print("\nPreview:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
