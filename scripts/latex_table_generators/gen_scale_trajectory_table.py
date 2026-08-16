"""
Scale trajectory table across two model families with size variation:
  - Qwen3-VL  (2B, 4B, 8B) : VLM and LM backbone, both complete
  - LLaVA-Vicuna (7B, 13B) : VLM and LM backbone, both complete

Metrics (blind condition, variant C / original questions):
  - VQA Accuracy : from responses_model_blind.csv (ground-truth match)
  - HM SBERT     : from pair_cache_cleaned_blind.parquet (human-model alignment)

Outputs: latex/AAAI2026/LaTeX/tables/scale_trajectory_qwen3vl.tex
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables" / "supp"

# Families: list of (size_label, vlm_id, lm_id)
FAMILIES = {
    "Qwen3-VL": [
        ("2B", "Qwen3-VL-2B",      "Qwen3-VL-2B (LM)"),
        ("4B", "Qwen3-VL-4B",      "Qwen3-VL-4B (LM)"),
        ("8B", "Qwen3-VL-8B",      "Qwen3-VL-8B (LM)"),
    ],
    "LLaVA-Vicuna": [
        ("7B",  "LLaVA-Vicuna",      "LLaVA-Vicuna (LM)"),
        ("13B", "LLaVA-Vicuna-13B",  "LLaVA-Vicuna-13B (LM)"),
    ],
}

MIN_N = 80


def load_accuracy(variant: str = "C") -> dict:
    df = pd.read_csv(ROOT / "analysis" / "session2" / "exports" / "responses_model_blind.csv")
    df = df[df["variant"] == variant]
    all_ids = [m for rows in FAMILIES.values() for _, v, l in rows for m in (v, l)]
    out = {}
    for m in all_ids:
        sub = df[df["model"] == m]
        out[m] = (sub["accuracy"].mean(), len(sub)) if len(sub) >= MIN_N else (None, len(sub))
    return out


def load_sbert(variant: str = "C") -> dict:
    df = pd.read_parquet(
        ROOT / "analysis" / "session2" / "exports" / "pair_cache_cleaned_blind.parquet"
    )
    hm = df[(df["pair_type"] == "HM") & (df["variant"] == variant)]
    all_ids = [m for rows in FAMILIES.values() for _, v, l in rows for m in (v, l)]
    out = {}
    for m in all_ids:
        sub = hm[hm["subject_2"] == m]
        out[m] = (sub["sbert_score"].mean(), len(sub)) if len(sub) >= MIN_N else (None, len(sub))
    return out


def fmt(val):
    return "---" if val is None else f"{val:.3f}"


def cell(val, best):
    if val is None:
        return r"\multicolumn{1}{c}{---}"
    s = f"{val:.3f}"
    return r"\textbf{" + s + "}" if abs(val - best) < 1e-6 else s


def main():
    acc   = load_accuracy("C")
    sbert = load_sbert("C")

    # Console summary
    for fam, rows in FAMILIES.items():
        print(f"\n{fam}")
        for size, vlm, lm in rows:
            va = fmt(acc.get(vlm, (None,))[0])
            vs = fmt(sbert.get(vlm, (None,))[0])
            la = fmt(acc.get(lm,  (None,))[0])
            ls = fmt(sbert.get(lm,  (None,))[0])
            print(f"  {size}: VLM acc={va} sbert={vs} | LM acc={la} sbert={ls}")

    # Gather all values per column for bold-best
    all_vlm_acc   = [acc.get(v,(None,))[0]   for rows in FAMILIES.values() for _,v,_ in rows]
    all_vlm_sbert = [sbert.get(v,(None,))[0] for rows in FAMILIES.values() for _,v,_ in rows]
    all_lm_acc    = [acc.get(l,(None,))[0]   for rows in FAMILIES.values() for _,_,l in rows]
    all_lm_sbert  = [sbert.get(l,(None,))[0] for rows in FAMILIES.values() for _,_,l in rows]
    best_va = max(v for v in all_vlm_acc   if v is not None)
    best_vs = max(v for v in all_vlm_sbert if v is not None)
    best_la = max(v for v in all_lm_acc    if v is not None)
    best_ls = max(v for v in all_lm_sbert  if v is not None)

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Family}} & \multirow{2}{*}{\textbf{Params}} &",
        r"\multicolumn{2}{c}{\textbf{VLM}} & \multicolumn{2}{c}{\textbf{LM backbone}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"& & \textbf{Acc}$\uparrow$ & \textbf{SBERT}$\uparrow$"
        r" & \textbf{Acc}$\uparrow$ & \textbf{SBERT}$\uparrow$ \\",
        r"\midrule",
    ]

    for fam, rows in FAMILIES.items():
        n = len(rows)
        first = True
        for size, vlm, lm in rows:
            va = acc.get(vlm,   (None,))[0]
            vs = sbert.get(vlm, (None,))[0]
            la = acc.get(lm,    (None,))[0]
            ls = sbert.get(lm,  (None,))[0]
            fam_cell = (r"\multirow{" + str(n) + r"}{*}{" + fam + "}") if first else ""
            lines.append(
                f"{fam_cell} & {size} & "
                f"{cell(va, best_va)} & {cell(vs, best_vs)} & "
                f"{cell(la, best_la)} & {cell(ls, best_ls)} \\\\"
            )
            first = False
        lines.append(r"\midrule")

    # Replace last \midrule with \bottomrule
    lines[-1] = r"\bottomrule"

    lines += [
        r"\end{tabular}",
        r"\caption{Scale trajectory for Qwen3-VL and LLaVA-Vicuna under the blind condition "
        r"(variant C, original questions). "
        r"\textbf{Acc}: VQA accuracy against ground-truth labels. "
        r"\textbf{SBERT}: mean HM semantic similarity to human answers. "
        r"LM backbone strips the vision encoder; VLM receives a blank image. "
        r"In both families, LM backbone SBERT exceeds the paired VLM at every size, "
        r"while VQA accuracy does not track SBERT monotonically. "
        r"\textbf{Bold}: best per column across both families.}",
        r"\label{tab:scale_trajectory_qwen3vl}",
        r"\end{table}",
    ]

    out = OUT_DIR / "scale_trajectory_qwen3vl.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
