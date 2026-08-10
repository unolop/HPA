"""
Generate per-model degradation tables (Δ = sHH_orig − sHH_pron) for operation
and entity groups, saved as LaTeX to the tables directory.

Rows:  Human LOO (reference) + each model, grouped by class
Cols:  groups sorted by human Δ (descending)
Color: orange  = model over-degrades vs human by > THRESH
       blue    = model under-degrades vs human by > THRESH
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))
from helpers import filter_abstained_pairs

sys.path.insert(0, str(ROOT))
from analysis.utils.vqa import VQAAnswerMapper

EXPORTS   = ROOT / "analysis/session2/exports"
TABLE_DIR = ROOT / "latex/AAAI2026/LaTeX/tables"

OVER_THRESH  =  0.03   # model_delta - human_delta > this → orange
UNDER_THRESH = -0.03   # model_delta - human_delta < this → blue

MODEL_GROUPS = {
    "VLMs": [
        "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B",
        "LLaVA-Mistral", "LLaVA-Vicuna",
    ],
    "Backbone Decoders": [
        "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)",
        "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    ],
    "SA-LLMs": [
        "Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B",
    ],
}

MODEL_LABEL = {
    "InternVL-8B":          "InternVL",
    "Qwen3-VL-8B":          "Qwen3-VL",
    "LLaVA-1.5-7B":         "LLaVA-1.5",
    "LLaVA-Mistral":        "LLaVA-Mistral",
    "LLaVA-Vicuna":         "LLaVA-Vicuna",
    "InternVL-8B (LM)":     "InternVL",
    "Qwen3-VL-8B (LM)":     "Qwen3-VL",
    "LLaVA-1.5 (LM)":       "LLaVA-1.5",
    "LLaVA-Mistral (LM)":   "LLaVA-Mistral",
    "LLaVA-Vicuna (LM)":    "LLaVA-Vicuna",
    "Qwen3-8B":             "Qwen3",
    "Qwen3-8B (think)":     "Qwen3 (think)",
    "Qwen2.5-7B-Instruct":  "Qwen2.5",
    "Vicuna-7B":            "Vicuna",
}

OP_COL_LABEL = {
    "ident": "Identity", "know": "World K.", "act": "Action",
    "cause": "Causality", "attr": "Attribute", "spat": "Spatial",
    "comp": "Comparison", "temp": "Temporal", "text": "Text R.",
}
ENT_COL_LABEL = {
    "product": "Product", "animal": "Animal", "food": "Food",
    "object": "Object", "place": "Place", "person": "Person",
    "vehicle": "Vehicle", "other": "Other", "text": "Text",
}


def load_data():
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)
    # free-text only
    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other")
                 for qid, ann in mapper.annotations.items()}
    keep_qids = {qid for qid, atype in qid2atype.items() if atype == "other"}
    pc = pc[pc["question_id"].astype(int).isin(keep_qids)]

    keep_models = {m for ms in MODEL_GROUPS.values() for m in ms}
    hh = pc[pc["pair_type"] == "HH"].copy()
    hm = pc[(pc["pair_type"] == "HM") & pc["subject_2"].isin(keep_models)].copy()
    return hh, hm


def compute_deltas(df, dim):
    """Return (group → delta) for each model/subject (or 'Human' for hh)."""
    var = df.groupby([dim, "variant"])["sbert_score"].mean().unstack("variant")
    delta = {}
    for grp in var.index:
        c = var.loc[grp, "C"] if "C" in var.columns else np.nan
        a = var.loc[grp, "A"] if "A" in var.columns else np.nan
        delta[grp] = c - a
    return delta


def fmt_cell(val, human_val):
    """Format a delta cell with color based on over/under-degradation."""
    if np.isnan(val):
        return "---"
    num = f"{val:+.2f}" if val >= 0 else f"{val:.2f}"
    if np.isnan(human_val):
        return num
    gap = val - human_val
    if gap > OVER_THRESH:
        return rf"\cellcolor{{orange!25}}{num}"
    elif gap < UNDER_THRESH:
        return rf"\cellcolor{{blue!10}}{num}"
    else:
        return num


def make_table(hh, hm, dim, col_label_map, label, caption, filename):
    # Human deltas
    human_delta = compute_deltas(hh, dim)
    groups = sorted(human_delta.keys(),
                    key=lambda g: human_delta[g], reverse=True)

    col_labels = [col_label_map.get(g, g) for g in groups]
    n_cols = len(groups)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.92}")
    lines.append(
        rf"\caption{{{caption}}}"
    )
    lines.append(rf"\label{{{label}}}")

    # tabular: model col + n_cols value cols
    col_spec = r">{\raggedright\arraybackslash}p{2.1cm}" + "r" * n_cols
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # rotated column headers
    rot_heads = " & ".join(
        rf"\rotatebox{{60}}{{\textbf{{{c}}}}}" for c in col_labels
    )
    lines.append(rf"\textbf{{Model}} & {rot_heads} \\")
    lines.append(r"\midrule")

    # Human reference row
    human_cells = " & ".join(
        f"{human_delta.get(g, float('nan')):+.2f}"
        if not np.isnan(human_delta.get(g, float("nan")))
        else "---"
        for g in groups
    )
    lines.append(
        rf"\rowcolor{{gray!12}}\textbf{{Human LOO}} & {human_cells} \\"
    )
    lines.append(r"\midrule")

    # Per-class model rows
    cls_keys = list(MODEL_GROUPS.keys())
    for ci, (cls_name, models) in enumerate(MODEL_GROUPS.items()):
        n_models = len(models)
        lines.append(
            rf"\multicolumn{{{n_cols + 1}}}{{c}}{{\textit{{{cls_name}}}}} \\"
        )
        lines.append(r"\midrule")
        for mi, model in enumerate(models):
            sub = hm[hm["subject_2"] == model]
            if sub.empty:
                continue
            model_delta = compute_deltas(sub, dim)
            label_str = MODEL_LABEL.get(model, model)
            cells = []
            for g in groups:
                val = model_delta.get(g, float("nan"))
                hval = human_delta.get(g, float("nan"))
                cells.append(fmt_cell(val, hval))
            row = rf"{label_str} & " + " & ".join(cells) + r" \\"
            lines.append(row)
        if ci < len(cls_keys) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = TABLE_DIR / filename
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


def main():
    hh, hm = load_data()

    make_table(
        hh, hm, "op", OP_COL_LABEL,
        label="tab:op_degradation_per_model",
        caption=(
            r"Per-model original-to-pronominalized SBERT degradation ($\Delta = S_\mathrm{Orig} - S_\mathrm{Pron}$) "
            r"by operation group (free-text questions only; columns sorted by human $\Delta$, descending). "
            r"\colorbox{orange!25}{Orange}: model over-degrades vs.\ human by $>{+}0.03$. "
            r"\colorbox{blue!10}{Blue}: model under-degrades vs.\ human by $>{-}0.03$."
        ),
        filename="op_degradation_per_model.tex",
    )

    make_table(
        hh, hm, "ent", ENT_COL_LABEL,
        label="tab:ent_degradation_per_model",
        caption=(
            r"Per-model original-to-pronominalized SBERT degradation ($\Delta = S_\mathrm{Orig} - S_\mathrm{Pron}$) "
            r"by entity group (free-text questions only; columns sorted by human $\Delta$, descending). "
            r"\colorbox{orange!25}{Orange}: model over-degrades vs.\ human by $>{+}0.03$. "
            r"\colorbox{blue!10}{Blue}: model under-degrades vs.\ human by $>{-}0.03$."
        ),
        filename="ent_degradation_per_model.tex",
    )


if __name__ == "__main__":
    main()
