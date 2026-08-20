"""
Generate group_band_coverage.tex — per-model band coverage table.

For each op/ent group column:
  - Human LOO row: shows mean sHH value (reference)
  - Group mean rows (VLM / Backbone / SA-LLM): colored cell if within/below/above human LOOCV p5–p95 band
  - Individual model rows: same check

Band computation: for each (dim, grp), LOO across participants — each participant's mean
sbert_score for that group, then p5/p95 across participants (text-answer-type questions only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

from figures.helpers import filter_abstained_pairs
from analysis.utils.vqa import VQAAnswerMapper

OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/group_band_coverage.tex"

MODEL_GROUPS = {
    "VLM": ["InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone Decoder": [
        "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)",
        "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    ],
    "Standalone LLM": [
        "Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B",
    ],
}
GROUP_LABELS = {"VLM": "VLM", "Backbone Decoder": "Backbone", "Standalone LLM": "SA-LLM"}

OP_GROUPS = [
    ("act", "Act."), ("attr", "Attr."), ("cause", "Caus."), ("comp", "Comp."),
    ("count", "Count"), ("exist", "Exist."), ("ident", "Ident."), ("know", "Know."), ("spat", "Spat."),
]
ENT_GROUPS = [
    ("animal", "Animal"), ("food", "Food"), ("object", "Object"), ("other", "Ent-Oth"),
    ("person", "Person"), ("place", "Place"), ("product", "Prod."), ("text", "Text"), ("vehicle", "Veh."),
]

def load_data():
    pc = pd.read_parquet(ROOT / "analysis/session2/exports/pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)

    # Filter to free-text questions
    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other")
                 for qid, ann in mapper.annotations.items()}
    text_qids = {qid for qid, atype in qid2atype.items() if atype == "other"}
    pc = pc[pc["question_id"].astype(int).isin(text_qids)].copy()

    hh = pc[pc["pair_type"] == "HH"].copy()
    all_models = {m for ms in MODEL_GROUPS.values() for m in ms}
    hm = pc[(pc["pair_type"] == "HM") & pc["subject_2"].isin(all_models)].copy()
    return hh, hm


def compute_loocv_bands(hh: pd.DataFrame) -> dict:
    """Return {(dim, grp): (p5, p95, mean_sHH)} for groups with enough data."""
    result = {}
    for dim in ("op", "ent"):
        for grp, grp_df in hh.groupby(dim):
            per_part = [
                pdata["sbert_score"].mean()
                for _, pdata in grp_df.groupby("subject_1")
                if len(pdata) >= 2
            ]
            mean_shh = grp_df["sbert_score"].mean()
            if len(per_part) >= 5:
                arr = np.array(per_part)
                result[(dim, grp)] = (
                    float(np.percentile(arr, 5)),
                    float(np.percentile(arr, 95)),
                    float(mean_shh),
                )
    return result


def band_cell(score: float, band) -> str:
    if band is None:
        return "---"
    p5, p95, _ = band
    val = f"{score:.3f}".lstrip("0")
    if score < p5:
        return rf"\cellcolor[HTML]{{F7EBD9}}{val}"
    elif score > p95:
        return rf"\cellcolor[HTML]{{FFCCCC}}{val}"
    else:
        return rf"\cellcolor[HTML]{{D9EAF3}}{val}"


def main() -> None:
    hh, hm = load_data()
    bands = compute_loocv_bands(hh)

    all_groups = (
        [("op", key, label) for key, label in OP_GROUPS] +
        [("ent", key, label) for key, label in ENT_GROUPS]
    )

    # Filter to groups that have a band
    valid_groups = [(dim, key, label) for dim, key, label in all_groups if (dim, key) in bands]

    op_valid = [(dim, key, label) for dim, key, label in valid_groups if dim == "op"]
    ent_valid = [(dim, key, label) for dim, key, label in valid_groups if dim == "ent"]
    n_op = len(op_valid)
    n_ent = len(ent_valid)
    n_total = n_op + n_ent

    col_spec = "l" + "c" * n_total

    # Header labels
    op_labels = [label for _, _, label in op_valid]
    ent_labels = [label for _, _, label in ent_valid]
    all_labels = op_labels + ent_labels

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\caption[Per-group mean $\sHM$ vs.\ human LOOCV SBERT band]"
        r"{Per-group mean $\sHM$ against the human LOOCV SBERT p5--p95 band (free-text questions, abstention-filtered). "
        r"\colorbox[HTML]{D9EAF3}{Blue}~= within human band; "
        r"\colorbox[HTML]{F7EBD9}{Orange}~= below band; "
        r"\colorbox[HTML]{FFCCCC}{Red}~= above band (over-aligned). "
        r"Human row shows mean $\sHH$ per group as reference. "
        r"Group mean rows aggregate all models within the class.}",
        r"\label{tab:group_band_coverage}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Multicolumn header
    multirow_header = r"\multirow{2}{*}{\textbf{Model}}"
    if n_op > 0 and n_ent > 0:
        multirow_header += (
            r" & \multicolumn{" + str(n_op) + r"}{c}{\textbf{Operation groups}}"
            r" & \multicolumn{" + str(n_ent) + r"}{c}{\textbf{Entity groups}} \\"
        )
        lines.append(multirow_header)
        lines.append(
            r"\cmidrule(lr){2-" + str(1 + n_op) + r"}"
            r"\cmidrule(lr){" + str(2 + n_op) + "-" + str(1 + n_op + n_ent) + r"}"
        )
    else:
        lines.append(multirow_header + r" \\")

    # Rotated column labels
    lines.append(
        r"& " + " & ".join(rf"\rotatebox{{60}}{{\textbf{{{c}}}}}" for c in all_labels) + r" \\"
    )
    lines.append(r"\midrule")

    # Human LOO row
    human_cells = []
    for dim, key, _ in valid_groups:
        band = bands.get((dim, key))
        if band is None:
            human_cells.append("---")
        else:
            _, _, mean_shh = band
            human_cells.append(f"{mean_shh:.3f}".lstrip("0"))
    lines.append(r"\textbf{Human (sHH)} & " + " & ".join(human_cells) + r" \\")
    lines.append(r"\midrule")

    # Model group sections
    group_names = list(MODEL_GROUPS.keys())
    for gi, (group_name, models) in enumerate(MODEL_GROUPS.items()):
        # Group mean row
        mean_cells = []
        for dim, key, _ in valid_groups:
            sub = hm[hm["subject_2"].isin(models)]
            sub_grp = sub[sub[dim] == key]
            if len(sub_grp) == 0:
                mean_cells.append("---")
            else:
                score = sub_grp["sbert_score"].mean()
                band = bands.get((dim, key))
                mean_cells.append(band_cell(score, band))
        label = GROUP_LABELS[group_name]
        lines.append(
            rf"\rowcolor{{gray!12}}\textit{{{label} Mean}} & " + " & ".join(mean_cells) + r" \\"
        )

        # Individual model rows
        for model in models:
            model_cells = []
            for dim, key, _ in valid_groups:
                sub = hm[(hm["subject_2"] == model) & (hm[dim] == key)]
                if len(sub) == 0:
                    model_cells.append("---")
                else:
                    score = sub["sbert_score"].mean()
                    band = bands.get((dim, key))
                    model_cells.append(band_cell(score, band))
            lines.append(rf"\quad {model} & " + " & ".join(model_cells) + r" \\")

        if gi < len(group_names) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
