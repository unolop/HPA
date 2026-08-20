"""
Generator for the yes/no response distribution table (appendix).

Layout: one row per model, columns: yes% | Oth%
Blind value shown; instruction effect Δ in colored parentheses (teal=+, red=−).
Abstention column shows all abstentions (hard + soft) via shared classifier.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/yesno_distribution_all_models.tex"

EXPORTS = ROOT / "analysis/session2/exports"
VQA_ANN_TRAIN = ROOT / "dataset/vqa/v2_mscoco_train2014_annotations.json"
VQA_ANN_VAL   = ROOT / "dataset/vqa/v2_mscoco_val2014_annotations.json"
SUBSET_JSONL   = ROOT / "dataset/vqa/vqa1k_control_study_subset.jsonl"

from analysis.utils.abstention import classify as _abst_classify, is_abstained

def classify(resp: str) -> str:
    r = str(resp).strip().lower()
    if is_abstained(_abst_classify(resp, None)): return "abst"
    if re.match(r"^yes\b", r): return "yes"
    if re.match(r"^no\b",  r): return "no"
    return "other"


# ── model display / ordering ───────────────────────────────────────────────
GROUPS = {
    "VLM": [
        ("InternVL-1B",        "InternVL3.5", "1B"),
        ("InternVL-2B",        "InternVL3.5", "2B"),
        ("InternVL-8B",        "InternVL3.5", "8B"),
        ("LLaVA-1.5-7B",       "LLaVA-1.5",  "7B"),
        ("LLaVA-Mistral",      "LLaVA-Mistral", "7B"),
        ("LLaVA-Vicuna",       "LLaVA-Vicuna",  "7B"),
        ("LLaVA-Vicuna-13B",   "LLaVA-Vicuna",  "13B"),
        ("Qwen3-VL-2B",        "Qwen3-VL",   "2B"),
        ("Qwen3-VL-4B",        "Qwen3-VL",   "4B"),
        ("Qwen3-VL-8B",        "Qwen3-VL",   "8B"),
    ],
    "Backbone Decoder": [
        ("InternVL-1B (LM)",       "InternVL3.5", "1B"),
        ("InternVL-2B (LM)",       "InternVL3.5", "2B"),
        ("InternVL-8B (LM)",       "InternVL3.5", "8B"),
        ("LLaVA-1.5 (LM)",        "LLaVA-1.5",  "7B"),
        ("LLaVA-Mistral (LM)",    "LLaVA-Mistral", "7B"),
        ("LLaVA-Vicuna (LM)",     "LLaVA-Vicuna",  "7B"),
        ("LLaVA-Vicuna-13B (LM)", "LLaVA-Vicuna",  "13B"),
        ("Qwen3-VL-2B (LM)",      "Qwen3-VL",   "2B"),
        ("Qwen3-VL-4B (LM)",      "Qwen3-VL",   "4B"),
        ("Qwen3-VL-8B (LM)",      "Qwen3-VL",   "8B"),
    ],
    "Standalone LLM": [
        ("Mistral-7B",           "Mistral",          "7B"),
        ("Phi-3.5-mini",         "Phi-3.5-mini",     "3.8B"),
        ("Qwen2.5-7B-Instruct",  "Qwen2.5-Instruct", "7B"),
        ("Qwen3-0.6B",           "Qwen3",            "0.6B"),
        ("Qwen3-1.7B",           "Qwen3",            "1.7B"),
        ("Qwen3-4B",             "Qwen3",            "4B"),
        ("Qwen3-8B",             "Qwen3",            "8B"),
        ("Qwen3-32B",            "Qwen3",            "32B"),
        ("Qwen3-0.6B (think)",   "Qwen3 (think)",    "0.6B"),
        ("Qwen3-1.7B (think)",   "Qwen3 (think)",    "1.7B"),
        ("Qwen3-4B (think)",     "Qwen3 (think)",    "4B"),
        ("Qwen3-8B (think)",     "Qwen3 (think)",    "8B"),
        ("Qwen3-32B (think)",    "Qwen3 (think)",    "32B"),
        ("Vicuna-7B",            "Vicuna",           "7B"),
        ("Vicuna-13B",           "Vicuna",           "13B"),
    ],
}
GROUP_LABELS = {
    "VLM":             "Vision-Language Models",
    "Backbone Decoder": "VLM Backbone Decoders",
    "Standalone LLM":  "Standalone LLMs",
}

# ── human reference ───────────────────────────────────────────────────────
# GT yes rate from VQA annotations; human yes rate from human study
# computed separately and hardcoded for the 26 yes/no questions
HUMAN_YES  = 65.4   # mean yes% across 40 participants
GT_YES     = 34.6   # ground-truth yes rate


def load_yn_ids() -> set[int]:
    """Return question IDs in the study subset that are yes/no type."""
    qid2type: dict[int, str] = {}
    for path in (VQA_ANN_TRAIN, VQA_ANN_VAL):
        with open(path) as f:
            for a in json.load(f)["annotations"]:
                qid2type[a["question_id"]] = a["answer_type"]
    with open(SUBSET_JSONL) as f:
        subset_ids = {json.loads(l)["question_id"] for l in f}
    return {qid for qid in subset_ids if qid2type.get(qid) == "yes/no"}


def compute_dist(df: pd.DataFrame, yn_ids: set[int]) -> dict[str, dict[str, float]]:
    """Return {model: {yes, no, abst, other}} as percentages (all 3 variants)."""
    sub = df[df["question_id"].isin(yn_ids)].copy()
    sub["cat"] = sub["response"].apply(classify)
    result: dict[str, dict[str, float]] = {}
    for model, grp in sub.groupby("model"):
        vc = grp["cat"].value_counts(normalize=True) * 100
        result[str(model)] = {c: vc.get(c, 0.0) for c in ("yes", "no", "abst", "other")}
    return result


def _fmt(v: float) -> str:
    return f"{v:.1f}" if v > 0 else "0"


def color_yes(v: float) -> str:
    """blue = yes-majority (≥50%), orange = no-majority (<50%)."""
    if v >= 50:
        return r"\cellcolor[HTML]{D9EAF3}"
    return r"\cellcolor[HTML]{F7EBD9}"


def gray_shade(v: float, max_v: float) -> str:
    """Gray cellcolor proportional to value (0 = no shade, max = gray!20)."""
    if max_v <= 0 or v <= 0:
        return ""
    intens = round((v / max_v) * 4) * 5   # steps: 0, 5, 10, 15, 20
    intens = max(0, min(20, intens))
    if intens == 0:
        return ""
    return rf"\cellcolor{{gray!{intens}}}"


def _delta_cell(blind_val: float, inst_val: float, shade: str = "") -> str:
    """Format blind value with optional gray shading and colored delta bracket."""
    delta = inst_val - blind_val
    val_str = _fmt(blind_val)
    if abs(delta) < 1e-9:
        return rf"{shade}{val_str}"
    sign = "+" if delta >= 0 else ""
    color = "[HTML]{0072B2}" if delta >= 0 else "[HTML]{CC7A00}"
    delta_str = rf"{{\scriptsize (\textcolor{color}{{{sign}{delta:.1f}}})}}"
    return rf"{shade}{val_str}\,{delta_str}"


def yes_cell(v: float) -> str:
    shade = r"\cellcolor{gray!15}" if v >= 50 else ""
    return rf"{shade}{_fmt(v)}"


def build_table(blind_dist: dict, inst_dist: dict) -> str:
    # Collect all model rows first to compute normalization maxima
    all_rows: list[tuple[str, str, str, dict, dict]] = []
    for grp, members in GROUPS.items():
        for key, disp, sz in members:
            blind = blind_dist.get(key, {})
            inst  = inst_dist.get(key, {})
            if blind or inst:
                all_rows.append((grp, disp, sz, blind, inst))

    max_oth  = max((r[3].get("other", 0.0) for r in all_rows), default=1.0) or 1.0
    max_abst = max((r[3].get("abst",  0.0) for r in all_rows), default=1.0) or 1.0

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(
        r"\caption{Yes/no response distributions for all models on the 26 yes/no questions "
        r"(all 3 variants pooled, $N{\leq}78$). "
        r"\textbf{Blind} and \textbf{Inst} columns show yes\% without and with instruction. "
        r"\textbf{Oth\%} = substantive non-yes/no outputs (blind); "
        r"\textbf{Abst\%} = all abstentions---hard and soft---detected by the shared classifier (blind). "
        r"\colorbox{gray!15}{Gray} = yes-majority (${\geq}50\%$). "
        r"Human and GT rows shown for reference.}"
    )
    lines.append(r"\label{tab:yesno_distribution_all}")
    lines.append(r"\begin{tabular}{ll cccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Sz}} "
        r"& \multicolumn{2}{c}{\textbf{yes\%}} & \multirow{2}{*}{\textbf{Oth\%}} & \multirow{2}{*}{\textbf{Abst\%}} \\"
    )
    lines.append(r"\cmidrule(lr){3-4}")
    lines.append(r" & & \textbf{Blind} & \textbf{Inst} & & \\")
    lines.append(r"\midrule")

    # Reference rows
    lines.append(
        rf"\textbf{{Human}} & -- & {yes_cell(HUMAN_YES)} & --- & 0.0 & 0.0 \\"
    )
    lines.append(
        rf"\textbf{{GT}} & -- & {yes_cell(GT_YES)} & --- & --- & --- \\"
    )

    prev_grp  = None
    prev_disp = None
    for grp, disp, sz, blind, inst in all_rows:
        if grp != prev_grp:
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{6}}{{c}}{{\textit{{{GROUP_LABELS[grp]}}}}}\\ ")
            lines.append(r"\midrule")
            prev_grp  = grp
            prev_disp = None

        name_cell = disp if disp != prev_disp else ""
        prev_disp = disp

        b_yes  = blind.get("yes",   0.0)
        i_yes  = inst.get("yes",    0.0)
        b_oth  = blind.get("other", 0.0)
        b_abst = blind.get("abst",  0.0)

        oth_cell  = rf"{gray_shade(b_oth,  max_oth)}{_fmt(b_oth)}"
        abst_cell = rf"{gray_shade(b_abst, max_abst)}{_fmt(b_abst)}"

        lines.append(rf"{name_cell} & {sz} & {yes_cell(b_yes)} & {yes_cell(i_yes)} & {oth_cell} & {abst_cell} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print("Loading yes/no question IDs…")
    yn_ids = load_yn_ids()
    print(f"  {len(yn_ids)} yes/no questions in study subset")

    blind_df = pd.read_csv(EXPORTS / "responses_model_blind.csv")
    inst_df  = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")

    blind_dist = compute_dist(blind_df, yn_ids)
    inst_dist  = compute_dist(inst_df,  yn_ids)

    tex = build_table(blind_dist, inst_dist)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"Saved: {OUT}")
    print(tex)
