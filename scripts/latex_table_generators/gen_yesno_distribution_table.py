"""
Generator for the yes/no response distribution table (appendix).

Layout: one row per model, columns: yes% | Oth%
Blind value shown; instruction effect Δ in colored parentheses (teal=+, red=−).
Abstention column removed.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/yesno_distribution_all_models.tex"

EXPORTS = ROOT / "analysis/session2/exports"
VQA_ANN_TRAIN = ROOT / "dataset/vqa/v2_mscoco_train2014_annotations.json"
VQA_ANN_VAL   = ROOT / "dataset/vqa/v2_mscoco_val2014_annotations.json"
SUBSET_JSONL   = ROOT / "dataset/vqa/vqa1k_control_study_subset.jsonl"

# ── abstention keyword regex ───────────────────────────────────────────────
ABS_RE = re.compile(
    r"\b(none|nothing|nowhere|unanswerable|unknown|cannot|can'?t|"
    r"no image|no picture|no photo|blank|empty|unable|n/a|no scene|"
    r"not\s+(?:visible|present|shown|provided|available|given|applicable)|"
    r"without|i\s+don'?t\s+know|i\s+cannot|impossible to)\b",
    re.I,
)

def classify(resp: str) -> str:
    r = str(resp).strip().lower()
    if ABS_RE.search(r):      return "abst"
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
    """blue = human-like (≥50%), orange = below GT (≤35%), else plain."""
    if v >= 50:
        return r"\cellcolor[HTML]{D9EAF3}"
    if v <= 35:
        return r"\cellcolor[HTML]{F7EBD9}"
    return ""


def _delta_cell(blind_val: float, inst_val: float) -> str:
    """Format blind_val with colored Δ parenthetical showing inst effect."""
    delta = inst_val - blind_val
    sign = "+" if delta >= 0 else ""
    color = "teal" if delta >= 0 else "red"
    delta_str = rf"{{\scriptsize (\textcolor{{{color}}}{{{sign}{delta:.1f}}})}}"
    return rf"{_fmt(blind_val)}\,{delta_str}"


def build_table(blind_dist: dict, inst_dist: dict) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(
        r"\caption{Yes/no response distributions for all models on the 26 yes/no questions "
        r"(all 3 variants pooled, $N{\leq}78$). "
        r"\textbf{Oth\%} = non-yes/no outputs (excluding soft abstentions). "
        r"Values show the blind condition; "
        r"{\scriptsize (\textcolor{teal}{$+\Delta$}/\textcolor{red}{$-\Delta$})} = instruction effect (Blind+Inst $-$ Blind). "
        r"\colorbox[HTML]{D9EAF3}{Blue} = yes-majority (${\geq}50\%$); "
        r"\colorbox[HTML]{F7EBD9}{Orange} = below GT yes rate (${\leq}35\%$). "
        r"Human and GT rows shown for reference.}"
    )
    lines.append(r"\label{tab:yesno_distribution_all}")
    lines.append(r"\begin{tabular}{ll cc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Sz} & \textbf{yes\%} & \textbf{Oth\%} \\")
    lines.append(r"\midrule")

    # Reference rows
    lines.append(
        rf"\textbf{{Human}} & -- & \cellcolor[HTML]{{D9EAF3}}{_fmt(HUMAN_YES)} & 0.0 \\"
    )
    lines.append(
        rf"\textbf{{GT}} & -- & \cellcolor[HTML]{{F7EBD9}}{_fmt(GT_YES)} & --- \\"
    )

    for grp, members in GROUPS.items():
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{4}}{{c}}{{\textit{{{GROUP_LABELS[grp]}}}}}\\"
        )
        lines.append(r"\midrule")

        prev_disp = None
        for key, disp, sz in members:
            blind = blind_dist.get(key, {})
            inst  = inst_dist.get(key, {})
            if not blind and not inst:
                continue

            name_cell = disp if disp != prev_disp else ""
            prev_disp = disp

            b_yes = blind.get("yes",   0.0)
            i_yes = inst.get("yes",    0.0)
            b_oth = blind.get("other", 0.0)
            i_oth = inst.get("other",  0.0)

            col_yes = color_yes(b_yes)
            yes_cell = rf"{col_yes}{_delta_cell(b_yes, i_yes)}"
            oth_cell = _delta_cell(b_oth, i_oth)

            lines.append(rf"{name_cell} & {sz} & {yes_cell} & {oth_cell} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
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
