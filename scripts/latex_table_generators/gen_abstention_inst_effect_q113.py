import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from utils.abstention import classify

EXPORTS = ROOT / "analysis/session2/exports"
TABLE_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables"

VARIANTS = ["C", "B", "A"]
VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}

SIZE_NUM = {
    "Qwen3-VL-2B": 2, "Qwen3-VL-4B": 4, "Qwen3-VL-8B": 8, "Qwen3-VL-32B": 32,
    "InternVL-1B": 1, "InternVL-2B": 2, "InternVL-8B": 8,
    "LLaVA-1.5-7B": 7, "LLaVA-Mistral": 7.01, "LLaVA-Vicuna": 7.02, "LLaVA-Vicuna-13B": 13,
    "Qwen3-VL-2B (LM)": 2, "Qwen3-VL-4B (LM)": 4, "Qwen3-VL-8B (LM)": 8, "Qwen3-VL-32B (LM)": 32,
    "InternVL-1B (LM)": 1, "InternVL-2B (LM)": 2, "InternVL-8B (LM)": 8,
    "LLaVA-1.5 (LM)": 7, "LLaVA-Vicuna (LM)": 7.02, "LLaVA-Mistral (LM)": 7.01, "LLaVA-Vicuna-13B (LM)": 13,
    "Qwen3-0.6B": 0.6, "Qwen3-1.7B": 1.7, "Qwen3-4B": 4, "Qwen3-8B": 8, "Qwen3-32B": 32,
    "Qwen2.5-7B-Instruct": 7, "Mistral-7B": 7.01, "Vicuna-7B": 7.02, "Vicuna-13B": 13,
    "Phi-3.5-mini": 3.8,
    "Qwen3-0.6B (think)": 0.6, "Qwen3-1.7B (think)": 1.7, "Qwen3-4B (think)": 4,
    "Qwen3-8B (think)": 8, "Qwen3-32B (think)": 32,
}

SIZE_LABEL = {
    "Qwen3-VL-2B": "2B", "Qwen3-VL-4B": "4B", "Qwen3-VL-8B": "8B", "Qwen3-VL-32B": "32B",
    "InternVL-1B": "1B", "InternVL-2B": "2B", "InternVL-8B": "8B",
    "LLaVA-1.5-7B": "7B", "LLaVA-Mistral": "7B", "LLaVA-Vicuna": "7B", "LLaVA-Vicuna-13B": "13B",
    "Qwen3-VL-2B (LM)": "2B", "Qwen3-VL-4B (LM)": "4B", "Qwen3-VL-8B (LM)": "8B", "Qwen3-VL-32B (LM)": "32B",
    "InternVL-1B (LM)": "1B", "InternVL-2B (LM)": "2B", "InternVL-8B (LM)": "8B",
    "LLaVA-1.5 (LM)": "7B", "LLaVA-Vicuna (LM)": "7B", "LLaVA-Mistral (LM)": "7B", "LLaVA-Vicuna-13B (LM)": "13B",
    "Qwen3-0.6B": "0.6B", "Qwen3-1.7B": "1.7B", "Qwen3-4B": "4B", "Qwen3-8B": "8B", "Qwen3-32B": "32B",
    "Qwen2.5-7B-Instruct": "7B", "Mistral-7B": "7B", "Vicuna-7B": "7B", "Vicuna-13B": "13B",
    "Phi-3.5-mini": "3.8B",
    "Qwen3-0.6B (think)": "0.6B", "Qwen3-1.7B (think)": "1.7B", "Qwen3-4B (think)": "4B",
    "Qwen3-8B (think)": "8B", "Qwen3-32B (think)": "32B",
}

DISPLAY_NAME = {
    "LLaVA-1.5-7B": "LLaVA-1.5",
    "LLaVA-Mistral": "LLaVA-1.6-Mistral",
    "LLaVA-Vicuna": "LLaVA-1.6-Vicuna",
    "LLaVA-Vicuna-13B": "LLaVA-1.6-Vicuna",
    "LLaVA-1.5 (LM)": "LLaVA-1.5",
    "LLaVA-Vicuna (LM)": "LLaVA-1.6-Vicuna",
    "LLaVA-Mistral (LM)": "LLaVA-1.6-Mistral",
    "LLaVA-Vicuna-13B (LM)": "LLaVA-1.6-Vicuna",
    "Phi-3.5-mini": "Phi-3.5",
}


def family_key(model: str) -> str:
    base = DISPLAY_NAME.get(model, model).replace(" (LM)", "").replace(" (think)", "")
    if base.startswith("InternVL"):
        return "InternVL"
    if base.startswith("LLaVA"):
        return "LLaVA"
    if base.startswith("Mistral"):
        return "Mistral"
    if base.startswith("Phi"):
        return "Phi"
    if base.startswith("Qwen2.5"):
        return "Qwen2.5"
    if base.startswith("Qwen3-VL"):
        return "Qwen3-VL"
    if base.startswith("Qwen3"):
        return "Qwen3"
    if base.startswith("Vicuna"):
        return "Vicuna"
    return base


def sort_models(models: list[str]) -> list[str]:
    return sorted(models, key=lambda m: (family_key(m).lower(), SIZE_NUM[m], DISPLAY_NAME.get(m, m).lower()))


def row_name(model: str, group: str) -> str:
    name = DISPLAY_NAME.get(model, model)
    if group == "VLM backbone decoder":
        name = name.replace(" (LM)", "")
    if group == "standalone LLM (think)":
        name = name.replace(" (think)", "")
    name = re.sub(r"-(?:0\.6|1|1\.7|2|3\.8|4|7|8|13|32)B(?=-|$)", "", name)
    return name.replace("--", "-").strip("- ")


def delta_text(delta: float) -> str:
    if abs(delta) < 0.05:
        return ""
    color = "teal" if delta > 0 else "red"
    return rf" \textcolor{{{color}}}{{({delta:+.1f})}}"


def fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}"


GROUPS = [
    ("Vision-Language Models", "VLM", sort_models([
        "Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B", "Qwen3-VL-32B",
        "InternVL-1B", "InternVL-2B", "InternVL-8B",
        "LLaVA-1.5-7B", "LLaVA-Vicuna", "LLaVA-Mistral", "LLaVA-Vicuna-13B",
    ])),
    ("VLM Backbone Decoders", "VLM backbone decoder", sort_models([
        "Qwen3-VL-2B (LM)", "Qwen3-VL-4B (LM)", "Qwen3-VL-8B (LM)", "Qwen3-VL-32B (LM)",
        "InternVL-1B (LM)", "InternVL-2B (LM)", "InternVL-8B (LM)",
        "LLaVA-1.5 (LM)", "LLaVA-Vicuna (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna-13B (LM)",
    ])),
    ("Standalone LLMs", "standalone LLM", sort_models([
        "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-32B",
        "Qwen2.5-7B-Instruct", "Mistral-7B", "Vicuna-7B", "Vicuna-13B", "Phi-3.5-mini",
    ])),
]


def load_frames():
    human = pd.read_csv(EXPORTS / "responses_human.csv")
    qids = set(human["question_id"].unique())
    inst = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")
    blind = pd.read_csv(EXPORTS / "responses_model_blind.csv")
    inst = inst[inst["question_id"].isin(qids)].copy()
    blind = blind[blind["question_id"].isin(qids)].copy()
    for df in (inst, blind):
        df["cls"] = df["response"].fillna("").astype(str).map(lambda x: classify(x, None))
    return inst, blind


def rates_for(df: pd.DataFrame, model: str, variant: str) -> tuple[float, float]:
    sub = df[(df["model"] == model) & (df["variant"] == variant)]
    if sub.empty:
        return float("nan"), float("nan")
    return (sub["cls"].eq("soft_abstained").mean(), sub["cls"].eq("hard_abstained").mean())


def render_table(lines: list[str], caption: str, label: str, one_variant: Optional[str] = None) -> str:
    lines_out = []
    lines_out.append(r"\begin{table}[t]")
    lines_out.append(r"\scriptsize")
    lines_out.append(r"\centering")
    lines_out.append(r"\setlength{\tabcolsep}{2pt}")
    lines_out.append(r"\resizebox{\columnwidth}{!}{%")
    lines_out.append("\n".join(lines))
    lines_out.append(r"}")
    lines_out.append(rf"\caption{{\small {caption}}}")
    lines_out.append(rf"\label{{{label}}}")
    lines_out.append(r"\end{table}")
    return "\n".join(lines_out) + "\n"


def build_vc(inst: pd.DataFrame, blind: pd.DataFrame) -> str:
    v = "C"
    lines = [r"\begin{tabular}{lc|cc}", r"\toprule",
             r"\textbf{Model} & \textbf{Size} & \textbf{Soft (w/)} & \textbf{Hard (w/)} \\",
             r"\midrule"]
    for title, group, models in GROUPS:
        lines.append(rf"\multicolumn{{4}}{{c}}{{\textit{{{title}}}}} \\")
        lines.append(r"\midrule")
        prev_name = None
        for model in models:
            name = row_name(model, group)
            size = SIZE_LABEL[model]
            soft_i, hard_i = rates_for(inst, model, v)
            soft_b, hard_b = rates_for(blind, model, v)
            soft = fmt_pct(soft_i) + delta_text((soft_i - soft_b) * 100)
            hard = fmt_pct(hard_i) + delta_text((hard_i - hard_b) * 100)
            cell_name = name if name != prev_name else ""
            lines.append(f"{cell_name} & {size} & {soft} & {hard} \\\\")
            prev_name = name
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    caption = (
        r"Per-model abstention on variant~C (113 questions) under the blind-awareness instruction. "
        r"Values report the \textit{w/ instruction} rate only; colored parentheses show the change in percentage points "
        r"relative to the blind-only (\textit{w/o}) condition."
    )
    return render_table(lines, caption, "tab:abstention_inst_effect_vC_q113", one_variant="C")


def build_vabc(inst: pd.DataFrame, blind: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{llc|cc}", r"\toprule",
             r"\textbf{Var.} & \textbf{Model} & \textbf{Size} & \textbf{Soft (w/)} & \textbf{Hard (w/)} \\",
             r"\midrule"]
    for v in VARIANTS:
        lines.append(rf"\multicolumn{{5}}{{c}}{{\textit{{{VARIANT_LABELS[v]}}}}} \\")
        lines.append(r"\midrule")
        for title, group, models in GROUPS:
            lines.append(rf"\multicolumn{{5}}{{l}}{{\scriptsize {title}}} \\")
            prev_name = None
            for model in models:
                name = row_name(model, group)
                size = SIZE_LABEL[model]
                soft_i, hard_i = rates_for(inst, model, v)
                soft_b, hard_b = rates_for(blind, model, v)
                soft = fmt_pct(soft_i) + delta_text((soft_i - soft_b) * 100)
                hard = fmt_pct(hard_i) + delta_text((hard_i - hard_b) * 100)
                variant_cell = v if prev_name is None else ""
                cell_name = name if name != prev_name else ""
                lines.append(f"{variant_cell} & {cell_name} & {size} & {soft} & {hard} \\\\")
                prev_name = name
            lines.append(r"\addlinespace[1pt]")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    caption = (
        r"Per-model abstention on the 113-question subset across all three control variants under the blind-awareness instruction. "
        r"Values report the \textit{w/ instruction} rate only; colored parentheses show the change in percentage points "
        r"relative to the blind-only (\textit{w/o}) condition."
    )
    return render_table(lines, caption, "tab:abstention_inst_effect_vABC_q113")


if __name__ == "__main__":
    inst_df, blind_df = load_frames()
    (TABLE_DIR / "abstention_inst_effect_vC_q113.tex").write_text(build_vc(inst_df, blind_df))
    (TABLE_DIR / "abstention_inst_effect_vABC_q113.tex").write_text(build_vabc(inst_df, blind_df))
    print(TABLE_DIR / "abstention_inst_effect_vC_q113.tex")
    print(TABLE_DIR / "abstention_inst_effect_vABC_q113.tex")
