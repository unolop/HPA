"""
Generate compact variant-C summary tables for the paper.

Outputs:
  - summary_vlm_decoder_vC_side_by_side.tex
  - summary_vlm_decoder_vC_slash.tex
  - summary_llm_vC.tex
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
EXPORTS = ROOT / "analysis/session2/exports"

pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
mr = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")
hr = pd.read_csv(EXPORTS / "responses_human.csv")

VARIANT = "C"

SIZE_LABEL = {
    "Qwen3-VL-2B": "2B", "Qwen3-VL-4B": "4B", "Qwen3-VL-8B": "8B", "Qwen3-VL-32B": "32B",
    "InternVL-1B": "1B", "InternVL-2B": "2B", "InternVL-8B": "8B",
    "LLaVA-1.5-7B": "7B", "LLaVA-Mistral": "7B", "LLaVA-Vicuna": "7B", "LLaVA-Vicuna-13B": "13B",
    "Qwen3-VL-2B (LM)": "2B", "Qwen3-VL-4B (LM)": "4B", "Qwen3-VL-8B (LM)": "8B", "Qwen3-VL-32B (LM)": "32B",
    "InternVL-1B (LM)": "1B", "InternVL-2B (LM)": "2B", "InternVL-8B (LM)": "8B",
    "LLaVA-1.5 (LM)": "7B", "LLaVA-Mistral (LM)": "7B", "LLaVA-Vicuna (LM)": "7B", "LLaVA-Vicuna-13B (LM)": "13B",
    "Qwen3-0.6B": "0.6B", "Qwen3-1.7B": "1.7B", "Qwen3-4B": "4B", "Qwen3-8B": "8B", "Qwen3-32B": "32B",
    "Qwen2.5-7B-Instruct": "7B", "Mistral-7B": "7B", "Vicuna-7B": "7B", "Vicuna-13B": "13B",
    "Phi-3.5-mini": "3.8B",
    "Qwen3-0.6B (think)": "0.6B", "Qwen3-1.7B (think)": "1.7B", "Qwen3-4B (think)": "4B",
    "Qwen3-8B (think)": "8B", "Qwen3-32B (think)": "32B",
}


def display_name(model):
    mapping = {
        "LLaVA-1.5-7B": "LLaVA-1.5",
        "LLaVA-Mistral": "LLaVA-1.6-Mistral",
        "LLaVA-Vicuna": "LLaVA-1.6-Vicuna",
        "LLaVA-Vicuna-13B": "LLaVA-1.6-Vicuna",
        "LLaVA-1.5 (LM)": "LLaVA-1.5",
        "LLaVA-Mistral (LM)": "LLaVA-1.6-Mistral",
        "LLaVA-Vicuna (LM)": "LLaVA-1.6-Vicuna",
        "LLaVA-Vicuna-13B (LM)": "LLaVA-1.6-Vicuna",
        "Qwen2.5-7B-Instruct": "Qwen2.5-Instruct",
    }
    return mapping.get(model, model)


def family_name(model):
    name = display_name(model).replace(" (LM)", "").replace(" (think)", "")
    if name.startswith("InternVL"):
        return "InternVL"
    if name.startswith("LLaVA"):
        return name
    if name.startswith("Qwen3-VL"):
        return "Qwen3-VL"
    if name.startswith("Qwen3"):
        return "Qwen3"
    if name.startswith("Qwen2.5"):
        return "Qwen2.5-Instruct"
    if name.startswith("Mistral"):
        return "Mistral"
    if name.startswith("Vicuna"):
        return "Vicuna"
    if name.startswith("Phi"):
        return "Phi-3.5"
    return name


def fmt_pct(x):
    if pd.isna(x):
        return "--"
    return f"{100 * x:.1f}"


def fmt_len(x):
    if pd.isna(x):
        return "--"
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def model_metrics(model):
    resp = mr[(mr["model"] == model) & (mr["variant"] == VARIANT)]["response"].fillna("")
    hm = pc[
        (pc["pair_type"] == "HM")
        & (pc["variant"] == VARIANT)
        & (pc["subject_2"] == model)
    ]
    acc = mr[(mr["model"] == model) & (mr["variant"] == VARIANT)]["accuracy"].mean()
    return {
        "acc": acc,
        "len": resp.str.split().apply(len).mean(),
        "sbert": hm["sbert_score"].mean(),
        "exact": hm["exact_score"].mean(),
    }


def human_metrics():
    hh = pc[(pc["pair_type"] == "HH") & (pc["variant"] == VARIANT)]
    resp = hr[hr["variant"] == VARIANT]["response"].fillna("")
    acc = hr[hr["variant"] == VARIANT].groupby("question_id")["accuracy"].mean().mean()
    return {
        "acc": acc,
        "len": resp.str.split().apply(len).mean(),
        "sbert": hh["sbert_score"].mean(),
        "exact": hh["exact_score"].mean(),
    }


VLM_PAIRS = [
    ("InternVL", "1B", "InternVL-1B", "InternVL-1B (LM)"),
    ("InternVL", "2B", "InternVL-2B", "InternVL-2B (LM)"),
    ("InternVL", "8B", "InternVL-8B", "InternVL-8B (LM)"),
    ("LLaVA-1.5", "7B", "LLaVA-1.5-7B", "LLaVA-1.5 (LM)"),
    ("LLaVA-1.6-Mistral", "7B", "LLaVA-Mistral", "LLaVA-Mistral (LM)"),
    ("LLaVA-1.6-Vicuna", "7B", "LLaVA-Vicuna", "LLaVA-Vicuna (LM)"),
    ("LLaVA-1.6-Vicuna", "13B", "LLaVA-Vicuna-13B", "LLaVA-Vicuna-13B (LM)"),
    ("Qwen3-VL", "2B", "Qwen3-VL-2B", "Qwen3-VL-2B (LM)"),
    ("Qwen3-VL", "4B", "Qwen3-VL-4B", "Qwen3-VL-4B (LM)"),
    ("Qwen3-VL", "8B", "Qwen3-VL-8B", "Qwen3-VL-8B (LM)"),
    ("Qwen3-VL", "32B", "Qwen3-VL-32B", "Qwen3-VL-32B (LM)"),
]

LLM_MODELS = [
    "Mistral-7B",
    "Phi-3.5-mini",
    "Qwen2.5-7B-Instruct",
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-32B",
    "Qwen3-0.6B (think)",
    "Qwen3-1.7B (think)",
    "Qwen3-4B (think)",
    "Qwen3-8B (think)",
    "Qwen3-32B (think)",
    "Vicuna-7B",
    "Vicuna-13B",
]


def write(path, text):
    path.write_text(text + "\n")
    print(f"Wrote {path}")


def build_vlm_side_by_side():
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\scriptsize")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{ll|cccc|cccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Family} & \textbf{Size} "
        r"& \multicolumn{4}{c|}{\textbf{VLM}}"
        r"& \multicolumn{4}{c}{\textbf{Backbone Decoder}} \\"
    )
    lines.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}")
    lines.append(
        r" & & \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{Exact}"
        r"& \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{Exact} \\"
    )
    lines.append(r"\midrule")
    h = human_metrics()
    lines.append(
        r"\textbf{Human (HH)} & -- & "
        + " & ".join([fmt_pct(h["acc"]), fmt_len(h["len"]), fmt_pct(h["sbert"]), fmt_pct(h["exact"])])
        + r" & \multicolumn{4}{c}{--} \\"
    )
    lines.append(r"\midrule")
    for family, size, vlm_model, dec_model in VLM_PAIRS:
        vm = model_metrics(vlm_model)
        dm = model_metrics(dec_model)
        lines.append(
            f"{family} & {size} & "
            + " & ".join([fmt_pct(vm["acc"]), fmt_len(vm["len"]), fmt_pct(vm["sbert"]), fmt_pct(vm["exact"])])
            + " & "
            + " & ".join([fmt_pct(dm["acc"]), fmt_len(dm["len"]), fmt_pct(dm["sbert"]), fmt_pct(dm["exact"])])
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\small Compact Blind+Inst summary on the original question variant (C): "
        r"vision-language models and their paired backbone decoders. "
        r"Acc = VQA accuracy (\%); Len = mean answer length in words; "
        r"SBERT and Exact = human--model agreement (\%).}"
    )
    lines.append(r"\label{tab:summary_vlm_decoder_vc_side}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_vlm_slash():
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\scriptsize")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{ll|cccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Family} & \textbf{Size} & \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{Exact} \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Values shown as VLM / Decoder}} \\")
    lines.append(r"\midrule")
    for family, size, vlm_model, dec_model in VLM_PAIRS:
        vm = model_metrics(vlm_model)
        dm = model_metrics(dec_model)
        lines.append(
            f"{family} & {size} & "
            + " & ".join([
                f"{fmt_pct(vm['acc'])} / {fmt_pct(dm['acc'])}",
                f"{fmt_len(vm['len'])} / {fmt_len(dm['len'])}",
                f"{fmt_pct(vm['sbert'])} / {fmt_pct(dm['sbert'])}",
                f"{fmt_pct(vm['exact'])} / {fmt_pct(dm['exact'])}",
            ])
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\small Slash-compressed Blind+Inst summary on the original question variant (C): "
        r"each cell reports VLM / paired backbone decoder. "
        r"Acc = VQA accuracy (\%); Len = mean answer length in words; "
        r"SBERT and Exact = human--model agreement (\%).}"
    )
    lines.append(r"\label{tab:summary_vlm_decoder_vc_slash}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_llm_table():
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\scriptsize")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{ll|cccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Family} & \textbf{Size} & \textbf{Acc} & \textbf{Len} & \textbf{SBERT} & \textbf{Exact} \\")
    lines.append(r"\midrule")
    h = human_metrics()
    lines.append(
        r"\textbf{Human (HH)} & -- & "
        + " & ".join([fmt_pct(h["acc"]), fmt_len(h["len"]), fmt_pct(h["sbert"]), fmt_pct(h["exact"])])
        + r" \\"
    )
    lines.append(r"\midrule")
    for model in LLM_MODELS:
        m = model_metrics(model)
        family = family_name(model)
        if "(think)" in model:
            family = f"{family} (think)"
        lines.append(
            f"{family} & {SIZE_LABEL[model]} & "
            + " & ".join([fmt_pct(m["acc"]), fmt_len(m["len"]), fmt_pct(m["sbert"]), fmt_pct(m["exact"])])
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\small Compact Blind+Inst summary on the original question variant (C) for standalone LLMs. "
        r"Acc = VQA accuracy (\%); Len = mean answer length in words; "
        r"SBERT and Exact = human--model agreement (\%).}"
    )
    lines.append(r"\label{tab:summary_llm_vc}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    write(out_dir / "summary_vlm_decoder_vC_side_by_side.tex", build_vlm_side_by_side())
    write(out_dir / "summary_vlm_decoder_vC_slash.tex", build_vlm_slash())
    write(out_dir / "summary_llm_vC.tex", build_llm_table())
