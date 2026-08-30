r"""
Comprehensive inst-blind table — all models, per-variant agreement metrics.

Shows Acc, Len, SBERT, chrF, ROUGE-1, Exact for every model across Original,
Weaker, Pronominalized variants.  Uses the canonical HPA abstention classifier
(analysis.utils.abstention) via figures.helpers.filter_abstained_pairs, so
values are consistent with gen_alignment_ranking_full_scale_ticks.py.

Output: latex/AAAI2026/LaTeX/tables/supp/comprehensive_inst_blind_table.tex
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figures.helpers import filter_abstained_pairs
from notebooks.helpers import _clean_answer_series
from analysis.utils.abstention import classify, is_abstained

EXPORTS = ROOT / "analysis/session2/exports"
OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/comprehensive_inst_blind_table.tex"

_STRIP_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# ── Model roster (display_name, group, family, size, pair_cache_id) ──────────
MODEL_ROSTER: list[tuple[str, str, str, str, str]] = [
    # VLMs
    ("InternVL3.5",    "VLM", "InternVL3.5",   "1B",  "InternVL-1B"),
    ("",               "VLM", "InternVL3.5",   "2B",  "InternVL-2B"),
    ("",               "VLM", "InternVL3.5",   "8B",  "InternVL-8B"),
    ("LLaVA-1.5",      "VLM", "LLaVA-1.5",     "7B",  "LLaVA-1.5-7B"),
    ("LLaVA-1.6-Mistral", "VLM", "LLaVA-Mistral", "7B", "LLaVA-Mistral"),
    ("LLaVA-1.6-Vicuna", "VLM", "LLaVA-Vicuna",  "7B", "LLaVA-Vicuna"),
    ("",               "VLM", "LLaVA-Vicuna",  "13B", "LLaVA-Vicuna-13B"),
    ("Qwen3-VL",       "VLM", "Qwen3-VL",      "2B",  "Qwen3-VL-2B"),
    ("",               "VLM", "Qwen3-VL",      "4B",  "Qwen3-VL-4B"),
    ("",               "VLM", "Qwen3-VL",      "8B",  "Qwen3-VL-8B"),
    ("",               "VLM", "Qwen3-VL",      "32B", "Qwen3-VL-32B"),
    # Backbone Decoders
    ("InternVL3.5",    "LM",  "InternVL3.5",   "1B",  "InternVL-1B (LM)"),
    ("",               "LM",  "InternVL3.5",   "2B",  "InternVL-2B (LM)"),
    ("",               "LM",  "InternVL3.5",   "8B",  "InternVL-8B (LM)"),
    ("LLaVA-1.5",      "LM",  "LLaVA-1.5",     "7B",  "LLaVA-1.5 (LM)"),
    ("LLaVA-1.6-Mistral", "LM", "LLaVA-Mistral", "7B", "LLaVA-Mistral (LM)"),
    ("LLaVA-1.6-Vicuna", "LM", "LLaVA-Vicuna",  "7B", "LLaVA-Vicuna (LM)"),
    ("",               "LM",  "LLaVA-Vicuna",  "13B", "LLaVA-Vicuna-13B (LM)"),
    ("Qwen3-VL",       "LM",  "Qwen3-VL",      "2B",  "Qwen3-VL-2B (LM)"),
    ("",               "LM",  "Qwen3-VL",      "4B",  "Qwen3-VL-4B (LM)"),
    ("",               "LM",  "Qwen3-VL",      "8B",  "Qwen3-VL-8B (LM)"),
    ("",               "LM",  "Qwen3-VL",      "32B", "Qwen3-VL-32B (LM)"),
    # Standalone LLMs
    ("Mistral",        "SA",  "Mistral",       "7B",  "Mistral-7B"),
    ("Phi-3.5-mini",   "SA",  "Phi-3.5",       "3.8B", "Phi-3.5-mini"),
    ("Qwen2.5-Instruct", "SA", "Qwen2.5",     "7B",  "Qwen2.5-7B-Instruct"),
    ("Qwen3",          "SA",  "Qwen3",         "0.6B", "Qwen3-0.6B"),
    ("",               "SA",  "Qwen3",         "1.7B", "Qwen3-1.7B"),
    ("",               "SA",  "Qwen3",         "4B",   "Qwen3-4B"),
    ("",               "SA",  "Qwen3",         "8B",   "Qwen3-8B"),
    ("",               "SA",  "Qwen3",         "32B",  "Qwen3-32B"),
    ("Vicuna",         "SA",  "Vicuna",        "7B",   "Vicuna-7B"),
    ("",               "SA",  "Vicuna",        "13B",  "Vicuna-13B"),
    # Think models
    ("Qwen3",          "Think", "Qwen3",       "0.6B", "Qwen3-0.6B (think)"),
    ("",               "Think", "Qwen3",       "1.7B", "Qwen3-1.7B (think)"),
    ("",               "Think", "Qwen3",       "4B",   "Qwen3-4B (think)"),
    ("",               "Think", "Qwen3",       "8B",   "Qwen3-8B (think)"),
    ("",               "Think", "Qwen3",       "32B",  "Qwen3-32B (think)"),
]

GROUP_TITLES = {
    "VLM":   "Vision-Language Models",
    "LM":    "VLM Backbone Decoders",
    "SA":    "Standalone LLMs",
    "Think": "Standalone LLMs with Thinking",
}

VARIANTS = ["C", "B", "A"]
VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
METRICS = ["acc", "len", "sbert", "chrf", "rouge1", "exact"]
METRIC_LABELS = ["Acc", "Len", "SBERT", "chrF", "ROUGE-1", "Exact"]


def compute_all() -> dict[str, dict[str, dict[str, float]]]:
    """Returns {model_id: {variant: {metric: value}}}.

    Agreement metrics (SBERT, chrF, ROUGE-1, Exact) are computed on
    free-text questions only (answer_type == 'other'), consistent with
    the sHM column in the ranking table.
    Acc and Len are computed on all question types.
    """
    from analysis.utils.vqa import VQAAnswerMapper

    # ── Load pair cache with canonical abstention filtering ──
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    pc = filter_abstained_pairs(pc)
    hm = pc[pc["pair_type"] == "HM"]
    hh = pc[pc["pair_type"] == "HH"]

    # ── Filter to free-text questions for agreement metrics ──
    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other")
                 for qid, ann in mapper.annotations.items()}
    ft_qids = {qid for qid, at in qid2atype.items() if at == "other"}
    hm_ft = hm[hm["question_id"].astype(int).isin(ft_qids)]
    hh_ft = hh[hh["question_id"].astype(int).isin(ft_qids)]

    # ── Load responses for Acc and Len (all question types) ──
    resp = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")
    resp["clean"] = _clean_answer_series(resp["response"].fillna("").astype(str))
    resp["clean"] = resp["clean"].apply(
        lambda x: _STRIP_THINK.sub("", x).strip() if isinstance(x, str) else x
    )

    all_ids = {mid for *_, mid in MODEL_ROSTER}
    result: dict[str, dict[str, dict[str, float]]] = {}

    for model_id in all_ids:
        result[model_id] = {}
        for v in VARIANTS:
            # ── Agreement metrics from pair_cache (free-text only) ──
            sub = hm_ft[(hm_ft["subject_2"] == model_id) & (hm_ft["variant"] == v)]
            d: dict[str, float] = {}
            d["sbert"]  = float(sub["sbert_score"].mean() * 100) if len(sub) > 0 else np.nan
            d["chrf"]   = float(sub["chrf_score"].mean() * 100)  if len(sub) > 0 else np.nan
            d["rouge1"] = float(sub["rouge1_score"].mean() * 100) if len(sub) > 0 else np.nan
            d["exact"]  = float(sub["exact_score"].mean() * 100) if len(sub) > 0 else np.nan

            # ── Acc and Len from responses (all question types) ──
            rsub = resp[(resp["model"] == model_id) & (resp["variant"] == v)]
            d["acc"] = float(rsub["accuracy"].mean() * 100) if len(rsub) > 0 else np.nan
            if len(rsub) > 0:
                # Len uses raw response (including think traces) for total verbosity
                words = rsub["response"].fillna("").apply(
                    lambda x: len(str(x).split()) if x else 0
                )
                d["len"] = float(words.mean())
            else:
                d["len"] = np.nan

            result[model_id][v] = d

    # ── Human row ──
    result["Human"] = {}
    for v in VARIANTS:
        sub = hh_ft[hh_ft["variant"] == v]
        d = {}
        d["sbert"]  = float(sub["sbert_score"].mean() * 100) if len(sub) > 0 else np.nan
        d["chrf"]   = float(sub["chrf_score"].mean() * 100)  if len(sub) > 0 else np.nan
        d["rouge1"] = float(sub["rouge1_score"].mean() * 100) if len(sub) > 0 else np.nan
        d["exact"]  = float(sub["exact_score"].mean() * 100) if len(sub) > 0 else np.nan

        # Human accuracy from responses (all question types)
        hresp = pd.read_csv(EXPORTS / "responses_human.csv") \
            if (EXPORTS / "responses_human.csv").exists() else None
        if hresp is not None:
            hsub = hresp[(hresp["variant"] == v)]
            d["acc"] = float(hsub["accuracy"].mean() * 100) if len(hsub) > 0 else np.nan
            words = hsub["response"].fillna("").apply(lambda x: len(str(x).split()))
            d["len"] = float(words.mean())
        else:
            d["acc"] = np.nan
            d["len"] = np.nan

        result["Human"][v] = d

    return result


def _fmt(val: float, metric: str) -> str:
    if not np.isfinite(val):
        return "---"
    if metric == "len":
        if val >= 10:
            return f"{val:.0f}"
        return f"{val:.1f}"
    if metric == "acc":
        return f"{val:.1f}"
    # agreement metrics: 1 decimal
    return f"{val:.1f}"


def _bold_best(vals: list[float], formatted: list[str], higher_better: bool = True) -> list[str]:
    """Bold the best value in a group of formatted strings."""
    finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
    if not finite:
        return formatted
    if higher_better:
        best_val = max(v for _, v in finite)
    else:
        best_val = min(v for _, v in finite)
    out = list(formatted)
    for i, v in finite:
        if abs(v - best_val) < 1e-9:
            out[i] = rf"\textbf{{{out[i]}}}"
            break  # only bold one
    return out


def build_table(data: dict) -> str:
    n_metric_cols = len(METRICS)  # 6
    n_variant_cols = n_metric_cols * len(VARIANTS)  # 18
    total_cols = 2 + n_variant_cols  # model + size + 18

    lines: list[str] = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Column spec
    col_spec = "lc" + "|".join(["c" * n_metric_cols] * len(VARIANTS))
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row 1: Model, Size, variant spans
    parts = [r"\textbf{Model}", r"\textbf{Size}"]
    for v in VARIANTS:
        parts.append(
            rf"\multicolumn{{{n_metric_cols}}}{{c{'|' if v != VARIANTS[-1] else ''}}}"
            rf"{{\textbf{{{VARIANT_LABELS[v]}}}}}"
        )
    lines.append(" & ".join(parts) + r" \\")

    # cmidrules
    for i, v in enumerate(VARIANTS):
        start = 3 + i * n_metric_cols
        end = start + n_metric_cols - 1
        lines.append(rf"\cmidrule(lr){{{start}-{end}}}")

    # Header row 2: metric names repeated per variant
    parts = ["", ""]
    for v in VARIANTS:
        for ml in METRIC_LABELS:
            parts.append(rf"\textbf{{{ml}}}")
    lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\midrule")

    # ── Human row ──
    parts = [r"\textbf{Human (N=40)}", "--"]
    for v in VARIANTS:
        hd = data.get("Human", {}).get(v, {})
        for m in METRICS:
            parts.append(_fmt(hd.get(m, np.nan), m))
    lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\midrule")

    # ── Model rows, grouped ──
    prev_group = None
    for display_name, group, family, size, model_id in MODEL_ROSTER:
        # Group header
        if group != prev_group:
            if prev_group is not None:
                lines.append(r"\midrule")
            lines.append(
                rf"\multicolumn{{{total_cols}}}{{c}}"
                rf"{{\textit{{{GROUP_TITLES[group]}}}}}"
                r" \\"
            )
            lines.append(r"\midrule")
            prev_group = group

        # Model name column
        if display_name:
            # Check if next rows share the family (multirow)
            count = sum(1 for dn, g, f, *_ in MODEL_ROSTER
                        if g == group and f == family)
            if count > 1:
                name_cell = rf"\multirow{{{count}}}{{*}}{{{display_name}}}"
            else:
                name_cell = display_name
        else:
            name_cell = ""

        # Collect per-variant values for bolding within group
        parts = [name_cell, size]
        model_data = data.get(model_id, {})

        # Get all values in this group for bolding
        group_models = [mid for _, g, *_, mid in MODEL_ROSTER if g == group]

        for v in VARIANTS:
            vd = model_data.get(v, {})
            # Collect this model's values and all group values for each metric
            for metric in METRICS:
                val = vd.get(metric, np.nan)
                all_group_vals = [
                    data.get(gm, {}).get(v, {}).get(metric, np.nan)
                    for gm in group_models
                ]
                formatted = _fmt(val, metric)
                # Bold best in group
                if np.isfinite(val):
                    finite_group = [v2 for v2 in all_group_vals if np.isfinite(v2)]
                    if finite_group:
                        higher_better = metric not in ("len",)  # len is neutral
                        if metric == "len":
                            pass  # don't bold len
                        elif higher_better and val >= max(finite_group) - 1e-9:
                            formatted = rf"\textbf{{{formatted}}}"
                parts.append(formatted)

        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("}")
    lines.append(
        r"\caption{\small Per-variant agreement metrics for all models in the "
        r"blind+instruction condition. "
        r"Acc = VQA accuracy (\%); Len = mean answer length in words; "
        r"SBERT, chrF, ROUGE-1, Exact = human--model (HM) agreement (\%), "
        r"averaged over 40 human responses per question, "
        r"on free-text questions only ($N{=}61$). "
        r"Human row shows human--human (HH) agreement. "
        r"Agreement metrics use abstention-filtered pairs "
        r"(canonical classifier; consistent with main analysis). "
        r"Bold = best in group per variant.}"
    )
    lines.append(r"\label{tab:comprehensive_inst_blind}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    print("Computing metrics (abstention-filtered)...")
    data = compute_all()

    # Verify against ranking table for a few models
    print("\nSpot-check sHM (total, all variants, filtered):")
    for mid in ["InternVL-8B", "LLaVA-1.5-7B", "Qwen3-8B", "Qwen3-0.6B"]:
        vals = [data.get(mid, {}).get(v, {}).get("sbert", np.nan) for v in VARIANTS]
        mean = np.nanmean(vals)
        print(f"  {mid:<22} C={vals[0]:.1f}  B={vals[1]:.1f}  A={vals[2]:.1f}  mean={mean:.1f}")

    tex = build_table(data)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
