"""Generate JS divergence blind vs inst_blind comparison table (variant C only).

Δ = JS(blind) − JS(inst_blind): positive = instruction improves alignment.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables"

from notebooks.helpers import (
    load_human_responses,
    _clean_answer_series,
    _answer_type_subset,
    _mark_abstentions,
    _distribution_stats_from_counts,
)
from analysis.utils.constants import MODEL_SIZE_B

MATCHED_MODELS = {
    "InternVL-8B", "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna",
    "InternVL-8B (LM)", "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)",
    "Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B",
}

GROUP_ORDER = ["VLM", "Backbone Decoder", "Standalone LLM", "Think"]
SECTION_TITLES = {
    "Backbone Decoder": "Backbone decoders",
    "VLM": "VLMs",
    "Standalone LLM": "Standalone LLMs",
    "Think": "Think models",
}


def infer_group(model: str) -> str:
    if "think" in model.lower():
        return "Think"
    if "(LM)" in model:
        return "Backbone Decoder"
    if "InternVL" in model or "Qwen3-VL" in model or "LLaVA" in model:
        return "VLM"
    return "Standalone LLM"


def display_name(model: str) -> str:
    name = model.replace(" (LM)", "")
    for s in ["-8B", "-7B", "-13B", "-32B", "-4B", "-2B", "-1B", "-1.7B", "-0.6B", "-Instruct"]:
        name = name.replace(s, "")
    name = name.replace("Vicuna-v1.5", "Vicuna").replace("  ", " ").strip().replace(" ()", "")
    if "(think)" in model:
        return "Qwen3 (think)"
    return name


def size_label(model: str) -> str:
    size = MODEL_SIZE_B.get(model)
    if size is None:
        return "--"
    return f"{int(size)}B" if float(size).is_integer() else f"{size:g}B"


def _read_export(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "analysis/session2/exports" / name)


def build_js_block(condition: str, variant: str = "C") -> pd.DataFrame:
    from analysis.utils.vqa import VQAAnswerMapper
    mapper = VQAAnswerMapper()
    mapper._load()
    qid2atype = {int(qid): ann.get("answer_type", "other") for qid, ann in mapper.annotations.items()}

    human = load_human_responses().copy()
    model = _read_export(f"responses_model_{condition}.csv").copy()

    human = human[human["variant"] == variant].copy()
    model = model[model["variant"] == variant].copy()
    model = model[model["model"].isin(MATCHED_MODELS)].copy()

    human["answer_type"] = human["question_id"].astype(int).map(qid2atype).fillna("other")
    model["answer_type"] = model["question_id"].astype(int).map(qid2atype).fillna("other")
    human["clean_response"] = _clean_answer_series(human["response"])
    model["clean_response"] = _clean_answer_series(model["response"])
    human = _mark_abstentions(human)
    model = _mark_abstentions(model)

    rows = []
    for model_name, model_grp in model.groupby("model"):
        vals = {}
        nqs = {}
        for atype in ("yes/no", "number"):
            h = _answer_type_subset(human, atype).copy()
            m = _answer_type_subset(model_grp, atype).copy()
            m = m[~m["is_abstained"]].copy()
            kept = set(zip(m["question_id"].astype(int), m["variant"].astype(str)))
            h = h[[qv in kept for qv in zip(h["question_id"].astype(int), h["variant"].astype(str))]].copy()
            nqs[atype] = len(kept)
            if h.empty or m.empty:
                vals[atype] = np.nan
                continue
            hc = h["clean_response"].value_counts()
            mc = m["clean_response"].value_counts()
            _, _, js, _, _ = _distribution_stats_from_counts(hc, mc)
            vals[atype] = float(js)
        rows.append({
            "Model": model_name,
            "Display": display_name(model_name),
            "Group": infer_group(model_name),
            "Size": size_label(model_name),
            f"yesno_{condition}": vals.get("yes/no", np.nan),
            f"count_{condition}": vals.get("number", np.nan),
            f"nq_yesno_{condition}": nqs.get("yes/no", 0),
            f"nq_count_{condition}": nqs.get("number", 0),
        })
    return pd.DataFrame(rows)


def fmt(v: float, bold: bool = False, underline: bool = False) -> str:
    if pd.isna(v):
        return "--"
    s = f"{v:.3f}".lstrip("0") or "0"
    if s.startswith("-0"):
        s = "-" + s[2:]
    if bold:
        return rf"\textbf{{{s}}}"
    if underline:
        return rf"\underline{{{s}}}"
    return s


def fmt_delta(v: float, bold: bool = False, underline: bool = False) -> str:
    if pd.isna(v):
        return "--"
    s = f"{v:+.3f}"
    # strip leading zero after sign
    if s.startswith("+0."):
        s = "+" + s[2:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    if bold:
        return rf"\textbf{{{s}}}"
    if underline:
        return rf"\underline{{{s}}}"
    return s


def highlight(series: pd.Series, higher_better: bool = False) -> dict:
    vals = series.dropna()
    if vals.empty:
        return {}
    if higher_better:
        best, second = vals.nlargest(2).values[:2] if len(vals) >= 2 else (vals.max(), np.nan)
    else:
        best, second = vals.nsmallest(2).values[:2] if len(vals) >= 2 else (vals.min(), np.nan)
    out = {}
    for idx, v in series.items():
        if pd.isna(v):
            continue
        out[idx] = ("bold" if np.isclose(v, best) else
                    "under" if not np.isnan(second) and np.isclose(v, second) else "")
    return out


def render(df: pd.DataFrame) -> str:
    cols = ["yesno_blind", "count_blind", "delta_yesno", "delta_count"]
    # JS is lower=better; delta = inst - blind, so negative delta = improvement
    hl = {c: highlight(df[c], higher_better=False) for c in cols}

    def cell_combined(idx, js_col, delta_col):
        """Render JS value with delta in parentheses with color+arrow direction."""
        js_v = df.loc[idx, js_col]
        d_v  = df.loc[idx, delta_col]
        h_js = hl[js_col].get(idx, "")
        h_d  = hl[delta_col].get(idx, "")
        js_str = fmt(js_v, bold=(h_js == "bold"), underline=(h_js == "under"))
        d_str  = fmt_delta(d_v, bold=(h_d == "bold"), underline=(h_d == "under"))
        if pd.isna(js_v):
            return "--"
        if not pd.isna(d_v):
            if d_v < -0.0005:
                d_str = rf"\textcolor[HTML]{{0072B2}}{{{d_str}\,$\downarrow$}}"
            elif d_v > 0.0005:
                d_str = rf"\textcolor[HTML]{{CC7A00}}{{{d_str}\,$\uparrow$}}"
        return f"{js_str} ({d_str})"

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{JS divergence to human answer distribution on original questions (variant~C). "
        r"$N_q$ = number of non-abstained questions used. "
        r"Values show JS w/o instruction; $\Delta$ = JS(w/ inst) $-$ JS(w/o inst) in parentheses. "
        r"\textcolor[HTML]{0072B2}{Blue\,$\downarrow$} = instruction reduces JS (improves alignment), "
        r"\textcolor[HTML]{CC7A00}{orange\,$\uparrow$} = worsens. "
        r"Bold = best (lowest) JS; underline = second-best per column.}",
        r"\label{tab:js_instruction_effect}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{$N_q$} & \multicolumn{2}{c}{JS (w/o inst.) \; ($\Delta$)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Model & Y/N & Cnt & Yes/No & Count \\",
        r"\midrule",
    ]

    for group in GROUP_ORDER:
        sub = df[df["Group"] == group]
        if sub.empty:
            continue
        lines.append(rf"\multicolumn{{5}}{{c}}{{\textit{{{SECTION_TITLES[group]}}}}} \\")
        lines.append(r"\midrule")
        for idx, row in sub.iterrows():
            nyn = int(row.get("nq_yesno_blind", 0))
            ncnt = int(row.get("nq_count_blind", 0))
            cells = [
                str(nyn) if nyn else "--",
                str(ncnt) if ncnt else "--",
                cell_combined(idx, "yesno_blind", "delta_yesno"),
                cell_combined(idx, "count_blind", "delta_count"),
            ]
            lines.append(f"{row['Display']} & {' & '.join(cells)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    for variant in ("C", "B", "A"):
        vname = {"C": "original", "B": "weaker", "A": "pronominalized"}[variant]
        print(f"\n=== Variant {variant} ({vname}) ===")
        blind = build_js_block("blind", variant=variant)
        inst  = build_js_block("inst_blind", variant=variant)

        df = blind.merge(inst[["Model", "yesno_inst_blind", "count_inst_blind"]], on="Model", how="inner")
        # keep blind Nq columns (Nq from blind condition is more informative re: abstention)
        df["delta_yesno"] = df["yesno_inst_blind"] - df["yesno_blind"]
        df["delta_count"]  = df["count_inst_blind"]  - df["count_blind"]
        df = df.sort_values(["Group", "Model"])

        suffix = {"C": "", "B": "_weaker", "A": "_pronominalized"}[variant]
        out = OUT_DIR / f"js_instruction_effect{suffix}.tex"
        out.write_text(render(df))
        print(f"Saved: {out}")

        print("  Group means:")
        for g in GROUP_ORDER:
            sub = df[df["Group"] == g]
            if sub.empty:
                continue
            print(f"    {g}: Δ yes/no={sub['delta_yesno'].mean():.3f}  Δ count={sub['delta_count'].mean():.3f}")


if __name__ == "__main__":
    main()
