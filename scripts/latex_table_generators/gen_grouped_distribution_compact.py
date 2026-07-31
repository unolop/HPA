import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables"
from notebooks.helpers import (
    grouped_distribution_long_table,
    load_human_responses,
    _clean_answer_series,
    _answer_type_subset,
    _mark_abstentions,
    _distribution_stats_from_counts,
)
from analysis.utils.constants import MODEL_SIZE_B
from analysis.utils.vqa import VQAAnswerMapper


GROUP_ORDER = ["Backbone Decoder", "VLM", "Standalone LLM", "Think"]
SECTION_TITLES = {
    "Backbone Decoder": "Backbone decoders",
    "VLM": "VLMs",
    "Standalone LLM": "Standalone LLMs",
    "Think": "Think models",
}

MATCHED_COMPACT_MODELS = {
    "InternVL-8B",
    "Qwen3-VL-8B",
    "LLaVA-1.5-7B",
    "LLaVA-Mistral",
    "LLaVA-Vicuna",
    "InternVL-8B (LM)",
    "Qwen3-VL-8B (LM)",
    "LLaVA-1.5 (LM)",
    "LLaVA-Mistral (LM)",
    "LLaVA-Vicuna (LM)",
    "Qwen3-8B",
    "Qwen3-8B (think)",
    "Qwen2.5-7B-Instruct",
    "Vicuna-7B",
}


def infer_group(model: str) -> str:
    if model == "Human LOO":
        return "Human baseline"
    if "think" in model.lower():
        return "Think"
    if "(LM)" in model:
        return "Backbone Decoder"
    if "InternVL" in model or "Qwen3-VL" in model or "LLaVA" in model:
        return "VLM"
    return "Standalone LLM"


def display_name(model: str) -> str:
    if model == "Human LOO":
        return model
    name = model.replace(" (LM)", "")
    name = name.replace("-8B", "").replace("-7B", "").replace("-13B", "").replace("-32B", "").replace("-4B", "").replace("-2B", "").replace("-1B", "").replace("-1.7B", "").replace("-0.6B", "")
    name = name.replace("-Instruct", "")
    name = name.replace("Qwen2.5", "Qwen2.5")
    name = name.replace("Qwen3-VL", "Qwen3-VL")
    name = name.replace("Qwen3", "Qwen3")
    name = name.replace("Vicuna-v1.5", "Vicuna")
    name = name.replace("  ", " ").strip()
    name = name.replace(" ()", "")
    if "(think)" in model:
        return "Qwen3 (think)"
    return name


def size_label(model: str) -> str:
    if model == "Human LOO":
        return "--"
    size = MODEL_SIZE_B.get(model)
    if size is None:
        return "--"
    if float(size).is_integer():
        return f"{int(size)}B"
    return f"{size:g}B"


def build_block(filter_abstention: bool, *, matched_only: bool) -> pd.DataFrame:
    frames = []
    for dim in ("op", "ent"):
        df = grouped_distribution_long_table(
            dim,
            condition="inst_blind",
            variants=("C", "B", "A"),
            filter_abstention=filter_abstention,
            answer_types=("yes/no", "number", "text"),
            include_human_baseline=True,
        ).copy()
        if matched_only:
            df = df[df["Model"].isin(MATCHED_COMPACT_MODELS) | (df["Model"] == "Human LOO")].copy()
        df["Dimension"] = "Operation" if dim == "op" else "Entity"
        df["Group"] = df["Model"].map(infer_group)
        df["Display"] = df["Model"].map(display_name)
        df["Size"] = df["Model"].map(size_label)
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(
        index=["Model", "Display", "Group", "Size"],
        columns=["Dimension", "Answer type"],
        values="Mean JS",
        aggfunc="first",
    )
    for dim in ["Operation", "Entity"]:
        for at in ["yes/no", "number", "text"]:
            if (dim, at) not in wide.columns:
                wide[(dim, at)] = np.nan
        wide[(dim, "Overall")] = wide[[(dim, "yes/no"), (dim, "number"), (dim, "text")]].mean(axis=1)
    wide = wide[[("Operation", "yes/no"), ("Operation", "number"), ("Operation", "text"), ("Operation", "Overall"),
                 ("Entity", "yes/no"), ("Entity", "number"), ("Entity", "text"), ("Entity", "Overall")]]
    wide = wide.reset_index()
    wide.columns = ["Model", "Display", "Group", "Size", "Op Yes/No", "Op Count", "Op Text", "Op Overall",
                    "Ent Yes/No", "Ent Count", "Ent Text", "Ent Overall"]
    return wide


def _read_export(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "analysis/session2/exports" / name)


def _qid_to_answer_type() -> dict[int, str]:
    mapper = VQAAnswerMapper()
    mapper._load()
    return {int(qid): ann.get("answer_type", "other") for qid, ann in mapper.annotations.items()}


def compute_human_loocv(filter_abstention: bool) -> dict[str, tuple[float, float]]:
    """Return {answer_type: (min_js, max_js)} across participants (LOOCV)."""
    qid2atype = _qid_to_answer_type()
    human = load_human_responses().copy()
    human = human[human["variant"].isin({"C", "B", "A"})].copy()
    human["answer_type"] = human["question_id"].astype(int).map(qid2atype).fillna("other")
    human["clean_response"] = _clean_answer_series(human["response"])
    human = _mark_abstentions(human)

    result = {}
    for answer_type in ("yes/no", "number"):
        sub = _answer_type_subset(human, answer_type).copy()
        if filter_abstention:
            sub = sub[~sub["is_abstained"]].copy()
        participant_js = []
        for participant, held_out in sub.groupby("participant"):
            others = sub[sub["participant"] != participant]
            other_counts = others["clean_response"].value_counts()
            held_counts = held_out["clean_response"].value_counts()
            if other_counts.empty or held_counts.empty:
                continue
            _, _, js, _, _ = _distribution_stats_from_counts(other_counts, held_counts)
            participant_js.append(float(js))
        if participant_js:
            result[answer_type] = (float(np.min(participant_js)), float(np.max(participant_js)))
    return result


def build_pooled_block(filter_abstention: bool, *, matched_only: bool) -> pd.DataFrame:
    qid2atype = _qid_to_answer_type()
    human = load_human_responses().copy()
    model = _read_export("responses_model_inst_blind.csv").copy()

    variants = {"C", "B", "A"}
    human = human[human["variant"].isin(variants)].copy()
    model = model[model["variant"].isin(variants)].copy()
    if matched_only:
        model = model[model["model"].isin(MATCHED_COMPACT_MODELS)].copy()

    human["answer_type"] = human["question_id"].astype(int).map(qid2atype).fillna("other")
    model["answer_type"] = model["question_id"].astype(int).map(qid2atype).fillna("other")
    human["clean_response"] = _clean_answer_series(human["response"])
    model["clean_response"] = _clean_answer_series(model["response"])
    human = _mark_abstentions(human)
    model = _mark_abstentions(model)

    rows = []
    for model_name, model_all in model.groupby("model"):
        vals: dict[str, float] = {}
        for answer_type in ("yes/no", "number", "text"):
            human_sub = _answer_type_subset(human, answer_type).copy()
            model_sub = _answer_type_subset(model_all, answer_type).copy()

            if filter_abstention:
                model_sub = model_sub[~model_sub["is_abstained"]].copy()
                kept_qvs = set(zip(model_sub["question_id"].astype(int), model_sub["variant"].astype(str)))
                human_sub = human_sub[
                    [qv in kept_qvs for qv in zip(human_sub["question_id"].astype(int), human_sub["variant"].astype(str))]
                ].copy()

            if human_sub.empty or model_sub.empty:
                vals[answer_type] = np.nan
                continue

            human_counts = human_sub["clean_response"].value_counts()
            model_counts = model_sub["clean_response"].value_counts()
            _, _, js, _, _ = _distribution_stats_from_counts(human_counts, model_counts)
            vals[answer_type] = float(js)

        rows.append(
            {
                "Model": model_name,
                "Display": display_name(model_name),
                "Group": infer_group(model_name),
                "Size": size_label(model_name),
                "Yes/No": vals.get("yes/no", np.nan),
                "Count": vals.get("number", np.nan),
                "Text": vals.get("text", np.nan),
            }
        )

    df = pd.DataFrame(rows)
    df["Overall"] = df[["Yes/No", "Count", "Text"]].mean(axis=1)
    return df


def fmt(v: float) -> str:
    if pd.isna(v):
        return "--"
    s = f"{v:.2f}"
    if s.startswith("0"):
        return s[1:]
    if s.startswith("-0"):
        return "-" + s[2:]
    return s


def highlight_cells(df: pd.DataFrame, metric_cols: list[str]) -> dict[tuple[int, str], str]:
    out = {}
    model_rows = df[df["Group"] != "Human baseline"].copy()
    for col in metric_cols:
        vals = model_rows[col].dropna().astype(float)
        uniq = sorted(vals.unique().tolist())
        best = uniq[0] if uniq else np.nan
        second = uniq[1] if len(uniq) > 1 else np.nan
        for idx, row in df.iterrows():
            v = row[col]
            text = fmt(v)
            if row["Group"] != "Human baseline" and not pd.isna(v):
                if np.isclose(v, best):
                    text = rf"\textbf{{{text}}}"
                elif not np.isnan(second) and np.isclose(v, second):
                    text = rf"\underline{{{text}}}"
            out[(idx, col)] = text
    return out


def render_table(df: pd.DataFrame, caption: str, label: str, *, include_size: bool) -> str:
    metric_cols = ["Op Yes/No", "Op Count", "Op Text", "Op Overall",
                   "Ent Yes/No", "Ent Count", "Ent Text", "Ent Overall"]
    styled = highlight_cells(df, metric_cols)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + ("ll" if include_size else "l") + r"rrrrrrrr}")
    lines.append(r"\toprule")
    if include_size:
        lines.append(r"& & \multicolumn{4}{c}{Operation} & \multicolumn{4}{c}{Entity} \\")
        lines.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}")
        lines.append(r"Model & Size & Yes/No & Count & Text & Overall & Yes/No & Count & Text & Overall \\")
        section_span = 10
    else:
        lines.append(r"& \multicolumn{4}{c}{Operation} & \multicolumn{4}{c}{Entity} \\")
        lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
        lines.append(r"Model & Yes/No & Count & Text & Overall & Yes/No & Count & Text & Overall \\")
        section_span = 9
    lines.append(r"\midrule")

    for group in GROUP_ORDER:
        sub = df[df["Group"] == group].copy()
        if sub.empty:
            continue
        lines.append(rf"\multicolumn{{{section_span}}}{{c}}{{\textbf{{{SECTION_TITLES[group]}}}}} \\")
        lines.append(r"\midrule")
        for idx, row in sub.iterrows():
            vals = " & ".join(styled[(idx, c)] for c in metric_cols)
            if include_size:
                lines.append(f"{row['Display']} & {row['Size']} & {vals} \\\\")
            else:
                lines.append(f"{row['Display']} & {vals} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def render_pooled_table(df: pd.DataFrame, caption: str, label: str, *,
                        include_size: bool,
                        loocv_ranges=None) -> str:
    # loocv_ranges: {"yes/no": (lo, hi), "number": (lo, hi)}
    col_to_atype = {"Yes/No": "yes/no", "Count": "number"}
    metric_cols = ["Yes/No", "Count"]
    styled = {}
    for col in metric_cols:
        atype = col_to_atype[col]
        lo, hi = (loocv_ranges[atype] if loocv_ranges and atype in loocv_ranges else (None, None))
        for idx, row in df.iterrows():
            v = row[col]
            if pd.isna(v):
                text = "--"
            else:
                s = f"{v:.3f}"
                if s.startswith("0"):
                    text = s[1:]
                elif s.startswith("-0"):
                    text = "-" + s[2:]
                else:
                    text = s
            if not pd.isna(v) and lo is not None and lo <= v <= hi:
                text = rf"\textbf{{{text}}}"
            styled[(idx, col)] = text

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\tiny")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{2.1pt}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + ("ll" if include_size else "l") + r"rr}")
    lines.append(r"\toprule")
    if include_size:
        lines.append(r"Model & Size & Yes/No & Count \\")
        section_span = 4
    else:
        lines.append(r"Model & Yes/No & Count \\")
        section_span = 3
    lines.append(r"\midrule")

    for group in GROUP_ORDER:
        sub = df[df["Group"] == group].copy()
        if sub.empty:
            continue
        lines.append(rf"\multicolumn{{{section_span}}}{{c}}{{\textbf{{{SECTION_TITLES[group]}}}}} \\")
        lines.append(r"\midrule")
        for idx, row in sub.iterrows():
            vals = " & ".join(styled[(idx, c)] for c in metric_cols)
            if include_size:
                lines.append(f"{row['Display']} & {row['Size']} & {vals} \\\\")
            else:
                lines.append(f"{row['Display']} & {vals} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_one(filter_abstention: bool, *, matched_only: bool, include_size: bool):
    if matched_only:
        stem = "hm_grouped_distribution_7b_compact"
        title = "Matched 7/8B"
        label = "tab:hm_grouped_distribution_7b_compact"
        pooled = build_pooled_block(filter_abstention, matched_only=True)
        pooled.loc[pooled["Group"] == "Think", "Group"] = "Standalone LLM"
        loocv = compute_human_loocv(filter_abstention)
        yn_lo, yn_hi = loocv.get("yes/no", (None, None))
        ct_lo, ct_hi = loocv.get("number", (None, None))
        def _fv(v): return f"{v:.3f}".lstrip("0") if v is not None else "?"
        loocv_note = (
            f"Human LOOCV range: Yes/No [{_fv(yn_lo)}, {_fv(yn_hi)}], "
            f"Count [{_fv(ct_lo)}, {_fv(ct_hi)}]. "
            "Bold = within human LOOCV range."
        )
        caption = (
            f"{title} blind-VQA answer-distribution JS divergence to human answers across all control variants "
            + ("after abstention filtering. " if filter_abstention else "(inclusive version). ")
            + "Lower is better. " + loocv_note
        )
        out = OUT_DIR / f"{stem}{'_filtered' if filter_abstention else ''}.tex"
        if filter_abstention:
            label += "_filtered"
        out.write_text(render_pooled_table(pooled, caption, label, include_size=False, loocv_ranges=loocv))
        print(out)

        full = build_block(filter_abstention, matched_only=True)
        full_out = OUT_DIR / f"hm_grouped_distribution_7b_grouped_full{'_filtered' if filter_abstention else ''}.tex"
        full_label = "tab:hm_grouped_distribution_7b_grouped_full"
        if filter_abstention:
            full_label += "_filtered"
        full_caption = (
            f"{title} grouped answer-distribution divergence to humans across all control variants "
            + ("after abstention filtering. " if filter_abstention else "(inclusive version). ")
            + "Entries are mean JS divergence (lower is better); Operation and Entity columns summarize yes/no, count, and text slices plus their overall average. "
            + "Model size is shown explicitly. Bold = best (lowest) among model rows per column; underline = second-best."
        )
        full_out.write_text(render_table(full, full_caption, full_label, include_size=True))
        print(full_out)
    else:
        df = build_block(filter_abstention, matched_only=False)
        stem = "hm_grouped_distribution_all_scales_compact"
        title = "All-scale"
        label = "tab:hm_grouped_distribution_all_scales_compact"
        suffix = "_filtered" if filter_abstention else ""
        out = OUT_DIR / f"{stem}{suffix}.tex"
        caption = (
            f"{title} grouped answer-distribution divergence to humans across all control variants "
            + ("after abstention filtering. " if filter_abstention else "(inclusive version). ")
            + "Entries are mean JS divergence (lower is better); Operation and Entity columns summarize yes/no, count, and text slices plus their overall average. "
            + ("Model size is shown explicitly. " if include_size else "")
            + "Bold = best (lowest) among model rows per column; underline = second-best."
        )
        if filter_abstention:
            label += "_filtered"
        out.write_text(render_table(df, caption, label, include_size=include_size))
        print(out)


if __name__ == "__main__":
    write_one(False, matched_only=True, include_size=False)
    write_one(True, matched_only=True, include_size=False)
    write_one(False, matched_only=False, include_size=True)
    write_one(True, matched_only=False, include_size=True)
