from __future__ import annotations

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
    _grouped_slice_metrics,
    _mark_abstentions,
    _prepare_distribution_category,
    _distribution_stats_from_counts,
)
from analysis.utils.vqa import VQAAnswerMapper
from analysis.utils.vqa import preprocess_answer

# Import LOO min-max range (same definition used in grouped_compact tables and paper text)
sys.path.insert(0, str(Path(__file__).parent))
from gen_grouped_distribution_compact import compute_human_loocv as _compute_loocv_minmax


GROUP_ORDER = ["VLM", "Backbone Decoder", "Standalone LLM"]
MODEL_ORDER = {
    "VLM": ["InternVL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna", "Qwen3-VL-8B"],
    "Backbone Decoder": ["InternVL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)", "Qwen3-VL-8B (LM)"],
    "Standalone LLM": ["Qwen2.5-7B-Instruct", "Qwen3-8B", "Qwen3-8B (think)", "Vicuna-7B"],
}
SECTION_TITLES = {
    "VLM": "VLMs",
    "Backbone Decoder": "Backbone Decoders",
    "Standalone LLM": "Standalone LLMs",
}
MODEL_SIZE = {
    "Qwen3-VL-8B": "8B",
    "InternVL-8B": "8B",
    "LLaVA-1.5-7B": "7B",
    "LLaVA-Mistral": "7B",
    "LLaVA-Vicuna": "7B",
    "Qwen3-VL-8B (LM)": "8B",
    "InternVL-8B (LM)": "8B",
    "LLaVA-1.5 (LM)": "7B",
    "LLaVA-Mistral (LM)": "7B",
    "LLaVA-Vicuna (LM)": "7B",
    "Qwen3-8B": "8B",
    "Qwen3-8B (think)": "8B",
    "Qwen2.5-7B-Instruct": "7B",
    "Mistral-7B": "7B",
    "Vicuna-7B": "7B",
}


def _display_name(model: str) -> str:
    if "(think)" in model:
        return "Qwen3 (think)"
    name = model.replace(" (LM)", "")
    for suffix in ("-8B", "-7B", "-13B", "-32B", "-4B", "-2B", "-1B", "-1.7B", "-0.6B", "-Instruct"):
        name = name.replace(suffix, "")
    if name == "Qwen3-VL":
        return "Qwen3-VL"
    if name == "LLaVA-1.5":
        return "LLaVA-1.5"
    if name == "LLaVA-Mistral":
        return "LLaVA-Mistral"
    if name == "LLaVA-Vicuna":
        return "LLaVA-Vicuna"
    if name == "InternVL":
        return "InternVL"
    if name == "Qwen2.5":
        return "Qwen2.5"
    if name == "Qwen3":
        return "Qwen3"
    if name == "Mistral":
        return "Mistral"
    if name == "Vicuna":
        return "Vicuna"
    return name.strip()


def _write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path)


def _zero_share(df: pd.DataFrame) -> float:
    cleaned = df["response"].fillna("").astype(str).map(lambda x: preprocess_answer(x, strip_think=True)).str.lower().str.strip()
    if len(cleaned) == 0:
        return float("nan")
    return float((cleaned == "0").mean() * 100.0)


def _read_export(name: str) -> pd.DataFrame:
    path = ROOT / "analysis/session2/exports" / name
    return pd.read_csv(path)


def _qid_to_answer_type() -> dict[int, str]:
    mapper = VQAAnswerMapper()
    mapper._load()
    return {int(qid): ann.get("answer_type", "other") for qid, ann in mapper.annotations.items()}


def _distribution_base_table_variants_with_models(
    condition: str,
    variants: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qid2atype = _qid_to_answer_type()
    model_keep = {m for models in MODEL_ORDER.values() for m in models}

    human = load_human_responses()
    model = _read_export(f"responses_model_{condition}.csv")

    human = human[human["variant"].isin(variants)].copy()
    model = model[(model["variant"].isin(variants)) & (model["model"].isin(model_keep))].copy()

    qvs = set(zip(human["question_id"].astype(int), human["variant"].astype(str)))
    human = human[[qv in qvs for qv in zip(human["question_id"].astype(int), human["variant"].astype(str))]].copy()
    model = model[[qv in qvs for qv in zip(model["question_id"].astype(int), model["variant"].astype(str))]].copy()

    human["answer_type"] = human["question_id"].astype(int).map(qid2atype).fillna("other")
    model["answer_type"] = model["question_id"].astype(int).map(qid2atype).fillna("other")
    human["clean_response"] = _clean_answer_series(human["response"])
    model["clean_response"] = _clean_answer_series(model["response"])
    return human, model


def _grouped_distribution_long_table_with_models(
    *,
    dimension: str,
    condition: str,
    variants: tuple[str, ...],
    filter_abstention: bool,
    answer_types: tuple[str, ...],
    min_questions_per_slice: int = 3,
) -> pd.DataFrame:
    human_all, model_all = _distribution_base_table_variants_with_models(condition=condition, variants=variants)
    human_all = _mark_abstentions(human_all)
    model_all = _mark_abstentions(model_all)

    rows = []
    for model_name, model_sub_all in model_all.groupby("model"):
        for answer_type in answer_types:
            model_sub = _answer_type_subset(model_sub_all, answer_type)
            human_sub = _answer_type_subset(human_all, answer_type)

            if filter_abstention:
                model_sub = model_sub[~model_sub["is_abstained"]].copy()
                kept_qvs = set(zip(model_sub["question_id"].astype(int), model_sub["variant"].astype(str)))
                human_sub = human_sub[
                    [qv in kept_qvs for qv in zip(human_sub["question_id"].astype(int), human_sub["variant"].astype(str))]
                ].copy()
                human_sub = human_sub[~human_sub["is_abstained"]].copy()

            slice_df = _grouped_slice_metrics(
                human_sub,
                model_sub,
                answer_type,
                dimension,
                min_questions_per_slice=min_questions_per_slice,
            )
            if slice_df.empty:
                continue

            rows.append(
                {
                    "Model": model_name,
                    "Answer type": answer_type,
                    "Mean JS": float(slice_df["js"].mean()),
                    "Mean TV": float(slice_df["tv"].mean()),
                    "N qv": int(slice_df["n_questions"].sum()),
                }
            )

    return pd.DataFrame(rows)


def build_zero_bias_count() -> str:
    human = load_human_responses()
    blind = _read_export("responses_model_blind.csv")
    inst = _read_export("responses_model_inst_blind.csv")
    qid2atype = _qid_to_answer_type()

    human = human.copy()
    blind = blind.copy()
    inst = inst.copy()
    for df in (human, blind, inst):
        df["answer_type"] = df["question_id"].astype(int).map(qid2atype).fillna("other")

    count_qids = set(human[(human["variant"] == "C") & (human["answer_type"] == "number")]["question_id"].astype(int).unique())
    human_c = human[(human["variant"] == "C") & (human["question_id"].isin(count_qids))].copy()
    blind_c = blind[(blind["variant"] == "C") & (blind["question_id"].isin(count_qids))].copy()
    inst_c = inst[(inst["variant"] == "C") & (inst["question_id"].isin(count_qids))].copy()

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Fraction of count-question answers equal to ``0'', by matched 7/8B model and condition (variant~C, 22 count questions). Human rate reflects natural scene plausibility; VLM zero-bias reflects treating the absence of an image as the absence of objects.}")
    lines.append(r"\label{tab:zero_bias_count}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Blind & +Instruction \\")
    lines.append(r"\midrule")
    for grp in GROUP_ORDER:
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{grp}}}}} \\")
        for model in MODEL_ORDER[grp]:
            b = _zero_share(blind_c[blind_c["model"] == model])
            i = _zero_share(inst_c[inst_c["model"] == model])
            lines.append(f"{_display_name(model)} & {b:.1f}\\% & {i:.1f}\\% \\\\")
        lines.append(r"\addlinespace[1pt]")
    lines.append(r"\midrule")
    lines.append(f"Human & \\multicolumn{{2}}{{c}}{{{_zero_share(human_c):.1f}\\%}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _fmt_delta(a: float, b: float) -> str:
    d = b - a
    color = "cyan!50!black" if d < 0 else "red!70!black"
    sign = "+" if d >= 0 else ""
    return f"{b:.3f} (\\textcolor{{{color}}}{{{sign}{d:.3f}}})"


def _fmt_js(v: float) -> str:
    s = f"{v:.3f}"
    if s.startswith("0"):
        return s[1:]
    if s.startswith("-0"):
        return "-" + s[2:]
    return s


def _style_val(val: float, metric: str, best: float, second: float) -> str:
    text = _fmt_js(val)
    if pd.isna(val):
        return text
    if metric in {"JS", "TV"}:
        if pd.notna(best) and abs(val - best) < 1e-12:
            return rf"\textbf{{{text}}}"
        if pd.notna(second) and abs(val - second) < 1e-12:
            return rf"\underline{{{text}}}"
    if metric == "Mode":
        if pd.notna(best) and abs(val - best) < 1e-12:
            return rf"\textbf{{{text}}}"
        if pd.notna(second) and abs(val - second) < 1e-12:
            return rf"\underline{{{text}}}"
    return text


def _bootstrap_loo_ci(
    human_sub: pd.DataFrame,
    category_order: list,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap 95% CI for human LOO JS and Mode by resampling (question_id, variant) pairs."""
    rng = np.random.default_rng(seed)
    qv_js: dict = {}
    qv_mode: dict = {}
    for (qid, variant), hq in human_sub.groupby(["question_id", "variant"]):
        js_list, mode_list = [], []
        for idx, row_h in hq.iterrows():
            answer = row_h["category"]
            rest = hq.drop(index=idx)
            if rest.empty:
                continue
            human_counts = rest["category"].value_counts().reindex(category_order, fill_value=0)
            loo_counts = pd.Series(0.0, index=category_order)
            if answer in loo_counts.index:
                loo_counts.loc[answer] = 1.0
            if human_counts.sum() == 0 or loo_counts.sum() == 0:
                continue
            _, _, js, _, _ = _distribution_stats_from_counts(human_counts, loo_counts)
            js_list.append(js)
            max_count = human_counts.max()
            modal_cats = set(human_counts[human_counts == max_count].index.tolist())
            mode_list.append(float(answer in modal_cats))
        if js_list:
            qv_js[(qid, variant)] = float(np.mean(js_list))
            qv_mode[(qid, variant)] = float(np.mean(mode_list))

    qv_keys = list(qv_js.keys())
    if not qv_keys:
        nan2 = (float("nan"), float("nan"))
        return nan2, nan2

    boot_js, boot_mode = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(qv_keys), size=len(qv_keys))
        boot_js.append(float(np.mean([qv_js[qv_keys[i]] for i in idx])))
        boot_mode.append(float(np.mean([qv_mode[qv_keys[i]] for i in idx])))

    js_ci = (float(np.percentile(boot_js, 2.5)), float(np.percentile(boot_js, 97.5)))
    mode_ci = (float(np.percentile(boot_mode, 2.5)), float(np.percentile(boot_mode, 97.5)))
    return js_ci, mode_ci


def _style_val_ci(val: float, metric: str, ci: tuple[float, float], is_median_winner: bool = False) -> str:
    """Bold if val falls within the human LOO 95% CI; underline if closest to human median."""
    text = _fmt_js(val)
    if pd.isna(val) or any(np.isnan(c) for c in ci):
        return rf"\underline{{{text}}}" if is_median_winner else text
    lo, hi = ci
    if lo <= val <= hi:
        text = rf"\textbf{{{text}}}"
    if is_median_winner:
        text = rf"\underline{{{text}}}"
    return text


def build_dist_inst_effect_js_summary() -> str:
    human_all, model_all = _distribution_base_table_variants_with_models(
        condition="inst_blind",
        variants=("C", "B", "A"),
    )
    human_all = _mark_abstentions(human_all)
    model_all = _mark_abstentions(model_all)

    rows = []
    for model_name, model_sub_all in model_all.groupby("model"):
        row = {"Model": model_name}
        for answer_type in ("yes/no", "number"):
            human_sub, model_sub, category_order = _prepare_distribution_category(
                human_all.copy(), model_sub_all.copy(), answer_type
            )

            model_sub = model_sub[~model_sub["is_abstained"]].copy()
            kept_qvs = set(zip(model_sub["question_id"].astype(int), model_sub["variant"].astype(str)))
            human_sub = human_sub[
                [qv in kept_qvs for qv in zip(human_sub["question_id"].astype(int), human_sub["variant"].astype(str))]
            ].copy()
            human_sub = human_sub[~human_sub["is_abstained"]].copy()

            js_vals = []
            tv_vals = []
            mode_hits = []
            for (qid, variant), hq in human_sub.groupby(["question_id", "variant"]):
                mq = model_sub[
                    (model_sub["question_id"].astype(int) == int(qid))
                    & (model_sub["variant"].astype(str) == str(variant))
                ]
                if hq.empty or mq.empty:
                    continue

                human_counts = hq["category"].value_counts().reindex(category_order, fill_value=0)
                model_counts = mq["category"].value_counts().reindex(category_order, fill_value=0)
                if human_counts.sum() == 0 or model_counts.sum() == 0:
                    continue

                _, _, js, tv, _ = _distribution_stats_from_counts(human_counts, model_counts)
                js_vals.append(js)
                tv_vals.append(tv)

                max_count = human_counts.max()
                modal_cats = set(human_counts[human_counts == max_count].index.tolist())
                model_cat = model_counts.idxmax()
                mode_hits.append(float(model_cat in modal_cats))

            row[(answer_type, "N")] = len(js_vals)
            row[(answer_type, "JS")] = float(pd.Series(js_vals).mean()) if js_vals else float("nan")
            row[(answer_type, "TV")] = float(pd.Series(tv_vals).mean()) if tv_vals else float("nan")
            row[(answer_type, "Mode")] = float(pd.Series(mode_hits).mean()) if mode_hits else float("nan")
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("Model")

    rankings = {}
    for answer_type in ("yes/no", "number"):
        for metric in ("JS", "TV", "Mode"):
            vals = summary.loc[summary.index != "Human LOO", (answer_type, metric)].dropna().astype(float)
            uniq = sorted(set(vals.tolist()), reverse=(metric == "Mode"))
            best = uniq[0] if uniq else float("nan")
            second = uniq[1] if len(uniq) > 1 else float("nan")
            rankings[(answer_type, metric)] = (best, second)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.2pt}")
    lines.append(r"\caption{Blind+instruction per-question categorical blind-VQA answer-distribution alignment to humans, pooled across all three variants and computed on the shared non-abstaining subset. Lower is better for JS/TV ($\downarrow$); higher is better for Mode ($\uparrow$). Best results are shown in \textbf{bold}; second-best are \underline{underlined}.}")
    lines.append(r"\label{tab:dist_inst_effect_js_summary}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccc@{\hspace{6pt}}cccc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Model}} &")
    lines.append(r"\multicolumn{4}{c|}{\textbf{Yes/No} ($N{=}78$)} &")
    lines.append(r"\multicolumn{4}{c}{\textbf{Count} ($N{=}78$)} \\")
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    lines.append(r"& $\mathbf{N_q}$ & \textbf{JS}$\downarrow$ & \textbf{TV}$\downarrow$ & \textbf{Mode}$\uparrow$ & $\mathbf{N_q}$ & \textbf{JS}$\downarrow$ & \textbf{TV}$\downarrow$ & \textbf{Mode}$\uparrow$ \\")
    lines.append(r"\midrule")
    for grp in GROUP_ORDER:
        lines.append(rf"\multicolumn{{9}}{{c}}{{\textbf{{{SECTION_TITLES[grp]}}}}} \\")
        lines.append(r"\midrule")
        for model in MODEL_ORDER[grp]:
            if model not in summary.index:
                continue
            yn_n = int(summary.at[model, ("yes/no", "N")])
            yn_js = float(summary.at[model, ("yes/no", "JS")])
            yn_tv = float(summary.at[model, ("yes/no", "TV")])
            yn_mode = float(summary.at[model, ("yes/no", "Mode")])
            ct_n = int(summary.at[model, ("number", "N")])
            ct_js = float(summary.at[model, ("number", "JS")])
            ct_tv = float(summary.at[model, ("number", "TV")])
            ct_mode = float(summary.at[model, ("number", "Mode")])
            yn_js_s = _style_val(yn_js, "JS", *rankings[("yes/no", "JS")])
            yn_tv_s = _style_val(yn_tv, "TV", *rankings[("yes/no", "TV")])
            yn_mode_s = _style_val(yn_mode, "Mode", *rankings[("yes/no", "Mode")])
            ct_js_s = _style_val(ct_js, "JS", *rankings[("number", "JS")])
            ct_tv_s = _style_val(ct_tv, "TV", *rankings[("number", "TV")])
            ct_mode_s = _style_val(ct_mode, "Mode", *rankings[("number", "Mode")])
            lines.append(
                f"{_display_name(model)} & {yn_n} & {yn_js_s} & {yn_tv_s} & {yn_mode_s} & {ct_n} & {ct_js_s} & {ct_tv_s} & {ct_mode_s} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_hm_grouped_distribution_7b_compact_filtered() -> str:
    variants = ("C", "B", "A")
    human_all, model_blind = _distribution_base_table_variants_with_models("blind", variants)
    _, model_inst = _distribution_base_table_variants_with_models("inst_blind", variants)
    human_all  = _mark_abstentions(human_all)
    model_blind = _mark_abstentions(model_blind)
    model_inst  = _mark_abstentions(model_inst)

    pooled_n = {}
    for answer_type in ("yes/no", "number"):
        human_sub = _answer_type_subset(human_all.copy(), answer_type)
        pooled_n[answer_type] = len(set(zip(human_sub["question_id"].astype(int), human_sub["variant"].astype(str))))

    # Human LOO baseline (from human data, no instruction delta)
    human_row = {"Model": "Human LOO"}
    _loo_human_sub: dict = {}
    _loo_category_order: dict = {}
    for answer_type in ("yes/no", "number"):
        human_sub = _answer_type_subset(human_all.copy(), answer_type)
        _, human_sub, category_order = _prepare_distribution_category(human_all.copy(), human_sub.copy(), answer_type)
        _loo_human_sub[answer_type] = human_sub
        _loo_category_order[answer_type] = category_order
        js_vals, mode_hits, qv_seen = [], [], set()
        for (qid, variant), hq in human_sub.groupby(["question_id", "variant"]):
            if hq.empty:
                continue
            qv_seen.add((int(qid), str(variant)))
            for idx, row_h in hq.iterrows():
                answer = row_h["category"]
                rest = hq.drop(index=idx)
                if rest.empty:
                    continue
                human_counts = rest["category"].value_counts().reindex(category_order, fill_value=0)
                loo_counts = pd.Series(0, index=category_order, dtype=float)
                if answer in loo_counts.index:
                    loo_counts.loc[answer] = 1.0
                if human_counts.sum() == 0 or loo_counts.sum() == 0:
                    continue
                _, _, js, _, _ = _distribution_stats_from_counts(human_counts, loo_counts)
                js_vals.append(js)
                max_count = human_counts.max()
                modal_cats = set(human_counts[human_counts == max_count].index.tolist())
                mode_hits.append(float(answer in modal_cats))
        human_row[(answer_type, "N")]    = len(qv_seen)
        human_row[(answer_type, "JS")]   = float(pd.Series(js_vals).mean()) if js_vals else float("nan")
        human_row[(answer_type, "Mode")] = float(pd.Series(mode_hits).mean()) if mode_hits else float("nan")

    # Bootstrap 95% CI for both JS and Mode
    loo_ci: dict = {}
    for answer_type in ("yes/no", "number"):
        js_ci, mode_ci = _bootstrap_loo_ci(
            _loo_human_sub[answer_type], _loo_category_order[answer_type]
        )
        loo_ci[(answer_type, "JS")]   = js_ci
        loo_ci[(answer_type, "Mode")] = mode_ci

    blind_metrics = _compute_js_mode_per_model(human_all, model_blind)
    inst_metrics  = _compute_js_mode_per_model(human_all, model_inst)

    # Compute count abstention rates for both blind and inst_blind (for ΔAbs)
    qid2atype = _qid_to_answer_type()
    model_inst_raw = _read_export("responses_model_inst_blind.csv").copy()
    model_blind_raw = _read_export("responses_model_blind.csv").copy()
    for df in (model_inst_raw, model_blind_raw):
        df["clean_response"] = _clean_answer_series(df["response"])
        _mark_abstentions(df)
        df["answer_type"] = df["question_id"].astype(int).map(qid2atype).fillna("other")
    model_inst_raw  = _mark_abstentions(model_inst_raw)
    model_blind_raw = _mark_abstentions(model_blind_raw)
    model_inst_raw["answer_type"]  = model_inst_raw["question_id"].astype(int).map(qid2atype).fillna("other")
    model_blind_raw["answer_type"] = model_blind_raw["question_id"].astype(int).map(qid2atype).fillna("other")

    abs_rates_inst:  dict[str, float] = {}
    abs_rates_blind: dict[str, float] = {}
    for model_name, grp in model_inst_raw[model_inst_raw["answer_type"] == "number"].groupby("model"):
        abs_rates_inst[model_name] = float(grp["is_abstained"].mean() * 100)
    for model_name, grp in model_blind_raw[model_blind_raw["answer_type"] == "number"].groupby("model"):
        abs_rates_blind[model_name] = float(grp["is_abstained"].mean() * 100)

    # Find closest-to-mean models per column (human LOO mean as target)
    model_set = {m for grp in MODEL_ORDER.values() for m in grp}
    median_winners: dict[str, str] = {}       # JS column winner per atype
    median_winners_mode: dict[str, str] = {}  # Mode column winner per atype
    for atype in ("yes/no", "number"):
        target_js = human_row[(atype, "JS")]
        candidates_js = [
            (m, abs(inst_metrics[m].get((atype, "JS"), float("inf")) - target_js))
            for m in model_set
            if m in inst_metrics and not pd.isna(inst_metrics[m].get((atype, "JS"), float("nan")))
        ]
        if candidates_js:
            median_winners[atype] = min(candidates_js, key=lambda x: x[1])[0]

        target_mode = human_row[(atype, "Mode")]
        candidates_mode = [
            (m, abs(inst_metrics[m].get((atype, "Mode"), float("inf")) - target_mode))
            for m in model_set
            if m in inst_metrics and not pd.isna(inst_metrics[m].get((atype, "Mode"), float("nan")))
        ]
        if candidates_mode:
            median_winners_mode[atype] = min(candidates_mode, key=lambda x: x[1])[0]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        "",
        r"\begin{tabular}{lcc@{\hspace{6pt}}cc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} &",
        rf"\multicolumn{{2}}{{c}}{{\textbf{{Yes/No}} ($N{{=}}{pooled_n['yes/no']}$)}} &",
        rf"\multicolumn{{2}}{{c}}{{\textbf{{Count}} ($N{{=}}{pooled_n['number']}$)}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"& \textbf{$D_{\mathrm{JS}}$} & \textbf{Mode} & \textbf{$D_{\mathrm{JS}}$} & \textbf{Abs(\%)} \\",
        r"\midrule",
    ]

    def _fmt_ci_cell(mean_val: float, lo: float, hi: float) -> str:
        v = _fmt_js(mean_val)
        ci = rf"{{\scriptsize [{_fmt_js(lo)}, {_fmt_js(hi)}]}}"
        return rf"\makecell[c]{{{v} \\ {ci}}}"

    loo_yn_js   = _fmt_ci_cell(float(human_row[("yes/no", "JS")]),   *loo_ci[("yes/no", "JS")])
    loo_yn_mode = _fmt_ci_cell(float(human_row[("yes/no", "Mode")]), *loo_ci[("yes/no", "Mode")])
    loo_ct_js   = _fmt_ci_cell(float(human_row[("number", "JS")]),   *loo_ci[("number", "JS")])
    loo_ct_mode = _fmt_ci_cell(float(human_row[("number", "Mode")]), *loo_ci[("number", "Mode")])
    lines.append(
        rf"\textbf{{Human LOO}} & {loo_yn_js} & {loo_yn_mode} & {loo_ct_js} & --- \\"
    )
    lines.append(r"\midrule")

    for grp in GROUP_ORDER:
        lines.append(rf"\multicolumn{{5}}{{c}}{{\textbf{{{SECTION_TITLES[grp]}}}}} \\")
        lines.append(r"\midrule")
        for model in MODEL_ORDER[grp]:
            if model not in inst_metrics:
                continue
            im = inst_metrics[model]
            cells = []
            for atype in ("yes/no", "number"):
                js_inst   = im.get((atype, "JS"), float("nan"))
                mode_inst = im.get((atype, "Mode"), float("nan"))
                is_winner      = (median_winners.get(atype) == model)
                is_mode_winner = (median_winners_mode.get(atype) == model)
                js_s   = _style_val_ci(js_inst,   "JS",   loo_ci[(atype, "JS")],   is_winner)
                mode_s = _style_val_ci(mode_inst, "Mode", loo_ci[(atype, "Mode")], is_mode_winner)
                if atype == "number":
                    abs_inst  = abs_rates_inst.get(model, float("nan"))
                    abs_blind = abs_rates_blind.get(model, float("nan"))
                    if not pd.isna(abs_inst) and not pd.isna(abs_blind):
                        d = abs_inst - abs_blind
                        if abs(d) < 0.05:
                            abs_s = f"{abs_inst:.1f}"
                        else:
                            sign = "+" if d >= 0 else ""
                            color = "cyan!50!black" if d < 0 else "red!70!black"
                            abs_s = rf"{abs_inst:.1f} {{\scriptsize (\textcolor{{{color}}}{{{sign}{d:.1f}}})}}"
                    else:
                        abs_s = f"{abs_inst:.1f}" if not pd.isna(abs_inst) else "---"
                    cells += [js_s, abs_s]
                else:
                    cells += [js_s, mode_s]
            lines.append(f"{_display_name(model)} & {' & '.join(cells)} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{$\djs$ divergence and modal-match rate vs.\ the human distribution. "
        r"Human LOO reports the leave-one-out \emph{mean}; human CI (LOO 95\% bootstrap): Y/N $\djs$ [.260, .282]; "
        r"Y/N Mode [.610, .692]; Count $\djs$ [.493, .554]; Count Mode [.451, .522]. "
        r"Abs = abstention rate; parenthetical values are with-instruction minus blind abstention "
        r"(\textcolor{cyan!50!black}{teal}: reduced abstention under instruction; \textcolor{red!70!black}{dark red}: increased abstention under instruction). "
        r"\colorbox{green!20}{Green}: within human CI. "
        r"\colorbox{orange!30}{Orange}: over-concentrated ($\djs$ below CI; Mode above CI). "
        r"\textbf{Bold}: closest to LOO mean per column; \underline{underline}: second-closest.}",
        r"\label{tab:hm_grouped_distribution_7b_compact_filtered}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _compute_js_mode_per_model(human_all: pd.DataFrame, model_all: pd.DataFrame) -> dict:
    """Return {model: {(answer_type, metric): value}} using per-question JS and Mode."""
    result = {}
    for model_name, model_sub_all in model_all.groupby("model"):
        row = {}
        for answer_type in ("yes/no", "number"):
            human_sub, model_sub, category_order = _prepare_distribution_category(
                human_all.copy(), model_sub_all.copy(), answer_type
            )
            model_sub = model_sub[~model_sub["is_abstained"]].copy()
            kept_qvs = set(zip(model_sub["question_id"].astype(int), model_sub["variant"].astype(str)))
            human_sub = human_sub[
                [qv in kept_qvs for qv in zip(human_sub["question_id"].astype(int), human_sub["variant"].astype(str))]
            ].copy()
            human_sub = human_sub[~human_sub["is_abstained"]].copy()
            js_vals, mode_hits = [], []
            for (qid, variant), hq in human_sub.groupby(["question_id", "variant"]):
                mq = model_sub[
                    (model_sub["question_id"].astype(int) == int(qid))
                    & (model_sub["variant"].astype(str) == str(variant))
                ]
                if hq.empty or mq.empty:
                    continue
                human_counts = hq["category"].value_counts().reindex(category_order, fill_value=0)
                model_counts = mq["category"].value_counts().reindex(category_order, fill_value=0)
                if human_counts.sum() == 0 or model_counts.sum() == 0:
                    continue
                _, _, js, _, _ = _distribution_stats_from_counts(human_counts, model_counts)
                js_vals.append(js)
                max_count = human_counts.max()
                modal_cats = set(human_counts[human_counts == max_count].index.tolist())
                mode_hits.append(float(model_counts.idxmax() in modal_cats))
            row[(answer_type, "N")] = len(js_vals)
            row[(answer_type, "JS")] = float(pd.Series(js_vals).mean()) if js_vals else float("nan")
            row[(answer_type, "Mode")] = float(pd.Series(mode_hits).mean()) if mode_hits else float("nan")
        result[model_name] = row
    return result


def _fmt_delta_inline(delta: float) -> str:
    """Render Δ as small plain text for inline display in JS cell."""
    if pd.isna(delta):
        return ""
    s = f"{delta:+.3f}"
    if s.startswith("+0."):
        s = "+" + s[2:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return rf" {{\scriptsize ({s})}}"


def build_hm_distribution_blind_vs_inst() -> str:
    """Appendix table: blind vs instruction DJS (count + yes/no) with delta."""
    variants = ("C", "B", "A")
    human_all, model_blind = _distribution_base_table_variants_with_models("blind", variants)
    _, model_inst = _distribution_base_table_variants_with_models("inst_blind", variants)
    human_all   = _mark_abstentions(human_all)
    model_blind = _mark_abstentions(model_blind)
    model_inst  = _mark_abstentions(model_inst)

    bm = _compute_js_mode_per_model(human_all, model_blind)
    im = _compute_js_mode_per_model(human_all, model_inst)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular}{lcc@{\hspace{4pt}}cc@{\hspace{4pt}}cc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} &",
        r"\multicolumn{3}{c}{\textbf{Yes/No}} &",
        r"\multicolumn{3}{c}{\textbf{Count}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"& \textbf{Blind} & \textbf{+Inst} & $\mathbf{\Delta}$ & \textbf{Blind} & \textbf{+Inst} & $\mathbf{\Delta}$ \\",
        r"\midrule",
    ]

    for grp in GROUP_ORDER:
        lines.append(rf"\multicolumn{{7}}{{c}}{{\textbf{{{SECTION_TITLES[grp]}}}}} \\")
        lines.append(r"\midrule")
        for model in MODEL_ORDER[grp]:
            if model not in bm or model not in im:
                continue
            row_parts = []
            for atype in ("yes/no", "number"):
                b = bm[model].get((atype, "JS"), float("nan"))
                i = im[model].get((atype, "JS"), float("nan"))
                d = i - b
                color = "cyan!50!black" if d < -0.001 else ("red!70!black" if d > 0.001 else "black")
                sign = "+" if d >= 0 else ""
                d_s = rf"\textcolor{{{color}}}{{{sign}{_fmt_js(d)}}}"
                row_parts += [_fmt_js(b), _fmt_js(i), d_s]
            lines.append(f"{_display_name(model)} & {' & '.join(row_parts)} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{$\djs$ divergence to the human distribution under blind and blind+instruction conditions, "
        r"with $\Delta = \text{inst} - \text{blind}$ (\textcolor{cyan!50!black}{teal}: improvement; \textcolor{red!70!black}{dark red}: worsening). "
        r"Values computed on the shared non-abstaining subset per condition.}",
        r"\label{tab:hm_distribution_blind_vs_inst}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    _write(OUT_DIR / "hm_grouped_distribution_7b_compact_filtered.tex", build_hm_grouped_distribution_7b_compact_filtered())
    _write(OUT_DIR / "hm_distribution_blind_vs_inst.tex", build_hm_distribution_blind_vs_inst())
