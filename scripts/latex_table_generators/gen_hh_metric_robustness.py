from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from config import MIN_ANSWERS_DEFAULT
from helpers import load_cleaned_pair_cache, load_human_subset
from analysis.utils.vqa import preprocess_answer

OUT = ROOT / "latex" / "AAAI2026" / "LaTeX" / "tables" / "hh_metric_robustness.tex"
BOOT_N = 2000
RNG = np.random.default_rng(12345)

VARIANTS = [("C", "Orig."), ("B", "Wkr."), ("A", "Pron.")]
METRICS = [
    ("sbert_score", "SBERT", True, 3),
    ("chrf_score", "chrF", True, 3),
    ("rouge1_score", "ROUGE-1", True, 3),
    ("jaccard_score", "Jaccard", True, 3),
    ("entropy", "Entropy (bits)", False, 2),
]

OP_ORDER = ["exist", "know", "act", "count", "attr", "ident", "temp", "spat", "text", "cause", "comp"]
ENT_ORDER = ["food", "animal", "place", "object", "person", "product", "vehicle", "other", "text"]

OP_FULL_NAMES = {
    "act": "Action", "attr": "Attribute", "cause": "Causality",
    "comp": "Comparison", "count": "Count", "exist": "Existence",
    "ident": "Identity", "know": "World Know.", "spat": "Spatial",
    "temp": "Temporal", "text": "Text Reading", "other": "Other",
}
ENT_FULL_NAMES = {
    "animal": "Animal", "food": "Food", "object": "Object",
    "other": "Other", "person": "Person", "place": "Place",
    "product": "Product", "text": "Text", "vehicle": "Vehicle",
}


def bootstrap_ci(vals: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return (0.0, 0.0)
    idx = RNG.integers(0, arr.size, size=(BOOT_N, arr.size))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mean = arr.mean()
    return float(mean - lo), float(hi - mean)


def entropy_bits(series: pd.Series) -> float:
    vals = [preprocess_answer(x, strip_think=True).lower() for x in series.fillna("").astype(str)]
    vals = [v for v in vals if v]
    if not vals:
        return np.nan
    counts = pd.Series(vals).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


def summarize_pair_metric(hh: pd.DataFrame, group_col: str, metric_col: str) -> dict[tuple[str, str], tuple[float, float, int]]:
    qmeans = (
        hh.groupby([group_col, "variant", "question_id"])[metric_col]
        .mean()
        .reset_index()
    )
    out: dict[tuple[str, str], tuple[float, float, int]] = {}
    for (grp, var), sub in qmeans.groupby([group_col, "variant"]):
        vals = sub[metric_col].astype(float).to_numpy()
        mean = float(np.mean(vals)) if len(vals) else np.nan
        lo, hi = bootstrap_ci(vals)
        out[(grp, var)] = (mean, max(lo, hi), int(sub["question_id"].nunique()))
    return out


def summarize_entropy(human: pd.DataFrame, group_col: str) -> dict[tuple[str, str], tuple[float, float, int]]:
    qent = (
        human.groupby([group_col, "variant", "question_id"])["response"]
        .apply(entropy_bits)
        .reset_index(name="entropy")
    )
    out: dict[tuple[str, str], tuple[float, float, int]] = {}
    for (grp, var), sub in qent.groupby([group_col, "variant"]):
        vals = sub["entropy"].astype(float).to_numpy()
        mean = float(np.mean(vals)) if len(vals) else np.nan
        lo, hi = bootstrap_ci(vals)
        out[(grp, var)] = (mean, max(lo, hi), int(sub["question_id"].nunique()))
    return out


def fmt_num(x: float, digits: int, drop_zero: bool) -> str:
    s = f"{x:.{digits}f}"
    if drop_zero and s.startswith("0"):
        s = s[1:]
    if drop_zero and s.startswith("-0"):
        s = "-" + s[2:]
    return s


def build_section_rows(order: list[str], label: str, metric_maps: dict[str, dict], highlight_metrics: set[str], name_map: dict[str, str] | None = None) -> list[str]:
    metric_cols = [(mkey, var) for mkey, _, _, _ in METRICS[:-1] for var, _ in VARIANTS]
    best_vals = {}
    second_vals = {}
    for mkey, var in metric_cols:
        vals = []
        for grp in order:
            item = metric_maps[mkey].get((grp, var))
            if item is not None and np.isfinite(item[0]):
                vals.append(float(item[0]))
        uniq = sorted(set(vals), reverse=True)
        best_vals[(mkey, var)] = uniq[0] if uniq else np.nan
        second_vals[(mkey, var)] = uniq[1] if len(uniq) > 1 else np.nan

    lines = [
        r"\multicolumn{16}{l}{\textit{By " + label + r" type}} \\",
        r"\midrule",
    ]

    for grp in order:
        ns = []
        cells = []
        for mkey, mlabel, drop_zero, digits in METRICS:
            source = metric_maps[mkey]
            for var, _ in VARIANTS:
                item = source.get((grp, var))
                if item is None:
                    cells.append("--")
                    continue
                mean, err, n_q = item
                ns.append(n_q)
                main = fmt_num(mean, digits, drop_zero)
                err_s = fmt_num(err, digits, drop_zero)
                if mkey in highlight_metrics:
                    if np.isfinite(best_vals[(mkey, var)]) and math.isclose(mean, best_vals[(mkey, var)], rel_tol=1e-9, abs_tol=1e-9):
                        main = r"\textbf{" + main + "}"
                    elif np.isfinite(second_vals[(mkey, var)]) and math.isclose(mean, second_vals[(mkey, var)], rel_tol=1e-9, abs_tol=1e-9):
                        main = r"\underline{" + main + "}"
                cells.append(main)
                # CI-heavy appendix version kept here for reference:
                # cells.append(main + r"{\scriptsize$\pm$" + err_s + "}")
        n_med = int(np.median(ns)) if ns else 0
        display = name_map.get(grp, grp) if name_map else grp
        lines.append(f"{display} (n={n_med}) " + " & " + " & ".join(cells) + r" \\")
    return lines


def main() -> None:
    participants, common_qids, _, _ = load_human_subset(
        ROOT, min_answers=MIN_ANSWERS_DEFAULT, translate=False, verbose=False
    )
    pair_cache = load_cleaned_pair_cache(ROOT, condition="inst_blind", include_yesno=True, verbose=False)
    hh = pair_cache[(pair_cache["pair_type"] == "HH") & (pair_cache["question_id"].isin(common_qids))].copy()
    human = pd.read_csv(ROOT / "analysis/session2/exports/responses_human.csv")
    human = human[human["question_id"].isin(common_qids)].copy()

    metric_maps = {
        "sbert_score": summarize_pair_metric(hh, "op", "sbert_score"),
        "chrf_score": summarize_pair_metric(hh, "op", "chrf_score"),
        "rouge1_score": summarize_pair_metric(hh, "op", "rouge1_score"),
        "jaccard_score": summarize_pair_metric(hh, "op", "jaccard_score"),
        "entropy": summarize_entropy(human, "op"),
    }
    metric_maps_ent = {
        "sbert_score": summarize_pair_metric(hh, "ent", "sbert_score"),
        "chrf_score": summarize_pair_metric(hh, "ent", "chrf_score"),
        "rouge1_score": summarize_pair_metric(hh, "ent", "rouge1_score"),
        "jaccard_score": summarize_pair_metric(hh, "ent", "jaccard_score"),
        "entropy": summarize_entropy(human, "ent"),
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Human--human agreement by group across metrics and variants, with answer entropy (bits) as a final column (113-question inst-blind subset). Values are mean question-level scores. Best and second-best values are highlighted within each metric/variant column for agreement metrics only. SBERT captures semantic grouping (Existence/Action/World Know.\ $>$ Comparison/Identity/Text Reading), while surface-form metrics (chrF, ROUGE-1, Jaccard) show a flatter profile; entropy confirms that response diversity is structured primarily by operation type, not entity type.}",
        r"\label{tab:hh_metric_robustness}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrrrrrr}",
        r"\toprule",
        r"Group & \multicolumn{3}{c}{SBERT} & \multicolumn{3}{c}{chrF} & \multicolumn{3}{c}{ROUGE-1} & \multicolumn{3}{c}{Jaccard} & \multicolumn{3}{c}{Entropy (bits)} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13} \cmidrule(lr){14-16}",
        r" & Orig. & Wkr. & Pron. & Orig. & Wkr. & Pron. & Orig. & Wkr. & Pron. & Orig. & Wkr. & Pron. & Orig. & Wkr. & Pron. \\",
        r"\midrule",
    ]
    lines.extend(build_section_rows(OP_ORDER, "operation", metric_maps, {"sbert_score", "chrf_score", "rouge1_score", "jaccard_score"}, OP_FULL_NAMES))
    lines.append(r"\addlinespace[4pt]")
    lines.extend(build_section_rows(ENT_ORDER, "entity", metric_maps_ent, {"sbert_score", "chrf_score", "rouge1_score", "jaccard_score"}, ENT_FULL_NAMES))
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
