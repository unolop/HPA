"""
degradation_by_ent_op.py

Facet control-variant degradation ladders by entity or operation subgroup,
using the same small-multiple layout style as figures/r_by_ent_op/.

Outputs per condition (inst_blind / blind):
  - degradation_by_ent_sbert_<condition>.png
  - degradation_by_op_sbert_<condition>.png
  - degradation_by_ent_accuracy_<condition>.png
  - degradation_by_op_accuracy_<condition>.png
  - grouped_* variants using collapsed entity/op mappings

Run from repo root:
  conda run -n zero python figures/degradation_by_ent_op/degradation_by_ent_op.py
  conda run -n zero python figures/degradation_by_ent_op/degradation_by_ent_op.py --condition both
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from config import MIN_ANSWERS_DEFAULT, MODELS_7B, MODEL_GROUP, COLLAPSED_VARIANT_QIDS
from helpers import load_cleaned_pair_cache, load_human_subset, read_export
from utils.constants import MODEL_SIZE_B

FIGURES_ROOT = ROOT / "figures"
LATEX_FIGURES_ROOT = ROOT / "latex/AAAI2026/LaTeX/figures"

ENT_MIN_Q = 1
OP_MIN_Q = 7

MODEL_GROUP_COLORS = {
    "VLM": "#9E4A3A",                    # muted brick
    "VLM backbone decoder": "#C0843D",   # muted ochre
    "standalone LLM": "#5E8C61",         # muted sage
    "standalone LLM (think)": "#5B84B8", # muted steel blue
}
MODEL_GROUP_LABELS = {
    "VLM": "VLM",
    "VLM backbone decoder": "Backbone",
    "standalone LLM": "SA-LLM",
    "standalone LLM (think)": "Think",
}
GROUP_LS = {
    "VLM": "-",
    "VLM backbone decoder": "-",
    "standalone LLM": "-",
    "standalone LLM (think)": "-",
}
GROUP_ORDER = ["VLM", "VLM backbone decoder", "standalone LLM", "standalone LLM (think)"]
FACET_GRID = [
    ["Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna", "InternVL-8B"],
    ["Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)", "InternVL-8B (LM)"],
    ["Qwen3-8B", "Qwen3-8B (think)", "Mistral-7B", "Vicuna-7B", "Qwen2.5-7B-Instruct"],
]
ROW_NAMES = ["VLM", "Backbone\nDecoder", "Standalone\nLLM"]
PANEL_LABELS = {
    "Qwen3-VL-8B": "Qwen3-VL-8B",
    "LLaVA-1.5-7B": "LLaVA-1.5-7B",
    "LLaVA-Mistral": "LLaVA-Mistral",
    "LLaVA-Vicuna": "LLaVA-Vicuna",
    "InternVL-8B": "InternVL-8B",
    "Qwen3-VL-8B (LM)": "Q3-VL-8B",
    "LLaVA-1.5 (LM)": "LLaVA-1.5",
    "LLaVA-Mistral (LM)": "Mistral",
    "LLaVA-Vicuna (LM)": "Vicuna",
    "InternVL-8B (LM)": "InternVL",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-8B (think)": "Qwen3-8B (T)",
    "Mistral-7B": "Mistral-7B",
    "Vicuna-7B": "Vicuna-7B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
}
FAMILY_SPECS = {
    "qwen3": [
        ("Qwen3-VL-8B", "VLM"),
        ("Qwen3-VL-8B (LM)", "Backbone"),
        ("Qwen3-8B", "SA-LLM"),
        ("Qwen3-8B (think)", "Think"),
    ],
    "internvl": [
        ("InternVL-8B", "VLM"),
        ("InternVL-8B (LM)", "Backbone"),
        ("Qwen3-8B", "SA-LLM"),
    ],
    "llava_15": [
        ("LLaVA-1.5-7B", "VLM"),
        ("LLaVA-1.5 (LM)", "Backbone"),
        ("Vicuna-7B", "SA-LLM"),
    ],
    "llava_mistral": [
        ("LLaVA-Mistral", "VLM"),
        ("LLaVA-Mistral (LM)", "Backbone"),
        ("Mistral-7B", "SA-LLM"),
    ],
    "llava_vicuna": [
        ("LLaVA-Vicuna", "VLM"),
        ("LLaVA-Vicuna (LM)", "Backbone"),
        ("Vicuna-7B", "SA-LLM"),
    ],
}
FAMILY_COLORS = {
    "VLM": MODEL_GROUP_COLORS["VLM"],
    "Backbone": MODEL_GROUP_COLORS["VLM backbone decoder"],
    "SA-LLM": MODEL_GROUP_COLORS["standalone LLM"],
    "Think": MODEL_GROUP_COLORS["standalone LLM (think)"],
}
VARIANT_COLORS = {"C": "#274c77", "B": "#6b8fb3", "A": "#c8d7ea"}

VARIANTS = ["C", "B", "A"]
VAR_X = [0, 1, 2]
VAR_LABELS = ["Orig", "Weak", "Pron"]
EXCLUDED = {"Mistral-7B", "Phi-3.5-mini"}
BOOT_N = 1000

CT_TO_VARIANT = {"question": "C", "weaker_object": "B", "pronominalized": "A"}
EOS_TOKENS = {"<|im_end|>", "</s>", "<eos>", "<|endoftext|>", "<|end|>"}

VLM_DIR_TO_MODEL = {
    "InternVL3_5-8B": "InternVL-8B",
    "llava-1.5-7b-hf": "LLaVA-1.5-7B",
    "llava-v1.6-mistral-7b-hf": "LLaVA-Mistral",
    "llava-v1.6-vicuna-7b-hf": "LLaVA-Vicuna",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
}
LM_DECODER_DIR_TO_MODEL = {
    "InternVL3_5-8B": "InternVL-8B (LM)",
    "llava-1.5-7b-hf": "LLaVA-1.5 (LM)",
    "llava-v1.6-mistral-7b-hf": "LLaVA-Mistral (LM)",
    "llava-v1.6-vicuna-7b-hf": "LLaVA-Vicuna (LM)",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B (LM)",
}
BACKBONE_DIR_TO_MODEL = {
    "Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-8B_think": "Qwen3-8B (think)",
    "vicuna-7b-v1.5": "Vicuna-7B",
}
LOGIT_SOURCES = [
    (ROOT / "evaluation/logits/vlm/pretrained", VLM_DIR_TO_MODEL),
    (ROOT / "evaluation/logits/lm_decoder/pretrained", LM_DECODER_DIR_TO_MODEL),
    (ROOT / "evaluation/logits/backbone/pretrained", BACKBONE_DIR_TO_MODEL),
]

ENT_GROUP_MAP = {
    "object": "object",
    "person": "person",
    "animal": "animal",
    "food": "food",
    "vehicle": "visual-other",
    "product": "visual-other",
    "place": "visual-other",
    "text": "text",
    "other": "text",
}
ENT_GROUP_ORDER = ["object", "person", "animal", "food", "visual-other", "text"]

OP_GROUP_MAP = {
    "attr": "attr",
    "count": "count",
    "ident": "ident",
    "spat": "spat",
    "exist": "exist",
    "act": "act",
    "know": "other",
    "text": "other",
    "temp": "other",
    "comp": "other",
    "other": "other",
    "cause": "other",
}
OP_GROUP_ORDER = ["attr", "count", "ident", "spat", "exist", "act", "other"]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
    }
)


def _base_dir(condition: str) -> Path:
    return FIGURES_ROOT / f"degradation_by_ent_op_7b_{condition}"


def _latex_base_dir(condition: str) -> Path:
    return LATEX_FIGURES_ROOT / f"degradation_by_ent_op_7b_{condition}"


def _metric_dir(condition: str, metric: str) -> Path:
    out = _base_dir(condition) / metric
    out.mkdir(parents=True, exist_ok=True)
    return out


def _latex_metric_dir(condition: str, metric: str) -> Path:
    out = _latex_base_dir(condition) / metric
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_metric_file(condition: str, metric: str, fname: str, fig) -> None:
    out = _metric_dir(condition, metric) / fname
    latex_out = _latex_metric_dir(condition, metric) / fname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, latex_out)
    print(f"Saved: {out}")


def _copy_metric_file(condition: str, metric: str, src: Path, fname: str) -> None:
    out = _metric_dir(condition, metric) / fname
    latex_out = _latex_metric_dir(condition, metric) / fname
    shutil.copy(src, out)
    shutil.copy(src, latex_out)
    print(f"Saved CSV: {out}")


def _build_groups(qid_meta: pd.DataFrame, col: str, min_q: int, other_label: str) -> dict[str, list[str]]:
    counts = qid_meta[col].value_counts()
    groups: dict[str, list[str]] = {}
    other_vals: list[str] = []
    for val, n in counts.items():
        if n >= min_q:
            groups[val] = [val]
        else:
            other_vals.append(val)
    if other_vals:
        groups[other_label] = other_vals
    return groups


def _build_mapped_groups(qid_meta: pd.DataFrame, col: str, mapping: dict, order: list[str]) -> dict[str, list[str]]:
    mapped = qid_meta[[col]].copy()
    mapped["grouped"] = mapped[col].map(mapping)
    groups: dict[str, list[str]] = {}
    for name in order:
        vals = mapped.loc[mapped["grouped"] == name, col].dropna().unique().tolist()
        if vals:
            groups[name] = vals
    return groups


def _human_sbert(pair_cache_h: pd.DataFrame, variant: str) -> pd.Series:
    hh = pair_cache_h[(pair_cache_h["pair_type"] == "HH") & (pair_cache_h["variant"] == variant)]
    return hh.groupby("question_id")["sbert_score"].mean()


def _human_acc(human_df_full: pd.DataFrame, common_qids: list[int], variant: str) -> pd.Series:
    hv = human_df_full[
        (human_df_full["variant"] == variant)
        & (human_df_full["question_id"].isin(common_qids))
    ]
    return hv.groupby("question_id")["accuracy"].mean()


def _model_sbert(pair_cache_h: pd.DataFrame, model: str, variant: str) -> pd.Series:
    hm = pair_cache_h[
        (pair_cache_h["pair_type"] == "HM")
        & (pair_cache_h["variant"] == variant)
        & (pair_cache_h["subject_2"] == model)
    ]
    return hm.groupby("question_id")["sbert_score"].mean()


def _model_acc(raw_acc_h: pd.DataFrame, model: str, variant: str) -> pd.Series:
    mv = raw_acc_h[(raw_acc_h["model"] == model) & (raw_acc_h["variant"] == variant)]
    return mv.groupby("question_id")["accuracy"].mean()


def _pearson(x: pd.Series, y: pd.Series) -> float:
    common = list(set(x.index) & set(y.index))
    if len(common) < 3:
        return np.nan
    xv = x.loc[common].astype(float)
    yv = y.loc[common].astype(float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if len(xv) < 3 or xv.nunique() < 2 or yv.nunique() < 2:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def _fisher_r_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(r) or n <= 3:
        return (np.nan, np.nan)
    # Avoid infinite z at |r|=1.
    r_clip = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r_clip)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = 1.959963984540054
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return float(lo), float(hi)


def _fisher_pool(rs: list[float]) -> float:
    vals = [float(r) for r in rs if np.isfinite(r) and abs(r) < 1.0]
    if not vals:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(vals))))


def _bootstrap_mean_ci(vals: np.ndarray, n_boot: int = BOOT_N) -> tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _human_loo_subgroup_rows(
    pair_cache_h: pd.DataFrame,
    qid_meta: pd.DataFrame,
    groups: dict[str, list[str]],
    group_col: str,
) -> list[dict]:
    hh = pair_cache_h[pair_cache_h["pair_type"] == "HH"].copy()
    participants = sorted(set(hh["subject_1"]).union(hh["subject_2"]))
    rows: list[dict] = []

    for subgroup, subgroup_vals in groups.items():
        subgroup_qids = set(qid_meta[qid_meta[group_col].isin(subgroup_vals)].index.tolist())
        hh_sub = hh[hh["question_id"].isin(subgroup_qids)].copy()
        if hh_sub.empty:
            continue

        per_participant = []
        for pid in participants:
            lhs = hh_sub[(hh_sub["subject_1"] == pid) | (hh_sub["subject_2"] == pid)]
            rhs = hh_sub[(hh_sub["subject_1"] != pid) & (hh_sub["subject_2"] != pid)]
            if lhs.empty or rhs.empty:
                continue

            lhs_q = lhs.groupby(["question_id", "variant"], as_index=False).agg(loo_hm=("sbert_score", "mean"))
            rhs_q = rhs.groupby(["question_id", "variant"], as_index=False).agg(loo_hh=("sbert_score", "mean"))
            merged = lhs_q.merge(rhs_q, on=["question_id", "variant"], how="inner")
            if merged.empty:
                continue

            row = {"participant": pid, "n": 0}
            ns = []
            for v in VARIANTS:
                vs = merged[merged["variant"] == v]
                if len(vs) < 3:
                    row[v] = np.nan
                    continue
                row[v] = _pearson(vs.set_index("question_id")["loo_hh"], vs.set_index("question_id")["loo_hm"])
                ns.append(len(vs))
            if any(np.isfinite(row[v]) for v in VARIANTS):
                row["n"] = int(np.median(ns)) if ns else 0
                per_participant.append(row)

        if not per_participant:
            continue

        part_df = pd.DataFrame(per_participant)
        for v in VARIANTS:
            pooled = _fisher_pool(part_df[v].dropna().tolist())
            rows.append(
                {
                    "model": "Human",
                    "grp": "Human",
                    "subgroup": subgroup,
                    "variant": v,
                    "value": pooled,
                    "n": int(part_df["n"].median()) if "n" in part_df and len(part_df) else 0,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                }
            )
    return rows


def load_confidence_frames(condition: str, common_qids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sem_df = pd.read_json(ROOT / "dataset/vqa/vqa1k_semantics.jsonl", lines=True)[["question_id", "ent", "op"]]

    human_files = sorted(glob.glob(str(ROOT / "evaluation/humans/by_participant/*.json")))
    human_rows = []
    for fp in human_files:
        with open(fp) as f:
            data = json.load(f)
        pid = data.get("code", Path(fp).stem)
        for ans in data.get("answers", []):
            qid = ans["question_id"]
            if qid not in common_qids:
                continue
            raw_conf = ans.get("confidence", 3)
            human_rows.append(
                {
                    "question_id": qid,
                    "participant": pid,
                    "variant": ans.get("variant", "C"),
                    "confidence": (raw_conf - 1) / 4.0,
                }
            )
    human_df = pd.DataFrame(human_rows).merge(sem_df, on="question_id", how="left")

    model_rows = []
    keep_models = set(MODELS_7B)
    for base_dir, mapping in LOGIT_SOURCES:
        for dir_name, display in mapping.items():
            if display not in keep_models:
                continue
            fpath = base_dir / dir_name / f"vqa_1k_control_{condition}.jsonl"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    qid = ex["question_id"]
                    if qid not in common_qids:
                        continue
                    logits = ex.get("generated_logits", {})
                    for ct, variant in CT_TO_VARIANT.items():
                        logit_data = logits.get(ct)
                        if not logit_data:
                            continue
                        probs = [
                            np.exp(t["logprob"])
                            for t in logit_data.get("content", [])
                            if t.get("token") not in EOS_TOKENS
                        ]
                        if not probs:
                            continue
                        model_rows.append(
                            {
                                "question_id": qid,
                                "model": display,
                                "variant": variant,
                                "confidence": float(np.mean(probs)),
                            }
                        )
    model_df = pd.DataFrame(model_rows).merge(sem_df, on="question_id", how="left")
    return human_df, model_df


def compute_ladder_df(
    pair_cache_h: pd.DataFrame,
    raw_acc_h: pd.DataFrame,
    human_df_full: pd.DataFrame,
    common_qids: list[int],
    qid_meta: pd.DataFrame,
    groups: dict[str, list[str]],
    group_col: str,
    models: list[str],
    *,
    metric: str,
    human_conf_h: pd.DataFrame | None = None,
    model_conf_h: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    if metric == "sbert":
        rows.extend(_human_loo_subgroup_rows(pair_cache_h, qid_meta, groups, group_col))
    for v in VARIANTS:
        if metric == "sbert":
            human_score = _human_sbert(pair_cache_h, v)
        elif metric == "accuracy":
            human_score = _human_acc(human_df_full, common_qids, v)
        elif metric == "confidence":
            assert human_conf_h is not None and model_conf_h is not None
            human_score = human_conf_h[human_conf_h["variant"] == v].groupby("question_id")["confidence"].mean()
        else:
            raise ValueError(metric)
        for subgroup, subgroup_vals in groups.items():
            subgroup_qids = qid_meta[qid_meta[group_col].isin(subgroup_vals)].index
            h_common = list(set(human_score.index) & set(subgroup_qids))
            if h_common:
                h_vals = human_score.loc[h_common].astype(float)
                h_vals = h_vals[np.isfinite(h_vals)]
                if len(h_vals) > 0:
                    h_mean, h_lo, h_hi = _bootstrap_mean_ci(h_vals.values)
                    rows.append(
                        {
                            "model": "Human",
                            "grp": "Human",
                            "subgroup": subgroup,
                            "variant": v,
                            "value": float(h_mean),
                            "n": int(len(h_vals)),
                            "ci_lo": float(h_lo) if np.isfinite(h_lo) else np.nan,
                            "ci_hi": float(h_hi) if np.isfinite(h_hi) else np.nan,
                        }
                    )

            for model in models:
                if metric == "sbert":
                    model_score = _model_sbert(pair_cache_h, model, v)
                elif metric == "accuracy":
                    model_score = _model_acc(raw_acc_h, model, v)
                elif metric == "confidence":
                    model_score = model_conf_h[(model_conf_h["model"] == model) & (model_conf_h["variant"] == v)].groupby("question_id")["confidence"].mean()
                else:
                    raise ValueError(metric)
                common = list(set(model_score.index) & set(subgroup_qids))
                if len(common) < 3:
                    continue
                vals = model_score.loc[common].astype(float)
                vals = vals[np.isfinite(vals)]
                if len(vals) < 3:
                    continue
                m_mean, m_lo, m_hi = _bootstrap_mean_ci(vals.values)
                rows.append(
                    {
                        "model": model,
                        "grp": MODEL_GROUP.get(model, "standalone LLM"),
                        "subgroup": subgroup,
                        "variant": v,
                        "value": float(m_mean),
                        "n": int(len(vals)),
                        "ci_lo": float(m_lo) if np.isfinite(m_lo) else np.nan,
                        "ci_hi": float(m_hi) if np.isfinite(m_hi) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def compute_r_ladder_df(
    pair_cache_h: pd.DataFrame,
    raw_acc_h: pd.DataFrame,
    human_df_full: pd.DataFrame,
    common_qids: list[int],
    qid_meta: pd.DataFrame,
    groups: dict[str, list[str]],
    group_col: str,
    models: list[str],
    *,
    metric: str,
    human_conf_h: pd.DataFrame | None = None,
    model_conf_h: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for v in VARIANTS:
        if metric == "sbert":
            human_score = _human_sbert(pair_cache_h, v)
        elif metric == "accuracy":
            human_score = _human_acc(human_df_full, common_qids, v)
        elif metric == "confidence":
            assert human_conf_h is not None and model_conf_h is not None
            human_score = human_conf_h[human_conf_h["variant"] == v].groupby("question_id")["confidence"].mean()
        else:
            raise ValueError(metric)

        for subgroup, subgroup_vals in groups.items():
            subgroup_qids = qid_meta[qid_meta[group_col].isin(subgroup_vals)].index
            for model in models:
                if metric == "sbert":
                    model_score = _model_sbert(pair_cache_h, model, v)
                elif metric == "accuracy":
                    model_score = _model_acc(raw_acc_h, model, v)
                elif metric == "confidence":
                    model_score = model_conf_h[(model_conf_h["model"] == model) & (model_conf_h["variant"] == v)].groupby("question_id")["confidence"].mean()
                else:
                    raise ValueError(metric)
                common = list(set(human_score.index) & set(model_score.index) & set(subgroup_qids))
                if len(common) < 3:
                    continue
                r = _pearson(human_score.loc[common], model_score.loc[common])
                if not np.isfinite(r):
                    continue
                ci_lo, ci_hi = _fisher_r_ci(r, len(common))
                rows.append(
                    {
                        "model": model,
                        "grp": MODEL_GROUP.get(model, "standalone LLM"),
                        "subgroup": subgroup,
                        "variant": v,
                        "value": float(r),
                        "n": int(len(common)),
                        "ci_lo": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
                        "ci_hi": float(ci_hi) if np.isfinite(ci_hi) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _axis_limits(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    if df.empty:
        return (0.0, 1.0)
    vals = df["value"].dropna().astype(float)
    lo = float(vals.min())
    hi = float(vals.max())
    span = hi - lo
    pad = max(0.015, span * 0.12)
    if metric == "accuracy":
        return max(0.0, lo - pad), min(1.0, hi + pad)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def _group_mean_and_ci(gsub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for v in VARIANTS:
        vsub = gsub[gsub["variant"] == v]
        model_vals = vsub.groupby("model")["value"].mean().dropna().astype(float).values
        if len(model_vals) == 0:
            means.append(np.nan)
            lows.append(np.nan)
            highs.append(np.nan)
            continue
        mean, lo, hi = _bootstrap_mean_ci(model_vals)
        means.append(mean)
        lows.append(lo)
        highs.append(hi)
    return np.array(means, dtype=float), np.array(lows, dtype=float), np.array(highs, dtype=float)


def plot_ladders(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    fname: str,
    metric_label: str,
    *,
    condition: str,
    metric: str,
):
    groups = {k: v for k, v in groups.items() if k in set(df["subgroup"].unique())}
    n_panels = len(groups)
    ncols = n_panels
    nrows = 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 1.7, 2.35),
        sharey=True,
    )
    axes = np.array(axes).flatten()
    ymin, ymax = _axis_limits(df, "accuracy" if "Accuracy" in metric_label else "sbert")

    for ax_idx, (subgroup, _) in enumerate(groups.items()):
        ax = axes[ax_idx]
        sub = df[df["subgroup"] == subgroup]

        hsub = sub[sub["model"] == "Human"]
        if not hsub.empty:
            ys_h = []
            lo_h = []
            hi_h = []
            for v in VARIANTS:
                vsub = hsub[hsub["variant"] == v]
                ys_h.append(float(vsub["value"].mean()) if not vsub.empty else np.nan)
                lo_h.append(float(vsub["ci_lo"].mean()) if "ci_lo" in vsub and not vsub.empty else np.nan)
                hi_h.append(float(vsub["ci_hi"].mean()) if "ci_hi" in vsub and not vsub.empty else np.nan)
            ys_h = np.asarray(ys_h, dtype=float)
            lo_h = np.asarray(lo_h, dtype=float)
            hi_h = np.asarray(hi_h, dtype=float)
            valid_h = np.isfinite(ys_h)
            if np.any(valid_h):
                ax.plot(
                    np.asarray(VAR_X)[valid_h],
                    ys_h[valid_h],
                    color="#424242",
                    lw=1.6,
                    ls="--",
                    marker="D",
                    markersize=4.2,
                    markerfacecolor="white",
                    markeredgecolor="#424242",
                    zorder=4,
                )
                ci_valid_h = valid_h & np.isfinite(lo_h) & np.isfinite(hi_h)
                if np.any(ci_valid_h):
                    x_ci = np.asarray(VAR_X)[ci_valid_h]
                    y_ci = ys_h[ci_valid_h]
                    yerr = np.vstack([
                        np.maximum(0, y_ci - lo_h[ci_valid_h]),
                        np.maximum(0, hi_h[ci_valid_h] - y_ci),
                    ])
                    ax.errorbar(
                        x_ci, y_ci, yerr=yerr, color="#424242", lw=1.0, ls="none",
                        capsize=2.3, capthick=0.9, elinewidth=0.9, zorder=3
                    )

        for g in GROUP_ORDER:
            gsub = sub[sub["grp"] == g]
            if gsub.empty:
                continue
            color = MODEL_GROUP_COLORS[g]
            ls = GROUP_LS[g]

            for model, mdf in gsub.groupby("model"):
                ys_m = [mdf[mdf["variant"] == v]["value"].mean() for v in VARIANTS]
                if all(np.isnan(y) for y in ys_m):
                    continue
                ax.plot(VAR_X, ys_m, color=color, lw=0.7, ls=ls, alpha=0.22, zorder=1)

            ys_mean, ys_lo, ys_hi = _group_mean_and_ci(gsub)
            valid = np.isfinite(ys_mean) & np.isfinite(ys_lo) & np.isfinite(ys_hi)
            if np.any(valid):
                ax.fill_between(
                    np.array(VAR_X)[valid],
                    ys_lo[valid],
                    ys_hi[valid],
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                    zorder=2,
                )
            ax.plot(
                VAR_X,
                ys_mean,
                color=color,
                lw=1.8,
                ls=ls,
                marker="o",
                markersize=4.8,
                zorder=3,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

        ax.set_title(subgroup, fontsize=11, fontweight="bold")
        ax.set_xticks(VAR_X)
        ax.set_xticklabels(VAR_LABELS, fontsize=9)
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(ymin, ymax)
        if ax_idx % ncols == 0:
            ax.set_ylabel(metric_label, fontsize=9)

        ns = sub[sub["model"] == "Human"]["n"].unique()
        if len(ns) > 0:
            ax.annotate(
                f"n≈{int(np.median(ns))}q",
                xy=(0.97, 0.04),
                xycoords="axes fraction",
                fontsize=7.5,
                ha="right",
                color="#666",
            )

    for ax_idx in range(n_panels, len(axes)):
        axes[ax_idx].set_visible(False)

    handles = [
        mlines.Line2D([], [], color="#424242", lw=1.6, ls="--", marker="D", markersize=4.2, label="Human")
    ] + [
        mlines.Line2D(
            [],
            [],
            color=MODEL_GROUP_COLORS[g],
            lw=1.8,
            ls=GROUP_LS[g],
            marker="o",
            markersize=5,
            label=MODEL_GROUP_LABELS[g],
        )
        for g in GROUP_ORDER
    ]
    fig.legend(
        handles=handles,
        fontsize=7,
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    _save_metric_file(condition, metric, fname, fig)
    plt.close(fig)


def _plot_row(fig, gs, row_idx: int, ncols: int, df: pd.DataFrame, groups: dict, metric_label: str, row_label: str,
             ymin: float | None = None, ymax: float | None = None):
    """Plot one row of the combined figure (op or ent)."""
    metric = "accuracy" if "Accuracy" in metric_label else "sbert"
    if ymin is None or ymax is None:
        ymin, ymax = _axis_limits(df, metric)
    for col_idx, (subgroup, _) in enumerate(groups.items()):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        sub = df[df["subgroup"] == subgroup]

        hsub = sub[sub["model"] == "Human"]
        if not hsub.empty:
            ys_h = []
            lo_h = []
            hi_h = []
            for v in VARIANTS:
                vsub = hsub[hsub["variant"] == v]
                ys_h.append(float(vsub["value"].mean()) if not vsub.empty else np.nan)
                lo_h.append(float(vsub["ci_lo"].mean()) if "ci_lo" in vsub and not vsub.empty else np.nan)
                hi_h.append(float(vsub["ci_hi"].mean()) if "ci_hi" in vsub and not vsub.empty else np.nan)
            ys_h = np.asarray(ys_h, dtype=float)
            lo_h = np.asarray(lo_h, dtype=float)
            hi_h = np.asarray(hi_h, dtype=float)
            valid_h = np.isfinite(ys_h)
            if np.any(valid_h):
                ax.plot(np.asarray(VAR_X)[valid_h], ys_h[valid_h], color="#424242", lw=1.6, ls="--", marker="D",
                        markersize=4.2, markerfacecolor="white", markeredgecolor="#424242", zorder=4)
                ci_valid_h = valid_h & np.isfinite(lo_h) & np.isfinite(hi_h)
                if np.any(ci_valid_h):
                    x_ci = np.asarray(VAR_X)[ci_valid_h]
                    y_ci = ys_h[ci_valid_h]
                    yerr = np.vstack([np.maximum(0, y_ci - lo_h[ci_valid_h]), np.maximum(0, hi_h[ci_valid_h] - y_ci)])
                    ax.errorbar(x_ci, y_ci, yerr=yerr, color="#424242", lw=1.0, ls="none",
                                capsize=2.3, capthick=0.9, elinewidth=0.9, zorder=3)

        for g in GROUP_ORDER:
            gsub = sub[sub["grp"] == g]
            if gsub.empty:
                continue
            color = MODEL_GROUP_COLORS[g]
            ls = GROUP_LS[g]
            for model, mdf in gsub.groupby("model"):
                ys_m = [mdf[mdf["variant"] == v]["value"].mean() for v in VARIANTS]
                if all(np.isnan(y) for y in ys_m):
                    continue
                ax.plot(VAR_X, ys_m, color=color, lw=0.7, ls=ls, alpha=0.22, zorder=1)
            ys_mean, ys_lo, ys_hi = _group_mean_and_ci(gsub)
            valid = np.isfinite(ys_mean) & np.isfinite(ys_lo) & np.isfinite(ys_hi)
            if np.any(valid):
                ax.fill_between(
                    np.array(VAR_X)[valid],
                    ys_lo[valid],
                    ys_hi[valid],
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                    zorder=2,
                )
            ax.plot(
                VAR_X,
                ys_mean,
                color=color,
                lw=1.8,
                ls=ls,
                marker="o",
                markersize=4.8,
                zorder=3,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

        ax.set_title(subgroup, fontsize=8.5, fontweight="bold")
        ax.set_xticks(VAR_X)
        ax.set_xticklabels(VAR_LABELS, fontsize=7)
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="y", labelsize=7)
        if col_idx == 0:
            ax.set_ylabel(metric_label, fontsize=7.5)
            ax.annotate(row_label, xy=(-0.45, 0.5), xycoords="axes fraction",
                        fontsize=8, fontweight="bold", rotation=90, va="center", ha="center")
        else:
            ax.set_yticklabels([])

        ns = sub[sub["model"] == "Human"]["n"].unique()
        if len(ns) > 0:
            ax.annotate(f"n≈{int(np.median(ns))}q", xy=(0.97, 0.04),
                        xycoords="axes fraction", fontsize=6, ha="right", color="#666")

    for col_idx in range(len(groups), ncols):
        fig.add_subplot(gs[row_idx, col_idx]).set_visible(False)


def plot_combined(
    df_op: pd.DataFrame,
    df_ent: pd.DataFrame,
    op_groups: dict,
    ent_groups: dict,
    fname: str,
    metric_label: str,
    *,
    condition: str,
    metric: str,
):
    """Generate a combined 2-row figure: op (top) + ent (bottom)."""
    op_groups = {k: v for k, v in op_groups.items() if k in set(df_op["subgroup"].unique())}
    ent_groups = {k: v for k, v in ent_groups.items() if k in set(df_ent["subgroup"].unique())}
    ncols = max(len(op_groups), len(ent_groups))

    # Shared y limits across both rows
    metric = "accuracy" if "Accuracy" in metric_label else "sbert"
    ymin, ymax = _axis_limits(pd.concat([df_op, df_ent]), metric)

    fig = plt.figure(figsize=(ncols * 1.7, 4.2))
    gs = fig.add_gridspec(2, ncols, hspace=0.38, wspace=0.22)

    _plot_row(fig, gs, 0, ncols, df_op, op_groups, metric_label, "Operation", ymin, ymax)
    _plot_row(fig, gs, 1, ncols, df_ent, ent_groups, metric_label, "Entity", ymin, ymax)

    handles = [
        mlines.Line2D([], [], color="#424242", lw=1.6, ls="--", marker="D",
                      markersize=4.2, label="Human")
    ] + [
        mlines.Line2D([], [], color=MODEL_GROUP_COLORS[g], lw=1.8, ls=GROUP_LS[g],
                      marker="o", markersize=5, label=MODEL_GROUP_LABELS[g])
        for g in GROUP_ORDER
    ]
    fig.legend(handles=handles, fontsize=7, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=True, handletextpad=0.4, columnspacing=0.8)
    fig.subplots_adjust(bottom=0.10)

    _save_metric_file(condition, metric, fname, fig)
    plt.close(fig)


def plot_r_combined(
    df_op: pd.DataFrame,
    df_ent: pd.DataFrame,
    op_groups: dict,
    ent_groups: dict,
    fname: str,
    metric_label: str,
    *,
    condition: str,
    metric: str,
):
    op_groups = {k: v for k, v in op_groups.items() if k in set(df_op["subgroup"].unique())}
    ent_groups = {k: v for k, v in ent_groups.items() if k in set(df_ent["subgroup"].unique())}
    ncols = max(len(op_groups), len(ent_groups))
    vals = pd.concat([df_op["value"], df_ent["value"]]).dropna().astype(float)
    ymin = min(-0.25, float(vals.min()) - 0.05) if len(vals) else -0.25
    ymax = max(0.95, float(vals.max()) + 0.05) if len(vals) else 0.95

    fig = plt.figure(figsize=(ncols * 1.7, 4.2))
    gs = fig.add_gridspec(2, ncols, hspace=0.38, wspace=0.22)

    for row_idx, (df, groups, row_label) in enumerate([(df_op, op_groups, "Operation"), (df_ent, ent_groups, "Entity")]):
        for col_idx, (subgroup, _) in enumerate(groups.items()):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            sub = df[df["subgroup"] == subgroup]
            hsub = sub[sub["model"] == "Human"]
            if not hsub.empty:
                ys_h = []
                for v in VARIANTS:
                    vsub = hsub[hsub["variant"] == v]
                    ys_h.append(float(vsub["value"].mean()) if not vsub.empty else np.nan)
                ys_h = np.asarray(ys_h, dtype=float)
                valid_h = np.isfinite(ys_h)
                if np.any(valid_h):
                    ax.plot(
                        np.asarray(VAR_X)[valid_h],
                        ys_h[valid_h],
                        color="#424242",
                        lw=1.6,
                        ls="--",
                        marker="D",
                        markersize=4.2,
                        markerfacecolor="white",
                        markeredgecolor="#424242",
                        zorder=4,
                    )
            for g in GROUP_ORDER:
                gsub = sub[sub["grp"] == g]
                if gsub.empty:
                    continue
                color = MODEL_GROUP_COLORS[g]
                for model, mdf in gsub.groupby("model"):
                    ys_m = [mdf[mdf["variant"] == v]["value"].mean() for v in VARIANTS]
                    if all(np.isnan(y) for y in ys_m):
                        continue
                    ax.plot(VAR_X, ys_m, color=color, lw=0.7, alpha=0.22, zorder=1)
                ys_mean, ys_lo, ys_hi = _group_mean_and_ci(gsub)
                valid = np.isfinite(ys_mean) & np.isfinite(ys_lo) & np.isfinite(ys_hi)
                if np.any(valid):
                    yerr = ys_mean[valid] - ys_lo[valid]
                    ax.errorbar(np.array(VAR_X)[valid], ys_mean[valid], yerr=yerr,
                                color=color, lw=1.8, marker="o", markersize=4.8,
                                capsize=2.5, capthick=1.0, elinewidth=1.0,
                                zorder=3, markerfacecolor=color,
                                markeredgecolor="white", markeredgewidth=0.5)
            ax.axhline(0.0, color="#9a9a9a", lw=0.8, ls=":")
            ax.set_title(subgroup, fontsize=8.5, fontweight="bold")
            ax.set_xticks(VAR_X)
            ax.set_xticklabels(VAR_LABELS, fontsize=7)
            ax.set_xlim(-0.3, 2.3)
            ax.set_ylim(ymin, ymax)
            ax.tick_params(axis="y", labelsize=7)
            if col_idx == 0:
                ax.set_ylabel(f"Mean Pearson r\n({metric_label})", fontsize=7.5)
                ax.annotate(row_label, xy=(-0.45, 0.5), xycoords="axes fraction",
                            fontsize=8, fontweight="bold", rotation=90, va="center", ha="center")
            else:
                ax.set_yticklabels([])
            ns = hsub["n"].unique() if not hsub.empty else sub["n"].unique()
            if len(ns) > 0:
                ax.annotate(f"n≈{int(np.median(ns))}q", xy=(0.97, 0.04),
                            xycoords="axes fraction", fontsize=6, ha="right", color="#666")

    handles = [
        mlines.Line2D([], [], color="#424242", lw=1.6, ls="--", marker="D",
                      markersize=4.2, label="Human"),
    ] + [
        mlines.Line2D([], [], color=MODEL_GROUP_COLORS[g], lw=1.8, marker="o", markersize=5, label=MODEL_GROUP_LABELS[g])
        for g in GROUP_ORDER
    ]
    fig.legend(handles=handles, fontsize=7, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=True, handletextpad=0.4, columnspacing=0.8)
    fig.subplots_adjust(bottom=0.10)

    _save_metric_file(condition, metric, fname, fig)
    plt.close(fig)


def plot_family_combined(
    df_op: pd.DataFrame,
    df_ent: pd.DataFrame,
    op_groups: dict,
    ent_groups: dict,
    family_key: str,
    *,
    condition: str,
    metric: str,
    metric_label: str,
    use_r: bool = False,
):
    op_groups = {k: v for k, v in op_groups.items() if k in set(df_op["subgroup"].unique())}
    ent_groups = {k: v for k, v in ent_groups.items() if k in set(df_ent["subgroup"].unique())}
    family = FAMILY_SPECS[family_key]
    rows = [(m, lab) for m, lab in family if m in set(df_op["model"]).union(set(df_ent["model"]))]
    ncols = max(len(op_groups), len(ent_groups))

    vals = pd.concat([df_op["value"], df_ent["value"]]).dropna().astype(float)
    if use_r:
        ymin = min(-0.25, float(vals.min()) - 0.05) if len(vals) else -0.25
        ymax = max(0.95, float(vals.max()) + 0.05) if len(vals) else 0.95
    else:
        metric_kind = "accuracy" if "Accuracy" in metric_label else "sbert"
        ymin, ymax = _axis_limits(pd.concat([df_op, df_ent]), metric_kind)

    fig = plt.figure(figsize=(ncols * 1.7, 3.9))
    gs = fig.add_gridspec(2, ncols, hspace=0.38, wspace=0.22)

    for row_idx, (df, groups, row_label) in enumerate([(df_op, op_groups, "Operation"), (df_ent, ent_groups, "Entity")]):
        for col_idx, (subgroup, _) in enumerate(groups.items()):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            sub = df[df["subgroup"] == subgroup]

            hsub = sub[sub["model"] == "Human"]
            if not hsub.empty and not use_r:
                ys_h = []
                lo_h = []
                hi_h = []
                for v in VARIANTS:
                    vsub = hsub[hsub["variant"] == v]
                    ys_h.append(float(vsub["value"].mean()) if not vsub.empty else np.nan)
                    lo_h.append(float(vsub["ci_lo"].mean()) if "ci_lo" in vsub and not vsub.empty else np.nan)
                    hi_h.append(float(vsub["ci_hi"].mean()) if "ci_hi" in vsub and not vsub.empty else np.nan)
                ys_h = np.asarray(ys_h, dtype=float)
                lo_h = np.asarray(lo_h, dtype=float)
                hi_h = np.asarray(hi_h, dtype=float)
                valid_h = np.isfinite(ys_h)
                if np.any(valid_h):
                    ax.plot(
                        np.asarray(VAR_X)[valid_h], ys_h[valid_h], color="#424242", lw=1.5, ls="--", marker="D",
                        markersize=4.0, markerfacecolor="white", markeredgecolor="#424242", zorder=4
                    )
                    ci_valid_h = valid_h & np.isfinite(lo_h) & np.isfinite(hi_h)
                    if np.any(ci_valid_h):
                        x_ci = np.asarray(VAR_X)[ci_valid_h]
                        y_ci = ys_h[ci_valid_h]
                        yerr = np.vstack([np.maximum(0, y_ci - lo_h[ci_valid_h]), np.maximum(0, hi_h[ci_valid_h] - y_ci)])
                        ax.errorbar(
                            x_ci, y_ci, yerr=yerr, color="#424242", lw=1.0, ls="none",
                            capsize=2.3, capthick=0.9, elinewidth=0.9, zorder=3
                        )

            for model, label in rows:
                msub = sub[sub["model"] == model]
                if msub.empty:
                    continue
                ys = [msub[msub["variant"] == v]["value"].mean() for v in VARIANTS]
                color = FAMILY_COLORS[label]
                if use_r:
                    ci_los = []
                    ci_his = []
                    for v in VARIANTS:
                        vsub = msub[msub["variant"] == v]
                        if vsub.empty:
                            ci_los.append(np.nan)
                            ci_his.append(np.nan)
                        else:
                            ci_los.append(float(vsub["ci_lo"].mean()) if "ci_lo" in vsub else np.nan)
                            ci_his.append(float(vsub["ci_hi"].mean()) if "ci_hi" in vsub else np.nan)
                    ys_arr = np.asarray(ys, dtype=float)
                    lo_arr = np.asarray(ci_los, dtype=float)
                    hi_arr = np.asarray(ci_his, dtype=float)
                    valid = np.isfinite(ys_arr)
                    if np.any(valid):
                        x_valid = np.asarray(VAR_X, dtype=float)[valid]
                        y_valid = ys_arr[valid]
                        ax.plot(
                            x_valid, y_valid, color=color, lw=1.9, marker="o", markersize=4.8, zorder=3,
                            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5, label=label
                        )
                        ci_valid = valid & np.isfinite(lo_arr) & np.isfinite(hi_arr)
                        if np.any(ci_valid):
                            x_ci = np.asarray(VAR_X, dtype=float)[ci_valid]
                            y_ci = ys_arr[ci_valid]
                            yerr = np.vstack([y_ci - lo_arr[ci_valid], hi_arr[ci_valid] - y_ci])
                            ax.errorbar(
                                x_ci, y_ci, yerr=yerr,
                                color=color, lw=1.2, capsize=2.5, capthick=0.9, elinewidth=0.9,
                                zorder=2, fmt="none", alpha=0.95
                            )
                else:
                    ci_los = []
                    ci_his = []
                    for v in VARIANTS:
                        vsub = msub[msub["variant"] == v]
                        ci_los.append(float(vsub["ci_lo"].mean()) if "ci_lo" in vsub and not vsub.empty else np.nan)
                        ci_his.append(float(vsub["ci_hi"].mean()) if "ci_hi" in vsub and not vsub.empty else np.nan)
                    ys_arr = np.asarray(ys, dtype=float)
                    lo_arr = np.asarray(ci_los, dtype=float)
                    hi_arr = np.asarray(ci_his, dtype=float)
                    valid = np.isfinite(ys_arr)
                    if np.any(valid):
                        x_valid = np.asarray(VAR_X, dtype=float)[valid]
                        y_valid = ys_arr[valid]
                        ax.plot(
                            x_valid, y_valid, color=color, lw=1.9, marker="o", markersize=4.8, zorder=3,
                            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5, label=label
                        )
                        ci_valid = valid & np.isfinite(lo_arr) & np.isfinite(hi_arr)
                        if np.any(ci_valid):
                            x_ci = np.asarray(VAR_X, dtype=float)[ci_valid]
                            y_ci = ys_arr[ci_valid]
                            yerr = np.vstack([y_ci - lo_arr[ci_valid], hi_arr[ci_valid] - y_ci])
                            ax.errorbar(
                                x_ci, y_ci, yerr=yerr,
                                color=color, lw=1.0, capsize=2.3, capthick=0.9, elinewidth=0.9,
                                zorder=2, fmt="none", alpha=0.95
                            )

            if use_r:
                ax.axhline(0.0, color="#9a9a9a", lw=0.8, ls=":")
            ax.set_title(subgroup, fontsize=8.5, fontweight="bold")
            ax.set_xticks(VAR_X)
            ax.set_xticklabels(VAR_LABELS, fontsize=7)
            ax.set_xlim(-0.3, 2.3)
            ax.set_ylim(ymin, ymax)
            ax.tick_params(axis="y", labelsize=7)
            if col_idx == 0:
                ylab = f"Mean Pearson r\n({metric_label})" if use_r else metric_label
                ax.set_ylabel(ylab, fontsize=7.5)
                ax.annotate(
                    row_label, xy=(-0.45, 0.5), xycoords="axes fraction",
                    fontsize=8, fontweight="bold", rotation=90, va="center", ha="center"
                )
            else:
                ax.set_yticklabels([])

            ns = sub["n"].unique() if use_r else sub[sub["model"] == "Human"]["n"].unique()
            if len(ns) > 0:
                ax.annotate(
                    f"n≈{int(np.median(ns))}q", xy=(0.97, 0.04), xycoords="axes fraction",
                    fontsize=6, ha="right", color="#666"
                )

    handles = []
    if not use_r:
        handles.append(
            mlines.Line2D([], [], color="#424242", lw=1.5, ls="--", marker="D",
                          markersize=4.0, markerfacecolor="white", markeredgecolor="#424242", label="Human")
        )
    for _, label in rows:
        handles.append(
            mlines.Line2D([], [], color=FAMILY_COLORS[label], lw=1.9, marker="o",
                          markersize=4.8, label=label)
        )
    fig.legend(
        handles=handles, fontsize=7, ncol=len(handles), loc="lower center",
        bbox_to_anchor=(0.5, 0.0), frameon=True, handletextpad=0.4, columnspacing=0.8
    )
    fig.subplots_adjust(bottom=0.10)

    fname = f"{'r_' if use_r else ''}{family_key}.png"
    _save_metric_file(condition, metric, fname, fig)
    plt.close(fig)


def export_table(df: pd.DataFrame, groups: dict[str, list[str]], fname_csv: str, *, condition: str, metric: str):
    subgroup_order = list(groups.keys())
    agg = (
        df[df["model"] != "Human"]
        .groupby(["grp", "subgroup", "variant"])["value"]
        .mean()
        .reset_index()
    )
    human = (
        df[df["model"] == "Human"][["subgroup", "variant", "value"]]
        .drop_duplicates()
        .assign(grp="Human")
    )
    agg = pd.concat([agg, human], ignore_index=True)
    pivot = agg.pivot_table(index="subgroup", columns=["variant", "grp"], values="value")
    pivot = pivot.reindex(subgroup_order)
    tmp = ROOT / ".tmp_degradation_by_ent_op.csv"
    pivot.to_csv(tmp, float_format="%.4f")
    _copy_metric_file(condition, metric, tmp, fname_csv)
    tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        default="inst_blind",
        choices=["inst_blind", "blind", "both"],
        help="Inference condition to use",
    )
    parser.add_argument(
        "--free_text_only",
        action="store_true",
        help="Skip yes/no extension and use the free-text subset only for a faster export.",
    )
    args = parser.parse_args()
    conditions = ["inst_blind", "blind"] if args.condition == "both" else [args.condition]

    print(f"Loading human data (min_answers={MIN_ANSWERS_DEFAULT})...")
    participants, common_qids, human_df_full, _ = load_human_subset(
        ROOT,
        min_answers=MIN_ANSWERS_DEFAULT,
        translate=False,
        verbose=False,
    )
    print(f"  {len(common_qids)} questions  |  {len(participants)} participants")

    models = [m for m in MODELS_7B if m in MODEL_SIZE_B and m not in EXCLUDED]

    for cond in conditions:
        print(f"\nLoading data for condition='{cond}'...")
        pair_cache = load_cleaned_pair_cache(
            ROOT,
            condition=cond,
            include_yesno=not args.free_text_only,
            verbose=False,
        )
        pair_cache_h = pair_cache[pair_cache["question_id"].isin(common_qids)].copy()
        raw_acc = read_export(ROOT, f"responses_model_{cond}.csv")
        raw_acc_h = raw_acc[raw_acc["question_id"].isin(common_qids)].copy()
        human_conf_h, model_conf_h = load_confidence_frames(cond, common_qids)

        if args.free_text_only:
            free_text_qids = sorted(pair_cache_h[pair_cache_h["pair_type"] == "HH"]["question_id"].unique())
            pair_cache_h = pair_cache_h[pair_cache_h["question_id"].isin(free_text_qids)].copy()
            raw_acc_h = raw_acc_h[raw_acc_h["question_id"].isin(free_text_qids)].copy()
            common_qids_cond = free_text_qids
        else:
            common_qids_cond = common_qids

        # Exclude questions with collapsed variant hierarchy from degradation analysis
        deg_qids = [q for q in common_qids_cond if q not in COLLAPSED_VARIANT_QIDS]
        pair_cache_deg = pair_cache_h[~pair_cache_h["question_id"].isin(COLLAPSED_VARIANT_QIDS)].copy()
        raw_acc_deg = raw_acc_h[~raw_acc_h["question_id"].isin(COLLAPSED_VARIANT_QIDS)].copy()
        print(f"  Degradation analysis: {len(deg_qids)}/{len(common_qids_cond)} questions "
              f"(excluded {len(COLLAPSED_VARIANT_QIDS)} collapsed-variant questions)")

        qid_meta = (
            pair_cache_deg[pair_cache_deg["pair_type"] == "HH"][["question_id", "ent", "op"]]
            .drop_duplicates("question_id")
            .set_index("question_id")
        )
        ent_groups_mapped = _build_mapped_groups(qid_meta, "ent", ENT_GROUP_MAP, ENT_GROUP_ORDER)
        op_groups_mapped = _build_mapped_groups(qid_meta, "op", OP_GROUP_MAP, OP_GROUP_ORDER)

        for metric, metric_label in [("sbert", "Mean HM SBERT"), ("accuracy", "Mean accuracy"), ("confidence", "Mean confidence")]:
            print(f"\n-- {metric_label} [{cond}] --")
            df_ent_grouped = compute_ladder_df(
                pair_cache_deg, raw_acc_deg, human_df_full, deg_qids, qid_meta, ent_groups_mapped, "ent", models, metric=metric,
                human_conf_h=human_conf_h, model_conf_h=model_conf_h,
            )
            df_op_grouped = compute_ladder_df(
                pair_cache_deg, raw_acc_deg, human_df_full, deg_qids, qid_meta, op_groups_mapped, "op", models, metric=metric,
                human_conf_h=human_conf_h, model_conf_h=model_conf_h,
            )
            rdf_ent_grouped = compute_r_ladder_df(
                pair_cache_deg, raw_acc_deg, human_df_full, deg_qids, qid_meta, ent_groups_mapped, "ent", models, metric=metric,
                human_conf_h=human_conf_h, model_conf_h=model_conf_h,
            )
            rdf_op_grouped = compute_r_ladder_df(
                pair_cache_deg, raw_acc_deg, human_df_full, deg_qids, qid_meta, op_groups_mapped, "op", models, metric=metric,
                human_conf_h=human_conf_h, model_conf_h=model_conf_h,
            )

            plot_ladders(df_ent_grouped, ent_groups_mapped, "ent.png", metric_label, condition=cond, metric=metric)
            plot_ladders(df_op_grouped, op_groups_mapped, "op.png", metric_label, condition=cond, metric=metric)
            plot_combined(df_op_grouped, df_ent_grouped, op_groups_mapped, ent_groups_mapped, "combined.png", metric_label, condition=cond, metric=metric)
            plot_r_combined(rdf_op_grouped, rdf_ent_grouped, op_groups_mapped, ent_groups_mapped, "r_combined.png", metric_label, condition=cond, metric=metric)

            for family_key in FAMILY_SPECS:
                plot_family_combined(
                    df_op_grouped, df_ent_grouped, op_groups_mapped, ent_groups_mapped,
                    family_key, condition=cond, metric=metric, metric_label=metric_label, use_r=False,
                )
                plot_family_combined(
                    rdf_op_grouped, rdf_ent_grouped, op_groups_mapped, ent_groups_mapped,
                    family_key, condition=cond, metric=metric, metric_label=metric_label, use_r=True,
                )

            export_table(df_ent_grouped, ent_groups_mapped, "table_ent.csv", condition=cond, metric=metric)
            export_table(df_op_grouped, op_groups_mapped, "table_op.csv", condition=cond, metric=metric)

    for cond in conditions:
        print(f"\nDone. All outputs -> {_base_dir(cond)}")


if __name__ == "__main__":
    main()
