"""
Per-group SBERT r strip plot (named model rows, unique per-model colors).

Y-axis: each model on its own named row, grouped into VLM / Backbone Decoder /
        Standalone LLM sections; human participants in a dedicated row at bottom.
X-axis: raw per-question Pearson r (HM vs HH SBERT).

Two outputs:
  alignment_profile_sbert_r_groups.png        — free-text questions only
  alignment_profile_sbert_r_groups_allq.png   — all questions (incl. categorical)

Human p5–p95 band: gray axvspan (not in legend per user request).
Human median: black axvline.
Model colors: unique per model (tab20 dark/light pairs for VLM/LM families).
Model markers: family-based (consistent with scatter figure).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.stats import pearsonr as _pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.utils.vqa import VQAAnswerMapper

# ── constants ─────────────────────────────────────────────────────────────────
EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "figures/alignment_profile"
OUT_DIR.mkdir(exist_ok=True)

MODEL_ORDER = {
    "VLM": [
        "InternVL-1B", "InternVL-2B", "InternVL-8B",
        "Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B", "Qwen3-VL-32B",
        "LLaVA-1.5-7B", "LLaVA-Mistral",
        "LLaVA-Vicuna", "LLaVA-Vicuna-13B",
        "Phi-3.5-mini",
    ],
    "Backbone Decoder": [
        "InternVL-1B (LM)", "InternVL-2B (LM)", "InternVL-8B (LM)",
        "Qwen3-VL-2B (LM)", "Qwen3-VL-4B (LM)", "Qwen3-VL-8B (LM)", "Qwen3-VL-32B (LM)",
        "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)",
        "LLaVA-Vicuna (LM)", "LLaVA-Vicuna-13B (LM)",
    ],
    "Standalone LLM": [
        "Qwen3-0.6B", "Qwen3-0.6B (think)",
        "Qwen3-1.7B", "Qwen3-1.7B (think)",
        "Qwen3-4B",   "Qwen3-4B (think)",
        "Qwen3-8B",   "Qwen3-8B (think)",
        "Qwen3-32B",  "Qwen3-32B (think)",
        "Qwen2.5-7B-Instruct", "Mistral-7B",
        "Vicuna-7B", "Vicuna-13B",
    ],
}
ALL_MODELS = [m for grp in MODEL_ORDER.values() for m in grp]

# One color per model family — same across VLM / LM / Standalone rows
_FAMILY_COLOR = {
    "InternVL":   "#E67E22",  # orange
    "Qwen3":      "#1F77B4",  # blue  (Qwen3-VL + Qwen3 standalone)
    "LLaVA-1.5":  "#2CA02C",  # green
    "Mistral":    "#D62728",  # red   (LLaVA-Mistral + Mistral-7B)
    "Vicuna":     "#9467BD",  # purple
    "Qwen2.5":    "#17BECF",  # teal
    "Phi":        "#8C564B",  # brown
}

def _fam(m: str) -> str:
    if m.startswith("InternVL"):      return "InternVL"
    if m.startswith("Qwen3-VL") or m.startswith("Qwen3"):  return "Qwen3"
    if m.startswith("LLaVA-1.5"):     return "LLaVA-1.5"
    if "Mistral" in m:                return "Mistral"
    if "Vicuna" in m:                 return "Vicuna"
    if m.startswith("Qwen2.5"):       return "Qwen2.5"
    if m.startswith("Phi"):           return "Phi"
    return "Phi"

MODEL_COLORS = {m: _FAMILY_COLOR[_fam(m)] for m in ALL_MODELS}

# Marker per family
def _marker(m: str) -> str:
    if m.startswith("InternVL"):     return "o"
    if m.startswith("Qwen3-VL"):     return "X"
    if m.startswith("Qwen3"):        return "X"
    if m.startswith("LLaVA-1.5"):    return "D"
    if "Mistral" in m:               return "^"
    if "Vicuna" in m:                return "v"
    if m.startswith("Qwen2.5"):      return "s"
    if m.startswith("Phi"):          return "p"
    return "o"

MODEL_STYLE = {m: _marker(m) for m in ALL_MODELS}
# think models use "P" marker
for m in ALL_MODELS:
    if "(think)" in m:
        MODEL_STYLE[m] = "P"

# Marker size proportional to log(param_count)
_SIZE_B = {
    "InternVL-1B": 0.9, "InternVL-1B (LM)": 0.9,
    "InternVL-2B": 2.0, "InternVL-2B (LM)": 2.0,
    "InternVL-8B": 8.0, "InternVL-8B (LM)": 8.0,
    "Qwen3-VL-2B": 2.0,  "Qwen3-VL-2B (LM)": 2.0,
    "Qwen3-VL-4B": 4.0,  "Qwen3-VL-4B (LM)": 4.0,
    "Qwen3-VL-8B": 8.0,  "Qwen3-VL-8B (LM)": 8.0,
    "Qwen3-VL-32B": 32.0, "Qwen3-VL-32B (LM)": 32.0,
    "LLaVA-1.5-7B": 7.0, "LLaVA-1.5 (LM)": 7.0,
    "LLaVA-Mistral": 7.0, "LLaVA-Mistral (LM)": 7.0,
    "LLaVA-Vicuna": 7.0,  "LLaVA-Vicuna (LM)": 7.0,
    "LLaVA-Vicuna-13B": 13.0, "LLaVA-Vicuna-13B (LM)": 13.0,
    "Phi-3.5-mini": 3.8,
    "Qwen3-0.6B": 0.6, "Qwen3-0.6B (think)": 0.6,
    "Qwen3-1.7B": 1.7, "Qwen3-1.7B (think)": 1.7,
    "Qwen3-4B": 4.0,   "Qwen3-4B (think)": 4.0,
    "Qwen3-8B": 8.0,   "Qwen3-8B (think)": 8.0,
    "Qwen3-32B": 32.0, "Qwen3-32B (think)": 32.0,
    "Qwen2.5-7B-Instruct": 7.0,
    "Mistral-7B": 7.0,
    "Vicuna-7B": 7.0, "Vicuna-13B": 13.0,
}
_LOG_LO, _LOG_HI = np.log(0.6), np.log(32.0)

def _marker_s(m: str) -> float:
    b = _SIZE_B.get(m, 7.0)
    t = (np.log(b) - _LOG_LO) / (_LOG_HI - _LOG_LO)
    return float(20 + t * 100)  # range [20, 120]

# y position per row (inverted axis: 3=top, 0=bottom)
GROUP_Y  = {"VLM": 3, "Backbone Decoder": 2, "Standalone LLM": 1}
HUMAN_Y  = 0
GROUP_LABELS = {
    "VLM":              "VLM",
    "Backbone Decoder": "Backbone\nDecoder",
    "Standalone LLM":   "Standalone\nLLM",
}


# ── helper ────────────────────────────────────────────────────────────────────
def _qid2atype() -> dict[int, str]:
    mapper = VQAAnswerMapper()
    mapper._load()
    return {int(qid): ann.get("answer_type", "other") for qid, ann in mapper.annotations.items()}


# ── compute Pearson r ─────────────────────────────────────────────────────────
def compute_pearson_r(
    pair_cache: pd.DataFrame,
    all_questions: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Per-question Pearson r between mean HM SBERT and mean HH SBERT.
    If all_questions=False (default): free-text questions only.
    If all_questions=True: all question types (incl. yes/no, count).
    Returns (model_r, human_r).  human_r uses LOO.
    Note: abstentions are NOT pre-filtered from pair_cache; they contribute
    low but non-zero SBERT scores.
    """
    if all_questions:
        mask = pd.Series(True, index=pair_cache.index)
    else:
        qid2atype = _qid2atype()
        ft_qids = {qid for qid, atype in qid2atype.items() if atype == "other"}
        mask = pair_cache["question_id"].isin(ft_qids)

    hh = pair_cache[(pair_cache["pair_type"] == "HH") & mask].copy()
    hm = pair_cache[(pair_cache["pair_type"] == "HM") & mask].copy()

    ref = hh.groupby(["question_id", "variant"])["sbert_score"].mean()

    # Mistral-7B excluded: uniform near-zero SBERT from verbose outputs makes
    # its r a measurement artifact (r≈−0.19 all-q; −0.07 free-text), not alignment signal.
    EXCLUDE_MODELS = {"Mistral-7B"}

    model_r: dict[str, float] = {}
    for model, grp in hm.groupby("subject_2"):
        if model not in ALL_MODELS or model in EXCLUDE_MODELS:
            continue
        m_mean = grp.groupby(["question_id", "variant"])["sbert_score"].mean()
        common = m_mean.index.intersection(ref.index)
        if len(common) < 20:
            continue
        r, _ = _pearsonr(m_mean[common], ref[common])
        model_r[model] = float(r)

    human_r: dict[str, float] = {}
    for p, grp in hh.groupby("subject_1"):
        p_mean  = grp.groupby(["question_id", "variant"])["sbert_score"].mean()
        loo_ref = hh[hh["subject_1"] != p].groupby(
            ["question_id", "variant"])["sbert_score"].mean()
        common  = p_mean.index.intersection(loo_ref.index)
        if len(common) < 20:
            continue
        r, _ = _pearsonr(p_mean[common], loo_ref[common])
        human_r[p] = float(r)

    return model_r, human_r


# ── plot ──────────────────────────────────────────────────────────────────────
MODEL_ORDER_7B = {
    "VLM": ["Qwen3-VL-8B", "InternVL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna"],
    "Backbone Decoder": ["Qwen3-VL-8B (LM)", "InternVL-8B (LM)", "LLaVA-1.5 (LM)",
                         "LLaVA-Mistral (LM)", "LLaVA-Vicuna (LM)"],
    "Standalone LLM": ["Qwen2.5-7B-Instruct", "Qwen3-8B", "Qwen3-8B (think)",
                       "Mistral-7B", "Vicuna-7B"],
}


def make_figure(
    model_r: dict[str, float],
    human_r: dict[str, float],
    model_order: dict | None = None,
    model_r_blind: dict[str, float] | None = None,  # hollow markers if provided
    xlabel: str = "Per-question Pearson $r$",
) -> plt.Figure:
    if model_order is None:
        model_order = MODEL_ORDER

    h_vals = np.array(list(human_r.values()))
    h_p5   = float(np.percentile(h_vals, 5))
    h_p95  = float(np.percentile(h_vals, 95))
    h_med  = float(np.median(h_vals))

    all_r_vals = [v for v in model_r.values() if np.isfinite(v)]
    if model_r_blind:
        all_r_vals += [v for v in model_r_blind.values() if np.isfinite(v)]
    x_lo = round(min(0.35, min(all_r_vals) - 0.05) * 10) / 10 if all_r_vals else 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    fig.subplots_adjust(left=0.20, right=0.95, top=0.97, bottom=0.38)

    # Human band (not in legend)
    ax.axvspan(h_p5, h_p95, color="#dddddd", zorder=1)
    ax.axvline(h_med, color="black", lw=1.5, zorder=2)

    # Human participants row
    for h_val in h_vals:
        ax.scatter(h_val, HUMAN_Y, marker="*", s=40, color="#888888",
                   alpha=0.5, zorder=4, linewidths=0)
    ax.axhline(0.55, color="#cccccc", lw=0.8, linestyle="--", zorder=1)

    # Helper to draw one set of dots
    def _draw_dots(r_dict, filled: bool):
        for grp, models in model_order.items():
            y_center = GROUP_Y[grp]
            n = len(models)
            sorted_models = sorted(models, key=lambda m: r_dict.get(m, float("nan")))
            offsets = np.linspace(-0.18, 0.18, n) if n > 1 else [0.0]
            for model, dy in zip(sorted_models, offsets):
                r_val = r_dict.get(model, float("nan"))
                if not np.isfinite(r_val):
                    continue
                color = MODEL_COLORS[model]
                mk    = MODEL_STYLE.get(model, "o")
                sz    = _marker_s(model)
                if filled:
                    ax.scatter(r_val, y_center + dy, marker=mk, s=sz,
                               color=color, zorder=5,
                               linewidths=0.5, edgecolors="#333333")
                else:
                    ax.scatter(r_val, y_center + dy, marker=mk, s=sz,
                               facecolors="white", edgecolors=color,
                               zorder=5, linewidths=1.2)

    # Blind (hollow) drawn first so filled sits on top
    if model_r_blind:
        _draw_dots(model_r_blind, filled=False)
    _draw_dots(model_r, filled=True)

    # Axes
    ytick_pos = [GROUP_Y[g] for g in model_order] + [HUMAN_Y]
    ytick_lbl = [GROUP_LABELS[g] for g in model_order] + ["Participants"]
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lbl, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xlim(x_lo, 1.02)
    ax.set_ylim(-0.5, 3.55)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)

    # Legend: family colors/markers + condition fill + human star
    FAMILY_LEGEND = [
        (_FAMILY_COLOR["InternVL"],  "o", "InternVL"),
        (_FAMILY_COLOR["Qwen3"],     "X", "Qwen3-VL / Qwen3"),
        (_FAMILY_COLOR["LLaVA-1.5"],"D", "LLaVA-1.5"),
        (_FAMILY_COLOR["Mistral"],   "^", "LLaVA-Mistral / Mistral"),
        (_FAMILY_COLOR["Vicuna"],    "v", "LLaVA-Vicuna / Vicuna"),
        (_FAMILY_COLOR["Qwen3"],     "P", "Qwen3 (think)"),
        (_FAMILY_COLOR["Qwen2.5"],   "s", "Qwen2.5"),
    ]
    handles = [
        mlines.Line2D([], [], color=c, marker=mk, lw=0,
                      markersize=6, markeredgecolor="white", markeredgewidth=0.4,
                      label=name)
        for c, mk, name in FAMILY_LEGEND
    ]
    if model_r_blind:
        handles += [
            mlines.Line2D([], [], color="#555555", marker="o", lw=0, markersize=6,
                          markerfacecolor="#555555", markeredgecolor="white",
                          markeredgewidth=0.4, label="w/ instruction"),
            mlines.Line2D([], [], color="#555555", marker="o", lw=0, markersize=6,
                          markerfacecolor="white", markeredgecolor="#555555",
                          markeredgewidth=1.2, label="w/o instruction"),
        ]
    handles += [mlines.Line2D([], [], color="#888888", marker="*", lw=0,
                               markersize=6, alpha=0.7, label="Participants")]
    ax.legend(handles=handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, fontsize=7.5, frameon=False,
              handlelength=1.2, columnspacing=0.8, handletextpad=0.4)

    return fig


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading pair_cache...")
    pair_cache = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")

    latex_dir = ROOT / "latex/AAAI2026/LaTeX/figures/alignment_profile"
    latex_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    pair_cache_blind = pd.read_parquet(EXPORTS / "pair_cache_cleaned_blind.parquet")

    # Compute r for both conditions × both question scopes
    cond_data = {}
    for cond_label, pc in [("inst", pair_cache), ("blind", pair_cache_blind)]:
        for allq in [False, True]:
            scope = "allq" if allq else "ft"
            key = f"{cond_label}_{scope}"
            print(f"\nComputing Pearson r ({key})...")
            model_r, human_r = compute_pearson_r(pc, all_questions=allq)
            h_vals = np.array(list(human_r.values()))
            print(f"Human r: median={np.median(h_vals):.3f}, "
                  f"p5={np.percentile(h_vals,5):.3f}, p95={np.percentile(h_vals,95):.3f}")
            cond_data[key] = (model_r, human_r)

    # Save versions
    # (suffix, model_order, inst_key, blind_key_or_None)
    versions = [
        ("_7b",      MODEL_ORDER_7B, "inst_ft",   None),           # 7B, inst, free-text only
        ("_7b_inst", MODEL_ORDER_7B, "inst_ft",   "blind_ft"),     # 7B, both conditions, free-text
        ("",         MODEL_ORDER,    "inst_ft",   None),            # all-scales, inst, free-text
        ("_allq",    MODEL_ORDER,    "inst_allq", None),            # all-scales, inst, all questions
    ]
    for suffix, morder, inst_key, blind_key in versions:
        model_r_inst,  human_r = cond_data[inst_key]
        model_r_blind = cond_data[blind_key][0] if blind_key else None  # type: ignore[index]
        xl = ("Per-question Pearson $r$  (all question types; Mistral-7B excluded)"
              if "allq" in inst_key
              else "Per-question Pearson $r$")
        fig = make_figure(model_r_inst, human_r, model_order=morder,
                          model_r_blind=model_r_blind, xlabel=xl)
        fname = f"alignment_profile_sbert_r_groups{suffix}.png"
        out_path = OUT_DIR / fname
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        shutil.copy(out_path, latex_dir / fname)
        print(f"Saved: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
