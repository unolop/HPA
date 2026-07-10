"""Heatmaps for matched 7/8B HH-HM grouped SBERT views."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "figures"))
sys.path.insert(0, str(ROOT))
import plot_style  # noqa: F401

from analysis.utils.abstention import classify, is_abstained

EXPORTS = ROOT / "analysis" / "session2" / "exports"
OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)

MODELS_7B = [
    # VLM
    "InternVL-8B",
    "Qwen3-VL-8B",
    "LLaVA-1.5-7B",
    "LLaVA-Mistral",
    "LLaVA-Vicuna",
    # Backbone decoder
    "InternVL-8B (LM)",
    "Qwen3-VL-8B (LM)",
    "LLaVA-1.5 (LM)",
    "LLaVA-Mistral (LM)",
    "LLaVA-Vicuna (LM)",
    # Standalone LLM
    "Qwen3-8B",
    "Qwen3-8B (think)",
    "Qwen2.5-7B-Instruct",
    "Vicuna-7B",
    "Mistral-7B",
]


def _group_slices(models: list[str]) -> list[tuple[str, list[str]]]:
    return [
        (
            "VLM",
            [m for m in models if "(LM)" not in m and m not in ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B"]],
        ),
        ("Backbone\nDecoder", [m for m in models if "(LM)" in m]),
        ("Standalone\nLLM", [m for m in models if m in ["Qwen3-8B", "Qwen3-8B (think)", "Qwen2.5-7B-Instruct", "Vicuna-7B", "Mistral-7B"]]),
    ]


def _filter_hm_abstentions(pc: pd.DataFrame) -> pd.DataFrame:
    model = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")
    model = model[model["model"].isin(MODELS_7B)].copy()
    model["is_abstained"] = model["response"].astype(str).apply(lambda x: is_abstained(classify(x, None)))
    keep = model.loc[~model["is_abstained"], ["question_id", "variant", "model"]].copy()
    keep["question_id"] = keep["question_id"].astype(int)
    hm = pc[pc["pair_type"] == "HM"].copy()
    hm["question_id"] = hm["question_id"].astype(int)
    hm = hm.merge(
        keep.rename(columns={"model": "subject_2"}),
        on=["question_id", "variant", "subject_2"],
        how="inner",
    )
    hh = pc[pc["pair_type"] == "HH"].copy()
    return pd.concat([hh, hm], ignore_index=True)


def make_heatmap(
    pc: pd.DataFrame,
    *,
    key: str,
    label: str,
    suffix: str,
    title_extra: str = "",
):
    hh = pc[pc["pair_type"] == "HH"].copy()
    hh_grp = hh.groupby(key).agg(sbert=("sbert_score", "mean")).reset_index()

    hm = pc[pc["pair_type"] == "HM"].copy()
    models = [m for m in MODELS_7B if m in hm["subject_2"].unique()]
    hm_grp = (
        hm[hm["subject_2"].isin(models)]
        .groupby(["subject_2", key])
        .agg(sbert=("sbert_score", "mean"))
        .reset_index()
    )

    hm_piv = hm_grp.pivot(index="subject_2", columns=key, values="sbert").reindex(models)
    slice_order = hh_grp.sort_values("sbert", ascending=False)[key].tolist()
    slice_order = [x for x in slice_order if x in hm_piv.columns]
    hm_piv = hm_piv[slice_order]

    hh_row = hh_grp.set_index(key)["sbert"].reindex(slice_order)
    full = pd.concat(
        [pd.DataFrame([hh_row.values], columns=slice_order, index=["Human–Human"]), hm_piv]
    )

    fig_w = max(10, 0.9 * len(slice_order) + 6)
    fig, ax = plt.subplots(figsize=(fig_w, 8))
    sns.heatmap(
        full.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean SBERT", "shrink": 0.8},
        ax=ax,
        vmin=0.15,
        vmax=0.70,
    )

    ax.axhline(1, color="black", linewidth=2)

    idx = 1
    for grp_label, grp_models in _group_slices(models):
        n = len(grp_models)
        if n == 0:
            continue
        ax.text(len(slice_order) + 0.6, idx + n / 2, grp_label, fontsize=10, fontweight="bold", va="center", ha="left")
        idx += n
        ax.axhline(idx, color="black", linewidth=1.5, linestyle="--")

    ax.set_xlabel(label)
    ax.set_ylabel("")
    ax.set_title(f"HM SBERT by {label} (7B scale){title_extra}")
    plt.tight_layout()

    out = OUTDIR / f"heatmap_7b_{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    pc_cleaned = EXPORTS / "pair_cache_cleaned.parquet"
    if not pc_cleaned.exists():
        print("pair_cache_cleaned.parquet not found")
        sys.exit(1)

    pc = pd.read_parquet(pc_cleaned)

    human = pd.read_csv(EXPORTS / "responses_human.csv")
    vC = human[human["variant"] == "C"]
    q_yn = vC.groupby("question_id").apply(
        lambda g: (g["response"].astype(str).str.lower().str.strip().isin(["yes", "no"])).mean() > 0.5,
        include_groups=False,
    )
    yn_qids = set(q_yn[q_yn].index)

    pc_ft = pc[~pc["question_id"].isin(yn_qids)].copy()
    pc_filt = _filter_hm_abstentions(pc)
    pc_ft_filt = pc_filt[~pc_filt["question_id"].isin(yn_qids)].copy()

    # Operation heatmaps
    make_heatmap(pc, key="op", label="Operation Type", suffix="op_all", title_extra=" — all questions")
    make_heatmap(pc_ft, key="op", label="Operation Type", suffix="op_freetext", title_extra=" — free-text only")
    make_heatmap(pc_filt, key="op", label="Operation Type", suffix="op_all_filtered", title_extra=" — all questions, HM filtered")
    make_heatmap(pc_ft_filt, key="op", label="Operation Type", suffix="op_freetext_filtered", title_extra=" — free-text only, HM filtered")

    # Entity heatmaps
    make_heatmap(pc, key="ent", label="Entity Type", suffix="ent_all", title_extra=" — all questions")
    make_heatmap(pc_ft, key="ent", label="Entity Type", suffix="ent_freetext", title_extra=" — free-text only")
    make_heatmap(pc_filt, key="ent", label="Entity Type", suffix="ent_all_filtered", title_extra=" — all questions, HM filtered")
    make_heatmap(pc_ft_filt, key="ent", label="Entity Type", suffix="ent_freetext_filtered", title_extra=" — free-text only, HM filtered")
