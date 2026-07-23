"""
Generate answer-distribution figures for blind / inst-blind HM analysis.

Outputs are written under condition folders:
  figures/answer_distribution/blind/{yesno,number}/
  figures/answer_distribution/inst_blind/{yesno,number}/
  figures/answer_distribution/blind_abstsplit/{yesno,number}/
  figures/answer_distribution/inst_blind_abstsplit/{yesno,number}/

The condition is encoded by folder, not filename, e.g.:
  inst_blind/yesno/combined.png
  inst_blind/yesno/qwen3_vl_8b.png
  inst_blind_abstsplit/number/combined.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from figures.helpers import clear_output_plots, read_response_exports
from analysis.utils.vqa import VQAAnswerMapper, extract_number, bin_number, preprocess_answer
from analysis.utils.abstention import classify, is_abstained


OUT_BASE = ROOT / "figures" / "answer_distribution"

TRIPLETS = [
    {"slug": "qwen3_vl_8b", "vlm": "Qwen3-VL-8B", "decoder": "Qwen3-VL-8B (LM)", "llm": "Qwen3-8B"},
    {"slug": "internvl_8b", "vlm": "InternVL-8B", "decoder": "InternVL-8B (LM)", "llm": "Qwen3-8B"},
    {"slug": "internvl_8b_think", "vlm": "InternVL-8B", "decoder": "InternVL-8B (LM)", "llm": "Qwen3-8B", "extras": ["Qwen3-8B (think)"]},
    {"slug": "llava_15_7b", "vlm": "LLaVA-1.5-7B", "decoder": "LLaVA-1.5 (LM)", "llm": "Vicuna-7B"},
    {"slug": "llava_mistral_7b", "vlm": "LLaVA-Mistral", "decoder": "LLaVA-Mistral (LM)", "llm": "Mistral-7B"},
    {"slug": "llava_vicuna_7b", "vlm": "LLaVA-Vicuna", "decoder": "LLaVA-Vicuna (LM)", "llm": "Vicuna-7B"},
]

YN_COLORS = {"yes": "#6BA292", "no": "#C97A6A", "others": "#B5B5B5"}
NUM_COLORS = {
    "0": "#DADADA",
    "1": "#BFD7EA",
    "2–3": "#9EC1A3",
    "4–5": "#7FB3A2",
    "6–10": "#5C8D89",
    "11–20": "#4C6A92",
    ">20": "#8E5A5A",
    "others": "#B5B5B5",
}
NUM_ORDER = ["0", "1", "2–3", "4–5", "6–10", "11–20", ">20", "others"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=["blind", "inst_blind", "all"],
        default="all",
        help="Which condition to export.",
    )
    parser.add_argument(
        "--filter-abstentions",
        action="store_true",
        help="Filter hard/soft abstentions from model outputs and append _abstsplit to filenames.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Export only the combined all-model figures.",
    )
    parser.add_argument(
        "--triplets-only",
        action="store_true",
        help="Export only the family triplet figures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing pngs in the selected condition folders before exporting.",
    )
    return parser.parse_args()


def condition_folder(condition: str, filter_abstentions: bool) -> Path:
    suffix = "_abstsplit" if filter_abstentions else ""
    return OUT_BASE / f"{condition}{suffix}"


def ensure_condition_dirs(condition: str, filter_abstentions: bool) -> tuple[Path, Path]:
    base = condition_folder(condition, filter_abstentions)
    yesno_dir = base / "yesno"
    number_dir = base / "number"
    yesno_dir.mkdir(parents=True, exist_ok=True)
    number_dir.mkdir(parents=True, exist_ok=True)
    return yesno_dir, number_dir


def clean_output(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .apply(lambda x: preprocess_answer(x, strip_think=True))
        .str.lower()
    )


def classify_yn(text: str) -> str:
    text = str(text).strip()
    if text == "yes" or text.startswith("yes "):
        return "yes"
    if text == "no" or text.startswith("no "):
        return "no"
    return "others"


def number_bin(text: str) -> str:
    value = extract_number(text)
    return bin_number(value) if value is not None else "others"


def build_stack(df: pd.DataFrame, value_col: str, source_order: list[str]) -> pd.DataFrame:
    counts = df.value_counts(["source", value_col]).reset_index(name="count")
    counts["proportion"] = counts.groupby("source")["count"].transform(lambda x: x / x.sum())
    return (
        counts.pivot(index="source", columns=value_col, values="proportion")
        .fillna(0)
        .reindex(source_order)
        .fillna(0)
    )


def plot_yn(stack_df: pd.DataFrame):
    cats = [c for c in ["yes", "no", "others"] if c in stack_df.columns]
    fig, ax = plt.subplots(figsize=(7.8, 0.58 * stack_df.shape[0] + 0.8))
    stack_df[cats].plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[YN_COLORS[c] for c in cats],
        width=0.75,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for i, src in enumerate(stack_df.index):
        cum = 0.0
        for cat in cats:
            val = float(stack_df.loc[src, cat])
            if val > 0.04:
                ax.text(
                    cum + val / 2,
                    i,
                    f"{val * 100:.0f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                )
            cum += val
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=3, frameon=False)
    plt.tight_layout()
    return fig


def plot_num(stack_df: pd.DataFrame):
    order = [c for c in NUM_ORDER if c in stack_df.columns]
    fig, ax = plt.subplots(figsize=(11.5, 0.58 * stack_df.shape[0] + 0.8))
    stack_df[order].plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[NUM_COLORS[c] for c in order],
        width=0.75,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for i, src in enumerate(stack_df.index):
        cum = 0.0
        for cat in order:
            val = float(stack_df.loc[src, cat])
            if val > 0.04:
                ax.text(
                    cum + val / 2,
                    i,
                    f"{val * 100:.0f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                )
            cum += val
    ax.legend(
        bbox_to_anchor=(0.5, -0.18),
        loc="upper center",
        ncol=len(order),
        frameon=False,
        fontsize=10,
    )
    plt.tight_layout()
    return fig


def save(fig, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(path)


def build_human_and_gt(exports: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, set[int], dict[int, str]]:
    mapper = VQAAnswerMapper()
    mapper._load()

    human_src = exports["human"]
    human_sub = human_src[human_src["variant"] == "C"].copy()
    human_qid_set = set(human_sub["question_id"].astype(int).unique())

    qid2atype = {
        int(qid): ann.get("answer_type", "other")
        for qid, ann in mapper.annotations.items()
    }

    human_df = pd.DataFrame(
        {
            "qid": human_sub["question_id"].astype(int),
            "answer_type": human_sub["question_id"].astype(int).map(qid2atype).fillna("other"),
            "source": "Humans",
            "output": clean_output(human_sub["response"]),
        }
    )

    gt_rows = []
    for qid in sorted(human_qid_set):
        answers = mapper.get_answers(qid)
        atype = qid2atype.get(qid, "other")
        for ans in answers:
            gt_rows.append(
                {
                    "qid": qid,
                    "answer_type": atype,
                    "source": "Ground Truth",
                    "output": preprocess_answer(ans, strip_think=True).lower(),
                }
            )
    gt_df = pd.DataFrame(gt_rows)
    return human_df, gt_df, human_qid_set, qid2atype


def build_model_df(
    exports: dict[str, pd.DataFrame],
    *,
    condition: str,
    qids: set[int],
    qid2atype: dict[int, str],
    filter_abstentions: bool,
) -> pd.DataFrame:
    key = "model_blind" if condition == "blind" else "model_inst_blind"
    sub = exports[key].copy()
    sub = sub[sub["variant"] == "C"].copy()
    sub = sub[sub["question_id"].astype(int).isin(qids)].copy()
    sub["qid"] = sub["question_id"].astype(int)
    sub["answer_type"] = sub["qid"].map(qid2atype).fillna("other")
    sub["output"] = clean_output(sub["response"])
    if filter_abstentions:
        sub = sub[sub["output"].apply(lambda x: not is_abstained(classify(x, None)))].copy()
    return sub


def filename(stem: str) -> str:
    return f"{stem}.png"


def export_triplet(
    triplet: dict[str, str],
    *,
    model_df: pd.DataFrame,
    human_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    yesno_dir: Path,
    number_dir: Path,
):
    model_names = [triplet["vlm"], triplet["decoder"], triplet["llm"]] + triplet.get("extras", [])
    subset = model_df[model_df["model"].isin(model_names)].copy()
    subset["source"] = subset["model"]
    base_df = pd.concat(
        [
            human_df[["qid", "answer_type", "source", "output"]],
            gt_df[["qid", "answer_type", "source", "output"]],
            subset[["qid", "answer_type", "source", "output"]],
        ],
        ignore_index=True,
    )
    row_order = ["Humans", "Ground Truth"] + model_names

    yesno_df = base_df[base_df["answer_type"] == "yes/no"].copy()
    yesno_df["yn"] = yesno_df["output"].apply(classify_yn)
    yn_stack = build_stack(yesno_df, "yn", row_order)
    save(plot_yn(yn_stack), yesno_dir, filename(triplet["slug"]))

    number_df = base_df[base_df["answer_type"] == "number"].copy()
    number_df["number_bin"] = number_df["output"].apply(number_bin)
    num_stack = build_stack(number_df, "number_bin", row_order)
    save(plot_num(num_stack), number_dir, filename(triplet["slug"]))


def export_combined(
    *,
    model_df: pd.DataFrame,
    human_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    yesno_dir: Path,
    number_dir: Path,
):
    sub = model_df.copy()
    sub["source"] = sub["model"]
    base_df = pd.concat(
        [
            human_df[["qid", "answer_type", "source", "output"]],
            gt_df[["qid", "answer_type", "source", "output"]],
            sub[["qid", "answer_type", "source", "output"]],
        ],
        ignore_index=True,
    )
    source_order = ["Humans", "Ground Truth"] + sorted(sub["model"].unique().tolist())

    yesno_df = base_df[base_df["answer_type"] == "yes/no"].copy()
    yesno_df["yn"] = yesno_df["output"].apply(classify_yn)
    yn_stack = build_stack(yesno_df, "yn", source_order)
    save(plot_yn(yn_stack), yesno_dir, filename("combined"))

    number_df = base_df[base_df["answer_type"] == "number"].copy()
    number_df["number_bin"] = number_df["output"].apply(number_bin)
    num_stack = build_stack(number_df, "number_bin", source_order)
    save(plot_num(num_stack), number_dir, filename("combined"))


args = parse_args()

exports = read_response_exports(ROOT, check_completeness=False)
human_df, gt_df, human_qid_set, qid2atype = build_human_and_gt(exports)

conditions = ["blind", "inst_blind"] if args.condition == "all" else [args.condition]
do_combined = not args.triplets_only
do_triplets = not args.combined_only

for condition in conditions:
    yesno_dir, number_dir = ensure_condition_dirs(condition, args.filter_abstentions)
    if args.overwrite:
        clear_output_plots(yesno_dir, overwrite=True)
        clear_output_plots(number_dir, overwrite=True)
    model_df = build_model_df(
        exports,
        condition=condition,
        qids=human_qid_set,
        qid2atype=qid2atype,
        filter_abstentions=args.filter_abstentions,
    )
    if do_combined:
        export_combined(
            model_df=model_df,
            human_df=human_df,
            gt_df=gt_df,
            yesno_dir=yesno_dir,
            number_dir=number_dir,
        )
    if do_triplets:
        for triplet in TRIPLETS:
            export_triplet(
                triplet,
                model_df=model_df,
                human_df=human_df,
                gt_df=gt_df,
                yesno_dir=yesno_dir,
                number_dir=number_dir,
            )
