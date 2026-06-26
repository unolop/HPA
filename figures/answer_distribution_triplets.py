"""
Export per-family answer-distribution figures for matched 7/8B triplets.

Each figure keeps the same horizontal stacked-bar structure as the existing
answer-distribution plots, but focuses on a single family triplet:

  Humans
  paired VLM
  paired VLM backbone decoder
  paired standalone LLM

Outputs are saved to:
  figures/answer_distribution/

By default this exports both `blind` and `inst_blind` figures for:
  - Qwen3-VL-8B  -> Qwen3-VL-8B (LM)  -> Qwen3-8B
  - InternVL-8B  -> InternVL-8B (LM)  -> Qwen3-8B
  - LLaVA-1.5-7B -> LLaVA-1.5 (LM)    -> Vicuna-7B
  - LLaVA-Mistral -> LLaVA-Mistral (LM) -> Mistral-7B
  - LLaVA-Vicuna -> LLaVA-Vicuna (LM) -> Vicuna-7B

Run from repo root:
  conda run -n zero python figures/answer_distribution_triplets.py --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from config import VLM_BASE_LLM
from helpers import clear_output_plots, read_response_exports
from utils.vqa import VQAAnswerMapper, bin_number, extract_number

parser = argparse.ArgumentParser()
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete existing plot files in the output folder before exporting.",
)
parser.add_argument(
    "--condition",
    choices=["blind", "inst_blind", "all"],
    default="all",
    help="Which condition(s) to export.",
)
args = parser.parse_args()

OUT_DIR = ROOT / "figures" / "answer_distribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(OUT_DIR, overwrite=args.overwrite)

TRIPLETS = [
    {
        "slug": "qwen3_vl_8b",
        "vlm": "Qwen3-VL-8B",
        "decoder": "Qwen3-VL-8B (LM)",
        "llm": "Qwen3-8B",
    },
    {
        "slug": "internvl_8b",
        "vlm": "InternVL-8B",
        "decoder": "InternVL-8B (LM)",
        "llm": "Qwen3-8B",
    },
    {
        "slug": "llava_15_7b",
        "vlm": "LLaVA-1.5-7B",
        "decoder": "LLaVA-1.5 (LM)",
        "llm": "Vicuna-7B",
    },
    {
        "slug": "llava_mistral_7b",
        "vlm": "LLaVA-Mistral",
        "decoder": "LLaVA-Mistral (LM)",
        "llm": "Mistral-7B",
    },
    {
        "slug": "llava_vicuna_7b",
        "vlm": "LLaVA-Vicuna",
        "decoder": "LLaVA-Vicuna (LM)",
        "llm": "Vicuna-7B",
    },
]

CONDITIONS = ["blind", "inst_blind"] if args.condition == "all" else [args.condition]
COND_PREFIX = {
    "blind": "blind",
    "inst_blind": "inst_blind",
}

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

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


def clean_output(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


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
    stack = counts.pivot(index="source", columns=value_col, values="proportion").fillna(0)
    return stack.reindex(source_order).dropna(how="all")


def plot_yn(stack_df: pd.DataFrame) -> plt.Figure:
    cats = [c for c in ["yes", "no", "others"] if c in stack_df.columns]
    n_rows = stack_df.shape[0]
    fig, ax = plt.subplots(figsize=(7.8, 0.58 * n_rows + 0.8))
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
        cum = 0
        for cat in cats:
            val = stack_df.loc[src, cat]
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


def plot_num(stack_df: pd.DataFrame) -> plt.Figure:
    order = [c for c in NUM_ORDER if c in stack_df.columns]
    n_rows = stack_df.shape[0]
    fig, ax = plt.subplots(figsize=(11.5, 0.58 * n_rows + 0.8))
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
        cum = 0
        for cat in order:
            val = stack_df.loc[src, cat]
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


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  [answer_distribution] {name}")


print("Loading VQA annotations...")
mapper = VQAAnswerMapper()
mapper._load()
qid2atype = {int(qid): ann.get("answer_type", "other") for qid, ann in mapper.annotations.items()}

print("Loading response exports...")
exports = read_response_exports(ROOT)
human_src = exports["human"]
human_sub = human_src[human_src["variant"] == "C"].copy()
human_qid_set = set(human_sub["question_id"].astype(int).unique())
N_HUMANS = human_src["participant"].nunique()

human_df = pd.DataFrame(
    {
        "qid": human_sub["question_id"].astype(int),
        "answer_type": human_sub["question_id"].astype(int).map(qid2atype).fillna("other"),
        "source": "Humans",
        "condition": "inst_blind",
        "output": clean_output(human_sub["response"]),
    }
)

model_rows: list[pd.DataFrame] = []
for condition, export_key in [("blind", "model_blind"), ("inst_blind", "model_inst_blind")]:
    sub = exports[export_key].copy()
    sub = sub[sub["variant"] == "C"].copy()
    sub = sub[sub["question_id"].astype(int).isin(human_qid_set)].copy()
    sub["qid"] = sub["question_id"].astype(int)
    sub["answer_type"] = sub["qid"].map(qid2atype).fillna("other")
    sub["condition"] = condition
    sub["output"] = clean_output(sub["response"])
    model_rows.append(sub[["qid", "answer_type", "model", "condition", "output"]])

model_df = pd.concat(model_rows, ignore_index=True)

human_yesno_q = int((human_df["answer_type"] == "yes/no").sum() / max(N_HUMANS, 1))
human_number_q = int((human_df["answer_type"] == "number").sum() / max(N_HUMANS, 1))
print(f"Human study: q_yesno={human_yesno_q}, q_number={human_number_q}, participants={N_HUMANS}")

for triplet in TRIPLETS:
    base_llm = VLM_BASE_LLM.get(triplet["vlm"])
    if base_llm != triplet["llm"]:
        print(
            f"Warning: configured LLM for {triplet['vlm']} is {triplet['llm']}, "
            f"but VLM_BASE_LLM says {base_llm}"
        )

    source_labels = {
        "Humans": "Humans",
        triplet["vlm"]: triplet["vlm"],
        triplet["decoder"]: triplet["decoder"],
        triplet["llm"]: triplet["llm"],
    }

    for condition in CONDITIONS:
        subset = model_df[
            (model_df["condition"] == condition)
            & (model_df["model"].isin([triplet["vlm"], triplet["decoder"], triplet["llm"]]))
        ].copy()
        subset["source"] = subset["model"]

        base_df = pd.concat(
            [
                human_df[["qid", "answer_type", "source", "condition", "output"]],
                subset[["qid", "answer_type", "source", "condition", "output"]],
            ],
            ignore_index=True,
        )

        yesno_order = [
            source_labels["Humans"],
            source_labels[triplet["vlm"]],
            source_labels[triplet["decoder"]],
            source_labels[triplet["llm"]],
        ]
        yesno_df = base_df[base_df["answer_type"] == "yes/no"].copy()
        yesno_df["source"] = yesno_df["source"].map(
            {
                "Humans": yesno_order[0],
                triplet["vlm"]: yesno_order[1],
                triplet["decoder"]: yesno_order[2],
                triplet["llm"]: yesno_order[3],
            }
        )
        yn_stack = build_stack(
            yesno_df.assign(yn=yesno_df["output"].apply(classify_yn)),
            "yn",
            yesno_order,
        )
        save(
            plot_yn(yn_stack),
            f"{COND_PREFIX[condition]}_{triplet['slug']}_yesno.png",
        )

        number_order = [
            source_labels["Humans"],
            source_labels[triplet["vlm"]],
            source_labels[triplet["decoder"]],
            source_labels[triplet["llm"]],
        ]
        number_df = base_df[base_df["answer_type"] == "number"].copy()
        number_df["source"] = number_df["source"].map(
            {
                "Humans": number_order[0],
                triplet["vlm"]: number_order[1],
                triplet["decoder"]: number_order[2],
                triplet["llm"]: number_order[3],
            }
        )
        num_stack = build_stack(
            number_df.assign(number_bin=number_df["output"].apply(number_bin)),
            "number_bin",
            number_order,
        )
        save(
            plot_num(num_stack),
            f"{COND_PREFIX[condition]}_{triplet['slug']}_number.png",
        )

print("\nDone. All figures saved to:", OUT_DIR)
