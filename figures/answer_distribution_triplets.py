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

from config import MODEL_GROUP, VLM_BASE_LLM
from helpers import clear_output_plots, read_response_exports
from utils.abstention import classify, is_abstained
from utils.constants import MODEL_FAMILY, MODEL_FAMILY_COLORS, MODEL_SIZE_B
from utils.load_session import clean_answer
from utils.vqa import VQAAnswerMapper, bin_number, extract_number, preprocess_answer

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
parser.add_argument(
    "--filter_abstentions",
    action="store_true",
    help="Drop soft/hard abstentions before computing answer distributions.",
)
parser.add_argument(
    "--split_abstentions",
    action="store_true",
    help="Keep abstentions but split them into hard/soft categories instead of folding them into others.",
)
args = parser.parse_args()

subdir = "answer_distribution"
if args.filter_abstentions:
    subdir = "answer_distribution_filtered"
elif args.split_abstentions:
    subdir = "answer_distribution_abstsplit"
OUT_DIR = ROOT / "figures" / subdir
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
YN_COLORS_SPLIT = {
    "yes": "#6BA292",
    "no": "#C97A6A",
    "soft_abst": "#C4956A",
    "hard_abst": "#7A7A7A",
    "others": "#B5B5B5",
}
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
NUM_COLORS_SPLIT = {
    **NUM_COLORS,
    "soft_abst": "#C4956A",
    "hard_abst": "#7A7A7A",
}
NUM_ORDER = ["0", "1", "2–3", "4–5", "6–10", "11–20", ">20", "others"]
NUM_ORDER_SPLIT = ["0", "1", "2–3", "4–5", "6–10", "11–20", ">20", "soft_abst", "hard_abst", "others"]

# Display labels for legend (clean terminology)
DISPLAY_LABELS = {
    "soft_abst": "Soft Abstention",
    "hard_abst": "Hard Abstention",
    "others": "Others",
    "yes": "Yes",
    "no": "No",
}

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
    return series.fillna("").astype(str).apply(lambda x: preprocess_answer(x, strip_think=True)).str.lower()


def is_substantive(text: str) -> bool:
    cleaned = clean_answer(str(text))
    return not is_abstained(classify(cleaned, None))


def abstention_bucket(text: str) -> str | None:
    cleaned = clean_answer(str(text))
    cls = classify(cleaned, None)
    if cls == "soft_abstained":
        return "soft_abst"
    if cls == "hard_abstained":
        return "hard_abst"
    return None


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


def build_stack(df: pd.DataFrame, value_col: str, source_order: list[str],
                ensure_cols: list[str] | None = None) -> pd.DataFrame:
    counts = df.value_counts(["source", value_col]).reset_index(name="count")
    counts["proportion"] = counts.groupby("source")["count"].transform(lambda x: x / x.sum())
    stack = counts.pivot(index="source", columns=value_col, values="proportion").fillna(0)
    stack = stack.reindex(source_order).fillna(0)
    if ensure_cols:
        for col in ensure_cols:
            if col not in stack.columns:
                stack[col] = 0.0
    return stack


def plot_yn(stack_df: pd.DataFrame) -> plt.Figure:
    order = ["yes", "no", "soft_abst", "hard_abst", "others"] if args.split_abstentions else ["yes", "no", "others"]
    color_map = YN_COLORS_SPLIT if args.split_abstentions else YN_COLORS
    cats = [c for c in order if c in stack_df.columns]
    n_rows = stack_df.shape[0]
    fig, ax = plt.subplots(figsize=(7.8, 0.58 * n_rows + 0.8))
    plot_df = stack_df[cats].rename(columns=DISPLAY_LABELS)
    display_cats = [DISPLAY_LABELS.get(c, c) for c in cats]
    plot_df[display_cats].plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[color_map[c] for c in cats],
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
    legend_cols = min(5, len(cats)) if args.split_abstentions else 3
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=legend_cols, frameon=False)
    plt.tight_layout()
    return fig


def plot_num(stack_df: pd.DataFrame) -> plt.Figure:
    base_order = NUM_ORDER_SPLIT if args.split_abstentions else NUM_ORDER
    color_map = NUM_COLORS_SPLIT if args.split_abstentions else NUM_COLORS
    order = [c for c in base_order if c in stack_df.columns]
    n_rows = stack_df.shape[0]
    fig, ax = plt.subplots(figsize=(11.5, 0.58 * n_rows + 0.8))
    plot_df = stack_df[order].rename(columns=DISPLAY_LABELS)
    display_order = [DISPLAY_LABELS.get(c, c) for c in order]
    plot_df[display_order].plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[color_map[c] for c in order],
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
        ncol=5 if args.split_abstentions else len(order),
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
if args.filter_abstentions:
    human_df = human_df[human_df["output"].apply(is_substantive)].copy()

model_rows: list[pd.DataFrame] = []
for condition, export_key in [("blind", "model_blind"), ("inst_blind", "model_inst_blind")]:
    sub = exports[export_key].copy()
    sub = sub[sub["variant"] == "C"].copy()
    sub = sub[sub["question_id"].astype(int).isin(human_qid_set)].copy()
    sub["qid"] = sub["question_id"].astype(int)
    sub["answer_type"] = sub["qid"].map(qid2atype).fillna("other")
    sub["condition"] = condition
    sub["output"] = clean_output(sub["response"])
    if args.filter_abstentions:
        sub = sub[sub["output"].apply(is_substantive)].copy()
    model_rows.append(sub[["qid", "answer_type", "model", "condition", "output"]])

model_df = pd.concat(model_rows, ignore_index=True)

human_yesno_q = int((human_df["answer_type"] == "yes/no").sum() / max(N_HUMANS, 1))
human_number_q = int((human_df["answer_type"] == "number").sum() / max(N_HUMANS, 1))
print(f"Human study: q_yesno={human_yesno_q}, q_number={human_number_q}, participants={N_HUMANS}")

# Build ground truth row from VQA annotations (10 annotators per question)
gt_rows = []
for qid in human_qid_set:
    ann = mapper.annotations.get(int(qid), mapper.annotations.get(str(qid), {}))
    atype = ann.get("answer_type", "other")
    for a in ann.get("answers", []):
        gt_rows.append({
            "qid": int(qid),
            "answer_type": atype,
            "source": "Ground Truth",
            "condition": "inst_blind",
            "output": preprocess_answer(a["answer"], strip_think=False).lower(),
        })
gt_df = pd.DataFrame(gt_rows)

for triplet in TRIPLETS:
    base_llm = VLM_BASE_LLM.get(triplet["vlm"])
    if base_llm != triplet["llm"]:
        print(
            f"Warning: configured LLM for {triplet['vlm']} is {triplet['llm']}, "
            f"but VLM_BASE_LLM says {base_llm}"
        )

    # Build model list — 3 or 4 tiers depending on whether think is present
    model_keys = ["vlm", "decoder", "llm"]
    if "think" in triplet and triplet["think"]:
        model_keys.append("think")
    model_names = [triplet[k] for k in model_keys]

    source_labels = {"Humans": "Humans"}
    source_labels.update({triplet[k]: triplet[k] for k in model_keys})

    for condition in CONDITIONS:
        subset = model_df[
            (model_df["condition"] == condition)
            & (model_df["model"].isin(model_names))
        ].copy()
        subset["source"] = subset["model"]

        base_df = pd.concat(
            [
                human_df[["qid", "answer_type", "source", "condition", "output"]],
                gt_df[["qid", "answer_type", "source", "condition", "output"]],
                subset[["qid", "answer_type", "source", "condition", "output"]],
            ],
            ignore_index=True,
        )

        row_order = ["Humans", "Ground Truth"] + [triplet[k] for k in model_keys]
        source_map = {"Humans": "Humans", "Ground Truth": "Ground Truth"}
        source_map.update({triplet[k]: triplet[k] for k in model_keys})

        # Yes/No
        yesno_df = base_df[base_df["answer_type"] == "yes/no"].copy()
        yesno_df["source"] = yesno_df["source"].map(source_map)
        non_model = {"Humans", "Ground Truth"}
        if args.split_abstentions:
            yesno_df["yn"] = yesno_df.apply(
                lambda r: (abstention_bucket(r["output"]) if r["source"] not in non_model else None) or classify_yn(r["output"]),
                axis=1,
            )
        else:
            yesno_df["yn"] = yesno_df["output"].apply(classify_yn)
        yn_ensure = ["yes", "no", "soft_abst", "hard_abst", "others"] if args.split_abstentions else None
        yn_stack = build_stack(yesno_df, "yn", row_order, ensure_cols=yn_ensure)
        save(
            plot_yn(yn_stack),
            f"{COND_PREFIX[condition]}_{triplet['slug']}_yesno"
            f"{'_filtered' if args.filter_abstentions else ''}"
            f"{'_abstsplit' if args.split_abstentions else ''}.png",
        )

        # Number
        number_df = base_df[base_df["answer_type"] == "number"].copy()
        number_df["source"] = number_df["source"].map(source_map)
        if args.split_abstentions:
            number_df["number_bin"] = number_df.apply(
                lambda r: (abstention_bucket(r["output"]) if r["source"] not in non_model else None) or number_bin(r["output"]),
                axis=1,
            )
        else:
            number_df["number_bin"] = number_df["output"].apply(number_bin)
        num_ensure = NUM_ORDER_SPLIT if args.split_abstentions else None
        num_stack = build_stack(number_df, "number_bin", row_order, ensure_cols=num_ensure)
        save(
            plot_num(num_stack),
            f"{COND_PREFIX[condition]}_{triplet['slug']}_number"
            f"{'_filtered' if args.filter_abstentions else ''}"
            f"{'_abstsplit' if args.split_abstentions else ''}.png",
        )

# ── Combined all-models stacked bar plots ──────────────────────────────────
# Group ordering: Humans → Ground Truth → VLMs → Backbone Decoders → Standalone LLMs
# Each group gets a distinct background band color for visual separation.

GROUP_ORDER = ["VLM", "VLM backbone decoder", "standalone LLM"]
GROUP_BAND_COLORS = {
    "VLM": "#E8EAF6",                   # light indigo
    "VLM backbone decoder": "#FFF3E0",   # light orange
    "standalone LLM": "#E8F5E9",         # light green
}


def _model_sort_key(model: str):
    family = MODEL_FAMILY.get(model, model)
    size = float(MODEL_SIZE_B.get(model, 999.0))
    return (family.lower(), size, model.lower())


def plot_combined_yn(stack_df: pd.DataFrame, group_spans: dict) -> plt.Figure:
    """Yes/no stacked bars for all models with group background bands."""
    order = ["yes", "no", "soft_abst", "hard_abst", "others"] if args.split_abstentions else ["yes", "no", "others"]
    color_map = YN_COLORS_SPLIT if args.split_abstentions else YN_COLORS
    cats = [c for c in order if c in stack_df.columns]
    n_rows = stack_df.shape[0]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * n_rows + 1.2))
    plot_df = stack_df[cats].rename(columns=DISPLAY_LABELS)
    display_cats = [DISPLAY_LABELS.get(c, c) for c in cats]
    plot_df[display_cats].plot(
        kind="barh", stacked=True, ax=ax,
        color=[color_map[c] for c in cats], width=0.75,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    # Group background bands
    for grp, (start, end) in group_spans.items():
        if grp in GROUP_BAND_COLORS:
            ax.axhspan(start - 0.5, end + 0.5, color=GROUP_BAND_COLORS[grp], alpha=0.3, zorder=0)
    # Percentage labels
    for i, src in enumerate(stack_df.index):
        cum = 0
        for cat in cats:
            val = stack_df.loc[src, cat]
            if val > 0.04:
                ax.text(cum + val / 2, i, f"{val * 100:.0f}%",
                        ha="center", va="center", color="white",
                        fontsize=8, fontweight="bold")
            cum += val
    legend_cols = min(5, len(cats)) if args.split_abstentions else 3
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=legend_cols, frameon=False, fontsize=9)
    plt.tight_layout()
    return fig


def plot_combined_num(stack_df: pd.DataFrame, group_spans: dict) -> plt.Figure:
    """Number stacked bars for all models with group background bands."""
    base_order = NUM_ORDER_SPLIT if args.split_abstentions else NUM_ORDER
    color_map = NUM_COLORS_SPLIT if args.split_abstentions else NUM_COLORS
    order = [c for c in base_order if c in stack_df.columns]
    n_rows = stack_df.shape[0]
    fig, ax = plt.subplots(figsize=(12, 0.42 * n_rows + 1.2))
    plot_df = stack_df[order].rename(columns=DISPLAY_LABELS)
    display_order = [DISPLAY_LABELS.get(c, c) for c in order]
    plot_df[display_order].plot(
        kind="barh", stacked=True, ax=ax,
        color=[color_map[c] for c in order], width=0.75,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for grp, (start, end) in group_spans.items():
        if grp in GROUP_BAND_COLORS:
            ax.axhspan(start - 0.5, end + 0.5, color=GROUP_BAND_COLORS[grp], alpha=0.3, zorder=0)
    for i, src in enumerate(stack_df.index):
        cum = 0
        for cat in order:
            val = stack_df.loc[src, cat]
            if val > 0.04:
                ax.text(cum + val / 2, i, f"{val * 100:.0f}%",
                        ha="center", va="center", color="white",
                        fontsize=8, fontweight="bold")
            cum += val
    ax.legend(
        bbox_to_anchor=(0.5, -0.04), loc="upper center",
        ncol=5 if args.split_abstentions else len(order),
        frameon=False, fontsize=9,
    )
    plt.tight_layout()
    return fig


print("\n── Combined all-models plots ──")
for condition in CONDITIONS:
    cond_models = model_df[model_df["condition"] == condition]["model"].unique()
    # Build ordered model list: group by MODEL_GROUP, sort within group
    grouped_models = {}
    for grp in GROUP_ORDER:
        grp_models = sorted(
            [m for m in cond_models if MODEL_GROUP.get(m, "standalone LLM") == grp],
            key=_model_sort_key,
        )
        if grp_models:
            grouped_models[grp] = grp_models

    # Build source order: Humans, Ground Truth, then models by group
    source_order = ["Humans", "Ground Truth"]
    group_spans = {}  # group → (start_idx, end_idx) for background bands
    idx = len(source_order)
    for grp, models in grouped_models.items():
        group_spans[grp] = (idx, idx + len(models) - 1)
        source_order.extend(models)
        idx += len(models)

    # Build combined df
    model_sub = model_df[
        (model_df["condition"] == condition)
        & (model_df["model"].isin(set().union(*grouped_models.values())))
    ].copy()
    model_sub["source"] = model_sub["model"]

    combined_df = pd.concat([
        human_df[["qid", "answer_type", "source", "condition", "output"]],
        gt_df[["qid", "answer_type", "source", "condition", "output"]],
        model_sub[["qid", "answer_type", "source", "condition", "output"]],
    ], ignore_index=True)

    # Yes/No
    non_model_combined = {"Humans", "Ground Truth"}
    yn_df = combined_df[combined_df["answer_type"] == "yes/no"].copy()
    if args.split_abstentions:
        yn_df["yn"] = yn_df.apply(
            lambda r: (abstention_bucket(r["output"]) if r["source"] not in non_model_combined else None) or classify_yn(r["output"]),
            axis=1,
        )
    else:
        yn_df["yn"] = yn_df["output"].apply(classify_yn)
    yn_ensure = ["yes", "no", "soft_abst", "hard_abst", "others"] if args.split_abstentions else None
    yn_stack = build_stack(yn_df, "yn", source_order, ensure_cols=yn_ensure)
    save(
        plot_combined_yn(yn_stack, group_spans),
        f"{COND_PREFIX[condition]}_combined_yesno"
        f"{'_filtered' if args.filter_abstentions else ''}"
        f"{'_abstsplit' if args.split_abstentions else ''}.png",
    )

    # Number
    num_df = combined_df[combined_df["answer_type"] == "number"].copy()
    if args.split_abstentions:
        num_df["number_bin"] = num_df.apply(
            lambda r: (abstention_bucket(r["output"]) if r["source"] not in non_model_combined else None) or number_bin(r["output"]),
            axis=1,
        )
    else:
        num_df["number_bin"] = num_df["output"].apply(number_bin)
    num_ensure = NUM_ORDER_SPLIT if args.split_abstentions else None
    num_stack = build_stack(num_df, "number_bin", source_order, ensure_cols=num_ensure)
    save(
        plot_combined_num(num_stack, group_spans),
        f"{COND_PREFIX[condition]}_combined_number"
        f"{'_filtered' if args.filter_abstentions else ''}"
        f"{'_abstsplit' if args.split_abstentions else ''}.png",
    )

print("\nDone. All figures saved to:", OUT_DIR)
