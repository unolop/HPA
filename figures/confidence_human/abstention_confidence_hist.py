from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from analysis.utils.abstention import classify

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "confidence"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

EXPORTS = ROOT / "analysis" / "session2" / "exports"
RESPONSES = EXPORTS / "responses_model_inst_blind.csv"
SEMANTICS = ROOT / "dataset" / "vqa" / "vqa1k_semantics.jsonl"

CT_TO_VARIANT = {"question": "C", "weaker_object": "B", "pronominalized": "A"}
EOS_TOKENS = {"<|im_end|>", "</s>", "<eos>", "<|endoftext|>", "<|end|>"}

VLM_DIR_TO_MODEL = {
    "InternVL3_5-1B": ("InternVL-1B", "VLM"),
    "InternVL3_5-2B": ("InternVL-2B", "VLM"),
    "InternVL3_5-8B": ("InternVL-8B", "VLM"),
    "llava-1.5-7b-hf": ("LLaVA-1.5-7B", "VLM"),
    "llava-v1.6-mistral-7b-hf": ("LLaVA-Mistral", "VLM"),
    "llava-v1.6-vicuna-7b-hf": ("LLaVA-Vicuna", "VLM"),
    "llava-v1.6-vicuna-13b-hf": ("LLaVA-Vicuna-13B", "VLM"),
    "Qwen3-VL-2B-Instruct": ("Qwen3-VL-2B", "VLM"),
    "Qwen3-VL-4B-Instruct": ("Qwen3-VL-4B", "VLM"),
    "Qwen3-VL-8B-Instruct": ("Qwen3-VL-8B", "VLM"),
    "Qwen3-VL-32B-Instruct": ("Qwen3-VL-32B", "VLM"),
}
LM_DECODER_DIR_TO_MODEL = {
    "InternVL3_5-1B": ("InternVL-1B (LM)", "Backbone decoder"),
    "InternVL3_5-2B": ("InternVL-2B (LM)", "Backbone decoder"),
    "InternVL3_5-8B": ("InternVL-8B (LM)", "Backbone decoder"),
    "llava-1.5-7b-hf": ("LLaVA-1.5 (LM)", "Backbone decoder"),
    "llava-v1.6-mistral-7b-hf": ("LLaVA-Mistral (LM)", "Backbone decoder"),
    "llava-v1.6-vicuna-7b-hf": ("LLaVA-Vicuna (LM)", "Backbone decoder"),
    "llava-v1.6-vicuna-13b-hf": ("LLaVA-Vicuna-13B (LM)", "Backbone decoder"),
    "Qwen3-VL-2B-Instruct": ("Qwen3-VL-2B (LM)", "Backbone decoder"),
    "Qwen3-VL-4B-Instruct": ("Qwen3-VL-4B (LM)", "Backbone decoder"),
    "Qwen3-VL-8B-Instruct": ("Qwen3-VL-8B (LM)", "Backbone decoder"),
    "Qwen3-VL-32B-Instruct": ("Qwen3-VL-32B (LM)", "Backbone decoder"),
}
BACKBONE_DIR_TO_MODEL = {
    "Mistral-7B-Instruct-v0.2": ("Mistral-7B", "Standalone LLM"),
    "Phi-3.5-mini-instruct": ("Phi-3.5-mini", "Standalone LLM"),
    "Qwen2.5-7B-Instruct": ("Qwen2.5-7B-Instruct", "Standalone LLM"),
    "Qwen3-0.6B": ("Qwen3-0.6B", "Standalone LLM"),
    "Qwen3-1.7B": ("Qwen3-1.7B", "Standalone LLM"),
    "Qwen3-4B": ("Qwen3-4B", "Standalone LLM"),
    "Qwen3-8B": ("Qwen3-8B", "Standalone LLM"),
    "Qwen3-32B": ("Qwen3-32B", "Standalone LLM"),
    "vicuna-7b-v1.5": ("Vicuna-7B", "Standalone LLM"),
    "vicuna-13b-v1.5": ("Vicuna-13B", "Standalone LLM"),
}
BACKBONE_THINK_DIR_TO_MODEL = {
    "Qwen3-0.6B_think": ("Qwen3-0.6B (think)", "Think"),
    "Qwen3-1.7B_think": ("Qwen3-1.7B (think)", "Think"),
    "Qwen3-4B_think": ("Qwen3-4B (think)", "Think"),
    "Qwen3-8B_think": ("Qwen3-8B (think)", "Think"),
    "Qwen3-32B_think": ("Qwen3-32B (think)", "Think"),
}

LOGIT_SOURCES = [
    (ROOT / "evaluation" / "logits" / "vlm" / "pretrained", VLM_DIR_TO_MODEL),
    (ROOT / "evaluation" / "logits" / "lm_decoder" / "pretrained", LM_DECODER_DIR_TO_MODEL),
    (ROOT / "evaluation" / "logits" / "backbone" / "pretrained", BACKBONE_DIR_TO_MODEL),
    (ROOT / "evaluation" / "logits" / "backbone" / "pretrained", BACKBONE_THINK_DIR_TO_MODEL),
]


def save(fig: plt.Figure, name: str) -> None:
    out = OUT_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_DIR / name)
    plt.close(fig)
    print(f"saved {out}")


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(RESPONSES)
    df["abstention_class"] = df["response"].fillna("").astype(str).map(lambda x: classify(x, None))
    df["abstention_group"] = np.where(
        df["abstention_class"].isin(["hard_abstained", "soft_abstained"]),
        "Abstained",
        "Committed",
    )
    return df[["question_id", "model", "variant", "ent", "op", "abstention_class", "abstention_group"]]


def load_confidence() -> pd.DataFrame:
    sem_df = pd.read_json(SEMANTICS, lines=True)[["question_id", "ent", "op"]]
    rows: list[dict] = []
    for base_dir, mapping in LOGIT_SOURCES:
        for dir_name, (display, model_group) in mapping.items():
            fpath = base_dir / dir_name / "vqa_1k_control_inst_blind.jsonl"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    qid = ex["question_id"]
                    logits = ex.get("generated_logits", {})
                    for ct, variant in CT_TO_VARIANT.items():
                        logit_data = logits.get(ct)
                        if not logit_data:
                            continue
                        probs = [
                            np.exp(tok["logprob"])
                            for tok in logit_data.get("content", [])
                            if tok.get("token") not in EOS_TOKENS
                        ]
                        if not probs:
                            continue
                        rows.append(
                            {
                                "question_id": qid,
                                "model": display,
                                "model_group": model_group,
                                "variant": variant,
                                "confidence": float(np.mean(probs)),
                            }
                        )
    return pd.DataFrame(rows).merge(sem_df, on="question_id", how="left")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("abstention_group")["confidence"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary["pct_of_rows"] = summary["count"] / len(df)
    return summary


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    order = ["hallucinated_correct", "hallucinated_wrong", "soft_abstained", "hard_abstained", "degenerate"]
    summary = (
        df.groupby("abstention_class")["confidence"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary["pct_of_rows"] = summary["count"] / len(df)
    summary["abstention_class"] = pd.Categorical(summary["abstention_class"], categories=order, ordered=True)
    return summary.sort_values("abstention_class")


def summarize_by_group(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["model_group", "abstention_group"])["confidence"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    return summary


MODELS_7B = {
    "InternVL-8B", "InternVL-8B (LM)",
    "LLaVA-1.5-7B", "LLaVA-1.5 (LM)",
    "LLaVA-Mistral", "LLaVA-Mistral (LM)",
    "LLaVA-Vicuna", "LLaVA-Vicuna (LM)",
    "Qwen3-VL-8B", "Qwen3-VL-8B (LM)",
    "Mistral-7B", "Qwen2.5-7B-Instruct", "Qwen3-8B", "Vicuna-7B",
    "Qwen3-8B (think)",
}


def plot_committed_vs_abstained(df: pd.DataFrame, suffix: str = "", title_suffix: str = "") -> None:
    bins = np.linspace(0.0, 1.0, 31)
    colors = {"Committed": "#2a6fbb", "Abstained": "#c23b22"}

    committed = df.loc[df["abstention_group"] == "Committed", "confidence"].to_numpy()
    abstained = df.loc[df["abstention_group"] == "Abstained", "confidence"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.0, 3.0),
                           gridspec_kw={"left": 0.12, "right": 0.98, "bottom": 0.13, "top": 0.92})
    ax.hist(committed, bins=bins, density=True, alpha=0.55, color=colors["Committed"], label=f"Committed (n={len(committed):,})")
    ax.hist(abstained, bins=bins, density=True, alpha=0.55, color=colors["Abstained"], label=f"Abstained (n={len(abstained):,})")
    ax.axvline(np.median(committed), color=colors["Committed"], linestyle="--", linewidth=1.3)
    ax.axvline(np.median(abstained), color=colors["Abstained"], linestyle="--", linewidth=1.3)
    ax.set_title(f"Committed vs abstained{title_suffix}")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean token probability")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    save(fig, f"abstention_committed_vs_abstained_inst_blind{suffix}.png")


def plot_type_split(df: pd.DataFrame, suffix: str = "", title_suffix: str = "") -> None:
    bins = np.linspace(0.0, 1.0, 31)
    colors = {"Committed": "#2a6fbb", "soft_abstained": "#f28e2b", "hard_abstained": "#8c1d18"}

    committed = df.loc[df["abstention_group"] == "Committed", "confidence"].to_numpy()
    soft = df.loc[df["abstention_class"] == "soft_abstained", "confidence"].to_numpy()
    hard = df.loc[df["abstention_class"] == "hard_abstained", "confidence"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.0, 3.0),
                           gridspec_kw={"left": 0.12, "right": 0.98, "bottom": 0.13, "top": 0.92})
    ax.hist(committed, bins=bins, density=True, histtype="step", linewidth=1.8, color=colors["Committed"], label="Committed")
    ax.hist(soft, bins=bins, density=True, alpha=0.50, color=colors["soft_abstained"], label=f"Soft abst. (n={len(soft):,})")
    ax.hist(hard, bins=bins, density=True, alpha=0.45, color=colors["hard_abstained"], label=f"Hard abst. (n={len(hard):,})")
    ax.axvline(np.median(soft), color=colors["soft_abstained"], linestyle="--", linewidth=1.2)
    ax.axvline(np.median(hard), color=colors["hard_abstained"], linestyle="--", linewidth=1.2)
    ax.set_title(f"Abstention type split{title_suffix}")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean token probability")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    save(fig, f"abstention_vs_confidence_hist_inst_blind{suffix}.png")


def main() -> None:
    labels = load_labels()
    conf = load_confidence()
    merged = conf.merge(labels, on=["question_id", "model", "variant", "ent", "op"], how="inner")

    overall = summarize(merged)
    by_class = summarize_by_class(merged)
    by_group = summarize_by_group(merged)

    overall.to_csv(OUT_DIR / "abstention_vs_confidence_summary_inst_blind.csv", index=False)
    by_class.to_csv(OUT_DIR / "abstention_vs_confidence_by_class_inst_blind.csv", index=False)
    by_group.to_csv(OUT_DIR / "abstention_vs_confidence_by_group_inst_blind.csv", index=False)

    shutil.copy(OUT_DIR / "abstention_vs_confidence_summary_inst_blind.csv", LATEX_DIR / "abstention_vs_confidence_summary_inst_blind.csv")
    shutil.copy(OUT_DIR / "abstention_vs_confidence_by_class_inst_blind.csv", LATEX_DIR / "abstention_vs_confidence_by_class_inst_blind.csv")
    shutil.copy(OUT_DIR / "abstention_vs_confidence_by_group_inst_blind.csv", LATEX_DIR / "abstention_vs_confidence_by_group_inst_blind.csv")

    print("\nOverall:")
    print(overall.to_string(index=False))
    print("\nBy class:")
    print(by_class.to_string(index=False))
    print("\nBy group:")
    print(by_group.to_string(index=False))

    plot_committed_vs_abstained(merged)
    plot_type_split(merged)
    merged_7b = merged[merged["model"].isin(MODELS_7B)]
    plot_committed_vs_abstained(merged_7b, suffix="_7b", title_suffix=" (7/8B)")
    plot_type_split(merged_7b, suffix="_7b", title_suffix=" (7/8B)")


if __name__ == "__main__":
    main()
