from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from utils.vqa import VQAAnswerMapper, vqa_accuracy  # noqa: E402
from helpers import load_cleaned_pair_cache  # noqa: E402

THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
VARIANT_TO_CT = {"C": "question", "B": "weaker_object", "A": "pronominalized"}
VARIANT_ORDER = ["C", "B", "A"]
VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
VARIANT_COLORS = {"C": "#4C78A8", "B": "#F58518", "A": "#54A24B"}
CONDITIONS = {
    "blind": "vqa_1k_control_blind.jsonl",
    "inst_blind": "vqa_1k_control_inst_blind.jsonl",
}
CONDITION_LABELS = {"blind": "Blind", "inst_blind": "Inst-Blind"}

MODEL_PAIRS = [
    ("Qwen3-0.6B", "Qwen3-0.6B (think)", 0.6),
    ("Qwen3-1.7B", "Qwen3-1.7B (think)", 1.7),
    ("Qwen3-4B", "Qwen3-4B (think)", 4.0),
    ("Qwen3-8B", "Qwen3-8B (think)", 8.0),
    ("Qwen3-32B", "Qwen3-32B (think)", 32.0),
]

THINK_JSONL = {
    "Qwen3-0.6B (think)": ROOT / "evaluation/logits/backbone/pretrained/Qwen3-0.6B_think",
    "Qwen3-1.7B (think)": ROOT / "evaluation/logits/backbone/pretrained/Qwen3-1.7B_think",
    "Qwen3-4B (think)": ROOT / "evaluation/logits/backbone/pretrained/Qwen3-4B_think",
    "Qwen3-8B (think)": ROOT / "evaluation/logits/backbone/pretrained/Qwen3-8B_think",
    "Qwen3-32B (think)": ROOT / "evaluation/logits/backbone/pretrained/Qwen3-32B_think",
}

NOTHINK_JSONL = {
    "Qwen3-0.6B": ROOT / "evaluation/logits/backbone/pretrained/Qwen3-0.6B",
    "Qwen3-1.7B": ROOT / "evaluation/logits/backbone/pretrained/Qwen3-1.7B",
    "Qwen3-4B": ROOT / "evaluation/logits/backbone/pretrained/Qwen3-4B",
    "Qwen3-8B": ROOT / "evaluation/logits/backbone/pretrained/Qwen3-8B",
    "Qwen3-32B": ROOT / "evaluation/logits/backbone/pretrained/Qwen3-32B",
}

OUT_DIR = ROOT / "figures" / "think_eda"
OUT_STEM = "think_delta_vs_length_with_blind"


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def extract_think(text: str) -> str:
    match = THINK_RE.search(text)
    return match.group(1).strip() if match else ""


def load_metrics_by_variant(jsonl_path: Path, *, strip_think_tags: bool) -> dict[str, dict[str, float]]:
    mapper = VQAAnswerMapper()
    scores: dict[str, list[float]] = {v: [] for v in VARIANT_ORDER}
    confidences: dict[str, list[float]] = {v: [] for v in VARIANT_ORDER}
    think_lengths: dict[str, list[int]] = {v: [] for v in VARIANT_ORDER}
    with open(jsonl_path) as handle:
        for line in handle:
            ex = json.loads(line)
            qid = ex["question_id"]
            gt = mapper.get_answers(qid)
            answers = ex.get("generated_answers", {})
            logits = ex.get("generated_logits", {})
            for variant, ct in VARIANT_TO_CT.items():
                raw_ans = answers.get(ct, "")
                if strip_think_tags:
                    think_text = extract_think(raw_ans)
                    think_lengths[variant].append(len(think_text.split()) if think_text.strip() else 0)
                ans = raw_ans
                if strip_think_tags:
                    ans = strip_think(ans)
                ans = ans.strip()
                if ans and gt:
                    scores[variant].append(vqa_accuracy(ans, gt))
                lp_dict = logits.get(ct, {})
                token_list = lp_dict.get("content", []) if isinstance(lp_dict, dict) else []
                lp_values = [tok["logprob"] for tok in token_list if isinstance(tok, dict) and "logprob" in tok]
                if not lp_values:
                    continue
                if strip_think_tags:
                    final_lps = []
                    in_think = True
                    has_closed_think = "</think>" in answers.get(ct, "")
                    if has_closed_think:
                        for tok in token_list:
                            if not isinstance(tok, dict) or "logprob" not in tok:
                                continue
                            if in_think:
                                if "</think>" in tok.get("token", ""):
                                    in_think = False
                            else:
                                final_lps.append(tok["logprob"])
                    conf = float(np.mean(final_lps)) if final_lps else float(np.mean(lp_values))
                else:
                    conf = float(np.mean(lp_values))
                confidences[variant].append(conf)
    return {
        variant: {
            "accuracy": sum(scores[variant]) / len(scores[variant]) if scores[variant] else np.nan,
            "mean_confidence": float(np.mean(confidences[variant])) if confidences[variant] else np.nan,
            "mean_think_words": float(np.mean(think_lengths[variant])) if think_lengths[variant] else np.nan,
        }
        for variant in VARIANT_ORDER
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hm_frames = {}
    for condition in CONDITIONS:
        hm_cache = load_cleaned_pair_cache(ROOT, condition=condition, include_yesno=True, verbose=False)
        hm_sub = hm_cache[
            (hm_cache["pair_type"] == "HM")
            & (hm_cache["subject_2"].isin([m for pair in MODEL_PAIRS for m in pair[:2]]))
        ].copy()
        hm_frames[condition] = (
            hm_sub.groupby(["subject_2", "variant"], as_index=False)["sbert_score"]
            .mean()
            .rename(columns={"subject_2": "model", "sbert_score": "hm_sbert"})
        )

    acc_rows = []
    for base_model, think_model, size in MODEL_PAIRS:
        for condition, filename in CONDITIONS.items():
            think_metrics = load_metrics_by_variant(THINK_JSONL[think_model] / filename, strip_think_tags=True)
            base_metrics = load_metrics_by_variant(NOTHINK_JSONL[base_model] / filename, strip_think_tags=False)
            for variant in VARIANT_ORDER:
                acc_rows.append(
                    {
                        "size_b": size,
                        "base_model": base_model,
                        "think_model": think_model,
                        "condition": condition,
                        "variant": variant,
                        "mean_think_words": think_metrics[variant]["mean_think_words"],
                        "think_accuracy": think_metrics[variant]["accuracy"],
                        "base_accuracy": base_metrics[variant]["accuracy"],
                        "think_confidence": think_metrics[variant]["mean_confidence"],
                        "base_confidence": base_metrics[variant]["mean_confidence"],
                    }
                )
    acc_df = pd.DataFrame(acc_rows)
    rows = []
    for base_model, think_model, size in MODEL_PAIRS:
        for variant in VARIANT_ORDER:
            for condition in CONDITIONS:
                hm_mean = hm_frames[condition]
                hm_think = hm_mean.rename(columns={"model": "think_model", "hm_sbert": "think_hm_sbert"})
                hm_base = hm_mean.rename(columns={"model": "base_model", "hm_sbert": "base_hm_sbert"})
                think_hm = hm_think[(hm_think["think_model"] == think_model) & (hm_think["variant"] == variant)]
                base_hm = hm_base[(hm_base["base_model"] == base_model) & (hm_base["variant"] == variant)]
                acc_row = acc_df[
                    (acc_df["base_model"] == base_model)
                    & (acc_df["think_model"] == think_model)
                    & (acc_df["condition"] == condition)
                    & (acc_df["variant"] == variant)
                ]
                rows.append(
                    {
                        "size_b": size,
                        "size_label": f"{size:g}B",
                        "base_model": base_model,
                        "think_model": think_model,
                        "condition": condition,
                        "variant": variant,
                        "mean_think_words": float(acc_row["mean_think_words"].iloc[0]),
                        "delta_hm_sbert": float(think_hm["think_hm_sbert"].iloc[0] - base_hm["base_hm_sbert"].iloc[0]),
                        "delta_accuracy": float(acc_row["think_accuracy"].iloc[0] - acc_row["base_accuracy"].iloc[0]),
                        "delta_confidence": float(acc_row["think_confidence"].iloc[0] - acc_row["base_confidence"].iloc[0]),
                        "think_hm_sbert": float(think_hm["think_hm_sbert"].iloc[0]),
                        "base_hm_sbert": float(base_hm["base_hm_sbert"].iloc[0]),
                        "think_accuracy": float(acc_row["think_accuracy"].iloc[0]),
                        "base_accuracy": float(acc_row["base_accuracy"].iloc[0]),
                        "think_confidence": float(acc_row["think_confidence"].iloc[0]),
                        "base_confidence": float(acc_row["base_confidence"].iloc[0]),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["size_b", "condition", "variant"])
    df.to_csv(OUT_DIR / f"{OUT_STEM}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharex=True)
    panel_specs = [
        ("delta_hm_sbert", r"$\Delta$ HM SBERT (think $-$ no-think)"),
        ("delta_accuracy", r"$\Delta$ Accuracy (think $-$ no-think)"),
        ("delta_confidence", r"$\Delta$ Mean answer logprob (think $-$ no-think)"),
    ]

    for ax, (metric, ylabel) in zip(axes, panel_specs):
        for (size, condition), g in df.groupby(["size_b", "condition"]):
            g = g.sort_values("variant", key=lambda s: s.map({v: i for i, v in enumerate(VARIANT_ORDER)}))
            ax.plot(
                g["mean_think_words"],
                g[metric],
                color="#C7C7C7",
                linewidth=1.0,
                zorder=1,
            )
        for _, row in df.iterrows():
            ax.scatter(
                row["mean_think_words"],
                row[metric],
                s=56,
                facecolors=VARIANT_COLORS[row["variant"]] if row["condition"] == "inst_blind" else "none",
                edgecolor=VARIANT_COLORS[row["variant"]],
                linewidth=0.7,
                zorder=3,
            )
            ax.annotate(
                row["size_label"],
                (row["mean_think_words"], row[metric]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#333333",
            )
        ax.axhline(0, color="#999999", linestyle="--", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
        ax.grid(axis="x", alpha=0.08)

    axes[0].set_title("Does more visible thinking improve alignment?")
    axes[1].set_title("Does it improve task accuracy?")
    axes[2].set_title("Are final answers more confident?")
    for ax in axes:
        ax.set_xlabel("Mean think words")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=VARIANT_COLORS[v], markeredgecolor="white",
               markeredgewidth=0.7, markersize=7, label=VARIANT_LABELS[v])
        for v in VARIANT_ORDER
    ]
    legend_handles += [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="#666666",
               markeredgewidth=0.8, markersize=7, label="Inst-Blind"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#666666",
               markeredgewidth=1.0, markersize=7, label="Blind"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / f"{OUT_STEM}.png", dpi=180, bbox_inches="tight")

    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {OUT_DIR / f'{OUT_STEM}.png'}")
    print(f"Saved: {OUT_DIR / f'{OUT_STEM}.csv'}")


if __name__ == "__main__":
    main()
