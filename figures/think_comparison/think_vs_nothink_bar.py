"""
Bar chart comparing Qwen3 think vs non-think across metrics:
SBERT (HM), Accuracy, chrF, and Entropy (first-token top-5 logprob entropy).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy

SCRIPT_DIR = Path(__file__).resolve().parent        # figures/think_comparison/
FIGURES_DIR = SCRIPT_DIR.parent                       # figures/
REPO = FIGURES_DIR.parent                             # repo root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from figures.helpers import save_fig

EXPORTS = REPO / "analysis/session2/exports"
LOGIT_DIR = REPO / "evaluation/logits/backbone/pretrained"
OUT_DIR = Path(__file__).resolve().parent

SIZES = ["0.6B", "1.7B", "4B", "8B", "32B"]
CONDITION = "vqa_1k_control_inst_blind.jsonl"


# ── Data loading (per-question level) ────────────────────────────────────────

def load_accuracy_perq() -> pd.DataFrame:
    """Per-(model, question) accuracy from responses_model_inst_blind."""
    df = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")
    df = df[df["model"].str.startswith("Qwen3-") & ~df["model"].str.contains("VL")]
    return df[["model", "question_id", "accuracy"]].copy()


def load_hm_sbert_chrf_perq() -> pd.DataFrame:
    """Per-(model, question) mean HM SBERT and chrF from pair_cache."""
    pc = pd.read_parquet(EXPORTS / "pair_cache.parquet")
    hm = pc[
        ((pc["subject_group_1"] == "human") & (pc["subject_group_2"] != "human"))
        | ((pc["subject_group_2"] == "human") & (pc["subject_group_1"] != "human"))
    ].copy()
    hm["model"] = np.where(
        hm["subject_group_1"] == "human", hm["subject_2"], hm["subject_1"]
    )
    hm = hm[hm["model"].str.startswith("Qwen3-") & ~hm["model"].str.contains("VL")]
    # Average over human raters per (model, question)
    return (
        hm.groupby(["model", "question_id"])[["sbert_score", "chrf_score"]]
        .mean()
        .reset_index()
        .rename(columns={"sbert_score": "sbert", "chrf_score": "chrf"})
    )


def load_entropy_perq() -> pd.DataFrame:
    """Per-(model, question) first-token top-5 entropy (bits)."""
    rows = []
    for size in SIZES:
        for think in [False, True]:
            folder = f"Qwen3-{size}_think" if think else f"Qwen3-{size}"
            model_name = f"Qwen3-{size} (think)" if think else f"Qwen3-{size}"
            path = LOGIT_DIR / folder / CONDITION
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    qid = rec.get("question_id")
                    logits = rec.get("generated_logits", {}).get("question", {})
                    tokens = logits.get("content", [])
                    if not tokens:
                        continue
                    top5 = tokens[0].get("top_logprobs", [])
                    if not top5:
                        continue
                    probs = np.exp([t["logprob"] for t in top5])
                    probs = probs / probs.sum()
                    rows.append({
                        "model": model_name,
                        "question_id": qid,
                        "entropy": float(sp_entropy(probs, base=2)),
                    })
    return pd.DataFrame(rows)


# ── Aggregation ──────────────────────────────────────────────────────────────

def build_comparison_df() -> pd.DataFrame:
    """Build per-question long-form dataframe with all metrics."""
    acc = load_accuracy_perq()
    hm = load_hm_sbert_chrf_perq()
    ent = load_entropy_perq()

    df = acc.merge(hm, on=["model", "question_id"], how="outer")
    df = df.merge(ent, on=["model", "question_id"], how="outer")

    df["think"] = df["model"].str.contains("(think)", regex=False)
    df["size"] = (
        df["model"]
        .str.extract(r"Qwen3-(\S+?)(?:\s|$)")[0]
        .str.replace("(think)", "", regex=False)
    )
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-model mean and SEM for each metric."""
    metrics = ["sbert", "accuracy", "chrf", "entropy"]
    records = []
    for model, mdf in df.groupby("model"):
        row = {"model": model, "think": mdf["think"].iloc[0], "size": mdf["size"].iloc[0]}
        for m in metrics:
            vals = mdf[m].dropna()
            row[m] = vals.mean() if len(vals) else np.nan
            row[f"{m}_sem"] = vals.sem() if len(vals) > 1 else 0.0
        records.append(row)
    return pd.DataFrame(records).sort_values("size")


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_bars(agg: pd.DataFrame) -> None:
    metrics = ["sbert", "accuracy", "chrf", "entropy"]
    metric_labels = {
        "sbert": "SBERT (HM)", "accuracy": "Accuracy",
        "chrf": "chrF", "entropy": "Entropy (bits)",
    }

    sizes = sorted(agg["size"].unique(), key=lambda s: float(s.replace("B", "")))
    x = np.arange(len(sizes))
    n_metrics = len(metrics)
    total_bar_width = 0.72
    bar_w = total_bar_width / (n_metrics * 2)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    hatches = ["", "//"]

    for mi, metric in enumerate(metrics):
        for ti, (think_val, think_label, hatch) in enumerate(
            [(False, "no-think", hatches[0]), (True, "think", hatches[1])]
        ):
            vals, errs = [], []
            for size in sizes:
                row = agg[(agg["size"] == size) & (agg["think"] == think_val)]
                if len(row) > 0 and pd.notna(row[metric].values[0]):
                    vals.append(row[metric].values[0])
                    errs.append(row[f"{metric}_sem"].values[0])
                else:
                    vals.append(0)
                    errs.append(0)

            offset = (mi * 2 + ti - (n_metrics * 2 - 1) / 2) * bar_w
            ax.bar(
                x + offset,
                vals,
                yerr=errs,
                width=bar_w * 0.92,
                color=colors[mi],
                alpha=0.85 if not think_val else 0.55,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.5,
                capsize=2,
                error_kw={"lw": 0.8, "capthick": 0.8},
                label=f"{metric_labels[metric]} ({think_label})",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Qwen3-{s}" for s in sizes], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Qwen3 SA-LLM: think vs no-think across metrics", fontsize=13)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        fontsize=8,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, OUT_DIR, "qwen3_think_vs_nothink_metrics.png")
    plt.close(fig)


def plot_delta(perq: pd.DataFrame) -> None:
    """Bar chart showing delta (think − nothink) per metric per size, with error bars."""
    metrics = ["sbert", "accuracy", "chrf", "entropy"]
    metric_labels = {
        "sbert": "SBERT (HM)", "accuracy": "Accuracy",
        "chrf": "chrF", "entropy": "Entropy (bits)",
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    sizes = sorted(perq["size"].unique(), key=lambda s: float(s.replace("B", "")))
    x = np.arange(len(sizes))
    n_metrics = len(metrics)
    bar_w = 0.72 / n_metrics

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for mi, metric in enumerate(metrics):
        deltas, delta_sems = [], []
        for size in sizes:
            nt = perq[(perq["size"] == size) & (~perq["think"])]
            tk = perq[(perq["size"] == size) & (perq["think"])]
            # Match on question_id for paired difference
            merged = nt[["question_id", metric]].dropna().merge(
                tk[["question_id", metric]].dropna(),
                on="question_id", suffixes=("_nt", "_tk"),
            )
            if len(merged) > 0:
                diff = merged[f"{metric}_tk"] - merged[f"{metric}_nt"]
                deltas.append(diff.mean())
                delta_sems.append(diff.sem() if len(diff) > 1 else 0.0)
            else:
                deltas.append(0)
                delta_sems.append(0)

        offset = (mi - (n_metrics - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            deltas,
            yerr=delta_sems,
            width=bar_w * 0.88,
            color=colors[mi],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            capsize=2.5,
            error_kw={"lw": 0.8, "capthick": 0.8},
            label=metric_labels[metric],
        )

    ax.axhline(0, color="black", lw=0.8, ls="-")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Qwen3-{s}" for s in sizes], fontsize=11)
    ax.set_ylabel("Δ (think − no-think)", fontsize=12)
    ax.set_title("Qwen3 SA-LLM: effect of thinking on metrics", fontsize=13)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=n_metrics, fontsize=9, frameon=True,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, OUT_DIR, "qwen3_think_vs_nothink_delta.png")
    plt.close(fig)


def main() -> None:
    perq = build_comparison_df()
    agg = aggregate(perq)
    print(agg.to_string(index=False))
    plot_bars(agg)
    plot_delta(perq)


if __name__ == "__main__":
    main()
