from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "think_eda"

THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
MODELS = [
    ("Qwen3-0.6B", ROOT / "evaluation/logits/backbone/pretrained/Qwen3-0.6B_think"),
    ("Qwen3-1.7B", ROOT / "evaluation/logits/backbone/pretrained/Qwen3-1.7B_think"),
    ("Qwen3-4B", ROOT / "evaluation/logits/backbone/pretrained/Qwen3-4B_think"),
    ("Qwen3-8B", ROOT / "evaluation/logits/backbone/pretrained/Qwen3-8B_think"),
    ("Qwen3-32B", ROOT / "evaluation/logits/backbone/pretrained/Qwen3-32B_think"),
]
FILES = {
    "Blind": "vqa_1k_control_blind.jsonl",
    "Inst-Blind": "vqa_1k_control_inst_blind.jsonl",
}
COLORS = {"Blind": "#9AA0A6", "Inst-Blind": "#2F6C9E"}


def get_think(text: str) -> str:
    match = THINK_RE.search(text)
    return match.group(1) if match else ""


def load_think_lengths(path: Path) -> list[int]:
    lengths: list[int] = []
    with open(path) as handle:
        for line in handle:
            # Only inspect the generated_answers portion; the token-logit payload
            # is much larger and unnecessary for trace-length histograms.
            answer_region = line.split('"generated_logits"', 1)[0]
            matches = THINK_RE.findall(answer_region)
            if matches:
                lengths.extend(len(match.split()) if match.strip() else 0 for match in matches)
    return lengths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 5.8), sharex=True, sharey=True)
    axes = axes.flatten()
    bins = np.linspace(0, 1600, 41)

    for ax, (model_name, model_dir) in zip(axes, MODELS):
        for condition, filename in FILES.items():
            lengths = load_think_lengths(model_dir / filename)
            rows.extend(
                {
                    "model": model_name,
                    "condition": condition,
                    "think_words": length,
                }
                for length in lengths
            )
            arr = np.array(lengths)
            ax.hist(
                arr,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=COLORS[condition],
                label=f"{condition} (mean={arr.mean():.0f})",
            )
            ax.axvline(arr.mean(), color=COLORS[condition], linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(model_name, fontsize=10)
        ax.set_xlim(0, 1600)
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    axes[5].axis("off")
    for ax in axes[:3]:
        ax.set_xlabel("")
    for ax in axes[3:5]:
        ax.set_xlabel("Think words")
    for idx in [0, 3]:
        axes[idx].set_ylabel("Density")

    fig.suptitle("Think-length distributions with vs without instruction", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "think_length_inst_vs_blind.png", dpi=180, bbox_inches="tight")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["model", "condition"], as_index=False)["think_words"]
        .agg(["mean", "median", "quantile"])
        .reset_index()
    )
    # Replace the generic quantile column with p95 from the raw grouped data.
    p95 = df.groupby(["model", "condition"])["think_words"].quantile(0.95).rename("p95").reset_index()
    summary = (
        df.groupby(["model", "condition"], as_index=False)["think_words"]
        .agg(mean="mean", median="median")
        .merge(p95, on=["model", "condition"], how="left")
    )
    summary.to_csv(OUT_DIR / "think_length_inst_vs_blind_summary.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nSaved: {OUT_DIR / 'think_length_inst_vs_blind.png'}")
    print(f"Saved: {OUT_DIR / 'think_length_inst_vs_blind_summary.csv'}")


if __name__ == "__main__":
    main()
