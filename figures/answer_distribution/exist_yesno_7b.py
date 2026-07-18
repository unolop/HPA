from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from analysis.utils.abstention import classify as classify_abstention, is_abstained
from analysis.utils.vqa import preprocess_answer


EXPORTS = ROOT / "analysis" / "session2" / "exports"
OUT_DIR = ROOT / "figures" / "answer_distribution" / "exist_yesno_7b"
LATEX_OUT_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "answer_distribution" / "exist_yesno_7b"

FAMILIES = [
    {
        "slug": "internvl_8b",
        "title": "InternVL-8B",
        "rows": [
            ("Human", "Human"),
            ("InternVL-8B", "InternVL-8B"),
            ("InternVL-8B (LM)", "InternVL-8B (LM)"),
            ("Qwen3-8B", "Qwen3-8B"),
        ],
    },
    {
        "slug": "qwen3_8b",
        "title": "Qwen3-8B",
        "rows": [
            ("Human", "Human"),
            ("Qwen3-VL-8B", "Qwen3-VL-8B"),
            ("Qwen3-VL-8B (LM)", "Qwen3-VL-8B (LM)"),
            ("Qwen3-8B", "Qwen3-8B"),
            ("Qwen3-8B (think)", "Qwen3-8B (think)"),
        ],
    },
    {
        "slug": "llava_15_7b",
        "title": "LLaVA-1.5-7B",
        "rows": [
            ("Human", "Human"),
            ("LLaVA-1.5-7B", "LLaVA-1.5-7B"),
            ("LLaVA-1.5 (LM)", "LLaVA-1.5 (LM)"),
            ("Vicuna-7B", "Vicuna-7B"),
        ],
    },
    {
        "slug": "llava_mistral_7b",
        "title": "LLaVA-Mistral",
        "rows": [
            ("Human", "Human"),
            ("LLaVA-Mistral", "LLaVA-Mistral"),
            ("LLaVA-Mistral (LM)", "LLaVA-Mistral (LM)"),
            ("Mistral-7B", "Mistral-7B"),
        ],
    },
    {
        "slug": "llava_vicuna_7b",
        "title": "LLaVA-Vicuna",
        "rows": [
            ("Human", "Human"),
            ("LLaVA-Vicuna", "LLaVA-Vicuna"),
            ("LLaVA-Vicuna (LM)", "LLaVA-Vicuna (LM)"),
            ("Vicuna-7B", "Vicuna-7B"),
        ],
    },
]

YN_COLORS = {
    "yes": "#6BA292",
    "no": "#C97A6A",
    "others": "#B5B5B5",
    "abstain": "#7A7A7A",
}


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


def classify_model_bucket(text: str) -> str:
    cleaned = preprocess_answer(text, strip_think=True)
    abst_cls = classify_abstention(cleaned, None)
    if is_abstained(abst_cls):
        return "abstain"
    return classify_yn(cleaned.lower())


def load_frames(variants: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    human = pd.read_csv(EXPORTS / "responses_human.csv")
    human = human[(human["op"] == "exist") & (human["variant"].isin(variants))].copy()
    human["source"] = "Human"
    human["yn"] = clean_output(human["response"]).apply(classify_yn)

    model = pd.read_csv(EXPORTS / "responses_model_inst_blind.csv")
    model = model[(model["op"] == "exist") & (model["variant"].isin(variants))].copy()
    model["yn"] = model["response"].fillna("").astype(str).apply(classify_model_bucket)
    return human, model


def build_stack(df: pd.DataFrame, source_order: list[str]) -> pd.DataFrame:
    counts = df.value_counts(["source", "yn"]).reset_index(name="count")
    counts["proportion"] = counts.groupby("source")["count"].transform(lambda x: x / x.sum())
    return (
        counts.pivot(index="source", columns="yn", values="proportion")
        .fillna(0.0)
        .reindex(source_order)
        .fillna(0.0)
    )


def draw_family(ax: plt.Axes, family: dict[str, object], human: pd.DataFrame, model: pd.DataFrame) -> None:
    rows = family["rows"]
    source_order = [label for _, label in rows]

    frames = [human[["source", "yn"]]]
    for model_name, row_label in rows:
        if model_name == "Human":
            continue
        sub = model[model["model"] == model_name].copy()
        sub["source"] = row_label
        frames.append(sub[["source", "yn"]])
    plot_df = pd.concat(frames, ignore_index=True)
    stack = build_stack(plot_df, source_order)
    cats = [c for c in ["yes", "no", "others", "abstain"] if c in stack.columns]

    stack[cats].plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[YN_COLORS[c] for c in cats],
        width=0.64,
        legend=False,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for i, src in enumerate(stack.index):
        cum = 0.0
        for cat in cats:
            val = float(stack.loc[src, cat])
            if val > 0.08:
                ax.text(
                    cum + val / 2,
                    i,
                    f"{val * 100:.0f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.5,
                    fontweight="bold",
                )
            cum += val
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.axhline(len(stack.index) - 0.5, color="#DDDDDD", linewidth=0.9)


def build_family_figure(variants: tuple[str, ...], family: dict[str, object], out_name: str) -> Path:
    human, model = load_frames(variants)
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 0.88 * len(family["rows"]) + 0.45), sharex=True)
    draw_family(ax, family, human, model)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=YN_COLORS["yes"]),
        plt.Rectangle((0, 0), 1, 1, color=YN_COLORS["no"]),
        plt.Rectangle((0, 0), 1, 1, color=YN_COLORS["others"]),
        plt.Rectangle((0, 0), 1, 1, color=YN_COLORS["abstain"]),
    ]
    fig.legend(
        handles,
        ["yes", "no", "others", "abstain"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    ax.set_xlabel("Proportion", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def copy_to_latex(path: Path) -> None:
    LATEX_OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, LATEX_OUT_DIR / path.name)


def main() -> None:
    outputs = []
    for family in FAMILIES:
        outputs.append(
            build_family_figure(
                ("C",),
                family,
                f"vC_exist_yesno_{family['slug']}_7b_inst_blind_abstsplit.png",
            )
        )
        outputs.append(
            build_family_figure(
                ("A", "B", "C"),
                family,
                f"vABC_exist_yesno_{family['slug']}_7b_inst_blind_abstsplit.png",
            )
        )
    for path in outputs:
        copy_to_latex(path)
        print(path)


if __name__ == "__main__":
    main()
