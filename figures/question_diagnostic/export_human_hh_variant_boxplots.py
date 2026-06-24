from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "figures/question_diagnostic"
LATEX_DIR = ROOT / "latex/AAAI2026/LaTeX/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_LABELS = {
    "C": "Original",
    "B": "Weaker",
    "A": "Pronominalized",
}
VARIANT_ORDER = ["Original", "Weaker", "Pronominalized"]
VARIANT_PALETTE = {
    "Original": "#1f77b4",
    "Weaker": "#ff7f0e",
    "Pronominalized": "#2ca02c",
}

OP_ORDER = ["ident", "count", "attr", "act", "spat"]
OP_LABELS = {
    "ident": "Identity",
    "count": "Count",
    "attr": "Attribute",
    "act": "Action",
    "spat": "Spatial",
}
ENT_MERGE = {
    "person": "person",
    "animal": "animal",
    "object": "object",
    "food": "food",
    "other": "other",
    "product": "other",
    "place": "other",
    "vehicle": "other",
    "text": "other",
}
ENT_ORDER = ["person", "animal", "object", "food", "other"]
ENT_LABELS = {
    "person": "Person",
    "animal": "Animal",
    "object": "Object",
    "food": "Food",
    "other": "Other",
}


def load_long_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    human = pd.read_csv(EXPORTS / "responses_human.csv")
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")

    meta = (
        human[human["variant"] == "C"]
        .drop_duplicates("question_id")[["question_id", "ent", "op"]]
        .copy()
    )
    meta["ent_group"] = meta["ent"].map(ENT_MERGE).fillna("other")

    rows = []
    for v_code, v_name in VARIANT_LABELS.items():
        hh_v = (
            pc[(pc["variant"] == v_code) & (pc["pair_type"] == "HH")]
            .groupby("question_id")["sbert_score"]
            .mean()
            .rename("hh_sbert")
            .reset_index()
        )
        tmp = hh_v.merge(meta, on="question_id", how="left")
        tmp["variant_name"] = v_name
        rows.append(tmp)

    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df[long_df["op"].isin(OP_ORDER) & long_df["ent_group"].isin(ENT_ORDER)].copy()
    long_df["op_label"] = long_df["op"].map(OP_LABELS)
    long_df["ent_label"] = long_df["ent_group"].map(ENT_LABELS)
    return long_df, meta


def style_axes(ax: plt.Axes, ylabel: str = "HH SBERT") -> None:
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)


def add_n_labels(ax: plt.Axes, labels: list[str], counts: dict[str, int]) -> None:
    ax.set_xticklabels([f"{lab}\n(n={counts.get(lab, 0)})" for lab in labels], rotation=0)


def export_op_plot(df: pd.DataFrame) -> None:
    plot_df = df[df["op"].isin(OP_ORDER)].copy()
    counts = (
        plot_df[plot_df["variant_name"] == "Original"]
        .groupby("op_label")["question_id"]
        .nunique()
        .to_dict()
    )
    order = (
        plot_df[plot_df["variant_name"] == "Original"]
        .groupby("op_label")["hh_sbert"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.boxplot(
        data=plot_df,
        x="op_label",
        y="hh_sbert",
        hue="variant_name",
        order=order,
        hue_order=VARIANT_ORDER,
        palette=VARIANT_PALETTE,
        width=0.56,
        fliersize=2.2,
        linewidth=0.9,
        ax=ax,
    )
    style_axes(ax)
    add_n_labels(ax, order, counts)
    ax.legend(title="Variant", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=True, fontsize=11, title_fontsize=11)
    fig.tight_layout()

    out = OUT_DIR / "human_hh_sbert_by_op_variant.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(LATEX_DIR / out.name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_ent_plot(df: pd.DataFrame) -> None:
    plot_df = df[df["ent_group"].isin(ENT_ORDER)].copy()
    counts = (
        plot_df[plot_df["variant_name"] == "Original"]
        .groupby("ent_label")["question_id"]
        .nunique()
        .to_dict()
    )
    order = (
        plot_df[plot_df["variant_name"] == "Original"]
        .groupby("ent_label")["hh_sbert"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.boxplot(
        data=plot_df,
        x="ent_label",
        y="hh_sbert",
        hue="variant_name",
        order=order,
        hue_order=VARIANT_ORDER,
        palette=VARIANT_PALETTE,
        width=0.56,
        fliersize=2.2,
        linewidth=0.9,
        ax=ax,
    )
    style_axes(ax)
    add_n_labels(ax, order, counts)
    ax.legend(title="Variant", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=True, fontsize=11, title_fontsize=11)
    fig.tight_layout()

    out = OUT_DIR / "human_hh_sbert_by_entity_variant.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(LATEX_DIR / out.name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df, _ = load_long_df()
    export_op_plot(df)
    export_ent_plot(df)
    print("Saved human HH seaborn boxplots to figures/question_diagnostic and latex figure dir.")


if __name__ == "__main__":
    main()
