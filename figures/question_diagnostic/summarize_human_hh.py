from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "analysis/session2/exports"


def top_answers(df: pd.DataFrame, n: int = 5) -> str:
    counts = Counter(str(x).strip() for x in df["response"] if str(x).strip())
    return "; ".join(f"{answer} ({count})" for answer, count in counts.most_common(n))


def build_ranked() -> pd.DataFrame:
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    human = pd.read_csv(EXPORTS / "responses_human.csv")

    hh = (
        pc[pc["pair_type"] == "HH"]
        .groupby(["question_id", "variant"], as_index=False)["sbert_score"]
        .mean()
        .rename(columns={"sbert_score": "hh_sbert"})
    )

    meta = human.groupby(["question_id", "variant"], as_index=False).agg(
        question_en=("question_en", "first"),
        op=("op", "first"),
        ent=("ent", "first"),
        accuracy=("accuracy", "mean"),
    )
    ranked = meta.merge(hh, on=["question_id", "variant"])
    return ranked, human


def print_variant_summaries(ranked: pd.DataFrame, human: pd.DataFrame) -> None:
    for variant in ["C", "B", "A"]:
        sub = (
            ranked[ranked.variant == variant]
            .sort_values("hh_sbert", ascending=False)
            .reset_index(drop=True)
        )
        print(f"===== VARIANT {variant} TOP 10 =====")
        for _, row in sub.head(10).iterrows():
            ans = top_answers(
                human[(human.question_id == row.question_id) & (human.variant == variant)]
            )
            print(
                f"{int(row.question_id)} | {row.hh_sbert:.3f} | "
                f"{row.op}/{row.ent} | {row.question_en} | {ans}"
            )
        print(f"===== VARIANT {variant} BOTTOM 10 =====")
        for _, row in sub.tail(10).iterrows():
            ans = top_answers(
                human[(human.question_id == row.question_id) & (human.variant == variant)]
            )
            print(
                f"{int(row.question_id)} | {row.hh_sbert:.3f} | "
                f"{row.op}/{row.ent} | {row.question_en} | {ans}"
            )

        top = sub.head(10)
        bottom = sub.tail(10)
        print(f"===== VARIANT {variant} COUNTS =====")
        print("top_ops", top.op.value_counts().to_dict())
        print("top_ents", top.ent.value_counts().to_dict())
        print("bottom_ops", bottom.op.value_counts().to_dict())
        print("bottom_ents", bottom.ent.value_counts().to_dict())


def print_degradation(ranked: pd.DataFrame, human: pd.DataFrame) -> None:
    pivot = ranked.pivot(index="question_id", columns="variant", values="hh_sbert").reset_index()
    meta_c = ranked[ranked.variant == "C"][["question_id", "question_en", "op", "ent"]]
    deg = meta_c.merge(pivot, on="question_id")
    deg["delta_CA"] = deg["C"] - deg["A"]
    print("===== TOP 10 C->A DEGRADATION =====")
    for _, row in deg.sort_values("delta_CA", ascending=False).head(10).iterrows():
        ans_c = top_answers(human[(human.question_id == row.question_id) & (human.variant == "C")])
        ans_a = top_answers(human[(human.question_id == row.question_id) & (human.variant == "A")])
        print(
            f"{int(row.question_id)} | {row.delta_CA:.3f} | "
            f"{row.op}/{row.ent} | {row.question_en} | "
            f"C: {ans_c} | A: {ans_a}"
        )


def main() -> None:
    ranked, human = build_ranked()
    print_variant_summaries(ranked, human)
    print_degradation(ranked, human)


if __name__ == "__main__":
    main()
