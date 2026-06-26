from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "analysis/session2/exports"
OUT = ROOT / "analysis/human_answers"


def shannon(values) -> float:
    vals = [str(v).strip() for v in values if str(v).strip()]
    if not vals:
        return float("nan")
    counts = Counter(vals)
    n = sum(counts.values())
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def main() -> None:
    pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
    human = pd.read_csv(EXPORTS / "responses_human.csv")

    hh = (
        pc[pc["pair_type"] == "HH"]
        .groupby(["question_id", "variant"], as_index=False)
        .agg(hh_sbert=("sbert_score", "mean"))
    )

    meta = human.groupby(["question_id", "variant"], as_index=False).agg(
        question_en=("question_en", "first"),
        op=("op", "first"),
        ent=("ent", "first"),
        entropy=("response", shannon),
        accuracy=("accuracy", "mean"),
        gt=("gt", "first"),
    )

    wide = (
        human.sort_values(["question_id", "variant", "participant"])
        .pivot_table(
            index=["question_id", "variant"],
            columns="participant",
            values="response",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    ranked = meta.merge(hh, on=["question_id", "variant"], how="left").merge(
        wide, on=["question_id", "variant"], how="left"
    )

    for variant in ["C", "B", "A"]:
        sub = (
            ranked[ranked["variant"] == variant]
            .sort_values("hh_sbert", ascending=False)
            .reset_index(drop=True)
            .copy()
        )
        sub.insert(1, "rank", range(1, len(sub) + 1))
        sub = sub.drop(columns=["variant"])
        cols = [
            "question_id",
            "rank",
            "question_en",
            "op",
            "ent",
            "hh_sbert",
            "entropy",
            "accuracy",
            "gt",
        ]
        participant_cols = [c for c in sub.columns if c not in cols]
        sub = sub[cols + participant_cols]
        out = OUT / f"hh_ranked_v{variant}.csv"
        sub.to_csv(out, index=False)
        print(f"wrote {out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
