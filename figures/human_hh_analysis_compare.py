"""
Compare human-only HH SBERT summaries between:
  - q88 free-text only
  - q113 yes/no-inclusive

Exports side-by-side CSV tables for entity and operation type.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from utils.vqa import preprocess_answer

EXPORTS = ROOT / "analysis/session2/exports"
OUT_DIR = ROOT / "figures/human_hh_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_hh(include_yesno: bool) -> pd.DataFrame:
    pair_df = pd.read_parquet(EXPORTS / "pair_cache.parquet")
    if not include_yesno:
        return pair_df[pair_df["pair_type"] == "HH"].copy()

    yesno = pd.read_csv(EXPORTS / "answer_pairs_yesno.csv")
    cache = np.load(EXPORTS / "embeddings_all-mpnet-base-v2.npz", allow_pickle=True)
    strings = cache["strings"].tolist()
    vectors = cache["vectors"]
    lookup = {s: vectors[i] for i, s in enumerate(strings)}

    a1 = yesno["answer_1"].fillna("").apply(preprocess_answer)
    a2 = yesno["answer_2"].fillna("").apply(preprocess_answer)
    scores = []
    for s1, s2 in zip(a1, a2):
        v1, v2 = lookup.get(s1), lookup.get(s2)
        if v1 is None or v2 is None:
            scores.append(np.nan)
        else:
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            scores.append(float(np.dot(v1, v2) / (n1 * n2)) if n1 and n2 else 0.0)
    yesno["sbert_score"] = scores

    cols = ["question_id", "variant", "ent", "op", "pair_type", "sbert_score"]
    hh = pd.concat(
        [
            pair_df[pair_df["pair_type"] == "HH"][cols],
            yesno[yesno["pair_type"] == "HH"][cols],
        ],
        ignore_index=True,
    )
    return hh


def summarize(hh: pd.DataFrame, group_col: str, tag: str) -> pd.DataFrame:
    summary = (
        hh.groupby([group_col, "variant"])["sbert_score"]
        .mean()
        .rename("hh_sbert")
        .reset_index()
    )
    wide = summary.pivot(index=group_col, columns="variant", values="hh_sbert").reset_index()
    wide.columns.name = None
    rename = {c: f"{tag}_{c}" for c in ["C", "B", "A"] if c in wide.columns}
    return wide.rename(columns=rename)


def main() -> None:
    hh88 = load_hh(include_yesno=False)
    hh113 = load_hh(include_yesno=True)

    ent = summarize(hh88, "ent", "q88").merge(
        summarize(hh113, "ent", "q113"), on="ent", how="outer"
    ).sort_values("q113_C", ascending=False)
    op = summarize(hh88, "op", "q88").merge(
        summarize(hh113, "op", "q113"), on="op", how="outer"
    ).sort_values("q113_C", ascending=False)

    ent_path = OUT_DIR / "human_hh_sbert_entity_q88_vs_q113.csv"
    op_path = OUT_DIR / "human_hh_sbert_op_q88_vs_q113.csv"
    ent.to_csv(ent_path, index=False)
    op.to_csv(op_path, index=False)
    print(f"Saved: {ent_path}")
    print(f"Saved: {op_path}")


if __name__ == "__main__":
    main()
