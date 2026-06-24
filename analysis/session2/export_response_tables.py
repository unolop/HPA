"""
Rebuild canonical session-2 human/model response exports from raw files.

This replaces the notebook-only export cell with a scriptable path so figure
exporters can rely on up-to-date response tables.

Outputs:
  analysis/session2/exports/responses_human.csv
  analysis/session2/exports/responses_model_control.csv
  analysis/session2/exports/responses_model_blind.csv
  analysis/session2/exports/responses_model_inst_blind.csv

Usage:
  conda run -n zero python analysis/session2/export_response_tables.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis"))

from config import MIN_ANSWERS_DEFAULT
from utils.load_session import load_human_data
from utils.model_registry import MODEL_TYPE, default_all_models
from utils.normalize_korean import clean_korean_answer
from utils.vqa import preprocess_answer, vqa_accuracy


VARIANT_ORDER = ["C", "B", "A"]
CT_MAP = {"question": "C", "weaker_object": "B", "pronominalized": "A"}
COND_MAP = {
    "vqa_1k_control": "control",
    "vqa_1k_control_blind": "blind",
    "vqa_1k_control_inst_blind": "inst_blind",
}


def _load_question_text_maps(participants: list[dict]) -> tuple[dict, dict]:
    q_en_map: dict[tuple[int, str], str] = {}
    q_kr_map: dict[tuple[int, str], str] = {}
    for p in participants:
        for ans in p.get("answers", []):
            key = (int(ans["question_id"]), ans.get("variant", "C"))
            if key not in q_en_map:
                q_en_map[key] = ans.get("question_en", "")
                q_kr_map[key] = ans.get("question_kr", "")
    return q_en_map, q_kr_map


def _build_model_rows(
    all_models: dict,
    common_qids: set[int],
    mapper,
    condition_key: str,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    acc_rows: list[dict] = []

    for label, (results_dir, mdir) in all_models.items():
        path = results_dir / mdir / f"{condition_key}.jsonl"
        if not path.exists():
            continue

        seen = set()
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    print(f"MALFORMED JSON skipped in {path}")
                    continue

                qid = int(ex["question_id"])
                if qid not in common_qids:
                    continue

                gt = mapper.get_answers(qid)
                if not gt:
                    continue

                ga = ex.get("generated_answers", {}) or {}
                for ct, var in CT_MAP.items():
                    ans = preprocess_answer(ga.get(ct, ""), strip_think=True)
                    if not ans:
                        continue
                    key = (label, qid, var)
                    if key in seen:
                        continue
                    seen.add(key)
                    acc = float(vqa_accuracy(ans, gt))
                    rows.append({
                        "question_id": qid,
                        "variant": var,
                        "model": label,
                        "model_group": MODEL_TYPE.get(label, ""),
                        "accuracy": acc,
                        "response": ans,
                    })
                    acc_rows.append({
                        "question_id": qid,
                        "variant": var,
                        "model": label,
                        "accuracy": acc,
                    })

    return rows, acc_rows


def main(*, translate: bool = True) -> None:
    participants, common_qids, df, mapper = load_human_data(
        ROOT,
        min_answers=MIN_ANSWERS_DEFAULT,
        translate=translate,
        verbose=True,
    )
    common_qids = set(int(q) for q in common_qids)
    all_models = default_all_models(ROOT)

    out_dir = ROOT / "analysis/session2/exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = (
        df[["question_id", "ent", "op"]]
        .drop_duplicates("question_id")
        .set_index("question_id")
    )
    gt_map = {qid: " | ".join(mapper.get_answers(qid) or []) for qid in common_qids}
    q_en_map, q_kr_map = _load_question_text_maps(participants)

    h_avg = (
        df.groupby(["question_id", "variant"])["accuracy"]
        .mean()
        .rename("human_avg_acc")
        .reset_index()
    )

    def add_meta(df_in: pd.DataFrame) -> pd.DataFrame:
        df_in = df_in.merge(meta.reset_index(), on="question_id", how="left")
        df_in["gt"] = df_in["question_id"].map(gt_map)
        return df_in

    # Human export
    hrows = df[
        ["question_id", "variant", "participant", "answer_kr", "answer_en", "accuracy"]
    ].copy()
    hrows["canonical_kr"] = hrows["answer_kr"].apply(clean_korean_answer)
    hrows["response"] = hrows["answer_en"]
    hrows["question_en"] = hrows.apply(
        lambda r: q_en_map.get((int(r.question_id), r.variant), ""), axis=1
    )
    hrows["question_kr"] = hrows.apply(
        lambda r: q_kr_map.get((int(r.question_id), r.variant), ""), axis=1
    )
    hrows = add_meta(hrows)
    hrows = hrows[
        [
            "question_id",
            "ent",
            "op",
            "variant",
            "question_en",
            "question_kr",
            "gt",
            "participant",
            "accuracy",
            "answer_kr",
            "canonical_kr",
            "response",
        ]
    ]
    hrows = hrows.merge(h_avg, on=["question_id", "variant"], how="left")
    hrows["model_avg_acc"] = np.nan
    hrows = hrows[
        [
            "question_id",
            "ent",
            "op",
            "variant",
            "question_en",
            "question_kr",
            "gt",
            "human_avg_acc",
            "model_avg_acc",
            "participant",
            "accuracy",
            "answer_kr",
            "canonical_kr",
            "response",
        ]
    ]
    hrows = hrows.sort_values(["question_id", "variant", "participant"]).reset_index(drop=True)
    hpath = out_dir / "responses_human.csv"
    hrows.to_csv(hpath, index=False)
    print(f"Human  -> {hpath}  ({len(hrows)} rows, {hrows.participant.nunique()} participants)")

    # Build all model rows first so model_avg_acc can be based on inst_blind over all models.
    cond_rows: dict[str, list[dict]] = {}
    cond_acc_rows: dict[str, list[dict]] = {}
    for cond_key, cond_name in COND_MAP.items():
        rows, acc_rows = _build_model_rows(all_models, common_qids, mapper, cond_key)
        cond_rows[cond_name] = rows
        cond_acc_rows[cond_name] = acc_rows
        print(
            f"Loaded raw model rows for {cond_name}: {len(rows)} "
            f"rows across {len({r['model'] for r in rows})} models"
        )

    inst_acc_rows = cond_acc_rows.get("inst_blind", [])
    if inst_acc_rows:
        m_avg = (
            pd.DataFrame(inst_acc_rows)
            .groupby(["question_id", "variant"])["accuracy"]
            .mean()
            .rename("model_avg_acc")
            .reset_index()
        )
    else:
        m_avg = pd.DataFrame(columns=["question_id", "variant", "model_avg_acc"])

    for cond_name, rows in cond_rows.items():
        if not rows:
            print(f"[{cond_name}] no model rows; skipping")
            continue
        mrows = pd.DataFrame(rows)
        mrows["question_en"] = mrows.apply(
            lambda r: q_en_map.get((int(r.question_id), r.variant), ""), axis=1
        )
        mrows["question_kr"] = mrows.apply(
            lambda r: q_kr_map.get((int(r.question_id), r.variant), ""), axis=1
        )
        mrows = mrows.merge(h_avg, on=["question_id", "variant"], how="left")
        mrows = mrows.merge(m_avg, on=["question_id", "variant"], how="left")
        mrows = add_meta(mrows)
        mrows = mrows[
            [
                "question_id",
                "ent",
                "op",
                "variant",
                "question_en",
                "question_kr",
                "gt",
                "human_avg_acc",
                "model_avg_acc",
                "model",
                "model_group",
                "accuracy",
                "response",
            ]
        ]
        mrows = mrows.sort_values(
            ["question_id", "variant", "model_group", "model"]
        ).reset_index(drop=True)
        mpath = out_dir / f"responses_model_{cond_name}.csv"
        mrows.to_csv(mpath, index=False)
        print(
            f"Model [{cond_name}] -> {mpath}  "
            f"({len(mrows)} rows, {mrows.model.nunique()} models)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild canonical session-2 human/model response exports."
    )
    parser.add_argument(
        "--no_translate",
        action="store_true",
        help="Skip the API-backed Korean answer translation step and use only the local map/normalizer.",
    )
    args = parser.parse_args()
    main(translate=not args.no_translate)
