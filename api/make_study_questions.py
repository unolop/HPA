"""
Generate experiment JSON files for the human study (inst_blind condition).

For each of the 3 control types (question, weaker_object, pronominalized),
produces one JSON file in experiment/ with fields:
  { "question_id": int, "question": str, "question_kr": str }

Korean translation is done via GPT-4.1-mini in batches.
Note: question.json contains only new questions not already in human_vqa.csv;
weaker_object and pronominalized contain all 100 sampled questions.
"""
import json, os, re, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.gpt_utils import call_api

ROOT      = Path(__file__).parent.parent
OUT_DIR   = ROOT / "experiment"
OUT_DIR.mkdir(exist_ok=True)

LOGITS_FILE = ROOT / "evaluation/logits/pretrained/Qwen3-VL-8B-Instruct/vqa_1k_control.jsonl"
SAMPLE_CSV  = ROOT / "analysis/csv/human_study_sample.csv"
HVQA_CSV    = ROOT / "analysis/csv/human_vqa.csv"

CONTROL_TYPES = ["question", "weaker_object", "pronominalized"]
EXPERIMENT_NAME = "human_study"
BATCH_SIZE = 40

# ── Translation schema ────────────────────────────────────────────────────────
SYSTEM_TRANSLATE = (
    "You are a professional Korean translator specializing in survey questions. "
    "Translate each English question to natural, concise Korean. "
    "Preserve the meaning exactly. Return ONLY a JSON object {\"results\": [str, ...]} "
    "with one Korean string per input question, in the same order."
)

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["results"],
    "additionalProperties": False
}

def translate_batch(texts: list[str]) -> list[str]:
    """Translate a batch of English questions to Korean."""
    return call_api(SYSTEM_TRANSLATE, TRANSLATE_SCHEMA, "translate", texts)

def strip_prompt_wrapper(text: str) -> str:
    """Remove 'Question: ... Answer the question using a single word or phrase. \nAnswer:' wrapper."""
    text = re.sub(r'^Question:\s*', '', text.strip())
    text = re.sub(r'\s*Answer the question using a single word or phrase\.?\s*\n?Answer:\s*$', '', text)
    return text.strip()

def main():
    # ── Load sample and identify new questions (not in V1 human study) ─────────
    sample = pd.read_csv(SAMPLE_CSV)
    hvqa   = pd.read_csv(HVQA_CSV)
    v1_qids = set(hvqa["question_id"])
    new_qids = set(sample[~sample["question_id"].isin(v1_qids)]["question_id"])
    print(f"New questions (not in V1): {len(new_qids)}")

    # ── Load control variants from logits ──────────────────────────────────────
    rows = {}
    with open(LOGITS_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d["question_id"] in new_qids:
                rows[d["question_id"]] = {
                    ct: strip_prompt_wrapper(d.get(ct, ""))
                    for ct in CONTROL_TYPES
                }

    print(f"Control variants loaded: {len(rows)}")
    missing = new_qids - set(rows.keys())
    if missing:
        print(f"WARNING: {len(missing)} question_ids not found in logits: {missing}")

    # ── For each control type: translate + save JSON ───────────────────────────
    for ct in CONTROL_TYPES:
        print(f"\n── {ct} ──")
        entries = [
            {"question_id": qid, "question": rows[qid][ct]}
            for qid in sorted(rows.keys())
            if rows[qid].get(ct)
        ]

        # Translate in batches
        all_kr = []
        texts  = [e["question"] for e in entries]
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            print(f"  Translating batch {i//BATCH_SIZE + 1}/{-(-len(texts)//BATCH_SIZE)} ({len(batch)} items)...")
            kr_batch = translate_batch(batch)
            all_kr.extend(kr_batch)

        assert len(all_kr) == len(entries), f"Translation length mismatch: {len(all_kr)} vs {len(entries)}"

        # Build output
        out = [
            {
                "question_id": e["question_id"],
                "question":    e["question"],
                "question_kr": kr,
            }
            for e, kr in zip(entries, all_kr)
        ]

        out_path = OUT_DIR / f"{EXPERIMENT_NAME}_{ct}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(out)} questions → {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
