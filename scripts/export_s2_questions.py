#!/usr/bin/env python3
"""
Export question variants for experiment/s2.csv using dataset/vqa/vqa1k_control.jsonl.

Outputs three JSON files:
  - experiment/s2_question.json
  - experiment/s2_weaker_object.json
  - experiment/s2_pronominalized.json

Each record has:
  - question_id
  - question
  - question_kr

By default the script translates English questions into Korean using the
Google translate public endpoint and stores a local cache so reruns are stable
and cheap. Pass --no-translate to skip translation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_S2_CSV = REPO_ROOT / "experiment" / "s2.csv"
DEFAULT_CONTROL_JSONL = REPO_ROOT / "dataset" / "vqa" / "vqa1k_control.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "experiment"
DEFAULT_CACHE = REPO_ROOT / "experiment" / "translation_cache_s2_ko.json"
KEYS = ("question", "weaker_object", "pronominalized")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export s2 question variants to JSON.")
    parser.add_argument("--s2-csv", type=Path, default=DEFAULT_S2_CSV,
                        help=f"Path to s2.csv (default: {DEFAULT_S2_CSV})")
    parser.add_argument("--control-jsonl", type=Path, default=DEFAULT_CONTROL_JSONL,
                        help=f"Path to vqa1k_control.jsonl (default: {DEFAULT_CONTROL_JSONL})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE,
                        help=f"Translation cache JSON (default: {DEFAULT_CACHE})")
    parser.add_argument("--target-lang", default="ko",
                        help="Target language code for question_kr (default: ko)")
    parser.add_argument("--no-translate", action="store_true",
                        help="Do not call the translation endpoint; copy English into question_kr.")
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Ignore existing cache and rebuild translations.")
    return parser.parse_args()


def load_s2_qids(path: Path) -> List[int]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "question_id" not in reader.fieldnames:
            raise ValueError(f"{path} is missing required column 'question_id'")
        qids = [int(row["question_id"]) for row in reader]
    if len(qids) != len(set(qids)):
        raise ValueError(f"{path} contains duplicate question_id values")
    return qids


def load_control_rows(path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[int(row["question_id"])] = row
    return rows


def load_cache(path: Path, overwrite: bool) -> Dict[str, str]:
    if overwrite or not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: Path, cache: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.write("\n")


def translate_text(text: str, target_lang: str) -> str:
    params = urlencode({
        "client": "gtx",
        "sl": "en",
        "tl": target_lang,
        "dt": "t",
        "q": text,
    })
    url = f"https://translate.googleapis.com/translate_a/single?{params}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return "".join(part[0] for part in payload[0] if part and part[0]).strip()


def translate_many(
    texts: Iterable[str],
    *,
    target_lang: str,
    cache: Dict[str, str],
    enabled: bool,
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    unique_texts = list(dict.fromkeys(texts))
    for idx, text in enumerate(unique_texts, start=1):
        if text in cache:
            results[text] = cache[text]
            continue
        if not enabled:
            cache[text] = text
            results[text] = text
            continue

        last_err = None
        for attempt in range(3):
            try:
                translated = translate_text(text, target_lang)
                cache[text] = translated
                results[text] = translated
                break
            except Exception as err:  # pragma: no cover - best effort network path
                last_err = err
                time.sleep(1.0)
        else:
            raise RuntimeError(f"Failed to translate text after retries: {text!r}") from last_err

        if idx % 25 == 0:
            print(f"Translated {idx}/{len(unique_texts)} unique strings", file=sys.stderr)
    return {text: cache[text] for text in unique_texts}


def build_variant_records(
    qids: List[int],
    control_rows: Dict[int, dict],
    key: str,
    translations: Dict[str, str],
) -> List[dict]:
    records = []
    missing = []
    empty = []
    for qid in qids:
        row = control_rows.get(qid)
        if row is None:
            missing.append(qid)
            continue
        text = (row.get(key) or "").strip()
        if not text:
            empty.append(qid)
            continue
        records.append({
            "question_id": qid,
            "question": text,
            "question_kr": translations[text],
        })

    if missing:
        raise KeyError(f"{len(missing)} question_id values were not found in control JSONL. "
                       f"First few: {missing[:10]}")
    if empty:
        raise ValueError(f"{len(empty)} rows have empty '{key}' text. First few: {empty[:10]}")
    return records


def main() -> int:
    args = parse_args()

    qids = load_s2_qids(args.s2_csv)
    control_rows = load_control_rows(args.control_jsonl)
    cache = load_cache(args.cache_path, overwrite=args.overwrite_cache)

    variant_texts: Dict[str, List[str]] = {key: [] for key in KEYS}
    for qid in qids:
        row = control_rows.get(qid)
        if row is None:
            continue
        for key in KEYS:
            text = (row.get(key) or "").strip()
            if text:
                variant_texts[key].append(text)

    all_texts: List[str] = []
    for key in KEYS:
        all_texts.extend(variant_texts[key])
    translations = translate_many(
        all_texts,
        target_lang=args.target_lang,
        cache=cache,
        enabled=not args.no_translate,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key in KEYS:
        records = build_variant_records(qids, control_rows, key, translations)
        out_path = args.out_dir / f"s2_{key}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote {out_path} ({len(records)} rows)")

    save_cache(args.cache_path, cache)
    print(f"Wrote {args.cache_path} ({len(cache)} cached translations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
