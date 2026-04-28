"""
Helpers for translating unmapped Korean human answers into kr_answer_map.json.

This keeps the OpenAI translation loop out of notebooks so notebook execution
does not depend on `tqdm.notebook` / ipywidgets.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .normalize_korean import clean_korean_answer, normalize_korean_answer


_KR_RE = re.compile(r"[가-힣]")
_DEFAULT_MAP_PATH = Path(__file__).resolve().parent / "kr_answer_map.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PREPROCESSING_DIR = _REPO_ROOT / "preprocessing"

if str(_PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING_DIR))

from preprocess import ask_gpt, setup_openai_client, translate_prompt  # noqa: E402


def collect_canonical_queue(
    participants: list[dict[str, Any]],
    mapping: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str], int]:
    """Return raw Korean answers, canonical translation queue, and skip count."""
    raw_to_question: dict[str, str] = {}
    value_map = mapping["map"]

    for participant in participants:
        for answer in participant.get("answers", []):
            raw = str(answer.get("answer_text", "")).strip()
            if _KR_RE.search(raw) and raw not in raw_to_question:
                raw_to_question[raw] = answer.get("question_en", "")

    canonical_to_question: dict[str, str] = {}
    skipped = 0
    for raw, question in raw_to_question.items():
        canonical = clean_korean_answer(raw)
        if value_map.get(raw, "") not in ("", None):
            skipped += 1
            continue
        if value_map.get(canonical, "") not in ("", None):
            skipped += 1
            continue
        if canonical not in canonical_to_question:
            canonical_to_question[canonical] = question

    return raw_to_question, canonical_to_question, skipped


def translate_unmapped_korean_answers(
    participants: list[dict[str, Any]],
    map_path: str | Path = _DEFAULT_MAP_PATH,
    *,
    model: str = "gpt-4o-mini",
    client: Any = None,
    save_every: int = 25,
    show_samples: int = 8,
) -> dict[str, Any]:
    """
    Translate canonical Korean answers missing from kr_answer_map.json.

    Returns a stats dict so notebooks can inspect what happened without
    re-parsing stdout.
    """
    map_path = Path(map_path)
    with open(map_path, encoding="utf-8") as f:
        kr_map = json.load(f)

    raw_to_question, canonical_to_question, skipped = collect_canonical_queue(
        participants, kr_map
    )

    print(f"Raw Korean strings      : {len(raw_to_question)}")
    print(f"Already translated      : {skipped}")
    print(f"Unique canonicals to translate: {len(canonical_to_question)}")
    if canonical_to_question and show_samples:
        print("\nSample canonical forms:")
        for canonical, question in list(canonical_to_question.items())[:show_samples]:
            print(f"  {canonical!r:25}  (q: {question[:50]!r})")

    translated_count = 0
    if canonical_to_question:
        if client is None:
            client = setup_openai_client()

        if client:
            iterator = tqdm(canonical_to_question.items(), desc="Translating")
            for idx, (canonical, question) in enumerate(iterator, start=1):
                prompt = translate_prompt(question, canonical)
                english = ask_gpt(client, prompt, model=model)
                english = english.strip("\"' ").split("\n")[0]
                kr_map["map"][canonical] = english
                translated_count += 1

                if save_every > 0 and idx % save_every == 0:
                    with open(map_path, "w", encoding="utf-8") as f:
                        json.dump(kr_map, f, ensure_ascii=False, indent=2)

            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(kr_map, f, ensure_ascii=False, indent=2)

            print(
                f"\n✓ Translated {translated_count} canonical strings → {map_path}"
            )
        else:
            print("⚠ No OpenAI client — set OPENAI_API_KEY and re-run.")
    else:
        print("\n✓ All Korean answers already mapped.")

    # Clear the memoized normalizer cache so later notebook cells see updates.
    normalize_korean_answer.__defaults__[0].clear()

    return {
        "raw_korean_strings": len(raw_to_question),
        "already_translated": skipped,
        "canonicals_to_translate": len(canonical_to_question),
        "translated": translated_count,
        "map_path": str(map_path),
    }
