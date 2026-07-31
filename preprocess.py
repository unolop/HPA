"""
Preprocessing pipeline for generating control question variants.

Takes the base 1k VQA question set and uses GPT to generate anchor-removed
variants (A: weaker_object/deictic_removed, B: object_removed, C: pronominalized)
as well as semantic annotations (operation type, entity type, answer type).

Usage
-----
# Generate control variants (variants A/B/C):
python preprocess.py --mode CTL \
    --input dataset/vqa/vqav2_1k_val.json \
    --output dataset/vqa/vqa1k_control.jsonl

# Generate semantic annotations only:
python preprocess.py --mode SEM \
    --input dataset/vqa/vqav2_1k_val.json \
    --output dataset/vqa/vqa1k_sem.jsonl

# Add pronominalized variant to an existing JSONL (resume-safe):
python preprocess.py --mode MERGE_PRONOMINALIZED \
    --output dataset/vqa/vqa1k_control.jsonl

Requires OPENAI_API_KEY in environment or .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from api.extract_vqa import get_vqa_questions, process, merge_ctl_field
from api.prompts import SYSTEM_CTL_PRONOMINALIZED, CTL_PRONOMINALIZED_SCHEMA
from dataset.paths import VQA_1K


def load_base_questions(input_path: str) -> list[dict]:
    """Load questions from a VQA JSON file (standard v2 format or flat list)."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if "questions" in data:
        return data["questions"]
    raise ValueError(f"Unrecognised format in {input_path!r}")


def count_completed(output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0
    with open(output_path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate control question variants via GPT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["CTL", "SEM", "MERGE_PRONOMINALIZED"],
        help=(
            "CTL: generate all 4 control variants (deictic_removed, object_removed, "
            "weaker_object, pronominalized). "
            "SEM: extract semantic annotations (op, ent, ans, etc.). "
            "MERGE_PRONOMINALIZED: add pronominalized field to an existing JSONL."
        ),
    )
    parser.add_argument(
        "--input",
        default=str(VQA_1K),
        help="Path to the base VQA question JSON (default: dataset/vqa/vqav2_1k_val.json).",
    )
    parser.add_argument(
        "--output",
        default="dataset/vqa/vqa1k_control.jsonl",
        help="Path to write (or update) the output JSONL.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Max questions per API call (default: 50).",
    )
    args = parser.parse_args()

    if args.mode == "MERGE_PRONOMINALIZED":
        if not os.path.exists(args.output):
            print(f"Error: output file {args.output!r} does not exist.", file=sys.stderr)
            sys.exit(1)
        print(f"Adding pronominalized variants to {args.output!r} ...")
        merge_ctl_field(
            jsonl_path=args.output,
            field="pronominalized",
            system=SYSTEM_CTL_PRONOMINALIZED,
            schema=CTL_PRONOMINALIZED_SCHEMA,
            batch_size=args.batch_size,
        )
        return

    questions = get_vqa_questions(input_path=args.input)
    already_done = count_completed(args.output)
    if already_done:
        print(f"Resuming: {already_done} records already in {args.output!r}, skipping.")
        questions = questions[already_done:]

    if not questions:
        print("Nothing to do — output is already complete.")
        return

    print(f"Processing {len(questions)} questions in {args.mode!r} mode → {args.output!r}")
    process(questions, mode=args.mode, output_path=args.output, batch_size=args.batch_size)
    total = count_completed(args.output)
    print(f"Done. {total} records written to {args.output!r}.")


if __name__ == "__main__":
    main()
