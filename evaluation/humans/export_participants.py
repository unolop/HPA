#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split aggregated participant JSON files into one JSON file per participant. "
            "When duplicate participant codes are found, keep the most complete record."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="evaluation/humans",
        help="Directory containing aggregated participant JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/humans/by_participant",
        help="Directory where one-participant-per-file JSONs will be written.",
    )
    parser.add_argument(
        "--include-zero-answers",
        action="store_true",
        help="Also export participants with zero completed answers. Default skips them.",
    )
    parser.add_argument(
        "--delete-source-files",
        action="store_true",
        help="Delete processed aggregate source JSON files after a successful export.",
    )
    return parser.parse_args()


def sanitize_piece(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r'[\\/:"*?<>|]+', "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "unknown"


def extract_mmdd(participant: dict) -> str:
    for key in ("created_at", "completed_at", "session_1_started_at"):
        value = participant.get(key)
        if not value:
            continue
        try:
            dt = datetime.fromisoformat(value)
            return dt.strftime("%m%d")
        except ValueError:
            continue
    return "0000"


def completed_count(participant: dict) -> int:
    answers = participant.get("answers") or []
    return len(answers)


def participant_rank(participant: dict) -> tuple:
    return (
        completed_count(participant),
        participant.get("completed_at") or "",
        participant.get("session_3_completed_at") or "",
        participant.get("session_2_completed_at") or "",
        participant.get("session_1_completed_at") or "",
        participant.get("created_at") or "",
    )


def load_participants(input_dir: Path):
    participants = []
    for path in sorted(input_dir.glob("*.json")):
        data = json.load(open(path, encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for participant in data:
            participants.append((path, participant))
    return participants


def choose_best_records(participants):
    grouped = defaultdict(list)
    for source_path, participant in participants:
        grouped[participant.get("code")].append((source_path, participant))

    chosen = {}
    duplicate_report = {}
    for code, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda row: participant_rank(row[1]), reverse=True)
        chosen[code] = rows_sorted[0]
        if len(rows_sorted) > 1:
            duplicate_report[code] = rows_sorted
    return chosen, duplicate_report


def build_filename(participant: dict) -> str:
    mmdd = extract_mmdd(participant)
    tag = sanitize_piece(participant.get("tag", "unknown"))
    code = sanitize_piece(participant.get("code", "unknown"))
    name = sanitize_piece((participant.get("info") or {}).get("name", "unknown"))
    count = completed_count(participant)
    return f"{mmdd}_{tag}_{code}_{name}_{count}.json"


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(input_dir.glob("*.json"))
    participants = load_participants(input_dir)
    chosen, duplicate_report = choose_best_records(participants)

    written = []
    for code, (source_path, participant) in sorted(chosen.items()):
        count = completed_count(participant)
        if count == 0 and not args.include_zero_answers:
            continue
        filename = build_filename(participant)
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(participant, f, ensure_ascii=False, indent=2)
        written.append((out_path, source_path, count))

    print(f"Loaded participant records: {len(participants)}")
    print(f"Unique participant codes: {len(chosen)}")
    print(f"Duplicate participant codes: {len(duplicate_report)}")
    for code, rows in sorted(duplicate_report.items()):
        print(f"DUPLICATE {code}")
        for source_path, participant in rows:
            print(
                "  "
                f"source={source_path.name} "
                f"answers={completed_count(participant)} "
                f"status={participant.get('session_status')} "
                f"name={(participant.get('info') or {}).get('name')}"
            )

    print(f"Wrote files: {len(written)}")
    for out_path, source_path, answer_count in written:
        print(f"  {out_path} <- {source_path.name} ({answer_count} answers)")

    if args.delete_source_files:
        deleted = []
        for source_path in source_files:
            source_path.unlink()
            deleted.append(source_path.name)
        print(f"Deleted source files: {len(deleted)}")
        for name in deleted:
            print(f"  {name}")


if __name__ == "__main__":
    main()
