from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RANKED = ROOT / "analysis/human_answers/hh_ranked_vC.csv"
OUT_DIR = ROOT / "analysis/human_answers"

FIXED_COLS = {
    "question_id",
    "rank",
    "question_en",
    "op",
    "ent",
    "hh_sbert",
    "entropy",
    "accuracy",
    "gt",
}


def fmt_num(val: str, digits: int) -> str:
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return "NA"


def parse_float(val: str) -> float | None:
    try:
        return float(val)
    except Exception:
        return None


def top_answers(row: dict[str, str], participant_cols: list[str], k: int = 5) -> str:
    vals = [row[c].strip() for c in participant_cols if row.get(c) and row[c].strip()]
    cnt = Counter(vals)
    return ", ".join([f'"{ans}" ({n})' for ans, n in cnt.most_common(k)])


def format_block(row: dict[str, str], participant_cols: list[str]) -> str:
    return (
        f"{row.get('op', ''):<6} | {row.get('ent', ''):<7} | "
        f"HH={fmt_num(row.get('hh_sbert', ''), 3)} | "
        f"H={fmt_num(row.get('entropy', ''), 2)} | "
        f"acc={fmt_num(row.get('accuracy', ''), 3)}\n"
        f"  Q: \"{row.get('question_en', '')}\"\n"
        f"  Top answers: {top_answers(row, participant_cols)}\n"
    )


def main() -> None:
    with RANKED.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    participant_cols = [c for c in rows[0].keys() if c not in FIXED_COLS]
    valid_rows = [r for r in rows if parse_float(r.get("hh_sbert", "")) is not None]

    exports = {
        "top10": valid_rows[:10],
        "bottom10": list(reversed(valid_rows[-10:])),
    }

    for name, subset in exports.items():
        text = "\n".join(format_block(r, participant_cols) for r in subset)
        out = OUT_DIR / f"hh_examples_{name}_vC.txt"
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
