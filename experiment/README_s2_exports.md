# S2 Question Export

This export maps `question_id` values from `experiment/s2.csv` to the text
variants stored in `dataset/vqa/vqa1k_control.jsonl` and writes three JSON
files:

- `experiment/s2_question.json`
- `experiment/s2_weaker_object.json`
- `experiment/s2_pronominalized.json`

Each JSON record has:

```json
{
  "question_id": 441840002,
  "question": "How many men are there?",
  "question_kr": "남자가 몇 명 있나요?"
}
```

## How to Run

From the repo root:

```bash
python3 scripts/export_s2_questions.py
```

This will:

1. read `experiment/s2.csv`
2. look up each `question_id` in `dataset/vqa/vqa1k_control.jsonl`
3. export the `question`, `weaker_object`, and `pronominalized` fields
4. translate each English question to Korean as `question_kr`
5. cache translations in `experiment/translation_cache_s2_ko.json`

## Reproducibility Notes

- The mapping is deterministic because it is keyed directly by `question_id`.
- The output order matches the row order in `experiment/s2.csv`.
- Korean translations are cached locally in
  `experiment/translation_cache_s2_ko.json` so reruns stay stable.
- If the cache already exists, cached translations are reused.

## Useful Options

Skip translation and just copy English into `question_kr`:

```bash
python3 scripts/export_s2_questions.py --no-translate
```

Rebuild the translation cache from scratch:

```bash
python3 scripts/export_s2_questions.py --overwrite-cache
```

Use a different input/output path:

```bash
python3 scripts/export_s2_questions.py \
  --s2-csv experiment/s2.csv \
  --control-jsonl dataset/vqa/vqa1k_control.jsonl \
  --out-dir experiment
```

## Failure Modes

- If a `question_id` in `experiment/s2.csv` is missing from
  `dataset/vqa/vqa1k_control.jsonl`, the script stops with an error.
- If translation fails repeatedly, the script stops so the exports do not mix
  translated and untranslated rows silently.
