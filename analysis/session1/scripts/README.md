# analysis/scripts/

Scripts for question sampling and human study preparation.

## sample_questions.py

Samples N questions from vqa1k for the human study with balanced ent groups and intelligent prioritization.

```
python analysis/scripts/sample_questions.py [--n 100] [--seed 42] [--output analysis/csv/human_study_sample.csv]
```

**Arguments:**
- `--n` — total questions (default: 100)
- `--seed` — random seed (default: 42)
- `--output` — output CSV path
- `--flip-threshold` — min flip rate for priority 4 (default: 0.5)

**Quota allocation:** equal N//9 per ent group (animal, food, object, other, person, place, product, text, vehicle), with remainder distributed to larger groups. All groups are capped at their pool size.

**Priority ordering (within each ent group):**
1. Tier A — overconfident wrong in blind (canonical hallucination)
2. Tier B — answer flip across control variants (linguistically sensitive)
3. Tier C — consensus wrong across models
4. High inst_blind flip rate (≥ 0.5) — instruction unlocks a strong prior
5. Previous human study questions (374 from `analysis/csv/human_vqa.csv`)
6. Fill: remaining questions sorted by descending blind accuracy

**Output columns:** `question_id, image_id, question_text, ent, op, op_raw, w, blind_acc, inst_blind_acc, flip_rate, mean_conf_blind, tier_A, tier_B, tier_C, tiers_str, is_prev_study, priority, selection_reason`

**Sample run (n=100, seed=42):**
- 12 ent groups × ~11 questions each = 100 total
- Tier A (overconfident wrong): 26/100
- Previous study overlap: 40/100
- Mean blind acc: 0.301 (biased toward harder questions)
