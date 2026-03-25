# Data Issues & Scoring Pitfalls

Issues found during analysis that affect result validity. Check these before writing any
numbers into the paper.

---

## 1. Qwen3-VL `answers` Field is Self-Referential

**Affected files:** `evaluation/logits/pretrained/Qwen3-VL-*/vqa_1k_control.jsonl`

The `answers` field in Qwen3 control logits files contains the model's OWN predictions
from an earlier image-conditioned run, NOT the VQAv2 annotator ground truth. Scoring
control variant outputs against this field gives accuracy relative to the model's own
baseline behavior, not against human ground truth.

**Evidence:** 97.3% of outputs in the control file match `answers[0]` for Qwen3-VL-8B.
This is impossibly high for real accuracy.

**Fix:** Use `vqav2_1k_val.json` as the GT source for all models. Load GT once and join
on `question_id` before scoring.

---

## 2. LLaVA Tokenizer: Multi-Word Answers Concatenated Without Spaces

**Affected files:** All LLaVA logits files

When decoding multi-token answers from `generated_logits`, tokens are joined without spaces,
producing outputs like `"Onbuilding"` instead of `"on building"`, `"taking picture"` →
`"takingpicture"`. This systematically underestimates accuracy for multi-word answers.

**Fix options:**
1. Use the `output` field from scored/ files rather than reconstructing from logits
2. If logits are needed for confidence, align on `question_id` to the scored/ output field
   and only use logits for the probability values, not the decoded text

---

## 3. Abstention Regex: `blank` Matches "blanket"

**Affected:** Any abstention analysis using the current regex pattern

The pattern `blank` fires on the legitimate answer `"blanket"`. Verified: 2 of Qwen3-VL-8B's
3 "hard abstentions" were actually the answer `"blanket"`.

**Fix:**
```python
# BAD
r"blank"

# GOOD — use word boundary
r"\bblank\b"
```

Full recommended regex (word-boundary-safe):
```python
refusal_re = re.compile(
    r"cannot|can't|don't see|no image|unable|not sure|unclear"
    r"|i cannot|i don't|can not|without|no visual|not visible"
    r"|cannot see|i'm not|i am not|\bblank\b",
    re.I
)
```

---

## 4. Scoring Regime Mismatch: 10-Annotator vs. Single GT

**Affects:** Degradation curve comparisons between baseline and control variants

- The VQAv2 baseline uses **10-annotator soft accuracy** (VQA eval: answer matches ≥3/10
  annotators → 1.0; 2/10 → 0.67; 1/10 → 0.33)
- Control variant scoring has used **single canonical GT** exact match in some notebooks

These are incompatible for degradation curve analysis. The 10-annotator score is always
higher than single-GT exact match, making control variants appear to drop in accuracy
even if behavior is unchanged.

**Fix:** For all degradation analysis, use the same scoring function (VQAv2 soft accuracy
with the original 10-annotator pool) for ALL control types. The annotators were asked about
the original question — their answers should also apply to paraphrased control variants
(deictic_removed, object_removed, etc.) since the referent is the same. Only `pronominalized`
may need separate judgment.

---

## 5. Qwen3-VL-4B: No Usable Control Data

`evaluation/logits/pretrained/Qwen3-VL-4B-Instruct/vqa_1k_control.jsonl` has
`generated_answers=None` for all records. This model cannot contribute to control variant
analysis until re-run.

Also: Qwen3-VL-32B only has `vqa_1k_control_inst_blind.jsonl` — no blind (without inst)
condition. Missing: direct blind comparison for the largest and best-performing model.

---

## 6. Inst Abstention Artifacts (Qwen3 Long-Form Refusals)

Qwen3-VL-8B produces MORE hard abstentions under inst (12) than blind (3). These are
verbose responses like `"Based on the instruction to imagine an appropriate image, I would
say..."` that happen to trigger the abstention regex.

These are not genuine abstentions — they are instruction-following verbosity artifacts.
When counting abstentions, filter out responses where the refusal phrase appears after
position 10+ tokens, or use a stricter regex requiring the phrase in the first sentence.

**Quick filter:** `len(output.split()) <= 6 and refusal_re.search(output)` captures genuine
one-phrase refusals; longer responses are likely verbose Qwen3 artifacts.

---

## 7. Coverage Gaps in Blind Condition

Only 4 of 7 models have the blind condition (no inst):
- ✓ Qwen3-VL-8B-Instruct
- ✓ llava-1.5-7b-hf
- ✓ llava-v1.6-mistral-7b-hf
- ✓ llava-v1.6-vicuna-7b-hf
- ✗ Qwen3-VL-32B-Instruct (only has inst_blind)
- ✗ Qwen3-VL-4B-Instruct (only has inst_blind, with broken generated_answers)
- ✗ llava-v1.6-vicuna-13b-hf (only has inst_blind)

For the paper, you can only do blind/inst comparison analysis on the 4 covered models.
Consider running blind condition for Qwen3-VL-32B — the 32B model is the most interesting
for showing how scale affects prior activation and abstention.
