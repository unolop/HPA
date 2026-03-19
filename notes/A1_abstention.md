# A1: Abstention Analysis Plan

## Pipeline Verdict

The pipeline is correct — no `blind_inst` instruction is added under `--condition blind`.
The model receives a blank image and a question ending in `\nAnswer:` with no guidance on
what to do.

## Known Issue: Baked-In Instruction

The prompt still contains `"Answer the question using a single word or phrase."` — baked
into `VQADataset_json`, not controlled by condition. This may suppress natural abstentions
because the model is explicitly told to answer.

If you want to measure natural abstention rates, add a separate condition that strips this
instruction: just `"Question: {q}\nAnswer:"` for blind.

---

## What to Measure

Does the model recognize it can't answer from a blank image, and what predicts whether it
tries anyway?

---

## Step 1 — Abstention Detection

Classify each blind output into one of:

| Class | Definition |
|-------|-----------|
| `abstained` | Output contains a refusal phrase |
| `hallucinated_correct` | Output matches GT despite blank image |
| `hallucinated_wrong` | Plausible VQA-style answer, doesn't match GT |
| `degenerate` | Empty, `\n`, or `"Answer:"` repeated |

```python
refusal_re = re.compile(
    r"cannot|can't|don't see|no image|unable|not sure|unclear|n/a", re.I
)
```

---

## Step 2 — Abstention Rate by Tier

Cross-tabulate abstention rate against A–E tiers.

**Hypothesis:** Tier A questions (model has strong priors) will have the lowest abstention
rate — the model hallucinates confidently even with a blank image. Tier D/E questions may
have higher abstention because the referential anchor (the entity name) is what makes the
model confident.

---

## Step 3 — Abstention Rate by Op

Text-reading questions (`op=text`) should abstain most — a model legitimately can't read a
sign from a blank image. Count and attribute questions will hallucinate more freely.

This is the cleanest operationalization of "which cognitive operations are grounded vs.
prior-driven."

---

## Step 4 — Confidence on Abstained vs. Answered Outputs

Compare `mean(confidence)` between abstained and answered outputs.

**Hypothesis:** answered outputs have higher mean token probability — the model commits more
strongly when it confabulates. This ties logprob calibration back to abstention behavior.

---

## Step 5 — Blind vs. Inst Condition Comparison

For the same question, what fraction of models:

| Transition | Interpretation |
|-----------|----------------|
| answered blind → answered inst | Stable hallucination — pure text prior |
| abstained blind → answered inst | Instruction unlocks the prior |
| abstained blind → abstained inst | Genuinely hard even with permission |

The **abstained blind → answered inst** group is most interesting: the model knows the
answer when allowed to use priors but suppresses it without explicit permission. Strong
signal that inst answers come from training statistics, not visual grounding.

---

## Step 6 — Tie to Tier

A gold question is one that:
1. Gives a confident wrong answer in standard evaluation
2. Abstains under blind
3. Confidently answers under inst

This is the canonical example of a "corpus prior" question for the human study.
