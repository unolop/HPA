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

## Empirical Findings (Qwen3-VL-8B, LLaVA series — 1K blind outputs)

### Hard Abstention Rate is Near-Zero

| Model | n | Hard abs | Soft abs | Combined |
|-------|---|----------|----------|----------|
| Qwen3-VL-8B-Instruct | 1000 | 3 (0.30%) | 129 (12.9%) | 132 (13.2%) |
| llava-1.5-7b-hf | 1000 | 1 (0.10%) | 187 (18.7%) | 188 (18.8%) |
| llava-v1.6-mistral-7b-hf | 1000 | 0 (0.00%) | 158 (15.8%) | 158 (15.8%) |
| llava-v1.6-vicuna-7b-hf | 1000 | 2 (0.20%) | 113 (11.3%) | 115 (11.5%) |

**Hard abstention** = explicit refusal phrases ("I cannot", "no image", etc.)
**Soft abstention** = outputs like `nothing`, `none`, `unanswerable`, `nowhere`, `no sign`

The baked-in `"Answer the question using a single word or phrase."` instruction is almost
entirely suppressing explicit refusals. This is the expected behavior — models comply with
the directive and hallucinate instead.

**The real abstention signal is soft abstention (11–19%)**. These outputs are the model
hedging ("nothing", "none") rather than committing to a specific hallucination. They are
worth separating from genuine hallucinations in the analysis.

### Regex Warning: False Positives

The word `blank` in the regex fires on the legitimate answer `"blanket"`. Of Qwen3's 3
"hard abstentions", 2 are actually `"blanket"`. The only genuine hard abstention was
`"no image"`.

**Fix:** Use word-boundary anchoring for `blank`:
```python
refusal_re = re.compile(
    r"cannot|can't|don't see|no image|unable|not sure|unclear"
    r"|i cannot|i don't|can not|without|no visual|not visible"
    r"|cannot see|i'm not|i am not|\bblank\b",  # \b prevents matching "blanket"
    re.I
)
```

### Soft Abstention Collapse Under Inst

When given the inst instruction ("imagine an appropriate image..."), soft abstentions
largely convert to specific answers:

| Model | soft-abs blind | soft-abs inst | collapse rate |
|-------|----------------|---------------|---------------|
| Qwen3-VL-8B | 12.9% | 3.8% | **71.9%** |
| llava-v1.6-mistral | 15.8% | 7.8% | 51.9% |
| llava-v1.6-vicuna-7b | 11.3% | 9.5% | 28.3% |
| llava-1.5-7b | 18.7% | 16.0% | **17.1%** |

This ordering is highly interpretable: Qwen3 is strongly instruction-following, so the inst
prompt almost completely activates its prior. LLaVA 1.5, by contrast, barely registers the
instruction — it just hallucinates regardless of whether permission is given. The collapse
rate is essentially a measure of how much a model's hallucination behavior is
instruction-gated vs. always-on.

**Paper framing:** LLaVA 1.5 represents "unconditional hallucination" — it answers freely
without requiring permission. Qwen3 represents "conditional prior activation" — it has a
strong prior but gatekeeps it behind the instruction. This distinction is mechanistically
interesting.

### Inst Does Not Eliminate Abstention

Counterintuitively, Qwen3-VL-8B produces MORE hard abstentions under inst (12 vs 3) than
blind. These are long-form responses like `"Based on the instruction to imagine an
appropriate image..."` that get caught by the regex. This is performative abstention —
the model signals compliance with the instruction format while nominally refusing. It's
not genuine uncertainty; it's a different failure mode (verbosity + instruction following)
that is an artifact of Qwen3's training.

**Recommendation:** For hard abstention counting, filter out responses > 15 tokens as likely
Qwen3 "instruction-following refusals" rather than genuine blind abstentions. Or use a
stricter regex that requires the refusal phrase to appear in the first 10 tokens.

---

## What to Measure

Does the model recognize it can't answer from a blank image, and what predicts whether it
tries anyway?

---

## Step 1 — Abstention Detection

Classify each blind output into one of:

| Class | Definition |
|-------|-----------|
| `hard_abstained` | Explicit refusal phrase (use word-boundary-anchored regex) |
| `soft_abstained` | nothing / none / unanswerable / nowhere — hedging without committing |
| `hallucinated_correct` | Output matches GT despite blank image |
| `hallucinated_wrong` | Plausible VQA-style answer, doesn't match GT |
| `degenerate` | Empty, `\n`, or "Answer:" repeated |

Note: soft abstentions are the more meaningful category — hard abstentions are near-zero
because the prompt instructs the model to answer.

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

**Expected finding:** `op=text` → high soft abstention ("nothing" for reading questions);
`op=exist` and `op=count` → zero bias ("no", "0") rather than abstention;
`op=attr` (color) → high-confidence hallucination ("black", "white").

---

## Step 4 — Confidence on Abstained vs. Answered Outputs

Compare `mean(confidence)` between abstained and answered outputs.

**Empirical result (Qwen3-VL-8B blind):**
- Hard abstentions: mean LP = -0.37
- Answered outputs: mean LP = -0.26

Direction is correct (abstentions less confident), but effect size is small. The real test
is comparing soft-abstained outputs ("nothing") vs. committed hallucinations — soft
abstentions should have lower LP than strong hallucinations like specific color/object names.

**Confidence by condition (Qwen3-VL-8B):**
| Condition | Mean LP | High conf (LP > -0.5) |
|-----------|---------|----------------------|
| Blind | -0.258 | 80.8% |
| Inst blind | -0.177 | 89.5% |

The inst instruction raises confidence uniformly +0.08 LP across all control types.
This is consistent with "instruction unlocks prior commitment": the model becomes more
certain when given permission to use its priors.

---

## Step 5 — Blind vs. Inst Condition Comparison

For the same question, what fraction of models:

| Transition | Interpretation |
|-----------|----------------|
| answered blind → answered inst | Stable hallucination — pure text prior |
| soft-abstained blind → answered inst | Instruction unlocks the prior |
| soft-abstained blind → soft-abstained inst | Genuinely hard even with permission |

**Empirical result (Qwen3-VL-8B, 994 matched records):**
- Same answer across conditions: 66.4% (question type), up to 75.1% (subject_ablated)
- 33.6% of answers change between blind and inst — larger than expected
- Soft abstention collapse rate: 71.9% (see table above)
- Yes/no shift: blind strongly biases toward "no" (70%); inst shifts toward "yes" (55%)

**The yes-bias shift is interesting:** models give "no" when they have no information
(default uncertainty = negative), but shift toward "yes" when instructed to imagine a
plausible scenario. This means the "no" bias in blind is not necessarily linguistic prior —
it may be a form of epistemic humility. The "yes" shift under inst reflects corpus prior
(most described scenes exist).

The **soft-abstained blind → answered inst** group is most interesting: the model knows the
answer when allowed to use priors but suppresses it without explicit permission. Strong
signal that inst answers come from training statistics, not visual grounding.

---

## Step 6 — Tie to Tier

A gold question is one that:
1. Gives a confident wrong answer in standard evaluation
2. Soft-abstains under blind (outputs "nothing", "none", "nowhere")
3. Confidently answers under inst

This is the canonical example of a "corpus prior" question for the human study.

---

## Output Distribution (Blind, Qwen3-VL-8B)

Top outputs: `no`(27%), `yes`(12%), `0`(9%), `black`(8%), `nothing`(4%), `none`(4%),
`nowhere`(2%), `white`(2%), `unanswerable`(2%)

**Zero bias:** 75% of numeric blind answers are "0" — the model's null answer for count
questions is zero, not "unknown". This is a meaningful prior: "if I can't see it, there
aren't any."

**No bias:** 70% of yes/no blind answers are "no" — models default to negation under
uncertainty, possibly learned from training examples where things-are-absent get "no".

**Color bias:** "black" dominates color responses (16%) under blind — the modal color in
training data? or a genuine uncertainty-default?

Under inst, content-specific answers appear (`red`, `standing`, `blue`, `grass`, `apple`,
`cat`, `stop`) that are absent from blind's top outputs. These are the activated priors.
