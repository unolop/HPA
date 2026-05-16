# Alignment: Training-Free Interventions for Prior Exploitation

Focus: reduce linguistic prior reliance at **inference time**, using behavioral
signals already characterized in the diagnostic study. Human data (h(q)) used
for evaluation only — never as a training or tuning target.

---

## Core Principle

**No model updates. Correct using inference-time signals.**

The overconfidence finding (80–90% of blind tokens at logprob > −0.5) and the
instruction-gated spectrum (Qwen3-VL: 67% collapse, LLaVA-1.5: 17%) provide
two complementary handles:
1. The model already emits uncertainty signals (soft abstentions, logprob) — we
   can act on them without touching weights.
2. The model's prior gate responds to prompt framing — redirecting the gate costs
   one sentence.

---

## Dataset Assets

| Asset | Source | Role |
|-------|--------|------|
| Blind logprobs | Model (no image) | Confidence thresholding signal |
| Inst_blind logprobs | Model (explicit no-image instruction) | Gate-activation signal |
| Soft abstention rate | Derived from model outputs | Baseline uncertainty estimate |
| Answer-flip rate blind→inst | Derived from model outputs | Gate-sensitivity measure |
| h(q) human difficulty | 36 participants × 113 questions | **Evaluation only** |
| VQA v2 annotations (10/q) | Original dataset | Calibration upper bound |

---

## Approach 1: Confidence-Based Selective Abstention (Post-Hoc, No Training)

**Core idea:** Withhold model answers when blind logprob confidence falls below
a per-question-type threshold. High-confidence blind outputs cluster on bias
patterns (``no'', ``0'', ``black'') — a threshold selectively suppresses these
without touching general VQA performance.

**Implementation:**
```
abstain(q) = True  if mean_logprob(blind, q) < τ_type(q)
```
where `τ_type` is tuned per answer type (yes/no, count, color) on VQA accuracy
— not on h(q).

**What to measure:**
- % answers withheld per model (coverage vs. precision tradeoff)
- VQA accuracy on remaining accepted answers (does selectivity improve precision?)
- Pearson r(selected_acc, h(q)) vs. baseline (does filtering improve alignment with human difficulty as a byproduct?)
- ECE on accepted answers vs. all answers

**Data needed:** Existing blind logprob JSONL files. No new inference.

**Models to run:** Qwen3-VL-8B (primary; has logprob data), LLaVA family.

**Expected result:** Abstaining on low-confidence answers raises accuracy on
accepted answers and reduces model-specific bias outputs (``0'', ``no''). The
accepted-answer difficulty curve should align better with h(q) as biased
low-information answers are removed.

---

## Approach 2: Conditional Abstention Prompting (Zero-Shot, No Training)

**Core idea:** Replace the "imagine the scene" instruction with a skeptical
instruction: *"Only answer if you are confident you can do so from the question
text alone; otherwise say 'I need the image'."* This inverts the permission
signal and tests whether models have self-knowledge of prior reliance.

**Why this is interesting:**
- The standard inst condition shows 67% of soft-abstentions collapse for Qwen3-VL-8B.
  The same gate should work in reverse: asking for explicit uncertainty expression.
- LLaVA-1.5 (17% collapse) has a weak or absent gate — the conditional prompt
  will likely have little effect, confirming unconditional prior exploitation.
- The contrast between models makes architectural claims testable without training.

**Conditions to run:**
| Condition | Prompt |
|-----------|--------|
| `blind` | No instruction (baseline) |
| `inst_blind` | "Answer as if you can see the image" (existing) |
| `skeptical_blind` | "Only answer if confident from question text alone; else say 'I need the image'" |

**What to measure:**
- % explicit abstentions ("I need the image") per model — does the gate engage?
- Distribution shift on yes/no and count answers (does the prior bias reduce?)
- Accuracy on non-abstained answers (does selectivity improve precision?)
- SBERT similarity to human answers (does the conditional prompt improve alignment?)

**Data needed:** One new inference run per model with the skeptical prompt.

---

## Approach 3: Language Prior Correction (Contrastive Decoding, No Training)

**Core idea:** Divide VLM (with image) logprobs by blind logprobs to cancel the
linguistic prior contribution to the answer.

```
p_corrected(a | image, q)  ∝  p_vlm(a | image, q) / p_blind(a | q)^λ
```

- `λ = 0` → original VLM output
- `λ = 1` → full prior removal
- Tune λ on VQA accuracy (not on h(q))

**Why this matters:** Directly operationalizes the McCoy mechanism — subtracting
the corpus-frequency signal from the visual-grounding signal. Prior correction
should most improve accuracy on questions where model-specific biases diverge
from ground truth (count, yes/no type).

**What to measure:**
- VQA accuracy on 1k control set as λ varies (primary)
- Yes/no distribution shift (does ``no'' bias reduce with λ > 0?)
- Count distribution shift (does ``0'' bias reduce?)
- Pearson r(corrected_acc, h(q)) — does alignment with human difficulty emerge?

**Data needed:** Existing blind logprobs + VLM image-conditioned logprobs from
`vqa_1k_control`. No new inference needed for already-run models.

---

## Experiment Plan

### Phase 1 — Confidence Thresholding (Immediate, No New Inference)
- [ ] Extract per-question mean logprob from blind JSONL files for all models
- [ ] Fit per-answer-type thresholds on VQA accuracy (τ_yes/no, τ_count, τ_other)
- [ ] Measure coverage / precision tradeoff curve
- [ ] Validate: Pearson r(accepted_acc, h(q)) before/after

### Phase 2 — Conditional Abstention Prompt (One New Inference Run)
- [ ] Write skeptical prompt variant; add to inference pipeline
- [ ] Run Qwen3-VL-8B and LLaVA-1.5 (contrasting endpoints of the gate spectrum)
- [ ] Compare abstention rates, answer distributions, SBERT human alignment
- [ ] Report: does the gate work in both directions?

### Phase 3 — Contrastive Decoding (Requires Sighted Logprobs)
- [ ] Verify sighted (with-image) logprob files exist for target models
- [ ] Implement λ-sweep; tune on VQA accuracy
- [ ] Report accuracy + distribution bias reduction across λ values
- [ ] Validate: h(q) alignment as byproduct

### Phase 4 — Human Alignment Validation
- [ ] For each intervention: compute SBERT similarity to human answers (per model)
- [ ] Compare Pearson r(corrected_acc, h(q)) before/after each approach
- [ ] Report behavioral shift: blind answer distribution, soft abstention rate
- [ ] Check: do prior-correction gains concentrate on model-specific bias types
  (count, yes/no) rather than shared-prior types (exist, action)?

---

## Key Hypotheses

1. **Confidence thresholding reduces model-specific bias outputs** — abstaining on
   low-LP answers removes the ``0'' and ``no'' defaults disproportionately, since
   these are the overconfident-but-wrong cases
2. **Conditional prompt engages the same gate as inst** — Qwen3-VL models that
   collapse under inst should also respond to the skeptical instruction; LLaVA
   models with weak gates will not
3. **Prior correction improves h(q) alignment as a byproduct** — removing the
   corpus-frequency signal should make the difficulty curve more human-like
   without using h(q) in any optimization

---

## Notes

- Primary model: **Qwen3-VL-8B** (strongest instruction sensitivity, most
  interesting for Approaches 2 and 3)
- Contrast model: **LLaVA-1.5** (unconditional exploiter, tests limits of Approach 2)
- h(q) is a **held-out test set** — do not tune any threshold or hyperparameter on it
- VQA v2 annotations (10 annotators, with image) are the calibration upper bound
  for Approach 1: overconfidence gap = blind_LP − human_agreement_with_image
- All approaches are additive: confidence thresholding + conditional prompt could
  be combined; contrastive decoding is independent
