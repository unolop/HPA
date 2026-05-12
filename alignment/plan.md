# Alignment: Reducing Linguistic Prior Exploitation in VLMs

Using the human study dataset (h(q) difficulty curve, blind/inst_blind logprobs, variant triples A/B/C)
to reduce hallucination and shortcut behavior in vision-language models.

---

## Dataset Assets

| Asset | Description |
|-------|-------------|
| `h(q)` | Human difficulty per question — mean accuracy across 36 participants |
| Blind logprobs | Model output distribution with no image (pure linguistic prior) |
| Inst_blind logprobs | Model output with explicit "no image" instruction |
| Variant triples | A (pronominalized), B (weaker_object), C (original) per question |
| Answer-flip rate | % of answers that change blind → inst (instruction sensitivity) |
| Soft abstention | "none/nowhere/unanswerable" outputs under blind condition |

---

## Approach 1: Language Prior Correction (Test-Time)

**Core idea:** Divide vision model logprobs by blind model logprobs to cancel the linguistic prior.

```
p_corrected(a | image, q)  ∝  p_vlm(a | image, q) / p_blind(a | q)^λ
```

- `λ = 0` → original VLM output (no correction)
- `λ = 1` → full prior removal
- `λ` can be tuned on the human-difficulty curve

**Why it works:** The blind model IS the language prior. Dividing it out forces the model to rely on visual evidence.

**What to measure:**
- VQA accuracy on 1k control set before/after correction
- Alignment with h(q): does Pearson r(model_acc, h(q)) improve?
- ECE / reliability diagram: does calibration improve?

**Data needed:** Existing blind logprobs + VLM (image-conditioned) logprobs — no new inference required for already-run models.

**References:** Ramakrishnan et al. (2018), Chen et al. (2020 — CSS), Liang et al. (2020 — LMHM)

---

## Approach 2: Human-Calibrated DPO

**Core idea:** Build preference pairs from the human difficulty signal and fine-tune with DPO.

**Pair construction:**
- **Rejected:** model's confident blind answer on questions where `h(q) < 0.3` (humans also fail → pure prior exploitation)
- **Chosen:** abstention / hedged response OR the correct ground-truth answer

**High-value training examples:**
- Questions with high answer-flip rate (blind→inst): model is instruction-sensitive, meaning prior exploitation is gated
- Soft abstention examples (none/nowhere/unanswerable) as natural "chosen" responses for hard questions

**Training target:** Model should express uncertainty on low-h(q) questions without an image, and answer confidently when visual evidence is available.

**Variants:**
- `DPO-hard`: only questions with `h(q) < 0.3`
- `DPO-all`: full difficulty spectrum, weight pairs by `(1 - h(q))`
- `DPO-abstain`: chosen = explicit uncertainty expression ("I cannot answer without seeing the image")

---

## Approach 3: Variant Consistency Fine-tuning

**Core idea:** A model exploiting corpus-frequency shortcuts gives inconsistent answers across A/B/C rephrasings of the same question. Visual grounding should be rephrasing-invariant.

**Loss:** Consistency penalty across variants

```
L_consistency = KL(p(a | q_A) || p(a | q_B)) + KL(p(a | q_A) || p(a | q_C))
```

Applied in blind condition — if blind answers are inconsistent across variants, the model is exploiting surface-level corpus patterns, not semantic content.

**Fine-tuning:**
- SFT or auxiliary loss added to standard VQA training
- No additional human labels needed — variant triples are already in the dataset

**What to measure:**
- Variance of blind answers across A/B/C variants before/after
- Whether consistency training reduces blind accuracy (desirable: model should be less sure without image)
- Whether VQA accuracy with image is preserved

---

## Approach 4: Difficulty-Aware Confidence Calibration

**Core idea:** Post-hoc calibration using h(q) as the target confidence. No model weight changes.

**Method:**
1. Train a question difficulty predictor: `f(question) → h_predicted(q)` using {question text, entity type, operation type} as features, trained on the 113+ labeled questions
2. At test time, predict `h_predicted(q)` for any new question
3. Apply isotonic regression or temperature scaling: map raw logprob → calibrated confidence anchored to `h_predicted(q)`

**Simpler version:** Just fit a per-question-type temperature T using your existing data
- e.g., yes/no questions: T_yesno; color questions: T_color; count questions: T_count

**What to measure:**
- ECE before/after calibration
- Reliability diagram: does calibrated confidence track h(q)?
- Brier score on blind predictions

---

## Approach 5: Abstention Fine-tuning

**Core idea:** The "Answer using a single word or phrase" instruction suppresses all explicit refusals (hard abstention: 0–0.3%). Fine-tune to recover appropriate uncertainty expression.

**Steps:**
1. Remove the "single word" constraint from the prompt template
2. For questions with `h(q) < 0.2` AND `blind_accuracy < 0.2`: generate target responses expressing uncertainty ("I cannot determine this without seeing the image")
3. For questions with `h(q) > 0.7`: keep standard confident answers
4. SFT on this mixed dataset

**Risk:** May over-suppress answers on genuinely answerable questions. Need careful held-out evaluation.

---

## Experiment Plan

### Phase 1 — Analysis (no training, immediate)
- [ ] Implement language prior correction (λ-sweep on existing logprobs)
- [ ] Compute ECE / reliability diagrams for all models
- [ ] Measure Pearson r(corrected_acc, h(q)) vs r(original_acc, h(q))
- [ ] Quantify variant inconsistency rate per model under blind condition

### Phase 2 — Calibration (post-hoc, no fine-tuning)
- [ ] Fit isotonic regression on {blind_logprob → h(q)} mapping
- [ ] Train question difficulty predictor (generalize h(q) to unseen questions)
- [ ] Evaluate calibration on held-out question split

### Phase 3 — Fine-tuning
- [ ] Construct DPO preference pairs from human study data
- [ ] Fine-tune Qwen3-VL-8B with DPO-hard (h(q) < 0.3 filter)
- [ ] Fine-tune with variant consistency auxiliary loss
- [ ] Evaluate: blind accuracy (should decrease), image-conditioned accuracy (should be preserved), h(q) alignment (should improve)

### Phase 4 — Evaluation
- [ ] Human study on fine-tuned model: does h(q) alignment improve?
- [ ] Compare behavioral shift: blind → inst flip rate before/after fine-tuning
- [ ] Measure soft abstention rate change
- [ ] Report ECE, Pearson r, Krippendorff α with humans before/after

---

## Key Hypotheses

1. **Prior correction improves h(q) alignment** — removing the language prior should make the model's difficulty curve look more like the human difficulty curve
2. **Consistency training reduces blind overconfidence** — without changing image-conditioned performance
3. **DPO on hard questions teaches appropriate uncertainty** — without degrading general VQA performance
4. **Calibrated models transfer better** — a model calibrated to h(q) on 113 questions should generalize to the full 1k question set

---

## Notes

- All fine-tuning should use **Qwen3-VL-8B** as the base (strongest alignment signal in existing data)
- Evaluation baseline: existing blind/inst_blind logprob files — no need to re-run inference for Phase 1
- Human study for Phase 4 can reuse the existing experimental pipeline (by_participant JSON format)
- Keep VQA accuracy with image as the primary guardrail — alignment should not come at the cost of visual capability
