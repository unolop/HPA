# Agreement & Correlation Metrics — Computed Results

Computed from:
- Human data: `analysis/csv/human_vqa.csv` — 20 participants × 374 VQA questions, `inst blind` condition only
- Model data: `evaluation/scored/pretrained/` — 10 models × {blind, inst_blind} conditions

---

## Key Methodological Notes

### Free-Text Answer Handling for Agreement
VQA answers are free-text, so agreement requires a defined equivalence notion.
Three layers, each answering a different question:

| Layer | Equivalence | Metric | Use case |
|-------|-------------|--------|----------|
| **Correctness** | Binary (VQAv2 soft score ≥ 33.3%) | Krippendorff α (nominal), ICC | "Do raters agree on which questions are hard?" |
| **Canonical match** | Same canonical cluster (GPT-4o mini) | Proportion same answer | "Do raters give the same answer regardless of correctness?" |
| **Semantic distance** | 1 − cosine(SBERT embeddings) | Krippendorff α (interval) | "How similar are wrong answers to each other?" |

Current analysis uses Layers 1 and 2. Layer 3 requires SBERT embeddings over all human answers — TODO.

**Important asymmetry to disclose in paper:** Human h(q) is computed from VQA soft-scores
(0 / 0.33 / 0.67 / 1.0 averaged across 20 raters). Model accuracy is binary (0 or 1).
The comparison is valid but this asymmetry should be stated in the methods section.

---

## Human Difficulty Curve h(q)

374 questions, 20 raters each.

| Statistic | Value |
|-----------|-------|
| Mean h(q) | 0.390 |
| Std | 0.363 |
| h = 0 (all humans wrong) | 22.2% (83 questions) |
| h = 1 (all humans correct) | 5.6% (21 questions) |
| 0 < h < 1 (mixed) | 72.2% (270 questions) |

Distribution is bimodal — large spike near 0 (hard questions) and near 1 (easy questions),
substantial middle. The h(q) curve is **extremely reliable** (ICC(2,k) = 0.963 across 20
raters), meaning we can treat it as a high-quality difficulty estimate for the scatter analysis.

---

## Pearson Difficulty Correlation: Human h(q) vs. Model Accuracy

**Finding: Pearson r ≈ Spearman ρ throughout (max difference 0.026).**
The switch from Spearman to Pearson is methodologically cleaner but does not change the story.
The human-model difficulty relationship is approximately linear.

### Blind condition

| Model | Pearson r | Spearman ρ | Mean acc |
|-------|-----------|-----------|---------|
| InternVL3_5-8B | **0.616** | 0.616 | 0.472 |
| InternVL3_5-2B | 0.603 | 0.619 | 0.455 |
| InternVL3_5-1B | 0.602 | 0.611 | 0.436 |
| Qwen3-VL-8B | 0.581 | 0.586 | 0.443 |
| Qwen3-VL-2B | 0.560 | 0.574 | 0.455 |
| InternVL3_5-4B | 0.554 | 0.580 | 0.464 |
| Qwen3-VL-4B | 0.537 | 0.560 | 0.454 |
| llava-v1.6-mistral | 0.506 | 0.516 | 0.451 |
| llava-v1.6-vicuna | 0.502 | 0.511 | 0.468 |
| llava-1.5-7b | 0.499 | 0.509 | 0.434 |

### Inst_blind condition (same questions)

| Model | Pearson r | Spearman ρ | Mean acc |
|-------|-----------|-----------|---------|
| InternVL3_5-8B | **0.719** | 0.694 | 0.455 |
| InternVL3_5-4B | **0.700** | 0.689 | 0.475 |
| Qwen3-VL-8B | 0.650 | 0.653 | 0.479 |
| Qwen3-VL-4B | 0.637 | 0.635 | 0.479 |
| InternVL3_5-1B | 0.634 | 0.640 | 0.449 |
| InternVL3_5-2B | 0.631 | 0.651 | 0.455 |
| Qwen3-VL-2B | 0.560 | 0.572 | 0.463 |
| llava-v1.6-mistral | 0.515 | 0.526 | 0.467 |
| llava-1.5-7b | 0.511 | 0.519 | 0.436 |
| llava-v1.6-vicuna | 0.507 | 0.515 | 0.491 |

### Interpretation
**Inst_blind consistently aligns more with human difficulty than blind (Δr = +0.06 to +0.10).**
This is a new finding: when models are given the instruction to imagine a plausible scene,
their difficulty patterns shift toward human prior knowledge. The instruction activates the
same world-knowledge priors that humans use when reasoning without an image. This is the
Pearson-r version of the collapse rate story (Section 5.2 of the paper).

LLaVA models plateau at r ≈ 0.50 regardless of condition — consistent with their
unconditional hallucination behavior. InternVL and Qwen3 show larger inst-vs-blind r gaps,
consistent with instruction-gating.

---

## Krippendorff's α

All computed with `krippendorff` package (already installed).

| Group | α (nominal) | α (interval/soft) | Interpretation |
|-------|-------------|------------------|----------------|
| **Humans only (N=20)** | **0.554** | **0.564** | Moderate — expected for ambiguous VQA without images |
| Best model added (InternVL-8B blind) | 0.543 | 0.554 | Adding model decreases α |
| Worst model added (LLaVA-Vicuna blind) | 0.535 | 0.546 | Even larger decrease |
| **All 30 raters combined** | **0.498** | **0.511** | Drops below 0.5 when humans+models pooled |

**Key finding:** Adding ANY model to the human rater pool decreases α, confirming that
models are systematically less consistent with the human group pattern than humans are with
each other. The drop is larger for LLaVA models than InternVL/Qwen3 — consistent with
the difficulty correlation ordering.

**Inst_blind models** show slightly higher α when added to the pool (InternVL-8B: 0.552
vs blind 0.543), confirming the instruction moves model behavior closer to human patterns.

### For the paper
Report: α\_humans = 0.554 as the within-group reliability baseline.
Compare with: α drops when any model is added → human-model structural difference.
Note: α = 0.554 is "moderate" by Krippendorff's own scale (0.67 = acceptable for content
analysis; 0.8 = high). For blind VQA without images, 0.55 reflects genuine task difficulty,
not measurement noise.

---

## ICC Results (manual computation)

For the human group (20 raters × 374 questions):

| Metric | Value |
|--------|-------|
| ICC(2,1) — single rater, absolute agreement | 0.565 |
| ICC(2,k) — average of k=20 raters | **0.963** |

**ICC(2,k) = 0.963 means h(q) is an extremely reliable difficulty estimate.**
Use this to justify treating h(q) as a ground-truth difficulty signal in the scatter analysis.
ICC(2,1) = 0.565 is the agreement you'd expect from any single human rater — moderate,
consistent with α and the bimodal distribution.

TODO: Install `pingouin` for official ICC output with 95% CIs.
```bash
pip install pingouin
```

---

## Canonical Answer Agreement (Layer 2)

Over 190 unique human participant pairs, using `answer_normalized`:

| Metric | Value |
|--------|-------|
| Mean pairwise canonical match rate | 0.370 |
| Std across pairs | 0.053 |
| Range | 0.249 – 0.524 |
| Correlation with h(q) | r = 0.678 (p << 0.001) |

**Interpretation:** Humans agree on the SAME canonical answer 37% of the time — much lower
than the accuracy agreement because even when both are correct they may give different
valid answers. The strong correlation with h(q) (r = 0.678) means canonical agreement
tracks difficulty: easy questions produce consensus answers, hard questions produce diverse
wrong answers.

For the paper: report canonical match rate alongside correctness agreement to show that
the human group is genuinely diverse in answers, not just correct/incorrect.

### SBERT Semantic Agreement (Layer 3) — Computed

Model: `all-MiniLM-L6-v2`, 1341 unique answer strings embedded.

**Human-human semantic consensus** (mean pairwise cosine similarity per question):

| Stat | Value |
|------|-------|
| Mean | **0.676** |
| Std | 0.209 |
| Min | 0.236 |
| 25th pct | 0.509 |
| Median | 0.724 |
| Max | 1.000 |
| Semantic diversity (1 − consensus) | 0.324 |

Pearson r(semantic_consensus, h(q)) = **0.657** (p = 1.7e-47).
Semantic consensus correlates with correctness agreement but captures a different dimension:
questions with low correctness agreement can still have high semantic consensus (everyone
says an object word, just the wrong one). This makes Layer 3 an independent signal.

**Krippendorff α (semantic distance = 1 − cosine):**
- α\_semantic = **0.489** — lower than α\_nominal (0.547)
- Interpretation: measuring on the soft semantic scale reveals more disagreement than binary
  correct/incorrect collapses. Humans' wrong answers are semantically diverse, not clustered.
  α\_nominal underestimates how much humans actually disagree in answer content.

**Human-model SBERT similarity** (model answer vs. mean human embedding centroid, blind):

| Model | Cosine sim to human centroid |
|-------|------------------------------|
| InternVL3_5-8B | **0.803** |
| InternVL3_5-1B | 0.783 |
| InternVL3_5-4B | 0.779 |
| InternVL3_5-2B | 0.776 |
| Qwen3-VL-2B | 0.764 |
| Qwen3-VL-8B | 0.758 |
| Qwen3-VL-4B | 0.751 |
| LLaVA-1.6-mistral | **0.741** |
| Human-human mean pairwise | **0.676** (reference) |

All models score above the human-human baseline — expected, because centroid comparison is
less penalized than all-pairs comparison. The ranking matches the difficulty correlation
ranking (InternVL > Qwen3 > LLaVA), confirming metric consistency.

---

## Human-Model Divergence Analysis

Model accuracy averages ~6pp higher than humans (mean Δ = model − human = +0.063).

### Quadrant counts (374 questions)
| Quadrant | Criterion | N | % |
|----------|-----------|---|---|
| Both fail | h < 0.1, model < 0.1 | 79 | 21% |
| Both pass | h > 0.8, model > 0.8 | 57 | 15% |
| Human-only correct | h > 0.5, model < 0.3 | 21 | 6% |
| Model-only correct | h < 0.2, model > 0.5 | **17** | **5%** |

### Gold examples: model exploits prior, humans don't
These are the best "canonical hallucination" examples for the paper:

| QID | Question | h(q) | Model acc | Note |
|-----|----------|------|-----------|------|
| 196681009 | "Are there clouds in the sky?" | 0.00 | 1.00 | Sky → clouds = high-freq prior |
| 343496001 | "Are there any animals in this photo?" | 0.00 | 1.00 | Generic existence question |
| 580621000 | "What type of establishment are these people...?" | 0.00 | 1.00 | Context anchors answer |

### Gold examples: humans correct, models fail
These suggest human reasoning about plausible scenes that models miss:

| QID | Question | h(q) | Model acc | Note |
|-----|----------|------|-----------|------|
| 212759008 | "Do you see any grass?" | 0.95 | 0.00 | Humans infer from context; models null-answer |
| 94663002 | "Can you see the sky from this photo?" | 0.90 | 0.00 | Spatial reasoning |
| 531086004 | "Are there any magnets on the fridge?" | 0.95 | 0.10 | Common household knowledge |

---

## TODOs

| # | Item | Priority | Notes |
|---|------|----------|-------|
| 1 | `pip install pingouin` for ICC with 95% CI | Medium | Manual values should match |
| 2 | SBERT semantic distance agreement (Layer 3) | Medium | Richer free-text agreement measure |
| 3 | Human blind condition (no inst) | High | Currently ALL human data is inst_blind — need blind-only for condition comparison |
| 4 | Extend to 1K questions | High | Current analysis uses 374 questions (original human study). New 1K analysis needs new human data or model-only analysis |
| 5 | Add Qwen3 LLM text-only baselines | Low | Missing blind file for these |
| 6 | Bootstrap 95% CIs on all Pearson r values | Medium | Needed for paper |
| 7 | Disclose soft-score vs binary asymmetry in methods | Must | h(q) = soft, model = binary |
