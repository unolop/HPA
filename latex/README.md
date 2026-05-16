# AAAI Paper — Figures & Analysis Plan

**Title:** When Models Answer Without Seeing: Human-Calibrated Diagnosis of Linguistic Prior Exploitation in Vision-Language Models  
**Track:** AAAI Human-AI Alignment (fallback: main track)  
**Date:** 2026-05-14

---

## Paper Framing & Core Thesis

The paper's central claim is not just that VLMs answer without images — it is that **some of their language priors are human-like while others are model-specific failure modes**. This distinction is the main contribution of §6 (Human Alignment).

Analogy: Poliak et al. (2018) showed NLI models exploit hypothesis-only cues. McCoy et al. (2024) traced this to corpus frequency. We extend both: we measure *which* priors are shared with humans and *which* question-answer types drive that sharing.

---

## §6 Focus: Human–Model Prior Alignment

### Key Finding: Three Regimes of Prior Sharing

**1. Shared prior (genuine human-like linguistic behaviour)**
- Question types: yes/no (exist), action questions
- Entity types: animal, place, food
- Jaccard overlap (top-3 answers): ~0.93 for yes/no, ~0.55 for action
- Per-question r (human acc vs model acc): 0.90 for animal, 0.91 for place
- Interpretation: models and humans converge on the same answer from the same linguistic pattern. For these questions, model blind behaviour is essentially an imitation of human uncertainty.

**2. Partial alignment (shared structure, different magnitude)**
- Question types: attribute, world knowledge
- Entity types: person, food, other
- r ~0.6–0.8, Jaccard ~0.34–0.42
- Interpretation: models capture the easy/hard structure of human difficulty but diverge on ambiguous cases.

**3. Model-specific prior (not human-like, model failure mode)**
- Question types: count (numerical), text/OCR, temporal
- Entity types: object, product
- Jaccard ~0.05–0.13, count r = −0.387
- Model behaviour: collapses to "0" for count (70% of VLM count answers), "no" for yes/no (62% blind)
- Human behaviour: counts are distributed (2 > 1 > 3 > 4 > 0), yes/no lean toward "yes" (67%)
- **Count r is NEGATIVE**: when humans do well on count questions, models tend to do worse and vice versa — a meaningful anti-alignment.

### Supporting Statistics

| Metric | Value |
|---|---|
| SBERT HM pairwise similarity | 0.511 |
| SBERT HH pairwise similarity | 0.512 |
| ICC(2,36) human difficulty | 0.963 |
| Pearson r (human vs VLM inst_blind per question) | 0.605 |
| Pearson r (human vs standalone LLM) | 0.622 |
| Backbone decoder ≈ human–human SBERT | ~matched |

### Backbone Decoder Insight (important for §6)
When the vision encoder is removed (backbone LM only):
- LLaVA-1.5 VLM yes-rate: 0% → LM yes-rate: 57%
- LLaVA-Mistral VLM: 0% → LM: 29%
- LLaVA-Vicuna VLM: 0% → LM: 71%

The vision encoder shifts the yes/no prior toward "no". The backbone LM is closer to human yes-distribution. This is evidence that the "no" bias is a VLM-specific failure mode, not a linguistic prior.

---

## Figures in `latex/AnonymousSubmission/LaTeX/figures/`

### Ready to use

| File | Description | Paper section |
|---|---|---|
| `fig_hm_alignment.png` | Left: Jaccard overlap by question type. Right: Pearson r by entity type. Two-panel. | §6 Human Alignment |
| `fig_hm_quadrant.png` | Per-question scatter: human acc (x) vs VLM acc (y), colour = Jaccard overlap. Four quadrants: shared failure / shared knowledge / human-only / model-only. r=0.605 | §6 Human Alignment |
| `fig_dist_answer_bias.png` | Left: yes/no distribution (model vs human vs GT). Right: count distribution (model vs human). VLM blind. | §4 Answer Distribution Biases |
| `abstention_rates.png` | Output classification (hard-abstained / soft-abstained / hallucinated correct / wrong) by model, blind vs inst_blind | §5 Characterizing Prior Reliance |
| `abstention_collapse.png` | Soft abstention collapse rate per model (blind → inst_blind) | §5 Instruction-Gated Hallucination |
| `abstention_bias.png` | Top-15 output frequencies with class colour, blind vs blind+inst | §4 / §5 |
| `lm_decoder_top_answers.png` | Top answer distributions pooled across 9 models, LM decoder vs VLM, blind vs inst_blind | §5 / §6 |

### Still needed (paper has placeholder boxes)

| Label | Description | Source |
|---|---|---|
| `fig_degradation` | Accuracy + confidence across control variants C→B→A, one line per model | Requires unified scoring; see notes/data_issues.md |
| `fig_collapse` | Bar chart: soft abstention collapse rate per model | Use `abstention_collapse.png` or regenerate |
| `fig_mg` | Multimodal Gain by correctness quadrant (VQA + MMStar) | analysis/session2/12_align_quadrant.ipynb |

---

## VLM Family Behavioral Summary

### LLaVA (1.5, Mistral, Vicuna) — unconditional prior exploiters
- Yes-rate 0%, zero-rate 95–100%: hardest biases in the dataset
- Low change rate (14–24%): instruction-resistant
- LLaVA-1.5 collapse: negative (instruction increases hedging, not decreases)
- **Model-specific failure mode**, not human-like

### Qwen3-VL-8B — instruction-gated
- Same blind biases (0% yes, 91% zero) but 67% soft abstention collapse and 32% answer change
- The instruction "unlocks" latent uncertainty awareness
- Most human-like in terms of responsiveness

### InternVL (1B, 2B, 8B) — inconsistent across scale
- Yes-rate erratic: 1B=71%, 2B=14%, 8B=71% — no monotonic trend
- High change rates (29–43%), zero soft abstention
- No systematic prior exploitation pattern

### Standalone LLM (Qwen3 family)
- Scale-dominated: <2B ≈ random (8–9% acc), 8B–32B ≈ 27–37%
- Mistral-7B and Vicuna-13B: degenerate (0%) — chat fine-tuned without image context
- Phi-3.5-mini: always "yes" (100%), 0% zero-rate — opposite extreme to LLaVA
- Think mode: minimal benefit, hurts small models (higher instability)

---

## What Kinds of Questions Show Shared Language Priors?

### Shared (model ≈ human):
- **"Is there a ___?"** type yes/no questions (exist): both converge on same answer ~93% of the time
- **Action questions** ("Is she eating it?", "Is it on?"): ~55% overlap
- **Animal/place/food entities**: difficulty curves almost perfectly correlated (r > 0.77)
- **World knowledge** ("Are cats really afraid of water?"): both get it right for the same reasons

### Diverged (model ≠ human):
- **Count questions**: model says "0", humans give distributed counts. r = −0.37 (anti-aligned)
- **Text/OCR and temporal questions**: near-zero overlap, models have no useful prior
- **Object/product entities**: low r (0.13 / −0.48), model difficulty doesn't match human difficulty

---

## Human Study (Session 2)
- N = 36 participants (15M / 21F, age 18–34)
- 113 questions (VQA 1K subset), variant C (original), inst_blind condition
- ICC(2,36) = 0.963 — highly reliable difficulty curve
- Human data is **inst_blind only** — limitation to note in paper

---

## TODO for Next Session
- [ ] Generate `fig_mg` (multimodal gain quadrant) from `12_align_quadrant.ipynb`
- [ ] Rewrite §6.3 to use the three-regime framework above
- [ ] Add `fig_hm_alignment.png` and `fig_hm_quadrant.png` into paper.tex
- [ ] Update paper text with count r = −0.387 and backbone decoder yes-rate reversal finding
- [ ] Verify image ablation results (gray/noise runs for Qwen3-VL-8B) when tmux completes
