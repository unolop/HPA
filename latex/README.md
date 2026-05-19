# AAAI Paper — Figures & Analysis Plan

**Title:** When Models Answer Without Seeing: Human-Calibrated Diagnosis of Linguistic Prior Exploitation in Vision-Language Models  
**Track:** AAAI Human-AI Alignment (fallback: main track)  
**Date:** 2026-05-18

---

## Paper Framing & Core Thesis

The paper's main claim is now:

**Blind prior exploitation is pervasive, but the model group that most closely matches human blind priors is the VLM backbone decoder, not the full VLM.**

The supporting contrast is equally important:
- full VLMs and standalone LLMs do not fail in the same way
- some priors are shared with humans
- others are model-specific failure modes

The paper should stay focused on diagnosis and alignment, not intervention.

---

## Intended Results Order

### F1. Prior exploitation is pervasive
- Models answer a substantial fraction of VQA questions correctly without any image.
- This establishes that blind VQA is meaningfully exploitable through language-only priors.

### F2. VLM backbone decoders are the human-like group
- This is the main finding.
- Backbone decoders achieve the highest human-model semantic agreement.
- They are closer to human blind priors than full VLMs or standalone LLMs.

### F3. Full VLMs and standalone LLMs show different failure regimes
- Full VLMs show stronger null-hypothesis blind defaults such as no-bias and zero-bias.
- Standalone LLMs are more heterogeneous and often further from human answer semantics.

### F4. Blind success is structured, not random
- Models overproduce defaults such as:
  - `no` for yes/no
  - `0` for count
  - `black` for color
- These patterns show that blind accuracy comes from structured shortcuts, not general robust reasoning.

### F5. Question / entity analyses explain where alignment holds and breaks
- Stronger overlap:
  - yes/no and action questions
  - animal and place entities
- Divergence:
  - count questions
  - object / product / OCR-heavy settings

### F6. Blind models are overconfident
- Blind answers are often produced with high mean token log-probability even when wrong.
- Instruction raises commitment more than it improves grounding.

---

## Current Paper Structure

The draft should read in this order:

1. Blind exploitation exists.
2. The main architectural result is that backbone decoders are closest to humans.
3. Full VLMs and standalone LLMs diverge in different ways.
4. Bias patterns and entity/question-type analyses explain why.
5. Instruction and confidence analyses characterize how the priors are expressed.

This means:
- `Human-Model Alignment` is the scientific center of the paper.
- `Distribution biases`, `instruction gating`, and `confidence` are supporting mechanism sections.
- `Control degradation` is useful diagnostic evidence, but should not overshadow the main alignment result.

---

## Figures in `latex/AnonymousSubmission/LaTeX/figures/`

### Main figures

| File | Role | Paper use |
|---|---|---|
| `fig_overview.png` | Main early overview | Blind success + human relation + control degradation |
| `fig_scatter_agreement.png` | Main alignment figure | Shows backbone decoders are closest to humans |
| `fig_dist_answer_bias.png` | Bias evidence | Shows blind defaults are structured |
| `fig_hm_alignment.png` | Entity-type support | Explains where alignment holds and breaks |
| `fig_instruction_effect.png` | Behavioral mechanism | Shows instruction-gated vs unconditional prior expression |

### Secondary / appendix-leaning figures

| File | Role |
|---|---|
| `fig_hm_quadrant.png` | Useful support for per-question divergence, but not the main figure |
| `sbert_heatmap_groups.png` | Structural confirmation of model-group separation |
| `abstention_rates.png` | More detailed abstention breakdown |
| `abstention_collapse.png` | Model-wise collapse detail |
| `lm_decoder_top_answers.png` | Supporting qualitative comparison |

---

## Figure Priority

If space or attention is limited, the ranking should be:

1. `fig_overview.png`
2. `fig_scatter_agreement.png`
3. `fig_dist_answer_bias.png`
4. `fig_hm_alignment.png`
5. `fig_instruction_effect.png`

If one figure is demoted first, it should usually be:
- `fig_hm_quadrant.png`

---

## Human Alignment Focus

### Main architectural claim
- `VLM backbone decoder` is the closest group to human blind priors.
- This should be stated more strongly than any one question-type breakdown.

### Supporting entity / question findings
- Strong alignment:
  - yes/no existence
  - action
  - animal
  - place
- Moderate alignment:
  - food
  - person
  - other
- Divergence:
  - count
  - object
  - product
  - OCR / text-heavy

### Caution
- Do not overstate this as “the same prior” in a fully literal sense.
- Safer wording:
  - “substantial overlap with human blind priors”
  - “closest match to human blind reasoning”
  - “human-like prior structure”

---

## Human Study

- N = 36 participants (15M / 21F, age 18–34)
- 113 questions from the VQA 1K subset
- Human comparison exports are based on the matched condition used in the session 2 alignment analyses
- ICC(2,36) = 0.963

Important limitation:
- human comparisons are not interchangeable across every blind prompt setting
- condition matching should be stated carefully in the paper

---

## What To Keep Out of the Main AAAI Story

- intervention proposals as a major paper contribution
- overly detailed per-notebook derivations
- fragile claims from partially stabilized analyses
- too many parallel stories in the abstract

The paper is already strong on results. The main task is focus, not expansion.

---

## Next Useful Cleanups

- tighten the abstract further around the decoder result
- keep the intro findings list in the `F1 -> F6` order above
- make sure `Which Models Are Most Human-Like?` remains the conceptual center of §6
- keep entity-type analysis as support, not the top-level story
