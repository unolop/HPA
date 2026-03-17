# Research Ideas & Suggestions

---

## Core Research Question
Can VLMs answer visual questions correctly without seeing images?
If yes → they are exploiting **linguistic priors / shortcuts** rather than visual understanding.

---

## Analysis Ideas

### High Priority

#### 1. Confidence Calibration (ECE + Reliability Diagram)
- Compute **Expected Calibration Error (ECE)** from logprob-based confidence
- Plot reliability diagrams: confidence bins vs. actual accuracy
- Key question: are models *overconfident when wrong* under blind conditions?
- Compare blind vs. visual condition calibration
- Strong standalone figure for the paper

#### 2. Control Type Degradation Curve
- Plot accuracy/confidence across the 5 variants per model:
  `question → deictic_removed → object_removed → weaker_object → subject_ablated`
- Shows *how much linguistic weakening is needed before model breaks*
- Quantifies shortcut reliance on a single axis
- One line per model → clean multi-model comparison figure

#### 3. Human vs. Model Comparison
- Align human confidence ratings (1–5 scale from `dataset/humans/`) with model logprob confidence
- Humans answering blind → lower confidence + lower accuracy (expected)
- If models don't show this → strong hallucination/overconfidence signal
- Could be a key table in the paper

#### 4. Answer Consistency Across Control Variants
- For the same `question_id`, do models give the same answer across all 5 variants?
- Consistency score = frequency of mode answer across variants
- High consistency = model ignores question wording = pure linguistic shortcut
- Low consistency = model is sensitive to wording = some visual/semantic processing

#### 5. Blind Accuracy Gap by Question Category
- Compute `Δ = acc_blind − acc_visual` per answer type (yes/no, number, color, location, other)
- Yes/no expected to have highest blind accuracy (easiest to guess)
- Counting expected to be lowest (requires actual visual grounding)
- Would explain *which question types* are most vulnerable

#### 6. Pre/Post Finetuning Confidence Shift
- Does SFT/JS training on blind VQA lower model confidence on blind questions?
- Compare confidence distributions before and after finetuning
- Goal: finetuning should produce *lower confidence + lower accuracy* on blind (better calibrated)
- If not → the model learned to answer differently but still is not calibrated

### Medium Priority

#### 7. Taxonomy of Shortcut Questions
- Use blind accuracy to cluster questions into:
  - *Purely linguistic shortcuts*: high blind accuracy regardless of model
  - *Visual-dependent*: low blind accuracy across all models
  - *Model-specific shortcuts*: high for some models, low for others
- Useful as a dataset contribution

#### 8. Attention Map Analysis (Mechanistic Evidence)
- For LLaVA/InternVL with image present: compare attention on image tokens vs. text tokens
- Between normal and blind/inst conditions
- Would provide mechanistic evidence for shortcut behavior
- Requires additional inference pass with attention outputs

---

## Paper Framing — Reviewer-Guided Reframe

### Primary Reframe (from reviewer)
Don't lead with "hallucination". Lead with **benchmark diagnosis + corpus frequency**.

**Suggested intro framing:**
> "Just as hypothesis-only baselines exposed annotation artifacts in NLI [Poliak 2018],
> we ask: how much of VQA performance is achievable without visual information?
> We show that VLMs exploit linguistic priors shaped by their training distribution [McCoy 2024],
> and use human blind VQA responses — which lack this bias — as a calibration reference."

### New Experiment to Add (from reviewer)
- Correlate blind accuracy with **answer frequency in VQA training distribution**
  - Proxy: use VQA v2 training split answer frequency as a corpus frequency estimate
  - Plot: blind accuracy vs. answer prior probability (scatter per question)
  - Expected: high correlation for LLMs/VLMs, lower for humans
  - This directly operationalizes the McCoy [1] mechanism

### Related Work to Expand
- Psycholinguistics: human fast/slow thinking (System 1 vs. System 2), error patterns
- NLI shortcut literature: hypothesis-only baselines, annotation artifacts
- LMs as corpus frequency models: McCoy et al. 2024

---

## Paper Framing Ideas (Original)

### Angle 1: "Linguistic Prior Exploitation in VLMs"
- The control question variants are the unique contribution
- Frame as measuring *how much of VQA performance is explained by linguistic priors alone*
- Position control ablations as a new measurement methodology

### Angle 2: "Can Finetuning Fix Shortcut Behavior?"
- Intervention study: measure shortcuts → train against them → measure again
- Clean before/after narrative using SFT and JS-divergence training
- Human data as the calibration target

### Angle 3: "Blind VQA as a Diagnostic Tool"
- Propose blind VQA evaluation as a standard diagnostic for any VLM
- Argue models should be tested blind before deployment
- Your dataset + protocol as a community contribution

---

## Model Coverage

### Current Models (ACL paper)
| Model | Family | Size | Status |
|-------|--------|------|--------|
| LLaVA 1.5 | LLaVA | 7B | ✓ |
| LLaVA 1.6 Mistral | LLaVA | 7B | ✓ |
| LLaVA 1.6 Vicuna | LLaVA | 7B | ✓ |
| Qwen3-VL | Qwen | 8B, 32B | ✓ |
| InternVL 3.5 | InternVL | 1B–8B | ✗ can't run on new GPU |

### InternVL Replacement Options
Since InternVL can't run on the new environment, options ranked by priority:

1. **Qwen2.5-VL-7B** ← recommended
   - Same Qwen family but older generation, provides within-family comparison (Qwen2.5 vs Qwen3)
   - Widely used benchmark, well-supported in ms-swift
   - Very VRAM-efficient

2. **Phi-3.5-Vision-Instruct (4.2B)**
   - Microsoft architecture — genuinely different family from LLaVA/Qwen
   - Lightweight, good on text-heavy/reasoning tasks
   - Good replacement if you need architectural diversity

3. **LLaMA-3.2-Vision-11B**
   - Meta family, good diversity, but 11B may hit same VRAM ceiling as InternVL

### Do you need a replacement?
Arguably **no** for AAAI, given:
- You already have 3 families: LLaVA (3 variants), Qwen3-VL (2 sizes), + LLM baselines
- Adding LLM decoder-only from existing VLMs gives you a new axis (VLM vs its own LLM backbone)
- The AAAI story is about the *phenomenon* across families, not model breadth

If reviewers push on coverage, add Qwen2.5-VL-7B (minimal extra work, already in ms-swift).

### LLM Decoder-Only Plan
Extract the language backbone from existing VLMs for text-only inference:
| VLM | LLM backbone | HuggingFace ID |
|-----|-------------|----------------|
| LLaVA 1.5 | Vicuna 7B | `lmsys/vicuna-7b-v1.5` |
| LLaVA 1.6 Mistral | Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.2` |
| LLaVA 1.6 Vicuna | Vicuna 7B | `lmsys/vicuna-7b-v1.5` |
| Qwen3-VL 8B | Qwen3 8B | `Qwen/Qwen3-8B` |
- Already have Qwen3-8B in project (`evaluation/logits/pretrained/`)
- Lets you ask: does the VLM perform *better or worse* than its own LLM backbone on blind VQA?
- If VLM > LLM backbone blind → vision encoder somehow leaks text bias
- If VLM ≈ LLM backbone blind → behavior is purely from LLM component

---

## Additional Datasets: OK-VQA and TextVQA

### What We Have
- **OK-VQA**: 5K+ questions, normal + blind scored for 7 models — fully inference-complete
- **TextVQA**: 5K questions, normal + blind scored for 7+ models — fully inference-complete

### Should We Include Them?

**TextVQA — YES, as a negative control (strongly recommended)**
- Requires reading text *in the image* → blind accuracy should be near zero
- This is a natural lower bound that anchors the whole argument
- If VQA blind accuracy = 40%, MMStar blind = 25%, TextVQA blind ≈ 5% → clear gradient
- Strengthens the claim: "models exploit linguistic priors *specifically when answers are in the language distribution*"
- Very cheap to include — data already collected

**OK-VQA — YES, as an upper bound / interesting contrast**
- Requires outside knowledge (not visual) → blind accuracy expected to be *high*
- Questions like "What country is this currency from?" — answerable from question text alone
- Would show: OK-VQA blind ≈ VQA blind >> TextVQA blind → confirms the "visual necessity" gradient
- Also interesting to compare with human blind performance on OK-VQA questions

**Recommended dataset lineup for AAAI:**

| Dataset | Expected blind accuracy | Role in paper |
|---------|------------------------|---------------|
| VQA v2 (1K) | ~40-50% | Main dataset, with control variants |
| OK-VQA | ~35-45% | Knowledge-based questions, high linguistic prior |
| MMStar | ~25% (random) | Visual reasoning, low linguistic prior |
| TextVQA | ~5-10% | Image-text reading, near-zero linguistic prior |

This gradient is a powerful figure — one axis showing "how much vision is actually needed" across benchmark types.

### VQA v2 Diversity Concern
VQA v2 is indeed biased toward yes/no and simple attribute questions. The entity categorization (ent) partially addresses this. The OK-VQA + TextVQA comparison also addresses it by showing behavior across different question *types* rather than just different entity groups.

---

## Human Study — Question Sampling

### New Notebook
`analysis/notebook/question_sampling.ipynb` — samples ~100–150 questions from 1K control set

### Sampling Tiers
| Tier | Criterion | What it captures |
|------|-----------|-----------------|
| A | High confidence + wrong (conf>0.75, acc<0.2) | Overconfident hallucination |
| B | Answer changes across control variants | Genuine linguistic sensitivity |
| C | All models give same wrong answer | Shared corpus prior |
| D | Large accuracy drop: question → subject_ablated | Shortcut-dependent questions |

### Sampling Tiers (updated)
| Tier | Criterion | What it captures |
|------|-----------|-----------------|
| A | High confidence + wrong (conf>0.75, acc<0.2) | Overconfident hallucination |
| B | Answer changes across control variants | Linguistic sensitivity |
| C | ≥80% of models wrong on baseline | Shared corpus prior (hard questions) |
| D | Large accuracy drop: question → subject_ablated | Shortcut-dependent questions |
| E | **Large confidence drop: question → subject_ablated** | Model becomes uncertain when linguistic cues weakened |

Tier E is the new addition — captures questions where models lose confidence but may still be correct (or newly wrong). Especially interesting for the calibration analysis.

### Practical Guidance — 300-400 Questions Per Person

**Target: 300 questions × 5 variants per question**
- Each question shown in all 5 control forms
- Previous study: 641 questions ≈ 1 hour → 300 × 5 = 1500 item presentations
- If shown one variant at a time: ~1500 judgments ≈ 2.5 hours — too long
- **Better approach:** show all 5 variants together per question as a batch
  - "Here are 5 versions of the same question, answer each briefly"
  - ~300 question groups ≈ 75–90 minutes — feasible

**What question types to include:**
- Yes, include **all entity types** (object, person, animal, place, food, etc.)
- For statistical significance per group: need ~30+ questions per entity type
- With 9 entity types × 30 = 270 minimum — matches the 300 target well

**Should we sample by confidence difference (model vs. human)?**
- YES — add a Tier F: questions where existing human data (from original study) shows
  *high confidence* but model shows *low confidence* (or vice versa)
- This reveals the human-model calibration mismatch most directly
- Can use existing `dataset/humans/` data to identify these question_ids
- Only applies to the ~374 questions already annotated — use as a priority pool

**Statistical significance considerations:**
- 300 questions × 9 entity types → ~33 per group → sufficient for pairwise comparisons
- With 20 new participants: ~6,000 responses per entity group → solid CIs
- Use the same leave-one-rater-out jackknife as the original study for inter-rater agreement
- Balance sampling: don't let `object` (most common) dominate — cap at 60/300 = 20%

**Recommended final sampling breakdown (300 total):**
- ~60 from Tier D/E (interesting degradation cases)
- ~60 from Tier A/C (hallucination / hard questions)
- ~60 from Tier B (answer flip)
- ~120 random stratified by ent (ensures entity coverage)
- Cap each entity at 20% → max ~60 per entity type
- Export: `analysis/csv/sampled_control_questions.csv` with tier labels for manual review

---

## Organization TODOs

- [ ] Centralize all hardcoded paths into a single `config.py` or `.env`
  - Currently scattered across notebooks: `/home/david/...`, `/home/yuna/...`
  - Affects portability and reproducibility
- [ ] Create a `results_index.csv` or `manifest.json`
  - Track which `(model, dataset, condition)` combos have been run
  - Include file path, row count, date
  - Replace the ad-hoc `os.listdir` + file-size checks in notebooks
- [ ] Extract reusable analysis code from large notebooks into `analysis/utils/`
  - Notebooks `question_type.ipynb` (1.4MB) and `correlations.ipynb` (633KB) are too large
  - Keep notebooks thin: load data → call util function → display

---

## AAAI Full Paper Plan

### What's Already in the ACL Short Paper (do NOT repeat as-is)
- Blind inference protocol (blank image + instruction condition)
- N=20 human participants, 641 questions (374 VQA + 267 MMStar)
- Pretrained model accuracy: humans vs. LLMs vs. VLMs
- Answer pattern comparison (yes/no, number distributions)
- SFT + JS finetuning on N=10/15 subjects, single random split
- Human-model alignment (ρ) before/after finetuning
- Multimodal Gain (MG) analysis and correctness quadrants

### What's New for AAAI (the delta)

#### 1. Reframing (intro + related work)
- Lead with benchmark diagnosis, not hallucination — Poliak [2] analogy explicitly
- Add corpus frequency framing — McCoy [1] as mechanistic explanation
- Expand related work: NLI shortcut literature, psycholinguistics (System 1/2), LMs as corpus frequency models

#### 2. Control Question Variants (main new contribution)
- 5 linguistic ablations per question: `question → deictic_removed → object_removed → weaker_object → subject_ablated`
- Degradation curve: accuracy/confidence drop across variants per model
- Answer consistency score across variants
- Quantifies shortcut reliance on a continuous axis — not in ACL paper at all

#### 3. Confidence & Calibration Analysis (new)
- ECE + reliability diagrams from logprob-based confidence
- Are models overconfident when wrong under blind conditions?
- Pre/post finetuning confidence shift
- Human confidence (Likert 1–5) vs. model logprob confidence comparison

#### 4. Corpus Frequency Correlation (new experiment from reviewer)
- Correlate blind accuracy with answer frequency in VQA v2 training split
- Scatter plot: answer prior probability vs. blind accuracy per question
- Expected: high correlation for LLMs/VLMs, lower for humans
- Directly operationalizes McCoy [1] mechanism

#### 5. Robustness Fixes (fixes ACL limitations)
- **K-fold finetuning** (5-fold or LOO on N=20 participants)
  - Report mean ± std of ρ instead of single-split values
  - Focus on InternVL 3.5 8B + Qwen3-VL 8B for main results
  - Other models in supplementary with single split
  - Infrastructure already exists: `train_kfold_sft.py`, `dataset/folds/`
  - pane 3 currently running fold 1 of LLaVA-Mistral adaptive_kl-0.5
- **Bootstrap CIs** on all pretrained model accuracy + human accuracy numbers

#### 6. Scale-Up
- Expand VQA from 374 → 1K questions (already collected)
- Add Qwen3-VL-32B results (already have logits)
- Keep MMStar for cross-benchmark generalization

### Robustness Strategy Summary
| Result type | Method | Status |
|-------------|--------|--------|
| Pretrained model accuracy | Bootstrap 95% CI | Not done |
| Human accuracy | Bootstrap 95% CI (jackknife already used for inter-rater) | Not done |
| Finetuning ρ scores | K-fold (5-fold or LOO) mean ± std | In progress (pane 3) |
| Control question degradation | Report across all 7 models | Not done |

### Key Figures for AAAI
1. **Degradation curve** — accuracy/confidence across 5 control variants, one line per model
2. **Calibration / reliability diagram** — confidence bins vs. accuracy, blind vs. visual condition
3. **Corpus frequency scatter** — answer prior probability vs. blind accuracy
4. **Human vs. model confidence** — distribution comparison with bootstrap CIs
5. **Finetuning results table** — mean ± std ρ across folds (replaces Figure 3 from ACL)

---

## Next Steps (TODO)

### Immediate
- [x] Fill in `meta_review.md` with reviewer feedback ✓
- [ ] Run confidence/VQA accuracy analysis in `confidence.ipynb` (cells added, needs execution)
- [ ] Wait for `llava-v1.6-mistral-7b-hf adaptive_kl-0.5` training to finish (tmux pane 3, ~2d remaining)

### Short Term — Analysis
- [ ] Implement degradation curve plot (control_type × accuracy/confidence per model)
- [ ] Compute ECE and plot reliability diagrams per model × condition
- [ ] Align human confidence (Likert) with model logprob confidence for comparison
- [ ] Compute answer consistency score across 5 control variants per question
- [ ] Run corpus frequency correlation experiment (VQA v2 train split answer frequencies)
- [ ] Add bootstrap CIs to all pretrained + human accuracy numbers

### Short Term — Training
- [ ] Run k-fold (5-fold) finetuning for InternVL 3.5 8B — SFT and JS objectives
- [ ] Run k-fold finetuning for Qwen3-VL 8B — SFT and JS objectives
- [ ] Run remaining LLaVA-Mistral folds after pane 3 finishes

### Medium Term
- [ ] Run inference on 1K VQA condition (scale-up from 374)
- [ ] Analyze pre/post finetuning confidence shift
- [ ] Build shortcut question taxonomy (cluster by blind accuracy across models)
- [ ] Write human vs. model calibration comparison notebook

### Paper Writing
- [ ] Rewrite intro with benchmark diagnosis framing + Poliak analogy
- [ ] Expand related work: NLI shortcuts, psycholinguistics, corpus frequency
- [ ] Write control questions section (new main contribution)
- [ ] Finalize 5 key figures listed above
- [ ] Replace single-split finetuning results with k-fold mean ± std
