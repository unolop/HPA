# Hypothesis-Only Baselines in NLI — Paper Notes

**Title:** Hypothesis Only Baselines in Natural Language Inference
**Authors:** Adam Poliak, Jason Naradowsky, Aparajita Haldar, Rachel Rudinger, Benjamin Van Durme
**Published:** *SEM (Joint Conference on Lexical and Computational Semantics), 2018
**Link:** https://aclanthology.org/S18-2023/
**arXiv:** https://arxiv.org/abs/1805.01042
**GitHub:** https://github.com/azpoliak/hypothesis-only-NLI

---

## Core Idea

Natural Language Inference (NLI) is supposed to test whether a **hypothesis** (claim) is entailed, contradicted, or neutral with respect to a **premise** (evidence). The premise is the visual/contextual evidence; the hypothesis is the claim to evaluate.

The paper trains a model using *only the hypothesis* — completely ignoring the premise — and finds that this model **significantly outperforms a majority-class baseline on 10 out of 10 NLI datasets**.

**Conclusion:** NLI benchmarks contain annotation artifacts — statistical patterns in how annotators write hypotheses that allow models to predict the label without any evidence.

---

## Methodology

- Train a simple bag-of-words / decomposable-attention classifier on hypothesis only
- Evaluate on 10 NLI datasets: SNLI, MultiNLI, SciTail, MPE, Sick, QA-NLI, RTE, FN+, Spouses, Breaking NLI
- Compare to: majority class baseline, full premise+hypothesis model
- Find: hypothesis-only > majority on all 10 datasets; on some datasets approaches full model performance

**Example artifact discovered:**
- Hypotheses containing "not" or "no" → likely contradiction
- Hypotheses with negation or extreme language → biased toward entailment/contradiction
- Annotators use specific phrasing patterns for each label class

---

## Key Results

| Dataset | Majority baseline | Hypothesis-only |
|---------|------------------|-----------------|
| SNLI | 34.3% | 69.0% |
| MultiNLI | 35.4% | 67.0% |
| SciTail | 53.0% | 57.3% |
| Breaking NLI | 33.3% | 63.8% |

On SNLI: hypothesis-only achieves 69% vs. the full model's ~88% — more than halfway there using zero evidence.

---

## The Analogy to Our Work

This paper is the **most direct analogue** to what we are doing. The mapping is exact:

| NLI | Our VQA study |
|-----|---------------|
| Premise | Image |
| Hypothesis | Question |
| Task | Predict the label / answer |
| Hypothesis-only model | Blind VLM / LLM |
| Annotation artifact | Linguistic prior in VQA questions |

**The core question is identical:** *How much of benchmark performance is achievable without the evidence the benchmark is supposed to test?*

In NLI: the answer is "a lot — annotators write biased hypotheses."
In our VQA work: the answer is "a lot — questions contain statistical cues to the answer."

---

## What This Means for Our Framing

### Recommended Introduction Framing
> "Just as Poliak et al. [2018] showed that NLI benchmarks can be partially solved
> using hypothesis-only baselines — revealing annotation artifacts and linguistic priors —
> we ask whether VQA benchmarks exhibit analogous vulnerabilities. We find that VLMs
> can answer 40–50% of VQA questions correctly without any visual input, suggesting
> that VQA questions themselves contain systematic cues to the answer."

### Positioning as Benchmark Diagnosis
The Poliak paper catalyzed a wave of NLI dataset scrutiny (SNLI artifacts, adversarial NLI, etc.). Our work can position itself as doing the same for VQA:
- "We propose blind VQA evaluation as an analogous diagnostic for VQA benchmarks"
- This frames our dataset/protocol as a community contribution, not just a model analysis

### Strengthening the Contribution
Poliak only identified the problem; they had no human data and no intervention.
Our work goes further:
1. We have human blind VQA responses → shows what *natural* non-artifact performance looks like
2. We show the mechanism (control variants, corpus frequency)
3. We provide a finetuning fix → closes the loop from diagnosis to intervention

---

## Important Citations That Built on Poliak

These appeared after Poliak and you may want to cite to show the lineage:
- **Gururangan et al. (2018)** "Annotation Artifacts in Natural Language Inference Data" (ACL) — confirmed Poliak findings, deeper artifact analysis
- **Nie et al. (2020)** "Adversarial NLI" — built harder datasets to address the artifact problem
- **McCoy et al. (2019)** "Right for the Wrong Reasons" — showed models use shallow heuristics in NLI

The fact that the NLI community took ~2 years to seriously address the Poliak finding, and that it spawned several major papers, suggests there is a natural narrative arc: we are identifying for VQA what Poliak identified for NLI, but with richer methodology.

---

## How to Cite

**In intro:** cite as the motivating analogy — first sentence or first paragraph
**In related work:** under "benchmark artifacts and shortcuts"
**In methodology:** "We follow the spirit of hypothesis-only baselines [Poliak 2018] by evaluating VLMs with question-only inputs"

---

## Key Differences (to address if a reviewer asks)

| | Poliak NLI | Our VQA work |
|--|-----------|-------------|
| Input modality | Text only | Vision + Language |
| Source of artifact | Annotator writing style | Training data statistics |
| Human comparison | No | Yes (N=20+) |
| Intervention | No | Yes (SFT + JS finetuning) |
| Control variants | No | Yes (5 linguistic ablations) |
| Confidence analysis | No | Yes (logprob-based) |
