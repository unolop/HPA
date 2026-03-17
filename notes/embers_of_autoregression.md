# Embers of Autoregression — Paper Notes

**Title:** Embers of Autoregression Show How Large Language Models Are Shaped by the Problem They Are Trained to Solve
**Authors:** R. Thomas McCoy, Shunyu Yao, Dan Friedman, Mathew D. Hardy, Thomas L. Griffiths (Princeton)
**Published:** PNAS 121(41), 2024
**Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11474099/ (open access)
**GitHub:** https://github.com/tommccoy1/embers-of-autoregression

---

## Core Argument

LLMs are fundamentally shaped by their training objective: **next-word prediction over internet text**. This single fact predicts both their strengths and their surprising failures. The authors call this the **teleological approach** — understanding a system by the goal it was trained to achieve.

The key implication: LLMs are not general reasoning engines. They are optimised to produce outputs that match the statistical distribution of their training data. When a task aligns with that distribution, they succeed; when it doesn't, they fail in predictable ways.

---

## The Three Factors

The paper proposes that LLM accuracy is predictably influenced by three probability-based factors:

| Factor | Description | Effect |
|--------|-------------|--------|
| **P(task)** | How often is this type of task performed in training data? | Rare tasks → lower accuracy |
| **P(output)** | How probable is the target output in the training distribution? | Low-probability outputs → lower accuracy |
| **P(input)** | How common is the input context in training data? | Rare input types → lower accuracy |

---

## Key Experiments & Results

**Cipher decoding (deterministic task):**
- GPT-4 decoding a simple cipher: **51% accuracy** when output is a high-probability sentence
- Same task, low-probability output: **13% accuracy**
- The task is fully deterministic — probability should be irrelevant — yet it dominates performance

**Five models tested:** GPT-3.5, GPT-4, Claude 3, Llama 3, Gemini 1.0
**11 tasks** across different probability regimes
**Consistent finding:** All models show the same pattern — performance tracks output probability even on tasks where probability is normatively irrelevant

**Other examples:**
- Word counting, list reversal: fail despite seeming simple — because exact counts/orders are rarely reproduced verbatim in training text
- Common knowledge retrieval: high accuracy — matches training distribution well

---

## Connection to Our Work

### Direct Mechanistic Explanation
This paper provides the theoretical mechanism behind our blind VQA findings:

> When a VLM correctly answers a VQA question without seeing an image, it is producing
> a high-probability completion of the question text based on its training distribution —
> not reasoning visually.

Specifically:
- VQA questions like *"What color is the fire hydrant?"* → the answer "red" is a high-P(output) completion
- *"How many people are in the scene?"* → "2" or "3" are higher P than "17"
- Models are not guessing — they are completing statistically probable sequences

### Predicting Which Questions Are Hard vs. Easy Blind
The three factors make concrete predictions for our blind VQA results:
- **High blind accuracy:** questions with high-P(output) answers in the training distribution (common objects, yes/no, common colors)
- **Low blind accuracy:** questions requiring rare or image-specific outputs (exact text in image → TextVQA, specific spatial arrangements)
- **Entity type effects:** `person` and `object` questions dominate training data → high P(output) → high blind accuracy; `text` entity questions → low P → low blind accuracy

### Proposed Experiment (from reviewer suggestion)
Test the P(output) hypothesis directly:
- Compute answer frequency in VQA v2 training split for each question
- Correlate with blind accuracy across models
- Expected: Spearman ρ > 0.3 between answer frequency and blind accuracy
- This would be a novel quantitative test of the McCoy mechanism applied to VQA

---

## How to Cite / Position

**In related work:** "McCoy et al. [1] show that LLM outputs are biased toward high-probability training-distribution completions. We provide empirical evidence that this mechanism extends to VLMs on visual question answering: blind accuracy correlates with answer frequency in the training distribution."

**In discussion:** Use to explain *why* VLMs succeed without images — not as a failure mode but as a principled consequence of autoregressive pretraining.

**Differentiation from our work:**
- McCoy studies pure text LLMs; we study VLMs with a visual modality
- McCoy uses artificial constructed tasks; we use real-world VQA benchmarks
- We add the human comparison — McCoy has no human data on the same tasks
- We test an intervention (finetuning); McCoy is purely diagnostic

---

## Key Quote to Use

> "LLMs should not be evaluated as if they are humans, but should instead be treated as a
> distinct type of system — one that has been shaped by its own particular set of pressures."

This supports our framing that human blind VQA behavior is the right reference point precisely *because* humans are not shaped by corpus frequency in the same way.
