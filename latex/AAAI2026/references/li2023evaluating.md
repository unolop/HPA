# Li et al. (2023) — Evaluating Object Hallucination in Large Vision-Language Models (POPE)

**Citation key:** `li2023evaluating`
**Venue:** EMNLP 2023
**PDF:** `pdfs/li2023evaluating.pdf`
**arXiv:** https://arxiv.org/abs/2305.10355

## What the paper does

Proposes **POPE (Polling-based Object Probing Evaluation)**, a yes/no probing benchmark for measuring object hallucination in LVLMs. For each image, the benchmark asks "Is there a [object] in the image?" where objects are sampled under three conditions: random, popular (frequent across images), and adversarial (objects that co-occur with present objects).

Key findings:
- VLMs hallucinate objects at substantial rates, especially under adversarial sampling
- Models show a **yes-bias** on POPE (they tend to say "yes" even when objects are absent)
- The yes-bias correlates with the frequency of "yes" answers in instruction-tuning data

POPE is now one of the most widely used benchmarks for VLM hallucination evaluation.

## Link to our paper

**Complementary diagnostics: POPE's yes-bias vs. our no-bias.** POPE documents a yes-bias in VLMs on object-existence yes/no questions. Our blind VQA work finds the opposite: a 62% no-bias on VQA v2.0 yes/no questions under blind conditions (vs. 33% for humans). These two results together suggest that yes/no bias direction depends on the training distribution — POPE-style instruction-tuning pushes toward "yes," while VQA v2.0 blind pretraining pushes toward "no."

Our work provides a **mechanistic complement** to POPE: POPE measures the *symptom* (hallucinated objects), while our blind VQA setup exposes the *prior source* (the model's answer distribution without any image input).

**Specific placement:** VLM Hallucination and Shortcut Learning (Related Work) — noted as a yes/no probing benchmark whose direction of bias differs from ours, positioning us as mechanistic complement.
