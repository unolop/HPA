# Jabri, Joulin & van der Maaten (2016) — Revisiting Visual Question Answering Baselines

**Citation key:** `jabri2016revisiting`
**Venue:** ECCV 2016
**PDF:** `pdfs/jabri2016revisiting.pdf`
**arXiv:** https://arxiv.org/abs/1606.08390

## What the paper does

Revisits what constitutes a strong VQA baseline. Key finding: a simple multi-label binary classifier that takes the *answer as input* (along with the question and image) and predicts whether the answer is correct achieves surprisingly competitive performance — raising serious questions about whether high accuracy on VQA requires genuine visual reasoning.

The paper shows that text-only models using word embeddings of question+answer pairs can match or approach full visual models on VQA v1.0 Real Open-Ended and Multiple Choice tracks. The strong text-only performance reveals that VQA v1.0 benchmarks are substantially solvable through question-answer co-occurrence statistics, without any visual understanding.

## Link to our paper

This is the most direct historical precedent for our blind VQA setup. Jabri et al. showed in 2016 that text-only models achieve substantial VQA accuracy without images; our work extends this to modern VLMs on VQA v2.0, adding: (1) a human reference dataset, (2) the backbone decoder model group, (3) instruction and confidence analysis, and (4) a control-variant perturbation suite.

We cite this in the introduction and in the Hypothesis-Only section of Related Work to establish that the "blind VQA" diagnostic predates our work and that our contribution is the human-calibrated, architecture-comparative, and confidence-extended version of that tradition.

**Specific placement:** Introduction paragraph 2 (Poliak analogy → VQA history) and Hypothesis-Only and Shortcut Baselines (Related Work).
