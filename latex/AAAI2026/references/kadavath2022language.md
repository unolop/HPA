# Kadavath et al. (2022) — Language Models (Mostly) Know What They Know

**Citation key:** `kadavath2022language`
**Venue:** arXiv preprint (Anthropic)
**PDF:** `pdfs/kadavath2022language.pdf`
**arXiv:** https://arxiv.org/abs/2207.05221

## What the paper does

Investigates whether large language models can accurately predict when they will answer questions correctly — i.e., whether LLMs have calibrated *self-knowledge*. Studies Claude models at various scales on diverse multiple-choice and true/false tasks.

Key findings:
- Larger LLMs are well-calibrated when asked "Is the following statement true/false?" or "Which answer is correct?" in the right format
- Models can distinguish questions they will get right from those they will get wrong, especially at scale
- However, self-knowledge degrades when models are asked to reason about chains of thought or novel formats
- Calibration improves with model scale (larger = better-calibrated self-knowledge)

The paper introduces the "P(True)" evaluation framework: directly ask the model to estimate the probability its own answer is correct.

## Link to our paper

**Calibrated uncertainty under instruction vs. our finding.** Kadavath et al. show that LLMs *can* express calibrated uncertainty about their own correctness under appropriate prompting. Our instruction condition (Blind+Inst) is precisely such a prompt — we explicitly tell the model no image is present and ask it to reason from a plausible imagined scene. This *should* trigger appropriate uncertainty acknowledgment.

Instead, we observe the opposite: model token confidence *increases* under the instruction, soft abstentions collapse (Qwen3-VL: 72% collapse rate), and ECE worsens. This is a direct contrast with Kadavath et al.'s finding — their models become more calibrated under uncertainty-eliciting prompts; ours become more committed (hallucination rather than acknowledged uncertainty).

This contrast sharpens our Calibration section finding: VLMs under visual deprivation do not self-regulate the way LLMs can, suggesting the instruction-tuning for VLMs does not include appropriate epistemic humility about missing visual input.

**Specific placement:** Model Calibration and Overconfidence (Related Work).
