F1. Prior exploitation is pervasive

Models answer a substantial fraction of VQA questions correctly without any image input, often around 40--70% depending on question type and model family. This establishes the basic diagnosis: the benchmark is meaningfully exploitable through language-only priors.

F2. VLM backbone decoders are the human-like group

This is the main finding. The VLM backbone decoders are the model group that aligns most closely with human blind priors. On free-text agreement, they reach semantic similarity to humans that is near the human-human ceiling, while full VLMs are lower and standalone LLMs diverge further. This suggests that the language backbone exposed through the decoder-only setting best captures the prior structure shared with humans.

F3. Full VLMs and standalone LLMs show different failure regimes

Although both exploit blind priors, they do not do so in the same way. Full VLMs show stronger model-specific blind defaults such as extreme no-bias and zero-bias, while standalone LLMs drift further from human answer semantics overall. The result is not a single "model prior" phenomenon but distinct behavioral regimes across model groups.

F4. Blind success is structured, not random

The blind answers follow clear default patterns rather than diverse plausible guesses. Models overproduce "no" on yes/no questions, "0" on count questions, and frequent defaults such as "black" for color. These patterns show that blind accuracy is driven by systematic shortcut structure, not by general robust reasoning.

F5. Instruction unlocks stronger prior expression

Adding the explicit no-image / imagine-a-plausible-scene instruction does not make models more cautious. Instead, it reduces soft abstention, increases answer commitment, and often raises confidence. The instruction acts as a permission signal that releases latent prior-based answering rather than improving grounding.

F6. Blind models are overconfident

Blind answers are often produced with high mean token log-probability even when they are wrong. Confidence does not reliably track correctness, and instruction often increases commitment further. The failure mode is therefore not just hallucination, but confident hallucination.
