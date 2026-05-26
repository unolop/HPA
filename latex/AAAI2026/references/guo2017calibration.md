# Guo et al. (2017) — On Calibration of Modern Neural Networks

**Citation key:** `guo2017calibration`
**Venue:** ICML 2017
**PDF:** `pdfs/guo2017calibration.pdf`
**arXiv:** https://arxiv.org/abs/1706.04599

## What the paper does

Identifies that **modern deep neural networks are systematically miscalibrated**: their confidence scores (softmax output) do not reflect the true probability of correctness. Contrary to shallow networks from the 1990s, deep networks trained with modern techniques (batch norm, large capacity, weight decay) are overconfident — high predicted probabilities do not correspond to high accuracy.

Key contributions:
- **Reliability diagrams** to visualize calibration gaps
- **Expected Calibration Error (ECE)** as a summary metric
- **Temperature scaling** as a simple, effective post-hoc calibration method (one parameter calibrates the entire model)

The paper demonstrates that scaling the softmax logits by a single learned temperature dramatically improves calibration across ResNet, VGG, and other architectures.

## Link to our paper

**Calibration failure as a distinct diagnostic dimension.** Guo et al. establish the canonical framework for measuring overconfidence in neural networks. We extend this to the *generative VLM* setting under blind conditions: mean token log-probability remains high even for wrong committed blind answers, creating a calibration failure that is invisible to accuracy-only evaluation.

We use ECE as a metric (blind ECE: VLM 0.337, standalone LLM 0.642) directly following the Guo et al. framework. Their finding that overconfidence is a systematic property of modern networks — not just noise — reinforces our claim that blind overconfidence reflects a structural failure of the VLM training objective rather than an incidental artifact.

We also contrast with the instruction condition: Guo et al. show calibration can be post-hoc corrected; we show that our instruction effectively *worsens* calibration by releasing latent priors, which is the opposite of what a calibration-aware model should do.

**Specific placement:** Model Calibration and Overconfidence (Related Work) and the methods paragraph of the Introduction (overconfidence motivation sentence citing `guo2017calibration`).
