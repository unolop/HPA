# Torralba & Efros (2011) — Unbiased Look at Dataset Bias

**Citation key:** `torralba2011unbiased`
**Venue:** CVPR 2011
**PDF:** `pdfs/torralba2011unbiased.pdf`
**Source:** http://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf

## What the paper does

Demonstrates that major vision benchmark datasets (Caltech-101, PASCAL, LabelMe, SUN, ImageNet, etc.) carry strong sampling biases that are easily learnable by classifiers. A model trained on one dataset performs poorly on others even for the same semantic categories, and classifiers can identify *which* dataset an image came from with high accuracy purely from low-level statistics — revealing that datasets are far from representative of the visual world.

Key contributions:
- Introduces cross-dataset generalization as a diagnostic: train on dataset A, test on dataset B
- Shows dataset-specific bias is encoded in texture, color, and viewpoint statistics
- Argues that benchmark performance is inflated by dataset-specific shortcuts rather than genuine generalization

## Link to our paper

This is the foundational reference for "dataset bias = exploitable shortcut" in computer vision. We cite it in the introduction and related work to place blind VQA in a long tradition of benchmark-bias diagnostics: just as Torralba & Efros showed that vision classifiers exploit dataset-specific image statistics rather than learning true visual concepts, our work shows that VLMs exploit textual prior statistics rather than learning to ground visual questions. The parallel is direct.

**Specific placement:** Hypothesis-Only and Shortcut Baselines (Related Work), and the second paragraph of the Introduction when contextualizing Poliak's NLI artifact finding.
