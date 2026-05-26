# Zhang et al. (2016) — Yin and Yang: Balancing and Answering Binary Visual Questions

**Citation key:** `zhang2016yin`
**Venue:** CVPR 2016
**PDF:** `pdfs/zhang2016yin.pdf`
**arXiv:** https://arxiv.org/abs/1511.05099

## What the paper does

Identifies and corrects a **yes-bias** in VQA v1.0: models trained on VQA answer "yes" to binary (yes/no) questions at far above chance rates because the dataset contains ~60% "yes" answers. The paper responds by constructing complementary abstract scene pairs where the same question has "yes" for one scene and "no" for the other — eliminating the ability to exploit answer frequency without looking at the image.

This work directly motivated the complementary-image pairing approach later generalized in VQA v2.0 (Goyal et al. 2017).

## Link to our paper

**The yes → no bias flip.** Zhang et al. documented that VQA v1.0 models had a yes-bias (~60% "yes" answers). Our work finds the opposite on VQA v2.0: VLMs default to "no" at ~62% under blind conditions. This is a meaningful comparison: the redesign intended to eliminate yes-bias produced a no-bias instead, suggesting that models exploit whatever distributional skew exists in training rather than learning visual grounding. VQA v2.0 fixed the benchmark; it did not fix the underlying exploitability.

This framing is unique to our paper — no prior work connecting yes-bias (Zhang 2016) to no-bias (our finding) under blind VQA v2.0 exists.

**Specific placement:** Introduction paragraph 2 (motivating VQA v2.0 history) and Hypothesis-Only and Shortcut Baselines (Related Work), explicitly noting the yes → no bias flip.
