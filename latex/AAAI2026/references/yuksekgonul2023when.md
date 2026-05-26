# Yüksekgönül et al. (2023) — When and Why Vision-Language Models Behave like Bags-of-Words

**Citation key:** `yuksekgonul2023when`
**Venue:** ICLR 2023 (Oral, top 5%)
**PDF:** `pdfs/yuksekgonul2023when.pdf`
**arXiv:** https://arxiv.org/abs/2210.01936

## What the paper does

Introduces the **ARO benchmark** (Attribution, Relation, Order) to test whether VLMs (CLIP and variants) understand the *structure* of language rather than treating inputs as bags of words. Three sub-tasks:
- **VG-Attribution**: "a red car near a blue house" vs. "a blue car near a red house"
- **VG-Relation**: spatial/relational understanding ("cat on the sofa" vs. "sofa on the cat")
- **COCO/Flickr Order**: sentence-level word order sensitivity

Key finding: State-of-the-art VLMs (CLIP, FLAVA, ViLT) perform barely above chance on these tasks — shuffling word order or swapping attributes does not significantly change model scores. Models behave like bags-of-words even though they use transformer architectures with positional encodings.

## Link to our paper

**Compositional failure → image-ignorance.** Yüksekgönül et al. show VLMs fail at compositional binding *within* visual-language pairs (they can't tell "A on B" from "B on A"). Our blind VQA work shows a more extreme version of the same failure: models effectively ignore the *entire image* and produce answers from text statistics alone.

Together with our work, these results bound the grounding failure from two angles:
- Yüksekgönül et al.: even when the image *is* present, VLMs fail to correctly bind it to linguistic structure
- Our work: models answer VQA questions correctly *without* any image, revealing the prior-only baseline

We cite ARO in the VLM Hallucination section to establish that our image-ignorance finding fits within a broader pattern of shallow visual-linguistic integration.

**Specific placement:** VLM Hallucination and Shortcut Learning (Related Work).
