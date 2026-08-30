# Future Research Directions & Related Work
_Last updated: 2026-08-30_

---

## 1. Activation Steering for Hallucination & Visual Grounding

### Key Papers

**CAST — Conditional Activation Steering** (ICLR 2025 Spotlight)
- arXiv: https://arxiv.org/abs/2409.05907
- Code: https://github.com/IBM/activation-steering
- Extracts behavior vector + condition vector via PCA on contrastive examples. Behavior vector is only applied when `cosine_sim(hidden_state, condition_vector) > θ`. Text-only LLMs (LLaMA, Qwen, OLMo). Multi-condition logical composition (OR) supported.

**ASD — Activation Steering Decoding** (ACL 2025)
- ACL: https://aclanthology.org/2025.acl-long.634
- Steering vector = `mean(factual token activations) − mean(hallucinated token activations)` from 100-image calibration set. Bidirectional: runs π⁺ (h+λv) and π⁻ (h−λv) in parallel, final logit = `(1+α)·logit(π⁺) − α·logit(π⁻)`. Tested on LLaVA-1.5 and Qwen-VL. POPE +2.88–8.54%, CHAIR_S 51→40.

**ShiftDC** (local repo: /home/david/Desktop/yuna/ShiftDC)
- Inference-time safety intervention for VLMs via PyTorch forward hooks (no weight changes).
- At inference: captions image → extracts text-only activations → extracts VL activations → projects `(act_VL − act_text)` onto safety direction → subtracts during generation.
- Safety direction: `mean(safe activations) − mean(unsafe activations)` from ~100 calibration pairs.
- Models: LLaVA-1.5, LLaVA-1.6, MiniGPT-4, ShareGPT4V, Qwen-VL.
- Key files: `shiftdc/hooks.py` (ActivationStore, ActivationPatcher), `shiftdc/safety_direction.py`, `shiftdc/shiftdc.py`.

**GrAInS** (ACL 2026)
- arXiv: https://arxiv.org/abs/2507.18043
- Gradient-based attribution (Integrated Gradients) to find which tokens drive hallucination. Norm-preserving steering: `h̃ = (h + λv) × ||h||/||h + λv||`. Tested on LLaVA-1.6-7B and Qwen2.5-VL-7B.

---

### Our 4-Vector Epistemic Steering Framework (Proposed)

Extract one steering vector per epistemic condition:

| Vector | Condition | Contrastive pair | Captures | Data needed |
|---|---|---|---|---|
| **V1** | Blind → Abstention | blind+abstained vs. blind+confident | "No evidence" direction | Blind hidden states + abstention labels |
| **V2a** | Blind+inst → Human imagination | human inst_blind vs. model inst_blind | Human-calibrated prior direction | Human responses + model hidden states |
| **V2b** | Blind+inst → Imagination | inst_blind vs. blind | "Instruction adds imagination" direction | Two-condition hidden states |
| **V3** | Sighted → Correct answer | sighted+correct vs. sighted+wrong | Visual grounding direction | Sighted hidden states + GT correctness labels |
| **V4** | Perturbed image → Abstention | noise/gray+uncertain vs. noise/gray+confident | "Contradictory evidence" direction | Perturbed-image hidden states |

**Key insight**: Abstention collapse rate (Qwen3-VL-8B: 72%, LLaVA-1.5: 17%) predicts how well-separated V1/V2 are in activation space → predicts how effective steering will be per model.

---

### Methodology: Mathematical Formulation

#### Hidden State Extraction

For a transformer with $L$ layers, at inference time the input sequence is:

$$[\text{system}]\ [\text{question}]\ [\text{Answer:}]$$

During the **prefill pass** (processing all input tokens simultaneously), the hidden state at layer $\ell$ and the last input token position is:

$$h_i^{(\ell)} \in \mathbb{R}^{d_{\text{model}}} \quad \text{(last input token, layer } \ell \text{, sample } i\text{)}$$

Captured via PyTorch `register_forward_hook` on transformer layer $\ell$. The prefill pass is identified by `seq_len > 1`; decoding steps (`seq_len = 1`) are skipped. For VLMs, visual tokens are prepended before the text sequence — the last input token is still the final text token (`Answer:`), and the hook captures it correctly at position `[-1]` of the prefill hidden state.

#### Layer Selection

Extract candidate layers $\ell \in \{12, 14, 16, 18, 20, 22\}$ (middle 40–70% of depth for 32-layer models). Select best layer by **Fisher discriminant ratio**:

$$\ell^* = \arg\max_\ell \frac{\|\bar{h}_{\mathcal{A}}^{(\ell)} - \bar{h}_{\mathcal{B}}^{(\ell)}\|^2}{\sigma_{\mathcal{A}}^{2(\ell)} + \sigma_{\mathcal{B}}^{2(\ell)}}$$

where $\mathcal{A}, \mathcal{B}$ are the two contrastive classes and $\sigma^2$ is the mean variance across dimensions. Alternatively, use cosine similarity between class means: $\cos(\bar{h}_{\mathcal{A}}^{(\ell)},\ \bar{h}_{\mathcal{B}}^{(\ell)})$ — lower = better separation.

#### V1: Abstention Direction (Blind Condition)

Let $\mathcal{A}$ = blind samples where model soft-abstains (regex: "nothing", "none", "unanswerable", etc.) and $\mathcal{C}$ = blind samples where model answers confidently.

$$\mathbf{v}_1^{(\ell)} = \frac{1}{|\mathcal{A}|}\sum_{i \in \mathcal{A}} h_i^{(\ell)} - \frac{1}{|\mathcal{C}|}\sum_{i \in \mathcal{C}} h_i^{(\ell)}$$

No GT labels required — abstention labels come from existing blind inference outputs.

#### V3: Visual Grounding Direction (Sighted Condition)

Let $\mathcal{R}$ = sighted samples where model answer is correct (exact match for yes/no and count; $\text{SBERT}(\text{pred}, \text{GT}) > 0.6$ for free-text) and $\mathcal{W}$ = incorrect sighted samples.

$$\mathbf{v}_3^{(\ell)} = \frac{1}{|\mathcal{R}|}\sum_{i \in \mathcal{R}} h_i^{(\ell)} - \frac{1}{|\mathcal{W}|}\sum_{i \in \mathcal{W}} h_i^{(\ell)}$$

GT correctness labels from VQAv2 ground truth (no new human collection needed).

#### Steering Application

At inference time, apply the unit-normalized direction vector via a forward hook at layer $\ell^*$:

$$\tilde{h}^{(\ell^*)} = h^{(\ell^*)} + \lambda \hat{\mathbf{v}}, \quad \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

**Norm-preserving variant** (GrAInS, ACL 2026):

$$\tilde{h}^{(\ell^*)} = \left(h^{(\ell^*)} + \lambda \hat{\mathbf{v}}\right) \cdot \frac{\|h^{(\ell^*)}\|}{\|h^{(\ell^*)} + \lambda \hat{\mathbf{v}}\|}$$

$\lambda$ is tuned on a held-out split of the calibration set.

#### Contrastive Decoding (Post-hoc, No Retraining)

Using existing blind logprobs from current inference runs:

$$\text{logit}_{\text{final}}(w) = (1 + \alpha) \cdot \text{logit}_{\text{sighted}}(w) - \alpha \cdot \text{logit}_{\text{blind}}(w)$$

**Human-calibrated variant**: set $\alpha$ per-question using the human difficulty curve $h(q)$:

$$\alpha(q) = \alpha_0 \cdot h(q)$$

Questions with high $h(q)$ (easy blind = strong shortcut) receive stronger subtraction. $\alpha_0$ is set per model using abstention collapse rate.

#### Stability Validation

- **Leave-one-out**: extract direction $N$ times leaving one sample out; check $\cos(\mathbf{v}_{-i}, \mathbf{v}_{\text{full}}) > 0.95$
- **Cross-type generalization**: extract on yes/no questions, apply to count questions; if POPE improves → captures genuine epistemic direction, not question-specific noise
- **Random baseline**: steer with random unit vector at same $\lambda$; must not improve POPE

---

### 3-Month Implementation Plan

**Phase 1 — Immediate (no new inference)**
1. Contrastive decoding with fixed $\alpha$ per model → test on POPE/HallusionBench
2. Human-calibrated $\alpha(q) = \alpha_0 \cdot h(q)$ variant → compare to fixed-$\alpha$ baseline

**Phase 2 — 1 month (re-run with hidden states)**
1. Re-run blind inference on vqa_1k_control with `output_hidden_states=True`, save last-token hidden states at layers `[12,14,16,18,20,22]`
2. Re-run sighted inference same way
3. Extract V1 (abstention labels already computed) + V3 (VQAv2 GT)
4. Layer selection via Fisher discriminant ratio
5. Apply via ShiftDC-style hooks → eval on POPE, HallusionBench

**Phase 3 — 2–3 months**
1. Conditional steering (CAST-style condition vector) as ablation
2. V2a if sighted human study completed
3. Attention analysis: do hallucinating responses show blind-like attention to visual tokens?

**What is NOT needed**: no standalone LM decoder re-runs (blind VLM condition already serves as the no-vision baseline).

**Target benchmarks**: POPE, HallusionBench, CHAIR, AMBER

---

### Notes on Architecture and Resources

- **Which layers**: middle 40–70% of depth. For 32-layer models: layers 12–22. Qwen3-VL-8B may differ — sweep independently.
- **Memory during inference**: model (4-bit) ~5 GB + hook tensors ~negligible → TITAN RTX (24 GB) is sufficient
- **Disk**: save last-token only at 6 candidate layers → ~64 KB per sample → ~64 MB for 1k questions
- **Layer architecture**: each transformer "layer" = one full block (attention + FFN + residuals). No sub-layers. "Every 2nd layer" = sampling the search space (12, 14, 16...), not a special structure.
- **Conditional vector (CAST)**: deferred to Phase 3. Core intervention works unconditionally first.
- **Sample size**: 1k questions >> literature calibration sets (ASD: 100, ShiftDC: 100, RepEng: 128). Sufficient even for free-text (SBERT-threshold labels) and after train/test split.

**Key insight**: Abstention collapse rate (Qwen3-VL-8B: 72%, LLaVA-1.5: 17%) predicts how well-separated V1/V2 are in activation space → predicts how effective steering will be per model.

**Target benchmarks**: POPE, HallusionBench, CHAIR, VQAv2, RefCOCO, AMBER

---

## 2. Counting in VLMs

### Problem
- Subitizing range (0–4): ~70–80% accuracy; estimation range (5–10): ~40–55%; >10: largely fails
- Zero-bias: models answer "0" 75–95% of the time in blind condition; bleeds into sighted condition
- Models undercount systematically (conservative bias, opposite to humans who overcount under time pressure)
- Occlusion and clustering are primary failure drivers

### Key Papers

| Paper | Venue | Link |
|---|---|---|
| TallyQA (249K counting QA) | AAAI 2019 | https://arxiv.org/abs/1810.12440 |
| Learning To Count Everything (FSC-147) | CVPR 2021 | https://arxiv.org/abs/2104.08391 |
| Teaching CLIP to Count to Ten (CountBench) | ICCV 2023 | https://arxiv.org/abs/2302.12066 |
| Counting to Four is still a Chore for VLMs | 2025 | https://arxiv.org/abs/2604.10039 |
| Your VLM Can't Even Count to 20 | 2024 | https://arxiv.org/abs/2510.04401 |
| LVLM-COUNT (divide-and-conquer) | 2024 | https://arxiv.org/abs/2412.00686 |
| CounterCount (zero-aware prompt) | 2025 | https://arxiv.org/abs/2605.17826 |
| The Count Is There, but Misaligned | 2025 | https://arxiv.org/abs/2607.09544 |
| CountGD (NeurIPS 2024, grounding-first) | NeurIPS 2024 | https://arxiv.org/abs/2407.04619 |
| CounTX (open-vocab counting) | BMVC 2023 | https://arxiv.org/abs/2306.01851 |
| GroundingDINO | ECCV 2024 | https://arxiv.org/abs/2303.05499 |

### How Models Are Improved (Main Paradigms)
1. **Prompt engineering**: list-then-count, zero-aware ("say 0 if absent"), region-by-region scan
2. **Grounding-first pipeline**: detect with GroundingDINO → count bounding boxes (CountGD)
3. **Divide and conquer**: partition image into patches → count per patch (≤4 each) → sum (LVLM-COUNT)
4. **Density map regression**: classical CV head, not natively in VLMs
5. **Cardinality-aware fine-tuning**: contrastive pairs N vs N±1; ordinal loss (Teaching CLIP to Count)
6. **Inference-time probe correction**: linear probe on intermediate activations → correct output logit (Count Is There but Misaligned)

**Note**: LVLM-COUNT found CoT ("think step-by-step") *hurts* counting. Zero-bias under blind/text-only condition is currently uncharted — our work may be the first systematic documentation.

### Proposed Human Data Collection for Counting
1. **Count prior distribution** (blind, no image): humans distribute confidence across bins (0,1,2–3,4–5,6–10,>10) for each count question type → measures reasonable linguistic prior; gap to model zero-distribution = calibration target
2. **Subitizing benchmark**: controlled images with 1–15 objects + human RT → defines subitizing vs. estimation boundary empirically per question type
3. **Count range collection**: humans give estimate + confidence interval with images → distributional target matching model logprob format

---

## 3. Creativity & Imagination Benchmarks

### Visual Imagination / Ambiguity

| Benchmark | What it measures | Link |
|---|---|---|
| Hyperphantasia | Mental visualization without external stimuli; abstract synthetic puzzles (NeurIPS 2025) | https://arxiv.org/abs/2507.11932 |
| ImagineBench | 5 types of grounded mental imagery reasoning: Event Progression, Perspective Transformation, Functional Intent, Counterfactual State, Structural Completion (19.8K samples) | https://openreview.net/forum?id=dXPCskbh2q |
| VOPE | Hallucination in voluntary imagination tasks (story writing from image); low-hallucination models on facts = high-hallucination on imagination | https://arxiv.org/abs/2511.13420 |
| Mirage (Machine Mental Imagery) | Latent visual tokens interleaved during decoding; "thinking visually" without generating pixels | https://arxiv.org/abs/2506.17218 |
| MindCube | Spatial mental modeling from limited views; human ceiling 95%, best VLM ~48% (NeurIPS 2025 Oral) | https://arxiv.org/abs/2506.21458 |
| Creation-MMBench | Creative image-conditioned tasks: poetry, story continuation, artistic reinterpretation (765 cases, ICCV 2025) | https://arxiv.org/abs/2503.14478 |
| GuessBench | Creative sensemaking from abstract Minecraft builds; performance correlates with corpus frequency | https://arxiv.org/abs/2506.00814 |
| AmbiBench | Ambiguous images (duck/rabbit style); humans flexibly shift, VLMs collapse to dominant feature | https://openreview.net/forum?id=R2dCGaqzYW |
| IllusionBench+ | Classical + real-world optical illusions (1,051 images, 5,548 QA pairs) | https://arxiv.org/abs/2501.00848 |
| SemVink + HC-Bench | Hidden content in optical illusions; VLMs 0–5% accuracy (EMNLP 2025) | https://arxiv.org/abs/2506.02803 |
| **Rorschach for MLLMs** | Full Rorschach protocol on GPT-4o, Grok 3, Gemini 2.0; bias projection with zero ground truth | https://arxiv.org/abs/2604.18437 |
| FIQ (Figural Interpretation Quest) | Human vs. AI creative interpretation of abstract ambiguous figures; humans rated more creative | https://www.tandfonline.com/doi/full/10.1080/10447318.2024.2345430 |

### Creative Text / Divergent Thinking

| Benchmark | What it measures | Link |
|---|---|---|
| Divergent Creativity (DAT + AUT at scale) | DAT, AUT, haiku, story vs. 100K humans; LLMs beat average human on DAT | https://arxiv.org/abs/2405.13012 |
| LiveIdeaBench | Scientific idea generation from single keyword; 1,180 keywords, 22 domains | https://arxiv.org/abs/2412.17596 |
| C²-Eval | Convergent + divergent creativity across LLMs and VLMs; U-O-S scoring | https://arxiv.org/abs/2510.04009 |
| CreBench | Idea → process → product; 2.2K multimodal samples, 79.2K human feedback (AAAI 2026) | https://arxiv.org/abs/2511.13626 |
| CreativityPrism | Cross-domain creativity framework for LLMs | https://arxiv.org/abs/2510.20091 |
| CREATE | Associative creativity / remote association in LLMs | https://arxiv.org/pdf/2603.09970 |
| IDEAFix | Creative defixation — breaking fixed thinking patterns | https://arxiv.org/abs/2606.00875 |
| TTCW (Torrance Test for Writing) | TTCT adapted for LLM outputs: fluency, flexibility, originality, elaboration | https://www.emergentmind.com/topics/torrance-test-of-creative-writing-ttcw |
| DAT probing | Semantic divergence in LLM word associations vs. human norms | https://arxiv.org/abs/2310.11158 |
| Artificial Phantasia | Emergent mental imagery in text-only LLMs | https://arxiv.org/abs/2509.23108 |

### Counterfactual / "What If" Imagination

| Benchmark | What it measures | Link |
|---|---|---|
| CounterVQA | Counterfactual causal reasoning in VLMs for video | https://arxiv.org/abs/2511.19923 |
| MindEdit-Bench | Object-level counterfactual spatial reasoning; correct answer is absent from all input images | https://arxiv.org/abs/2607.00491 |
| HalluSegBench | Counterfactual segmentation hallucination; vision-driven vs. label-driven errors | https://arxiv.org/abs/2506.21546 |

### Human vs. AI Creativity Comparison

| Paper | Finding | Link |
|---|---|---|
| Stable Diffusion vs. human visual creativity | Persistent human–AI creativity gap for skilled artists | https://arxiv.org/abs/2511.16814 |
| Best humans still outperform AI (AUT) | Top 10% of humans exceed LLMs; average humans do not | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10502005/ |
| AI takes the Torrance Test | GPT-4 scores top 1% of humans on TTCT | https://www.sciencedirect.com/science/article/pii/S2713374523000249 |

---

### Connection to Our Study

**Our blind condition IS a Rorschach test** — zero factual ground truth, model must project from internal priors. Our human responses provide a baseline that the published Rorschach MLLMs paper (arXiv:2604.18437) lacks entirely.

**Mapping our conditions to benchmarks**:

| Our condition | Benchmark analogue | What it measures |
|---|---|---|
| Blind → confident answer | Rorschach for MLLMs | Prior projection with zero visual evidence |
| Inst_blind → human response | VOPE voluntary imagination | Calibrated imagination under instruction |
| Sighted → correct answer | ImagineBench / MindCube | Grounded visual reasoning beyond visible |
| Noise image → abstention | AmbiBench / IllusionBench+ | Uncertainty under contradictory/ambiguous visual input |

**VOPE finding is a direct prediction for our work**: models with high abstention collapse (Qwen3-VL-8B: 72%) should also score high on VOPE's imagination-hallucination scale. Low-collapse models (LLaVA-1.5: 17%) should be more consistent across factual and imaginative tasks.

**GuessBench corpus-frequency result** directly operationalizes our McCoy (2024) mechanism: performance drops for rare concepts = linguistic prior exploitation is frequency-driven.

---

## 4. Human Data Collection Ideas (Future Studies)

### Sighted Human Responses (highest priority)
Collect human answers to same 113 questions with images visible. Completes the 2×2 (blind/sighted × human/model). Enables:
- Visual necessity score per question (sighted ≠ blind → image actually matters)
- True alignment target for calibration (sighted humans, not blind humans)
- DPO pairs: (sighted answer, blind answer) where they differ

### Count Prior Distribution Elicitation
Blind condition, count questions only. Ask humans to distribute 10 points across answer bins (0, 1, 2–3, 4–5, 6–10, >10). Directly measures reasonable linguistic prior for counts. 20 raters × 200–300 count questions from VQAv2.

### Instruction Sensitivity in Humans
Test same instruction manipulation on humans across 3 conditions: no instruction / "imagine a typical scene" / "do not guess". Currently only one human instruction condition. Measures whether 33.6% answer shift (Qwen3-VL-8B) is a model-specific failure or universal cognitive response.

### Visual Necessity Annotation
Per question (text only): "Can this be answered correctly without seeing the image?" (Yes/Partially/No). 5 annotators × 113 questions = ~565 judgments. Generalizable to any VQA benchmark. Directly validates h(q) curve.

### Rorschach-Style Ambiguity Study
10 Rorschach plates (public domain, 1921) × 20–30 humans. Free description + confidence + alternatives + RT. Human normative data from Exner already available for comparison. Models have likely seen these online — use diffusion midpoints instead for guaranteed OOD stimuli.

### Inkblot / Ambiguity Probing (Novel Stimuli)
Use diffusion model midpoints (halfway through denoising) — guaranteed OOD, controllable ambiguity level. Better than classic Rorschach plates or abstract art (both compromised by training data exposure).

---

## 5. Training & Calibration Directions

### Contrastive Decoding (post-hoc, no retraining)
```
logit_final = (1+α) · logit_sighted − α · logit_blind
```
Blind logprobs already collected. Apply immediately on POPE/HallusionBench. α can be set per-model using abstention collapse rate.

### DPO on Variant Pairs
Pronominalization variants (C→A) where sHM degrades significantly:
- chosen: original question + human majority answer
- rejected: pronominalized question + model's drifted answer
Targets shortcut behavior directly. Should transfer to GQA, Winoground, VQAv2 balanced.

### SFT on Human Response Distribution
40 humans × 113 questions × 3 variants ≈ 13,560 (response, question) pairs. Weight by human confidence. Teaches model the human prior distribution. Compatible with existing `training/train_human_alignment.py` + LoRA setup.

### Confidence Calibration
Temperature scaling / isotonic regression: model logprob → human-calibrated confidence using 40-person confidence ratings. Lightweight, no retraining, generalizes to any benchmark where overconfidence is a problem.

### R-Tuning / Refusal Training
Train model to abstain on questions where blind accuracy ≈ 0. Blind accuracy data already available for all 113 questions × models.
