# HPA — Human-calibrated Prior Analysis of Vision-Language Models

**Paper:** *When Models Answer Without Seeing: Human-Calibrated Diagnosis of Linguistic Prior Exploitation in Vision-Language Models*

We probe VLMs on VQA questions with the image replaced by a blank, measuring how much models exploit linguistic priors rather than visual content, and compare this to human question difficulty.

---

## Environment

```bash
conda activate zero        # Python 3.9, ms-swift, transformers
export HF_HOME=/home/david/Desktop/yuna/.cache/hf
export OPENAI_API_KEY=<your key>   # required for preprocessing only
```

---

## Pipeline

### Step 1 — Preprocess: generate control variants

```bash
python preprocess.py --mode CTL \
    --input  dataset/vqa/vqav2_1k_val.json \
    --output dataset/vqa/vqa1k_control.jsonl
```

This calls GPT to generate three anchor-removed variants per question:

| Variant | Field | What changes |
|---|---|---|
| A | `weaker_object` / `deictic_removed` | Object replaced by hypernym; deictic words removed |
| B | `object_removed` | Specific object replaced by "object" |
| C | `pronominalized` | All content noun phrases replaced by pronouns |

Resume-safe: already-processed records are skipped automatically.

---

### Step 2 — Inference

```bash
python evaluation/inference.py \
    --model  Qwen/Qwen3-VL-8B-Instruct \
    --model_type vlm \
    --dataset vqa_1k \
    --condition _control_blind \
    --savedir evaluation/logits/vlm
```

**`--model_type`**

| Value | Description |
|---|---|
| `vlm` | Full VLM — image + text |
| `lm` | Backbone decoder — text only (no vision encoder) |

**`--condition`**

| Flag | Dataset | Image | Instruction |
|---|---|---|---|
| `_control_blind` | `vqa1k_control.jsonl` (variants A/B/C) | blank | none |
| `_control_inst_blind` | `vqa1k_control.jsonl` | blank | yes |
| `_control` | `vqa1k_control.jsonl` | real image | none |
| `_blind` | `vqav2_1k_val.json` (base only) | blank | none |

Use `--fill_missing` to resume from a partial JSONL.

---

### Step 2b — Extract Hidden States (for activation steering)

```bash
python evaluation/extract_hidden_states.py \
    --model  Qwen/Qwen3-VL-8B-Instruct \
    --model_type vlm \
    --dataset vqa_1k_control \
    --condition _blind \
    --layers 12,14,16,18,20,22 \
    --resume
```

Captures last-input-token hidden states at specified transformer layers during the prefill pass via forward hooks. Output: `evaluation/hidden_states/pretrained/{model}/{dataset}{condition}.pt` mapping `question_id → {variant → {'hidden_states': Tensor(n_layers, d_model), 'layers': [...]}}`.

Run for both `--condition _blind` and `--condition ''` (sighted) to extract V1 (abstention direction) and V3 (visual grounding direction) respectively. See `notes/future_directions.md §1` for the full methodology.

---

### Step 3 — Score

```bash
python evaluation/score_results.py --results_dir evaluation/logits/vlm
```

---

## Models

| Group | Models |
|---|---|
| VLM | Qwen3-VL-{2B,4B,8B,32B}, InternVL3.5-{2B,8B}, LLaVA-1.5-7B, LLaVA-Mistral-7B, LLaVA-Vicuna-7B |
| VLM backbone decoder | Qwen3-VL (LM), InternVL (LM), LLaVA-1.5 (LM), LLaVA-Mistral (LM), LLaVA-Vicuna (LM) |
| Standalone LLM | Qwen3-{0.6B–32B}, Qwen2.5-7B, Mistral-7B, Vicuna-7B |
| Standalone LLM (think) | Qwen3-{0.6B–32B} with chain-of-thought |

---

## Related Work — Activation Steering & Counting Diagnosis

Key papers for the follow-up project on activation steering for visual grounding. Full notes with methodology and links in `notes/future_directions.md`.

**Activation steering:**

| Paper | Venue | Key contribution |
|---|---|---|
| CAST | ICLR 2025 | Conditional steering — applies only when `cosim(h, condition_vector) > θ` |
| ASD | ACL 2025 | Bidirectional logit contrast + steering vector from factual vs hallucinated activations; POPE +2.88–8.54% |
| ShiftDC | — | Inference-time VLM safety via PyTorch hooks; safety direction = mean(safe)−mean(unsafe) activations |
| GrAInS | ACL 2026 | Norm-preserving steering `h̃ = (h+λv)·‖h‖/‖h+λv‖`; integrated gradients for token attribution |

**Counting diagnosis:**

| Paper | Venue | Key contribution |
|---|---|---|
| The Count Is There, but Misaligned | arXiv 2026 | Proves correct count IS in hidden states; failure = readout misalignment (SVCCA); error-detector → self-correction (+15.6 acc pts) |
| CounterCount | 2025 | Zero-aware prompting to reduce zero-bias |
| LVLM-COUNT | 2024 | Divide-and-conquer patching (≤4 objects per patch); CoT *hurts* counting |

PDF: `latex/AAAI2026/references/pdfs/el-shangiti2026count_misaligned.pdf`
