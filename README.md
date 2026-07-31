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
