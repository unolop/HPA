# HPA — Human-calibrated Prior Analysis of Vision-Language Models

**Paper:** *When Models Answer Without Seeing: Human-Calibrated Diagnosis of Linguistic Prior Exploitation in Vision-Language Models*
**Target:** AAAI Human-AI Alignment track

Models are prompted on VQA questions with the image replaced by a blank/uninformative image. The study measures how much VLMs exploit linguistic priors rather than visual content, and compares this to human question difficulty.

---

## Repository Structure

```
HPA/
├── evaluation/          # Inference + scoring pipeline
│   ├── inference.py         # Main inference engine (all models + conditions)
│   ├── inference_api.py     # API-based inference (Gemini etc.)
│   ├── utils.py             # Dataset loading, prompt formatting
│   ├── score_results.py     # Score JSONL outputs → VQA accuracy
│   ├── score_humans.py      # Process raw human study responses
│   └── logits/              # Inference outputs (JSONL per model × condition)
│       ├── vlm/             # Full VLMs (image + text)
│       ├── lm_decoder/      # VLM backbone decoder (text-only)
│       ├── backbone_nothink/# Standalone LLMs, no chain-of-thought
│       ├── backbone_think/  # Standalone LLMs, chain-of-thought enabled
│       └── ablation/        # Image-type ablation (gray/white/noise vs blank)
│
├── analysis/
│   ├── session2/            # Main analysis notebooks (see §Analysis Notebooks)
│   │   ├── *.ipynb              # Core notebooks (numbered 01–14)
│   │   ├── export_agreement_heatmaps.py  # Generates appendix heatmap figures
│   │   ├── exports/             # Cached CSV/parquet outputs (model+human responses)
│   │   ├── quadrants/           # Per-question quadrant CSVs
│   │   └── archive/             # Exploratory / deprecated notebooks
│   ├── utils/               # Shared analysis utilities (see §Utils)
│   │   ├── model_registry.py
│   │   ├── load_session.py
│   │   ├── vqa.py
│   │   ├── agreement.py
│   │   ├── abstention.py
│   │   ├── constants.py
│   │   ├── corr.py
│   │   ├── quadrants.py
│   │   ├── normalize_korean.py
│   │   └── preprocessing/   # Data preprocessing scripts
│   └── run_paper_figures.py # Generates all main paper figures
│
├── dataset/             # VQA question sets + control stimuli
│   ├── vqa/
│   │   ├── vqav2_1k_val.json      # 1 000-question VQA v2 subset
│   │   └── vqa1k_control.jsonl    # Control stimuli (+ pronominalized/weaker variants)
│   ├── blank_224.png              # All-black 224×224 (default blind image)
│   ├── gray_224.png               # Uniform gray 128
│   ├── white_224.png              # All-white 255
│   └── noise_224.png              # Random pixels (seed=42)
│
├── scripts/             # Shell scripts for batch inference
├── latex/               # Paper manuscript + figures
│   └── AnonymousSubmission/LaTeX/
│       ├── paper.tex
│       └── figures/     # All paper figures (generated)
├── training/            # Fine-tuning outputs
└── logs/                # Inference logs
```

---

## Evaluation Pipeline

### 1. Inference

**Script:** `evaluation/inference.py`

Runs a model on VQA questions under a specified condition, saving logprobs to JSONL.

```bash
conda run -n zero python evaluation/inference.py \
  --model <HF model id>        \   # e.g. Qwen/Qwen3-VL-8B-Instruct
  --model_type <vlm|lm>        \   # vlm = full VLM; lm = text-only (no vision)
  --dataset vqa_1k             \   # dataset name
  --condition _control_blind   \   # condition suffix (see below)
  --savedir evaluation/logits/vlm  # output root
```

**Conditions:**

| Flag | Dataset loaded | Image | Instruction |
|---|---|---|---|
| `_control_blind` | `vqa1k_control.jsonl` (+ variants A/B/C) | blank | none |
| `_control_inst_blind` | `vqa1k_control.jsonl` | blank | yes |
| `_control` | `vqa1k_control.jsonl` | real image | none |
| `_blind` | `vqav2_1k_val.json` (base only) | blank | none |

**Image override** (for ablation studies):
```bash
--image_override gray|white|noise   # default: blank
```

**Quantization:**
```bash
--quantization_bit 4   # required for 32B models
--fill_missing         # resume from partial JSONL (safe to re-run)
```

**Output:** `<savedir>/pretrained/<model_dir>/<dataset><condition>[_<image>].jsonl`
Each row: `{id, question, answers, generated_answers, generated_logits, ...control_variants}`

---

### 2. Export Responses

**Notebook:** `analysis/session2/09_align_human.ipynb`

Reads all model JSONL files via `model_registry.py`, preprocesses answers with `preprocess_answer()`, and exports:

| File | Contents |
|---|---|
| `exports/responses_model_blind.csv` | All models, blind condition, all variants A/B/C |
| `exports/responses_model_inst_blind.csv` | All models, inst_blind condition |
| `exports/responses_model_control.csv` | All models, control (real image) condition |
| `exports/responses_human.csv` | 36 participants × 113 questions, inst_blind |

Re-run this notebook after adding new models or completing inference.

---

### 3. Score & Analyse

**`analysis/run_paper_figures.py`** — generates all main paper figures from the CSV exports:

```bash
conda run -n zero python analysis/run_paper_figures.py
```

Output goes to `latex/AnonymousSubmission/LaTeX/figures/`.

**Modular export scripts** (each writes to its own `figures/` subfolder):

| Script | Output folder | What it generates |
|---|---|---|
| `export_agreement_scatter.py` | `agreement_scatter/` | Per-model SBERT×exact-match scatter; SBERT vs. scale (Qwen3) |
| `figures/hh_hm/variants_family.py` | `hh_hm/variants_family/` | Family-level HM agreement across C→B→A with HH ceiling |
| `export_scale_plots.py --metric agreement` | `agreement_scale/` | Agreement vs. model parameter count |
| `export_scale_plots.py --metric accuracy` | `accuracy_scale/` | Accuracy vs. model parameter count |
| `export_accuracy_variants.py` | `accuracy_lineplot/` | Accuracy degradation across variants C→B→A (per group/family/model/7B) |
| `export_accuracy_scatter.py --agg model_groups` | `accuracy_scatter/` | Per-question human vs. model accuracy scatter, aggregated by group |
| `export_accuracy_scatter.py --agg model_family` | `accuracy_scatter/` | Same scatter aggregated by model family |
| `export_accuracy_quadrant.py` | `accuracy_scatter/` | Per-question Jaccard-coloured scatter: blind×inst and inst×inst |
| `export_answer_dist.py` | `answer_dist_barplot/` | Yes/no and number answer distributions |
| `export_instruction_effect.py` | `instruction_effect/` | Blind→inst_blind soft/hard abstention and response-change rates |
| `export_entity_analysis.py` | `entity_analysis/` | Entity distribution, Pearson r, SBERT, degradation, instruction sensitivity |

`run_paper_figures.py` is a thin orchestrator that calls all of the above in sequence.

---

## Figure Naming Convention

All export scripts follow a consistent folder + filename pattern.

**Folder**: `{phenomenon}_{plot_type}`
- phenomenon: `accuracy`, `agreement`, `answer_dist`, `instruction_effect`, `entity`
- plot_type: `lineplot`, `scatter`, `heatmap`, `barplot`, `scale`

**Filename tokens** (joined by `_`, only include what varies for that script):

| Token | Values | Meaning |
|---|---|---|
| `{condition}` | `blind`, `inst_blind`, `control` | Model inference condition |
| `v{VAR}` | `vC`, `vB`, `vA`, `vABC` | Control variant(s) plotted |
| `{metric}` | `sbert`, `simcse`, `bertscore`, `chrf`, `rouge1`, `jaccard`, `exact` | Agreement scoring method |
| `{qtype}` | `open`, `yn`, `number` | Question type filter |
| `{agg}` | `groups`, `family`, `models`, `7b` | Aggregation level |
| `q{N}` | e.g. `q113` | Number of questions |
| `h{N}` | e.g. `h40` | Number of human participants (human-study subset only) |

**Examples:**
- `accuracy_lineplot/inst_blind_vABC_groups.png` — accuracy degradation across C→B→A, aggregated by model group, inst_blind condition
- `accuracy_scatter/inst_blind_vC_family_q113_h40.png` — per-question scatter (human vs. model accuracy), variant C, aggregated by model family, 113 questions, 40 humans
- `hh_hm/variants_family/inst_blind_sbert_family_vABC_q113_yesno.png` — family-level SBERT agreement across variants with HH ceiling
- `answer_dist_barplot/blind_vC_yn_q67_h40.png` — yes/no distribution, blind condition, 67 questions
- `instruction_effect/soft_abstention_vC_q1000.png` — soft abstention dumbbell, full 1000-question set
- `agreement_scale/inst_blind_models_sbert_q113_h40.png` — SBERT agreement vs. model scale, per-model points

---

### 4. Human Scoring

**Script:** `evaluation/score_humans.py`

Processes raw human participant JSON files from `evaluation/humans/by_participant/` into `exports/responses_human.csv`.

---

## Analysis Notebooks

All core notebooks are in `analysis/session2/`. Run in order — each exports CSVs consumed by later notebooks and `run_paper_figures.py`.

| Notebook | Paper § | What it does |
|---|---|---|
| `01_setup.ipynb` | §3 | Characterises the 113-question stimulus set; entity/operator breakdown |
| `02_setup_embeddings.ipynb` | App | t-SNE of 1k VQA question embeddings → `sampling_tsne.png` |
| `04_prior_accuracy.ipynb` | §4 | VLM accuracy across all tiers (VLM / LM decoder / standalone) × conditions; Table 1 |
| `04b_prior_inst_comparison.ipynb` | §4 | Focused blind→inst_blind instruction effect; response change rates |
| `06_prior_lm_decoder.ipynb` | §5 | Three-tier text-only comparison; backbone yes-rate reversal (VLM "no" bias is vision-specific) |
| `07_char_confidence.ipynb` | §4 | Logprob confidence analysis; overconfidence under blind (80–90% LP > −0.5) |
| `08_char_abstention.ipynb` | §5 | Soft abstention collapse rates per model; generates `abstention_rates.png`, `abstention_collapse.png` |
| `09_align_human.ipynb` | §6 | Loads human study (36 participants, 113 Qs); computes difficulty curve ICC=0.963; **exports all CSVs** |
| `10_align_agreement.ipynb` | §6 | Pairwise inter-rater agreement (SBERT/chrF/ROUGE-1/BERTScore/SimCSE); builds `pair_cache.parquet` |
| `12_align_quadrant.ipynb` | §6.2 | Per-question scatter: human acc (x) vs VLM acc (y); → `fig_hm_quadrant.png` |
| `13_answer_dist.ipynb` | §6.1 | Yes/No and number answer distributions; **save manually** → `fig13_yn_*.png`, `fig13_num_*.png` |
| `14_entity_type_analysis.ipynb` | §6.2 | Pearson r (human vs VLM accuracy) per entity type; three-regime finding |

Exploratory and supporting notebooks are in `analysis/session2/archive/`.

---

## Utils Reference (`analysis/utils/`)

### `model_registry.py`
Canonical model list and group assignments.

- `MODEL_TYPE` — `{display_label → group_name}` (VLM / VLM backbone decoder / standalone LLM / standalone LLM (think))
- `default_all_models(base)` — full model registry: `{label → (results_dir, model_dir_name)}`
- `backbone_nothink_models(base)` — Qwen3 + Mistral + Vicuna + Phi standalone LLMs (no CoT)
- `backbone_think_models(base)` — Qwen3 standalone LLMs with chain-of-thought
- `model_groups(base)` — models grouped by type for plotting
- `flatten_all_models(base)` — `{label → concrete results path}`

### `load_session.py`
Central data loader for notebooks.

- `load_human_data(base)` → `(participants, common_qids, df, mapper)` — loads 36 human participants, computes per-question difficulty
- `load_model_results(base, q_ids)` → `(model_q_acc, MODEL_TYPE, MODELS, ALL_MODELS)` — loads blind/inst_blind accuracy for all registered models; silently skips missing files

### `vqa.py`
VQA answer normalisation and accuracy scoring.

- `preprocess_answer(text)` — full VQA preprocessing pipeline: `strip_think_answer` → `process_punctuation` → `process_digit_article`; used by all models and humans
- `vqa_accuracy(prediction, gt_answers)` → float — standard VQA v2 accuracy (min(count/3, 1))
- `VQAAnswerMapper` — maps raw answers to normalised canonical forms

### `agreement.py`
Pairwise inter-rater agreement metrics.

Computes agreement between any two answer lists using:
- **Exact match** — binary (normalised)
- **Jaccard** — token-level overlap
- **ROUGE-1** — unigram F1
- **chrF** — character n-gram F1
- **SBERT** — cosine similarity (`all-mpnet-base-v2`); cached in `embeddings_all-mpnet-base-v2.npz`
- **SimCSE** — cosine similarity (`sup-simcse-roberta-large`); cached in `embeddings_sup-simcse-roberta-large.npz`
- **BERTScore** — precision/recall/F1 from `microsoft/deberta-xlarge-mnli`

Pairwise results cached in `exports/pair_cache.parquet` to avoid recomputation.

### `abstention.py`
Classifies model outputs into five categories.

- `classify(output, gt_answers)` → `str` — returns one of: `hard_abstained`, `soft_abstained`, `hallucinated_correct`, `hallucinated_wrong`, `degenerate`
- `is_abstained(cls)` → `bool`
- **Hard abstention:** explicit refusal patterns (`"cannot"`, `"no image"`, `"I cannot see"`, etc.)
- **Soft abstention:** hedge words (`"nothing"`, `"unknown"`, `"unanswerable"`, etc.)

### `constants.py`
Shared plotting constants — colours, orders, labels.

- `MODEL_ORDER`, `MODEL_LABEL` — canonical VLM display order + labels
- `COND_LABEL`, `COND_COLOR`, `CONDITIONS` — condition formatting (blind / inst_blind / control)
- `VARIANT_ORDER`, `VARIANT_LABELS`, `VARIANT_COLORS` — degradation variants A/B/C
- `GROUP_COLORS`, `TIER_ORDER`, `TIER_COLORS`, `TIER_STYLE` — model group colours + plot styles
- `CLASS_ORDER`, `CLASS_COLOR`, `CLASS_LABEL` — abstention class taxonomy colours
- `CONTROL_TYPES`, `CT_LABELS`, `CT_TO_VARIANT` — control question variant names
- `VLM_MODELS`, `BB_MODELS`, `TRIPLES` — model lists for tier comparison plots

### `corr.py`
Pearson/Spearman correlation helpers for human-model alignment.

- Computes per-entity-type Pearson r between human difficulty and VLM accuracy
- Used by `14_entity_type_analysis.ipynb` and `run_paper_figures.py`

### `quadrants.py`
Per-question quadrant assignment for the human×model accuracy scatter.

- `assign_quadrant(human_acc, model_acc, threshold)` → Q1–Q4 label
- Used by `12_align_quadrant.ipynb` and `run_paper_figures.py`

### `normalize_korean.py`
Korean text normalisation for human participant answers (translation + romanisation).

### `preprocessing/`
Data preparation scripts (run once):

- `preprocess.py` — tokenisation, condition mapping, confidence extraction
- `aggregate.py` — aggregates responses by answer type and model
- `process_raw_human_responses.py` — converts raw human JSON → standardised format
- `build_training_data.py` — builds fine-tuning datasets from responses
- `create_gt_answer_from_blind.py` — infers ground-truth from blind responses

---

## Paper Figures

All figures are generated by running `analysis/run_paper_figures.py` (or each export script individually).

| Figure | Script | Output path |
|---|---|---|
| Per-model agreement scatter (SBERT × exact) | `export_agreement_scatter.py` | `agreement_scatter/inst_blind_vC_sbert_exact_models*.png` |
| SBERT alignment vs. scale (Qwen3) | `export_agreement_scatter.py` | `agreement_scatter/inst_blind_vC_sbert_scale_qwen3*.png` |
| Per-question quadrant scatter (blind) | `export_accuracy_quadrant.py` | `accuracy_scatter/blind_vC_jaccard_vlm*.png` |
| Per-question quadrant scatter (inst_blind, annotated) | `export_accuracy_quadrant.py` | `accuracy_scatter/inst_blind_vC_jaccard_vlm_annotated*.png` |
| Entity type + Pearson r 2-panel | `export_entity_analysis.py` | `entity_analysis/fig_hm_alignment*.png` |
| Family-level HM variant figure | `figures/hh_hm/variants_family.py` | `hh_hm/variants_family/inst_blind_sbert_family_vABC_q113_yesno.png` |
| Soft/hard abstention dumbbell | `export_instruction_effect.py` | `instruction_effect/soft_abstention_vC_*.png` |
| Response change rate | `export_instruction_effect.py` | `instruction_effect/response_change_vC_*.png` |
| Accuracy degradation C→B→A | `export_accuracy_variants.py` | `accuracy_lineplot/{cond}_vABC_groups.png` |
| Agreement by variant lineplots | `export_agreement_variants.py` | `agreement_lineplot/inst_blind_vABC_{metric}_groups.png` |
| Abstention rates / collapse | `08_char_abstention.ipynb` *(run manually)* | `figures/abstention_*.png` |

---

## Models

| Group | Models |
|---|---|
| VLM | Qwen3-VL-{2B,4B,8B,32B}, InternVL3.5-{1B,2B,8B}, LLaVA-1.5-7B, LLaVA-Mistral-7B, LLaVA-Vicuna-7B |
| VLM backbone decoder | Qwen3-VL-32B (LM), LLaVA-1.5 (LM), LLaVA-Mistral (LM), LLaVA-Vicuna (LM) |
| Standalone LLM | Qwen3-{0.6B,1.7B,4B,8B,32B}, Qwen2.5-7B, Mistral-7B, Vicuna-{7B,13B}, Phi-3.5-mini |
| Standalone LLM (think) | Qwen3-{0.6B,1.7B,4B,8B,32B} with chain-of-thought |

---

## Inference Scripts

| Script | Purpose |
|---|---|
| `scripts/run_after_4b.sh` | Wait for Qwen3-4B fill, then run all pending 32B jobs |
| `scripts/run_ablation_image_type.sh` | Image-type ablation (blank/gray/white/noise) on Qwen3-VL-4B |
| `scripts/run_fill_32b.sh` | Fill incomplete 32B inference conditions |
| `scripts/run_fill_lm_decoder_32b.sh` | Fill lm_decoder/Qwen3-VL-32B inst_blind |

Monitor running jobs: `tail -f logs/<script>.log`

---

## Environment

```bash
conda activate zero        # main environment (Python 3.9, ms-swift, transformers)
export HF_HOME=/home/david/Desktop/yuna/.cache/hf
```

Git push requires SSH key:
```bash
export GIT_SSH_COMMAND="ssh -i /home/david/Desktop/yuna/.ssh/id_ed25519"
```
