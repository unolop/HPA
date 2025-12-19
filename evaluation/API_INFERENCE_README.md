# API Model Inference (GPT & Gemini)

This guide explains how to run inference using API models (OpenAI GPT and Google Gemini) with the same structure as local model inference.

## Overview

The `inference_api.py` script provides API-based inference that:
- Uses the same datasets as local models (mmstar, spubench, vqa_1k, vqa_5k)
- Supports the same conditions (blind, inst_blind, sys_inst_blind)
- Saves results in the same JSONL format
- Supports resume functionality
- Handles both vision (VLM) and text-only (LM) modes

## Setup

### 1. Install Required Packages

```bash
pip install openai google-generativeai pillow
```

### 2. Set API Keys

#### OpenAI (GPT models)
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

#### Google (Gemini models)
```bash
export GOOGLE_API_KEY="your-google-api-key-here"
```

You can also add these to your `~/.bashrc` or `~/.zshrc` to make them persistent.

## Supported Models

### OpenAI GPT Models
- **gpt-4o** (latest GPT-4 with vision)
- **gpt-4o-mini** (smaller, faster GPT-4o)
- **gpt-4-turbo** (GPT-4 Turbo with vision)
- **gpt-3.5-turbo** (text-only, cheaper)

### Google Gemini Models
- **gemini-1.5-pro** (most capable)
- **gemini-1.5-flash** (faster, cheaper)
- **gemini-pro** (older version)
- **gemini-pro-vision** (with vision support)

## Usage

### Basic Usage

#### Single Model Inference

```bash
# OpenAI GPT-4o on MMStar
python evaluation/inference_api.py \
    --provider openai \
    --model gpt-4o \
    --dataset mmstar \
    --condition "" \
    --model_type vlm

# Google Gemini on VQA
python evaluation/inference_api.py \
    --provider gemini \
    --model gemini-1.5-pro \
    --dataset vqa_1k \
    --condition "" \
    --model_type vlm

# Blind condition (no images)
python evaluation/inference_api.py \
    --provider openai \
    --model gpt-4o \
    --dataset mmstar \
    --condition "_blind" \
    --model_type vlm
```

### Batch Processing

Run all API models on all datasets:

```bash
./evaluation/run_api_inference.sh
```

This will run:
- All GPT models (gpt-4o, gpt-4o-mini, gpt-4-turbo)
- All Gemini models (gemini-1.5-pro, gemini-1.5-flash)
- On all datasets (mmstar, spubench, vqa_1k)
- With all conditions (default, _blind, _inst_blind)

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--provider` | str | `openai` | API provider: `openai` or `gemini` |
| `--model` | str | `gpt-4o` | Model name (see supported models above) |
| `--model_type` | str | `vlm` | Model type: `vlm` (vision) or `lm` (text-only) |
| `--dataset` | str | `mmstar` | Dataset: `mmstar`, `spubench`, `vqa_1k`, `vqa_5k` |
| `--condition` | str | `""` | Condition suffix: `""`, `_blind`, `_inst_blind`, `_sys_inst_blind` |
| `--savedir` | str | `/home/work/yuna/HPA/evaluation/results` | Output directory |
| `--resume` | flag | False | Resume from existing output file |
| `--max_token_length` | int | 4096 | Maximum output tokens |

## Output Format

Results are saved in JSONL format (same as local models):

```
/home/work/yuna/HPA/evaluation/results/api/{model_name}/{dataset}{condition}.jsonl
```

### Example Output Structure

Each line is a JSON object:

```json
{
    "pid": 0,
    "question": "What is in the image?",
    "output": "The image shows a cat sitting on a couch.",
    "qid": "12345",
    "answer": "A",
    "category": "coarse perception"
}
```

## Conditions Explained

### Default (no suffix)
- **Full images provided**
- Standard inference

```bash
--condition ""
```

### Blind (`_blind`)
- **Blank image** (224x224 white image)
- Tests text-only understanding

```bash
--condition "_blind"
```

### Instruction Blind (`_inst_blind`)
- **Blank image + instruction**
- Tells model to imagine an image

```bash
--condition "_inst_blind"
```

### System Instruction Blind (`_sys_inst_blind`)
- **System message + blank image + instruction**
- Uses system prompt

```bash
--condition "_sys_inst_blind"
```

## Examples

### 1. Run GPT-4o on MMStar (all conditions)

```bash
# Normal (with images)
python evaluation/inference_api.py \
    --provider openai \
    --model gpt-4o \
    --dataset mmstar \
    --condition ""

# Blind (no images)
python evaluation/inference_api.py \
    --provider openai \
    --model gpt-4o \
    --dataset mmstar \
    --condition "_blind"

# Instruction blind
python evaluation/inference_api.py \
    --provider openai \
    --model gpt-4o \
    --dataset mmstar \
    --condition "_inst_blind"
```

### 2. Run Gemini on VQA with Resume

```bash
# Start inference
python evaluation/inference_api.py \
    --provider gemini \
    --model gemini-1.5-pro \
    --dataset vqa_1k \
    --resume

# If interrupted, resume will skip already processed items
```

### 3. Text-Only Model (GPT-3.5)

```bash
python evaluation/inference_api.py \
    --provider openai \
    --model gpt-3.5-turbo \
    --dataset vqa_1k \
    --model_type lm \
    --condition "_blind"
```

## Error Handling

The script includes:
- **Automatic retries** with exponential backoff (3 attempts)
- **Error logging** for failed items
- **Resume support** to continue from interruptions
- **API rate limit handling**

## Cost Estimation

### OpenAI Pricing (approximate, check latest pricing)
- **GPT-4o**: $5/1M input tokens, $15/1M output tokens
- **GPT-4o-mini**: $0.15/1M input tokens, $0.60/1M output tokens
- **GPT-4-turbo**: $10/1M input tokens, $30/1M output tokens

### Google Gemini Pricing (approximate)
- **Gemini 1.5 Pro**: $3.5/1M input tokens, $10.5/1M output tokens
- **Gemini 1.5 Flash**: $0.35/1M input tokens, $1.05/1M output tokens

### Estimated Costs per Dataset
- **MMStar** (1,500 questions): ~$5-20 per model depending on model choice
- **VQA 1k** (1,000 questions): ~$3-15 per model
- **SpuBench**: Similar to MMStar

**Tip**: Start with mini/flash models for testing, then use full models for final results.

## Comparison with Local Models

| Feature | Local Models (inference.py) | API Models (inference_api.py) |
|---------|---------------------------|------------------------------|
| Cost | GPU compute | API tokens |
| Speed | Depends on GPU | Depends on API rate limits |
| Setup | Model download + GPU | API key only |
| Output Format | ✅ Same JSONL | ✅ Same JSONL |
| Datasets | ✅ Same datasets | ✅ Same datasets |
| Conditions | ✅ Same conditions | ✅ Same conditions |
| Resume | ✅ Supported | ✅ Supported |

## Troubleshooting

### 1. API Key Not Found
```
Error: OPENAI_API_KEY not found
```
**Solution**: Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### 2. Rate Limit Errors
```
Error: Rate limit exceeded
```
**Solution**: The script will automatically retry. For persistent issues:
- Use `--resume` to continue later
- Wait a few minutes between runs
- Consider upgrading API tier

### 3. Image Encoding Errors
```
Error: Failed to encode image
```
**Solution**: Check that image files exist and are valid:
```bash
ls -lh /home/work/yuna/HPA/data/blank_224.png
```

### 4. Out of Memory
```
Error: Image too large
```
**Solution**: API models handle this automatically (images are sent as base64). If issues persist, check image file sizes.

## Integration with Existing Pipeline

The API inference results integrate seamlessly with the existing evaluation pipeline:

```bash
# 1. Run API inference
python evaluation/inference_api.py --provider openai --model gpt-4o --dataset mmstar

# 2. Evaluate results (use existing scoring scripts)
python evaluation/score_models.py \
    --results_dir /home/work/yuna/HPA/evaluation/results/api/gpt-4o

# 3. Compare with local models
python analysis/compare_models.py \
    --local_results /home/work/yuna/HPA/evaluation/results/pretrained \
    --api_results /home/work/yuna/HPA/evaluation/results/api
```

## Notes

1. **Vision Support**: GPT-4o, GPT-4-turbo, and all Gemini models support vision. GPT-3.5-turbo is text-only.

2. **Token Limits**: Default is 4096 tokens. Increase if needed:
   ```bash
   --max_token_length 8192
   ```

3. **Temperature**: Fixed at 0 for reproducibility. Modify in `inference_api.py` if needed.

4. **Image Format**: Images are automatically encoded to base64 for API requests.

5. **Resume Functionality**: Always use `--resume` to avoid re-processing completed items.

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the code in `evaluation/inference_api.py`
- Check API provider documentation:
  - OpenAI: https://platform.openai.com/docs
  - Google: https://ai.google.dev/docs
