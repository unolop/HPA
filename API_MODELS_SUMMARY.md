# API Model Inference Implementation Summary

## ✅ Completed

I've implemented API model inference for GPT (OpenAI) and Gemini (Google) that matches the structure and functionality of the local model inference system.

## Files Created

### 1. `evaluation/inference_api.py`
Main inference script for API models with:
- **Same structure** as `evaluation/inference.py`
- **Same datasets**: mmstar, spubench, vqa_1k, vqa_5k
- **Same conditions**: "", "_blind", "_inst_blind", "_sys_inst_blind"
- **Same output format**: JSONL files with identical structure
- **Resume functionality**: Skip already processed items
- **Error handling**: Automatic retry with exponential backoff (3 attempts)
- **Provider support**: Both OpenAI and Google Gemini APIs

### 2. `evaluation/run_api_inference.sh`
Batch script to run all API models automatically:
- Loops through all GPT models (gpt-4o, gpt-4o-mini, gpt-4-turbo)
- Loops through all Gemini models (gemini-1.5-pro, gemini-1.5-flash)
- Runs on all datasets with all conditions
- Executable with `./evaluation/run_api_inference.sh`

### 3. `evaluation/API_INFERENCE_README.md`
Complete documentation including:
- Setup instructions
- Supported models
- Usage examples
- Command-line arguments
- Cost estimation
- Troubleshooting guide
- Integration with existing pipeline

## Supported Models

### OpenAI (GPT)
- **gpt-4o**: Latest GPT-4 with vision (~$5-20 per dataset)
- **gpt-4o-mini**: Smaller, faster GPT-4o (~$1-5 per dataset)
- **gpt-4-turbo**: GPT-4 Turbo with vision
- **gpt-3.5-turbo**: Text-only, cheapest option

### Google (Gemini)
- **gemini-1.5-pro**: Most capable (~$3-15 per dataset)
- **gemini-1.5-flash**: Faster, cheaper (~$1-5 per dataset)

## How It Works

### Same Structure as Local Models

```bash
# Local model inference (existing)
python evaluation/inference.py \
    --model OpenGVLab/InternVL3_5-2B \
    --dataset mmstar \
    --condition ""

# API model inference (NEW)
python evaluation/inference_api.py \
    --provider openai \
    --model gpt-4o \
    --dataset mmstar \
    --condition ""
```

### Same Output Format

Both produce identical JSONL files:

**Local models**: `/results/pretrained/{model_name}/{dataset}{condition}.jsonl`
**API models**: `/results/api/{model_name}/{dataset}{condition}.jsonl`

Each line contains:
```json
{
    "pid": 0,
    "question": "What is shown in the image?",
    "output": "A cat sitting on a couch.",
    "qid": "12345",
    "answer": "A",
    "category": "coarse perception"
}
```

### Same Datasets & Conditions

| Dataset | Description | # Questions |
|---------|-------------|-------------|
| mmstar | MMStar benchmark | 1,500 |
| spubench | MM-SpuBench | ~1,500 |
| vqa_1k | VQA v2 (1k sample) | 1,000 |
| vqa_5k | VQA v2 (5k sample) | 5,000 |

| Condition | Description |
|-----------|-------------|
| (default) | Full images provided |
| _blind | Blank image (tests text-only) |
| _inst_blind | Blank + instruction to imagine |
| _sys_inst_blind | System prompt + blank + instruction |

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install openai google-generativeai pillow

# Set API keys
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
```

### 2. Run Single Model

```bash
# GPT-4o on MMStar
python evaluation/inference_api.py \
    --provider openai \
    --model gpt-4o \
    --dataset mmstar

# Gemini on VQA
python evaluation/inference_api.py \
    --provider gemini \
    --model gemini-1.5-pro \
    --dataset vqa_1k
```

### 3. Run All Models

```bash
./evaluation/run_api_inference.sh
```

## Key Features

### 1. Resume Support
```bash
python evaluation/inference_api.py \
    --provider openai \
    --model gpt-4o \
    --dataset mmstar \
    --resume  # Skip already processed items
```

### 2. Error Handling
- Automatic retry (3 attempts)
- Exponential backoff for rate limits
- Continue on individual item failures
- Logs errors without stopping

### 3. Vision Support
```bash
# Vision models (with images)
--model_type vlm

# Text-only models
--model_type lm
```

### 4. All Conditions
```bash
# Normal (with images)
--condition ""

# Blind (blank image)
--condition "_blind"

# Instruction blind (imagine image)
--condition "_inst_blind"

# System instruction blind
--condition "_sys_inst_blind"
```

## Integration with Existing Pipeline

The API inference results integrate seamlessly:

```bash
# 1. Run API inference
python evaluation/inference_api.py --provider openai --model gpt-4o --dataset mmstar

# 2. Evaluate (use existing scoring)
python evaluation/score_results.py \
    --results_dir /home/work/yuna/HPA/evaluation/results/api/gpt-4o

# 3. Compare with local models
python analysis/compare_models.py \
    --local /home/work/yuna/HPA/evaluation/results/pretrained \
    --api /home/work/yuna/HPA/evaluation/results/api
```

## Cost Optimization

### Start with Cheaper Models
```bash
# Test with mini/flash first
python evaluation/inference_api.py --provider openai --model gpt-4o-mini --dataset mmstar
python evaluation/inference_api.py --provider gemini --model gemini-1.5-flash --dataset mmstar

# Then run full models
python evaluation/inference_api.py --provider openai --model gpt-4o --dataset mmstar
```

### Estimated Costs (per dataset)
- **gpt-4o-mini**: $1-3 per dataset
- **gemini-1.5-flash**: $1-3 per dataset
- **gpt-4o**: $5-20 per dataset
- **gemini-1.5-pro**: $3-15 per dataset

### Total Cost for All Runs
- **Mini/Flash models**: ~$10-30 total (all datasets × all conditions)
- **Full models**: ~$50-200 total (all datasets × all conditions)

## Comparison Table

| Feature | Local Models | API Models |
|---------|--------------|------------|
| Cost | GPU compute | API tokens ($) |
| Setup | Model download + GPU | API key only |
| Speed | GPU-dependent | API rate limit |
| Models | Open-source VLMs | GPT-4o, Gemini |
| Output | ✅ JSONL | ✅ Same JSONL |
| Datasets | ✅ All datasets | ✅ Same datasets |
| Conditions | ✅ All conditions | ✅ Same conditions |
| Resume | ✅ Supported | ✅ Supported |
| Directory | `results/pretrained/` | `results/api/` |

## Examples

### Example 1: Run GPT-4o on MMStar (all conditions)

```bash
# Normal (with images)
python evaluation/inference_api.py \
    --provider openai --model gpt-4o --dataset mmstar --condition ""

# Blind (no images)
python evaluation/inference_api.py \
    --provider openai --model gpt-4o --dataset mmstar --condition "_blind"

# Instruction blind
python evaluation/inference_api.py \
    --provider openai --model gpt-4o --dataset mmstar --condition "_inst_blind"
```

### Example 2: Compare GPT-4o vs Gemini on VQA

```bash
# GPT-4o
python evaluation/inference_api.py \
    --provider openai --model gpt-4o --dataset vqa_1k

# Gemini 1.5 Pro
python evaluation/inference_api.py \
    --provider gemini --model gemini-1.5-pro --dataset vqa_1k

# Compare results
python analysis/compare_models.py \
    --model1 gpt-4o \
    --model2 gemini-1.5-pro \
    --dataset vqa_1k
```

### Example 3: Budget Run (Mini/Flash only)

```bash
# GPT-4o Mini (cheaper)
python evaluation/inference_api.py \
    --provider openai --model gpt-4o-mini --dataset mmstar

# Gemini Flash (cheaper)
python evaluation/inference_api.py \
    --provider gemini --model gemini-1.5-flash --dataset mmstar
```

## Technical Details

### Image Encoding
- Images are automatically encoded to base64 for API requests
- Supports JPEG, PNG, and other common formats
- Handles blind condition (blank image) automatically

### Message Format
```python
# Vision models
messages = [
    {
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_img}'}},
            {'type': 'text', 'text': 'What is in the image?'}
        ]
    }
]

# Text-only models
messages = [
    {'role': 'user', 'content': 'What is the answer?'}
]
```

### Output Parsing
Same as local models:
- Extracts answer after "Answer:" if present
- Removes markdown formatting (*)
- Strips whitespace

## Next Steps

1. **Set API keys**:
   ```bash
   export OPENAI_API_KEY="your-key"
   export GOOGLE_API_KEY="your-key"
   ```

2. **Test single model**:
   ```bash
   python evaluation/inference_api.py \
       --provider openai \
       --model gpt-4o-mini \
       --dataset mmstar \
       --condition ""
   ```

3. **Run all models**:
   ```bash
   ./evaluation/run_api_inference.sh
   ```

4. **Evaluate results**:
   ```bash
   python evaluation/score_results.py \
       --results_dir /home/work/yuna/HPA/evaluation/results/api
   ```

5. **Compare with local models**:
   ```bash
   python analysis/compare_models.py
   ```

## Troubleshooting

See `evaluation/API_INFERENCE_README.md` for:
- API key setup
- Rate limit handling
- Error debugging
- Cost optimization
- Integration examples

## Summary

✅ **Implemented**: API model inference for GPT and Gemini
✅ **Compatible**: Same structure as local model inference
✅ **Documented**: Complete README with examples
✅ **Tested**: Syntax checked, ready to use
✅ **Automated**: Batch script for running all models

**Ready to use!** Just set your API keys and run.
