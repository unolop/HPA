# Dataset Processing Scripts

Simple and modular scripts for processing evaluation results.

## Files

- **eval_utils.py**: Shared utility functions (scoring, parsing, loading)
- **process_dataset.py**: Process a single dataset + condition combination
- **process_all_datasets.py**: Automatically process all datasets and conditions

## Usage

### Option 1: Process All Datasets Automatically

```bash
cd evaluation
python process_all_datasets.py \
    --data_dir /home/work/yuna/HPA/data \
    --output_dir /home/work/yuna/HPA/data/combined \
    --use_encoder  # Optional: compute embedding similarity for VQA
```

This will:
- Find all result files in `models/`, `humans/`, `finetuned/` directories
- Process each dataset + condition combination
- Create combined JSONL files and summary JSON files

### Option 2: Process Specific Dataset

```bash
cd evaluation
python process_dataset.py \
    --dataset mmstar \
    --condition inst_blind \
    --models /path/to/model1.jsonl /path/to/model2.jsonl \
    --humans /path/to/human1.jsonl \
    --finetuned /path/to/finetuned1.jsonl \
    --output_dir /home/work/yuna/HPA/data/combined
```

Available datasets: `mmstar`, `spubench`, `vqa1k`, `vqa5k`

Available conditions: `inst_blind`, `blind`, `sys_inst_blind`, or empty for no condition

## Output Files

For each dataset + condition:

**Combined JSONL**: `{dataset}_{condition}_combined.jsonl`
- All results from all sources in one file
- Each item has `source`, `model`, `correct`, `embedding_similarity` fields
- Finetuned models also have `training_method` field

**Summary JSON**: `{dataset}_{condition}_summary.json`
```json
{
  "dataset": "mmstar",
  "condition": "inst_blind",
  "total_items": 1500,
  "overall": {
    "num_items": 1500,
    "num_correct": 850,
    "accuracy": 0.5667,
    "embedding_similarity": 0.0
  },
  "by_source": {
    "models": {...},
    "finetuned": {...},
    "humans": {...}
  },
  "by_category": {
    "coarse perception | image scene": {...},
    ...
  }
}
```

## Features

- **Automatic file discovery**: Finds all matching files
- **Flexible filtering**: By dataset and condition
- **Source tracking**: Models, humans, finetuned kept separate
- **Multi-choice scoring**: Pattern matching for A/B/C/D answers
- **VQA scoring**: VQA accuracy (min(matches/3, 1.0))
- **Embedding similarity**: Sentence transformer similarity for VQA (optional)
- **Category grouping**: mmstar uses both category and l2_category
- **Training method tracking**: Finetuned models include training_method field

## Examples

Process only VQA 1k with no condition:
```bash
python process_dataset.py \
    --dataset vqa1k \
    --models data/models/*vqa1k.jsonl \
    --finetuned data/finetuned/*vqa1k.jsonl \
    --output_dir data/combined \
    --use_encoder
```

Process mmstar with inst_blind condition:
```bash
python process_dataset.py \
    --dataset mmstar \
    --condition inst_blind \
    --models data/models/*mmstar_inst_blind.jsonl \
    --humans data/humans/*mmstar*.jsonl \
    --finetuned data/finetuned/*mmstar_inst_blind.jsonl \
    --output_dir data/combined
```
