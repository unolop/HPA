# Fix for Human MC/VQA Results with Nested Structures

## Problem

When saving human results to CSV, nested structures (lists, dicts) in the data get converted to string representations:

```python
# Before: This creates strings like "['A', 'B', 'C']" instead of actual lists
pd.DataFrame(choice_results).to_csv('human_mc_per_question.csv')
```

When you read it back:
```python
df = pd.read_csv('human_mc_per_question.csv')
print(df['extracted_choices'][0])
# Output: "['A', 'B', 'C']"  ❌ String, not a list!
```

## Solution

Now saves **both JSONL (for code) and CSV (for viewing)**:

### 1. JSONL Files (for code/analysis)
- **Preserves nested structures** (lists, dicts, etc.)
- Use for all programmatic analysis
- Files:
  - `human_vqa_per_question.jsonl`
  - `human_mc_per_question.jsonl`

### 2. CSV Files (for quick viewing)
- Human-readable in Excel/spreadsheet apps
- **Nested structures are strings** (not for code use)
- Files:
  - `human_vqa_per_question.csv`
  - `human_mc_per_question.csv`

## How to Use

### ✅ Correct: Load JSONL files

```python
from evaluation.score_humans import load_human_results

# Load MC results
df = load_human_results('/home/work/yuna/HPA/evaluation/scored/humans/human_mc_per_question.jsonl')

# Now nested structures are properly parsed!
print(df['extracted_choices'][0])
# Output: ['A', 'B', 'C', 'A']  ✅ Actual list!

print(df['accuracies'][0])
# Output: [1, 0, 1, 1]  ✅ Actual list!

# Access specific elements
df['first_choice'] = df['extracted_choices'].apply(lambda x: x[0])
```

### ❌ Wrong: Load CSV files for analysis

```python
# DON'T do this for code/analysis
df = pd.read_csv('human_mc_per_question.csv')

print(df['extracted_choices'][0])
# Output: "['A', 'B', 'C', 'A']"  ❌ String, not a list!
```

### Manual JSONL Loading

If you can't import the helper function:

```python
import json
import pandas as pd

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

df = load_jsonl('/path/to/human_mc_per_question.jsonl')
```

## Example: Working with MC Results

```python
from evaluation.score_humans import load_human_results

# Load MC results
df = load_human_results('/home/work/yuna/HPA/evaluation/scored/humans/human_mc_per_question.jsonl')

# Access nested data
for idx, row in df.iterrows():
    qid = row['qid']
    choices = row['extracted_choices']  # ✅ This is a list!
    accuracies = row['accuracies']      # ✅ This is a list!

    # Count correct responses
    correct_count = sum(accuracies)
    total_count = len(accuracies)

    print(f"Q{qid}: {correct_count}/{total_count} correct")

# Aggregate statistics
df['num_correct'] = df['accuracies'].apply(sum)
df['num_total'] = df['accuracies'].apply(len)
df['agreement_rate'] = df['num_correct'] / df['num_total']
```

## Example: Working with VQA Results

```python
from evaluation.score_humans import load_human_results

# Load VQA results
df = load_human_results('/home/work/yuna/HPA/evaluation/scored/humans/human_vqa_per_question.jsonl')

# Access nested data
for idx, row in df.iterrows():
    qid = row['qid']
    answers = row['answers']              # ✅ List of strings
    accuracies = row['accuracies']        # ✅ List of floats
    confidences = row['confidences']      # ✅ List of floats
    gt_answers = row['gt_answers']        # ✅ List of ground truth answers

    if 'visual_similarities' in row:
        similarities = row['visual_similarities']  # ✅ List of floats

    print(f"Q{qid}: Mean accuracy = {row['mean_accuracy']:.3f}")
```

## Changes in score_humans.py

### Before (CSV only):
```python
# Save MC results
choice_output = os.path.join(output_dir, 'human_mc_per_question.csv')
pd.DataFrame(choice_results).to_csv(choice_output)
```

### After (JSONL + CSV):
```python
# Save MC results (as JSONL to preserve nested structures)
choice_output_jsonl = os.path.join(output_dir, 'human_mc_per_question.jsonl')
with open(choice_output_jsonl, 'w', encoding='utf-8') as f:
    for result in choice_results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

# Also save as CSV for quick viewing (but nested structures will be strings)
choice_output_csv = os.path.join(output_dir, 'human_mc_per_question.csv')
pd.DataFrame(choice_results).to_csv(choice_output_csv, index=False)
```

## Benefits

1. **JSONL preserves data types**: Lists stay as lists, dicts stay as dicts
2. **CSV for viewing**: Open in Excel/spreadsheet to browse results
3. **Best of both worlds**: Code uses JSONL, humans view CSV
4. **No data loss**: All information is preserved in JSONL

## Quick Reference

| Format | Use For | Nested Structures |
|--------|---------|-------------------|
| `.jsonl` | Code/analysis | ✅ Preserved |
| `.csv` | Viewing in Excel | ❌ Converted to strings |

**Always use `.jsonl` for programmatic analysis!**
