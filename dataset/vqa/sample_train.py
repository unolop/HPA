"""
Sample 1000 questions from VQA v2 training set, stratified to match
the answer_type × question_type distribution of the validation 1K subset.

Outputs:
  dataset/vqa/vqa1k_train.json       — sampled training questions (same format as vqav2_1k_val.json)
  dataset/vqa/vqa1k_train.jsonl      — JSONL format for inference pipeline

Run:
  conda run -n aide python dataset/vqa/sample_train.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
N_SAMPLE = 1000

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/david/Desktop/yuna/data')

VAL_1K = ROOT / 'dataset/vqa/vqav2_1k_val.json'
TRAIN_Q = DATA / 'v2_OpenEnded_mscoco_train2014_questions.json'
TRAIN_A = ROOT / 'dataset/vqa/v2_mscoco_train2014_annotations.json'

OUT_JSON = ROOT / 'dataset/vqa/vqa1k_train.json'
OUT_JSONL = ROOT / 'dataset/vqa/vqa1k_train.jsonl'

# ── 1. Compute target distribution from val 1K ──────────────────────────────
val = json.load(open(VAL_1K))
# Stratify by (answer_type, question_type) joint distribution
val_strata = Counter((q['answer_type'], q['question_type']) for q in val)
print(f'Val 1K: {len(val)} questions, {len(val_strata)} strata')

# ── 2. Load training data ────────────────────────────────────────────────────
print('Loading training questions…')
train_q = json.load(open(TRAIN_Q))['questions']
q_by_id = {q['question_id']: q for q in train_q}

print('Loading training annotations…')
train_a = json.load(open(TRAIN_A))['annotations']
a_by_id = {a['question_id']: a for a in train_a}

# Merge questions + annotations
train_merged = []
for qid, q in q_by_id.items():
    a = a_by_id.get(qid)
    if a is None:
        continue
    train_merged.append({
        'image_id': q['image_id'],
        'question_id': qid,
        'question_type': a['question_type'],
        'question': q['question'],
        'answers': a['answers'],
        'multiple_choice_answer': a['multiple_choice_answer'],
        'answer_type': a['answer_type'],
    })
print(f'Training merged: {len(train_merged)} questions')

# ── 3. Group training questions by stratum ───────────────────────────────────
train_by_stratum = defaultdict(list)
for q in train_merged:
    key = (q['answer_type'], q['question_type'])
    train_by_stratum[key].append(q)

# ── 4. Stratified sampling ──────────────────────────────────────────────────
random.seed(SEED)
sampled = []
shortfall = 0

for stratum, n_needed in val_strata.most_common():
    pool = train_by_stratum.get(stratum, [])
    if len(pool) >= n_needed:
        sampled.extend(random.sample(pool, n_needed))
    else:
        sampled.extend(pool)
        shortfall += n_needed - len(pool)
        print(f'  Warning: stratum {stratum} has {len(pool)}/{n_needed} available')

# Fill shortfall from remaining questions (proportional to answer_type)
if shortfall > 0:
    used_ids = {q['question_id'] for q in sampled}
    remaining = [q for q in train_merged if q['question_id'] not in used_ids]
    random.shuffle(remaining)

    # Try to fill from same answer_type proportionally
    at_deficit = Counter()
    for stratum, n_needed in val_strata.items():
        pool = train_by_stratum.get(stratum, [])
        if len(pool) < n_needed:
            at_deficit[stratum[0]] += n_needed - len(pool)

    for q in remaining:
        if shortfall <= 0:
            break
        at = q['answer_type']
        if at_deficit.get(at, 0) > 0:
            sampled.append(q)
            at_deficit[at] -= 1
            shortfall -= 1

print(f'\nSampled: {len(sampled)} questions')

# ── 5. Verify distribution match ────────────────────────────────────────────
sampled_at = Counter(q['answer_type'] for q in sampled)
sampled_qt = Counter(q['question_type'] for q in sampled)

print(f'\nanswer_type comparison:')
for at in ['other', 'yes/no', 'number']:
    val_n = sum(v for (a, _), v in val_strata.items() if a == at)
    sam_n = sampled_at.get(at, 0)
    print(f'  {at:10s}  val={val_n:4d}  train_sample={sam_n:4d}')

print(f'\nquestion_type comparison (top 10):')
val_qt = Counter(q['question_type'] for q in val)
for qt, val_n in val_qt.most_common(10):
    sam_n = sampled_qt.get(qt, 0)
    print(f'  {qt:25s}  val={val_n:3d}  train_sample={sam_n:3d}')

# Unique images
n_images = len(set(q['image_id'] for q in sampled))
print(f'\nUnique images in sample: {n_images}')

# ── 6. Save ─────────────────────────────────────────────────────────────────
json.dump(sampled, open(OUT_JSON, 'w'), indent=2)
print(f'Saved: {OUT_JSON}')

with open(OUT_JSONL, 'w') as f:
    for q in sampled:
        f.write(json.dumps(q) + '\n')
print(f'Saved: {OUT_JSONL}')
