#!/usr/bin/env python3
"""
2. prepare_training_data.py - Convert Preprocessed Data to Training Format

Creates JSONL files ready for training with:
- Black placeholder images
- Optional instruction prefix
- Confidence scores for soft supervision
- Train/val splits

Usage:
    python prepare_training_data.py \
        --processed_dir ./processed_data/mmstar \
        --questions_csv ./data/mmstar_questions.csv \
        --output_dir ./training_data/mmstar \
        --with_instruction \
        --create_split
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import sys 
sys.path.append('/home/work/yuna/HPA')
from dataset.vqav2 import VQADataset

def read_questions(dataset='s1', answer_type='text'): 
    questions_path = f"/home/work/yuna/HPA/experiments/questions/{dataset}.csv" 
    questions = []
    if answer_type == 'text': 
        vqa_val = VQADataset()
        vqa_val = vqa_val.get_by_qid() 

    if questions_path.endswith('.csv'):
        import csv
        with open(questions_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row.get('qid', row.get('question_id', '')))
                atype = row.get('answer_type', '') 
                
                if atype != answer_type: 
                    continue 

                if answer_type == 'choice': 
                    answer = row.get('answer', row.get('multiple_choice_answer', ''))
                    # annot = annotations[qid] 
                    # annot['answer'] = answer 
                    ### TODO Load the images for mmstar and mmspu and save into PIL image below 
                    # questions.append(annot) 

                else: # vqa questions 
                    questions.append(vqa_val[qid]) 
    else:
        with open(questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data if isinstance(data, list) else data.get('questions', [])
        for item in items:
            qid = str(item.get('qid', item.get('question_id', '')))
            questions.append({
                'qid': qid,
                'question': item.get('question', item.get('question_en', '')),
                'answer': item.get('answer', ''),
                'image_id': item.get('image_id', ''),
                'category': item.get('category', ''),
            })

    return questions 

def aggregate_responses(
    responses: List[Dict],
    by_cluster: bool = False,
) -> List[Dict]:
    """
    Aggregate individual responses into training examples.
    
    Args:
        responses: List of individual responses
        by_cluster: If True, aggregate by cluster_id; else by exact answer
        
    Returns:
        List of aggregated training examples
    """
    # Group responses
    grouped = defaultdict(list)
    
    for r in responses:
        qid = r['qid']
        
        if by_cluster and r.get('cluster_id') is not None:
            key = (qid, r['cluster_id'])
            answer = r.get('cluster_answer', r['answer_normalized'])
        else:
            key = (qid, r['answer_normalized'])
            answer = r['answer_normalized']
        
        grouped[key].append({
            'answer': answer,
            'confidence': r['confidence'],
            'time_spent': r.get('time_spent', 0),
            'question': r.get('question', ''),
            'category': r.get('category', ''),
        })
    
    # Aggregate each group
    aggregated = []
    
    for (qid, group_key), items in grouped.items():
        # Use most common answer in group (for display)
        from collections import Counter
        answers = [item['answer'] for item in items]
        answer = Counter(answers).most_common(1)[0][0]
        
        # Average confidence
        avg_confidence = np.mean([item['confidence'] for item in items])
        
        # Average time
        times = [item['time_spent'] for item in items if item['time_spent'] > 0]
        avg_time = np.mean(times) if times else 0
        
        aggregated.append({
            'qid': qid,
            'answer': answer,
            'confidence': round(avg_confidence, 2),
            'num_responses': len(items),
            'time_spent_seconds': round(avg_time, 2),
            'question': items[0]['question'],
            'category': items[0]['category'],
        })
    
    print(f"✓ Aggregated {len(responses)} responses into {len(aggregated)} training examples")
    return aggregated


# =============================================================================
# JSONL Creation
# =============================================================================

def create_training_jsonl(
    data: List[Dict],
    output_path: str,
    black_image_path: str,
    min_confidence: float = None,
) -> int:
    """
    Create training JSONL file with MC formatting.
    
    Format:
    {
        "images": ["black_image.png"],
        "conversations": [
            {"role": "user", "content": "<image>\n{instruction}{question}"},
            {"role": "assistant", "content": "{answer}"}
        ],
        "confidence": 4.2,
        "qid": "664",
        ...
    }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = []
    filtered_conf = 0
    filtered_resp = 0
    
    for item in data:
        # Apply filters
        if min_confidence and item['confidence'] < min_confidence:
            filtered_conf += 1
            continue 
        
        question = item.get('question', '')
        if not question:
            continue
        
        answer = item['answer']
        options = item.get('options', None)
        
        # Format question with options if available
        if options:
            if isinstance(options, str):
                try:
                    import ast
                    options = ast.literal_eval(options)
                except:
                    options = None
            
            if options and isinstance(options, list):
                choices_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                question = (
                    f"Question: {question}\n"
                    f"{choices_text}\n"
                    "Provide only the letter corresponding to the correct choice (A, B, C, or D).\n"
                    "Answer:"
                )
        
        example = {
            "images": [black_image_path],
            "conversations": [
                {
                    "role": "user",
                    "content": f"<image>\n{question}"
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ],
            "confidence": item['confidence'],
            "qid": item['qid'],
            "num_responses": item.get('num_responses', 1),
            "category": item.get('category', ''),
        }
        
        examples.append(example)
    
    # Write JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✓ Created {output_path} ({len(examples)} examples)")
    if filtered_conf > 0:
        print(f"  Filtered by confidence: {filtered_conf}")
    
    return len(examples)


def create_individual_jsonl(
    responses: List[Dict],
    output_path: str,
    black_image_path: str,
    instruction_prefix: str = "",
) -> int:
    """
    Create training JSONL from individual (non-aggregated) responses.
    
    Each response becomes one training example.
    Handles multiple choice formatting with options.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = []
    
    for r in responses:
        question = r.get('question', '')
        answer = r.get('answer_normalized', r.get('answer', ''))
        options = r.get('options', None)
        formatted_question = f"{question}\n{instruction_prefix}" 
        
        if not question or not answer:
            continue
        
        if options:
            if isinstance(options, str):
                try:
                    import ast
                    options = ast.literal_eval(options)
                except:
                    options = None
            
            if options and isinstance(options, list):
                choices_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                formatted_question = (
                    f"Question: {question}\n{instruction_prefix}\n"
                    f"{choices_text}\n"
                    "Provide only the letter corresponding to the correct choice (A, B, C, or D).\n"
                    "Answer:"
                )
        
        example = {
            "images": [black_image_path],
            "conversations": [
                {
                    "role": "user",
                    "content": f"<image>\n{formatted_question}"
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ],
            "confidence": r.get('confidence', 3),
            "qid": r['qid'],
            "category": r.get('category', ''),
        }
        
        examples.append(example)
    
    # Shuffle
    np.random.shuffle(examples)
    
    # Write JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✓ Created {output_path} ({len(examples)} examples)")
    return len(examples)


# =============================================================================
# Train/Val Split
# =============================================================================

def create_train_val_split(
    jsonl_path: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Split JSONL file into train and validation sets.
    
    Splits by question ID to avoid data leakage.
    """
    np.random.seed(seed)
    
    # Load examples
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Group by qid
    by_qid = defaultdict(list)
    for ex in examples:
        by_qid[ex['qid']].append(ex)
    
    # Split qids
    qids = list(by_qid.keys())
    np.random.shuffle(qids)
    
    split_idx = int(len(qids) * train_ratio)
    train_qids = set(qids[:split_idx])
    val_qids = set(qids[split_idx:])
    
    # Assign examples
    train_examples = [ex for ex in examples if ex['qid'] in train_qids]
    val_examples = [ex for ex in examples if ex['qid'] in val_qids]
    
    # Save
    base_path = Path(jsonl_path)
    train_path = base_path.parent / f"{base_path.stem}_train.jsonl"
    val_path = base_path.parent / f"{base_path.stem}_val.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✓ Train: {train_path} ({len(train_examples)} examples, {len(train_qids)} questions)")
    print(f"✓ Val: {val_path} ({len(val_examples)} examples, {len(val_qids)} questions)")
    
    return str(train_path), str(val_path)


# =============================================================================
# Ground Truth Data Preparation (for ablations A1, A2)
# =============================================================================

def create_gt_training_data(
    questions: str,
    output_path: str,
    images_dir: str = None,
    use_real_images: bool = False,
    instruction_prefix: str = "",
) -> int:
    
    # Create training examples
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = []
    
    for q in questions:
        if not q['question'] or not q['answer']:
            continue
        
        if use_real_images and images_dir and q['image_id']:
            image_path = f"{images_dir}/{q['image_id']}.jpg"
            if not os.path.exists(image_path):
                image_path = black_image_path
        else:
            image_path = black_image_path
        
        example = {
            "images": [image_path],
            "conversations": [
                {
                    "role": "user",
                    "content": f"<image>\n{instruction_prefix}{q['question']}"
                },
                {
                    "role": "assistant",
                    "content": q['answer']
                }
            ],
            "confidence": 5.0,  # GT has max confidence
            "qid": q['qid'],
            "category": q.get('category', ''),
            "is_gt": True,
        }
        
        examples.append(example)
    
    # Write
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✓ Created GT training data: {output_path} ({len(examples)} examples)")
    return len(examples)


# =============================================================================
# Main Pipeline
# =============================================================================

def prepare_training_data(
    processed_dir: str,
    questions_path: str,
    output_dir: str,
    # by_cluster: bool = False,
    # create_split: bool = True,
    train_ratio: float = 0.9,
    min_confidence: float = None,
    # min_responses: int = 1,
    images_dir: str = None,
):
    """
    Main function to prepare training data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📦 PREPARING TRAINING DATA")
    print("=" * 60)

    black_image_path = "/home/work/yuna/HPA/data/blank_224.png"
    blind_instruction = "Note: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n"
    
    # Load preprocessed responses (handle both JSON and JSONL)
    responses_path = processed_dir
    if os.path.isdir(processed_dir):
        responses_path = os.path.join(processed_dir, 'individual_responses.json')
    
    if not os.path.exists(responses_path):
        print(f"❌ Not found: {responses_path}")
        print("   Run preprocess_answers.py first!")
        return
    
    # Load based on file extension
    responses = []
    if responses_path.endswith('.jsonl'):
        with open(responses_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    responses.append(json.loads(line))
    else:
        with open(responses_path, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    
    print(f"✓ Loaded {len(responses)} preprocessed responses")
    
    # Create training data Aggregated version 
    # if aggregate:
    # aggregated = aggregate_responses(responses, by_cluster=False) # TODO: no clustering answers now 
    # jsonl_path = os.path.join(output_dir, 'train_aggregated.jsonl')
    
    # Also create individual (shuffled) version
    individual_path = os.path.join(output_dir, 'train_individual.jsonl')
    create_individual_jsonl(
        responses,
        individual_path,
        black_image_path,
    )
    # create_train_val_split(individual_path, train_ratio)

    print("\n📋 Creating GT training data...") 
    # A2: GT + black images
    gt_blind_path = os.path.join(output_dir, 'train_gt_blind.jsonl')
    create_gt_training_data(
        questions_path,
        gt_blind_path,
        use_real_images=False,
    )
    
    # A1: GT + real images (if images_dir provided)
    # if images_dir:
    gt_real_path = os.path.join(output_dir, 'train_gt_images.jsonl')
    create_gt_training_data(
        questions_path,
        gt_real_path,
        images_dir=images_dir,
        use_real_images=True,
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 OUTPUT FILES")
    print("=" * 60)
    for f in sorted(Path(output_dir).glob('*.jsonl')):
        with open(f, 'r') as file:
            count = sum(1 for _ in file)
        print(f"   {f.name}: {count} examples")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from preprocessed responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        
Examples:
  # Basic usage (aggregated + split)
  python prepare_training_data.py \\
      --processed_dir ./processed_data/mmstar \\
      --questions_csv ./data/mmstar_questions.csv \\
      --output_dir ./training_data/mmstar \\
      --with_instruction --create_split

  # With clustering and GT data
  python prepare_training_data.py \\
      --processed_dir ./processed_data/mmstar \\
      --questions_csv ./data/mmstar_questions.csv \\
      --output_dir ./training_data/mmstar \\
      --with_instruction --by_cluster --also_create_gt
        """
    )
    
    parser.add_argument("--processed_dir", type=str, required=True,
                        help="Directory with preprocessed data (from preprocess_answers.py)")
    parser.add_argument("--questions_csv", type=str, required=True,
                        help="Questions file (for GT data)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for training files")
    
    parser.add_argument("--with_instruction", action="store_true",
                        help="Add blind instruction prefix")
    parser.add_argument("--no_aggregate", action="store_true",
                        help="Don't aggregate (use individual responses)")
    parser.add_argument("--by_cluster", action="store_true",
                        help="Aggregate by cluster (requires clustering in preprocessing)")
    
    parser.add_argument("--create_split", action="store_true",
                        help="Create train/val split")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Train set ratio")
    
    parser.add_argument("--min_confidence", type=float, default=None,
                        help="Filter examples below this confidence")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Directory with real images (for A1)")
    
    args = parser.parse_args()
    
    prepare_training_data(
        processed_dir=args.processed_dir,
        questions_path=args.questions_csv,
        output_dir=args.output_dir,
        with_instruction=args.with_instruction,
        aggregate=not args.no_aggregate,
        by_cluster=args.by_cluster,
        create_split=args.create_split,
        train_ratio=args.train_ratio,
        min_confidence=args.min_confidence,
        images_dir=args.images_dir,
    )


if __name__ == "__main__":
    main()