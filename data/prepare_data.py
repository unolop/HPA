#!/usr/bin/env python3
"""
Usage:
    python prepare_blind_vqa_data.py \
        --csv_files p1.csv p2.csv p3.csv \
        --questions_file questions.json \
        --output_path train.jsonl \
        --aggregate
"""

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

def load_questions_from_csv(csv_path: str) -> Dict[str, dict]:
    """Load questions from your CSV format."""
    questions = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row['qid'])
            questions[qid] = {
                'question': row.get('question_en', row.get('question', '')),
                'category': row.get('category', ''),
                'l2_category': row.get('l2_category', ''),
                'answer': row.get('answer', row.get('multiple_choice_answer', '')),
                'options': row.get('options', ''),
                'answer_type': row.get('answer_type', ''),
                'image_id': row.get('image_id', ''),
            }
    
    return questions

def load_annotations(annotations_path: str, dataset_type: str = "vqav2") -> Dict[str, dict]:
    """
    Load answer annotations from VQA dataset file.
    
    Args:
        annotations_path: Path to annotations JSON file
        dataset_type: Dataset type
        
    Returns:
        Dictionary mapping qid -> {answers: [...], answer_type: ...}
    """
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    annotations = {}
    
    if dataset_type in ["vqav2", "okvqa"]:
        # VQAv2 format: {"annotations": [{"question_id": 123, "answers": [{"answer": "yes"}, ...]}]}
        for ann in data.get("annotations", data):
            qid = str(ann["question_id"])
            annotations[qid] = {
                "answers": [a["answer"] for a in ann.get("answers", [])],
                "answer_type": ann.get("answer_type"),
                "question_type": ann.get("question_type"),
            }
    
    elif dataset_type == "gqa":
        # GQA has answers in questions file
        pass
    
    print(f"Loaded {len(annotations)} annotations")
    return annotations


def create_black_image(output_path: str, size: Tuple[int, int] = (448, 448)) -> str:
    """Create black placeholder image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    black_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img = Image.fromarray(black_array)
    img.save(output_path)
    
    print(f"Created black image: {output_path}")
    return str(output_path)


def process_csv_files(
    csv_files,
    questions: Dict[str, dict],
    output_path: str,
    black_image_path: str = "black_image.png",
    instruction_prefix: str = "Note: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n",
    aggregate: bool = True,
    min_responses: int = 1,
) -> Tuple[str, dict]:
    """
    Process participant CSV files and create training JSONL.
    
    Args:
        csv_files: List of participant CSV file paths
        questions: Question mapping from load_questions()
        output_path: Output JSONL path
        black_image_path: Path to black placeholder image
        instruction_prefix: Instruction to prepend to questions
        aggregate: Whether to aggregate responses across participants
        min_responses: Minimum responses needed to include a question
        
    Returns:
        Tuple of (output_path, statistics_dict)
    """
    # Collect all responses
    responses = defaultdict(list)
    
    for csv_file in glob(csv_files): 
        print(f"Processing: {csv_file}")
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row['qid'])
                
                if qid not in questions:
                    continue
                
                responses[qid].append({
                    'answer': row['answer'],
                    'confidence': int(row['confidence']),
                    'time_spent': float(row['time_spent_seconds']),
                    'source_file': csv_file,
                })
    
    # Prepare output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_questions': 0,
        'total_examples': 0,
        'confidence_distribution': defaultdict(int),
        'avg_responses_per_question': 0,
        'skipped_low_response': 0,
    }
    
    examples = []
    
    if aggregate:
        # Group by (qid, answer) and average confidence
        for qid, resp_list in responses.items():
            if len(resp_list) < min_responses:
                stats['skipped_low_response'] += 1
                continue
            
            question_data = questions[qid]
            question_text = question_data['question']
            
            # Group by answer
            answer_groups = defaultdict(list)
            for r in resp_list:
                answer_groups[r['answer']].append(r)
            
            for answer, group in answer_groups.items():
                avg_confidence = sum(r['confidence'] for r in group) / len(group)
                avg_time = sum(r['time_spent'] for r in group) / len(group)
                
                # Round confidence to nearest integer for binning
                confidence_bin = round(avg_confidence)
                stats['confidence_distribution'][confidence_bin] += 1
                
                example = {
                    "images": [black_image_path],
                    "conversations": [
                        {
                            "role": "user",
                            "content": f"<image>\n{instruction_prefix}{question_text}"
                        },
                        {
                            "role": "assistant", 
                            "content": str(answer)
                        }
                    ],
                    "confidence": round(avg_confidence, 2),
                    "num_responses": len(group),
                    "qid": qid,
                    "time_spent_seconds": round(avg_time, 2),
                }
                examples.append(example)
            
            stats['total_questions'] += 1
    else:
        # Keep all individual responses
        for qid, resp_list in responses.items():
            if len(resp_list) < min_responses:
                stats['skipped_low_response'] += 1
                continue
            
            question_data = questions[qid]
            question_text = question_data['question']
            
            for r in resp_list:
                stats['confidence_distribution'][r['confidence']] += 1
                
                example = {
                    "images": [black_image_path],
                    "conversations": [
                        {
                            "role": "user",
                            "content": f"<image>\n{instruction_prefix}{question_text}"
                        },
                        {
                            "role": "assistant",
                            "content": str(r['answer'])
                        }
                    ],
                    "confidence": r['confidence'],
                    "qid": qid,
                    "time_spent_seconds": r['time_spent'],
                }
                examples.append(example)
            
            stats['total_questions'] += 1
    
    # Write examples
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    stats['total_examples'] = len(examples)
    stats['avg_responses_per_question'] = len(examples) / max(stats['total_questions'], 1)
    stats['confidence_distribution'] = dict(stats['confidence_distribution'])
    
    print(f"\n📊 Statistics:")
    print(f"   Total questions: {stats['total_questions']}")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Avg responses/question: {stats['avg_responses_per_question']:.2f}")
    print(f"   Confidence distribution: {stats['confidence_distribution']}")
    print(f"   Skipped (low response): {stats['skipped_low_response']}")
    print(f"\n✅ Saved to: {output_path}")
    
    return str(output_path), stats


def create_train_val_split(
    jsonl_path: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Split JSONL file into train and validation sets.
    
    Args:
        jsonl_path: Path to JSONL file
        train_ratio: Ratio for training set
        seed: Random seed
        
    Returns:
        Tuple of (train_path, val_path)
    """
    np.random.seed(seed)
    
    # Load examples
    examples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Shuffle
    np.random.shuffle(examples)
    
    # Split
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # Save
    base_path = Path(jsonl_path)
    train_path = base_path.parent / f"{base_path.stem}_train.jsonl"
    val_path = base_path.parent / f"{base_path.stem}_val.jsonl"
    
    with open(train_path, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    
    with open(val_path, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Created train set: {train_path} ({len(train_examples)} examples)")
    print(f"Created val set: {val_path} ({len(val_examples)} examples)")
    
    return str(train_path), str(val_path)


def analyze_confidence_statistics(jsonl_path: str) -> dict:
    """
    Analyze confidence distribution and other statistics.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        Statistics dictionary
    """
    confidences = []
    times = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            confidences.append(example['confidence'])
            if 'time_spent_seconds' in example:
                times.append(example['time_spent_seconds'])
    
    stats = {
        'count': len(confidences),
        'confidence_mean': np.mean(confidences),
        'confidence_std': np.std(confidences),
        'confidence_min': np.min(confidences),
        'confidence_max': np.max(confidences),
        'confidence_median': np.median(confidences),
    }
    
    if times:
        stats.update({
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_median': np.median(times),
        })
    
    # Confidence histogram
    hist, bins = np.histogram(confidences, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    stats['confidence_histogram'] = {
        int(bins[i]): int(hist[i]) for i in range(len(hist))
    }
    
    print("\n📈 Confidence Analysis:")
    print(f"   Mean: {stats['confidence_mean']:.2f} ± {stats['confidence_std']:.2f}")
    print(f"   Median: {stats['confidence_median']:.2f}")
    print(f"   Range: [{stats['confidence_min']:.2f}, {stats['confidence_max']:.2f}]")
    print(f"   Distribution: {stats['confidence_histogram']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare blind VQA data for soft supervised learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--csv_files", type=str, default="/home/work/yuna/HPA/results/humans/all_results_20251202_112501/*/*.csv",
                        help="Participant CSV files")
    parser.add_argument("--questions_file", type=str, required=True,
                        help="Questions JSON file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSONL path")
    
    parser.add_argument("--dataset_type", type=str, default="vqav2",
                        choices=["vqav2", "gqa", "okvqa", "custom"],
                        help="VQA dataset type")
    parser.add_argument("--black_image_path", type=str, default="black_image.png",
                        help="Path for black placeholder image")
    parser.add_argument("--instruction_prefix", type=str,
                        default="Note: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n",
                        help="Instruction prefix for questions")
    
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate responses across participants")
    parser.add_argument("--min_responses", type=int, default=1,
                        help="Minimum responses to include question")
    
    parser.add_argument("--create_split", action="store_true",
                        help="Create train/val split")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Train split ratio")
    
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze output statistics")
    
    args = parser.parse_args()
    
    # Create black image
    create_black_image(args.black_image_path)
    
    # Load questions
    questions = load_questions_from_csv(args.questions_file) # , args.dataset_type 
    
    # Process CSV files
    output_path, stats = process_csv_files(
        csv_files=args.csv_files,
        questions=questions,
        output_path=args.output_path,
        black_image_path=args.black_image_path,
        instruction_prefix=args.instruction_prefix,
        aggregate=args.aggregate,
        min_responses=args.min_responses,
    )
    
    # Create train/val split if requested
    if args.create_split:
        create_train_val_split(output_path, args.train_ratio)
    
    # Analyze if requested
    if args.analyze:
        analyze_confidence_statistics(output_path)
    
    # Save stats
    stats_path = Path(args.output_path).parent / "data_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")


if __name__ == "__main__":
    main()