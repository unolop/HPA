#!/usr/bin/env python3
"""
Create ground truth training data from VQA v2 annotations.

Processes VQA ground truth annotations through the same aggregation pipeline
used for human study data (semantic clustering, confidence normalization).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

sys.path.append('..')
from dataset.vqav2 import VQADataset
from aggregate import AnswerAggregator


def vqa_annotations_to_responses(annotations: List[Dict], qid: int) -> List[Dict]:
    """
    Convert VQA v2 annotations to response format.

    VQA v2 has 10 annotators per question. Each answer appears with a frequency.
    We convert this to individual responses with confidences based on agreement.

    Args:
        annotations: List of 10 VQA answer dicts, each with 'answer' field
        qid: Question ID

    Returns:
        List of response dicts with answer and confidence
    """
    # Count answer frequencies
    answer_counts = {}
    for ann in annotations:
        answer = ann['answer'].strip().lower()
        answer_counts[answer] = answer_counts.get(answer, 0) + 1

    # Create responses with confidence based on agreement
    # VQA scoring: answer is correct if >=3 annotators agree
    # We map frequency to confidence:
    # 10/10 = 1.0, 9/10 = 0.95, ..., 3/10 = 0.5, 2/10 = 0.25, 1/10 = 0.1
    responses = []
    for answer, count in answer_counts.items():
        # Map count to confidence
        if count >= 3:
            # Majority answers: linear scale from 0.5 to 1.0
            confidence = 0.3 + (count / 10.0) * 0.7  # 3->0.51, 10->1.0
        else:
            # Minority answers: lower confidence
            confidence = count / 10.0  # 1->0.1, 2->0.2

        # Create response for each occurrence (to maintain vote weight)
        for _ in range(count):
            responses.append({
                'answer': answer,
                'confidence': confidence,
                'qid': qid,
            })

    return responses


def build_gt_training_data(
    dataset_name: str,
    output_path: str,
    subset_size: int = None,
    answer_type: str = 'text',
    blind: bool = True,
    use_clustering: bool = True,
    openai_model: str = 'gpt-4o-mini',
):
    """
    Build ground truth training data from VQA annotations.

    Args:
        dataset_name: 'vqa1k', 'vqa5k', etc (uses VQADataset)
        output_path: Output JSONL file path
        subset_size: Number of questions to sample (None = all)
        answer_type: 'text' or 'choice'
        blind: Use blank image
        use_clustering: Whether to use semantic clustering
        openai_model: Model for clustering
    """

    print("=" * 80)
    print("🏗️  Building Ground Truth Training Data")
    print("=" * 80)
    print(f"   Dataset: {dataset_name}")
    print(f"   Output: {output_path}")
    print(f"   Answer type: {answer_type}")
    print(f"   Semantic clustering: {use_clustering}")
    print("=" * 80)

    # Load VQA dataset
    print("\n[1/4] Loading VQA dataset...")
    if dataset_name == 'vqa1k':
        ds = VQADataset()
        import numpy as np
        import torch
        from torch.utils.data import Subset

        size = subset_size or 1000
        indices = np.random.choice(len(ds), size=min(size, len(ds)), replace=False)
        ds = Subset(ds, indices)
    elif dataset_name == 'vqa5k':
        ds = VQADataset()
        import numpy as np
        import torch
        from torch.utils.data import Subset

        size = subset_size or 5000
        indices = np.random.choice(len(ds), size=min(size, len(ds)), replace=False)
        ds = Subset(ds, indices)
    else:
        # Load from JSON
        from dataset.vqav2 import VQADataset_json
        ds = VQADataset_json(json_path=dataset_name)
        if subset_size:
            import numpy as np
            from torch.utils.data import Subset
            indices = np.random.choice(len(ds), size=min(subset_size, len(ds)), replace=False)
            ds = Subset(ds, indices)

    print(f"✓ Loaded {len(ds)} questions")

    # Setup aggregator
    print("\n[2/4] Setting up aggregator...")
    if use_clustering:
        from preprocess import setup_openai_client
        client = setup_openai_client()
        if client is None:
            print("⚠️  No OpenAI client, disabling clustering")
            use_clustering = False
    else:
        client = None

    aggregator = AnswerAggregator(
        use_clustering=use_clustering,
        client=client,
        model=openai_model if use_clustering else None
    )

    # Process each question
    print("\n[3/4] Processing annotations...")
    examples = []
    skipped = 0

    for item in tqdm(ds, desc="Processing"):
        # Get question and annotations
        if isinstance(item, dict):
            qid = item['question_id']
            question = item['question']
            annotations = item.get('answers', [])  # List of 10 answer dicts
            image_path = item.get('image_path', '/home/work/yuna/HPA/data/blank_224.png')
        else:
            # Dataset object
            qid = item['question_id'] if isinstance(item, dict) else getattr(item, 'question_id', None)
            question = item['question'] if isinstance(item, dict) else getattr(item, 'question', '')
            annotations = item.get('answers', []) if isinstance(item, dict) else getattr(item, 'answers', [])
            image_path = item.get('image_path') if isinstance(item, dict) else getattr(item, 'image_path', '/home/work/yuna/HPA/data/blank_224.png')

        if not annotations or not question:
            skipped += 1
            continue

        # Convert annotations to responses
        responses = vqa_annotations_to_responses(annotations, qid)

        if not responses:
            skipped += 1
            continue

        # Aggregate using your pipeline
        # Note: aggregator expects 'answer_normalized' and 'confidence' keys
        from preprocess import normalize_answer

        answers_with_conf = [
            {
                'answer_normalized': normalize_answer(r['answer']),
                'confidence': r['confidence']
            }
            for r in responses
        ]

        # Run aggregation (clustering + normalization)
        confidence_dist = aggregator.aggregate(answers_with_conf)

        # Sort by confidence (descending)
        sorted_items = sorted(confidence_dist.items(), key=lambda x: -x[1])
        unique_answers = [item[0] for item in sorted_items]
        unique_confidences = [item[1] for item in sorted_items]

        # Get highest confidence answer for conversation
        max_conf_idx = 0 if unique_confidences else 0
        top_answer = unique_answers[max_conf_idx] if unique_answers else ''

        # Use blank image if blind
        if blind:
            image_path = '/home/work/yuna/HPA/data/blank_224.png'
            question_text = f"<image>"
        else:
            question_text = f"<image>"

        # Create training example
        example = {
            'idx': len(examples),
            'images': [image_path],
            'qid': qid,
            'conversations': [
                {'role': 'user', 'content': question_text},
                {'role': 'assistant', 'content': top_answer}
            ],
            'labels': {
                'confidences': unique_confidences,
                'answers': unique_answers
            }
        }

        examples.append(example)

    print(f"✓ Processed {len(examples)} examples (skipped {skipped})")

    # Write output
    print("\n[4/4] Writing output...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"✓ Wrote {len(examples)} items to {output_path}")

    # Print sample
    if examples:
        print("\n" + "=" * 80)
        print("📋 Sample Example:")
        print("=" * 80)
        sample = examples[0]
        print(f"QID: {sample['qid']}")
        print(f"Question: {sample['conversations'][0]['content']}")
        print(f"Top answer: {sample['conversations'][1]['content']}")
        print(f"All answers: {sample['labels']['answers'][:5]}")
        print(f"Confidences: {[f'{c:.3f}' for c in sample['labels']['confidences'][:5]]}")
        print(f"Confidence sum: {sum(sample['labels']['confidences']):.3f}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Build ground truth training data from VQA annotations"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name or path: 'vqa1k', 'vqa5k', or path to JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Number of questions to sample (default: all)"
    )
    parser.add_argument(
        "--answer_type",
        type=str,
        default='text',
        choices=['text', 'choice'],
        help="Answer type"
    )
    parser.add_argument(
        "--no_clustering",
        action='store_true',
        help="Disable semantic clustering"
    )
    parser.add_argument(
        "--no_blind",
        action='store_true',
        help="Use real images instead of blank"
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default='gpt-4o-mini',
        help="OpenAI model for clustering"
    )

    args = parser.parse_args()

    build_gt_training_data(
        dataset_name=args.dataset,
        output_path=args.output,
        subset_size=args.subset_size,
        answer_type=args.answer_type,
        blind=not args.no_blind,
        use_clustering=not args.no_clustering,
        openai_model=args.openai_model,
    )


if __name__ == "__main__":
    main()
