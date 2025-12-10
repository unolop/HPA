#!/usr/bin/env python3
"""
score_human_results.py - Score and analyze human responses for VQA and MC tasks

Processes human responses from training data files, computes:
- VQA accuracy and embedding similarity for text answers
- MC accuracy for choice answers
- Confidence distributions and statistics
- Saves QID mappings by answer_type for comparison with models

Usage:
    python evaluation/score_human_results.py --text_data data/training/s1_text/train_agg_10_blind_inst.jsonl \
                                   --choice_data data/training/s1_choice/train_agg_10_blind_inst.jsonl \
                                   --output_dir evaluation/human_scored/
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

VQA_ANNOTATIONS_PATH = "/home/work/yuna/VLMEval/data/v2_mscoco_val2014_annotations.json"

CONF_MAP = {
    'yes': 1.0,
    'maybe': 0.5,
    'no': 0.01,
    '1': 0.05,
    '2': 0.25,
    '3': 0.5,
    '4': 0.75,
    '5': 1.0
}


# =============================================================================
# VQA Annotation Loading
# =============================================================================

class VQAAnswerMapper:
    """Maps question_id to list of ground truth answers from VQA annotations."""

    def __init__(self, annotations_path: str = VQA_ANNOTATIONS_PATH):
        self.annotations_path = annotations_path
        self._qid_to_answers = None

    def _load(self):
        """Load annotations and build lookup dict."""
        if self._qid_to_answers is not None:
            return

        if not os.path.exists(self.annotations_path):
            print(f"⚠️  VQA annotations not found: {self.annotations_path}")
            self._qid_to_answers = {}
            return

        print(f"   Loading VQA annotations from {self.annotations_path}...")
        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        self._qid_to_answers = {}
        for ann in annotations['annotations']:
            qid = int(ann['question_id'])
            answers = [a['answer'] for a in ann['answers']]
            self._qid_to_answers[qid] = answers

        print(f"   ✓ Loaded {len(self._qid_to_answers)} VQA annotations")

    def get_answers(self, question_id: int) -> List[str]:
        """Get list of 10 annotator answers for a question."""
        self._load()
        qid = int(question_id)
        return self._qid_to_answers.get(qid, [])


_vqa_mapper = None

def get_vqa_mapper() -> VQAAnswerMapper:
    """Get or create global VQA mapper."""
    global _vqa_mapper
    if _vqa_mapper is None:
        _vqa_mapper = VQAAnswerMapper()
    return _vqa_mapper


# =============================================================================
# Answer Normalization and Scoring
# =============================================================================

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (open-ended)."""
    if not answer:
        return ""
    answer = str(answer)

    # Remove <think>...</think> content if present
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL | re.IGNORECASE)

    answer = answer.lower().strip()
    # Remove articles
    for article in ['a ', 'an ', 'the ']:
        if answer.startswith(article):
            answer = answer[len(article):]
    # Remove punctuation
    import string
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(answer.split()).strip()


def vqa_accuracy(gt_answers: List[str], pred: str) -> float:
    """VQA accuracy: min(1, #matches / 3)."""
    pred = normalize_answer(pred)
    matches = sum([pred == normalize_answer(ans) for ans in gt_answers])
    return min(1.0, matches / 3.0)


def extract_mc_choice(output: str) -> str:
    """Extract the predicted answer (A, B, C, D) from model output."""
    if not output:
        return ""

    # Remove <think>...</think> content if present
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL | re.IGNORECASE)
    output = output.strip()

    # Pattern 1: Look for explicit answer statements
    patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is[:\s]*([A-D])",
        r"(?:the\s+)?(?:correct\s+)?answer[:\s]*([A-D])",
        r"(?:option\s+)?([A-D])\s+is\s+(?:the\s+)?correct",
        r"(?:I\s+)?(?:would\s+)?choose\s+(?:option\s+)?([A-D])",
        r"(?:I\s+)?(?:would\s+)?select\s+(?:option\s+)?([A-D])",
        r"^([A-D])(?:[:\.\)]|\s|$)",  # Answer at the start
        r"\n([A-D])(?:[:\.\)]|\s|$)",  # Answer after newline
        r"(?:Therefore|Thus|So|Hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(?:option\s+)?([A-D])",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Pattern 2: Last capital letter A-D
    matches = re.findall(r'\b([A-D])\b', output)
    if matches:
        return matches[-1].upper()

    # Pattern 3: Look for choice at end
    if output and output[-1].upper() in 'ABCD':
        return output[-1].upper()

    # Pattern 4: Check format like "A: Hanging Posters"
    match = re.match(r'^([A-D]):', output)
    if match:
        return match.group(1).upper()

    return ""


def get_encoder():
    """Lazy load sentence transformer."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2").to('cuda')
    except:
        return None


def answer_similarity(gt: str, pred: str, encoder) -> float:
    """Compute embedding similarity."""
    if encoder is None:
        return 0.0
    try:
        pred = pred.strip().lower()
        gt = str(gt).strip().lower()
        emb = encoder.encode([pred, gt])
        similarities = encoder.similarity(emb, emb)
        return float(similarities[1, 0])
    except:
        return 0.0


# =============================================================================
# Human Response Processing
# =============================================================================

def process_human_responses(
    data_path: str,
    answer_type: str,  # 'text' or 'choice'
    with_similarity: bool = False,
) -> Dict:
    """
    Process human responses from training data file.

    Args:
        data_path: Path to training JSONL file (e.g., train_agg_10_blind_inst.jsonl)
        answer_type: 'text' for VQA or 'choice' for MC
        with_similarity: Compute embedding similarity for VQA

    Returns:
        Dict with processed results, statistics, and QID mappings
    """
    print(f"\n{'='*60}")
    print(f"📊 Processing Human Responses: {answer_type.upper()}")
    print(f"   File: {os.path.basename(data_path)}")
    print(f"{'='*60}")

    # Load data
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"   Loaded {len(data)} questions")

    # Load VQA mapper if needed
    vqa_mapper = None
    if answer_type == 'text':
        vqa_mapper = get_vqa_mapper()

    # Load encoder if similarity requested
    encoder = None
    if with_similarity and answer_type == 'text':
        print("   Loading sentence transformer for similarity computation...")
        encoder = get_encoder()
        if encoder:
            print("   ✓ Encoder loaded")
        else:
            print("   ⚠️ Failed to load encoder, skipping similarity computation")

    # Process each question
    results = []
    qids = []

    # Statistics
    stats = {
        'total_questions': len(data),
        'total_responses': 0,
        'confidence_dist': defaultdict(int),
        'answer_dist': defaultdict(int),
        'question_type_dist': defaultdict(int),
        'category_dist': defaultdict(int),
    }

    for item in tqdm(data, desc=f"Processing {answer_type} responses"):
        qid = str(item['qid'])
        qids.append(qid)

        # Get human responses from labels
        confidences = item['labels']['confidences']
        answers = item['labels']['answers']

        # Get the "consensus" answer (highest confidence)
        consensus_answer = answers[0] if answers else ""
        consensus_confidence = confidences[0] if confidences else 0.0

        stats['total_responses'] += len(answers)

        # Compute accuracy based on answer type
        if answer_type == 'text':
            # VQA: Compare against ground truth annotators
            gt_answers = vqa_mapper.get_answers(int(qid))

            if gt_answers:
                # Accuracy of consensus answer
                acc = vqa_accuracy(gt_answers, consensus_answer)

                # Compute similarity if requested
                sim = 0.0
                if encoder and gt_answers:
                    # Use majority GT answer
                    majority_gt = max(set(gt_answers), key=gt_answers.count)
                    sim = answer_similarity(majority_gt, consensus_answer, encoder)

                result = {
                    'qid': qid,
                    'answer': consensus_answer,
                    'confidence': consensus_confidence,
                    'accuracy': float(acc),
                    'correct': acc >= 0.3,  # VQA threshold
                    'gt_answers': gt_answers,
                    'all_human_answers': answers,
                    'all_confidences': confidences,
                }

                if encoder:
                    result['answer_similarity'] = float(sim)

                # Extract question type from conversation if available
                if 'conversations' in item and len(item['conversations']) > 0:
                    question = item['conversations'][0]['content']
                    # Try to extract question type
                    for qt in ['what', 'where', 'when', 'who', 'how', 'why', 'which', 'is', 'are', 'does', 'do']:
                        if question.lower().startswith(qt):
                            result['question_type'] = qt
                            stats['question_type_dist'][qt] += 1
                            break
                    else:
                        result['question_type'] = 'other'
                        stats['question_type_dist']['other'] += 1

                results.append(result)

        else:  # choice
            # MC: Extract letter and compare with ground truth
            # For human responses, we need to get GT from somewhere
            # The consensus answer should already be the letter (A/B/C/D)

            # For now, we'll just store the response
            # Actual GT comparison would require loading the original MMStar dataset
            result = {
                'qid': qid,
                'answer': consensus_answer,
                'confidence': consensus_confidence,
                'all_human_answers': answers,
                'all_confidences': confidences,
            }

            results.append(result)

        # Update confidence distribution
        for conf in confidences:
            conf_bucket = round(conf, 1)
            stats['confidence_dist'][conf_bucket] += 1

        # Update answer distribution
        for ans in answers:
            stats['answer_dist'][ans] += 1

    # Compute aggregate statistics
    if answer_type == 'text' and results:
        accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
        stats['mean_accuracy'] = float(np.mean(accuracies)) if accuracies else 0.0
        stats['std_accuracy'] = float(np.std(accuracies)) if accuracies else 0.0
        stats['correct_count'] = sum(1 for r in results if r.get('correct', False))

        if encoder:
            similarities = [r['answer_similarity'] for r in results if 'answer_similarity' in r]
            stats['mean_similarity'] = float(np.mean(similarities)) if similarities else 0.0
            stats['std_similarity'] = float(np.std(similarities)) if similarities else 0.0

    # Convert defaultdicts to regular dicts for JSON serialization
    stats['confidence_dist'] = dict(stats['confidence_dist'])
    stats['answer_dist'] = dict(sorted(stats['answer_dist'].items(), key=lambda x: -x[1])[:20])  # Top 20
    stats['question_type_dist'] = dict(stats['question_type_dist'])
    stats['category_dist'] = dict(stats['category_dist'])

    print(f"\n📈 Summary Statistics:")
    print(f"   Questions: {stats['total_questions']}")
    print(f"   Total responses: {stats['total_responses']}")
    if answer_type == 'text':
        print(f"   Mean accuracy: {stats.get('mean_accuracy', 0):.4f} ± {stats.get('std_accuracy', 0):.4f}")
        print(f"   Correct: {stats.get('correct_count', 0)}/{len(results)}")
        if encoder:
            print(f"   Mean similarity: {stats.get('mean_similarity', 0):.4f} ± {stats.get('std_similarity', 0):.4f}")

    return {
        'results': results,
        'qids': qids,
        'statistics': stats,
        'answer_type': answer_type,
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score and analyze human responses for VQA and MC tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process VQA (text) responses
  python score_human_results.py --text_data data/training/s1_text/train_agg_10_blind_inst.jsonl --output_dir evaluation/human_scored/

  # Process MC (choice) responses
  python score_human_results.py --choice_data data/training/s1_choice/train_agg_10_blind_inst.jsonl --output_dir evaluation/human_scored/

  # Process both with similarity computation
  python score_human_results.py --text_data data/training/s1_text/train_agg_10_blind_inst.jsonl \
                                 --choice_data data/training/s1_choice/train_agg_10_blind_inst.jsonl \
                                 --output_dir evaluation/human_scored/ \
                                 --with_similarity
        """
    )

    parser.add_argument("--text_data", type=str, default=None,
                        help="Path to VQA (text) human responses JSONL")
    parser.add_argument("--choice_data", type=str, default=None,
                        help="Path to MC (choice) human responses JSONL")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for scored results and QID mappings")
    parser.add_argument("--with_similarity", action="store_true",
                        help="Compute answer similarity for VQA (requires sentence-transformers)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    # Process text (VQA) responses
    if args.text_data:
        text_results = process_human_responses(
            args.text_data,
            answer_type='text',
            with_similarity=args.with_similarity
        )

        # Save results
        output_path = os.path.join(args.output_dir, 'human_vqa_scored.jsonl')
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in text_results['results']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"   ✓ Saved scored results: {output_path}")

        # Save QIDs
        qids_path = os.path.join(args.output_dir, 'human_vqa_qids.json')
        with open(qids_path, 'w', encoding='utf-8') as f:
            json.dump({'qids': text_results['qids'], 'count': len(text_results['qids'])}, f, indent=2)
        print(f"   ✓ Saved QIDs: {qids_path}")

        # Save statistics
        stats_path = os.path.join(args.output_dir, 'human_vqa_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(text_results['statistics'], f, indent=2, ensure_ascii=False)
        print(f"   ✓ Saved statistics: {stats_path}")

        all_results['vqa'] = text_results

    # Process choice (MC) responses
    if args.choice_data:
        choice_results = process_human_responses(
            args.choice_data,
            answer_type='choice',
            with_similarity=False  # No similarity for MC
        )

        # Save results
        output_path = os.path.join(args.output_dir, 'human_mc_scored.jsonl')
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in choice_results['results']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"   ✓ Saved scored results: {output_path}")

        # Save QIDs
        qids_path = os.path.join(args.output_dir, 'human_mc_qids.json')
        with open(qids_path, 'w', encoding='utf-8') as f:
            json.dump({'qids': choice_results['qids'], 'count': len(choice_results['qids'])}, f, indent=2)
        print(f"   ✓ Saved QIDs: {qids_path}")

        # Save statistics
        stats_path = os.path.join(args.output_dir, 'human_mc_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(choice_results['statistics'], f, indent=2, ensure_ascii=False)
        print(f"   ✓ Saved statistics: {stats_path}")

        all_results['mc'] = choice_results

    print(f"\n{'='*60}")
    print(f"✅ Human response scoring complete!")
    print(f"   Results saved to: {args.output_dir}")
    print(f"{'='*60}")

    return all_results


if __name__ == '__main__':
    main()
