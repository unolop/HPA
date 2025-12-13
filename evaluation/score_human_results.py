#!/usr/bin/env python3
"""
score_human_results.py - Score and analyze human responses for VQA and MC tasks

Processes human responses from training data files, computes:
- VQA accuracy and embedding similarity for text answers
- MC accuracy for choice answers
- Confidence distributions and statistics
- Saves QID mappings by answer_type for comparison with models 
"""

import os 
import json
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict 
from tqdm import tqdm
from utils import * 

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


# Global mapper instance (lazy loaded)
_vqa_mapper = None

def get_vqa_mapper() -> VQAAnswerMapper:
    """Get or create global VQA mapper."""
    global _vqa_mapper
    if _vqa_mapper is None:
        _vqa_mapper = VQAAnswerMapper()
    return _vqa_mapper


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
