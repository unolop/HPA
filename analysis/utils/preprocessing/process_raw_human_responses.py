#!/usr/bin/env python3
"""
process_raw_human_responses.py - Process raw human responses and compute per-question metrics

Loads raw CSV data, computes per-question average accuracy and embedding similarity
against ground truth. Supports both VQA and MC tasks.

Usage:
    python evaluation/process_raw_human_responses.py \
        --human_data_dir data/humans/all_results_20251206_154732 \
        --session s1 \
        --output_dir evaluation/human_analysis/
"""

import os
import pandas as pd 
import re
import csv
import json
import argparse
from glob import glob 
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add this directory so preprocess.py is importable directly
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from preprocess import normalize_answer, preprocess_pipeline, CONF_MAP
from dataset.paths import VQA_ANNOT as VQA_ANNOTATIONS_PATH


# =============================================================================
# VQA Annotation Loading
# =============================================================================

class VQAAnswerMapper:
    """Maps question_id to list of ground truth answers from VQA annotations."""

    def __init__(self, annotations_path: str = VQA_ANNOTATIONS_PATH):
        self.annotations_path = annotations_path
        self._qid_to_answers = None
        self._qid_to_gt_visual = None  # Store visual GT for VQA

    def _load(self):
        """Load annotations and build lookup dict."""
        if self._qid_to_answers is not None:
            return

        if not os.path.exists(self.annotations_path):
            print(f"⚠️  VQA annotations not found: {self.annotations_path}")
            self._qid_to_answers = {}
            self._qid_to_gt_visual = {}
            return

        print(f"   Loading VQA annotations from {self.annotations_path}...")
        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        self._qid_to_answers = {}
        self._qid_to_gt_visual = {}

        for ann in annotations['annotations']:
            qid = int(ann['question_id'])
            # All 10 annotator answers
            answers = [a['answer'] for a in ann['answers']]
            self._qid_to_answers[qid] = answers

            # Multiple choice answer (consensus from humans who saw image)
            if 'multiple_choice_answer' in ann:
                self._qid_to_gt_visual[qid] = ann['multiple_choice_answer']

        print(f"   ✓ Loaded {len(self._qid_to_answers)} VQA annotations")

    def get_answers(self, question_id: int) -> List[str]:
        """Get list of 10 annotator answers for a question."""
        self._load()
        qid = int(question_id)
        return self._qid_to_answers.get(qid, [])

    def get_visual_gt(self, question_id: int) -> str:
        """Get visual ground truth (multiple choice answer from humans who saw image)."""
        self._load()
        qid = int(question_id)
        return self._qid_to_gt_visual.get(qid, "")


_vqa_mapper = None

def get_vqa_mapper() -> VQAAnswerMapper:
    """Get or create global VQA mapper."""
    global _vqa_mapper
    if _vqa_mapper is None:
        _vqa_mapper = VQAAnswerMapper()
    return _vqa_mapper


# =============================================================================
# MMStar Data Loading
# =============================================================================

def load_mmstar_annotations(session: str = "s1") -> Dict:
    """Load MMStar annotations from questions CSV and original dataset."""
    # Load from original model results to get full annotations
    mmstar_path = "/home/work/yuna/HPA/evaluation/results/pretrained/InternVL3_5-1B_mmstar_blind.jsonl"

    with open(mmstar_path, 'r', encoding='utf-8') as f:
        mmstar_data = [json.loads(line) for line in f]

    # Load questions
    questions_path = f"/home/work/yuna/HPA/dataset/questions/{session}.csv"
    questions = []
    with open(questions_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    # Filter for choice questions
    questions = [row for row in questions if row.get('answer_type') == 'choice']

    # Map questions to annotations
    annot = {}
    for row in questions:
        qid = row['qid']
        question = row.get('question_en', row.get('question', ''))

        # Match with mmstar data
        for data in mmstar_data:
            if question.strip()[:50] in data['question']:
                annot[qid] = {**row, **data}
                break

    return annot


# =============================================================================
# Scoring Functions
# =============================================================================

def vqa_accuracy(gt_answers: List[str], pred: str) -> float:
    """VQA accuracy: min(1, #matches / 3)."""
    pred = normalize_answer(pred)
    matches = sum([pred == normalize_answer(ans) for ans in gt_answers])
    return min(1.0, matches / 3.0)


def extract_mc_choice(output: str) -> str:
    """Extract the predicted answer (A, B, C, D) from model output."""
    if not output:
        return ""

    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL | re.IGNORECASE)
    output = output.strip()

    patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is[:\s]*([A-D])",
        r"(?:the\s+)?(?:correct\s+)?answer[:\s]*([A-D])",
        r"(?:option\s+)?([A-D])\s+is\s+(?:the\s+)?correct",
        r"(?:I\s+)?(?:would\s+)?choose\s+(?:option\s+)?([A-D])",
        r"(?:I\s+)?(?:would\s+)?select\s+(?:option\s+)?([A-D])",
        r"^([A-D])(?:[:\.\)]|\s|$)",
        r"\n([A-D])(?:[:\.\)]|\s|$)",
        r"(?:Therefore|Thus|So|Hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(?:option\s+)?([A-D])",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    matches = re.findall(r'\b([A-D])\b', output)
    if matches:
        return matches[-1].upper()

    if output and output[-1].upper() in 'ABCD':
        return output[-1].upper()

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
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")  # CPU fallback
        except:
            return None


def compute_similarity(gt: str, pred: str, encoder) -> float:
    """Compute embedding similarity."""
    if encoder is None or not gt or not pred:
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
# Data Loading
# =============================================================================

def get_responses_by_qid(
    answers: List[Dict],
    answer_type: str,
    vqa_mapper: VQAAnswerMapper = None,
    mmstar_annot: Dict = None
) -> Tuple[Dict, Dict]:
    """
    Group responses by QID and attach ground truth.

    Returns:
        responses_by_qid: {qid: [responses]}
        gt_by_qid: {qid: ground_truth_info}
    """
    answers = [a for a in answers if a.get("answer_type") == answer_type]
    responses_by_qid = defaultdict(list)
    gt_by_qid = {}
    
    for resp in answers:
        qid = str(resp['qid'])

        # Initialize ground truth for this QID if not seen
        if qid not in gt_by_qid:
            if answer_type == 'text' and vqa_mapper:
                gt_answers = vqa_mapper.get_answers(int(qid))
                visual_gt = vqa_mapper.get_visual_gt(int(qid))
                gt_by_qid[qid] = {
                    'gt_answers': gt_answers,
                    'visual_gt': visual_gt,
                    'answer_type': 'text'
                }
            elif answer_type == 'choice' and mmstar_annot and qid in mmstar_annot:
                old_qid = qid 
                question = resp.get('question', resp.get('question', '')) 
                for k, annot in mmstar_annot.items():
                    if question.strip()[:50] in annot['question'] : 
                        qid = k 
                        print(f"original qid found {k}")
                        break 
                annot['answer_type'] = 'choice'
                annot['human_question'] = question 
                gt_by_qid[qid] = annot  
                resp['old_qid'] = old_qid

        # Map confidence
        confidence = resp.get("confidence", 3)
        resp['confidence'] = CONF_MAP.get(str(confidence), 0.5)
        responses_by_qid[qid].append(resp)
    return dict(responses_by_qid), gt_by_qid


# =============================================================================
# Per-Question Processing
# =============================================================================

def process_question_responses(
    qid: str,
    responses: List[Dict],
    gt_info: Dict,
    encoder=None
) -> Dict:
    """
    Process all responses for a single question.

    Computes per-question average accuracy and similarity against ground truth.
    """
    answer_type = gt_info.get('answer_type', 'text')

    # Extract answers and confidences
    answers = [r['answer'] for r in responses]
    confidences = [r['confidence'] for r in responses]

    if answer_type == 'text':
        # VQA: Compute accuracy and similarity for each response
        gt_answers = gt_info['gt_answers']
        visual_gt = gt_info.get('visual_gt', '')

        if not gt_answers:
            return None

        accuracies = []
        gt_similarities = []  # Similarity to ground truth (10 annotators)
        visual_similarities = []  # Similarity to visual GT (humans who saw image)

        for answer in answers:
            # Accuracy against GT
            acc = vqa_accuracy(gt_answers, answer)
            accuracies.append(acc)

            # Similarity to GT (use majority answer)
            if encoder and gt_answers:
                majority_gt = max(set(gt_answers), key=gt_answers.count)
                sim_gt = compute_similarity(majority_gt, answer, encoder)
                gt_similarities.append(sim_gt)

            # Similarity to visual GT
            if encoder and visual_gt:
                sim_visual = compute_similarity(visual_gt, answer, encoder)
                visual_similarities.append(sim_visual)

        result = {
            'qid': qid,
            'answer_type': 'text',
            'num_responses': len(responses),
            'answers': answers,
            'confidences': confidences,
            'gt_answers': gt_answers,
            'visual_gt': visual_gt,
            # Averages
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            # Individual metrics
            'accuracies': accuracies,
        }

        if gt_similarities:
            result['mean_gt_similarity'] = float(np.mean(gt_similarities))
            result['std_gt_similarity'] = float(np.std(gt_similarities))
            result['gt_similarities'] = gt_similarities

        if visual_similarities:
            result['mean_visual_similarity'] = float(np.mean(visual_similarities))
            result['std_visual_similarity'] = float(np.std(visual_similarities))
            result['visual_similarities'] = visual_similarities

    else:  # choice
        gt_answer = gt_info.get('answer', '')
        choices = {
            1: "A", 
            2: "B", 
            3: "C", 
            4: "D", 
        }

        # Extract choices and compute accuracy
        extracted_choices = [choices[int(ans)] for ans in answers] # [extract_mc_choice(ans) for ans in answers]
        correct = [1 if choice == gt_answer.strip().upper()[0] else 0 for choice in extracted_choices]
        result = {
            'qid': qid,
            'answer_type': 'choice',
            'answer_type': 'text',
            'num_responses': len(responses),
            'answers': answers,
            'extracted_choices': extracted_choices,
            'confidences': confidences,
            # Averages
            'mean_accuracy': float(np.mean(correct)),
            'std_accuracy': float(np.std(correct)),
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            # Individual metrics
            'accuracies': correct,
        }
        result = {**result, **gt_info}

    return result


# =============================================================================
# Main Processing
# =============================================================================

def process_all_responses(
    human_data_dir: str,
    session: str,
    output_dir: str,
    with_similarity: bool = False
):
    """
    Process all raw human responses and compute per-question metrics.
    """
    print(f"\n{'='*60}")
    print(f"📊 Processing Raw Human Responses")
    print(f"   Session: {session}")
    print(f"   Data dir: {human_data_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Load questions
    questions_path = f"/home/work/yuna/HPA/dataset/questions/{session}.csv"
    # Load annotations
    print("\n📚 Loading annotations...")
    vqa_mapper = get_vqa_mapper()
    mmstar_annot = load_mmstar_annotations(session)

    # Load encoder if needed
    encoder = None
    if with_similarity:
        print("\n🔧 Loading sentence transformer...")
        encoder = get_encoder()
        if encoder:
            print("   ✓ Encoder loaded")
        else:
            print("   ⚠️ Failed to load encoder, skipping similarity")

    # Process VQA (text)
    print(f"\n{'='*60}")
    print("Processing VQA (text) responses...")
    print(f"{'='*60}")

    # Load raw responses
    # all_responses = load_raw_human_data(human_data_dir, questions_path) 
    all_responses = preprocess_pipeline(glob(f"{human_data_dir}/*/*.csv"), questions_path, output_dir)  
    text_responses_by_qid, text_gt = get_responses_by_qid(
        all_responses['responses']['text'], 'text', vqa_mapper=vqa_mapper 
    )
    print(f"   Found {len(text_responses_by_qid)} VQA questions")

    text_results = []
    for qid, responses in tqdm(text_responses_by_qid.items(), desc="Processing VQA"):
        if qid in text_gt:
            result = process_question_responses(qid, responses, text_gt[qid], encoder)
            if result:
                text_results.append(result)

    # Save VQA results
    text_output = os.path.join(output_dir, 'human_vqa_per_question.csv') 
    pd.DataFrame(text_results).to_csv(text_output) 
    # with open(text_output, 'w', encoding='utf-8') as f:
    #     for item in text_results:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\n   ✓ Saved: {text_output}")

    # Compute VQA statistics
    vqa_stats = {
        'num_questions': len(text_results),
        'total_responses': sum(r['num_responses'] for r in text_results),
        'mean_accuracy': float(np.mean([r['mean_accuracy'] for r in text_results])),
        'mean_confidence': float(np.mean([r['mean_confidence'] for r in text_results])),
        'correlation_conf_acc': float(np.corrcoef(
            [r['mean_confidence'] for r in text_results],
            [r['mean_accuracy'] for r in text_results]
        )[0, 1]) if len(text_results) > 1 else 0.0,
    }

    if with_similarity and any('mean_gt_similarity' in r for r in text_results):
        vqa_stats['mean_gt_similarity'] = float(np.mean([
            r['mean_gt_similarity'] for r in text_results if 'mean_gt_similarity' in r
        ]))
        vqa_stats['correlation_gt_sim_acc'] = float(np.corrcoef(
            [r['mean_gt_similarity'] for r in text_results if 'mean_gt_similarity' in r],
            [r['mean_accuracy'] for r in text_results if 'mean_gt_similarity' in r]
        )[0, 1])

        if any('mean_visual_similarity' in r for r in text_results):
            vqa_stats['mean_visual_similarity'] = float(np.mean([
                r['mean_visual_similarity'] for r in text_results if 'mean_visual_similarity' in r
            ]))
            vqa_stats['correlation_visual_sim_acc'] = float(np.corrcoef(
                [r['mean_visual_similarity'] for r in text_results if 'mean_visual_similarity' in r],
                [r['mean_accuracy'] for r in text_results if 'mean_visual_similarity' in r]
            )[0, 1])

    stats_output = os.path.join(output_dir, 'human_vqa_stats.json')
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(vqa_stats, f, indent=2)
    print(f"   ✓ Saved: {stats_output}")

    # Process MC (choice)
    print(f"\n{'='*60}")
    print("Processing MC (choice) responses...")
    print(f"{'='*60}")
    print(f"   Found {len(text_responses_by_qid)} VQA questions")

    choice_responses_by_qid, choice_gt = get_responses_by_qid(
        all_responses['responses']['choice'], 'choice', mmstar_annot=mmstar_annot
    )
    print(f"   Found {len(choice_responses_by_qid)} MC questions")

    choice_results = []
    for qid, responses in tqdm(choice_responses_by_qid.items(), desc="Processing MC"):
        if qid in choice_gt:
            result = process_question_responses(qid, responses, choice_gt[qid])
            if result:
                choice_results.append(result)

    # Save MC results
    choice_output = os.path.join(output_dir, 'human_mc_per_question.csv')
    pd.DataFrame(choice_results).to_csv(choice_output) 
    # with open(choice_output, 'w', encoding='utf-8') as f:
    #     for item in choice_results:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\n   ✓ Saved: {choice_output}")

    # Compute MC statistics
    mc_stats = {
        'num_questions': len(choice_results),
        'total_responses': sum(r['num_responses'] for r in choice_results),
        'mean_accuracy': float(np.mean([r['mean_accuracy'] for r in choice_results])),
        'mean_confidence': float(np.mean([r['mean_confidence'] for r in choice_results])),
        'correlation_conf_acc': float(np.corrcoef(
            [r['mean_confidence'] for r in choice_results],
            [r['mean_accuracy'] for r in choice_results]
        )[0, 1]) if len(choice_results) > 1 else 0.0,
    }

    stats_output = os.path.join(output_dir, 'human_mc_stats.json')
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(mc_stats, f, indent=2)
    print(f"   ✓ Saved: {stats_output}")

    # Save QID lists
    vqa_qids = [r['qid'] for r in text_results]
    mc_qids = [r['qid'] for r in choice_results]

    with open(os.path.join(output_dir, 'human_vqa_qids.json'), 'w') as f:
        json.dump({'qids': vqa_qids, 'count': len(vqa_qids)}, f, indent=2)

    with open(os.path.join(output_dir, 'human_mc_qids.json'), 'w') as f:
        json.dump({'qids': mc_qids, 'count': len(mc_qids)}, f, indent=2)

    print(f"\n{'='*60}")
    print("📈 Summary")
    print(f"{'='*60}")
    print(f"VQA Questions: {len(text_results)}")
    print(f"  Mean Accuracy: {vqa_stats['mean_accuracy']:.4f}")
    print(f"  Mean Confidence: {vqa_stats['mean_confidence']:.4f}")
    print(f"  Correlation (Conf-Acc): {vqa_stats['correlation_conf_acc']:.4f}")
    if 'mean_gt_similarity' in vqa_stats:
        print(f"  Mean GT Similarity: {vqa_stats['mean_gt_similarity']:.4f}")
        print(f"  Correlation (GT Sim-Acc): {vqa_stats['correlation_gt_sim_acc']:.4f}")
    if 'mean_visual_similarity' in vqa_stats:
        print(f"  Mean Visual Similarity: {vqa_stats['mean_visual_similarity']:.4f}")
        print(f"  Correlation (Visual Sim-Acc): {vqa_stats['correlation_visual_sim_acc']:.4f}")

    print(f"\nMC Questions: {len(choice_results)}")
    print(f"  Mean Accuracy: {mc_stats['mean_accuracy']:.4f}")
    print(f"  Mean Confidence: {mc_stats['mean_confidence']:.4f}")
    print(f"  Correlation (Conf-Acc): {mc_stats['correlation_conf_acc']:.4f}")

    print(f"\n✅ Processing complete! Results saved to: {output_dir}")

    return text_results, choice_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process raw human responses and compute per-question metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--human_data_dir", type=str,
                        default="/home/work/yuna/HPA/data/humans/all_results_20251206_154732",
                        help="Directory with raw human CSV files")
    parser.add_argument("--session", type=str, default="s1",
                        help="Session identifier (for questions file)")
    parser.add_argument("--output_dir", type=str, default="/home/work/yuna/HPA/evaluation/scored/humans",
                        help="Output directory for processed results")
    parser.add_argument("--with_similarity", action="store_true",
                        help="Compute embedding similarity (requires sentence-transformers)")

    args = parser.parse_args()

    process_all_responses(
        args.human_data_dir,
        args.session,
        args.output_dir,
        args.with_similarity
    )


if __name__ == '__main__':
    main()
