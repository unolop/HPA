#!/usr/bin/env python3
"""
6_analyze_calibration.py - Human Confidence Calibration Analysis

Analyzes how well human confidence predicts accuracy.
Only uses ACTUAL study data (excludes pilot with fake confidence=1).

Computes:
- Calibration curve (confidence level → accuracy)
- ECE (Expected Calibration Error)
- Per-participant calibration
- Response time analysis

Usage:
    python 6_analyze_calibration.py \
        --human_responses ./processed_data/actual/individual_responses.json \
        --model_results ./results/InternVL3_5-2B_mmstar_inst_blind.jsonl \
        --output_dir ./analysis/calibration
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
from scipy.stats import spearmanr


# =============================================================================
# Data Loading
# =============================================================================

def load_human_responses(path: str) -> List[Dict]:
    """Load human responses, excluding pilot data."""
    responses = []
    
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    responses.append(json.loads(line))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            responses = data if isinstance(data, list) else data.get('responses', [])
    
    # Filter out pilot data (has confidence=1 and participant_id='pilot')
    actual = [r for r in responses 
              if r.get('participant_id') != 'pilot' and r.get('source') != 'pilot']
    
    if len(actual) < len(responses):
        print(f"   ⚠️ Filtered {len(responses) - len(actual)} pilot responses")
    
    return actual


def load_ground_truth(path: str) -> Dict[str, str]:
    """Load ground truth from model results JSONL."""
    gt = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qid = str(item.get('qid', item.get('index', '')))
                if qid.endswith('.0'):
                    qid = qid[:-2]
                gt[qid] = item.get('answer', '')
    
    return gt


def normalize_answer(answer: str) -> str:
    """Normalize answer."""
    if not answer:
        return ""
    answer = str(answer).lower().strip()
    for article in ['a ', 'an ', 'the ']:
        if answer.startswith(article):
            answer = answer[len(article):]
    import string
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(answer.split()).strip()


def add_correctness(responses: List[Dict], gt: Dict[str, str]) -> List[Dict]:
    """Add correctness label to each response."""
    for r in responses:
        qid = str(r['qid'])
        if qid.endswith('.0'):
            qid = qid[:-2]
        
        if qid in gt:
            pred = normalize_answer(r.get('answer_normalized', r.get('answer', '')))
            truth = normalize_answer(gt[qid])
            r['correct'] = (pred == truth)
            r['ground_truth'] = gt[qid]
        else:
            r['correct'] = None
    
    return responses


# =============================================================================
# Calibration Analysis
# =============================================================================

def compute_calibration_curve(responses: List[Dict]) -> Dict[int, Dict]:
    """Compute accuracy for each confidence level."""
    by_conf = defaultdict(list)
    
    for r in responses:
        if r.get('correct') is not None:
            conf = int(r.get('confidence', 3))
            by_conf[conf].append(r['correct'])
    
    curve = {}
    for conf in sorted(by_conf.keys()):
        correct_list = by_conf[conf]
        acc = np.mean(correct_list)
        curve[conf] = {
            'accuracy': acc,
            'count': len(correct_list),
            'std_error': np.std(correct_list) / np.sqrt(len(correct_list)) if len(correct_list) > 1 else 0,
            'expected': conf / 5.0,  # Perfect calibration
        }
    
    return curve


def compute_calibration_metrics(responses: List[Dict]) -> Dict[str, float]:
    """Compute ECE, MCE, Brier score."""
    curve = compute_calibration_curve(responses)
    
    if not curve:
        return {}
    
    total = sum(c['count'] for c in curve.values())
    
    # ECE (Expected Calibration Error)
    ece = sum(
        (c['count'] / total) * abs(c['accuracy'] - c['expected'])
        for c in curve.values()
    )
    
    # MCE (Maximum Calibration Error)
    mce = max(abs(c['accuracy'] - c['expected']) for c in curve.values())
    
    # Brier Score
    brier_scores = []
    for r in responses:
        if r.get('correct') is not None:
            prob = r.get('confidence', 3) / 5.0
            outcome = 1.0 if r['correct'] else 0.0
            brier_scores.append((prob - outcome) ** 2)
    
    brier = np.mean(brier_scores) if brier_scores else 0
    
    # Over/under confidence
    overconf, underconf = [], []
    for r in responses:
        if r.get('correct') is not None:
            prob = r.get('confidence', 3) / 5.0
            outcome = 1.0 if r['correct'] else 0.0
            diff = prob - outcome
            if diff > 0:
                overconf.append(diff)
            else:
                underconf.append(-diff)
    
    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'mean_overconfidence': np.mean(overconf) if overconf else 0,
        'mean_underconfidence': np.mean(underconf) if underconf else 0,
        'pct_overconfident': len(overconf) / len(responses) if responses else 0,
    }


# =============================================================================
# Participant Analysis
# =============================================================================

def analyze_participants(responses: List[Dict]) -> Dict[str, Dict]:
    """Analyze each participant's calibration."""
    by_participant = defaultdict(list)
    
    for r in responses:
        pid = r.get('participant_id', 'unknown')
        by_participant[pid].append(r)
    
    results = {}
    
    for pid, p_responses in by_participant.items():
        valid = [r for r in p_responses if r.get('correct') is not None]
        
        if len(valid) < 5:
            continue
        
        accuracy = np.mean([r['correct'] for r in valid])
        mean_conf = np.mean([r.get('confidence', 3) for r in valid]) / 5.0
        
        # Calibration error per response
        cal_errors = [
            abs((r.get('confidence', 3) / 5.0) - (1.0 if r['correct'] else 0.0))
            for r in valid
        ]
        
        results[pid] = {
            'num_responses': len(valid),
            'accuracy': accuracy,
            'mean_confidence': mean_conf * 5,
            'calibration_error': np.mean(cal_errors),
            'overconfidence': mean_conf - accuracy,
        }
    
    return results


# =============================================================================
# Question Analysis
# =============================================================================

def analyze_questions(responses: List[Dict]) -> Dict[str, Dict]:
    """Analyze difficulty and confidence patterns per question."""
    by_qid = defaultdict(list)
    
    for r in responses:
        qid = str(r['qid'])
        by_qid[qid].append(r)
    
    results = {}
    
    for qid, q_responses in by_qid.items():
        valid = [r for r in q_responses if r.get('correct') is not None]
        if not valid:
            continue
        
        accuracy = np.mean([r['correct'] for r in valid])
        mean_conf = np.mean([r.get('confidence', 3) for r in valid])
        
        answers = [r.get('answer_normalized', r.get('answer', '')) for r in valid]
        agreement = Counter(answers).most_common(1)[0][1] / len(answers)
        
        results[qid] = {
            'num_responses': len(valid),
            'accuracy': accuracy,
            'mean_confidence': mean_conf,
            'agreement': agreement,
            'overconfidence': (mean_conf / 5.0) - accuracy,
            'question': valid[0].get('question', '')[:100],
            'category': valid[0].get('category', ''),
        }
    
    return results


def find_interesting_questions(question_stats: Dict[str, Dict], top_k: int = 10) -> Dict:
    """Find questions with notable patterns."""
    questions = list(question_stats.values())
    
    return {
        'overconfident': sorted(
            [q for q in questions if q['accuracy'] < 0.4],
            key=lambda x: -x['mean_confidence']
        )[:top_k],
        'underconfident': sorted(
            [q for q in questions if q['accuracy'] > 0.7],
            key=lambda x: x['mean_confidence']
        )[:top_k],
        'high_disagreement': sorted(
            questions, key=lambda x: x['agreement']
        )[:top_k],
    }


# =============================================================================
# Response Time Analysis
# =============================================================================

def analyze_response_time(responses: List[Dict]) -> Dict:
    """Analyze relationship between time, confidence, and accuracy."""
    valid = [r for r in responses 
             if r.get('correct') is not None and r.get('time_spent', 0) > 0]
    
    if len(valid) < 10:
        return {}
    
    times = np.array([r['time_spent'] for r in valid])
    confs = np.array([r.get('confidence', 3) for r in valid])
    correct = np.array([r['correct'] for r in valid])
    
    time_conf_r, _ = spearmanr(times, confs)
    time_acc_r, _ = spearmanr(times, correct)
    
    # Quartile analysis
    q25, q75 = np.percentile(times, [25, 75])
    
    fast = [r for r in valid if r['time_spent'] <= q25]
    slow = [r for r in valid if r['time_spent'] > q75]
    
    return {
        'time_confidence_correlation': time_conf_r,
        'time_accuracy_correlation': time_acc_r,
        'fast_accuracy': np.mean([r['correct'] for r in fast]) if fast else 0,
        'slow_accuracy': np.mean([r['correct'] for r in slow]) if slow else 0,
        'fast_confidence': np.mean([r.get('confidence', 3) for r in fast]) if fast else 0,
        'slow_confidence': np.mean([r.get('confidence', 3) for r in slow]) if slow else 0,
    }


# =============================================================================
# Main
# =============================================================================

def run_analysis(human_path: str, gt_path: str, output_dir: str) -> Dict:
    """Run full calibration analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🎯 HUMAN CALIBRATION ANALYSIS")
    print("=" * 60)
    
    # Load
    print("\n📂 Loading data...")
    responses = load_human_responses(human_path)
    print(f"   Responses: {len(responses)}")
    
    gt = load_ground_truth(gt_path)
    print(f"   Ground truth: {len(gt)} questions")
    
    responses = add_correctness(responses, gt)
    valid = [r for r in responses if r.get('correct') is not None]
    print(f"   Valid (with GT): {len(valid)}")
    
    # Overall stats
    overall_acc = np.mean([r['correct'] for r in valid])
    overall_conf = np.mean([r.get('confidence', 3) for r in valid])
    
    print(f"\n📊 Overall:")
    print(f"   Accuracy: {overall_acc:.4f}")
    print(f"   Mean confidence: {overall_conf:.2f}/5")
    
    # Calibration
    print("\n🎯 Calibration...")
    curve = compute_calibration_curve(valid)
    metrics = compute_calibration_metrics(valid)
    print(f"   ECE: {metrics.get('ece', 0):.4f}")
    print(f"   Brier: {metrics.get('brier', 0):.4f}")
    
    # Print calibration curve
    print("\n   Conf | Accuracy | Count")
    print("   -----|----------|------")
    for conf in sorted(curve.keys()):
        c = curve[conf]
        print(f"     {conf}  |  {c['accuracy']:.3f}   | {c['count']}")
    
    # Participants
    print("\n👥 Participant analysis...")
    participants = analyze_participants(valid)
    print(f"   Analyzed {len(participants)} participants")
    
    # Questions
    print("\n❓ Question analysis...")
    questions = analyze_questions(valid)
    interesting = find_interesting_questions(questions)
    
    # Response time
    print("\n⏱️ Response time...")
    time_analysis = analyze_response_time(valid)
    if time_analysis:
        print(f"   Time-accuracy corr: {time_analysis.get('time_accuracy_correlation', 0):.4f}")
    
    # Compile results
    results = {
        'overall': {
            'total_responses': len(responses),
            'valid_responses': len(valid),
            'accuracy': overall_acc,
            'mean_confidence': overall_conf,
            'num_participants': len(set(r.get('participant_id', '') for r in responses)),
            'num_questions': len(set(r['qid'] for r in responses)),
        },
        'calibration_curve': {str(k): v for k, v in curve.items()},
        'calibration_metrics': metrics,
        'participants': participants,
        'questions': questions,
        'interesting_questions': interesting,
        'time_analysis': time_analysis,
    }
    
    # Save
    with open(os.path.join(output_dir, 'calibration_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Human calibration analysis")
    parser.add_argument("--human_responses", type=str, required=True)
    parser.add_argument("--model_results", type=str, required=True,
                        help="Model results JSONL (for ground truth)")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    run_analysis(args.human_responses, args.model_results, args.output_dir)


if __name__ == "__main__":
    main()