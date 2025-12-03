#!/usr/bin/env python3
"""
5_analyze_human_model.py - Human-Model Comparison Analysis

Compares human blind VQA responses with model predictions.
Uses scored results from score_results.py.

Computes:
- Per-question accuracy correlation (human vs model)
- Agreement matrix (both correct, both wrong, etc.)
- Confidence-accuracy relationship
- Per-category breakdown
- Error pattern analysis

Usage:
    python 5_analyze_human_model.py \
        --human_responses ./processed_data/actual/individual_responses.json \
        --model_results ./results/InternVL3_5-2B_mmstar_inst_blind.jsonl \
        --output_dir ./analysis/human_model
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import confusion_matrix

try:
    from sklearn.metrics import cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =============================================================================
# Data Loading
# =============================================================================

def load_human_responses(path: str) -> List[Dict]:
    """Load human responses from JSON or JSONL (excludes pilot data)."""
    responses = []
    print(f"reading human responses: {path} ...")
    
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    responses.append(json.loads(line))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            responses = data if isinstance(data, list) else data.get('responses', [])
    
    # Filter out pilot data
    actual_responses = [
        r for r in responses 
        if r.get('participant_id') != 'pilot' and r.get('source') != 'pilot'
    ]
    
    if len(actual_responses) < len(responses):
        print(f"   ⚠️ Filtered out {len(responses) - len(actual_responses)} pilot responses")
    
    by_qid = defaultdict(list)
    choices = ['A', 'B', 'C', 'D']
    
    for r in actual_responses:
        qid = str(r['qid'])
        if qid.endswith('.0'):
            qid = qid[:-2]
        if r.get('answer_type', '') == 'choice':
            c = int(r['answer']) -1 
            r = choices[c] 
        by_qid[qid].append(r)

    return by_qid # actual_responses


def load_model_results(path: str) -> Dict[str, Dict]:
    """Load model results from JSONL. Returns qid -> result dict."""
    print(f"Loading model results path: {path}")
    results = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qid = str(item.get('qid', item.get('index', item.get('question_id', ''))))
                if qid.endswith('.0'):
                    qid = qid[:-2]
                results[qid] = {
                    'prediction': item.get('output', ''),
                    'ground_truth': item.get('answers', item.get('answer', '')), ### UPDATED from answer -> answers 
                    'correct': item.get('correct', False),
                    'category': item.get('category', item.get('l2_category', item.get('question_type', ''))),
                    'question': item.get('question', ''),
                }
    
    return results


def aggregate_human_by_question(qid_responses) -> Dict[str, Dict]:
    """Aggregate human responses by question ID."""
    
    aggregated = {}
    
    for qid, qid_responses in by_qid.items():
        answers = [r.get('answer_normalized', r.get('answer', '')) for r in qid_responses]
        answer_counts = Counter(answers)
        majority_answer = answer_counts.most_common(1)[0][0]
        agreement = answer_counts.most_common(1)[0][1] / len(answers)
        
        confidences = [r.get('confidence', 3) for r in qid_responses]
        mean_conf = np.mean(confidences)
        
        aggregated[qid] = {
            'majority_answer': majority_answer,
            'mean_confidence': mean_conf,
            'agreement': agreement,
            'num_responses': len(qid_responses),
            'answer_distribution': dict(answer_counts),
            'category': qid_responses[0].get('category', ''),
            'question': qid_responses[0].get('question', ''),
        }
    
    return aggregated


# =============================================================================
# Answer Comparison
# =============================================================================

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    answer = str(answer).lower().strip()
    for article in ['a ', 'an ', 'the ']:
        if answer.startswith(article):
            answer = answer[len(article):]
    import string
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(answer.split()).strip()


def answers_match(ans1: str, ans2: str) -> bool:
    """Check if two answers match."""
    return normalize_answer(ans1) == normalize_answer(ans2)


def extract_answer_by_type(answer: str, answer_type: str) -> str:
    """Extract answer based on question type.

    For multiple choice questions (answer_type='choice'), extracts the first
    letter A, B, C, or D from the answer string.
    For other types, returns the full answer.
    """
    if answer_type == 'choice':
        import re
        answer_str = str(answer).upper()

        # Try to find pattern like "A:", "B:", etc. (answer choice with colon)
        match = re.search(r'\b([ABCD]):', answer_str)
        if match:
            return match.group(1)

        # Try to find standalone letter at word boundary
        match = re.search(r'\b([ABCD])\b', answer_str)
        if match:
            return match.group(1)

        # Fallback: find first occurrence of A, B, C, or D
        for char in answer_str:
            if char in ['A', 'B', 'C', 'D']:
                return char

        # Last resort: first character
        return answer_str[0] if answer_str else ''
    else:
        # For free-form answers, return the full text
        return str(answer)


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_agreement_matrix(human_data: Dict, model_data: Dict) -> Dict:
    """Compute 2x2 agreement matrix."""
    shared_qids = set(human_data.keys()) & set(model_data.keys())
    
    matrix = {'both_correct': 0, 'both_wrong': 0, 'human_only': 0, 'model_only': 0}
    details = []
    
    for qid in shared_qids:
        hm = human_data[qid]
        m = model_data[qid]
        
        ### TODO: FOR VQA FREE ANSWER TYPE, VQA ACCURACY AND EMBEDDING SCORE SHOULD BE CALCULATED INSTEAD
        gt = m.get('ground_truth', '')
        # gt = [g['answer'] for g in gt] ### TODO: IF GT IS A LIST OF DICTIONARIES (FOR VQA)
        h_correct = []
        m_correct = []

        # Get answer_type from human responses (all responses for same question have same type)
        answer_type = hm[0].get('answer_type', '') if isinstance(hm, list) and len(hm) > 0 else ''

        for g in gt:
            # Extract model prediction based on answer type (MCQ vs free-form)
            model_answer = extract_answer_by_type(m['prediction'], answer_type)
            m_correct.append(answers_match(model_answer, g))

            for h in hm:
                try:
                    # Human answers are already processed in load_human_responses
                    h_correct.append(answers_match(h, g))
                except Exception as e:
                    breakpoint()

        h_correct = np.mean(h_correct) 
        m_correct = np.mean(m_correct)  # m.get('correct', False)
        breakpoint()
        if h_correct and m_correct:
            matrix['both_correct'] += 1
            outcome = 'both_correct'
        elif not h_correct and not m_correct:
            matrix['both_wrong'] += 1
            outcome = 'both_wrong'
        elif h_correct:
            matrix['human_only'] += 1
            outcome = 'human_only'
        else:
            matrix['model_only'] += 1
            outcome = 'model_only'
        
        details.append({
            'qid': qid, 'outcome': outcome,
            'human_answer': hm,
            'model_answer': m['prediction'],
            'ground_truth': gt,

            ### TODO CONFIDENCE VALUES FROM HUMANS 
            # 'human_confidence': h['mean_confidence'],
            # 'human_agreement': h['agreement'],
        })
    
    total = len(shared_qids)
    return {
        **matrix,
        'total': total,
        'agreement_rate': (matrix['both_correct'] + matrix['both_wrong']) / total if total else 0,
        'human_accuracy': (matrix['both_correct'] + matrix['human_only']) / total if total else 0,
        'model_accuracy': (matrix['both_correct'] + matrix['model_only']) / total if total else 0,
        'details': details,
    }


def compute_correlations(human_data: Dict, model_data: Dict) -> Dict:
    """Compute various correlations."""
    shared_qids = list(set(human_data.keys()) & set(model_data.keys()))
    
    if len(shared_qids) < 10:
        return {'error': 'Too few shared questions'}
    
    h_correct, m_correct, h_conf, h_agree = [], [], [], []
    
    for qid in shared_qids:
        hm, md = human_data[qid], model_data[qid]
        gt = md.get('ground_truth', '')

        ### EDITED:
        h_correct = []
        m_correct = []

        # Get answer_type from human responses
        answer_type = hm[0].get('answer_type', '') if isinstance(hm, list) and len(hm) > 0 else ''

        # HUMAN: % correct
        human_correct_rate = np.mean([1 if answers_match(h, gt) else 0 for h in hm])
        h_correct.append(human_correct_rate)

        # MODEL: 1 or 0
        model_answer = extract_answer_by_type(md['prediction'], answer_type)
        model_is_correct = 1 if answers_match(model_answer, gt) else 0
        m_correct.append(model_is_correct)

        # h_conf.append(h['mean_confidence'])
        # h_agree.append(h['agreement'])

    # print("h_conf:", h_conf[:20], "unique:", np.unique(h_conf))
    # print("h_agree:", h_agree[:20], "unique:", np.unique(h_agree))
    # print("m_correct:", m_correct[:20], "unique:", np.unique(m_correct))

    confusion = confusion_matrix(h_correct, m_correct, labels=[0,1])
    print("confusion matrix:\n", confusion)
    # print("expected matrix:\n", expected)
    # print("weighted expected sum:", np.sum(w_mat * expected))
    results = {'num_questions': len(shared_qids)}
    
    # Accuracy correlation
    if len(set(h_correct)) > 1 and len(set(m_correct)) > 1:
        r, p = spearmanr(h_correct, m_correct)
        results['accuracy_spearman'] = {'r': r, 'p': p}
    
    # Confidence → Model accuracy
    r, p = spearmanr(h_conf, m_correct)
    results['confidence_to_model_acc'] = {'r': r, 'p': p}
    
    # Agreement → Model accuracy
    r, p = spearmanr(h_agree, m_correct)
    results['agreement_to_model_acc'] = {'r': r, 'p': p}
    
    # Cohen's Kappa
    if HAS_SKLEARN:
        results['cohen_kappa'] = cohen_kappa_score(h_correct, m_correct)
    
    return results


def compute_category_breakdown(human_data: Dict, model_data: Dict) -> Dict:
    """Per-category statistics."""
    shared_qids = set(human_data.keys()) & set(model_data.keys())
    
    by_cat = defaultdict(lambda: {'h_correct': 0, 'm_correct': 0, 'both': 0, 'total': 0, 'conf': []})
    
    for qid in shared_qids:
        h, m = human_data[qid], model_data[qid]
        cat = h.get('category') or m.get('category', 'Unknown')
        gt = m.get('ground_truth', '')
        
        h_ok = answers_match(h['majority_answer'], gt)
        m_ok = m.get('correct', False)
        
        by_cat[cat]['total'] += 1
        by_cat[cat]['h_correct'] += int(h_ok)
        by_cat[cat]['m_correct'] += int(m_ok)
        by_cat[cat]['both'] += int(h_ok and m_ok)
        by_cat[cat]['conf'].append(h['mean_confidence'])
    
    return {
        cat: {
            'total': d['total'],
            'human_accuracy': d['h_correct'] / d['total'] if d['total'] else 0,
            'model_accuracy': d['m_correct'] / d['total'] if d['total'] else 0,
            'both_correct_rate': d['both'] / d['total'] if d['total'] else 0,
            'mean_confidence': np.mean(d['conf']) if d['conf'] else 0,
        }
        for cat, d in by_cat.items()
    }


def find_interesting_cases(human_data: Dict, model_data: Dict, top_k: int = 10) -> Dict:
    """Find interesting disagreement cases."""
    shared_qids = set(human_data.keys()) & set(model_data.keys())
    
    cases = []
    for qid in shared_qids:
        h, m = human_data[qid], model_data[qid]
        gt = m.get('ground_truth', '')
        h_ok = answers_match(h['majority_answer'], gt)
        m_ok = m.get('correct', False)
        
        cases.append({
            'qid': qid,
            'question': h.get('question', '')[:100],
            'human_answer': h['majority_answer'],
            'model_answer': m['prediction'],
            'ground_truth': gt,
            'human_correct': h_ok,
            'model_correct': m_ok,
            'confidence': h['mean_confidence'],
            'agreement': h['agreement'],
        })
    
    return {
        'human_right_model_wrong': sorted(
            [c for c in cases if c['human_correct'] and not c['model_correct']],
            key=lambda x: -x['confidence']
        )[:top_k],
        'model_right_human_wrong': sorted(
            [c for c in cases if not c['human_correct'] and c['model_correct']],
            key=lambda x: -x['confidence']
        )[:top_k],
        'both_wrong_high_conf': sorted(
            [c for c in cases if not c['human_correct'] and not c['model_correct']],
            key=lambda x: -x['confidence']
        )[:top_k],
    }


# =============================================================================
# Main
# =============================================================================

def run_analysis(human_path: str, model_path: str, output_dir: str) -> Dict:
    """Run full analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🔬 HUMAN-MODEL COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Load
    print("\n📂 Loading data...")
    human_responses = load_human_responses(human_path)
    # print(f"   Human responses: {len(human_responses)}") ### CHANGED -> to a dict of qids 
    
    model_data = load_model_results(model_path)
    print(f"   Model predictions: {len(model_data)}")
    
    ### do not aggregate human data 
    ''' 
    human_data = aggregate_human_by_question(human_responses) 
    print(f"   Human questions: {len(human_data)}")
    
    shared = set(human_data.keys()) & set(model_data.keys())
    print(f"   Shared questions: {len(shared)}")
    '''
    # Analyze
    print("\n🤝 Agreement matrix...")
    agreement = compute_agreement_matrix(human_responses, model_data)
    
    print("\n📈 Correlations...")
    correlations = compute_correlations(human_responses, model_data)
    
    print("\n📁 Category breakdown...")
    categories = compute_category_breakdown(human_responses, model_data)
    
    print("\n🔍 Interesting cases...")
    interesting = find_interesting_cases(human_responses, model_data)
    
    # Results
    results = {
        'agreement_matrix': {k: v for k, v in agreement.items() if k != 'details'},
        'correlations': correlations,
        'by_category': categories,
        'interesting_cases': interesting,
    }
    
    # Save
    with open(os.path.join(output_dir, 'human_model_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(os.path.join(output_dir, 'agreement_details.jsonl'), 'w') as f:
        for d in agreement['details']:
            f.write(json.dumps(d) + '\n')
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Human accuracy:   {agreement['human_accuracy']:.4f}")
    print(f"Model accuracy:   {agreement['model_accuracy']:.4f}")
    print(f"Agreement rate:   {agreement['agreement_rate']:.4f}")
    if 'cohen_kappa' in correlations:
        print(f"Cohen's κ:        {correlations['cohen_kappa']:.4f}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Human-model comparison analysis")
    parser.add_argument("--human_responses", type=str, required=True)
    parser.add_argument("--model_results", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    run_analysis(args.human_responses, args.model_results, args.output_dir)


if __name__ == "__main__":
    main()