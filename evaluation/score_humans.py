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
from analysis.utils.score import *   
sys.path.append(str(Path(__file__).parent.parent)) 
from preprocessing.preprocess import preprocess_pipeline 


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
class VQAAnswerMapper:
    """Maps question_id to list of ground truth answers from VQA annotations."""

    def __init__(self, annotations_path: str = VQA_ANNOTATIONS_PATH):
        self.annotations_path = annotations_path
        self._qid_to_answers = None
        self._qid_to_gt_visual = None  # Store visual GT for VQA
        self.annotations = {} 

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
            self.annotations[qid] = ann 
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
    not_matched = {}
    for row in questions:
        qid = row['qid']
        question = row.get('question_en', row.get('question', ''))
        matched = False 
        # DEBUGGED MATCHING LOGIC
        not_matched[qid] = []
        for data in mmstar_data:
            if (
                    (qid == "802" and str(data.get('pid')) == "715")
                    or
                    (qid == "287" and str(data.get('pid')) == "1499")
                    or
                    (qid == "588" and str(data.get('pid')) == "541") 
                    or
                    (qid == "214" and str(data.get('pid')) == "284") 
                ):
                    # print(f'manually matched {qid}')
                    data['human_qid'] = qid
                    data['original_question'] = data['question']
                    annot[qid] = data
                    matched = True
                    break  # Once manually matched, break out of the loop

            if question.strip()[:50] in data['question'] and qid not in annot:
                mmstar_options_str = data['question'].split('\nOptions: ')[-1] 
                mmstar_options = [opt.strip() for opt in mmstar_options_str.split(',')]
                row_options = row.get('options', '')
                row_options_list = [opt.strip() for opt in row_options.split(',')]
                matched = all(any(mo.lower()[:5] in ro.lower() for ro in row_options_list[:2]) for mo in mmstar_options[:2])
                
                if matched:
                    data['human_qid'] = qid
                    data['original_question'] = data['question']
                    annot[qid] = data
                    # print(f'[!] Matched options by relaxed approach for QID {qid}\n  MMStar: {mmstar_options}\n  Human : {row_options_list}\n  Data: {data}\n  Row:  {row}')
                    break
                else: 
                    not_matched[qid].append(data) 
                    
        if not matched: 
            # print(f"Not matching {row_options_list, mmstar_options}")  
            if len(not_matched[qid]) == 1 : 
                data = not_matched[qid][0] 
                data['human_qid'] = qid 
                data['original_question'] = data['question']
                annot[qid] = data
            else :
                breakpoint()          
                    
    print('Length of MMStar questions:', len(annot))  
    annot_keys = list(annot.keys())
    if len(annot_keys) != len(set(annot_keys)):
        dupes = set([k for k in annot_keys if annot_keys.count(k) > 1])
        print(f"[!] Duplicate QIDs found in annot: {dupes}") 
    return annot

# =============================================================================
# Data Loading
# =============================================================================

def get_responses_by_qid(
    answers: List[Dict],
    answer_type: str,
    vqa_mapper: VQAAnswerMapper = None,
    mmstar_annot: Dict = None
) -> Tuple[Dict, Dict]:
    answers = [a for a in answers if a.get("answer_type") == answer_type]
    responses_by_qid = defaultdict(list)
    gt_by_qid = {}
    
    for resp in answers:
        qid = str(resp['qid'])
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
                annot = mmstar_annot[qid]  
                # for k, annot in mmstar_annot.items():
                #     if question.strip()[:50] in annot['question'] : 
                #         qid = k 
                #         print(f"original qid found {k}")
                #         break 
                annot['answer_type'] = 'choice'
                # annot['human_question'] = question 
                gt_by_qid[qid] = annot  

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
    answer_type = gt_info.get('answer_type', 'text')
    '''
    process subject answers per question 
    returns a list (by subject number) of dictionary scored each answer  
    '''
    choices = {
        1: "A", 
        2: "B", 
        3: "C", 
        4: "D", 
    }
    results = []
    for i, resp in enumerate(responses):  
        # agreement = {}
        answer = normalize_answer(resp['answer'] )
        resp['qid'] = qid 

        if answer_type == 'text':
            gt_answers = gt_info['gt_answers']
            # visual_gt = gt_info.get('visual_gt', '')  # what is this ? 
            resp['correct'] = vqa_accuracy(gt_answers, answer)
            if resp['correct'] is None or (isinstance(resp['correct'], float) and np.isnan(resp['correct'])):
                print(f"⚠️  NaN or None value in 'correct': qid={qid}, answer={answer}, gt_answers={gt_answers}, resp={resp}")
            resp['answer_similarity'] = np.mean([compute_similarity(gt, answer, encoder) for gt in gt_answers]) 
            # for j, rj in enumerate(range(len(responses))):   # compute other answer simlarities 
            #     if encoder: 
            #         if i == j:  
            #             sim = 1 
            #         else: 
            #             sim = compute_similarity(answer, responses[rj]['answer'], encoder)  
            #         agreement[responses[rj]['participant_id']]=sim  
            print(resp) 
        else:  # choice
            gt_answer = gt_info.get('answer', '')
            choice = choices[int(answer)]  
            resp['correct'] = 1 if choice == gt_answer.strip().upper()[0] else 0  

            # for j, choice_j in enumerate(range(len(responses))): 
            #     if i != j: 
            #         a = 1 if choice == choice_j else 0  
            #         agreement[responses[choice_j]['participant_id']] = a  
        
        # resp['agreement'] = agreement 
        results.append({**resp, **gt_info})   

    return results 

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
    human_data_dir =f"/home/work/yuna/HPA/data/humans/{human_data_dir}" 
    print(f"   Session: {session}")
    print(f"   Data dir: {human_data_dir}")
    
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
    print("Processing VQA (text) responses...") 
    all_responses = preprocess_pipeline(glob(f"{human_data_dir}/*/*.csv"), questions_path, output_dir)  
    text_responses_by_qid, text_gt = get_responses_by_qid(
        all_responses['responses']['text'], 'text', vqa_mapper=vqa_mapper 
    )
    print(f"   Found {len(text_responses_by_qid)} VQA questions")

    text_output = os.path.join(output_dir, 'human_vqa_per_question.json')  
    text_results = []
    for qid, responses in tqdm(text_responses_by_qid.items(), desc="Processing VQA"):
        if qid in text_gt:
            result = process_question_responses(qid, responses, text_gt[qid], encoder)
            if result:
                if isinstance(result, list):
                    text_results.extend(result)
                else:
                    text_results.append(result)
                with open(text_output, 'w', encoding='utf-8') as f:
                    json.dump(text_results, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())

    # Compute VQA statistics
    # vqa_stats = {
    #     'num_questions': len(text_results),
    #     # 'total_responses': sum(r['num_responses'] for r in text_results),
    #     'mean_accuracy': float(np.mean([r['mean_accuracy'] for r in text_results])),
    #     'mean_confidence': float(np.mean([r['mean_confidence'] for r in text_results])),
    #     'correlation_conf_acc': get_pearsonr_correlation({"confidence":[r['mean_confidence'] for r in text_results],  
    #                                                         "accuracy":[r['mean_accuracy'] for r in text_results]})  ,                                                        
    # }

    # if with_similarity: # get correlations 
    #     if any('mean_visual_similarity' in r for r in text_results):
    #         visual_sim = np.array([
    #             r['mean_visual_similarity']
    #             for r in text_results
    #             if 'mean_visual_similarity' in r and 'mean_accuracy' in r
    #         ])

    #         accuracy = np.array([
    #             r['mean_accuracy']
    #             for r in text_results
    #             if 'mean_visual_similarity' in r and 'mean_accuracy' in r
    #         ])
    #         corr = get_pearsonr_correlation({"visual_sim":visual_sim, "accuracy":accuracy})  
    #         vqa_stats['correlation_visual_sim_acc'] = corr

    # stats_output = os.path.join(output_dir, 'human_vqa_stats.json')
    # with open(stats_output, 'w', encoding='utf-8') as f:
    #     json.dump(vqa_stats, f, indent=2)
    # print(f"   ✓ Saved: {stats_output}") 

    # Process MC (choice)
    print("Processing MC (choice) responses...")
    print(f"   Found {len(text_responses_by_qid)} VQA questions")

    choice_responses_by_qid, choice_gt = get_responses_by_qid(
        all_responses['responses']['choice'], 'choice', mmstar_annot=mmstar_annot
    )
    print(f"   Found {len(choice_responses_by_qid)} MC questions")

    choice_results = []
    for qid, responses in tqdm(choice_responses_by_qid.items(), desc="Processing MC"):
        choice_results.append(process_question_responses(qid, responses, choice_gt[qid])) 

    # Save MC results
    choice_output = os.path.join(output_dir, 'human_mc_per_question.json') 
    with open(choice_output, 'w', encoding='utf-8') as f: 
        json.dump(choice_results, f, indent=2) 
    print(f"\n   ✓ Saved: {choice_output}")

    # Compute MC statistics
    # mc_stats = {
    #     'num_questions': len(choice_results),
    #     'total_responses': sum(r['num_responses'] for r in choice_results),
    #     'mean_accuracy': float(np.mean([r['mean_accuracy'] for r in choice_results])),
    #     'mean_confidence': float(np.mean([r['mean_confidence'] for r in choice_results])),
    #     'correlation_conf_acc': get_pearsonr_correlation({"confidence":[r['mean_confidence'] for r in choice_results], 
    #                                                         "accuracy":[r['mean_accuracy'] for r in choice_results]})  
    # }
    # stats_output = os.path.join(output_dir, 'human_mc_stats.json')
    # with open(stats_output, 'w', encoding='utf-8') as f:
    #     json.dump(mc_stats, f, indent=2)
    # print(f"   ✓ Saved: {stats_output}")

    # print("📈 Summary")
    # print(f"VQA Questions: {len(text_results)}")
    # print(f"  Mean Accuracy: {vqa_stats['mean_accuracy']:.4f}")
    # print(f"  Mean Confidence: {vqa_stats['mean_confidence']:.4f}")
    # # print(f"  Correlation (Conf-Acc): {vqa_stats['correlation_conf_acc']:.4f}") 

    # print(f"\nMC Questions: {len(choice_results)}")
    # print(f"  Mean Accuracy: {mc_stats['mean_accuracy']:.4f}")
    # print(f"  Mean Confidence: {mc_stats['mean_confidence']:.4f}")
    # # print(f"  Correlation (Conf-Acc): {mc_stats['correlation_conf_acc']:.4f}")
    
    # print(f"\n✅ Processing complete! Results saved to: {output_dir}")

    return text_results, choice_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process raw human responses and compute per-question metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--human_data_dir", type=str, default="n20", 
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