#!/usr/bin/env python3
"""  
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
from preprocessing.preprocess import preprocess_pipeline, CONF_MAP
from analysis.utils.vqa import get_vqa_mapper, vqa_accuracy

def load_human_results(jsonl_path: str) -> pd.DataFrame: 
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

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
            print(resp) 
        else:  # choice
            gt_answer = gt_info.get('answer', '')
            choice = choices[int(answer)]  
            resp['correct'] = 1 if choice == gt_answer.strip().upper()[0] else 0  
 
        results.append({**resp, **gt_info})   

    return results  

def process_all_responses(
    human_data_dir: str,
    session: str,
    output_dir: str
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
    all_responses = preprocess_pipeline(glob(f"{human_data_dir}/*/*.csv"), questions_path, output_dir)  
    
    # Load annotations
    print("\n📚 Loading annotations...")
    vqa_mapper = get_vqa_mapper()
    mmstar_annot = load_mmstar_annotations(session) 
    encoder = get_encoder() 

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
        json.dump([item for sublist in choice_results for item in sublist] , f, indent=2) 
    print(f"\n   ✓ Saved: {choice_output}")

    # Process VQA (text)
    print("Processing VQA (text) responses...") 
    text_responses_by_qid, text_gt = get_responses_by_qid(
        all_responses['responses']['text'], 'text', vqa_mapper=vqa_mapper )
    print(f"   Found {len(text_responses_by_qid)} VQA questions")

    text_output = os.path.join(output_dir, 'human_vqa_per_question.json')  
    text_results = []
    for qid, responses in tqdm(text_responses_by_qid.items(), desc="Processing VQA"):
        if qid in text_gt:
            result = process_question_responses(qid, responses, text_gt[qid], encoder)
            text_results.append(result)
            with open(text_output, 'w', encoding='utf-8') as f:
                json.dump(text_results, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
    return text_results, choice_results


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
    

    args = parser.parse_args()

    process_all_responses(
        args.human_data_dir,
        args.session,
        args.output_dir
    )


if __name__ == '__main__':
    main()