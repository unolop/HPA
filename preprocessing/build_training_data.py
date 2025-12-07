from tqdm import tqdm
import argparse
import glob  
import json
import sys 
from typing import Optional
sys.path.append('..')
from preprocess import *
from dataset.vqav2 import VQADataset
from aggregate import AnswerAggregator 
import random
from utils import read_and_clean_jsonl


CONF_MAP = {'yes': 1.0, 
            'maybe': 0.5, 
            'no': 0.01, 
            '1': 0.05,
            '2': 0.25,
            '3': 0.5,
            '4': 0.75,
            '5': 1.0
}

def read_pilot_data(path): 
    '''
    read and normalize pilot data 
    '''
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                line = json.loads(line)
                line['answer'] = normalize_answer(line['answer'])
                data.append(line)
    return data

def get_responses_by_qid(answers, vqa=True, blind=True, set_confidence=None):  

    if blind: 
        prompt = "\nNote: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario."
    else: 
        prompt = "" 
    
    if vqa:  
        vqa_val = VQADataset(prompt=prompt)
        vqa_val = vqa_val.get_by_qid() 

    responses_by_qid = {}
    for resp in answers: 
        qid = resp['qid'] # .get("qid", '') 
        if qid not in responses_by_qid.keys() : 
            responses_by_qid[qid] = [] 

        confidence = resp.get("confidence", 3)  
        if set_confidence is not None: 
            confidence = set_confidence 
        confidence = CONF_MAP[str(confidence)]  
        resp['confidence'] = confidence
        
        if vqa: # get original annotations 
            vqa_annot = vqa_val[qid]
            vqa_annot.pop('answers', None) # remove the ground truth answers 
            resp = {**vqa_annot, **resp}
            resp['question'] = vqa_annot['question']
        responses_by_qid[qid].append(resp)  

    return responses_by_qid 

def sample_data(processed, pilot=None, n=20):

    missing_qids = []
    for qid in processed.keys():
        resp = processed[qid]
        original_count = len(resp)
        deficit = n - len(resp)

        if deficit < 0:
            # We have more than enough data, truncate to n
            resp = resp[:n]
            print(f"[QID {qid}] Truncated {original_count} → {len(resp)} responses")

        # Append pilot data if we still need more
        if len(resp) < n and pilot is not None:
            print(f"[QID {qid}] Need {n - len(resp)} more responses (have {len(resp)}/{n})")
            if qid in pilot.keys():
                pilot_data = pilot[qid]
                idx = 0
                while len(resp) < n and idx < len(pilot_data):
                    resp.append(pilot_data[idx])
                    idx += 1
                print(f"[QID {qid}] Added {idx} pilot responses → now have {len(resp)}/{n}")
                if len(resp) != n:
                    print(f"[QID {qid}] ⚠️ Still missing {n - len(resp)} responses after pilot data")
            else:
                missing_qids.append(qid)  # Fixed: was appending list to itself!
                print(f"[QID {qid}] ⚠️ No pilot data available, missing {n - len(resp)} responses")

        processed[qid] = resp

    return processed, missing_qids 

def build_training_data(
    ds,
    output_jsonl_path: str,
    client: Optional["OpenAI"] = None,
    verbose: bool = True
):  
    processed = []
    aggregator = AnswerAggregator(cache_path="/home/work/yuna/HPA/preprocessing/answer_clustering_cache.json", client=client)

    # Set image path based on output path
    if 'blind' in output_jsonl_path: 
        image_path = "/home/work/yuna/HPA/data/blank_224.png"
        print('processing blind image annotations') 
    else:
        # Default: use actual image paths from data
        image_path = None  # Will be set from data if available
    
    print(f"Saving into file {output_jsonl_path}")
    
    with open(output_jsonl_path, "w") as f:
        for i, (qid, data) in enumerate(tqdm(ds.items(), desc="Processing questions")):
            question = data[0]['question']
            
            # Use image_path from data if not using blind image
            item_image_path = image_path
            if item_image_path is None and 'image' in data[0]:
                item_image_path = data[0]['image']
            elif item_image_path is None:
                item_image_path = f"images/{qid}.jpg"  # Fallback
            confidence_dist = aggregator.aggregate(data) 
            sorted_items = sorted(confidence_dist.items(), key=lambda x: -x[1])
            unique_answers = [item[0] for item in sorted_items]
            unique_confidences = [item[1] for item in sorted_items]
            # Get the index of the highest value in unique_confidences
            max_conf_idx = unique_confidences.index(max(unique_confidences)) if unique_confidences else 0
            if 'blind' in output_jsonl_path:
                ans = unique_answers[max_conf_idx] if unique_answers else ''
            else:
                # Use highest confidence answer if available, otherwise fall back to ground truth
                ans = unique_answers[max_conf_idx] if unique_answers else data[0].get('multiple_choice_answer', '') 

            
            item = {
                'idx': i,
                'images': [item_image_path],
                'qid': qid,
                'conversations': [
                    {'role': 'user', 'content': f'<image>{question}'},
                    {'role': 'assistant', 'content': ans}
                ],
                'labels': {
                    'confidences': unique_confidences,
                    'answers': unique_answers
                }
            }
            
            processed.append(item)
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            f.flush()  # Force write to disk immediately so file updates per iteration
    
    print(f"Wrote {len(processed)} items to {output_jsonl_path}")
    # print(f"Cache stats: {aggregator.get_cache_stats()}")
    
    return processed


def parse_args():
    parser = argparse.ArgumentParser(description="Build training data for HPA project")
    parser.add_argument('--results', type=str, default="all_results_20251206_154732", 
                        help='Glob pattern or comma-separated list of input CSV files')
    parser.add_argument('--session', type=str, default="s1", 
                        help='Directory to store cleaned training data')
    parser.add_argument('--n', type=int, default=10, 
                        help='Number of samples per question (default 10)')
    parser.add_argument('--answer_type', type=str, default='text', 
                        help='(Optional) Path to blank/placeholder image')
    return parser.parse_args()

def main(args): 

    ### S1 answers 
    input_csvs = glob.glob(f"/home/work/yuna/HPA/data/humans/{args.results}/*/*.csv") 
    questions_path = f"/home/work/yuna/HPA/dataset/questions/{args.session}.csv"
    output_dir = f"/home/work/yuna/HPA/data/training/{args.session}_{args.answer_type}" 

    ### Get pilot data 
    pilot_data = read_pilot_data("/home/work/yuna/HPA/data/humans/_all_pilot_cleaned.jsonl")
    for d in pilot_data: 
        d['confidence'] = 3
    pilot_data = get_responses_by_qid(pilot_data) 
    print(f'Total # Pilot Questions: {len(pilot_data.keys())}')  # number of qids / questions 
    
    ### Actual data 

    # get data by answer type and normalized 
    processed = preprocess_pipeline(input_csvs, questions_path, output_dir) 
    processed = processed['responses'][args.answer_type] 
    processed = get_responses_by_qid(processed)
    print('Selected answer type:', args.answer_type) 
    print(f'Total # Questions: {len(processed.keys())}')  # number of qids / questions 
    
    # sample n number of data per question 
    data, missingq = sample_data(processed, pilot_data, n=args.n) 
    print('missing # questions from pilot: ', len(missingq))
    print(f"#Q pilot: {len(pilot_data.keys())}, #Q Experiments: {len(processed.keys())}")  
    
    # Save training data file
    if args.answer_type == 'text':
        # need to cluster answers for free-text responses
        build_training_data(
            ds=data,
            output_jsonl_path=f"{output_dir}/train_agg_{args.n}_blind_inst.jsonl" ,
            client=setup_openai_client()
        )
    elif args.answer_type == 'choice':
        # For multiple choice, no clustering needed - answers are already discrete
        build_training_data(
            ds=data,
            output_jsonl_path=f"{output_dir}/train_agg_{args.n}_blind_inst.jsonl",
            client=None  # No API needed for choice questions
        )
    else:
        raise ValueError(f"Unknown answer_type: {args.answer_type}. Expected 'text' or 'choice'.")

    ### Mixed dataset 
    data1 = read_and_clean_jsonl("/home/work/yuna/HPA/data/training/vqa1k_374.jsonl")
    data2 = read_and_clean_jsonl(f"{output_dir}/train_agg_{args.n}_blind_inst.jsonl")
    combined = data1 + data2
    random.shuffle(combined)
    # Do something with shuffled combined
    print(f"Loaded {len(data1)} + {len(data2)} = {len(combined)} examples. First example:")
    # Save the combined shuffled data to a new JSONL file
    combined_output_path = f"{output_dir}/vqa1k_{args.n}_blind_inst_mixed.jsonl"
    with open(combined_output_path, "w", encoding="utf-8") as fout:
        for ex in combined:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Combined dataset saved to {combined_output_path}")
    # np.random.shuffle(examples) 

if __name__ == '__main__': 
    args = parse_args()
    main(args) 