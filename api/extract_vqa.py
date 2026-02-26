import json 
from prompts import *
from gpt_utils import call_api, make_batches 
from ..dataset.vqav2 import VQADataset_json, VQADataset
from ..dataset.paths import VQA_IMAGE_DIR, VQA_QUESTIONS, VQA_ANNOT, VQA_1K 

BATCH_SIZE = 50 

def get_vqa_questions(dataset='VQA_1K'): 
    vqa1k = VQADataset_json(prompt='', \
                            image_dir_path=VQA_IMAGE_DIR, \
                            json_path=VQA_1K) 
    
    dataset = VQADataset( image_dir_path=VQA_IMAGE_DIR, 
                question_path=VQA_QUESTIONS, 
                annotations_path=VQA_ANNOT, 
                prompt='')  
                
    qids = [d['question_id'] for d in vqa1k]  # get the qids from subset 
    questions = [q for q in dataset.questions if q['question_id'] in qids]  # get the original questions 
    print(f"loaded {len(questions)} questions from VQA validation 1k \ne.g.")
    print(questions[0]) 

    return questions 

def process(questions, mode='CTL', output_path: str = "vqa_extracted.jsonl", batch_size: int = BATCH_SIZE):
    output = []
    batches = make_batches(questions, hard_max_items=batch_size) 

    if mode == 'CTL': 
        system = SYSTEM_CTL
        schema= CTL_SCHEMA 
    else: 
        system = SYSTEM_SEM 
        schema= SEM_SCHEMA  

    with open(output_path, "a", encoding="utf-8") as f:
        for batch in batches:  
            # Extract questions and call API
            questions_text = [q['question'] for q in batch]
            ctl_results = call_api(system, schema, "vqa_ctl_v1", questions_text) 
            
            # Merge results
            current_batch_processed = [
                {**s, **c} for s, c in zip(batch, ctl_results)
            ]
            
            # Write only the NEW items to the file immediately
            for item in current_batch_processed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            output.extend(current_batch_processed)
            print(f"Saved {len(output)} total items (Batch size: {len(batch)})")
            
    return output  

def main(args): 

    questions = get_vqa_questions() 
    mode=args.mode  
    output = process(questions, mode, output_path=f"./dataset/vqa1k_{mode}.jsonl" )   # control semantics 
    
    return output 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["SEM", "CTL"], help="Extraction mode: SEM or CTL")
    args = parser.parse_args()
    main(args)
