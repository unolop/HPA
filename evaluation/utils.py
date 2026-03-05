import os 
import json 
import torch 
import random 
import numpy as np 
from datasets import load_dataset
 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # cuDNN 관련 설정 (재현성을 위해 필수지만 속도가 느려질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 필요하다면 환경 변수까지 고정 (일부 특수 연산 대비)
    os.environ['PYTHONHASHSEED'] = str(seed)

def format_prompt(prompt, dataset='vqa_1k', condition=''): 

    if 'mmstar' in dataset:
        prompt += f"{prompt}\nProvide only the letter corresponding to the correct choice (A, B, C, or D)."
        
    else:  # spubench elif 'okvqa' textvqa in args.dataset: 
        prompt = f"Question: {prompt} Answer the question using a single word or phrase." 

    ### ADD BLIND INSTRUCTION     
    if 'inst' in condition: 
            prompt = f"{prompt}\nNote: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario."
    
    prompt += "\nAnswer:"

    return prompt  

def get_dataset(data_name:str, prompt:str=''): 
    print("Loading dataset...", data_name)
        
    if data_name == "mmstar":
        dataset = load_dataset("Lin-Chen/MMStar", split="val") 
    
    elif data_name == "spubench":
        dataset = load_dataset("mmbench/MM-SpuBench", split="train", trust_remote_code=True)
        
    elif data_name == "textvqa":
        dataset = load_dataset("lmms-lab/textvqa", split="validation")  
        
    elif data_name == "okvqa":
        dataset = load_dataset("lmms-lab/OK-VQA", split="val2014") 

    elif "vqa_1k" in data_name :
        from dataset.vqav2 import VQADataset_json
        
        if 'control' in data_name: 
            dataset = VQADataset_json(
                prompt=prompt,
                image_dir_path="/home/david/Desktop/yuna/data/val2014",
                json_path="/home/david/Desktop/yuna/HPA/dataset/vqa/vqa1k_control.jsonl",
            )
        else :
            dataset = VQADataset_json(prompt=prompt) 

    elif data_name == "vqa_5k":
        from dataset.vqav2 import VQADataset
        from torch.utils.data import Subset 

        with open('/home/work/yuna/HPA/dataset/s1_qids.json', 'r') as file:
            qids = json.load(file)

        dataset = VQADataset(prompt=prompt, filter_qids=qids) 
        dataset = Subset(dataset, np.random.choice(len(dataset), size=5000, replace=False))

    return dataset 

def skip_processed_idx(existing_keys, output_jsonl_path): 
    id_keys = ['idx', 'qid', 'question_id', 'index']
    current_item_id = None
    processed_ids=set()

    for key in id_keys:
        if key in existing_keys:
            current_item_id = key
            break
    
    if current_item_id is None : 
        current_item_id = 'pid'

    if os.path.exists(output_jsonl_path):
        print(f"{len(processed_ids)} processed {key} exists in: {output_jsonl_path}")
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip() 
                if line:  # Skip empty lines 
                    try:
                        item = json.loads(line)
                        # Assuming the ID field is called 'qid'
                        processed_ids.add(item[current_item_id])
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines  
    return processed_ids, current_item_id

def clean_logprobs(logprobs_data):
    return {
        "content": [
            {
                "token": t['token'],
                "logprob": t['logprob'],
                "top_logprobs": [
                    {"token": tl['token'], "logprob": tl['logprob']}
                    for tl in t['top_logprobs']
                ]
            }
            for t in logprobs_data['content']
        ]
    }
