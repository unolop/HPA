import os 
import re 
from tqdm import tqdm
import json
import numpy as np 
import torch
from utils import skip_processed_idx  
import sys 
sys.path.append('/home/work/yuna/HPA') 

system_message = "Note: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n"

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.empty_cache() 

DATASET_TYPE = {
    'mmstar': 'multi-choice', 
    # 'mmstar': 'multi-choice', 
    # 'mmstar': 'multi-choice', 
}

def build_prompt(line, for_llm=False): 
    prompt = line['question'] 
    if DATASET_TYPE(dataset) == 'multi-choice':
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
    elif DATASET_TYPE(dataset) == 'Y/N' and for_llm:
        prompt += 'Answer the question using a single word or phrase. \n'

    return dict(image=tgt_path, text=prompt)
    
def load_dataset(data_name:str, prompt:str=''): 
    print("Loading dataset...")
    
    if data_name == "mmstar":
        from datasets import load_dataset 
        dataset = load_dataset("Lin-Chen/MMStar", split="val") 
    
    elif data_name == "spubench":
        from datasets import load_dataset

        # Load annotations
        with open('/home/work/yuna/HPA/dataset/annotation.json', 'r', encoding='utf-8') as f:
            annot = json.load(f)

        dataset = []

        # Use non-streaming mode for faster loading (downloads once and caches)
        print("  Loading MM-SpuBench dataset from HuggingFace (will cache locally)...")
        ds = load_dataset("mmbench/MM-SpuBench", split="train", trust_remote_code=True)

        print(f"  Loaded {len(ds)} images, combining with {len(annot)} annotations...")

        for data, ann in zip(ds, annot):
            image = data['image']
            question = ann['question']
            choices = ann['choices']
            choices_text = "\n".join(choices)
            question = (
                f"Question: {question}\n{prompt}"
                f"{choices_text}\n"
                "Provide only the letter corresponding to the correct choice (A, B, C, or D).\n"
                "Answer:"
            )
            ann['question'] = question
            ann['image'] = image

            dataset.append(ann)

        print(f"  ✓ Prepared {len(dataset)} examples") 

    elif data_name == "vqa_1k":
        from dataset.vqav2 import VQADataset_json
        dataset = VQADataset_json(prompt=prompt)
        
    elif data_name == "vqa_5k":
        from dataset.vqav2 import VQADataset
        from torch.utils.data import Subset

        with open('/home/work/yuna/HPA/dataset/s1_qids.json', 'r') as file:
            qids = json.load(file)

        dataset = VQADataset(prompt=prompt, filter_qids=qids) 
        dataset = Subset(dataset, np.random.choice(len(dataset), size=5000, replace=False))

    return dataset 

def main(args):

    from swift.llm import (
        PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
    )
    from swift.tuners import Swift
    # Please adjust the following lines
    model = args.model

    template_type = None  # None: use the default template_type of the corresponding model
    default_system = None  # None: use the default system prompt of the corresponding model

    # Load model and dialogue template
    model, tokenizer = get_model_tokenizer(model, use_hf=True) #, max_pixels=448)
    if args.lora_path is not None:
        lora_checkpoint = safe_snapshot_download(args.lora_path)  # Change to your checkpoint_dir
        model = Swift.from_pretrained(model, lora_checkpoint)
    model.eval()
    template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system)
    engine = PtEngine.from_model_template(model, template, max_batch_size=1)
    request_config = RequestConfig(max_tokens=args.max_token_length, temperature=0)
    
    save_name = args.model.split('/')[-1]
    
    if args.lora_path is not None : 
        finetuned = args.lora_path.split('/')[-3] 
        save_name += f'/{finetuned}' 
    output_jsonl_path = f"{args.savedir}/{save_name}/{args.dataset}{args.condition}.jsonl"
    prompt= ''

    if 'blind' in args.condition : 
        print('blind condition')
        if 'inst' in args.condition and 'sys' not in args.condition:  
            print('instructions appending in vqav2 question')
            prompt = '\n' + system_message 
            
    dataset = load_dataset(args.dataset, prompt) 
    try: 
        processed_ids, current_item_id = skip_processed_idx(existing_keys=dataset[0].keys(), output_jsonl_path=output_jsonl_path)
    except Exception as e: 
        print(e)
        processed_ids = set()

    if not args.resume: 
        processed_ids = set()
        write_mode = 'w'
    else :
        write_mode = 'a' 

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, write_mode, encoding='utf-8') as f:
        for i in tqdm(range(len(dataset))):
            data = dataset[i] 
            data['pid'] = i 
            prompt = data['question'] 

            if 'blind' in args.condition: 
                data['image'] = "/home/work/yuna/HPA/data/blank_224.png"
                
            if processed_ids is not None: 
                if data[current_item_id] in processed_ids:
                    continue 
                
            if 'inst' in args.condition and 'mmstar' in args.dataset:
                prompt += system_message

            if 'vqa' not in args.dataset:
                prompt += "\nProvide only the letter corresponding to the correct choice (A, B, C, or D).\nAnswer:"
                # prompt += "\nAnswer with the option's letter from the given choices directly, such as answer letter 'A' only. \n"
            # if args.model_type == 'vlm' and 'vqa' not in args.dataset: # VLM MCQ  
                # prompt += '\n select the correct answer from the options above.'
        
            messages = [] 
            if 'sys_inst' in args.condition: 
                messages.append({'role': 'system', 'content': system_message}) 
            
            if args.model_type == 'vlm':  
                messages.append({'role': 'user', 'content': f'<image>{prompt}' })
                
                infer_request = InferRequest(
                    messages=messages,
                    images=[data['image']] 
                )

            else : 
                messages.append({'role': 'user', 'content': prompt})
                infer_request = InferRequest(
                    messages=messages 
                ) 

            resp_list = engine.infer([infer_request], request_config)
            output_text = resp_list[0].choices[0].message.content 
        
            matches = re.findall(r"Answer\s*:?\s*(.+)", output_text)
            if matches:
                output_text = matches[-1].strip().replace('*', '')
            else:
                output_text = output_text.strip().replace('*', '')

            # 6. 결과를 JSON 객체로 생성하고 JSONL에 즉시 작성
            data['output'] = output_text 
            data['question'] = prompt 
            print('Q:', prompt, 'Output:', data['output'], 'Ans:', output_jsonl_path)

            data.pop('image')
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            f.flush()  # 버퍼를 비워 파일에 즉시 쓰도록 강제

    print(f"모든 작업 완료. 결과가 '{output_jsonl_path}'에 저장되었습니다.")

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description="VQA Evaluation") 
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-2B", help="Model name") 
    parser.add_argument("--model_type", type=str, default="vlm")
    parser.add_argument("--resume", action="store_true") 
    parser.add_argument("--max_token_length", default=512) 
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA Path") 
    parser.add_argument("--dataset", type=str, default="mmstar", help="Dataset name") 
    parser.add_argument('--checkpoint', type=str, default=None, help='Pretrained checkpoint') 
    parser.add_argument('--savedir', type=str, default="/home/work/yuna/HPA/data/models", help='Save directory of inference') 
    parser.add_argument("--condition", type=str, default='')
    
    args = parser.parse_args() 
    main(args) 