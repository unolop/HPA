import os 
import re 
from tqdm import tqdm
import json
import numpy as np 
import torch 
import sys
from utils import clean_logprobs
sys.path.append('/home/david/Desktop/yuna/HPA')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.empty_cache() 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def format_prompt(question): 
    return f"Question: {question} Answer the question using a single word or phrase. \nAnswer:" 
 
def load_dataset(data_name:str, prompt:str=''): 
    print("Loading dataset...") 
    if data_name == "vqa_1k": 
        from dataset.vqav2 import VQADataset_json
        dataset = VQADataset_json(
            prompt=prompt,
            image_dir_path="/home/david/Desktop/yuna/data/val2014",
            json_path="/home/david/Desktop/yuna/HPA/dataset/vqa/vqa1k_control.jsonl",
        )
        
    return dataset 


def main(args):

    from swift.llm import (
        PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
    )
    from swift.tuners import Swift
    
    def get_output(args, data, prompt): 

        if 'blind' in args.condition: 
            data['image'] = "/home/david/Desktop/yuna/HPA/dataset/blank_224.png"
            prompt = format_prompt(prompt)
            
        messages = [] 
        
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
        response = resp_list[0].choices[0]
        output_text = response.message.content 
        logprobs_data = response.logprobs  # LogProbs object 

        matches = re.findall(r"Answer\s*:?\s*(.+)", output_text)
        if matches:
            output_text = matches[-1].strip().replace('*', '')
        else:
            output_text = output_text.strip().replace('*', '') 
        return output_text, logprobs_data

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
    request_config = RequestConfig( max_tokens=args.max_token_length,                         
                                logprobs=True,
                                top_logprobs=5, 
                                temperature=0)
    save_name = args.model.split('/')[-1]
    savedir = args.savedir 
    if args.lora_path is not None : 
        finetuned = args.lora_path.split('/')[-4] 
        finetuned += args.lora_path.split('/')[-3] 
        save_name += f'/{finetuned}' 
        savedir += '/finetuned'    
    else: 
        savedir += '/pretrained'     

    output_jsonl_path = f"{savedir}/{save_name}/{args.dataset}_control{args.condition}.jsonl" 
    prompt= ''
    dataset = load_dataset(args.dataset, prompt)  
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for data in tqdm(dataset):
            record_id = data.get('id', 'unknown') 
            generated_answers = {}
            generated_logits = {}

            for k, val in data.items():
                if k in ['id', 'image', 'answers'] or 'id' in k:
                    continue
                
                prompt = format_prompt(val)
                if 'inst' in args.condition:
                    prompt += "\nNote: No images provided..."

                try:
                    output_text, logprobs_data = get_output(args, data, prompt)
                    generated_logits[k] = clean_logprobs(logprobs_data)
                    generated_answers[k] = output_text 
                    
                except Exception as e:
                    print(f"Error processing {record_id} at {k}: {e}")
                    continue
            # breakpoint()
            # Build the final clean output object
            data['answers'] = generated_answers
            data['generated_logits'] = generated_logits
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            f.flush()

    print(f"모든 작업 완료. 결과가 '{output_jsonl_path}'에 저장되었습니다.")

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description="VQA Evaluation") 
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-2B", help="Model name") 
    parser.add_argument("--model_type", type=str, default="vlm")
    parser.add_argument("--resume", action="store_true") 
    parser.add_argument("--max_token_length", default=512) 
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA Path") 
    parser.add_argument("--dataset", type=str, default="vqa_1k", help="Dataset name") 
    parser.add_argument('--checkpoint', type=str, default=None, help='Pretrained checkpoint') 
    parser.add_argument('--savedir', type=str, default="/home/david/Desktop/yuna/HPA/evaluation/logits", help='Save directory of inference') 
    parser.add_argument("--condition", type=str, default='')
    
    args = parser.parse_args() 
    main(args) 