import os 
import re 
import csv
from tqdm import tqdm
from typing import List, Union
from PIL import Image
import json
import torch


def load_dataset(data_name:str):
    print("Loading dataset...")
    
    if data_name == "mmstar":
        from datasets import load_dataset 
        dataset = load_dataset("Lin-Chen/MMStar", split="val") 
        return dataset 

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
    request_config = RequestConfig(max_tokens=128, temperature=0)
    
    dataset = load_dataset(args.dataset) # , y_true, prompt 

    save_name = args.lora_path.split('/')[-3] if args.lora_path else args.model.split('/')[-1]
    output_jsonl_path = f"{args.savedir}/{save_name}_{args.dataset}.jsonl"
    
    id_keys = ['idx', 'qid', 'question_id', 'index']
    current_item_id = None

    for key in id_keys:
        if key in dataset[0].keys():
            current_item_id = key
            break

    if current_item_id is None : 
        current_item_id = 'qid'
        print(f'created new key = "qid" on {dataset[0].keys()}') 

    processed_ids = set()
    if os.path.exists(output_jsonl_path):
        print(f"'{output_jsonl_path}' exists.")
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
        print(f"총 {len(processed_ids)}개의 항목을 건너뜁니다.")

    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
        for i in tqdm(range(len(dataset))):
            data = dataset[i] 
            question = data['question'] 
            # 5. 중복 검사 및 건너뛰기
            try: 
                if processed_ids is not None: 
                    if data[current_item_id] in processed_ids:
                        continue 
            except Exception as e : 
                breakpoint()
            
            infer_request = InferRequest(
                messages=[{'role': 'user', 'content': f'<image>{question}'}],
                images=[data['image']]
            )

            resp_list = engine.infer([infer_request], request_config)
            output_text = resp_list[0].choices[0].message.content
            
            # 6. 결과를 JSON 객체로 생성하고 JSONL에 즉시 작성
            data['output'] = output_text 
            data.pop('image')
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            f.flush()  # 버퍼를 비워 파일에 즉시 쓰도록 강제

    print(f"모든 작업 완료. 결과가 '{output_csv_path}'에 저장되었습니다.")

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description="VQA Evaluation") 
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-2B", help="Model name") 
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA Path") 
    parser.add_argument("--dataset", type=str, default="mmstar", help="Dataset name") 
    parser.add_argument('--checkpoint', type=str, default=None, help='Pretrained checkpoint') 
    parser.add_argument('--savedir', type=str, default="/home/work/yuna/HPA/results/swift", help='Save directory of inference') 
    parser.add_argument("--gpu", type=str, default="0")
    
    args = parser.parse_args() 
    main(args) 