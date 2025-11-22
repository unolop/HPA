import io 
import os
import numpy as np
from PIL import Image 
import time
import glob 
import wandb
import json 
import argparse 
from tqdm import tqdm 
from pathlib import Path
from torch.utils.data import Subset
from torch.utils.data import DataLoader 
import sys 
from dataset.vqav2 import VQADataset
from utils.postprocessor import PostProcessor 
import sys 
from huggingface_hub import login
from transformers import BitsAndBytesConfig 
from peft import PeftModel, PeftConfig # get_peft_model, LoraConfig, TaskType,  
from sentence_transformers import SentenceTransformer
import numpy as np  
from utils.tokens import HFTOKEN

login(token = HFTOKEN)
file_path = Path(__file__)
root_dir = file_path.parent  # one level up
IMAGE_SIZE = 224
seed = 42

def load_model(args):

    ### Load models 
    if 'InternVL' in args.model: 
        from models.load_vlm import InternVL
        model = internlm3(args.model) 

def answer_similarity(encoder, answer, gt:list): 
    avg = [] 
    for gta in gt : 
        gta = gta['answer'] 
        embeddings = encoder.encode([answer, gta]) 
        # print(embeddings.shape) # [3, 384]
        similarities = encoder.similarity(embeddings, embeddings)
        avg.append(similarities[1,0]) 
        # print(similarities) 

    return np.mean(avg) 

def main(args): 
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}" 
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.empty_cache() 

    ### Load models 
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model 
    model = load_model(args) 
    postprocessor = PostProcessor() 
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    checkpoint = '_'.join(args.checkpoint.split('/')) if args.checkpoint is not None else 0 
    prefix = f"-{checkpoint}" if args.checkpoint else "" 
    prefix += "-blind" if args.blind else "" 
    savedir = f"{args.savedir}" 
    os.makedirs(savedir, exist_ok=True)  
    save_path = os.path.join(savedir, f"{args.dataset.split('/')[-1]}_{IMAGE_SIZE}_{model_name}{prefix}.jsonl" )
    print(save_path) 
    
    if os.path.exists(save_path) : 
        if args.skip_exist: 
            return 
        else: 
            with open(save_path, 'r', encoding='utf-8') as f:
                unique_ids = set()
                for line in f:
                    data = json.loads(line)
                    unique_ids.add(data['qid']) 
    results = []
            
    if args.dataset == "Lin-Chen/MMStar": 
        from datasets import load_dataset 
        dataset = load_dataset(args.dataset, split="val") 

        for data in tqdm(dataset): 
            image = Image.new('RGB', (224, 224), color=(0, 0, 0)) if args.blind else data['image'] 
            output = model.get_outputs(image=image, prompt=build_prompt(data['question'], dataset='mcq'))
            data['output'] = output 
            print(data['answer'], output, postprocessor.postprocess_answer(output), save_path) 
            data['processed_output'] = postprocessor.postprocess_answer(output)
            del data['image']  
            results.append(data)
            pd.DataFrame(results).to_csv(save_path, index=False)
            
    else:  ### VQA LOOP  
        dataset = VQADataset() 
        for i, data in tqdm(enumerate(dataset)):  
            if args.blind: 
                data['image'] = Image.new('RGB', (224, 224), color=(0, 0, 0))      # Black # White  color=(255, 255, 255))  
            
            # skip existign inference 
            qid = int(data.get("question_id", "").strip()) 
            if qid in unique_ids: 
                continue 

            with open(save_path, "w", encoding="utf-8") as fout:

                try: 
                    data['output'] = model.get_outputs(image=data['image'], prompt=data['question'])
                    data['processed_output'] = postprocessor.postprocess_answer(output) 
                    data['acc'] = vqa_score(processed_output, data['answers']) 
                    data['score'] = answer_similarity(processed_output, data['answers'])
                    results.append(data)  
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    print(f"output: {output},  processed_output: {postprocess_answer}, acc: {acc} Saved JSONL to: {save_path}")
                    
                except Exception as e:
                    print(f"Error in model output: {e}, {output}")
                    continue 

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name") 
    parser.add_argument("--image_dir_path", type=str, default="/home/work/yuna/VLMEval/datasets", help="Model name") 
    parser.add_argument('--checkpoint', default=None, help='Black and white')  
    parser.add_argument("--gpu", type=str, default="0") 
    parser.add_argument('--quantize', action='store_true', help='Enable 4-bit quantization') 
    parser.add_argument('--blind', action='store_true', help='Enable 4-bit quantization') 
    parser.add_argument('--skip_exist', action='store_true', help='Enable 4-bit quantization') 
    parser.add_argument('--custom_prompt', default=None, help='Enable 4-bit quantization') 
    parser.add_argument('--savedir', default='results', help='Enable 4-bit quantization') 
    parser.add_argument('--sample_size', default=1000, type=int)  
    parser.add_argument('--token_length', default=None, type=int, help='Black and white') 
    parser.add_argument("--dataset", type=str, default="vqav2", help="Dataset name") 
    
    args = parser.parse_args() 
    main(args) 