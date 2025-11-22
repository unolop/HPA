import json 
import torch 
import io 
import pandas as pd  
import os
import numpy as np
import re
import gc
from PIL import Image 
import time
import glob 
import wandb
import argparse 
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
import numpy as np 
from pathlib import Path
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence 
from transformers import Trainer, TrainingArguments 
from torch.utils.data import DataLoader 
import sys 
from dataset.vqav2 import VQADataset
from utils.postprocessor import PostProcessor 
from utils.eval import vqa_accuracy, answer_similarity 
# from huggingface_hub import login
from transformers import BitsAndBytesConfig 
from peft import PeftModel, PeftConfig # get_peft_model, LoraConfig, TaskType,  

# login(token = "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") # expired 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["GRADIO_SHARE"] = "1"
os.environ["WORLD_SIZE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def load_model(args):

    ### Load models 
    if 'internlm3' in args.model: 
        from models.load_lm import internlm3
        model = internlm3(args.model) 
    elif 'internlm2_5' in args.model or 'internlm2' in args.model: 
        from models.load_lm import internlm2_5
        model = internlm2_5(args.model) 
    elif 'Qwen2.5' in args.model: 
        from models.load_lm import Qwen2_5 
        model = Qwen2_5(args.model, max_new_tokens=args.token_length, device=args.gpu) 
    elif 'Qwen3.5' in args.model: 
        from models.load_lm import Qwen3 
        model = Qwen3(args.model) 
    elif 'vicuna' in args.model: 
        from models.load_lm import vicuna 
        model = vicuna(args.model) 
    else: 
        from models.load_lm import mistral
        model = mistral(args.model) 
    return model 

def main(args): 
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache() 
    
    # tokenizer = model.processor  
    model = load_model(args) 
    postprocessor = PostProcessor()
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model 
     
    savedir = f"{args.savedir}" 
    os.makedirs(savedir, exist_ok=True)  
    save_path= os.path.join(savedir, f"{model_name.split('/')[-1]}.jsonl" )
    print('saving to', save_path) 
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    if os.path.exists(save_path) and args.skip_exist: 
        print(save_path, 'exists' ) 
        return  
    else : 
        results = [] 
        
        ### LOADING IMAGES FOR VLMS 
        # image_path="/home/work/yuna/VLMEval/data/val2014" 
        # if blind: 
        #     image = Image.new('RGB', (224, 224), color=(0, 0, 0))      # Black # White  color=(255, 255, 255))  
        # else: 
        #     image = os.path.join(self.image_path, annot['image_id'])
        
        dataset = VQADataset()
        acc = [ ] 

        with open(save_path, 'a') as output_file:
            for i, data in tqdm(enumerate(dataset)):  
                try: 
                    output = model.get_outputs(prompt=data['question']) 
                    data = {**data, **output}
                    postprocess_answer = postprocessor.postprocess_answer(data['text']) 
                    data['postprocess_answer'] = postprocess_answer
                    data['acc'] = max(vqa_accuracy(data["answers"], postprocess_answer), vqa_accuracy(data["answers"], data['text']) )
                    data['score'] = answer_similarity(encoder, postprocess_answer, data['answers']) 
                    acc.append(data['acc']) #data.get('acc'), 0) 
                    print(f"overall acc {np.mean(acc)}% output: {data['text']},  processed_output: {postprocess_answer}, acc: {data['acc']} ")
                    results.append(data) 
                    
                except Exception as e:
                    breakpoint()
                    print(f"Error in model output: {e}, {output}")
                    continue 

                json_line = json.dumps(data)
                output_file.write(json_line + '\n')
                output_file.flush() # Forces buffer contents to be written to disk

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str, default="internlm/internlm3-8b-instruct", help="Model name") 
    parser.add_argument('--skip_exist', action='store_true', help='Enable 4-bit quantization') 
    parser.add_argument('--savedir', default='/home/work/yuna/HPA/results/vqa/LLM', help='Enable 4-bit quantization') 
    parser.add_argument('--token_length', default=None, type=int, help='Black and white') 
    # parser.add_argument("--image_dir_path", type=str, default="/home/work/yuna/VLMEval/datasets", help="Model name") 
    # parser.add_argument('--sample_size', default=1000, type=int)  
    parser.add_argument("--dataset", type=str, default="vqav2", help="Dataset name") 
    parser.add_argument("--gpu", type=str, default=1, help="Dataset name") 
    
    args = parser.parse_args() 
    main(args) 