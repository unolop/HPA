import re 
import os 
import json 
import glob 
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset 

class VQADataset(Dataset):
    def __init__(self, 
                 image_dir_path="/home/work/yuna/VLMEval/data/val2014", 
                 question_path="/home/work/yuna/VLMEval/data/v2_OpenEnded_mscoco_val2014_questions.json", 
                 annotations_path="/home/work/yuna/VLMEval/data/v2_mscoco_val2014_annotations.json", 
                 prompt='', 
                 filter_qids=[],
                ):  
        
        self.image_dir_path = image_dir_path
        self.prompt = prompt
        
        with open(question_path, 'r') as f:
            questions_data = json.load(f)
        self.questions = questions_data['questions']
    
        # Load annotations if provided
        if annotations_path is not None:
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)['annotations'] 

        if len(filter_qids): 
            self.annotations, self.questions = self.filter_dataset(filter_qids) 

    def filter_dataset(self, qids): 
        
        anns = []
        qs = [] 
        skipped=[] 
        
        for ann, q in zip(self.annotations, self.questions): 
            ### !!! question_id has to be int 
            if ann['question_id'] in qids: 
                skipped.append(ann['question_id'])
                continue 
            else: 
                anns.append(ann)
                qs.append(q)
        print( 'Skipped #',len(skipped))
        return anns, qs 
        
    def __len__(self):
        return len(self.questions) 

    def __getitem__(self, idx):
        q = self.questions[idx]
        ann = self.annotations[idx]

        question = q['question']
        question = f"Question: {question} Answer the question using a single word or phrase. \nAnswer:" 
        ann['question'] = self.prompt + question
        padded_id = str(ann['image_id']).zfill(12)
        filename = f"COCO_val2014_{padded_id}.jpg"
        image = os.path.join(self.image_dir_path, filename)
        ann = {**q, **ann} 
        ann['image'] = image
        return ann 

class VQADataset_json(Dataset):
    def __init__(self, 
                image_dir_path="/home/work/yuna/VLMEval/data/val2014", 
                json_path="/home/work/yuna/HPA/data/vqav2_1k_val.json",
                prompt=''): 
        self.prompt = prompt 
        self.image_dir_path=image_dir_path
        with open(os.path.join(json_path), 'r') as f:
            self.questions = json.load(f)
        
    def __len__(self):
        return len(self.questions) 

    def __getitem__(self, idx): 
        annot = self.questions[idx] 
        annot['question'] = self.prompt + self.questions[idx]['question']  
        annot['image'] = os.path.join(self.image_dir_path, annot['image_id'])
        return annot 