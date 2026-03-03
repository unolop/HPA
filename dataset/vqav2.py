import os 
import json 
from torch.utils.data import Dataset 

class VQADataset(Dataset):
    def __init__(self, 
                 image_dir_path="/home/work/yuna/data/data/val2014", 
                 question_path="/home/work/yuna/data/data/v2_OpenEnded_mscoco_val2014_questions.json", 
                 annotations_path="/home/work/yuna/data/data/v2_mscoco_val2014_annotations.json", 
                 prompt='',
                 format=True, 
                 filter_qids=[],
                ):  
        
        self.image_dir_path = image_dir_path
        self.prompt = prompt
        self.format = format 
        
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
        if self.format: 
            ann['question'] = f"Question: {question} Answer the question using a single word or phrase. {self.prompt}\nAnswer:" 
        else: 
            ann['question'] = question
        
        padded_id = str(ann['image_id']).zfill(12)
        filename = f"COCO_val2014_{padded_id}.jpg"
        image = os.path.join(self.image_dir_path, filename)
        ann = {**q, **ann} 
        ann['image'] = image

        return ann 
    
    def get_by_qid(self): # TODO: A bit repetitive with the getitem 
        """Return a dictionary of all annotations keyed by question_id."""
        qid_to_ann = {}
        
        for q, ann in zip(self.questions, self.annotations):
            qid = q.get('question_id') or ann.get('question_id')
            if qid is None:
                continue
                
            question = q['question']
            ann['question'] = f"Question: {question} Answer the question using a single word or phrase. {self.prompt}\nAnswer:" 
            padded_id = str(ann['image_id']).zfill(12)
            filename = f"COCO_val2014_{padded_id}.jpg"
            image = os.path.join(self.image_dir_path, filename)
            processed_ann = {**q, **ann}
            processed_ann['image'] = image
            qid_to_ann[int(qid)] = processed_ann
        
        return qid_to_ann 

class VQADataset_json(Dataset):
    def __init__(self, 
                image_dir_path="/home/david/Desktop/yuna/data/val2014", 
                json_path="/home/david/Desktop/yuna/HPA/dataset/vqa/vqav2_1k_val.json",
                prompt=''): 
        self.prompt = prompt 
        self.image_dir_path=image_dir_path
        
        if json_path.endswith('json'): 
            with open(json_path, 'r') as f:
                self.questions = json.load(f)
        else: 
            with open(json_path, 'r', encoding='utf-8') as f: 
                self.questions = [json.loads(line) for line in f if line.strip()]  
    
    def __len__(self):
        return len(self.questions) 

    def __getitem__(self, idx): 
        annot = self.questions[idx] 
        annot['question'] = self.prompt + self.questions[idx]['question'] 

        image_id = annot.get('image_id')
        if isinstance(image_id, int):
            padded_id = str(image_id).zfill(12)
            filename = f"COCO_val2014_{padded_id}.jpg" 
        else: 
            filename = str(image_id) 
        annot['image'] = os.path.join(self.image_dir_path, filename) 
        return annot 