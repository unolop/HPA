import re 
import os 
import json 
import glob 
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset 
import random 
import pandas as pd 
from tqdm import tqdm 


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class VQADataset(Dataset):
    def __init__(self, 
                json_path="/home/work/yuna/HPA/data/vqav2_1k/val.json"): # /home/work/yuna/HPA/data/vqav2_val.json"): 
        
        with open(os.path.join(json_path), 'r') as f:
            self.questions = json.load(f)
        
    def __len__(self):
        return len(self.questions) 

    def __getitem__(self, idx): 
        annot = self.questions[idx] 
        return annot 