
import torch, os, json, math, time, logging, random
from typing import Dict, List
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def set_seed(seed: int=123):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
