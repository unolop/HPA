import numpy as np 
import re  
import pandas as pd 
from glob import glob 
import sys 
sys.path.append('/home/work/yuna/HPA') 
from preprocessing.utils import MODELNAMES 
from .vqa import score_number, score_yes_no, VQAAnswerMapper, vqa_accuracy 

root_dir = '/home/work/yuna/HPA/evaluation/scored'

MODEL_DISPLAY_MAP = {
    # ---------- Humans ----------
    "Humans": "Humans",
    'Qwen3-0.6B' : 'Qwen3 (0.6B)',
    'Qwen3-4B' : 'Qwen3 (4B)' , 
    'Qwen3-8B-Base': 'Qwen3 Base (8B)', 
    'Qwen3-8B' : 'Qwen3 (8B)',  
    
    # ---------- InternVL ----------
    "InternVL3_5-1B": "InternVL 3.5 (1B)",
    "InternVL3_5-2B": "InternVL 3.5 (2B)",
    "InternVL3_5-4B": "InternVL 3.5 (4B)",
    "InternVL3_5-8B": "InternVL 3.5 (8B)",

    "InternVL3_5-8B_A1_vqa_gt": "InternVL 3.5 (8B) | JS VQA (GT)",
    "InternVL3_5-8B_A2_vqa_10_blind_inst": "InternVL 3.5 (8B) | JS‑Blind VQA (n=10)",
    "InternVL3_5-8B_A3_vqa_15_blind_inst": "InternVL 3.5 (8B) | JS‑Blind VQA (n=15)",
    "InternVL3_5-8B_A4_mmstar_15_blind_inst": "InternVL 3.5 (8B) | JS‑Blind MMStar (n=15)",
    "InternVL3_5-8B_SFT_vqa_15_blind_inst": "InternVL 3.5 (8B) | SFT‑Blind VQA",
    "InternVL3_5-8B_SFT_vqa_gt": "InternVL 3.5 (8B) | SFT‑VQA",
    'InternVL3_5-8B_SFT_mmstar_15_blind_inst' : "InternVL 3.5 (8B) | SFT‑Blind MMStar (n=15)", 

    # ---------- Qwen ----------
    "Qwen3-VL-2B-Instruct": "Qwen3‑VL (2B)",
    "Qwen3-VL-4B-Instruct": "Qwen3‑VL (4B)",
    "Qwen3-VL-8B-Instruct": "Qwen3‑VL (8B)",

    "Qwen3-VL-4B-Instruct_A1_vqa_gt": "Qwen3‑VL (4B) | JS VQA (GT)",
    "Qwen3-VL-4B-Instruct_A2_vqa_10_blind_inst": "Qwen3‑VL (4B) | JS‑Blind VQA (n=10)",
    # "Qwen3-VL-4B-Instruct_A1_JS_vqa_n15_blind_inst": "Qwen3‑VL‑4B | JS‑Blind VQA (n=15)", 
    "Qwen3-VL-4B-Instruct_A3_vqa_15_blind_inst": "Qwen3‑VL (4B) | JS‑Blind VQA (n=15)",
    "Qwen3-VL-4B-Instruct_A4_mmstar_15_blind_inst": "Qwen3‑VL (4B) | JS‑Blind MMStar (n=15)",
    "Qwen3-VL-4B-Instruct_SFT_vqa_15_blind_inst": "Qwen3‑VL (4B) | SFT‑Blind VQA",
    "Qwen3-VL-4B-Instruct_SFT_vqa_gt": "Qwen3‑VL (4B) | SFT‑VQA",

    "Qwen3-VL-8B-Instruct_A1_vqa_gt": "Qwen3‑VL (8B) | JS VQA (GT)",
    "Qwen3-VL-8B-Instruct_A2_vqa_10_blind_inst": "Qwen3‑VL (8B) | JS‑Blind VQA (n=10)",
    "Qwen3-VL-8B-Instruct_A3_vqa_15_blind_inst": "Qwen3‑VL (8B) | JS‑Blind VQA (n=15)",
    "Qwen3-VL-8B-Instruct_A4_mmstar_15_blind_inst": "Qwen3‑VL (8B) | JS‑Blind MMStar (n=15)",
    "Qwen3-VL-8B-Instruct_SFT_mmstar_15_blind_inst": "Qwen3‑VL (8B) | SFT‑Blind MMStar",
    "Qwen3-VL-8B-Instruct_SFT_vqa_15_blind_inst": "Qwen3‑VL (8B) | SFT‑Blind VQA",
    "Qwen3-VL-8B-Instruct_SFT_vqa_gt_sft": "Qwen3‑VL (8B) | SFT‑VQA",
    # ---------- LLaVA ----------
    "llava-v1.6-mistral-7b-hf": "LLaVA‑Mistral (7B)",
    "llava-v1.6-vicuna-7b-hf": "LLaVA‑Vicuna (7B)",
    "llava-1.5-7b-hf": "LLaVA‑1.5 (7B)",

    "llava-v1.6-mistral-7b-hf_A1_vqa_gt": "LLaVA‑Mistral (7B) | JS VQA (GT)",
    "llava-v1.6-mistral-7b-hf_A2_vqa_10_blind_inst": "LLaVA‑Mistral (7B) | JS‑Blind VQA (n=10)",
    "llava-v1.6-mistral-7b-hf_A3_vqa_15_blind_inst": "LLaVA‑Mistral (7B) | JS‑Blind VQA (n=15)",
    "llava-v1.6-mistral-7b-hf_A4_mmstar_15_blind_inst": "LLaVA‑Mistral (7B) | JS‑Blind MMStar (n=15)",
    "llava-v1.6-mistral-7b-hf_SFT_mmstar_15_blind_inst": "LLaVA‑Mistral (7B) | SFT‑Blind MMStar",
    "llava-v1.6-mistral-7b-hf_SFT_vqa_15_blind_inst": "LLaVA‑Mistral (7B) | SFT‑Blind VQA",
    "llava-v1.6-mistral-7b-hf_SFT_vqa_gt": "LLaVA‑Mistral (7B) | SFT‑VQA",
}

def get_encoder():
    """Lazy load sentence transformer."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2").to('cuda')
    except:
        return None
    

def compute_similarity(gt: str, pred: str, encoder) -> float:
    if encoder is None or not gt or not pred:
        return 0.0

    emb = encoder.encode(
        [pred.strip(), gt.strip()],
        normalize_embeddings=True
    )

    sim = float((emb[0] @ emb[1]))
    # print(gt, pred, sim) 
    return float(np.clip(sim, -1.0, 1.0)) 
    
def find_matching(f, targets): 
    for t in targets : 
        if t in f : 
            f = f.replace(f'{t}', '')   
            print(f'found {t} in {f}')
            return t, f 
    print(f"cannot find matching {f} in {targets}") 

def get_summary(dataset='mmstar'): 
    files = glob(f"{root_dir}/*/*{dataset}*.jsonl")  + glob(f"{root_dir}/*/*/*/*{dataset}*.jsonl") + glob(f"{root_dir}/*/*/*{dataset}*.jsonl") 
    
    dfs= []
    for f in files: 
        try: 
            df = pd.read_json(f, lines=True)
            df['filename'] = f 
            if 'finetuned' in f : 
                df['model'] = f.split('/')[-2].replace('fold_0', '')
                df['trained_dataset'] = 'VQA' if 'vqa' in f.split('/')[-2].lower() else 'MMStar'
                df['strategy'] = 'SFT' if 'sft' in f.lower() else 'JS' 
                df['blind'] = 'Blind' if 'blind' in f.lower() else 'GT' 
                
            else: 
                df['model'], f = find_matching(f, [model.split('/')[-1] for model in MODELNAMES])  
                df['trained_dataset'] = 'Pretrained'
                df['strategy'] = 'Pretrained'
                df['blind'] = 'Pretrained'
            df['condition'] = f.split('/')[-1][:-6].replace(f'_', ' ').replace('vqa 1k', '').replace('vqa 5k', '').replace(f'{dataset}', '').strip()
            dfs.append(df)
        except Exception as e: 
            print(e)

    df = pd.concat(dfs)
        
    df['correct'] = pd.to_numeric(df['correct'], errors='coerce')
    df['correct'] = (df['correct'] * 100).round(1)
    
    if "answer_similarity" in df.columns: 
        df['answer_similarity'] = pd.to_numeric(df['answer_similarity'], errors='coerce')
        df['answer_similarity'] = (df['answer_similarity'] * 100).round(1)
        
    if 'meta_model' not in df.keys(): 
        df['meta_model'] = 'open-source model'
    
    df['model_raw'] = df['model']
    # df['model'] = df['model'].map(MODEL_DISPLAY_MAP)  
    unmapped = (
        df.loc[df['model'].isna(), 'model']
        .unique()
        )

    print(unmapped) 
    return df 
    
def extract_mc_choice(output: str) -> str:
    """Extract the predicted answer (A, B, C, D) from model output."""
    if not output:
        return ""

    # Remove <think>...</think> content if present
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL | re.IGNORECASE)
    output = output.strip()

    # Pattern 1: Look for explicit answer statements
    patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is[:\s]*([A-D])",
        r"(?:the\s+)?(?:correct\s+)?answer[:\s]*([A-D])",
        r"(?:option\s+)?([A-D])\s+is\s+(?:the\s+)?correct",
        r"(?:I\s+)?(?:would\s+)?choose\s+(?:option\s+)?([A-D])",
        r"(?:I\s+)?(?:would\s+)?select\s+(?:option\s+)?([A-D])",
        r"^([A-D])(?:[:\.\)]|\s|$)",  # Answer at the start
        r"\n([A-D])(?:[:\.\)]|\s|$)",  # Answer after newline
        r"(?:Therefore|Thus|So|Hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(?:option\s+)?([A-D])",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Pattern 2: Last capital letter A-D
    matches = re.findall(r'\b([A-D])\b', output)
    if matches:
        return matches[-1].upper()

    # Pattern 3: Look for choice at end
    if output and output[-1].upper() in 'ABCD':
        return output[-1].upper()

    # Pattern 4: Check format like "A: Hanging Posters"
    match = re.match(r'^([A-D]):', output)
    if match:
        return match.group(1).upper()

    return ""


def find_id_key(existing_keys): 
    current_item_id = None
    id_keys = ['idx', 'qid', 'question_id', 'index']
    
    for key in id_keys:
        if key in existing_keys:
            current_item_id = key
            break

    if current_item_id is None : 
        current_item_id = 'pid'

    return current_item_id 

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (open-ended)."""
    if not answer:
        return ""
    answer = str(answer)

    # Remove <think>...</think> content if present
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL | re.IGNORECASE)

    answer = answer.lower().strip()
    # Remove articles
    for article in ['a ', 'an ', 'the ']:
        if answer.startswith(article):
            answer = answer[len(article):]
    # Remove punctuation
    import string
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(answer.split()).strip()

def mc_accuracy(gt: str, pred: str) -> bool:
    """Multiple choice accuracy."""
    gt_letter = gt.strip().upper()[0] if gt else ""
    pred_letter = extract_mc_choice(pred)
    return gt_letter == pred_letter

def exact_match(gt: str, pred: str) -> bool:
    """Exact match after normalization."""
    return normalize_answer(pred) == normalize_answer(gt) 