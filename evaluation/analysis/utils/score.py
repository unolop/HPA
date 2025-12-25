import os 
import numpy as np 
import re  
import json 
from utils.df import MODEL_DISPLAY_MAP 
from typing import List 
import pandas as pd 
from glob import glob 
import sys 
sys.path.append('/home/work/yuna/HPA') 
from preprocessing.utils import MODELNAMES 

root_dir = '/home/work/yuna/HPA/evaluation/scored'
VQA_ANNOTATIONS_PATH = "/home/work/yuna/VLMEval/data/v2_mscoco_val2014_annotations.json"

def get_encoder():
    """Lazy load sentence transformer."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2").to('cuda')
    except:
        return None
    
encoder=get_encoder() 

def compute_similarity(gt: str, pred: str, encoder=encoder) -> float:
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
            return t, f 
    print(f"cannot find matching {f} in {targets}") 

def get_summary(dataset='mmstar'): 
    files = glob(f"{root_dir}/*/*{dataset}*.jsonl")  + glob(f"{root_dir}/*/*/*/{dataset}*.jsonl") + glob(f"{root_dir}/*/*/{dataset}*.jsonl") 

    dfs= []
    for f in files: 
        try: 
            df = pd.read_json(f, lines=True)
            if 'finetuned' in f : 
                df['model'] = f.split('/')[-2].replace('fold_0', '')
            else: 
                df['model'], f = find_matching(f, [model.split('/')[-1] for model in MODELNAMES])  
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
        
    print(len(files) ) 
    # pt = df.pivot_table(
    #     index=['model'],  
    #     columns=['condition'], 
    #     values=['correct'],
    #     aggfunc=['mean', 'count']
    # )
    # pt = pt.round(4)
    df['model'] = df['model'].map(MODEL_DISPLAY_MAP)  
    df = df.drop_duplicates(subset=['condition', 'question_id', 'model'])  
    # pt.to_csv(f"./tables/summary_{dataset}.csv")
    return df 
    
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


def vqa_accuracy(gt_answers: List[str], pred: str) -> float:
    """VQA accuracy: min(1, #matches / 3)."""
    pred = normalize_answer(pred)
    matches = sum([pred == normalize_answer(ans) for ans in gt_answers])
    return min(1.0, matches / 3.0)


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





class VQAAnswerMapper:
    """
    Maps question_id to list of ground truth answers from VQA annotations.

    Usage:
        mapper = VQAAnswerMapper()
        answers = mapper.get_answers(123456)  # Returns list of 10 annotator answers
    """

    def __init__(self, annotations_path: str = VQA_ANNOTATIONS_PATH):
        self.annotations_path = annotations_path
        self._qid_to_answers = None  # Lazy load
        self._loading = False  # Prevent recursive loading

    def _load(self):
        """Load annotations and build lookup dict (lazy loading)."""
        if self._qid_to_answers is not None:
            return

        if self._loading:
            return

        self._loading = True

        # Check if file exists before trying to load
        if not os.path.exists(self.annotations_path):
            print(f"⚠️  VQA annotations not found: {self.annotations_path}")
            self._qid_to_answers = {}
            self._loading = False
            return

        print(f"📂 Loading VQA annotations (this may take a moment)...")

        try:
            with open(self.annotations_path, 'r') as f:
                data = json.load(f)

            annotations = data.get('annotations', data)

            # Build qid -> answers lookup
            self._qid_to_answers = {}
            for ann in annotations:
                qid = int(ann['question_id'])
                # Extract answer strings from annotator responses
                answers = [a['answer'] for a in ann['answers']]
                self._qid_to_answers[qid] = answers

            print(f"   ✓ Loaded {len(self._qid_to_answers)} question annotations")
        except Exception as e:
            print(f"⚠️  Failed to load VQA annotations: {e}")
            self._qid_to_answers = {}
        finally:
            self._loading = False
    
    def get_answers(self, question_id: int) -> List[str]:
        """
        Get list of ground truth answers for a question.
        
        Args:
            question_id: VQA question ID
            
        Returns:
            List of 10 annotator answers (strings)
        """
        self._load()
        qid = int(question_id)
        return self._qid_to_answers.get(qid, [])
    
    def get_majority_answer(self, question_id: int) -> str:
        """Get most common answer for a question."""
        answers = self.get_answers(question_id)
        if not answers:
            return ""
        from collections import Counter
        return Counter(answers).most_common(1)[0][0]
    
    def has_question(self, question_id: int) -> bool:
        """Check if question_id exists in annotations."""
        self._load()
        return int(question_id) in self._qid_to_answers

def skip_processed_idx(existing_keys, output_jsonl_path): 
    id_keys = ['idx', 'qid', 'question_id', 'index']
    current_item_id = None
    processed_ids=set()

    for key in id_keys:
        if key in existing_keys:
            current_item_id = key
            break
    
    if current_item_id is None : 
        current_item_id = 'pid'

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
        print(f"Skipping {len(processed_ids)} items.")
    return processed_ids, current_item_id

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


# =============================================================================
# Parsing Functions (from your processor.py)
# =============================================================================


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
    
    # Pattern 2: Look for the last standalone letter A-D
    matches = re.findall(r'\b([A-D])\b', output)
    if matches:
        return matches[-1].upper()
    
    # Pattern 3: Check if output is just a single letter
    if len(output) == 1 and output.upper() in 'ABCD':
        return output.upper()
    
    # Pattern 4: Check format like "A: Hanging Posters"
    match = re.match(r'^([A-D]):', output)
    if match:
        return match.group(1).upper()
    
    return ""


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


# =============================================================================
# Scoring Functions
# =============================================================================

def mc_accuracy(gt: str, pred: str) -> bool:
    """Multiple choice accuracy."""
    gt_letter = gt.strip().upper()[0] if gt else ""
    pred_letter = extract_mc_choice(pred)
    return gt_letter == pred_letter


def exact_match(gt: str, pred: str) -> bool:
    """Exact match after normalization."""
    return normalize_answer(pred) == normalize_answer(gt)

