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
    
    df['model_raw'] = df['model']
    df['model'] = df['model'].map(MODEL_DISPLAY_MAP)  
    unmapped = (
        df.loc[df['model'].isna(), 'model_raw']
        .unique()
        )

    print(unmapped) 
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
    """Maps question_id to list of ground truth answers from VQA annotations."""

    def __init__(self, annotations_path: str = VQA_ANNOTATIONS_PATH):
        self.annotations_path = annotations_path
        self._qid_to_answers = None
        self._qid_to_gt_visual = None  # Store visual GT for VQA
        self.annotations = {} 

    def _load(self):
        """Load annotations and build lookup dict."""
        if self._qid_to_answers is not None:
            return

        if not os.path.exists(self.annotations_path):
            print(f"⚠️  VQA annotations not found: {self.annotations_path}")
            self._qid_to_answers = {}
            self._qid_to_gt_visual = {}
            return

        print(f"   Loading VQA annotations from {self.annotations_path}...")
        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        self._qid_to_answers = {}
        self._qid_to_gt_visual = {}

        for ann in annotations['annotations']:
            qid = int(ann['question_id']) 
            self.annotations[qid] = ann 
            answers = [a['answer'] for a in ann['answers']] 
            self._qid_to_answers[qid] = answers

            # Multiple choice answer (consensus from humans who saw image)
            if 'multiple_choice_answer' in ann:
                self._qid_to_gt_visual[qid] = ann['multiple_choice_answer']

        print(f"   ✓ Loaded {len(self._qid_to_answers)} VQA annotations")

    def get_answers(self, question_id: int) -> List[str]:
        """Get list of 10 annotator answers for a question."""
        self._load()
        qid = int(question_id)
        return self._qid_to_answers.get(qid, [])

    def get_visual_gt(self, question_id: int) -> str:
        """Get visual ground truth (multiple choice answer from humans who saw image)."""
        self._load()
        qid = int(question_id)
        return self._qid_to_gt_visual.get(qid, "")


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


contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }

numbers = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }

punctuations = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

class PostProcessor: 
    
    def __init__(self):
        self.contractions = contractions
        self.manualMap = numbers 
        self.articles = ["a", "an", "the"]
        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(\,)(\d)")
        self.punct = punctuations 

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        # print(inText, outText)
 
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        # print(inText, 'out:', outText)
        return outText

    def postprocess_answer(self, answer):
        
        if isinstance(answer, str):
            answer = answer.split("\n")[-1].split("\t")[-1]
            # generation = generation.replace("\n", " ").replace("\t", " ").strip()

        if isinstance(answer, dict):
            answer = answer.get("answer", "")
        if isinstance(answer, list):
            answer = answer[0]
    
        if "[INST]" in answer and "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()
        
        if "inst" in answer:
            answer = answer.split("inst")[-1].strip()
            
        if "USER" in answer and "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[-1].strip()
        
        if "assistant" in answer:
            answer = answer.split('assistant')[-1].strip()
        
        if "Answer:" in answer: 
            answer = answer.split("Answer: ")[-1]

        if "answer:" in answer:  ## FOR LLAVA 1.5 
            answer = answer.split("answer: ")[-1]
        # else:
        #     answer = answer.strip().split()[-1] if isinstance(answer, str) else ""

        # Remove extraneous tokens
        for token in [
            '<s>', '</s>', '<|im_end|>', '[/INST]', '<|im_start|>', 
            'ASSISTANT:', 'assistant\n', 'inst'
        ]:
            answer = answer.replace(token, '')

        # answer = answer.strip()
        if not answer:
            answer = ""

        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        
        return answer
