import re
import os 
import json 
import pandas as pd 
import math
from typing import List , Union, Optional

VQA_ANNOTATIONS_PATH = "/home/david/Desktop/yuna/data/v2_mscoco_val2014_annotations.json"

NUMBER_WORDS = {
    "zero": 0,
    "none": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

NUMBER_REGEX = re.compile(
    r"""
    [-+]?                  # optional sign
    (?:
        \d*\.\d+ |         # decimal number
        \d+                # integer
    )
    """,
    re.VERBOSE
)
def vqa_accuracy(pred, gt_answers):
    # pred가 리스트인 경우 처리
    if isinstance(pred, list):
        pred = pred[0] if pred else ""
    
    if pred is None:
        return 0.0
    
    pred = str(pred).strip().lower()
    
    matches = 0
    for ans in gt_answers:
        # gt_answers가 딕셔너리 리스트인 경우: [{'answer': 'yes'}, {'answer': 'no'}, ...]
        if isinstance(ans, dict):
            gt = ans.get('answer', '')
        else:
            # 문자열 리스트인 경우: ['yes', 'no', ...]
            gt = ans
        
        if str(gt).strip().lower() == pred: 
            matches += 1
    
    acc = min(1.0, matches / 3.0)
    return acc 

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

def get_vqa_mapper() -> VQAAnswerMapper:
    """Get or create global VQA mapper."""
    global _vqa_mapper
    if _vqa_mapper is None:
        _vqa_mapper = VQAAnswerMapper()
    return _vqa_mapper
    
def extract_number_or_other(
    x: Union[str, int, float, None],
    *,
    return_int_if_possible: bool = True,
    lowercase: bool = True
) -> Optional[Union[int, float, str]]:

    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, (int, float)):
        if return_int_if_possible and isinstance(x, float) and x.is_integer():
            return int(x)
        return x

    # ---------------------------
    # Normalize string
    # ---------------------------
    s = str(x).strip()
    if lowercase:
        s = s.lower()

    if s == "":
        return None

    # ---------------------------
    # Regex numeric extraction
    # ---------------------------
    match = NUMBER_REGEX.search(s)
    if match:
        num = float(match.group())
        if return_int_if_possible and num.is_integer():
            return int(num)
        return num
    return s

def extract_number(text):
    if pd.isna(text):
        return None

    text = str(text).lower()

    # 1️⃣ digit-based number
    digit_match = re.search(r"-?\d+\.?\d*", text)
    if digit_match:
        return float(digit_match.group())

    # 2️⃣ word-based number (e.g., "one thing")
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            # print("extracted ", value)
            return float(value)
    return None 

def score_number(pred, gt, tol=1e-3):
    p = extract_number(pred)
    g = extract_number(gt) 
    if p is None or g is None:
        return 0.0
    is_p_int = isinstance(p, (int, float)) and float(p).is_integer()
    is_g_int = isinstance(g, (int, float)) and float(g).is_integer()
    if is_p_int and is_g_int:
        return float(int(p) == int(g))
    try:
        return float(abs(float(p) - float(g)) <= tol)
    except Exception:
        print('cannot score number')
        return 0.0

def score_yes_no(pred, gt):
    p = str(pred).strip().lower()
    g = str(gt).strip().lower()
    return float(p == g)

        
def yes_or_no(df):
    out = df['output'].astype(str).str.lower()

    df = df.copy()
    df['y/n'] = 'others'
    df.loc[out.str.contains('yes', na=False), 'y/n'] = 'yes'
    df.loc[out.str.contains('no', na=False), 'y/n'] = 'no'
    return df

def bin_number(x):
    if x == 'others':
        return 'others'
    if not isinstance(x, (int, float)):
        return 'others'

    if x == 0:
        return '0'
    elif x == 1:
        return '1'
    elif 2 <= x <= 3:
        return '2–3'
    elif 4 <= x <= 5:
        return '4–5'
    elif 6 <= x <= 10:
        return '6–10'
    elif 11 <= x <= 20:
        return '11–20'
    else:
        return '>20'

def score_answer_types(df, answer_column, gt_column):
    SCORERS = {
        "number": score_number,
        "yes/no": score_yes_no,
    }

    def score_row(row):
        atype = str(row["answer_type"]).strip().lower()  # 정규화 추가

        if atype in SCORERS:
            return SCORERS[atype](row[answer_column], row[gt_column])
        return float(row["correct"]) if "correct" in row and pd.notna(row["correct"]) else 0.0

    df = df.copy()
    df["correct"] = df.apply(score_row, axis=1)
    df["correct"] = df["correct"] * 100
    return df


