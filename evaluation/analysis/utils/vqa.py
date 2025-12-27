import re
import os 
import json 
import pandas as pd 
import math
from typing import List , Union, Optional

VQA_ANNOTATIONS_PATH = "/home/work/yuna/VLMEval/data/v2_mscoco_val2014_annotations.json"

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
    return acc * 100 

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
        
        # 숫자 단어 -> 숫자 매핑 추가
        self.word_to_num = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
            'once': '1', 'twice': '2', 'single': '1', 'double': '2', 'triple': '3',
        }

    def extract_number(self, text):
        """텍스트에서 숫자 추출"""
        text_lower = text.lower()
        
        # 1. 숫자 단어가 있으면 변환
        for word, num in self.word_to_num.items():
            if word in text_lower.split():
                return num
        
        # 2. 숫자가 직접 있으면 추출
        numbers_found = re.findall(r'\d+', text)
        if numbers_found:
            return numbers_found[0]
        
        return text 


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

        if "answer:" in answer:
            answer = answer.split("answer: ")[-1]

        for token in [
            '<s>', '</s>', '<|im_end|>', '[/INST]', '<|im_start|>', 
            'ASSISTANT:', 'assistant\n', 'inst'
        ]:
            answer = answer.replace(token, '')

        if not answer:
            answer = ""

        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        answer = self.extract_number(answer)
        
        return answer.strip()