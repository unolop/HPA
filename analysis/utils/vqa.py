import re
import os 
import json 
import pandas as pd 
import math
from typing import List , Union, Optional

import sys
from pathlib import Path
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent.parent))
from dataset.paths import VQA_ANNOT as VQA_ANNOTATIONS_PATH

CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've",
                "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
                "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
                "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
                "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
                "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
                "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
                "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
                "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
                "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
                "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
                "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
                "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
                "youll": "you'll", "youre": "you're", "youve": "you've"}
MANUAL_MAP = {
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
ARTICLES = ["a", "an", "the"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [';', r"/", '[', ']', '"', '{', '}',
         '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']
THINK_RE = re.compile(r'<think>.*?</think>\s*', flags=re.DOTALL)

_vqa_mapper = None

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


def strip_think_answer(text):
    return THINK_RE.sub('', str(text or '')).strip()


def process_punctuation(in_text):
    out_text = str(in_text or '')
    for punct in PUNCT:
        if (punct + ' ' in out_text or ' ' + punct in out_text) or re.search(COMMA_STRIP, out_text) is not None:
            out_text = out_text.replace(punct, '')
        else:
            out_text = out_text.replace(punct, ' ')
    return PERIOD_STRIP.sub("", out_text, re.UNICODE)


def process_digit_article(in_text):
    out_text = []
    temp_text = str(in_text or '').lower().split()
    for word in temp_text:
        word = MANUAL_MAP.setdefault(word, word)
        if word not in ARTICLES:
            out_text.append(word)
    for idx, word in enumerate(out_text):
        if word in CONTRACTIONS:
            out_text[idx] = CONTRACTIONS[word]
    return ' '.join(out_text)


def preprocess_answer(text, strip_think=True):
    out_text = str(text or '').replace('\n', ' ').replace('\t', ' ').strip()
    # Always drop hidden chain-of-thought wrappers before any other normalization.
    # `strip_think` remains only for backwards compatibility at call sites.
    out_text = strip_think_answer(out_text)
    out_text = process_punctuation(out_text)
    out_text = process_digit_article(out_text)
    return out_text.strip()
    
def vqa_accuracy(pred, gt_answers):
    # pred가 리스트인 경우 처리
    if isinstance(pred, list):
        pred = pred[0] if pred else ""

    if pred is None:
        return 0.0

    pred = preprocess_answer(pred)

    matches = 0
    for ans in gt_answers:
        if isinstance(ans, dict):
            gt = ans.get('answer', '')
        else:
            gt = ans

        if preprocess_answer(gt) == pred:
            matches += 1

    acc = min(1.0, matches / 3.0)
    return acc


class VQAEval:
    def __init__(self, vqa, vqaRes, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        self.params = {'question_id': vqa.getQuesIds()}
        self.contractions = CONTRACTIONS
        self.manualMap = MANUAL_MAP
        self.articles = ARTICLES
        self.periodStrip = PERIOD_STRIP
        self.commaStrip = COMMA_STRIP
        self.punct = PUNCT

    def evaluate(self, quesIds=None):
        if quesIds is None:
            quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        accQA = []
        accQuesType = {}
        accAnsType = {}
        print("computing accuracy")
        step = 0
        for quesId in quesIds:
            gtAcc = []
            resAns = preprocess_answer(res[quesId]['answer'])
            norm_answers = []
            for ansDic in gts[quesId]['answers']:
                norm_answers.append({
                    **ansDic,
                    'answer': preprocess_answer(ansDic['answer']),
                })

            for gtAnsDatum in norm_answers:
                otherGTAns = [item for item in norm_answers if item != gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item['answer'] == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            quesType = gts[quesId]['question_type']
            ansType = gts[quesId]['answer_type']
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)
            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)
            if step % 100 == 0:
                self.updateProgress(step / float(len(quesIds)))
            step = step + 1

        self.setAccuracy(accQA, accQuesType, accAnsType)
        print("Done computing accuracy")

    def processPunctuation(self, inText):
        return process_punctuation(inText)

    def processDigitArticle(self, inText):
        return process_digit_article(inText)

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy['overall'] = round(100 * float(sum(accQA)) / len(accQA), self.n)
        self.accuracy['perQuestionType'] = {
            quesType: round(100 * float(sum(accQuesType[quesType])) / len(accQuesType[quesType]), self.n)
            for quesType in accQuesType
        }
        self.accuracy['perAnswerType'] = {
            ansType: round(100 * float(sum(accAnsType[ansType])) / len(accAnsType[ansType]), self.n)
            for ansType in accAnsType
        }

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rFinshed Percent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), int(progress * 100), status)
        sys.stdout.write(text)
        sys.stdout.flush()

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
