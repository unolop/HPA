
import json, random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

@dataclass
class QAItem:
    qid: str
    question: str
    # human distribution over answers (canonicalized)
    human_probs: Dict[str, float]
    # optional explicit candidate list to score
    candidates: Optional[List[str]] = None

def normalize(s: str) -> str:
    return " ".join((s or "").lower().strip().split())

def build_items(human_jsonl: str, synonyms: Optional[Dict[str,str]]=None) -> List[QAItem]:
    # human_jsonl lines must contain: qid, question, answers (list of strings)
    from collections import Counter
    items = []
    for ex in load_jsonl(human_jsonl):
        qid = ex["qid"]
        question = ex.get("question","")
        answers = [normalize(a) for a in ex.get("answers", [])]
        if synonyms:
            answers = [synonyms.get(a, a) for a in answers]
        c = Counter(answers)
        N = sum(c.values()) or 1
        probs = {k: v / N for k,v in c.items()}
        cand = list(sorted(probs.keys()))
        items.append(QAItem(qid=qid, question=question, human_probs=probs, candidates=cand))
    return items

def split(items: List[QAItem], seed: int=123, ratios=(0.8,0.1,0.1)):
    rnd = random.Random(seed)
    idx = list(range(len(items)))
    rnd.shuffle(idx)
    n = len(items)
    n_tr = int(ratios[0]*n)
    n_va = int(ratios[1]*n)
    tr = [items[i] for i in idx[:n_tr]]
    va = [items[i] for i in idx[n_tr:n_tr+n_va]]
    te = [items[i] for i in idx[n_tr+n_va:]]
    return tr, va, te
