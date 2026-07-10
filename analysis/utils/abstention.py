"""
Unified abstention classifier for VLM blind/inst_blind outputs.

Replaces separate implementations in notebooks 03, 05, 12.

Usage
-----
from utils.abstention import classify, is_abstained

cls = classify(output_text, gt_answers)   # → str
if is_abstained(cls): ...

Design choices
--------------
Hard abstention: output contains an explicit image-reference refusal keyword
  (cannot, can't, no image, unable, not visible, blank, etc.).

Soft abstention: the full normalized answer is an *epistemic hedge* — a word
  that signals the model cannot or will not commit to an answer.
  Excluded from soft abstention:
    - "nothing"  → valid content answer (e.g. "nothing is in the bowl");
                   for number questions callers may map it to "0"
    - "maybe", "possibly", "perhaps", "uncertain", "unsure"
                 → partial yes/no attempts; treated as committed (other)
                   rather than abstention, so they appear in the answer
                   distribution alongside human responses
"""
from __future__ import annotations

import re
from typing import List, Optional

from .vqa import vqa_accuracy

# Hard abstention: explicit refusal to answer without seeing an image
_HARD_RE = re.compile(
    r'cannot|can\'t|don\'t see|no image|unable|not visible'
    r'|i cannot|i don\'t|can not|no visual|cannot see'
    r'|i\'m not able|i am not able|\bblank\b',
    re.IGNORECASE,
)

# Soft abstention: epistemic hedge words that signal the model won't commit.
# NOTE: "nothing", "maybe", "possibly", "perhaps", "uncertain", "unsure"
# are intentionally excluded — see module docstring.
SOFT_WORDS = frozenset({
    'unanswerable', 'unclear', 'unknown', 'nowhere',
    'n/a', 'na', 'indeterminate', 'unidentifiable', 'no sign',
    "can't tell", 'cannot tell', 'not clear', 'not visible',
})

# Degenerate: structurally empty answers
_DEGEN_WORDS = frozenset({'answer', 'a', 'the'})


def classify(output: str, gt_answers: Optional[List[str]]) -> str:
    """
    Classify a model output into one of five categories.

    Categories
    ----------
    hard_abstained       — explicit refusal ("I cannot see the image")
    soft_abstained       — epistemic hedge ("unknown", "unanswerable")
    hallucinated_correct — committed answer, happens to score > 0 on GT
    hallucinated_wrong   — committed wrong answer (incl. "nothing", "maybe")
    degenerate           — empty or structurally invalid output

    Parameters
    ----------
    output      : raw model output string
    gt_answers  : list of ground-truth answers (from VQA annotations), or None
    """
    if not output or not output.strip():
        return 'degenerate'

    out_norm = output.strip().lower().strip('.,!? ')

    if _HARD_RE.search(output):
        return 'hard_abstained'

    if out_norm in SOFT_WORDS:
        return 'soft_abstained'

    if out_norm in _DEGEN_WORDS:
        return 'degenerate'

    if gt_answers and vqa_accuracy(output, gt_answers) > 0:
        return 'hallucinated_correct'

    return 'hallucinated_wrong'


def is_abstained(cls: str) -> bool:
    """Return True for hard_abstained or soft_abstained."""
    return cls in ('hard_abstained', 'soft_abstained')
