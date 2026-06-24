"""
Korean → English answer normalization for VQA scoring.

The mapping lives in analysis/utils/kr_answer_map.json.
Edit that file to add or fix translations, then re-run the notebook.
"""
import re
import json
from pathlib import Path

_MAP_PATH = Path(__file__).parent / 'kr_answer_map.json'


def _load_map():
    with open(_MAP_PATH, encoding='utf-8') as f:
        return json.load(f)


def _build_regex(numbers: dict, counter_words: list) -> re.Pattern:
    cw = '|'.join(counter_words)
    return re.compile(
        r'^(' + '|'.join(sorted(numbers.keys(), key=len, reverse=True)) +
        r')(?:\s*(?:' + cw + r'))?[\s\.\,]?$'
    )


# Politeness/sentence endings to strip before map lookup or API translation.
# Ordered longest-first so greedy matching works correctly.
_ENDINGS = [
    '입니다', '이에요', '습니다', '합니다', '이라고', '이라면',
    '예요', '어요', '아요', '이야', '이다', '한다',
    '이요', '네요', '군요', '죠', '요',
]
_PUNCT_RE = re.compile(r'[\s.,!?~。~\u3002]+$')


def clean_korean_answer(text: str) -> str:
    """Strip trailing punctuation and common Korean politeness endings.

    Returns the canonical bare form used for deduplication before API calls
    and as a fallback lookup key in the map.

    Examples:
        '동물이요'  → '동물'
        '집이요'    → '집'
        '없어요'    → '없어'  (bare stem — still meaningful)
        '고양이입니다' → '고양이'
        '선글라스를 쓴다.' → '선글라스를 쓴다'
    """
    t = _PUNCT_RE.sub('', text.strip())
    for ending in _ENDINGS:
        if t.endswith(ending) and len(t) > len(ending):
            t = _PUNCT_RE.sub('', t[:-len(ending)].strip())
            break   # strip at most one ending
    return t


def normalize_korean_answer(text: str, _cache: dict = {}) -> str:
    """
    Normalize a Korean VQA answer to English for scoring against English GT.

    Lookup order:
      1. Direct lookup in kr_answer_map.json 'map' (highest priority — manual overrides)
      2. yes/no keywords
      3. Leading Arabic digits + optional Korean counter (e.g. '3명' → '3')
      4. Korean number words + optional counter
      5. Color words
      6. Misc common VQA words
      7. Already ASCII passthrough
      8. Return as-is (will score 0 — flagged as unnormalized)

    The map and compiled regex are cached after first call.
    """
    if not _cache:
        m = _load_map()
        _cache['map']     = m['map']
        _cache['yesno']   = m['yesno']
        _cache['colors']  = m['colors']
        _cache['numbers'] = m['numbers']
        _cache['misc']    = m['misc']
        _cache['regex']   = _build_regex(m['numbers'], m['counter_words'])

    t = text.strip()
    t_lower = t.lower()

    # 1. Direct map lookup — try raw form first, then cleaned canonical form
    if t in _cache['map'] and _cache['map'][t] not in (None, ''):
        return _cache['map'][t]
    canonical = clean_korean_answer(t)
    if canonical != t and canonical in _cache['map'] and _cache['map'][canonical] not in (None, ''):
        return _cache['map'][canonical]

    # 2. Yes/no keywords
    for kr, en in sorted(_cache['yesno'].items(), key=lambda x: -len(x[0])):
        if t_lower.startswith(kr):
            return en
    if t_lower in ('yes', 'no'):
        return t_lower

    # 3. Leading Arabic digits + optional Korean counter
    m = re.match(r'^(\d+)', t)
    if m:
        return m.group(1)

    # 4. Korean number words + optional counter
    m = _cache['regex'].match(t)
    if m:
        return _cache['numbers'].get(m.group(1), t)

    # 5. Color words
    for kr, en in sorted(_cache['colors'].items(), key=lambda x: -len(x[0])):
        if t.startswith(kr):
            return en

    # 6. Misc common VQA words
    # Use exact match for short keys (<=2 chars) to avoid false substring matches
    # (e.g. '위' matching '위험해서', '안' matching '안전')
    for kr, en in sorted(_cache['misc'].items(), key=lambda x: -len(x[0])):
        if len(kr) <= 2:
            if canonical == kr or t == kr:
                return en
        else:
            if kr in t:
                return en

    # 7. Already ASCII
    if re.match(r'^[a-zA-Z0-9\s\-\/\.\,\(\)]+$', t):
        return t.lower()

    # 8. Unmapped
    return t
