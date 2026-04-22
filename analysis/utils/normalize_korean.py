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

    # 1. Direct map lookup (includes manual translations)
    if t in _cache['map'] and _cache['map'][t] is not None:
        return _cache['map'][t]

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
    for kr, en in sorted(_cache['misc'].items(), key=lambda x: -len(x[0])):
        if kr in t:
            return en

    # 7. Already ASCII
    if re.match(r'^[a-zA-Z0-9\s\-\/\.\,\(\)]+$', t):
        return t.lower()

    # 8. Unmapped
    return t
