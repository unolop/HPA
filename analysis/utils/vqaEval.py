"""Backward-compatible shim; canonical VQA utilities now live in analysis.utils.vqa."""

from analysis.utils.vqa import (
    ARTICLES,
    COMMA_STRIP,
    CONTRACTIONS,
    MANUAL_MAP,
    PERIOD_STRIP,
    PUNCT,
    VQAEval,
    preprocess_answer,
    process_digit_article,
    process_punctuation,
    strip_think_answer,
)
