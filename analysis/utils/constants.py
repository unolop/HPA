"""
Shared plotting and analysis constants for all S2 notebooks.

Usage
-----
from utils.constants import (
    MODEL_ORDER, COND_LABEL, COND_COLOR,
    VARIANT_ORDER, VARIANT_LABELS, VARIANT_COLORS,
    CLASS_ORDER, CLASS_COLOR,
)
"""

# ── Pretrained VLM model list (blind / inst_blind analysis) ──────────────────
MODEL_ORDER = [
    'llava-1.5-7b-hf',
    'llava-v1.6-mistral-7b-hf',
    'llava-v1.6-vicuna-7b-hf',
    'InternVL3_5-1B',
    'InternVL3_5-2B',
    'InternVL3_5-8B',
    'Qwen3-VL-2B-Instruct',
    'Qwen3-VL-4B-Instruct',
    'Qwen3-VL-8B-Instruct',
]

MODEL_LABEL = {
    'llava-1.5-7b-hf':           'LLaVA-1.5-7B',
    'llava-v1.6-mistral-7b-hf':  'LLaVA-Mistral',
    'llava-v1.6-vicuna-7b-hf':   'LLaVA-Vicuna',
    'InternVL3_5-1B':            'InternVL-1B',
    'InternVL3_5-2B':            'InternVL-2B',
    'InternVL3_5-8B':            'InternVL-8B',
    'Qwen3-VL-2B-Instruct':      'Qwen3-VL-2B',
    'Qwen3-VL-4B-Instruct':      'Qwen3-VL-4B',
    'Qwen3-VL-8B-Instruct':      'Qwen3-VL-8B',
}

# ── Condition labels and colors (blind / inst_blind) ─────────────────────────
COND_LABEL = {
    'blind':      'Blind',
    'inst_blind': 'Blind + Inst',
    'control':    'Original (+image)',
}

COND_COLOR = {
    'blind':      '#c62828',
    'inst_blind': '#1565c0',
    'control':    '#2e7d32',
}

# ── Variant labels and colors (A / B / C) ────────────────────────────────────
VARIANT_ORDER = ['C', 'B', 'A']

VARIANT_LABELS = {
    'C': 'C (original)',
    'B': 'B (weaker obj)',
    'A': 'A (pronominalized)',
}

VARIANT_COLORS = {
    'C': '#2196F3',
    'B': '#FF9800',
    'A': '#E91E63',
}

# ── Abstention class taxonomy ─────────────────────────────────────────────────
CLASS_ORDER = [
    'hard_abstained',
    'soft_abstained',
    'hallucinated_correct',
    'hallucinated_wrong',
    'degenerate',
]

CLASS_COLOR = {
    'hard_abstained':       '#7b1fa2',
    'soft_abstained':       '#e65100',
    'hallucinated_correct': '#2e7d32',
    'hallucinated_wrong':   '#c62828',
    'degenerate':           '#90a4ae',
}

CLASS_LABEL = {
    'hard_abstained':       'Hard abstain',
    'soft_abstained':       'Soft abstain',
    'hallucinated_correct': 'Hallucinated (correct)',
    'hallucinated_wrong':   'Hallucinated (wrong)',
    'degenerate':           'Degenerate',
}
