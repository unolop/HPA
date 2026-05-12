"""
Shared plotting and analysis constants for all S2 notebooks.

Model groups: VLM | VLM backbone decoder | standalone LLM | standalone LLM (think)

Usage
-----
from utils.constants import (
    MODEL_ORDER, COND_LABEL, COND_COLOR,
    VARIANT_ORDER, VARIANT_LABELS, VARIANT_COLORS,
    GROUP_COLORS,
    CLASS_ORDER, CLASS_COLOR,
    CONTROL_TYPES, CT_LABELS, CT_TO_VARIANT,
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

CONDITIONS = [
    ('_control_blind', 'blind', COND_COLOR['blind']),
    ('_control_inst_blind', 'inst_blind', COND_COLOR['inst_blind']),
]

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

# ── Model group colors ───────────────────────────────────────────────────────
GROUP_COLORS = {
    'VLM':                    '#E91E63',
    'VLM backbone decoder':   '#2196F3',
    'standalone LLM':         '#FF9800',
    'standalone LLM (think)': '#4CAF50',
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

# ── Control-type constants for full VQA question ladders ─────────────────────
CONTROL_TYPES = ['question', 'deictic_removed', 'object_removed', 'weaker_object', 'pronominalized']

CT_LABELS = ['original', 'deictic\nremoved', 'object\nremoved', 'weaker\nobject', 'pronominalized']

CT_TO_VARIANT = {
    'question': 'C',
    'weaker_object': 'B',
    'pronominalized': 'A',
}

# ── Generic abstention lexicon for blind / inst_blind analysis ───────────────
ABSTAIN_TOKENS = [
    'none', 'nothing', 'unknown', 'unanswerable', 'no image',
    'cannot', "can't", 'unable', 'n/a', 'not visible', 'not shown',
]

# ── Tier-level plotting constants ────────────────────────────────────────────
TIER_ORDER = ['VLM', 'VLM backbone decoder', 'standalone LLM', 'standalone LLM (think)']

TIER_COLORS = {
    'VLM':                    '#2c3e50',
    'VLM backbone decoder':   '#e67e22',
    'standalone LLM':         '#27ae60',
    'standalone LLM (think)': '#1565c0',
}

TIER_STYLE = {
    'VLM':                    {'color': '#2c3e50', 'marker': 'o', 'ls': '-',  'lw': 2.2},
    'VLM backbone decoder':   {'color': '#e67e22', 'marker': 's', 'ls': '--', 'lw': 1.8},
    'standalone LLM':         {'color': '#27ae60', 'marker': '^', 'ls': ':',  'lw': 1.8},
    'standalone LLM (think)': {'color': '#1565c0', 'marker': 'D', 'ls': '-.',  'lw': 1.8},
}

# ── Model registries used in prior tier / decoder comparisons ────────────────
VLM_MODELS = [
    'Qwen3-VL-2B-Instruct', 'Qwen3-VL-4B-Instruct', 'Qwen3-VL-8B-Instruct',
    'InternVL3_5-1B', 'InternVL3_5-2B', 'InternVL3_5-8B',
    'llava-1.5-7b-hf', 'llava-v1.6-mistral-7b-hf', 'llava-v1.6-vicuna-7b-hf',
]

LM_MODELS = VLM_MODELS

BB_MODELS = [
    'Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-32B',
    'Mistral-7B',
    'Vicuna-7B', 'Vicuna-13B',
]

# (vlm_model, lm_model, backbone_model, display_label)
TRIPLES = [
    ('llava-v1.6-mistral-7b-hf', 'llava-v1.6-mistral-7b-hf', 'Mistral-7B-Instruct-v0.2', 'Mistral-7B'),
    ('llava-v1.6-vicuna-7b-hf', 'llava-v1.6-vicuna-7b-hf', 'vicuna-7b-v1.5', 'Vicuna-7B'),
    ('Qwen3-VL-4B-Instruct', 'Qwen3-VL-4B-Instruct', 'Qwen3-4B', 'Qwen3-4B'),
    ('Qwen3-VL-8B-Instruct', 'Qwen3-VL-8B-Instruct', 'Qwen3-8B', 'Qwen3-8B'),
]
