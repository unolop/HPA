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
    'Qwen3-VL-32B-Instruct',
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
    'Qwen3-VL-32B-Instruct':     'Qwen3-VL-32B',
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

# ── Model group colors (canonical — used by all notebooks and scripts) ────────
GROUP_COLORS = {
    'VLM':                    '#E53935',   # red
    'VLM backbone decoder':   '#E67E22',   # orange
    'standalone LLM':         '#2E7D32',   # green
    'standalone LLM (think)': '#8E24AA',   # purple
}

GROUP_ORDER = ['VLM backbone decoder', 'VLM', 'standalone LLM (think)', 'standalone LLM']

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

TIER_COLORS = GROUP_COLORS  # alias — use GROUP_COLORS everywhere

TIER_STYLE = {
    'VLM':                    {'color': '#E53935', 'marker': 'o', 'ls': '-',  'lw': 2.2, 'hollow': False},
    'VLM backbone decoder':   {'color': '#E67E22', 'marker': 'o', 'ls': ':',  'lw': 1.8, 'hollow': True},
    'standalone LLM':         {'color': '#2E7D32', 'marker': 's', 'ls': '-',  'lw': 1.8, 'hollow': False},
    'standalone LLM (think)': {'color': '#8E24AA', 'marker': 's', 'ls': ':',  'lw': 1.8, 'hollow': True},
}

# ── Model registries used in prior tier / decoder comparisons ────────────────
VLM_MODELS = [
    'Qwen3-VL-2B-Instruct', 'Qwen3-VL-4B-Instruct', 'Qwen3-VL-8B-Instruct', 'Qwen3-VL-32B-Instruct',
    'InternVL3_5-1B', 'InternVL3_5-2B', 'InternVL3_5-8B',
    'llava-1.5-7b-hf', 'llava-v1.6-mistral-7b-hf', 'llava-v1.6-vicuna-7b-hf',
]

LM_MODELS = VLM_MODELS  # same models, text-only decoder mode

BB_MODELS = [
    'Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-32B',
    'Mistral-7B',
    'Vicuna-7B', 'Vicuna-13B',
    'Qwen2.5-7B', 'Phi-3.5-mini',
]

# ── Model family membership (display-name → family) ─────────────────────────
MODEL_FAMILY = {
    # Qwen3-VL (vision-language)
    'Qwen3-VL-2B':          'Qwen3-VL',
    'Qwen3-VL-4B':          'Qwen3-VL',
    'Qwen3-VL-8B':          'Qwen3-VL',
    'Qwen3-VL-32B':         'Qwen3-VL',
    # Qwen3 standalone (nothink + think share one family)
    'Qwen3-0.6B':           'Qwen3',
    'Qwen3-1.7B':           'Qwen3',
    'Qwen3-4B':             'Qwen3',
    'Qwen3-8B':             'Qwen3',
    'Qwen3-32B':            'Qwen3',
    'Qwen3-0.6B (think)':   'Qwen3',
    'Qwen3-1.7B (think)':   'Qwen3',
    'Qwen3-4B (think)':     'Qwen3',
    'Qwen3-8B (think)':     'Qwen3',
    'Qwen3-32B (think)':    'Qwen3',
    # LLaVA — split by base model so each gets a distinct red shade
    'LLaVA-1.5-7B':         'LLaVA-1.5',
    'LLaVA-1.5 (LM)':       'LLaVA-1.5',
    'LLaVA-Mistral':        'LLaVA-Mistral',
    'LLaVA-Mistral (LM)':   'LLaVA-Mistral',
    'LLaVA-Vicuna':         'LLaVA-Vicuna',
    'LLaVA-Vicuna (LM)':    'LLaVA-Vicuna',
    # InternVL
    'InternVL-1B':          'InternVL',
    'InternVL-2B':          'InternVL',
    'InternVL-8B':          'InternVL',
    # Others
    'Mistral-7B':           'Mistral',
    'Vicuna-13B':           'Vicuna',
    'Phi-3.5-mini':         'Phi',
    'Qwen2.5-7B':           'Qwen2.5',
}

# ── Model family colors ───────────────────────────────────────────────────────
MODEL_FAMILY_COLORS = {
    'Qwen3-VL':  '#5C6BC0',   # indigo
    'Qwen3':     '#0288D1',   # light blue
    'LLaVA-1.5':     '#B71C1C',   # deep crimson
    'LLaVA-Mistral': '#E53935',   # medium red
    'LLaVA-Vicuna':  '#FF7043',   # red-orange
    'InternVL':  '#F9A825',   # amber yellow
    'Mistral':   '#2E7D32',   # green
    'Vicuna':    '#6A1B9A',   # purple
    'Phi':       '#00838F',   # teal
    'Qwen2.5':   '#558B2F',   # olive green
    'Human':     '#1565C0',   # blue (matches GROUP_COLORS human convention)
}

# ── Model parameter sizes (billions) ─────────────────────────────────────────
MODEL_SIZE_B = {
    'Qwen3-VL-2B':          2.0,
    'Qwen3-VL-4B':          4.0,
    'Qwen3-VL-8B':          8.0,
    'Qwen3-VL-32B':        32.0,
    'Qwen3-0.6B':           0.6,
    'Qwen3-1.7B':           1.7,
    'Qwen3-4B':             4.0,
    'Qwen3-8B':             8.0,
    'Qwen3-32B':           32.0,
    'Qwen3-0.6B (think)':   0.6,
    'Qwen3-1.7B (think)':   1.7,
    'Qwen3-4B (think)':     4.0,
    'Qwen3-8B (think)':     8.0,
    'Qwen3-32B (think)':   32.0,
    'LLaVA-1.5-7B':         7.0,
    'LLaVA-Mistral':        7.0,
    'LLaVA-Vicuna':         7.0,
    'LLaVA-1.5 (LM)':       7.0,
    'LLaVA-Mistral (LM)':   7.0,
    'LLaVA-Vicuna (LM)':    7.0,
    'InternVL-1B':          1.0,
    'InternVL-2B':          2.0,
    'InternVL-8B':          8.0,
    'Mistral-7B':           7.0,
    'Vicuna-13B':          13.0,
    'Phi-3.5-mini':         3.8,
    'Qwen2.5-7B':           7.0,
}

# ── Group marker shapes, hollow flags, and line styles ───────────────────────
# VLM / SA-LLM → circle / square (filled)
# Backbone / Think → same shape but hollow + dotted line
GROUP_MARKER = {
    'VLM':                    'o',   # circle filled
    'VLM backbone decoder':   'o',   # circle hollow
    'standalone LLM':         's',   # square filled
    'standalone LLM (think)': 's',   # square hollow
}

GROUP_HOLLOW = {
    'VLM':                    False,
    'VLM backbone decoder':   True,
    'standalone LLM':         False,
    'standalone LLM (think)': True,
}

GROUP_LINESTYLE = {
    'VLM':                    '-',
    'VLM backbone decoder':   ':',
    'standalone LLM':         '-',
    'standalone LLM (think)': ':',
}

# (vlm_model, lm_model, backbone_model, display_label)
TRIPLES = [
    ('llava-v1.6-mistral-7b-hf', 'llava-v1.6-mistral-7b-hf', 'Mistral-7B-Instruct-v0.2', 'Mistral-7B'),
    ('llava-v1.6-vicuna-7b-hf', 'llava-v1.6-vicuna-7b-hf', 'vicuna-7b-v1.5', 'Vicuna-7B'),
    ('Qwen3-VL-4B-Instruct', 'Qwen3-VL-4B-Instruct', 'Qwen3-4B', 'Qwen3-4B'),
    ('Qwen3-VL-8B-Instruct', 'Qwen3-VL-8B-Instruct', 'Qwen3-8B', 'Qwen3-8B'),
]
