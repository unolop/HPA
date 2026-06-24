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

from typing import Iterable, Optional

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
    'standalone LLM (think)': '#1E88E5',   # blue
}

GROUP_ORDER = ['VLM', 'VLM backbone decoder', 'standalone LLM', 'standalone LLM (think)']
GROUP_PAIR_ORDER = [
    ('VLM', 'VLM backbone decoder'),
    ('standalone LLM', 'standalone LLM (think)'),
]
GROUP_PAIR_PLOTS = [
    ('vlm_backbone', 'VLM + Backbone decoder', ('VLM', 'VLM backbone decoder')),
    ('llm_think', 'Standalone LLM + Think', ('standalone LLM', 'standalone LLM (think)')),
]

# Condition-aware plotting scheme for accuracy_vABC exports.
# - blind / inst_blind: paired plots
# - control: VLM-only plot (no backbone matching requirement)
ACCURACY_VABC_PLOT_SPECS = {
    'vlm_backbone': {
        'kind': 'paired',
        'title': 'VLM + Backbone decoder',
        'groups': ('VLM', 'VLM backbone decoder'),
    },
    'llm_think': {
        'kind': 'paired_qwen_think',
        'title': 'Standalone LLM + Think',
        'groups': ('standalone LLM', 'standalone LLM (think)'),
    },
    'vlm': {
        'kind': 'single_group',
        'title': 'VLM',
        'groups': ('VLM',),
    },
}

ACCURACY_VABC_COND_SCHEME = {
    'blind': ['vlm_backbone', 'llm_think'],
    'inst_blind': ['vlm_backbone', 'llm_think'],
    'control': ['vlm'],
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
TIER_ORDER = GROUP_ORDER

TIER_COLORS = GROUP_COLORS  # alias — use GROUP_COLORS everywhere

TIER_STYLE = {
    'VLM':                    {'color': '#E53935', 'marker': 'o', 'ls': '-',  'lw': 2.2, 'hollow': False},
    'VLM backbone decoder':   {'color': '#E67E22', 'marker': 'o', 'ls': ':',  'lw': 1.8, 'hollow': True},
    'standalone LLM':         {'color': '#2E7D32', 'marker': 's', 'ls': '-',  'lw': 1.8, 'hollow': False},
    'standalone LLM (think)': {'color': '#1E88E5', 'marker': 's', 'ls': ':',  'lw': 1.8, 'hollow': True},
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
    'Qwen3-VL-2B (LM)':     'Qwen3-VL',
    'Qwen3-VL-4B (LM)':     'Qwen3-VL',
    'Qwen3-VL-8B (LM)':     'Qwen3-VL',
    'Qwen3-VL-32B (LM)':    'Qwen3-VL',
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
    'LLaVA-Vicuna-13B':     'LLaVA-Vicuna',
    'LLaVA-Vicuna-13B (LM)':'LLaVA-Vicuna',
    # InternVL
    'InternVL-1B':          'InternVL',
    'InternVL-2B':          'InternVL',
    'InternVL-8B':          'InternVL',
    'InternVL-1B (LM)':     'InternVL',
    'InternVL-2B (LM)':     'InternVL',
    'InternVL-8B (LM)':     'InternVL',
    # Others
    'Mistral-7B':           'Mistral',
    'Vicuna-7B':           'Vicuna',
    'Vicuna-13B':           'Vicuna',
    'Phi-3.5-mini':         'Phi',
    'Qwen2.5-7B':           'Qwen2.5',
    'Qwen2.5-7B-Instruct':  'Qwen2.5',
}

# ── Model family colors ───────────────────────────────────────────────────────
MODEL_FAMILY_COLORS = {
    'Qwen3-VL':  '#5C6BC0',   # indigo
    'Qwen3':     '#0288D1',   # light blue
    'LLaVA-1.5':     '#B71C1C',   # deep crimson
    'LLaVA-Mistral': '#E53935',   # medium red
    'LLaVA-Vicuna':  '#FF7043',   # red-orange
    'InternVL':  '#F9A825',   # amber yellow
    'Mistral':   '#EF5350',   # soft red  — same hue as LLaVA-Mistral (#E53935)
    'Vicuna':    '#FF8A65',   # soft orange-red — same hue as LLaVA-Vicuna (#FF7043)
    'Phi':       '#78909C',   # blue-grey — no VLM counterpart, kept neutral
    'Qwen2.5':   '#558B2F',   # olive green
    'Human':     '#1565C0',   # blue (matches GROUP_COLORS human convention)
}

# ── Question t-SNE styling ───────────────────────────────────────────────────
TSNE_ENTITY_COLORS = {
    'person':  '#4E79A7',
    'object':  '#F28E2B',
    'animal':  '#59A14F',
    'food':    '#E15759',
    'place':   '#76B7B2',
    'other':   '#EDC948',
    'vehicle': '#B07AA1',
    'product': '#FF9DA7',
    'text':    '#9C755F',
}

TSNE_OP_COLORS = {
    'attr':   '#4E79A7',
    'ident':  '#F28E2B',
    'count':  '#E15759',
    'spat':   '#76B7B2',
    'act':    '#59A14F',
    'exist':  '#EDC948',
    'text':   '#B07AA1',
    'know':   '#FF9DA7',
    'comp':   '#9C755F',
    'cause':  '#BAB0AC',
    'temp':   '#D37295',
    'other':  '#A0CBE8',
}

TSNE_POOL_POINT_STYLE = {
    'marker': 'o',
    'size': 10,
    'alpha': 0.12,
    'edgecolor': 'none',
    'linewidth': 0.0,
}

TSNE_SUBSET_POINT_STYLE = {
    'marker': 'o',
    'size': 62,
    'alpha': 0.95,
    'edgecolor': 'none',
    'linewidth': 0.0,
}

TSNE_LABEL_STYLE = {
    'fontsize': 10.0,
    'fontweight': 'normal',
    'color': '#2F2F2F',
    'min_wrap_width': 18,
    'max_wrap_width': 42,
    'long_question_words': 8,
    'bbox_alpha': 0.22,
    'arrow_lw': 0.7,
    'arrow_alpha': 0.55,
    'radial_offset_x': 34,
    'radial_offset_y': 24,
    'offset_radius_points': (28, 40, 54, 70, 88),
    'offset_angle_jitter_deg': (-40, -22, 0, 22, 40, 65, -65),
    'min_box_gap_px': 10.0,
}

TSNE_LEGEND_STYLE = {
    'fontsize': 10.5,
    'markersize': 7.5,
    'ncol_max': 12,
    'handletextpad': 0.45,
    'columnspacing': 1.1,
    'bbox_y': -0.045,
}

TSNE_FIGURE_STYLE = {
    'single_figsize': (13.5, 6.7),
    'combined_figsize': (24.5, 6.8),
    'dpi': 180,
}

# ── Model parameter sizes (billions) ─────────────────────────────────────────
MODEL_SIZE_B = {
    'Qwen3-VL-2B':          2.0,
    'Qwen3-VL-4B':          4.0,
    'Qwen3-VL-8B':          8.0,
    'Qwen3-VL-32B':        32.0,
    'Qwen3-VL-2B (LM)':     2.0,
    'Qwen3-VL-4B (LM)':     4.0,
    'Qwen3-VL-8B (LM)':     8.0,
    'Qwen3-VL-32B (LM)':   32.0,
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
    'LLaVA-Vicuna-13B':    13.0,
    'LLaVA-1.5 (LM)':       7.0,
    'LLaVA-Mistral (LM)':   7.0,
    'LLaVA-Vicuna (LM)':    7.0,
    'LLaVA-Vicuna-13B (LM)':13.0,
    'InternVL-1B':          1.0,
    'InternVL-2B':          2.0,
    'InternVL-8B':          8.0,
    'InternVL-1B (LM)':     1.0,
    'InternVL-2B (LM)':     2.0,
    'InternVL-8B (LM)':     8.0,
    'Mistral-7B':           7.0,
    'Vicuna-7B':            7.0,
    'Vicuna-13B':          13.0,
    'Phi-3.5-mini':         3.8,
    'Qwen2.5-7B':           7.0,
    'Qwen2.5-7B-Instruct':  7.0,
}

# ── Scale-aware per-model styling (for non-7B per-model lineplots) ──────────
SCALE_STYLE_ALPHA_RANGE = (0.38, 0.92)
SCALE_STYLE_MARKERSIZE_RANGE = (5.0, 8.8)
SCALE_STYLE_LINEWIDTH_RANGE = (1.2, 2.2)
SCALE_STYLE_FILL_ALPHA_RANGE = (0.03, 0.12)
SCALE_STYLE_GROUPS = {'standalone LLM', 'standalone LLM (think)'}


def _reference_max_scale(reference_models: Optional[Iterable[str]] = None) -> Optional[float]:
    if reference_models is None:
        sizes = list(MODEL_SIZE_B.values())
    else:
        sizes = [MODEL_SIZE_B[m] for m in reference_models if m in MODEL_SIZE_B]
    if not sizes:
        return None
    return max(sizes)


def model_scale_fraction(model: str, reference_models: Optional[Iterable[str]] = None) -> float:
    """Return model size as a fraction of the max size in the given reference set."""
    size = MODEL_SIZE_B.get(model)
    max_scale = _reference_max_scale(reference_models)
    if size is None or max_scale in (None, 0):
        return 1.0
    frac = size / max_scale
    return max(0.0, min(1.0, frac))


def model_scale_alpha(model: str, reference_models: Optional[Iterable[str]] = None) -> float:
    lo, hi = SCALE_STYLE_ALPHA_RANGE
    frac = model_scale_fraction(model, reference_models)
    return lo + frac * (hi - lo)


def model_scale_markersize(model: str, reference_models: Optional[Iterable[str]] = None) -> float:
    lo, hi = SCALE_STYLE_MARKERSIZE_RANGE
    frac = model_scale_fraction(model, reference_models)
    return lo + frac * (hi - lo)


def model_scale_linewidth(model: str, reference_models: Optional[Iterable[str]] = None) -> float:
    lo, hi = SCALE_STYLE_LINEWIDTH_RANGE
    frac = model_scale_fraction(model, reference_models)
    return lo + frac * (hi - lo)


def model_scale_fill_alpha(model: str, reference_models: Optional[Iterable[str]] = None) -> float:
    lo, hi = SCALE_STYLE_FILL_ALPHA_RANGE
    frac = model_scale_fraction(model, reference_models)
    return lo + frac * (hi - lo)


def model_scale_style(model: str, reference_models: Optional[Iterable[str]] = None) -> dict:
    return {
        'alpha': model_scale_alpha(model, reference_models),
        'markersize': model_scale_markersize(model, reference_models),
    }


def model_scale_line_style(model: str, reference_models: Optional[Iterable[str]] = None) -> dict:
    """Scale-aware line plotting style for per-model curves."""
    return {
        'line_alpha': model_scale_alpha(model, reference_models),
        'linewidth': model_scale_linewidth(model, reference_models),
        'fill_alpha': model_scale_fill_alpha(model, reference_models),
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


def paired_base_model_name(model: str, group: Optional[str] = None) -> str:
    """Normalize paired-model labels across related groups for plot matching."""
    base = model
    if base.endswith(' (LM)'):
        base = base[:-5]
    if base.endswith(' (think)'):
        base = base[:-8]
    # Keep display names aligned across paired VLM/backbone labels.
    if base == 'LLaVA-1.5-7B':
        return 'LLaVA-1.5'
    return base

# (vlm_model, lm_model, backbone_model, display_label)
TRIPLES = [
    ('llava-v1.6-mistral-7b-hf', 'llava-v1.6-mistral-7b-hf', 'Mistral-7B-Instruct-v0.2', 'Mistral-7B'),
    ('llava-v1.6-vicuna-7b-hf', 'llava-v1.6-vicuna-7b-hf', 'vicuna-7b-v1.5', 'Vicuna-7B'),
    ('Qwen3-VL-4B-Instruct', 'Qwen3-VL-4B-Instruct', 'Qwen3-4B', 'Qwen3-4B'),
    ('Qwen3-VL-8B-Instruct', 'Qwen3-VL-8B-Instruct', 'Qwen3-8B', 'Qwen3-8B'),
]
