"""
analysis/utils — shared utilities for HPA analysis notebooks.

Modules
-------
model_registry  : canonical model list, group assignments (VLM / backbone / standalone / think)
load_session    : data loaders for human study + model results (load_human_data, load_model_results)
vqa             : VQA answer normalisation, accuracy scoring (preprocess_answer, vqa_accuracy)
agreement       : pairwise inter-rater agreement (SBERT, chrF, ROUGE-1, BERTScore, SimCSE, Jaccard)
abstention      : classify model outputs (hard/soft abstain, hallucinated correct/wrong, degenerate)
constants       : plotting constants — colours, orders, labels for conditions/variants/groups
corr            : Pearson/Spearman correlation helpers for human-model difficulty alignment
quadrants       : 2×2 per-question quadrant assignment (human×model accuracy)
normalize_korean: Korean text normalisation for human participant answers
"""

from .model_registry import (
    MODEL_TYPE,
    default_all_models,
    backbone_models,
    backbone_think_models,
    model_groups,
    flatten_all_models,
)

from .load_session import (
    load_human_data,
    load_model_results,
)

from .vqa import (
    preprocess_answer,
    vqa_accuracy,
    VQAAnswerMapper,
)

from .abstention import (
    classify,
    is_abstained,
    SOFT_WORDS,
)

from .constants import (
    MODEL_ORDER,
    MODEL_LABEL,
    COND_LABEL,
    COND_COLOR,
    CONDITIONS,
    VARIANT_ORDER,
    VARIANT_LABELS,
    VARIANT_COLORS,
    GROUP_COLORS,
    GROUP_ORDER,
    CLASS_ORDER,
    CLASS_COLOR,
    CLASS_LABEL,
    CONTROL_TYPES,
    CT_LABELS,
    CT_TO_VARIANT,
    ABSTAIN_TOKENS,
    TIER_ORDER,
    TIER_COLORS,
    TIER_STYLE,
    VLM_MODELS,
    LM_MODELS,
    BB_MODELS,
    TRIPLES,
    MODEL_FAMILY,
    MODEL_FAMILY_COLORS,
    MODEL_SIZE_B,
    GROUP_MARKER,
)

__all__ = [
    # model registry
    'MODEL_TYPE', 'default_all_models', 'backbone_models',
    'backbone_think_models', 'model_groups', 'flatten_all_models',
    # data loading
    'load_human_data', 'load_model_results',
    # vqa scoring
    'preprocess_answer', 'vqa_accuracy', 'VQAAnswerMapper',
    # abstention
    'classify', 'is_abstained', 'SOFT_WORDS',
    # constants
    'MODEL_ORDER', 'MODEL_LABEL', 'COND_LABEL', 'COND_COLOR', 'CONDITIONS',
    'VARIANT_ORDER', 'VARIANT_LABELS', 'VARIANT_COLORS',
    'GROUP_COLORS', 'GROUP_ORDER', 'CLASS_ORDER', 'CLASS_COLOR', 'CLASS_LABEL',
    'CONTROL_TYPES', 'CT_LABELS', 'CT_TO_VARIANT', 'ABSTAIN_TOKENS',
    'TIER_ORDER', 'TIER_COLORS', 'TIER_STYLE',
    'VLM_MODELS', 'LM_MODELS', 'BB_MODELS', 'TRIPLES',
    'MODEL_FAMILY', 'MODEL_FAMILY_COLORS', 'MODEL_SIZE_B', 'GROUP_MARKER',
]
