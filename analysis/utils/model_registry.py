"""Shared model registries and model-type groupings for S2 analysis."""

from __future__ import annotations

from pathlib import Path

MODEL_TYPE = {
    'Qwen3-VL-8B': 'VLM',
    'LLaVA-1.5-7B': 'VLM',
    'LLaVA-Mistral': 'VLM',
    'LLaVA-Vicuna': 'VLM',
    'InternVL-1B': 'VLM',
    'InternVL-2B': 'VLM',
    'InternVL-8B': 'VLM',
    'LLaVA-1.5 (LM)': 'LM-decoder',
    'LLaVA-Mistral (LM)': 'LM-decoder',
    'LLaVA-Vicuna (LM)': 'LM-decoder',
    'Qwen3-8B': 'Backbone LLM',
}


def default_all_models(base: Path) -> dict:
    """Return the canonical all-model registry keyed by display label."""
    base = Path(base)
    return {
        'Qwen3-VL-8B':        (base / 'evaluation/logits/pretrained', 'Qwen3-VL-8B-Instruct'),
        'LLaVA-1.5-7B':       (base / 'evaluation/logits/pretrained', 'llava-1.5-7b-hf'),
        'LLaVA-Mistral':      (base / 'evaluation/logits/pretrained', 'llava-v1.6-mistral-7b-hf'),
        'LLaVA-Vicuna':       (base / 'evaluation/logits/pretrained', 'llava-v1.6-vicuna-7b-hf'),
        'InternVL-1B':        (base / 'evaluation/logits/pretrained', 'InternVL3_5-1B'),
        'InternVL-2B':        (base / 'evaluation/logits/pretrained', 'InternVL3_5-2B'),
        'InternVL-8B':        (base / 'evaluation/logits/pretrained', 'InternVL3_5-8B'),
        'LLaVA-1.5 (LM)':     (base / 'evaluation/logits/lm_decoder/pretrained', 'llava-1.5-7b-hf'),
        'LLaVA-Mistral (LM)': (base / 'evaluation/logits/lm_decoder/pretrained', 'llava-v1.6-mistral-7b-hf'),
        'LLaVA-Vicuna (LM)':  (base / 'evaluation/logits/lm_decoder/pretrained', 'llava-v1.6-vicuna-7b-hf'),
        'Qwen3-8B':           (base / 'evaluation/logits/backbone/pretrained', 'Qwen3-8B'),
    }


def flatten_all_models(base: Path) -> dict:
    """Return display label -> concrete model results directory."""
    return {label: root / model_dir for label, (root, model_dir) in default_all_models(base).items()}
