"""Shared model registries and model-type groupings for S2 analysis."""

from __future__ import annotations

from pathlib import Path

MODEL_TYPE = {
    # VLM (full vision-language model)
    'Qwen3-VL-8B':        'VLM',
    'LLaVA-1.5-7B':       'VLM',
    'LLaVA-Mistral':      'VLM',
    'LLaVA-Vicuna':       'VLM',
    'InternVL-1B':        'VLM',
    'InternVL-2B':        'VLM',
    'InternVL-8B':        'VLM',
    # VLM backbone decoder (VLM language head, vision bypassed)
    'LLaVA-1.5 (LM)':    'VLM backbone decoder',
    'LLaVA-Mistral (LM)': 'VLM backbone decoder',
    'LLaVA-Vicuna (LM)':  'VLM backbone decoder',
    # standalone LLM — nothink (default)
    'Qwen3-0.6B':          'standalone LLM',
    'Qwen3-1.7B':          'standalone LLM',
    'Qwen3-4B':            'standalone LLM',
    'Qwen3-8B':            'standalone LLM',
    'Qwen3-32B':           'standalone LLM',
    'Mistral-7B':          'standalone LLM',
    'Vicuna-7B':           'standalone LLM',
    'Vicuna-13B':          'standalone LLM',
    'Qwen2.5-7B':          'standalone LLM',
    'Phi-3.5-mini':        'standalone LLM',
    # note: inference.py saves using HF model id leaf: Qwen2.5-7B-Instruct / Phi-3.5-mini-instruct
    # standalone LLM — think variants
    'Qwen3-0.6B (think)':  'standalone LLM',
    'Qwen3-1.7B (think)':  'standalone LLM',
    'Qwen3-4B (think)':    'standalone LLM',
    'Qwen3-8B (think)':    'standalone LLM',
    'Qwen3-32B (think)':   'standalone LLM',
}


def backbone_think_models(base: Path) -> dict:
    """Qwen3 backbone models run WITH thinking tokens (chain-of-thought enabled)."""
    base = Path(base)
    bt = base / 'evaluation/logits/backbone_think/pretrained'
    return {
        'Qwen3-0.6B (think)':  (bt, 'Qwen3-0.6B'),
        'Qwen3-1.7B (think)':  (bt, 'Qwen3-1.7B'),
        'Qwen3-4B (think)':    (bt, 'Qwen3-4B'),
        'Qwen3-8B (think)':    (bt, 'Qwen3-8B'),
        'Qwen3-32B (think)':   (bt, 'Qwen3-32B'),
    }


def backbone_nothink_models(base: Path) -> dict:
    """All backbone models run WITHOUT thinking tokens (clean outputs)."""
    base = Path(base)
    bn = base / 'evaluation/logits/backbone_nothink/pretrained'
    return {
        'Qwen3-0.6B':          (bn, 'Qwen3-0.6B'),
        'Qwen3-1.7B':          (bn, 'Qwen3-1.7B'),
        'Qwen3-4B':            (bn, 'Qwen3-4B'),
        'Qwen3-8B':            (bn, 'Qwen3-8B'),
        'Qwen3-32B':           (bn, 'Qwen3-32B'),
        'Mistral-7B':          (bn, 'Mistral-7B-Instruct-v0.2'),
        'Vicuna-7B':           (bn, 'vicuna-7b-v1.5'),
        'Vicuna-13B':          (bn, 'vicuna-13b-v1.5'),
        'Qwen2.5-7B':          (bn, 'Qwen2.5-7B-Instruct'),
        'Phi-3.5-mini':        (bn, 'Phi-3.5-mini-instruct'),
    }


def default_all_models(base: Path) -> dict:
    """Return the canonical all-model registry keyed by display label.

    Only includes models with confirmed inference results.
    Models missing inst_blind (vicuna-*b, Qwen3-32B) are included but will
    be skipped automatically by load_model_results when data is absent.
    """
    base = Path(base)
    lm = base / 'evaluation/logits/lm_decoder/pretrained'
    pt = base / 'evaluation/logits/pretrained'
    return {
        # ── VLM ──────────────────────────────────────────────────────────────
        'Qwen3-VL-8B':        (pt, 'Qwen3-VL-8B-Instruct'),
        'LLaVA-1.5-7B':       (pt, 'llava-1.5-7b-hf'),
        'LLaVA-Mistral':      (pt, 'llava-v1.6-mistral-7b-hf'),
        'LLaVA-Vicuna':       (pt, 'llava-v1.6-vicuna-7b-hf'),
        'InternVL-1B':        (pt, 'InternVL3_5-1B'),
        'InternVL-2B':        (pt, 'InternVL3_5-2B'),
        'InternVL-8B':        (pt, 'InternVL3_5-8B'),
        # ── VLM backbone decoder ─────────────────────────────────────────────
        'LLaVA-1.5 (LM)':    (lm, 'llava-1.5-7b-hf'),
        'LLaVA-Mistral (LM)':(lm, 'llava-v1.6-mistral-7b-hf'),
        'LLaVA-Vicuna (LM)': (lm, 'llava-v1.6-vicuna-7b-hf'),
        # ── standalone LLM ───────────────────────────────────────────────────
        **backbone_nothink_models(base),
    }


def flatten_all_models(base: Path) -> dict:
    """Return display label -> concrete model results directory."""
    return {label: root / model_dir for label, (root, model_dir) in default_all_models(base).items()}
