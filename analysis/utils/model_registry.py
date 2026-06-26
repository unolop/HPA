"""Shared model registries and model-type groupings for S2 analysis."""

from __future__ import annotations

from pathlib import Path

MODEL_TYPE = {
    # VLM (full vision-language model)
    'Qwen3-VL-2B':        'VLM',
    'Qwen3-VL-4B':        'VLM',
    'Qwen3-VL-8B':        'VLM',
    'Qwen3-VL-32B':       'VLM',
    'LLaVA-1.5-7B':       'VLM',
    'LLaVA-Mistral':      'VLM',
    'LLaVA-Vicuna':       'VLM',
    'LLaVA-Vicuna-13B':   'VLM',
    'InternVL-1B':        'VLM',
    'InternVL-2B':        'VLM',
    'InternVL-8B':        'VLM',
    # VLM backbone decoder (VLM language head, vision bypassed)
    'Qwen3-VL-2B (LM)':   'VLM backbone decoder',
    'Qwen3-VL-4B (LM)':   'VLM backbone decoder',
    'Qwen3-VL-8B (LM)':   'VLM backbone decoder',
    'Qwen3-VL-32B (LM)':  'VLM backbone decoder',
    'LLaVA-1.5 (LM)':    'VLM backbone decoder',
    'LLaVA-Mistral (LM)': 'VLM backbone decoder',
    'LLaVA-Vicuna (LM)':      'VLM backbone decoder',
    'LLaVA-Vicuna-13B (LM)':  'VLM backbone decoder',
    'InternVL-1B (LM)':   'VLM backbone decoder',
    'InternVL-2B (LM)':   'VLM backbone decoder',
    'InternVL-8B (LM)':   'VLM backbone decoder',
    # standalone LLM — nothink (default, no chain-of-thought)
    'Qwen3-0.6B':          'standalone LLM',
    'Qwen3-1.7B':          'standalone LLM',
    'Qwen3-4B':            'standalone LLM',
    'Qwen3-8B':            'standalone LLM',
    'Qwen3-32B':           'standalone LLM',
    'Mistral-7B':          'standalone LLM',
    'Vicuna-7B':           'standalone LLM',
    'Vicuna-13B':          'standalone LLM',
    'Qwen2.5-7B-Instruct': 'standalone LLM',
    'Phi-3.5-mini':        'standalone LLM',
    # standalone LLM — think variants (chain-of-thought enabled)
    'Qwen3-0.6B (think)':  'standalone LLM (think)',
    'Qwen3-1.7B (think)':  'standalone LLM (think)',
    'Qwen3-4B (think)':    'standalone LLM (think)',
    'Qwen3-8B (think)':    'standalone LLM (think)',
    'Qwen3-32B (think)':   'standalone LLM (think)',
}


def backbone_think_models(base: Path) -> dict:
    """Qwen3 backbone models run WITH thinking tokens (chain-of-thought enabled).
    Stored under backbone/pretrained/ with _think suffix in the model dir name.
    """
    base = Path(base)
    bn = base / 'evaluation/logits/backbone/pretrained'
    return {
        'Qwen3-0.6B (think)':  (bn, 'Qwen3-0.6B_think'),
        'Qwen3-1.7B (think)':  (bn, 'Qwen3-1.7B_think'),
        'Qwen3-4B (think)':    (bn, 'Qwen3-4B_think'),
        'Qwen3-8B (think)':    (bn, 'Qwen3-8B_think'),
        'Qwen3-32B (think)':   (bn, 'Qwen3-32B_think'),
    }


def backbone_models(base: Path) -> dict:
    """All backbone models run WITHOUT thinking tokens (clean outputs)."""
    base = Path(base)
    bn = base / 'evaluation/logits/backbone/pretrained'
    return {
        'Qwen3-0.6B':          (bn, 'Qwen3-0.6B'),
        'Qwen3-1.7B':          (bn, 'Qwen3-1.7B'),
        'Qwen3-4B':            (bn, 'Qwen3-4B'),
        'Qwen3-8B':            (bn, 'Qwen3-8B'),
        'Qwen3-32B':           (bn, 'Qwen3-32B'),
        'Mistral-7B':          (bn, 'Mistral-7B-Instruct-v0.2'),
        'Vicuna-7B':           (bn, 'vicuna-7b-v1.5'),
        'Vicuna-13B':          (bn, 'vicuna-13b-v1.5'),
        'Qwen2.5-7B-Instruct': (bn, 'Qwen2.5-7B-Instruct'),
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
    pt = base / 'evaluation/logits/vlm/pretrained'
    return {
        # ── VLM ──────────────────────────────────────────────────────────────
        'Qwen3-VL-2B':        (pt, 'Qwen3-VL-2B-Instruct'),
        'Qwen3-VL-4B':        (pt, 'Qwen3-VL-4B-Instruct'),
        'Qwen3-VL-8B':        (pt, 'Qwen3-VL-8B-Instruct'),
        'Qwen3-VL-32B':       (pt, 'Qwen3-VL-32B-Instruct'),
        'LLaVA-1.5-7B':       (pt, 'llava-1.5-7b-hf'),
        'LLaVA-Mistral':      (pt, 'llava-v1.6-mistral-7b-hf'),
        'LLaVA-Vicuna':       (pt, 'llava-v1.6-vicuna-7b-hf'),
        'LLaVA-Vicuna-13B':   (pt, 'llava-v1.6-vicuna-13b-hf'),
        'InternVL-1B':        (pt, 'InternVL3_5-1B'),
        'InternVL-2B':        (pt, 'InternVL3_5-2B'),
        'InternVL-8B':        (pt, 'InternVL3_5-8B'),
        # ── VLM backbone decoder ─────────────────────────────────────────────
        'Qwen3-VL-2B (LM)':  (lm, 'Qwen3-VL-2B-Instruct'),
        'Qwen3-VL-4B (LM)':  (lm, 'Qwen3-VL-4B-Instruct'),
        'Qwen3-VL-8B (LM)':  (lm, 'Qwen3-VL-8B-Instruct'),
        'Qwen3-VL-32B (LM)': (lm, 'Qwen3-VL-32B-Instruct'),
        'LLaVA-1.5 (LM)':    (lm, 'llava-1.5-7b-hf'),
        'LLaVA-Mistral (LM)':(lm, 'llava-v1.6-mistral-7b-hf'),
        'LLaVA-Vicuna (LM)':     (lm, 'llava-v1.6-vicuna-7b-hf'),
        'LLaVA-Vicuna-13B (LM)': (lm, 'llava-v1.6-vicuna-13b-hf'),
        'InternVL-1B (LM)':  (lm, 'InternVL3_5-1B'),
        'InternVL-2B (LM)':  (lm, 'InternVL3_5-2B'),
        'InternVL-8B (LM)':  (lm, 'InternVL3_5-8B'),
        # ── standalone LLM (nothink) ─────────────────────────────────────────
        **backbone_models(base),
        # ── standalone LLM (think / chain-of-thought) ────────────────────────
        **backbone_think_models(base),
    }


def model_groups(base: Path) -> dict:
    """Return the canonical model-group registry for plotting/analysis.

    Groups all models from default_all_models() by MODEL_TYPE, each entry
    exposing the shared results directory and the label→model_dir mapping.

    Returns
    -------
    dict[group_name → {'dir': Path, 'models': {display_label: model_dir_name}}]

    Groups (in insertion order):
      'VLM'                    — full vision-language models
      'VLM backbone decoder'   — VLM LM head with vision bypassed
      'standalone LLM'         — text-only LLMs, nothink mode
      'standalone LLM (think)' — text-only LLMs, chain-of-thought enabled
    """
    base = Path(base)
    groups: dict = {}
    for label, (results_dir, mdir) in default_all_models(base).items():
        gname = MODEL_TYPE.get(label)
        if gname is None:
            continue
        if gname not in groups:
            groups[gname] = {'dir': results_dir, 'models': {}}
        groups[gname]['models'][label] = mdir
    return groups


def flatten_all_models(base: Path) -> dict:
    """Return display label -> concrete model results directory."""
    return {label: root / model_dir for label, (root, model_dir) in default_all_models(base).items()}
