#!/usr/bin/env python3
"""
Extract last-input-token hidden states from VLM transformer layers.

For each sample, runs a short inference pass via PtEngine (handles image
preprocessing and template formatting) while capturing transformer layer
outputs via PyTorch forward hooks.

During the prefill pass (seq_len > 1) the hook fires once with all input
tokens. We extract the hidden state at position [-1] (the last input token,
i.e. "Answer:") for each target layer.  During decoding (seq_len == 1)
the hook is ignored.

Output: .pt file mapping
  question_id -> {
      variant_key -> {
          'hidden_states': Tensor(n_layers, d_model),  # float32, CPU
          'layers': [12, 14, ...],
      }
  }

Usage examples:
  # Blind condition, default layers
  python evaluation/extract_hidden_states.py \\
      --model Qwen/Qwen3-VL-8B-Instruct \\
      --condition _blind --dataset vqa_1k_control

  # Sighted condition
  python evaluation/extract_hidden_states.py \\
      --model Qwen/Qwen3-VL-8B-Instruct \\
      --condition '' --dataset vqa_1k_control \\
      --image_dir /path/to/val2014
"""

import gc
import json
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

# Bypass torch CVE-2025-32434 check (same as inference.py)
def _noop(): pass
try:
    import transformers.utils.import_utils as _tu
    _tu.check_torch_load_is_safe = _noop
except Exception:
    pass
try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = _noop
except Exception:
    pass

from utils import get_dataset, format_prompt, set_seed
from dataset.paths import (
    BLANK_IMAGE, GRAY_IMAGE, NOISE_IMAGE, WHITE_IMAGE, LOGITS_DIR
)

_IMAGE_OVERRIDE_MAP = {
    'blank': BLANK_IMAGE,
    'gray':  GRAY_IMAGE,
    'noise': NOISE_IMAGE,
    'white': WHITE_IMAGE,
}

DEFAULT_LAYERS = [12, 14, 16, 18, 20, 22]


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------

def find_transformer_layers(model):
    """
    Return the nn.ModuleList of transformer decoder blocks.

    Tries common attribute paths across LLaVA-1.5/1.6, Qwen3-VL, InternVL.
    Raises AttributeError with a helpful message if none match.
    """
    candidate_paths = [
        'language_model.model.layers',   # LLaVA-1.5/1.6, InternVL
        'model.layers',                  # Qwen3-VL, plain LLaMA
        'model.model.layers',            # some swift-wrapped models
        'transformer.h',                 # GPT-2 style
    ]
    for path in candidate_paths:
        obj = model
        try:
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            if hasattr(obj, '__len__') and len(obj) > 0:
                print(f"[layers] found at '{path}'  (n={len(obj)})")
                return obj
        except AttributeError:
            continue
    raise AttributeError(
        f"Cannot find transformer layers in {type(model).__name__}. "
        f"Tried: {candidate_paths}. "
        "Pass the correct path manually or add it to candidate_paths."
    )


# ---------------------------------------------------------------------------
# Hook-based capture
# ---------------------------------------------------------------------------

class HiddenStateCapture:
    """
    Context manager that registers forward hooks on selected transformer layers
    and collects last-input-token hidden states from the prefill pass.

    Usage:
        capturer = HiddenStateCapture(layers_module, [12, 14, 16])
        with capturer:
            engine.infer([req], request_config)
        hs_tensor = capturer.get_stacked()  # Tensor(n_layers, d_model)
    """

    def __init__(self, layers_module, target_layer_indices):
        self.layers_module = layers_module
        self.target_layer_indices = target_layer_indices
        self._hooks = []
        self.captured = {}   # layer_idx -> Tensor(d_model,), float32, CPU

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # Transformer block output: (hidden_states, ...) or just hidden_states
            hs = output[0] if isinstance(output, tuple) else output
            # Prefill: seq_len > 1  (all input tokens at once)
            # Decoding: seq_len == 1 (one token per step) → skip
            if hs.shape[1] > 1:
                # Last position = last input token ("Answer:" in the prompt)
                self.captured[layer_idx] = hs[0, -1, :].detach().cpu().float()
        return hook

    def __enter__(self):
        self.captured.clear()
        for idx in self.target_layer_indices:
            h = self.layers_module[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def is_complete(self):
        return set(self.captured.keys()) == set(self.target_layer_indices)

    def get_stacked(self):
        """Returns Tensor(n_layers, d_model) in target_layer_indices order."""
        return torch.stack([self.captured[i] for i in self.target_layer_indices])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed()

    from swift.llm import (
        PtEngine, RequestConfig, get_model_tokenizer, get_template, InferRequest
    )

    # --- Load model (same pattern as inference.py) ---
    extra_model_kwargs = (
        {'use_flash_attn': False}
        if args.attn_impl and args.attn_impl != 'flash_attn'
        else {}
    )
    quant_config = None
    if args.quantization_bit == 8:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization_bit == 4:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

    model, tokenizer = get_model_tokenizer(
        args.model, use_hf=True, attn_impl=args.attn_impl,
        model_kwargs=extra_model_kwargs,
        quantization_config=quant_config,
        model_type=args.swift_model_type,
    )
    model.eval()

    template_type = args.template_type
    enable_thinking = None
    if args.template_type == 'qwen3_nothinking':
        template_type = 'qwen3'
        enable_thinking = False
    else:
        template_type = template_type or model.model_meta.template
    template = get_template(
        template_type, tokenizer, enable_thinking=enable_thinking
    )

    # Short generation: we only need the prefill pass to fire the hooks.
    # max_tokens=16 keeps things safe across all answer types.
    engine = PtEngine.from_model_template(model, template, max_batch_size=1)
    request_config = RequestConfig(
        max_tokens=16, logprobs=False, temperature=0
    )

    # --- Discover and validate target layers ---
    target_layers = [int(x) for x in args.layers.split(',')]
    transformer_layers = find_transformer_layers(model)
    n_layers = len(transformer_layers)
    print(f"Model has {n_layers} transformer layers. Extracting: {target_layers}")
    out_of_range = [l for l in target_layers if not (0 <= l < n_layers)]
    if out_of_range:
        raise ValueError(
            f"Layer indices {out_of_range} out of range [0, {n_layers - 1}]"
        )

    # --- Dataset ---
    dataset = get_dataset(
        f"{args.dataset}{args.condition}",
        json_path=getattr(args, 'json_path', None),
        image_dir=getattr(args, 'image_dir', None),
    )
    control_types = (
        [k.strip() for k in args.control_types.split(',')]
        if args.control_types else None
    )

    # --- Output path ---
    save_name = args.model.split('/')[-1]
    savedir = args.savedir or str(
        Path(LOGITS_DIR).parent / 'hidden_states' / 'pretrained'
    )
    img_suffix = f'_{args.image_override}' if args.image_override != 'blank' else ''
    output_path = (
        Path(savedir) / save_name
        / f"{args.dataset}{args.condition}{img_suffix}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_path}")

    # --- Resume ---
    results = {}
    if args.resume and output_path.exists():
        results = torch.load(output_path, map_location='cpu', weights_only=True)
        print(f"Resuming: {len(results)} records already saved")

    # --- Extraction loop ---
    capturer = HiddenStateCapture(transformer_layers, target_layers)

    for i, data in enumerate(tqdm(dataset)):
        qid = str(data.get('question_id', data.get('qid', data.get('idx', i))))
        if qid in results:
            continue

        if 'blind' in args.condition:
            data['image'] = _IMAGE_OVERRIDE_MAP.get(args.image_override, BLANK_IMAGE)

        sample_result = {}

        for k, v in data.items():
            if not isinstance(v, str):
                continue
            if k in ['id', 'image', 'answers', 'generated_answers', 'generated_logits']:
                continue
            if 'id' in k:
                continue
            if control_types is not None and k not in control_types:
                continue

            prompt = format_prompt(data, k, args.dataset, args.condition)
            if args.model_type == 'vlm':
                req = InferRequest(
                    messages=[{'role': 'user', 'content': f'<image>{prompt}'}],
                    images=[data['image']],
                )
            else:
                req = InferRequest(
                    messages=[{'role': 'user', 'content': prompt}]
                )

            try:
                with capturer:
                    engine.infer([req], request_config)

                if not capturer.is_complete():
                    missing = set(target_layers) - set(capturer.captured.keys())
                    print(f"  WARNING qid={qid} key={k}: hooks missed layers {missing}")
                    continue

                sample_result[k] = {
                    'hidden_states': capturer.get_stacked(),  # (n_layers, d_model)
                    'layers': target_layers,
                }
                d_model = capturer.get_stacked().shape[-1]
                print(f"  [{qid}][{k}] shape=({len(target_layers)}, {d_model})")

            except (MemoryError, RuntimeError) as e:
                oom = isinstance(e, MemoryError) or 'out of memory' in str(e).lower()
                print(f"  {'OOM' if oom else 'Runtime'} error qid={qid} key={k}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                import traceback
                print(f"  Error qid={qid} key={k}: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache()

        if sample_result:
            results[qid] = sample_result

        if i % 50 == 0 and results:
            torch.save(results, output_path)
            print(f"  Checkpoint: {len(results)} records → {output_path}")

        if i % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    torch.save(results, output_path)
    size_mb = output_path.stat().st_size / 1e6 if output_path.exists() else 0
    print(f"\nDone. {len(results)} records → {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract VLM hidden states for activation steering"
    )
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model ID or local path')
    parser.add_argument('--model_type', type=str, default='vlm',
                        choices=['vlm', 'lm'])
    parser.add_argument('--dataset', type=str, default='vqa_1k_control')
    parser.add_argument('--condition', type=str, default='_blind',
                        help="Condition suffix, e.g. '_blind', '' for sighted, '_inst_blind'")
    parser.add_argument('--layers', type=str, default=','.join(map(str, DEFAULT_LAYERS)),
                        help='Comma-separated transformer layer indices (0-indexed). '
                             f'Default: {DEFAULT_LAYERS}')
    parser.add_argument('--savedir', type=str, default=None,
                        help='Output root directory (default: evaluation/hidden_states/pretrained)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip question_ids already present in the output file')
    parser.add_argument('--control_types', type=str, default=None,
                        help='Comma-separated variant keys to extract, e.g. '
                             '"original,pronominalized" (default: all string fields)')
    parser.add_argument('--image_override', type=str, default='blank',
                        choices=['blank', 'gray', 'noise', 'white'])
    parser.add_argument('--attn_impl', type=str, default=None,
                        help="Attention backend override, e.g. 'eager', 'sdpa'")
    parser.add_argument('--swift_model_type', type=str, default=None)
    parser.add_argument('--template_type', type=str, default=None)
    parser.add_argument('--quantization_bit', type=int, default=None,
                        help='BitsAndBytes quantization: 4 or 8')
    parser.add_argument('--json_path', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)

    args = parser.parse_args()
    main(args)
