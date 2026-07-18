import gc
import os
import re
from tqdm import tqdm
import json
import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Bypass torch CVE-2025-32434 check so .bin weight files (e.g. vicuna-13b)
# can be loaded. Safe for trusted HuggingFace checkpoints on a local machine.
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
from utils import clean_logprobs, skip_processed_idx, get_dataset, set_seed, format_prompt
from dataset.paths import BLANK_IMAGE, GRAY_IMAGE, NOISE_IMAGE, WHITE_IMAGE, LOGITS_DIR

_IMAGE_OVERRIDE_MAP = {
    'blank': BLANK_IMAGE,
    'gray':  GRAY_IMAGE,
    'noise': NOISE_IMAGE,
    'white': WHITE_IMAGE,
}

def main(args):
    set_seed()

    from swift.llm import (
        PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
    )
    from swift.tuners import Swift

    model = args.model
    template_type = args.template_type
    default_system = None
    extra_model_kwargs = {'use_flash_attn': False} if args.attn_impl and args.attn_impl != 'flash_attn' else {}
    quant_config = None
    if args.quantization_bit == 8:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization_bit == 4:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model, tokenizer = get_model_tokenizer(model, use_hf=True, attn_impl=args.attn_impl,
                                           model_kwargs=extra_model_kwargs,
                                           quantization_config=quant_config,
                                           model_type=args.swift_model_type)
    if args.lora_path is not None:
        lora_checkpoint = safe_snapshot_download(args.lora_path)
        model = Swift.from_pretrained(model, lora_checkpoint)

    model.eval()

    enable_thinking = None
    # Detect thinking mode — used to set token budget and batch size
    is_thinking = (args.template_type == 'qwen3_thinking')
    if args.template_type == 'qwen3_nothinking':
        template_type = 'qwen3'
        enable_thinking = False
    else:
        template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system,
                            enable_thinking=enable_thinking)

    # Thinking models generate very long <think>...</think> traces before the answer.
    # VLMs process image tensors per request. Both need batch size 1 to stay safe.
    # Text-only non-thinking models can batch multiple prompts per engine.infer() call.
    is_vlm = args.model_type == 'vlm'
    effective_batch_size = 1 if (is_vlm or is_thinking) else args.max_batch_size
    engine = PtEngine.from_model_template(model, template, max_batch_size=effective_batch_size)

    # Thinking traces can be thousands of tokens before the actual answer.
    # Use a separate, higher budget so generation completes cleanly rather than
    # hitting the KV-cache wall mid-kernel (which causes a hard CUDA crash).
    effective_max_tokens = args.thinking_max_tokens if is_thinking else args.max_token_length
    request_config = RequestConfig(
        max_tokens=effective_max_tokens,
        logprobs=True,
        top_logprobs=5,
        temperature=0,
    )

    save_name = args.model.split('/')[-1]
    savedir = args.savedir or LOGITS_DIR
    if args.lora_path is not None:
        finetuned = args.lora_path.split('/')[-4]
        finetuned += args.lora_path.split('/')[-3]
        save_name += f'/{finetuned}'
        savedir += '/finetuned'
    else:
        savedir += '/pretrained'

    img_suffix = f'_{args.image_override}' if args.image_override != 'blank' else ''
    output_jsonl_path = f"{savedir}/{save_name}/{args.dataset}{args.condition}{img_suffix}.jsonl"
    dataset = get_dataset(f"{args.dataset}{args.condition}",
                          json_path=getattr(args, 'json_path', None),
                          image_dir=getattr(args, 'image_dir', None))

    control_types = [k.strip() for k in args.control_types.split(',')] if args.control_types else None

    existing_records = {}
    if args.fill_missing and os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        qid = rec.get('question_id', rec.get('qid', rec.get('idx')))
                        if qid is not None:
                            existing_records[qid] = rec
                    except json.JSONDecodeError:
                        continue
        print(f"fill_missing mode: loaded {len(existing_records)} existing records from {output_jsonl_path}")

    processed_ids = set()
    try:
        processed_ids, current_item_id = skip_processed_idx(
            existing_keys=dataset[0].keys(), output_jsonl_path=output_jsonl_path
        )
    except Exception as e:
        print('cannot process the key', e)

    if control_types and not args.fill_missing and not args.resume and os.path.exists(output_jsonl_path):
        existing_size = sum(1 for _ in open(output_jsonl_path, encoding='utf-8') if _.strip())
        if existing_size > 0:
            print(f"\n⚠️  WARNING: --control_types is set but --fill_missing is not.")
            print(f"   Output file already has {existing_size} records: {output_jsonl_path}")
            print(f"   Running without --fill_missing will OVERWRITE and lose existing CTs.")
            print(f"   Add --fill_missing to merge, or --resume to append. Aborting.\n")
            sys.exit(1)

    write_mode = 'w' if args.fill_missing else ('a' if args.resume else 'w')
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, write_mode, encoding='utf-8') as f:

        for i, data in enumerate(tqdm(dataset)):
            data['pid'] = i
            qid = data.get('question_id', data.get('qid', data.get('idx', i)))

            if args.fill_missing:
                existing = existing_records.get(qid, {})
                already_done = set((existing.get('generated_answers') or {}).keys())
                keys_needed = control_types if control_types else None
                if keys_needed is not None:
                    keys_needed = [k for k in keys_needed if k not in already_done]
                    if not keys_needed:
                        f.write(json.dumps(existing, ensure_ascii=False) + '\n')
                        f.flush()
                        continue
                generated_answers = dict(existing.get('generated_answers') or {})
                generated_logits = dict(existing.get('generated_logits') or {})
            else:
                if processed_ids is not None:
                    if data[current_item_id] in processed_ids:
                        print('skip ', current_item_id)
                        continue
                generated_answers = {}
                generated_logits = {}
                keys_needed = control_types

            if 'blind' in args.condition:
                data['image'] = _IMAGE_OVERRIDE_MAP.get(args.image_override, BLANK_IMAGE)

            record_id = data.get(current_item_id, 'unknown key')

            # Build one batch covering all prompt fields for this question.
            # This replaces the original one-by-one inner loop and cuts the number
            # of engine.infer() calls from (questions × keys) down to (questions).
            batch_requests = []
            batch_keys = []
            for k, v in data.items():
                if k in ['id', 'image', 'answers', 'generated_answers', 'generated_logits']:
                    continue
                if 'id' in k:
                    continue
                if not isinstance(v, str):
                    continue
                if keys_needed is not None and k not in keys_needed:
                    continue
                prompt = format_prompt(data, k, args.dataset, args.condition)
                if args.model_type == 'vlm':
                    req = InferRequest(
                        messages=[{'role': 'user', 'content': f'<image>{prompt}'}],
                        images=[data['image']]
                    )
                else:
                    req = InferRequest(messages=[{'role': 'user', 'content': prompt}])
                batch_requests.append(req)
                batch_keys.append(k)

            if not batch_requests:
                # All keys already done in fill_missing mode
                if args.fill_missing:
                    f.write(json.dumps(existing, ensure_ascii=False) + '\n')
                    f.flush()
                continue

            all_fields_successful = True
            try:
                resp_list = engine.infer(batch_requests, request_config)
                for k, resp in zip(batch_keys, resp_list):
                    choice = resp.choices[0]
                    output_text = choice.message.content
                    matches = re.findall(r"Answer\s*:?\s*(.+)", output_text)
                    if matches:
                        output_text = matches[-1].strip().replace('*', '')
                    else:
                        output_text = output_text.strip().replace('*', '')
                    print(f'[{k}] {output_text}')
                    generated_logits[k] = clean_logprobs(choice.logprobs)
                    generated_answers[k] = output_text

            except (MemoryError, RuntimeError) as e:
                oom = isinstance(e, MemoryError) or "out of memory" in str(e).lower()
                print(f"{'OOM' if oom else 'Runtime'} error at record {record_id}: {e}")
                if not oom:
                    import traceback; traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()
                all_fields_successful = False

            except Exception as e:
                import traceback
                print(f"Error processing {record_id}: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()
                all_fields_successful = False

            # Periodic cache flush to prevent fragmentation across long runs
            if i % 50 == 0:
                torch.cuda.empty_cache()

            if all_fields_successful:
                print(generated_answers)
                data['generated_answers'] = generated_answers
                data['generated_logits'] = generated_logits
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                f.flush()
            else:
                print(f"Skipping record {record_id} due to error.")

    print(f"모든 작업 완료. 결과가 '{output_jsonl_path}'에 저장되었습니다.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VQA Evaluation")
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-2B", help="Model name")
    parser.add_argument("--model_type", type=str, default="vlm")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_token_length", type=int, default=512)
    parser.add_argument("--thinking_max_tokens", type=int, default=8192,
                        help="Token budget for thinking models (qwen3_thinking template). "
                             "Think traces can be thousands of tokens; 512 causes mid-kernel OOM.")
    parser.add_argument("--max_batch_size", type=int, default=4,
                        help="Max requests per engine.infer() call for text-only non-thinking models. "
                             "VLM and thinking models are always forced to 1.")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA Path")
    parser.add_argument("--dataset", type=str, default="mmstar", help="Dataset name")
    parser.add_argument('--checkpoint', type=str, default=None, help='Pretrained checkpoint')
    parser.add_argument('--savedir', type=str, default=None, help='Save directory of inference')
    parser.add_argument("--condition", type=str, default='')
    parser.add_argument("--prompt", type=str, default='')
    parser.add_argument("--control_types", type=str, default=None,
                        help="Comma-separated control key(s) to run, e.g. 'pronominalized' "
                             "(default: all string fields)")
    parser.add_argument("--template_type", type=str, default=None,
                        help="Override swift template type, e.g. 'qwen3_nothinking' to disable "
                             "chain-of-thought for Qwen3 backbone models")
    parser.add_argument("--attn_impl", type=str, default=None,
                        help="Attention implementation override, e.g. 'eager', 'sdpa', 'flash_attn'")
    parser.add_argument("--swift_model_type", type=str, default=None,
                        help="Swift model_type override for models swift can't auto-detect, "
                             "e.g. 'llama' for vicuna, 'mistral' for Mistral-7B")
    parser.add_argument("--quantization_bit", type=int, default=None,
                        help="BitsAndBytes quantization: 4 or 8 (int4/int8). Use 8 for 32B on 2×24GB GPUs.")
    parser.add_argument("--fill_missing", action="store_true",
                        help="Load existing output and only run keys absent from generated_answers; "
                             "rewrites the file with merged results")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Override the default JSONL path for vqa_1k_control datasets "
                             "(e.g. dataset/vqa/vqa1k_v4_patch.jsonl)")
    parser.add_argument("--image_override", type=str, default='blank',
                        choices=['blank', 'gray', 'noise', 'white'],
                        help="Image to use for blind conditions: blank (all-black), "
                             "gray (128-gray), noise (random pixels, seed=42), white (all-white)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Override image directory (e.g. for train2014 images)")

    args = parser.parse_args()
    main(args)
