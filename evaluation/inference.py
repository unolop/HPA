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

    def get_output(args, data, prompt): 
 
        messages = [] 
        
        if args.model_type == 'vlm':  
            messages.append({'role': 'user', 'content': f'<image>{prompt}' })
            infer_request = InferRequest(
                messages=messages,
                images=[data['image']] 
            )

        else : 
            messages.append({'role': 'user', 'content': prompt})
            infer_request = InferRequest(
                messages=messages 
            ) 

        resp_list = engine.infer([infer_request], request_config)
        response = resp_list[0].choices[0]
        output_text = response.message.content  

        matches = re.findall(r"Answer\s*:?\s*(.+)", output_text)
        if matches:
            output_text = matches[-1].strip().replace('*', '')
        else:
            output_text = output_text.strip().replace('*', '') 
        
        print(f'{prompt}\n{output_text}') 
        return output_text, response.logprobs  
    
    model = args.model
    template_type = args.template_type  # None: use the default for the model; 'qwen3_nothinking' to disable thinking
    default_system = None  # None: use the default system prompt of the corresponding model
    # When overriding attn_impl, also pass use_flash_attn=False for models like InternVL
    # whose __init__ hardcodes flash attn regardless of config (e.g. InternVLChatModel)
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
                                           model_type=args.swift_model_type) #, max_pixels=448)
    if args.lora_path is not None:
        lora_checkpoint = safe_snapshot_download(args.lora_path)  # Change to your checkpoint_dir
        model = Swift.from_pretrained(model, lora_checkpoint)
        
    model.eval()
    enable_thinking = None
    if args.template_type == 'qwen3_nothinking':
        template_type = 'qwen3'  # use the thinking-capable template
        enable_thinking = False  # but suppress thinking via empty <think></think> prefix
    else:
        template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system,
                            enable_thinking=enable_thinking)
    engine = PtEngine.from_model_template(model, template, max_batch_size=1)
    request_config = RequestConfig( max_tokens=args.max_token_length,                         
                                    logprobs=True,
                                    top_logprobs=5, 
                                    temperature=0)
    
    save_name = args.model.split('/')[-1]
    savedir = args.savedir or LOGITS_DIR
    if args.lora_path is not None : 
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

    # Which control-type keys to run (None = all string fields, as before)
    control_types = [k.strip() for k in args.control_types.split(',')] if args.control_types else None

    # --fill_missing: load existing records, only re-run keys absent from generated_answers
    existing_records = {}  # qid -> record (used in fill_missing mode)
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

    # Safety guard: --control_types without --fill_missing overwrites the existing file,
    # destroying any CTs already present. Warn loudly so this is never done by accident.
    if control_types and not args.fill_missing and not args.resume and os.path.exists(output_jsonl_path):
        existing_size = sum(1 for _ in open(output_jsonl_path, encoding='utf-8') if _.strip())
        if existing_size > 0:
            print(f"\n⚠️  WARNING: --control_types is set but --fill_missing is not.")
            print(f"   Output file already has {existing_size} records: {output_jsonl_path}")
            print(f"   Running without --fill_missing will OVERWRITE and lose existing CTs.")
            print(f"   Add --fill_missing to merge, or --resume to append. Aborting.\n")
            sys.exit(1)

    # fill_missing rewrites the whole file; otherwise append or overwrite
    write_mode = 'w' if args.fill_missing else ('a' if args.resume else 'w')

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, write_mode, encoding='utf-8') as f:

        for i, data in enumerate(tqdm(dataset)):
            data['pid'] = i
            qid = data.get('question_id', data.get('qid', data.get('idx', i)))

            if args.fill_missing:
                # Determine which keys are still missing for this record
                existing = existing_records.get(qid, {})
                already_done = set((existing.get('generated_answers') or {}).keys())
                keys_needed = control_types if control_types else None
                if keys_needed is not None:
                    keys_needed = [k for k in keys_needed if k not in already_done]
                    if not keys_needed:
                        # All requested keys already present — write existing record unchanged
                        f.write(json.dumps(existing, ensure_ascii=False) + '\n')
                        f.flush()
                        continue
                # Keep fresh data (properly formatted CT fields from VQADataset_json),
                # only carry over prior inference results from existing record.
                # Do NOT merge existing CT text fields — they may be bare text from
                # older runs and would overwrite the correctly-wrapped fresh data.
                generated_answers = dict(existing.get('generated_answers') or {})
                generated_logits = dict(existing.get('generated_logits') or {})
            else:
                if processed_ids is not None:
                    if data[current_item_id] in processed_ids:
                        print('skip ', current_item_id)
                        continue
                generated_answers = {}
                generated_logits = {}
                keys_needed = control_types  # None means all

            if 'blind' in args.condition:
                data['image'] = _IMAGE_OVERRIDE_MAP.get(args.image_override, BLANK_IMAGE)

            record_id = data.get(current_item_id, 'unknown key')
            all_fields_successful = True

            for k, v in data.items():
                if k in ['id', 'image', 'answers', 'generated_answers', 'generated_logits']:
                    continue
                if 'id' in k:
                    continue
                if not isinstance(v, str):
                    continue
                # Skip keys not in the requested control_types filter
                if keys_needed is not None and k not in keys_needed:
                    continue

                prompt = format_prompt(data, k, args.dataset, args.condition)
                try:
                    output_text, logprobs_data = get_output(args, data, prompt)
                    generated_logits[k] = clean_logprobs(logprobs_data)
                    generated_answers[k] = output_text

                except MemoryError as e:
                    print(f"OOM error processing {record_id} at {k}: {e}")
                    all_fields_successful = False
                    break

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM error processing {record_id} at {k}: {e}")
                        all_fields_successful = False
                        break
                    else:
                        print(f"Error processing {record_id} at {k}: {e}")
                        all_fields_successful = False
                        break

                except Exception as e:
                    import traceback
                    print(f"Error processing {record_id} at {k}: {e}")
                    traceback.print_exc()
                    all_fields_successful = False
                    break

            if all_fields_successful:
                print(generated_answers, generated_logits)
                data['generated_answers'] = generated_answers
                data['generated_logits'] = generated_logits
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                f.flush()
            else:
                print(f"Skipping saving record {record_id} due to previous error.")  

    print(f"모든 작업 완료. 결과가 '{output_jsonl_path}'에 저장되었습니다.")

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description="VQA Evaluation") 
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-2B", help="Model name") 
    parser.add_argument("--model_type", type=str, default="vlm")
    parser.add_argument("--resume", action="store_true") 
    parser.add_argument("--max_token_length", type=int, default=512) 
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