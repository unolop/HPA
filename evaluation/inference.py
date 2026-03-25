import os
import re
from tqdm import tqdm
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import clean_logprobs, skip_processed_idx, get_dataset, set_seed, format_prompt
from dataset.paths import BLANK_IMAGE, LOGITS_DIR

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
    template_type = None  # None: use the default template_type of the corresponding model
    default_system = None  # None: use the default system prompt of the corresponding model
    # When overriding attn_impl, also pass use_flash_attn=False for models like InternVL
    # whose __init__ hardcodes flash attn regardless of config (e.g. InternVLChatModel)
    extra_model_kwargs = {'use_flash_attn': False} if args.attn_impl and args.attn_impl != 'flash_attn' else {}
    model, tokenizer = get_model_tokenizer(model, use_hf=True, attn_impl=args.attn_impl,
                                           model_kwargs=extra_model_kwargs) #, max_pixels=448)
    if args.lora_path is not None:
        lora_checkpoint = safe_snapshot_download(args.lora_path)  # Change to your checkpoint_dir
        model = Swift.from_pretrained(model, lora_checkpoint)
        
    model.eval()
    template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system)
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

    output_jsonl_path = f"{savedir}/{save_name}/{args.dataset}{args.condition}.jsonl"
    dataset = get_dataset(f"{args.dataset}{args.condition}")

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
                # Merge: start from existing record so we preserve all previous outputs
                data = {**data, **(existing or {})}
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
                data['image'] = BLANK_IMAGE

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
                    print(f"Error processing {record_id} at {k}: {e}")
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
    parser.add_argument("--max_token_length", default=512) 
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA Path") 
    parser.add_argument("--dataset", type=str, default="mmstar", help="Dataset name") 
    parser.add_argument('--checkpoint', type=str, default=None, help='Pretrained checkpoint') 
    parser.add_argument('--savedir', type=str, default=None, help='Save directory of inference')
    parser.add_argument("--condition", type=str, default='')
    parser.add_argument("--prompt", type=str, default='')
    parser.add_argument("--control_types", type=str, default=None,
                        help="Comma-separated control key(s) to run, e.g. 'pronominalized' "
                             "(default: all string fields)")
    parser.add_argument("--attn_impl", type=str, default=None,
                        help="Attention implementation override, e.g. 'eager', 'sdpa', 'flash_attn'")
    parser.add_argument("--fill_missing", action="store_true",
                        help="Load existing output and only run keys absent from generated_answers; "
                             "rewrites the file with merged results")
    
    args = parser.parse_args() 
    main(args) 