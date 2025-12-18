import os
import re
from tqdm import tqdm
import json
import numpy as np
from analysis.utils import skip_processed_idx
import sys
import base64
from pathlib import Path
import time

sys.path.append('/home/work/yuna/HPA')

seed = 42
np.random.seed(seed)


def load_dataset(data_name: str, prompt: str = ''):
    """Load dataset (same as inference.py)"""
    print("Loading dataset...")

    if data_name == "mmstar":
        from datasets import load_dataset
        dataset = load_dataset("Lin-Chen/MMStar", split="val")

    elif data_name == "spubench":
        from datasets import load_dataset
        print("  Loading MM-SpuBench dataset from HuggingFace (will cache locally)...")
        dataset = load_dataset("mmbench/MM-SpuBench", split="train", trust_remote_code=True)
        print(f"  ✓ Prepared {len(dataset)} examples")

    elif data_name == "vqa_1k":
        from dataset.vqav2 import VQADataset_json
        dataset = VQADataset_json(prompt=prompt)

    elif data_name == "vqa_5k":
        from dataset.vqav2 import VQADataset
        from torch.utils.data import Subset

        with open('/home/work/yuna/HPA/dataset/s1_qids.json', 'r') as file:
            qids = json.load(file)

        dataset = VQADataset(prompt=prompt, filter_qids=qids)
        dataset = Subset(dataset, np.random.choice(len(dataset), size=5000, replace=False))

    return dataset


def encode_image_to_base64(image_path):
    """Encode image to base64 for API requests"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_openai_api(messages, image_path=None, model="gpt-4o", max_tokens=4096, temperature=0):
    """Call OpenAI API (GPT models)"""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Prepare messages with image if provided
    if image_path is not None and model in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
        # For vision models
        base64_image = encode_image_to_base64(image_path)

        # Convert messages to include image
        formatted_messages = []
        for msg in messages:
            if msg['role'] == 'user' and '<image>' in msg['content']:
                # Replace <image> with actual image
                text = msg['content'].replace('<image>', '')
                formatted_messages.append({
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            'type': 'text',
                            'text': text
                        }
                    ]
                })
            else:
                formatted_messages.append(msg)
    else:
        formatted_messages = messages

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise


def call_gemini_api(messages, image_path=None, model="gemini-1.5-pro", max_tokens=4096, temperature=0):
    """Call Google Gemini API"""
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Initialize model
    model_instance = genai.GenerativeModel(model)

    # Prepare content
    if image_path is not None:
        # For vision models
        from PIL import Image
        image = Image.open(image_path)

        # Extract text from messages
        prompt_text = ""
        for msg in messages:
            if msg['role'] == 'user':
                prompt_text += msg['content'].replace('<image>', '')
            elif msg['role'] == 'system':
                prompt_text = msg['content'] + "\n\n" + prompt_text

        content = [prompt_text, image]
    else:
        # Text only
        prompt_text = ""
        for msg in messages:
            if msg['role'] == 'user':
                prompt_text += msg['content']
            elif msg['role'] == 'system':
                prompt_text = msg['content'] + "\n\n" + prompt_text

        content = prompt_text

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model_instance.generate_content(
                content,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
            return response.text
        except Exception as e:
            print(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise


def main(args):
    # Model configuration
    model_name = args.model
    provider = args.provider  # 'openai' or 'gemini'

    # Save path configuration
    save_name = model_name.replace('/', '_').replace('.', '_')
    savedir = args.savedir
    savedir += '/api'  # API models go in separate directory

    output_jsonl_path = f"{savedir}/{save_name}/{args.dataset}{args.condition}.jsonl"

    # Prepare prompt
    prompt = ''
    if 'inst' in args.condition:
        prompt = f"\nNote: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n"

    # Load dataset
    dataset = load_dataset(args.dataset, prompt)

    if args.dataset == 'spubench':
        with open('/home/work/yuna/HPA/dataset/annotation.json', 'r', encoding='utf-8') as f:
            annot = json.load(f)

    # Check for resume
    try:
        processed_ids, current_item_id = skip_processed_idx(
            existing_keys=dataset[0].keys(),
            output_jsonl_path=output_jsonl_path
        )
    except Exception as e:
        print(e)
        processed_ids = set()

    if not args.resume:
        processed_ids = set()
        write_mode = 'w'
    else:
        write_mode = 'a'

    # Create output directory
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, write_mode, encoding='utf-8') as f:
        for i, data in enumerate(tqdm(dataset)):
            data['pid'] = i

            # Handle blind condition
            if 'blind' in args.condition:
                data['image'] = "/home/work/yuna/HPA/data/blank_224.png"

            # Skip if already processed
            if processed_ids is not None:
                if data[current_item_id] in processed_ids:
                    continue

            # Prepare prompt based on dataset
            if 'mmstar' in args.dataset:
                prompt_text = data['question']
                if 'inst' in args.condition:
                    prompt_text = f"{prompt_text}\nNote: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario."
                prompt_text += "\nProvide only the letter corresponding to the correct choice (A, B, C, or D).\nAnswer:"

            elif 'spubench' in args.dataset:
                ann = annot[i]
                image = data['image']
                question = ann['question']
                choices = ann['choices']
                choices_text = "\n".join(choices)
                prompt_text = (
                    f"Question: {question}\n"
                    f"{choices_text}\n"
                )
                if 'inst' in args.condition:
                    prompt_text = f"{prompt_text}\nNote: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario."

                prompt_text = f"{prompt_text}\nProvide only the letter corresponding to the correct choice (A, B, C, or D).\nAnswer:"
                ann['question'] = prompt_text

            else:  # VQA
                prompt_text = data['question']

            # Prepare messages
            messages = []
            if 'sys_inst' in args.condition:
                messages.append({'role': 'system', 'content': system_message})

            # Determine if we need image
            if args.model_type == 'vlm':
                messages.append({'role': 'user', 'content': f'<image>{prompt_text}'})
                image_path = data.get('image', None)
            else:
                messages.append({'role': 'user', 'content': prompt_text})
                image_path = None

            # Call appropriate API
            try:
                if provider == 'openai':
                    output_text = call_openai_api(
                        messages,
                        image_path=image_path,
                        model=model_name,
                        max_tokens=args.max_token_length,
                        temperature=0
                    )
                elif provider == 'gemini':
                    output_text = call_gemini_api(
                        messages,
                        image_path=image_path,
                        model=model_name,
                        max_tokens=args.max_token_length,
                        temperature=0
                    )
                else:
                    raise ValueError(f"Unknown provider: {provider}")

                # Clean output (same as inference.py)
                matches = re.findall(r"Answer\s*:?\s*(.+)", output_text)
                if matches:
                    output_text = matches[-1].strip().replace('*', '')
                else:
                    output_text = output_text.strip().replace('*', '')

                # Save results
                data['output'] = output_text
                data['question'] = prompt_text
                print('Q:', prompt_text[:100], '... Output:', data['output'], output_jsonl_path)

                # Remove image data before saving
                data.pop('image', None)
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                f.flush()

            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue

    print(f"All tasks completed. Results saved to '{output_jsonl_path}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API Model Inference (GPT/Gemini)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model name (e.g., gpt-4o, gpt-4o-mini, gemini-1.5-pro, gemini-1.5-flash)")
    parser.add_argument("--provider", type=str, choices=['openai', 'gemini'], default="openai",
                        help="API provider (openai or gemini)")
    parser.add_argument("--model_type", type=str, default="vlm",
                        help="Model type (vlm for vision, lm for text-only)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--max_token_length", type=int, default=4096,
                        help="Maximum token length")
    parser.add_argument("--dataset", type=str, default="mmstar",
                        help="Dataset name (mmstar, spubench, vqa_1k, vqa_5k)")
    parser.add_argument('--savedir', type=str, default="/home/work/yuna/HPA/evaluation/results",
                        help='Save directory for inference results')
    parser.add_argument("--condition", type=str, default='',
                        help="Condition suffix (_blind, _inst_blind, etc.)")

    args = parser.parse_args()
    main(args)
