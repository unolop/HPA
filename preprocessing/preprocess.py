#!/usr/bin/env python3

import os
import re
import csv
import json
import argparse
import atexit
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import sys 
import glob
from pathlib import Path
# Add parent directory to path for dataset imports
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))
from dataset.vqav2 import VQADataset

def create_train_val_split(
    jsonl_path: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Split JSONL file into train and validation sets.
    
    Splits by question ID to avoid data leakage.
    """
    np.random.seed(seed)
    
    # Load examples
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Group by qid
    by_qid = defaultdict(list)
    for ex in examples:
        by_qid[ex['qid']].append(ex)
    
    # Split qids
    qids = list(by_qid.keys())
    np.random.shuffle(qids)
    
    split_idx = int(len(qids) * train_ratio)
    train_qids = set(qids[:split_idx])
    val_qids = set(qids[split_idx:])
    
    # Assign examples
    train_examples = [ex for ex in examples if ex['qid'] in train_qids]
    val_examples = [ex for ex in examples if ex['qid'] in val_qids]
    
    # Save
    base_path = Path(jsonl_path)
    train_path = base_path.parent / f"{base_path.stem}_train.jsonl"
    val_path = base_path.parent / f"{base_path.stem}_val.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✓ Train: {train_path} ({len(train_examples)} examples, {len(train_qids)} questions)")
    print(f"✓ Val: {val_path} ({len(val_examples)} examples, {len(val_qids)} questions)")
    
    return str(train_path), str(val_path)


def read_questions(dataset='s1', answer_type='text'): 
    questions_path = f"/home/work/yuna/HPA/experiments/questions/{dataset}.csv" 
    questions = []
    if answer_type == 'text': 
        vqa_val = VQADataset()
        vqa_val = vqa_val.get_by_qid() 

    if questions_path.endswith('.csv'):
        with open(questions_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row.get('qid', row.get('question_id', '')))
                atype = row.get('answer_type', '') 
                if atype != answer_type: 
                    continue 

                if answer_type == 'choice': 
                    answer = row.get('answer', row.get('multiple_choice_answer', ''))
                    # annot = annotations[qid] 
                    # annot['answer'] = answer 
                    ### TODO Load the images for mmstar and mmspu and save into PIL image below 
                    # questions.append(annot) 

                else: # vqa questions 
                    questions.append(vqa_val[qid]) 
    else:
        with open(questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data if isinstance(data, list) else data.get('questions', [])
        for item in items:
            qid = str(item.get('qid', item.get('question_id', '')))
            questions.append({
                'qid': qid,
                'question': item.get('question', item.get('question_en', '')),
                'answer': item.get('answer', ''),
                'image_id': item.get('image_id', ''),
                'category': item.get('category', ''),
            })

    return questions 

# =============================================================================
# Translation Cache Manager
# =============================================================================

def clean_korean_from_options(options_str: str) -> str:
    """
    Remove Korean text from MC options string.
    
    Input: "A: In her hand 그녀의 손에, B: On her shoulder 그녀의 어깨에, ..."
    Output: "A: In her hand, B: On her shoulder, ..."
    
    Also handles list format: ["In her hand 그녀의 손에", "On her shoulder 그녀의 어깨에"]
    """
    import re
    import ast
    
    # Korean character pattern
    korean_pattern = r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]+'
    
    # Try parsing as list first
    if options_str.startswith('['):
        try:
            options_list = ast.literal_eval(options_str)
            cleaned = []
            for opt in options_list:
                # Remove Korean and extra whitespace
                cleaned_opt = re.sub(korean_pattern, '', str(opt)).strip()
                # Clean up multiple spaces
                cleaned_opt = ' '.join(cleaned_opt.split())
                cleaned.append(cleaned_opt)
            return str(cleaned)
        except:
            pass
    
    # Handle string format "A: text 한글, B: text 한글"
    cleaned = re.sub(korean_pattern, '', options_str)
    # Clean up multiple spaces and commas
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def parse_options_to_list(options_str: str) -> List[str]:
    """
    Parse options string to list of option texts (without A:, B:, etc.)
    
    Input: "A: Red, B: Blue, C: Green, D: Yellow"
    Output: ["Red", "Blue", "Green", "Yellow"]
    
    Also handles: ["Red", "Blue", "Green", "Yellow"]
    """
    import ast
    import re
    
    if not options_str:
        return []
    
    # Try parsing as list
    if options_str.startswith('['):
        try:
            return ast.literal_eval(options_str)
        except:
            pass
    
    # Parse "A: text, B: text" format
    options = []
    # Split by pattern like "A:", "B:", etc.
    parts = re.split(r'[A-D]:\s*', options_str)
    for part in parts:
        part = part.strip().rstrip(',').strip()
        if part:
            options.append(part)
    
    return options


class TranslationCache:
    """
    Persistent translation cache that saves automatically.
    
    - Loads from file on init
    - Saves on every N translations
    - Saves on program exit
    - Thread-safe for single process
    """
    
    def __init__(self, cache_file: str, save_every: int = 10):
        self.cache_file = cache_file
        self.save_every = save_every
        self.cache: Dict[str, str] = {}
        self.unsaved_count = 0
        
        # Load existing cache
        self._load()
        
        # Register save on exit
        atexit.register(self.save)
    
    def _load(self):
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"✓ Loaded {len(self.cache)} cached translations from {self.cache_file}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠ Could not load cache: {e}. Starting fresh.")
                self.cache = {}
        else:
            print(f"ℹ No cache file found. Will create: {self.cache_file}")
            self.cache = {}
    
    def save(self):
        """Save cache to file."""
        if self.unsaved_count > 0:
            # Ensure directory exists
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Saved {len(self.cache)} translations to {self.cache_file}")
            self.unsaved_count = 0
    
    def get(self, key: str) -> Optional[str]:
        """Get translation from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        """Set translation in cache and auto-save periodically."""
        self.cache[key] = value
        self.unsaved_count += 1
        
        # Auto-save every N translations
        if self.unsaved_count >= self.save_every:
            self.save()
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache
    
    def __len__(self):
        return len(self.cache)
    
    def __contains__(self, key: str):
        return key in self.cache


# =============================================================================
# Korean Detection & Translation
# =============================================================================

def contains_korean(text: str) -> bool:
    """Check if text contains Korean characters."""
    if not text or not isinstance(text, str):
        return False
    pattern = re.compile(r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]')
    return bool(pattern.search(text))


def setup_openai_client(api_key: str = None):
    """Setup OpenAI client for translation."""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ OpenAI package not installed. Run: pip install openai")
        return None

    # Try to get API key from various sources
    if api_key is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        api_key = os.getenv('OPENAI_API_KEY') or os.getenv('API_KEY')

    if not api_key:
        print("❌ No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return None

    return OpenAI(api_key=api_key)


def ask_gpt(client, prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Send a prompt to GPT and get the response.

    Args:
        client: OpenAI client instance
        prompt: The prompt to send
        model: Model to use (default: gpt-4o-mini)

    Returns:
        The text response from the model
    """
    if client is None:
        return ""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  Error calling GPT: {e}")
        return ""

def translate_prompt(question, answer): 
    prompt = f"""You are a precise translation assistant for VQA data.

    Task: Translate the Korean answer to English. Return ONLY the translated answer, nothing else.

    Rules:
    - Keep answer concise (similar length to original)
    - Use natural English phrasing
    - If it's a number, color, or simple noun, translate directly
    - Do not add explanations

    QUESTION: {question}
    KOREAN ANSWER: {answer}

    English:"""
    return prompt 

def translate_answers(
    responses: List[Dict],
    questions: Dict[str, Dict],
    cache: TranslationCache,
    client,
    model: str = "gpt-4o-mini",
) -> List[Dict]:
    """
    Translate all Korean answers in responses.
    
    Uses cache to avoid re-translating.
    Saves cache periodically and on completion.
    """
    # Count Korean answers needing translation
    to_translate = []
    for r in responses:
        answer = str(r.get('answer', ''))
        if contains_korean(answer) and not cache.has(answer):
            to_translate.append(r)
    
    print(f"\n🌐 Translation Status:")
    print(f"   Total responses: {len(responses)}")
    print(f"   Korean answers: {sum(1 for r in responses if contains_korean(str(r.get('answer', ''))))}")
    print(f"   Already cached: {sum(1 for r in responses if contains_korean(str(r.get('answer', ''))) and cache.has(str(r.get('answer', ''))))}")
    print(f"   Need translation: {len(to_translate)}")
    
    # Translate new answers
    if to_translate and client:
        print(f"\n📝 Translating {len(to_translate)} new answers...")
        
        for r in tqdm(to_translate, desc="Translating"):
            answer = str(r['answer'])
            qid = r.get('qid', '')
            question = questions.get(qid, {}).get('question', '')
            
            # Translate
            prompt = translate_prompt(question, answer)
            translated = ask_gpt(client, prompt, model)

            # Clean up common issues
            translated = translated.strip('"\'')
            if translated.lower().startswith('english:'):
                translated = translated[8:].strip()
            
            # Save to cache immediately
            cache.set(answer, translated)
        
        # Final save
        cache.save()
        print(f"✓ Translation complete. Cache now has {len(cache)} entries.")
    
    # Apply translations to all responses
    translated_responses = []
    for r in responses:
        r_copy = r.copy()
        answer = str(r_copy.get('answer', ''))
        
        if contains_korean(answer):
            r_copy['answer_original'] = answer
            r_copy['answer'] = cache.get(answer) or answer
        
        translated_responses.append(r_copy)
    
    return translated_responses


# =============================================================================
# Answer Normalization
# =============================================================================

def normalize_answer(answer: str) -> str:
    """
    Normalize answer for consistent comparison.
    
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Normalize whitespace
    """
    if not answer:
        return ""
    
    answer = str(answer).lower().strip()
    
    # Remove articles at start
    for article in ['a ', 'an ', 'the ']:
        if answer.startswith(article):
            answer = answer[len(article):]
    
    # Remove punctuation
    import string
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace
    answer = ' '.join(answer.split())
    
    return answer.lower().strip()


def normalize_number_words(answer: str) -> str:
    """Convert number words to digits for consistency."""
    number_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    }
    
    answer_lower = answer.lower().strip()
    return number_map.get(answer_lower, answer)


def load_human_responses(input_files: List[str], session_length:int) -> List[Dict]:
    """Load human responses from participant CSV, JSON, or JSONL files."""
    responses = []
    num_files= len(input_files)
    
    for input_file in input_files:
        participant_id = Path(input_file).parent.stem
        
        try:
            # Handle CSV files (original format)
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = list(csv.DictReader(f))
                # print(f"CSV file {input_file} has {len(reader)} rows")
                if len(reader) < session_length: 
                    print(f"{input_file} is incomplete, skip")
                    num_files-= 1
                    continue 
                for row in reader:
                    row['participant_id'] = participant_id
                    responses.append(row) 

        except Exception as e:
            print(f"⚠ Error loading {input_file}: {e}")
    
    print(f"✓ Loaded {len(responses)} responses from {num_files} files")
    return num_files, responses


def load_questions(questions_path: str) -> Dict[str, Dict]:
    """Load questions from CSV or JSON file."""
    questions = {}
    
    if questions_path.endswith('.csv'):
        with open(questions_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row.get('qid', row.get('question_id', row.get('index', ''))))
                questions[qid] = row
    print(f"✓ Loaded {len(questions)} questions from {questions_path}")
    return questions


# =============================================================================
# Main Pipeline
# =============================================================================

def preprocess_pipeline(
    input_csvs: List[str],
    questions_path: str,
    output_dir: str,
    cache_file: str = "/home/work/yuna/HPA/preprocessing/translation_cache.json",
    translate: bool = True,
    openai_model: str = "gpt-4o-mini",
) -> Dict:
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📋 ANSWER PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Set default cache file
    if cache_file is None:
        cache_file = os.path.join(output_dir, 'translation_cache.json')
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    questions = load_questions(questions_path)        
    n, responses = load_human_responses(input_csvs, session_length=len(questions)) 
    
    # Add question info to responses
    for r in responses:  
        qid = r['qid'] 
        question_dict = questions[qid]
        answer_type = question_dict['answer_type'] 
        question = question_dict['question_en'].replace('\n', '').strip()
        r['question'] = question  
        r['answer_type'] = answer_type 
        
    # Step 2: Translation
    if translate:
        print("\n[2/5] Translating Korean answers...")
        cache = TranslationCache(cache_file, save_every=10)
        client = setup_openai_client()
        
        if client:
            responses = translate_answers(responses, questions, cache, client, openai_model)
        else:
            print("⚠ Skipping translation (no API client)")
    else:
        print("\n[2/5] Skipping translation (disabled)")
    
    # Step 3: Normalize
    print("\n[3/5] Normalizing answers...")
    for r in responses:
        r['answer_raw'] = r['answer']
        if r['answer_type'] == 'choice': 
            r['answer_normalized'] = ['A', 'B', 'C', 'D'][int(r['answer_raw'])-1]
        else:
            r['answer_normalized'] = normalize_answer(r['answer_raw'])
            r['answer_normalized'] = normalize_number_words(r['answer_normalized'])
    
    # Create a dict to group responses by answer_type
    responses_by_type = defaultdict(list)
    for r in responses:
        responses_by_type[r['answer_type']].append(r)

    for answer_type, resp_list in responses_by_type.items():
        path = os.path.join(output_dir, f"cleaned_n{n}_{answer_type}.json")
        with open(path, "w", encoding='utf-8') as f:
            json.dump(resp_list, f, ensure_ascii=False, indent=2)
        print(f"   ✓ {path}")

    # Step 5: Save outputs
    print("\n[5/5] Saving outputs...")
    
    # Save statistics
    stats = {
        'total_responses': len(responses),
        'total_questions': len(set(r['qid'] for r in responses)),
        'total_participants': len(set(r['participant_id'] for r in responses)),
        'korean_translated': sum(1 for r in responses if 'answer_original' in r),
        'unique_answers': len(set(r['answer'] for r in responses)),
    }
    
    stats_path = os.path.join(output_dir, 'preprocessing_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✓ {stats_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PREPROCESSING SUMMARY")
    print("=" * 60)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print("=" * 60)
    
    return {
        'responses': responses_by_type,
        'questions': questions,
        'stats': stats,
        'output_dir': output_dir,
    }

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--questions_csv", type=str, required=True,
                        help="Questions file (CSV or JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    
    parser.add_argument("--translate", action="store_true",
                        help="Translate Korean answers to English")
    parser.add_argument("--cache_file", type=str, default=None,
                        help="Translation cache file (default: output_dir/translation_cache.json)")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini",
                        help="OpenAI model for translation")
    
    args = parser.parse_args()
    
    # Handle glob patterns
    input_files = []
    for pattern in args.input_csvs:
        input_files.extend(glob.glob(pattern))
    
    if not input_files:
        print(f"❌ No files found matching: {args.input_csvs}")
        return
    
    preprocess_pipeline(
        input_csvs=input_files,
        questions_path=args.questions_csv,
        output_dir=args.output_dir,
        cache_file=args.cache_file,
        translate=args.translate,
        cluster=args.cluster,
        cluster_threshold=args.cluster_threshold,
        openai_model=args.openai_model,
    )

if __name__ == "__main__":
    main()