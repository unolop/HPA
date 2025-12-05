#!/usr/bin/env python3
"""
1. preprocess_answers.py - Answer Preprocessing with Translation Caching

Handles:
- Korean → English translation with persistent caching
- Answer normalization  
- Optional semantic clustering for free-text answers

Usage:
    python preprocess_answers.py \
        --input_csvs ./human_data/mmstar/*.csv \
        --questions_csv ./data/mmstar_questions.csv \
        --output_dir ./processed_data/mmstar \
        --translate \
        --cache_file ./translation_cache.json
"""

import os
import re
import csv
import json
import argparse
import atexit
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from tqdm import tqdm

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


def translate_single(
    client,
    question: str,
    answer: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> str:
    """
    Translate a single Korean answer to English.
    
    Uses question context for better translation.
    Retries on failure.
    """
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

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low for consistency
                max_tokens=100,
            )
            result = response.choices[0].message.content.strip()
            
            # Clean up common issues
            result = result.strip('"\'')
            if result.lower().startswith('english:'):
                result = result[8:].strip()
            
            return result
            
        except Exception as e:
            print(f"  ⚠ Translation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)  # Wait before retry
    
    # Return original if all retries fail
    print(f"  ❌ Failed to translate: {answer}")
    return answer


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
            translated = translate_single(client, question, answer, model)
            
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
    
    return answer.strip()


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


# =============================================================================
# Optional: Semantic Clustering
# =============================================================================

def cluster_answers(
    responses: List[Dict],
    distance_threshold: float = 0.5,
) -> List[Dict]:
    """
    Cluster semantically similar answers per question.
    
    Requires: pip install sentence-transformers scikit-learn
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        print("⚠ Clustering requires: pip install sentence-transformers scikit-learn")
        print("  Skipping clustering, returning original responses.")
        return responses
    
    print("\n🔗 Clustering similar answers...")
    
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Group by question
    by_question = defaultdict(list)
    for i, r in enumerate(responses):
        by_question[r['qid']].append((i, r))
    
    # Cluster each question's answers
    clustered_responses = responses.copy()
    total_clusters = 0
    
    for qid, items in tqdm(by_question.items(), desc="Clustering"):
        indices = [i for i, r in items]
        answers = [r['answer_normalized'] for i, r in items]
        
        if len(set(answers)) <= 1:
            # All same answer, single cluster
            for idx in indices:
                clustered_responses[idx]['cluster_id'] = 0
                clustered_responses[idx]['cluster_answer'] = answers[0]
            total_clusters += 1
            continue
        
        # Get unique answers and their embeddings
        unique_answers = list(set(answers))
        embeddings = model.encode(unique_answers)
        
        if len(unique_answers) == 1:
            answer_to_cluster = {unique_answers[0]: 0}
        else:
            # Cluster
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='cosine',
                linkage='average',
            )
            labels = clustering.fit_predict(embeddings)
            answer_to_cluster = {a: int(l) for a, l in zip(unique_answers, labels)}
        
        # Find representative for each cluster (most common answer)
        cluster_answers = defaultdict(list)
        for i, r in items:
            cid = answer_to_cluster[r['answer_normalized']]
            cluster_answers[cid].append(r['answer_normalized'])
        
        cluster_representatives = {}
        for cid, ans_list in cluster_answers.items():
            from collections import Counter
            cluster_representatives[cid] = Counter(ans_list).most_common(1)[0][0]
        
        # Assign clusters to responses
        for idx, (i, r) in zip(indices, items):
            cid = answer_to_cluster[r['answer_normalized']]
            clustered_responses[idx]['cluster_id'] = cid
            clustered_responses[idx]['cluster_answer'] = cluster_representatives[cid]
        
        total_clusters += len(set(answer_to_cluster.values()))
    
    print(f"✓ Created {total_clusters} clusters from {len(responses)} responses")
    
    return clustered_responses


# =============================================================================
# Data Loading
# =============================================================================

def load_human_responses(input_files: List[str]) -> List[Dict]:
    """Load human responses from participant CSV, JSON, or JSONL files."""
    responses = []
    
    for input_file in input_files:
        participant_id = Path(input_file).stem
        
        try:
            if input_file.endswith('.jsonl'):
                # Handle JSONL files (pilot data)
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        # Clean qid (remove .0 from float conversion)
                        qid = str(row.get('qid', row.get('question_id', '')))
                        if qid.endswith('.0'):
                            qid = qid[:-2]
                        responses.append({
                            'qid': qid,
                            'answer': str(row.get('answer', row.get('processed_output', ''))),
                            'confidence': int(row.get('confidence', 1)),  # Default 1 for pilot
                            'time_spent': float(row.get('time_spent_seconds', 0)),
                            'participant_id': row.get('participant_id', participant_id),
                            'question': row.get('question', ''),
                            'question_type': row.get('question_type', ''),
                        })
            
            elif input_file.endswith('.json'):
                # Handle JSON files (list of responses)
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else data.get('responses', [])
                    for row in items:
                        qid = str(row.get('qid', row.get('question_id', '')))
                        if qid.endswith('.0'):
                            qid = qid[:-2]
                        responses.append({
                            'qid': qid,
                            'answer': str(row.get('answer', '')),
                            'confidence': int(row.get('confidence', 1)),
                            'time_spent': float(row.get('time_spent_seconds', 0)),
                            'participant_id': row.get('participant_id', participant_id),
                            'question': row.get('question', ''),
                            'question_type': row.get('question_type', ''),
                        })
            
            else:
                # Handle CSV files (original format)
                with open(input_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        responses.append({
                            'qid': str(row.get('qid', row.get('question_id', ''))),
                            'answer': str(row.get('answer', '')),
                            'confidence': int(row.get('confidence', 3)),
                            'time_spent': float(row.get('time_spent_seconds', 0)),
                            'participant_id': participant_id,
                        })

        except Exception as e:
            print(f"⚠ Error loading {input_file}: {e}")
    
    print(f"✓ Loaded {len(responses)} responses from {len(input_files)} files")
    return responses


def load_questions(questions_path: str) -> Dict[str, Dict]:
    """Load questions from CSV or JSON file."""
    questions = {}
    
    if questions_path.endswith('.csv'):
        with open(questions_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row.get('qid', row.get('question_id', row.get('index', ''))))
                questions[qid] = {
                    'question': row.get('question_en', row.get('question', '')),
                    'category': row.get('category', row.get('question_type', '')),
                    'l2_category': row.get('l2_category', ''),
                    'answer': row.get('answer', row.get('multiple_choice_answer', '')),
                    'answer_type': row.get('answer_type', ''),
                    'options': row.get('options', ''),
                }
    else:
        with open(questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle various JSON formats
        if isinstance(data, list):
            items = data
        elif 'questions' in data:
            items = data['questions']
        else:
            items = list(data.values()) if isinstance(data, dict) else []
        
        for item in items:
            if isinstance(item, dict):
                qid = str(item.get('qid', item.get('question_id', item.get('id', ''))))
                questions[qid] = {
                    'question': item.get('question', item.get('question_en', '')),
                    'category': item.get('category', ''),
                    'answer': item.get('answer', ''),
                }
    
    print(f"✓ Loaded {len(questions)} questions from {questions_path}")
    return questions


# =============================================================================
# Main Pipeline
# =============================================================================

def preprocess_pipeline(
    input_csvs: List[str],
    questions_path: str,
    output_dir: str,
    cache_file: str = None,
    translate: bool = True,
    cluster: bool = False,
    cluster_threshold: float = 0.5,
    openai_model: str = "gpt-4o-mini",
) -> Dict:
    """
    Main preprocessing pipeline.
    
    Steps:
    1. Load human responses
    2. Load questions
    3. Translate Korean answers (with caching)
    4. Normalize answers
    5. Optional: Cluster similar answers
    6. Save processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📋 ANSWER PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Set default cache file
    if cache_file is None:
        cache_file = os.path.join(output_dir, 'translation_cache.json')
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    responses = load_human_responses(input_csvs)
    questions = load_questions(questions_path)
    
    # Add question info to responses
    for r in responses:
        q = questions.get(r['qid'], {})
        r['question'] = q.get('question', '')
        r['category'] = q.get('category', '')
        r['answer_type'] = q.get('answer_type', '')
    
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
        r['answer_normalized'] = normalize_answer(r['answer'])
        r['answer_normalized'] = normalize_number_words(r['answer_normalized'])
    
    # Step 4: Optional clustering
    if cluster:
        print("\n[4/5] Clustering similar answers...")
        responses = cluster_answers(responses, cluster_threshold)
    else:
        print("\n[4/5] Skipping clustering (disabled)")
        for r in responses:
            r['cluster_id'] = None
            r['cluster_answer'] = r['answer_normalized']
    
    # Step 5: Save outputs
    print("\n[5/5] Saving outputs...")
    
    # Save individual responses (for calibration analysis)
    individual_path = os.path.join(output_dir, 'individual_responses.json')
    with open(individual_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    print(f"   ✓ {individual_path}")
    
    # Save statistics
    stats = {
        'total_responses': len(responses),
        'total_questions': len(set(r['qid'] for r in responses)),
        'total_participants': len(set(r['participant_id'] for r in responses)),
        'korean_translated': sum(1 for r in responses if 'answer_original' in r),
        'unique_answers': len(set(r['answer_normalized'] for r in responses)),
    }
    
    if cluster:
        stats['total_clusters'] = len(set((r['qid'], r['cluster_id']) for r in responses if r['cluster_id'] is not None))
    
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
        'responses': responses,
        'questions': questions,
        'stats': stats,
        'output_dir': output_dir,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess VQA answers with translation and normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing with translation
  python preprocess_answers.py \\
      --input_csvs ./human_data/*.csv \\
      --questions_csv ./data/questions.csv \\
      --output_dir ./processed \\
      --translate

  # With clustering for free-text answers
  python preprocess_answers.py \\
      --input_csvs ./human_data/*.csv \\
      --questions_csv ./data/questions.csv \\
      --output_dir ./processed \\
      --translate --cluster
        """
    )
    
    parser.add_argument("--input_csvs", type=str, nargs='+', required=True,
                        help="Human response CSV files (glob pattern supported)")
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
    
    parser.add_argument("--cluster", action="store_true",
                        help="Cluster semantically similar answers")
    parser.add_argument("--cluster_threshold", type=float, default=0.5,
                        help="Clustering distance threshold (0.3=tight, 0.7=loose)")
    
    args = parser.parse_args()
    
    # Handle glob patterns
    import glob
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