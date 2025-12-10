#!/usr/bin/env python3
"""
score_results.py - Score existing inference JSONL outputs

Works with your existing output format:
{"index": 2, "question": "...", "answer": "D", "output": "A: Hanging Posters", ...}

Computes:
- MC accuracy (extracts A/B/C/D from output)
- VQA accuracy (for open-ended)
- Embedding similarity
- Per-category breakdown

Usage:
    # Score single file
    python score_results.py --input results/InternVL3_5-2B_mmstar_inst_blind.jsonl --output_dir scored/

    # Score all files in directory
    python score_results.py --input_dir /home/work/yuna/HPA/results/swift/ --output_dir /home/work/yuna/HPA/results/swift/scored/

    # With embedding similarity for VQA datasets
    python score_results.py --input_dir results/ --output_dir scored/ --with_similarity
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from glob import glob


# =============================================================================
# Constants
# =============================================================================

DATASETS = ["mmstar", "spubench", "vqa_1k", "vqa_5k"]
CONDITIONS = ["_inst_blind", "", "_sys_inst_blind", "_blind"]
MODELNAMES = [
    "OpenGVLab/InternVL3_5-8B",
    "OpenGVLab/InternVL3_5-4B",
    "OpenGVLab/InternVL3_5-2B",
    "OpenGVLab/InternVL3_5-1B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-1.5-7b-hf",
]

DATASET_TYPE = {
    'mmstar': 'multi-choice',
    'spubench': 'multi-choice',
    'vqa_1k': 'open-ended',
    'vqa_5k': 'open-ended',
}

VQA_ANNOTATIONS_PATH = "/home/work/yuna/VLMEval/data/v2_mscoco_val2014_annotations.json"


# =============================================================================
# VQA Answer Mapping
# =============================================================================

class VQAAnswerMapper:
    """
    Maps question_id to list of ground truth answers from VQA annotations.

    Usage:
        mapper = VQAAnswerMapper()
        answers = mapper.get_answers(123456)  # Returns list of 10 annotator answers
    """

    def __init__(self, annotations_path: str = VQA_ANNOTATIONS_PATH):
        self.annotations_path = annotations_path
        self._qid_to_answers = None  # Lazy load
        self._loading = False  # Prevent recursive loading

    def _load(self):
        """Load annotations and build lookup dict (lazy loading)."""
        if self._qid_to_answers is not None:
            return

        if self._loading:
            return

        self._loading = True

        # Check if file exists before trying to load
        if not os.path.exists(self.annotations_path):
            print(f"⚠️  VQA annotations not found: {self.annotations_path}")
            self._qid_to_answers = {}
            self._loading = False
            return

        print(f"📂 Loading VQA annotations (this may take a moment)...")

        try:
            with open(self.annotations_path, 'r') as f:
                data = json.load(f)

            annotations = data.get('annotations', data)

            # Build qid -> answers lookup
            self._qid_to_answers = {}
            for ann in annotations:
                qid = int(ann['question_id'])
                # Extract answer strings from annotator responses
                answers = [a['answer'] for a in ann['answers']]
                self._qid_to_answers[qid] = answers

            print(f"   ✓ Loaded {len(self._qid_to_answers)} question annotations")
        except Exception as e:
            print(f"⚠️  Failed to load VQA annotations: {e}")
            self._qid_to_answers = {}
        finally:
            self._loading = False
    
    def get_answers(self, question_id: int) -> List[str]:
        """
        Get list of ground truth answers for a question.
        
        Args:
            question_id: VQA question ID
            
        Returns:
            List of 10 annotator answers (strings)
        """
        self._load()
        qid = int(question_id)
        return self._qid_to_answers.get(qid, [])
    
    def get_majority_answer(self, question_id: int) -> str:
        """Get most common answer for a question."""
        answers = self.get_answers(question_id)
        if not answers:
            return ""
        from collections import Counter
        return Counter(answers).most_common(1)[0][0]
    
    def has_question(self, question_id: int) -> bool:
        """Check if question_id exists in annotations."""
        self._load()
        return int(question_id) in self._qid_to_answers


# Global mapper instance (lazy loaded)
_vqa_mapper = None

def get_vqa_mapper() -> VQAAnswerMapper:
    """Get or create global VQA mapper."""
    global _vqa_mapper
    if _vqa_mapper is None:
        _vqa_mapper = VQAAnswerMapper()
    return _vqa_mapper


# =============================================================================
# Parsing Functions (from your processor.py)
# =============================================================================

def get_conditions(path, datasets=DATASETS, conditions=CONDITIONS, modelnames=MODELNAMES):
    """Extract model, dataset, condition from filename."""
    filename = os.path.basename(path).replace(".jsonl", "")

    tokens = filename.split("_")
    short_models = {m.split("/")[-1]: m for m in modelnames}

    model_short = None
    model_full = None
    for short, full in short_models.items():
        if short in path:
            model_short = short
            model_full = full
            break
    if 'finetuned' in path : 
        trained = path.split('/')[-2] 
        model_full += f"/{trained}" 

    dataset = None
    for d in datasets:
        if d in tokens or d in filename:
            dataset = d
            break

    condition = ""
    for c in conditions:
        if c != "" and c in filename:
            condition = c
            break

    return model_full, dataset, condition


def extract_mc_choice(output: str) -> str:
    """Extract the predicted answer (A, B, C, D) from model output."""
    if not output:
        return ""

    # Remove <think>...</think> content if present
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL | re.IGNORECASE)
    output = output.strip()

    # Pattern 1: Look for explicit answer statements
    patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is[:\s]*([A-D])",
        r"(?:the\s+)?(?:correct\s+)?answer[:\s]*([A-D])",
        r"(?:option\s+)?([A-D])\s+is\s+(?:the\s+)?correct",
        r"(?:I\s+)?(?:would\s+)?choose\s+(?:option\s+)?([A-D])",
        r"(?:I\s+)?(?:would\s+)?select\s+(?:option\s+)?([A-D])",
        r"^([A-D])(?:[:\.\)]|\s|$)",  # Answer at the start
        r"\n([A-D])(?:[:\.\)]|\s|$)",  # Answer after newline
        r"(?:Therefore|Thus|So|Hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(?:option\s+)?([A-D])",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Pattern 2: Look for the last standalone letter A-D
    matches = re.findall(r'\b([A-D])\b', output)
    if matches:
        return matches[-1].upper()
    
    # Pattern 3: Check if output is just a single letter
    if len(output) == 1 and output.upper() in 'ABCD':
        return output.upper()
    
    # Pattern 4: Check format like "A: Hanging Posters"
    match = re.match(r'^([A-D]):', output)
    if match:
        return match.group(1).upper()
    
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (open-ended)."""
    if not answer:
        return ""
    answer = str(answer)

    # Remove <think>...</think> content if present
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL | re.IGNORECASE)

    answer = answer.lower().strip()
    # Remove articles
    for article in ['a ', 'an ', 'the ']:
        if answer.startswith(article):
            answer = answer[len(article):]
    # Remove punctuation
    import string
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(answer.split()).strip()


# =============================================================================
# Scoring Functions
# =============================================================================

def mc_accuracy(gt: str, pred: str) -> bool:
    """Multiple choice accuracy."""
    gt_letter = gt.strip().upper()[0] if gt else ""
    pred_letter = extract_mc_choice(pred)
    return gt_letter == pred_letter


def vqa_accuracy(gt_answers: List[str], pred: str) -> float:
    """VQA accuracy: min(1, #matches / 3)."""
    pred = normalize_answer(pred)
    matches = sum([pred == normalize_answer(ans) for ans in gt_answers])
    return min(1.0, matches / 3.0)


def exact_match(gt: str, pred: str) -> bool:
    """Exact match after normalization."""
    return normalize_answer(pred) == normalize_answer(gt)


def get_encoder():
    """Lazy load sentence transformer."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2").to('cuda')
    except:
        return None


def answer_similarity(gt: str, pred: str, encoder) -> float:
    """Compute embedding similarity."""
    if encoder is None:
        return 0.0
    try:
        pred = pred.strip().lower()
        gt = str(gt).strip().lower()
        emb = encoder.encode([pred, gt])
        similarities = encoder.similarity(emb, emb)
        return float(similarities[1, 0])
    except:
        return 0.0


# =============================================================================
# Main Scoring
# =============================================================================

def score_file(
    input_path: str,
    output_path: str = None,
    skip_existing: bool = True,
    with_similarity: bool = False,
) -> Dict:
    """
    Score a single JSONL file and save processed version.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to save scored output (optional)
        skip_existing: If True, skip if output file already exists
        with_similarity: If True, compute answer similarity for VQA datasets

    Returns dict with accuracy, per-category breakdown, etc.
    """
    # Check if output already exists
    if skip_existing and output_path and os.path.exists(output_path):
        # INSERT_YOUR_CODE
        # If output_path exists and the input and output have the same number of lines, skip processing
        try:
            with open(input_path, 'r', encoding='utf-8') as fin:
                input_lines = sum(1 for _ in fin if _.strip())
            with open(output_path, 'r', encoding='utf-8') as fout:
                output_lines = sum(1 for _ in fout if _.strip())
            if input_lines == output_lines:
                # print(f"\n{'='*60}")
                # print(f"⏭️  Skipping (already processed, line count matches): {os.path.basename(input_path)}")
                # print(f"   Output exists: {output_path}")
                # print(f"{'='*60}")
                return None
        except Exception as e:
            # If any problem opening/reading, proceed to full scoring
            pass 

    # Parse filename
    model_full, dataset, condition = get_conditions(input_path)
    dataset_type = DATASET_TYPE.get(dataset, 'multi-choice')

    # print(f"\n{'='*60}")
    print(f"📊 Scoring: {input_path}") # {os.path.basename()} 
    # print(f"   Model: {model_full}")
    # print(f"   Dataset: {dataset} ({dataset_type})")
    # print(f"   Condition: {condition or '(none)'}") 
    # print(f"{'='*60}")

    # Load data first (fast operation)
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if not data:
        print("   ⚠️ Empty file!")
        return {}

    # Only load VQA mapper if we actually need it (lazy initialization)
    vqa_mapper = None
    # We'll create it on-demand inside the loop only if dataset_type is open-ended

    # Load encoder if similarity computation is requested
    encoder = None
    if with_similarity and dataset_type == 'open-ended':
        print("   Loading sentence transformer for similarity computation...")
        encoder = get_encoder()
        if encoder:
            print("   ✓ Encoder loaded")
        else:
            print("   ⚠️ Failed to load encoder, skipping similarity computation")

    # Score each example
    correct = 0
    total = 0
    by_category = defaultdict(lambda: {'correct': 0, 'total': 0})

    for item in data:
        output = item.get('output', '')
        category = item.get('category', item.get('l2_category', item.get('question_type', 'Unknown')))

        # Get ground truth answers and score
        if dataset_type == 'multi-choice':
            gt = item.get('answer', '')
            extracted_choice = extract_mc_choice(output)
            is_correct = gt.strip().upper()[0] == extracted_choice if gt and extracted_choice else False

            # Save extracted choice for MC datasets
            item['extracted_choice'] = extracted_choice
        else:
            # Open-ended: get all annotator answers
            qid = item.get('question_id', item.get('qid', item.get('index', None)))

            # Lazy load VQA mapper only when we encounter the first VQA item
            if vqa_mapper is None and qid is not None:
                vqa_mapper = get_vqa_mapper()

            if vqa_mapper and qid is not None:
                all_answers = vqa_mapper.get_answers(qid)
            else:
                # Fallback to item's answers field
                all_answers = item.get('answers', [item.get('answer', '')])
                if isinstance(all_answers, str):
                    all_answers = [all_answers]

            if all_answers:
                is_correct = vqa_accuracy(all_answers, output)
                # is_correct = acc >= 0.5  # Binary for counting

                # Compute answer similarity if requested
                if encoder and all_answers:
                    # Use majority answer for similarity computation
                    majority_ans = max(set(all_answers), key=all_answers.count)
                    sim = answer_similarity(majority_ans, output, encoder)
                    item['answer_similarity'] = float(sim)
            else:
                gt = item.get('answer', '')
                is_correct = exact_match(gt, output)

        # Save correctness
        item['correct'] = is_correct

        # Update counts
        correct += int(is_correct)
        total += 1
        by_category[category]['correct'] += int(is_correct)
        by_category[category]['total'] += 1

    # Save processed file with scores
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"   ✓ Saved scored file: {output_path}")
    
    # Compute metrics
    accuracy = correct / total if total > 0 else 0

    results = {
        'file': os.path.basename(input_path),
        'model': model_full,
        'dataset': dataset,
        'condition': condition,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'by_category': {},
    }

    # Per-category
    for cat, cat_data in by_category.items():
        cat_acc = cat_data['correct'] / cat_data['total'] if cat_data['total'] > 0 else 0
        results['by_category'][cat] = {
            'accuracy': cat_acc,
            'correct': cat_data['correct'],
            'total': cat_data['total'],
        }

    # Print results
    print(f"\n📈 Results:")
    print(f"   Accuracy: {accuracy:.4f} ({correct}/{total})")

    print(f"\n   Per-category:")
    for cat, cat_data in sorted(results['by_category'].items(), key=lambda x: -x[1]['accuracy']):
        print(f"      {cat}: {cat_data['accuracy']:.3f} ({cat_data['correct']}/{cat_data['total']})")

    return results


def score_directory(
    input_dir: str,
    skip_existing: bool = True,
    with_similarity: bool = False,
) -> pd.DataFrame:

    output_dir = f"/home/work/yuna/HPA/evaluation/scored/{input_dir}" 
    input_dir=f"/home/work/yuna/HPA/evaluation/results/{input_dir}" 

    files = sorted(glob(f"{input_dir}/*.jsonl") + glob(f"{input_dir}/*/*/*.jsonl"))

    if not files:
        print(f"❌ No files matching in {input_dir}")
        breakpoint()
        return pd.DataFrame()

    print(f"Found {len(files)} files to score")

    # Score all files
    all_results = []
    skipped_count = 0
    for f in files:
        # Determine output path
        filename = f.replace(f"{input_dir}/", '')
        out_path = os.path.join(output_dir, filename) 
        result = score_file(f, out_path, skip_existing=skip_existing) 
        if result:
            all_results.append(result)
        elif skip_existing and out_path and os.path.exists(out_path):
            skipped_count += 1

    # Print processing summary
    print(f"\n{'='*60}")
    print(f"✓ Processed {len(all_results)} files")
    if skipped_count > 0:
        print(f"⏭️  Skipped {skipped_count} files (already processed)")
    print(f"{'='*60}")

    # Create summary DataFrame
    df = pd.DataFrame(all_results)
    
    # Pivot for easy comparison
    if len(all_results) > 0:
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        # Group by model and condition
        summary = df.groupby(['model', 'dataset', 'condition'])['accuracy'].first().unstack(fill_value=0)
        print(summary.to_string())
    
    # Save if output_dir provided
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'scored_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Detailed results: {results_path}")
    
    # Save summary CSV
    csv_path = os.path.join(output_dir, 'summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Summary CSV: {csv_path}")
    
    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score inference JSONL outputs and save processed files with extracted choices and scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score single file
  python score_results.py --input results/InternVL3_5-2B_mmstar_inst_blind.jsonl --output_dir scored/

  # Score all files in directory (skips already processed)
  python score_results.py --input_dir results/ --output_dir scored/

  # Force reprocessing of all files
  python score_results.py --input_dir results/ --output_dir scored/ --force

  # Multiple files with glob
  python score_results.py --input "results/*mmstar*.jsonl" --output_dir scored/

  # With embedding similarity for VQA datasets
  python score_results.py --input_dir results/ --output_dir scored/ --with_similarity
        """
    )

    parser.add_argument("--input", type=str, nargs='*', default=None,
                        help="Input JSONL file(s)")
    parser.add_argument("--input_dir", type=str, default="None",
                        help="Directory with JSONL files")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing even if output files exist")
    parser.add_argument("--with_similarity", action="store_true",
                        help="Compute answer similarity for VQA datasets (requires sentence-transformers)")
    args = parser.parse_args()
    skip_existing = not args.force

    if args.input_dir:
        # Score directory
        score_directory(
            args.input_dir,
            skip_existing=skip_existing 
        )


if __name__ == "__main__":
    main()