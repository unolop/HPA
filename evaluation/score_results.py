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
"""

import os
import json
import argparse
import pandas as pd
from collections import defaultdict
from typing import Dict
from glob import glob
from utils import *  


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


def get_conditions(path, datasets=DATASETS, conditions=CONDITIONS, modelnames=MODELNAMES):
    """Extract model, dataset, condition from filename."""
    filename = os.path.basename(path).replace(".jsonl", "")

    tokens = filename.split("_")
    short_models = {m.split("/")[-1]: m for m in modelnames}

    model_full = None
    for short, full in short_models.items():
        if short in path:
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

# Global mapper instance (lazy loaded)
_vqa_mapper = None

def get_vqa_mapper() -> VQAAnswerMapper:
    """Get or create global VQA mapper."""
    global _vqa_mapper
    if _vqa_mapper is None:
        _vqa_mapper = VQAAnswerMapper()
    return _vqa_mapper

def get_spubench(): 
    '''
    data example :  
        {'core_attributes': ['compact design', 'branding', 'cosmetic container'],
        'spurious_attributes': ['round shape',
        'marble-like background',
        'shiny surface'],
        'spurious_correlation_type': ['Shape', 'Background'],
        'question': 'Question: Which feature best indicates the identity of the round object on the marble-like surface?\nA. The branding on it\nB. The marble-like background\nC. The round shape\nD. Its shiny surface\nProvide only the letter corresponding to the correct choice (A, B, C, or D).\nAnswer:',
        'choices': ['A. The branding on it',
        'B. The marble-like background',
        'C. The round shape',
        'D. Its shiny surface'],
        'answer': 'A',
        'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1444x2564>}
    '''
    # from inference import load_dataset 
    # spubench = load_dataset("spubench") 

    with open('/home/work/yuna/HPA/data/annotation.json', 'r', encoding='utf-8') as f: 
        annot = json.load(f) 
    return annot 

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
    print(f"📊 Scoring: {input_path}") 

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
        breakpoint() 
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