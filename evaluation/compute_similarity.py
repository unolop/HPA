#!/usr/bin/env python3
"""
Compute embedding similarity for VQA datasets.

Reads scored JSONL files and adds embedding_similarity field to each item.
Saves to similarity_scored/ directory.

Usage:
    # Single file
    python compute_similarity.py --input scored/vqa_1k.jsonl --output_dir similarity_scored/

    # All VQA files in directory
    python compute_similarity.py --input_dir scored/ --output_dir similarity_scored/

    # Specific files
    python compute_similarity.py --input scored/*vqa*.jsonl --output_dir similarity_scored/
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from typing import List, Dict


def get_encoder():
    """Load sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence transformer model...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        if os.system("nvidia-smi > /dev/null 2>&1") == 0:
            encoder = encoder.to('cuda')
            print("✓ Using CUDA")
        else:
            print("✓ Using CPU")
        return encoder
    except Exception as e:
        print(f"❌ Failed to load encoder: {e}")
        return None


def compute_similarity(gt_answers: List[str], pred: str, encoder) -> float:
    """
    Compute embedding similarity between ground truth answers and prediction.

    Args:
        gt_answers: List of ground truth answer strings
        pred: Model prediction string
        encoder: Sentence transformer model

    Returns:
        Average cosine similarity (0-1)
    """
    if not encoder or not gt_answers or not pred:
        return 0.0

    try:
        # Encode prediction
        pred = pred.strip().lower()
        pred_emb = encoder.encode([pred], convert_to_numpy=True)[0]

        # Encode all ground truth answers
        gt_embs = encoder.encode(gt_answers, convert_to_numpy=True)

        # Compute cosine similarity with each GT answer
        similarities = []
        for gt_emb in gt_embs:
            sim = np.dot(pred_emb, gt_emb) / (
                np.linalg.norm(pred_emb) * np.linalg.norm(gt_emb) + 1e-10
            )
            similarities.append(sim)

        return float(np.mean(similarities))
    except Exception as e:
        print(f"⚠️  Similarity computation failed: {e}")
        return 0.0


def is_vqa_file(filepath: str) -> bool:
    """Check if file is a VQA dataset."""
    filename = os.path.basename(filepath).lower()
    return 'vqa' in filename


def process_file(input_path: str, output_path: str, encoder) -> Dict:
    """
    Process a single file: add embedding similarity to each item.

    Returns summary stats.
    """
    # Check if VQA file
    if not is_vqa_file(input_path):
        print(f"⚠️  Skipping non-VQA file: {os.path.basename(input_path)}")
        return None

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    # Load data
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if not data:
        print("⚠️  Empty file!")
        return None

    print(f"Loaded {len(data)} items")

    # Compute similarity for each item
    similarities = []
    for i, item in enumerate(data):
        # Get ground truth answers
        all_answers = item.get('all_answers', item.get('answers', []))
        if not all_answers:
            gt = item.get('answer', item.get('ground_truth', ''))
            all_answers = [gt] if gt else []

        # Extract answer strings from dict format if needed
        if all_answers and isinstance(all_answers[0], dict):
            all_answers = [a.get('answer', '') for a in all_answers if 'answer' in a]

        # Get prediction
        pred = item.get('output', item.get('response', item.get('prediction', '')))

        # Compute similarity
        sim = compute_similarity(all_answers, pred, encoder)
        item['embedding_similarity'] = sim
        similarities.append(sim)

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(data)} items...")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Summary stats
    avg_sim = np.mean(similarities) if similarities else 0.0
    print(f"\n✓ Saved: {output_path}")
    print(f"  Average similarity: {avg_sim:.4f}")

    return {
        'file': os.path.basename(input_path),
        'num_items': len(data),
        'avg_similarity': float(avg_sim),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute embedding similarity for VQA files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python compute_similarity.py --input scored/vqa_1k.jsonl --output_dir similarity_scored/

  # All VQA files in directory
  python compute_similarity.py --input_dir scored/ --output_dir similarity_scored/

  # Specific files
  python compute_similarity.py --input scored/*vqa*.jsonl --output_dir similarity_scored/
        """
    )

    parser.add_argument("--input", type=str, nargs='*', default=None,
                        help="Input JSONL file(s)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory with scored JSONL files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for files with similarity scores")
    parser.add_argument("--pattern", type=str, default="*vqa*.jsonl",
                        help="File pattern for input_dir (default: *vqa*.jsonl)")

    args = parser.parse_args()

    # Load encoder
    encoder = get_encoder()
    if not encoder:
        print("❌ Failed to load encoder. Exiting.")
        return

    # Collect files to process
    files = []
    if args.input_dir:
        pattern_path = os.path.join(args.input_dir, args.pattern)
        files = sorted(glob(pattern_path))
    elif args.input:
        for input_path in args.input:
            if '*' in input_path:
                files.extend(glob(input_path))
            else:
                files.append(input_path)
    else:
        parser.print_help()
        return

    if not files:
        print("❌ No files found!")
        return

    print(f"\nFound {len(files)} files to process")

    # Process each file
    all_results = []
    for input_path in files:
        # Determine output path
        if args.input_dir:
            rel_path = os.path.relpath(input_path, args.input_dir)
            output_path = os.path.join(args.output_dir, rel_path)
        else:
            output_path = os.path.join(args.output_dir, os.path.basename(input_path))

        result = process_file(input_path, output_path, encoder)
        if result:
            all_results.append(result)

    # Print summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in all_results:
            print(f"{r['file']}: {r['avg_similarity']:.4f} ({r['num_items']} items)")

        # Save summary
        summary_path = os.path.join(args.output_dir, 'similarity_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Summary saved: {summary_path}")

    print(f"\n✅ Processed {len(all_results)} files")


if __name__ == '__main__':
    main()
