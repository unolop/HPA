#!/usr/bin/env python3
"""
Complete Preprocessing Pipeline for Human VQA Responses

Selects N participants, preprocesses responses, and creates training data.

Usage:
    python prepare_human_training_data.py \
        --human_results_dir outputs/results/humans/all_results_20251202_112501 \
        --questions_csv data/questions/s1.csv \
        --output_dir data/train \
        --num_participants 10 \
        --use_clustering \
        --translate
"""

import os
import sys
import json
import csv
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import preprocessing functions
from preprocess_answers import (
    TranslationCache,
    setup_openai_client,
    translate_answers,
    normalize_answer,
    normalize_number_words,
    cluster_answers,
    load_questions,
)
from prepare_training_data import (
    aggregate_responses,
    create_training_jsonl,
    create_individual_jsonl,
    create_train_val_split,
)


def select_participants(
    results_dir: str,
    num_participants: int = None,
    min_completion: float = 0.9,
    seed: int = 42,
) -> List[str]:
    """
    Select N participants from results directory.

    Args:
        results_dir: Directory with participant folders
        num_participants: Number to select (None = all)
        min_completion: Minimum completion rate (0.0-1.0)
        seed: Random seed for selection

    Returns:
        List of participant folder paths
    """
    random.seed(seed)

    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_dir}")

    # Find all participant folders
    participant_folders = [
        d for d in results_path.iterdir()
        if d.is_dir() and (d / "answers.csv").exists()
    ]

    print(f"Found {len(participant_folders)} participant folders")

    # Filter by completion rate
    valid_participants = []

    for folder in participant_folders:
        participant_json = folder / "participant.json"

        if participant_json.exists():
            with open(participant_json, 'r') as f:
                data = json.load(f)

            total = data.get('total_questions', 0)
            current = data.get('current_question', 0)
            completed = data.get('is_completed', False)

            if total > 0:
                completion_rate = current / total
            else:
                completion_rate = 0.0

            if completion_rate >= min_completion or completed:
                valid_participants.append({
                    'folder': folder,
                    'participant_id': data.get('participant_id', folder.name),
                    'name': data.get('name', 'Unknown'),
                    'completion': completion_rate,
                    'total_questions': current,
                })
        else:
            # No participant.json, include anyway
            valid_participants.append({
                'folder': folder,
                'participant_id': folder.name,
                'name': folder.name,
                'completion': 1.0,
                'total_questions': 0,
            })

    print(f"Valid participants (>{min_completion*100}% complete): {len(valid_participants)}")

    # Sort by completion rate (descending)
    valid_participants.sort(key=lambda x: x['completion'], reverse=True)

    # Select N participants
    if num_participants is None or num_participants >= len(valid_participants):
        selected = valid_participants
    else:
        # Randomly select from valid participants
        selected = random.sample(valid_participants, num_participants)

    print(f"\n📋 Selected {len(selected)} participants:")
    for p in selected:
        print(f"   {p['name'][:20]:20} - {p['completion']*100:5.1f}% ({p['total_questions']} questions)")

    return [p['folder'] for p in selected]


def load_participant_responses(
    participant_folders: List[Path],
    questions: Dict[str, Dict],
) -> List[Dict]:
    """
    Load responses from selected participant folders.

    Each participant has:
    - answers.csv: question_num, qid, answer, confidence, time_spent_seconds, answer_timestamp
    - participant.json: metadata
    """
    all_responses = []

    for folder in participant_folders:
        answers_csv = folder / "answers.csv"
        participant_json = folder / "participant.json"

        # Load participant metadata
        participant_id = folder.name
        participant_name = "Unknown"

        if participant_json.exists():
            with open(participant_json, 'r') as f:
                metadata = json.load(f)
                participant_id = metadata.get('participant_id', folder.name)
                participant_name = metadata.get('name', 'Unknown')

        # Load answers
        with open(answers_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)

            for row in reader:
                qid = str(row.get('qid', ''))
                answer = str(row.get('answer', ''))
                confidence = int(row.get('confidence', 3))
                time_spent = float(row.get('time_spent_seconds', 0))

                if not qid or not answer:
                    continue

                # Get question info
                q_info = questions.get(qid, {})

                all_responses.append({
                    'qid': qid,
                    'answer': answer,
                    'confidence': confidence,
                    'time_spent': time_spent,
                    'participant_id': participant_id,
                    'participant_name': participant_name,
                    'question': q_info.get('question', ''),
                    'category': q_info.get('category', ''),
                    'answer_type': q_info.get('answer_type', ''),
                    'options': q_info.get('options', ''),
                })

    print(f"✓ Loaded {len(all_responses)} responses from {len(participant_folders)} participants")
    return all_responses


def preprocess_responses(
    responses: List[Dict],
    questions: Dict[str, Dict],
    cache_file: str,
    translate: bool = True,
    cluster: bool = False,
    cluster_threshold: float = 0.5,
) -> List[Dict]:
    """
    Preprocess responses: translation, normalization, optional clustering.
    """
    print("\n" + "=" * 60)
    print("🔧 PREPROCESSING RESPONSES")
    print("=" * 60)

    # Translation
    if translate:
        print("\n[1/3] Translating Korean answers...")
        cache = TranslationCache(cache_file, save_every=10)
        client = setup_openai_client()

        if client:
            responses = translate_answers(responses, questions, cache, client)
        else:
            print("⚠ Skipping translation (no API client)")
    else:
        print("\n[1/3] Skipping translation")

    # Normalization
    print("\n[2/3] Normalizing answers...")
    for r in responses:
        r['answer_raw'] = r['answer']
        r['answer_normalized'] = normalize_answer(r['answer'])
        r['answer_normalized'] = normalize_number_words(r['answer_normalized'])

    # Clustering
    if cluster:
        print("\n[3/3] Clustering similar answers...")
        responses = cluster_answers(responses, cluster_threshold)
    else:
        print("\n[3/3] Skipping clustering")
        for r in responses:
            r['cluster_id'] = None
            r['cluster_answer'] = r['answer_normalized']

    print("✓ Preprocessing complete")
    return responses


def create_training_files(
    responses: List[Dict],
    output_dir: str,
    black_image_path: str,
    use_aggregation: bool = True,
    use_clustering: bool = False,
    with_instruction: bool = True,
    create_split: bool = True,
    train_ratio: float = 0.9,
):
    """
    Create training JSONL files.
    """
    print("\n" + "=" * 60)
    print("📦 CREATING TRAINING FILES")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Instruction prefix
    instruction = ""
    if with_instruction:
        instruction = "Note: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n"

    # Save raw preprocessed responses
    raw_path = os.path.join(output_dir, 'preprocessed_responses.json')
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved raw responses: {raw_path}")

    # Create aggregated version
    if use_aggregation:
        print("\n[Aggregation Mode]")
        aggregated = aggregate_responses(responses, by_cluster=use_clustering)

        jsonl_path = os.path.join(output_dir, 'train_aggregated.jsonl')
        create_training_jsonl(
            aggregated,
            jsonl_path,
            black_image_path,
            instruction,
            min_confidence=None,
            min_responses=1,
        )

        if create_split:
            create_train_val_split(jsonl_path, train_ratio)

    # Create individual version
    print("\n[Individual Mode]")
    individual_path = os.path.join(output_dir, 'train_individual.jsonl')
    create_individual_jsonl(
        responses,
        individual_path,
        black_image_path,
        instruction,
    )

    if create_split:
        create_train_val_split(individual_path, train_ratio)

    # Statistics
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   Total responses: {len(responses)}")
    print(f"   Unique questions: {len(set(r['qid'] for r in responses))}")
    print(f"   Participants: {len(set(r['participant_id'] for r in responses))}")
    if use_aggregation:
        print(f"   Aggregated examples: {len(aggregated)}")
    print(f"   Output directory: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Complete preprocessing pipeline for human VQA responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: Select 10 participants, aggregate, translate
  python prepare_human_training_data.py \\
      --human_results_dir outputs/results/humans/all_results_20251202_112501 \\
      --questions_csv data/questions/s1.csv \\
      --output_dir data/train \\
      --num_participants 10 \\
      --translate

  # Advanced: All participants, clustering, no translation
  python prepare_human_training_data.py \\
      --human_results_dir outputs/results/humans/all_results_20251202_112501 \\
      --questions_csv data/questions/s1.csv \\
      --output_dir data/train \\
      --use_clustering --cluster_threshold 0.4
        """
    )

    # Required
    parser.add_argument("--human_results_dir", type=str, required=True,
                        help="Directory with participant folders (e.g., all_results_20251202_112501)")
    parser.add_argument("--questions_csv", type=str, required=True,
                        help="Questions CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for training data")

    # Participant selection
    parser.add_argument("--num_participants", type=int, default=None,
                        help="Number of participants to select (None = all)")
    parser.add_argument("--min_completion", type=float, default=0.9,
                        help="Minimum completion rate (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for participant selection")

    # Preprocessing
    parser.add_argument("--translate", action="store_true",
                        help="Translate Korean answers to English")
    parser.add_argument("--cache_file", type=str, default=None,
                        help="Translation cache file (default: data/translation_cache.json)")
    parser.add_argument("--use_clustering", action="store_true",
                        help="Use semantic clustering instead of exact matching")
    parser.add_argument("--cluster_threshold", type=float, default=0.5,
                        help="Clustering distance threshold (0.3=tight, 0.7=loose)")

    # Training data format
    parser.add_argument("--no_aggregation", action="store_true",
                        help="Don't aggregate (use individual responses only)")
    parser.add_argument("--no_instruction", action="store_true",
                        help="Don't add instruction prefix")
    parser.add_argument("--no_split", action="store_true",
                        help="Don't create train/val split")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Train set ratio (default: 0.9)")

    # Paths
    parser.add_argument("--black_image_path", type=str,
                        default="data/blank_224.png",
                        help="Path to black placeholder image")

    args = parser.parse_args()

    # Set default cache file
    if args.cache_file is None:
        args.cache_file = "data/translation_cache.json"

    print("=" * 60)
    print("🚀 HUMAN VQA TRAINING DATA PREPARATION")
    print("=" * 60)
    print(f"Results dir: {args.human_results_dir}")
    print(f"Questions: {args.questions_csv}")
    print(f"Output: {args.output_dir}")
    print(f"Participants: {args.num_participants if args.num_participants else 'ALL'}")
    print(f"Translate: {args.translate}")
    print(f"Clustering: {args.use_clustering}")
    print("=" * 60)

    # Step 1: Load questions
    print("\n[1/4] Loading questions...")
    questions = load_questions(args.questions_csv)

    # Step 2: Select participants
    print("\n[2/4] Selecting participants...")
    participant_folders = select_participants(
        args.human_results_dir,
        args.num_participants,
        args.min_completion,
        args.seed,
    )

    # Step 3: Load and preprocess responses
    print("\n[3/4] Loading responses...")
    responses = load_participant_responses(participant_folders, questions)

    responses = preprocess_responses(
        responses,
        questions,
        args.cache_file,
        args.translate,
        args.use_clustering,
        args.cluster_threshold,
    )

    # Step 4: Create training files
    print("\n[4/4] Creating training files...")
    create_training_files(
        responses,
        args.output_dir,
        args.black_image_path,
        use_aggregation=not args.no_aggregation,
        use_clustering=args.use_clustering,
        with_instruction=not args.no_instruction,
        create_split=not args.no_split,
        train_ratio=args.train_ratio,
    )

    print("\n✅ Pipeline complete!")
    print(f"\nTraining data saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Check the data:")
    print(f"     cat {args.output_dir}/train_aggregated_train.jsonl | head -1 | jq")
    print("  2. Run training:")
    print("     bash experiments/train_a4_blind.sh")


if __name__ == "__main__":
    main()
