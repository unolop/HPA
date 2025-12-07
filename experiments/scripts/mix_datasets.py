#!/usr/bin/env python3
"""
Mix human blind VQA data with original visual VQA data.

This prevents catastrophic forgetting by maintaining visual capabilities
while learning from human responses.
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def mix_datasets(
    human_data: List[Dict],
    original_data: List[Dict],
    human_ratio: float = 0.8,
    seed: int = 42,
) -> List[Dict]:
    """
    Mix two datasets with specified ratio.

    Args:
        human_data: Human responses (blind VQA)
        original_data: Original dataset (with images)
        human_ratio: Proportion of human data (0.0-1.0)
        seed: Random seed

    Returns:
        Mixed dataset
    """
    random.seed(seed)

    total_size = len(human_data) + len(original_data)
    target_human = int(total_size * human_ratio)
    target_original = total_size - target_human

    print(f"Mixing datasets:")
    print(f"  Human data: {len(human_data)} examples")
    print(f"  Original data: {len(original_data)} examples")
    print(f"  Target ratio: {human_ratio:.1%} human / {1-human_ratio:.1%} original")

    # Sample to target sizes
    if len(human_data) > target_human:
        sampled_human = random.sample(human_data, target_human)
    else:
        # Oversample if needed
        sampled_human = human_data * (target_human // len(human_data) + 1)
        sampled_human = random.sample(sampled_human, target_human)

    if len(original_data) > target_original:
        sampled_original = random.sample(original_data, target_original)
    else:
        # Oversample if needed
        sampled_original = original_data * (target_original // len(original_data) + 1)
        sampled_original = random.sample(sampled_original, target_original)

    # Mark data sources
    for item in sampled_human:
        item['data_source'] = 'human_blind'

    for item in sampled_original:
        item['data_source'] = 'original_visual'

    # Combine and shuffle
    mixed = sampled_human + sampled_original
    random.shuffle(mixed)

    print(f"\nMixed dataset:")
    print(f"  Total: {len(mixed)} examples")
    print(f"  Human: {len(sampled_human)} ({len(sampled_human)/len(mixed)*100:.1f}%)")
    print(f"  Original: {len(sampled_original)} ({len(sampled_original)/len(mixed)*100:.1f}%)")

    return mixed


def main():
    parser = argparse.ArgumentParser(
        description="Mix human blind VQA with original visual VQA data"
    )

    parser.add_argument("--human_data", type=str, required=True,
                        help="Human blind VQA JSONL file")
    parser.add_argument("--original_data", type=str, required=True,
                        help="Original VQA JSONL file (with images)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output mixed JSONL file")
    parser.add_argument("--human_ratio", type=float, default=0.8,
                        help="Proportion of human data (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print("🔀 MIXING DATASETS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    human_data = load_jsonl(args.human_data)
    original_data = load_jsonl(args.original_data)

    # Mix
    mixed_data = mix_datasets(
        human_data,
        original_data,
        args.human_ratio,
        args.seed,
    )

    # Save
    print(f"\nSaving to: {args.output}")
    save_jsonl(mixed_data, args.output)

    print("\n✅ Dataset mixing complete!")


if __name__ == "__main__":
    main()
