#!/usr/bin/env python3
"""
Create GT answer version of blind instruction training data.

Takes blind instruction training data (which has human confidences) and:
1. Replaces the assistant answer with ground truth answer
2. Keeps the human confidence distribution in labels
3. Uses real images for VQA questions (not blank images)

This creates a hybrid training set that:
- Trains the model to generate GT answers (not blind answers)
- But still uses human confidence distributions for alignment loss
"""

import json
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from dataset.vqav2 import VQADataset


def load_vqa_annotations(prompt: str = ""):
    """Load VQA ground truth annotations."""
    print("Loading VQA ground truth annotations...")
    vqa_dataset = VQADataset(prompt=prompt)
    annotations_by_qid = vqa_dataset.get_by_qid()
    print(f"Loaded {len(annotations_by_qid)} VQA annotations")
    return annotations_by_qid


def replace_with_gt_answers(
    input_path: str,
    output_path: str,
    answer_type: str = "text",  # "text" or "choice"
):
    """
    Replace blind instruction answers with ground truth answers.

    Args:
        input_path: Path to blind instruction training data
        output_path: Path to save GT answer version
        answer_type: "text" for VQA (open-ended) or "choice" for multiple choice
    """
    print(f"\n{'=' * 80}")
    print(f"Creating GT Answer Version from Blind Instruction Data")
    print(f"{'=' * 80}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Type:   {answer_type}")
    print(f"{'=' * 80}\n")

    # Load blind instruction data
    print("Loading blind instruction data...")
    with open(input_path, 'r', encoding='utf-8') as f:
        blind_data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(blind_data)} samples from blind instruction data\n")

    # Load GT annotations based on type
    if answer_type == "text":
        # For VQA open-ended questions
        prompt = "\nNote: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario."
        gt_annotations = load_vqa_annotations(prompt=prompt)
    else:
        # For multiple choice questions
        # Load from mmstar or other source
        # For now, we'll use the first answer from labels as GT
        # (assuming it's the most confident one)
        gt_annotations = None
        print("Multiple choice mode: using most confident answer from labels as GT\n")

    # Process each item
    processed_data = []
    skipped = 0
    updated = 0

    print("Processing samples...")
    for item in tqdm(blind_data, desc="Replacing answers with GT"):
        qid = item['qid']

        # Get GT answer
        gt_answer = None
        gt_image = None

        if answer_type == "text" and gt_annotations:
            # Get from VQA annotations
            qid_int = int(qid) if str(qid).isdigit() else None
            qid_str = str(qid)

            if qid_int and qid_int in gt_annotations:
                ann = gt_annotations[qid_int]
                gt_answer = ann.get('multiple_choice_answer', '')
                gt_image = ann.get('image', '')
            elif qid_str in gt_annotations:
                ann = gt_annotations[qid_str]
                gt_answer = ann.get('multiple_choice_answer', '')
                gt_image = ann.get('image', '')
            else:
                print(f"  Warning: No GT annotation found for qid={qid}, skipping")
                skipped += 1
                continue

        else:
            # For multiple choice or if no GT annotations
            # Use the first (most confident) answer from labels as GT
            if 'labels' in item and 'answers' in item['labels']:
                answers = item['labels']['answers']
                if answers:
                    gt_answer = answers[0]  # Most confident answer
                else:
                    print(f"  Warning: No answers in labels for qid={qid}, skipping")
                    skipped += 1
                    continue
            else:
                print(f"  Warning: No labels found for qid={qid}, skipping")
                skipped += 1
                continue

        # Create new item with GT answer
        new_item = item.copy()

        # Update assistant answer in conversations
        if 'conversations' in new_item:
            for conv in new_item['conversations']:
                if conv['role'] == 'assistant':
                    # Replace blind answer with GT answer
                    conv['content'] = gt_answer
                    updated += 1
                    break

        # Update image path if we have GT image (for VQA)
        if gt_image and answer_type == "text":
            new_item['images'] = [gt_image]

        # Keep the human confidence distribution in labels
        # (this is what we want - GT answer but human confidences)

        processed_data.append(new_item)

    # Save processed data
    print(f"\nSaving processed data...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n{'=' * 80}")
    print(f"Summary:")
    print(f"{'=' * 80}")
    print(f"  Total input samples:    {len(blind_data)}")
    print(f"  Samples updated:        {len(processed_data)}")
    print(f"  Skipped (no GT):        {skipped}")
    print(f"  Assistant answers replaced: {updated}")
    print(f"\n  Output saved: {output_path}")
    print(f"{'=' * 80}\n")

    return processed_data


def main():
    parser = argparse.ArgumentParser(
        description="Replace blind instruction answers with ground truth answers"
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to blind instruction training data")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save GT answer version")
    parser.add_argument("--answer_type", type=str, default="text",
                        choices=["text", "choice"],
                        help="Answer type: text (VQA open-ended) or choice (multiple choice)")

    args = parser.parse_args()

    replace_with_gt_answers(
        input_path=args.input_path,
        output_path=args.output_path,
        answer_type=args.answer_type
    )


if __name__ == "__main__":
    main()
