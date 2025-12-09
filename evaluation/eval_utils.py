#!/usr/bin/env python3
"""
Utility functions for evaluation processing.
"""
import json
import re
from typing import Dict, List
import numpy as np


# Dataset type mappings
DATASET_TYPES = {
    'mmstar': 'multi-choice',
    'spubench': 'multi-choice',
    'vqa_1k': 'open-ended',
    'vqa_5k': 'open-ended',
    'vqa1k': 'open-ended',
    'vqa5k': 'open-ended',
}


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def save_jsonl(data: List[Dict], filepath: str):
    """Save to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_mc_choice(output: str) -> str:
    """Extract multiple choice answer (A, B, C, D) from output."""
    output = output.strip()

    patterns = [
        r"[Aa]nswer\s*(?:is)?\s*:?\s*\n*\s*([A-Da-d])",
        r"(?:correct|right)\s+(?:answer|choice)\s+(?:is)?\s*:?\s*\n*\s*([A-Da-d])",
        r"^([A-Da-d])[\.\)\s]",
        r"\n([A-Da-d])\s*:",
        r"([A-Da-d])$",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).upper()

    if len(output) == 1 and output.upper() in 'ABCD':
        return output.upper()

    if output and output[0].upper() in 'ABCD':
        return output[0].upper()

    return output


def mc_accuracy(gt: str, pred: str) -> bool:
    """Multiple choice accuracy."""
    gt_letter = gt.strip().upper()[0] if gt else ""
    pred_letter = extract_mc_choice(pred)
    return gt_letter == pred_letter


def normalize_answer(text: str) -> str:
    """Normalize answer text."""
    text = text.lower().strip()
    # Remove articles
    for article in ['a ', 'an ', 'the ']:
        if text.startswith(article):
            text = text[len(article):]
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def exact_match(gt: str, pred: str) -> bool:
    """Exact match after normalization."""
    return normalize_answer(gt) == normalize_answer(pred)


def vqa_accuracy(all_answers: List[str], pred: str) -> float:
    """VQA accuracy: min(#matches / 3, 1.0)."""
    pred_norm = normalize_answer(pred)
    matches = sum(1 for ans in all_answers if normalize_answer(ans) == pred_norm)
    return min(matches / 3.0, 1.0)


def answer_similarity(answers: List[str], pred: str, encoder) -> float:
    """Compute embedding similarity between answers and prediction."""
    try:
        pred_emb = encoder.encode([pred], convert_to_numpy=True)[0]
        answer_embs = encoder.encode(answers, convert_to_numpy=True)

        # Compute cosine similarity with each answer
        similarities = []
        for ans_emb in answer_embs:
            sim = np.dot(pred_emb, ans_emb) / (
                np.linalg.norm(pred_emb) * np.linalg.norm(ans_emb) + 1e-10
            )
            similarities.append(sim)

        return float(np.mean(similarities))
    except Exception as e:
        return 0.0


def parse_filename(filepath: str) -> Dict[str, str]:
    """
    Parse filename to extract metadata.

    Returns:
        {
            'source': 'models'|'humans'|'finetuned',
            'model': model name,
            'dataset': dataset name,
            'condition': condition string,
            'training_method': training method (for finetuned),
        }
    """
    filepath = str(filepath)

    # Determine source
    if '/humans/' in filepath:
        source = 'humans'
    elif '/finetuned/' in filepath:
        source = 'finetuned'
    elif '/models/' in filepath:
        source = 'models'
    else:
        source = 'unknown'

    # Get filename without path and extension
    filename = filepath.split('/')[-1].replace('.jsonl', '')

    # Extract condition
    conditions = ['sys_inst_blind', 'inst_blind', 'blind', '']
    condition = ''
    for cond in conditions:
        if cond and cond in filename:
            condition = cond
            filename = filename.replace(f'_{cond}', '').replace(cond, '')
            break

    # Special handling for humans
    if source == 'humans':
        condition = 'blind_inst' if 'blind' in filepath else condition

    # Extract model name
    model_names = [
        "InternVL3_5-8B", "InternVL3_5-4B", "InternVL3_5-2B", "InternVL3_5-1B",
        "Qwen3-VL-8B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-2B-Instruct",
        "llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-1.5-7b-hf",
    ]

    model = 'human' if source == 'humans' else None
    for m in model_names:
        if m in filename:
            model = m
            filename = filename.replace(f'{m}_', '').replace(m, '')
            break

    # Extract training method (for finetuned)
    training_method = None
    if source == 'finetuned':
        methods = ['alignment_js', 'standard', 'sft']
        for method in methods:
            if method in filename:
                training_method = method
                filename = filename.replace(f'{method}_', '').replace(method, '')
                break

    # Extract dataset
    dataset = filename.strip('_')
    for ds in ['mmstar', 'spubench', 'vqa_1k', 'vqa_5k', 'vqa1k', 'vqa5k']:
        if ds in dataset:
            dataset = ds.replace('_', '')  # Normalize to vqa1k, vqa5k
            break

    return {
        'source': source,
        'model': model or 'unknown',
        'dataset': dataset,
        'condition': condition,
        'training_method': training_method,
    }


def compute_accuracy(item: Dict, dataset_type: str, all_answers: List[str] = None) -> bool:
    """Compute accuracy for a single item."""
    gt = item.get('answer', item.get('ground_truth', ''))
    pred = item.get('output', item.get('response', item.get('prediction', '')))

    if not all_answers:
        all_answers = item.get('all_answers', item.get('answers', []))
        if not all_answers:
            all_answers = [gt] if gt else []

        # Extract answer strings from dict format if needed
        if all_answers and isinstance(all_answers[0], dict):
            all_answers = [a.get('answer', '') for a in all_answers if 'answer' in a]

    if dataset_type == 'multi-choice':
        return mc_accuracy(gt, pred)
    else:
        # VQA accuracy
        if len(all_answers) > 1:
            vqa_acc = vqa_accuracy(all_answers, pred)
            return vqa_acc >= 0.5
        else:
            return exact_match(gt, pred)


def get_category(item: Dict, dataset_name: str) -> str:
    """Get category for an item."""
    # For mmstar, use both category and l2_category
    if dataset_name == 'mmstar' and 'l2_category' in item:
        return f"{item.get('category', 'Unknown')} | {item.get('l2_category', 'Unknown')}"
    else:
        return item.get('category', item.get('question_type', 'Unknown'))
