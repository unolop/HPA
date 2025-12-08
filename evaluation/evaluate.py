#!/usr/bin/env python3
"""
4. evaluate_models.py - Unified Evaluation for Blind VQA

Integrates with:
- Your inference.py for model inference
- Your scoring.py for VQA accuracy and embedding similarity

Evaluates:
- Blind condition (black images)
- Original condition (real images)
- Computes accuracy, embedding similarity, per-category breakdown

Usage:
    # Evaluate a trained model
    python evaluate_models.py \
        --model OpenGVLab/InternVL3_5-2B \
        --lora_path ./output/A5/checkpoint-100 \
        --dataset mmstar \
        --condition _inst_blind \
        --output_dir ./eval_results/A5

    # Evaluate base model (zero-shot)
    python evaluate_models.py \
        --model OpenGVLab/InternVL3_5-2B \
        --dataset mmstar \
        --condition _inst_blind \
        --output_dir ./eval_results/A0
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np

import torch
torch.cuda.empty_cache()

# Seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# =============================================================================
# Constants
# =============================================================================

SYSTEM_MESSAGE = "Note: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n"

DATASET_TYPE = {
    'mmstar': 'multi-choice',
    'spubench': 'multi-choice',
    'vqa_1k': 'open-ended',
    'vqa_5k': 'open-ended',
}

BLACK_IMAGE_PATH = "/home/work/yuna/HPA/data/blank_224.png"


# =============================================================================
# Scoring Functions (from your scoring.py)
# =============================================================================

def get_encoder():
    """Lazy load sentence transformer encoder."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2").to('cuda')
    except Exception as e:
        print(f"⚠ Could not load SentenceTransformer: {e}")
        return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    answer = str(answer).lower().strip()
    # Remove articles
    for article in ['a ', 'an ', 'the ']:
        if answer.startswith(article):
            answer = answer[len(article):]
    # Remove punctuation
    import string
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(answer.split()).strip()


def extract_mc_choice(output: str) -> str:
    """Extract multiple choice answer (A, B, C, D) from output."""
    output = output.strip()

    # Try to find explicit answer patterns (more flexible to handle newlines)
    patterns = [
        r"[Aa]nswer\s*(?:is)?\s*:?\s*\n*\s*([A-Da-d])",  # "answer is:\n\nA" or "answer: A"
        r"(?:correct|right)\s+(?:answer|choice)\s+(?:is)?\s*:?\s*\n*\s*([A-Da-d])",  # "correct answer is:\n\nA"
        r"^([A-Da-d])[\.\)\s]",  # "A. " at start
        r"\n([A-Da-d])\s*:",  # "\nA:" format
        r"([A-Da-d])$",  # "A" at end
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # If output is just a single letter
    if len(output) == 1 and output.upper() in 'ABCD':
        return output.upper()

    # Return first letter if it's A-D
    if output and output[0].upper() in 'ABCD':
        return output[0].upper()

    return output


def vqa_accuracy(gt_answers: List[str], pred: str) -> float:
    """
    VQA accuracy: min(1, #matches / 3).
    
    gt_answers: list of ground truth answers
    pred: predicted answer string
    """
    pred = normalize_answer(pred)
    
    matches = sum([
        pred == normalize_answer(ans)
        for ans in gt_answers
    ])
    
    return min(1.0, matches / 3.0)


def exact_match(gt: str, pred: str) -> bool:
    """Check exact match after normalization."""
    return normalize_answer(pred) == normalize_answer(gt)


def mc_accuracy(gt: str, pred: str) -> bool:
    """Multiple choice accuracy."""
    gt_letter = gt.strip().upper()[0] if gt else ""
    pred_letter = extract_mc_choice(pred)
    return gt_letter == pred_letter


def answer_similarity(gt_answers: List[str], pred: str, encoder=None) -> float:
    """
    Compute embedding similarity between prediction and ground truth answers.
    
    gt_answers: list of ground truth strings
    pred: predicted answer string
    """
    if encoder is None:
        return 0.0
    
    pred = pred.strip().lower()
    scores = []
    
    for gta in gt_answers:
        gta = str(gta).strip().lower()
        try:
            emb = encoder.encode([pred, gta])
            similarities = encoder.similarity(emb, emb)
            scores.append(float(similarities[1, 0]))
        except Exception as e:
            continue
    
    return float(np.mean(scores)) if scores else 0.0


# =============================================================================
# Dataset Loading (from your inference.py)
# =============================================================================

def load_dataset(data_name: str, prompt: str = '') -> List[Dict]:
    """Load dataset for evaluation."""
    print(f"📂 Loading dataset: {data_name}")
    
    if data_name == "mmstar":
        from datasets import load_dataset
        dataset = load_dataset("Lin-Chen/MMStar", split="val")
        # Convert to list of dicts
        data_list = []
        for i, item in enumerate(dataset):
            data_list.append({
                'qid': str(i),
                'question': item['question'],
                'answer': item['answer'],
                'image': item['image'],
                'category': item.get('category', item.get('l2_category', '')),
                'options': item.get('options', ''),
            })
        return data_list
    
    elif data_name == "spubench":
        from datasets import load_dataset
        
        with open('/home/work/yuna/HPA/data/annotation.json', 'r', encoding='utf-8') as f:
            annot = json.load(f)
        
        data_list = []
        ds = load_dataset("mmbench/MM-SpuBench", streaming=True)['train']
        
        for i, (data, ann) in enumerate(zip(ds, annot)):
            image = data['image']
            question = ann['question']
            choices = ann['choices']
            choices_text = "\n".join(choices)
            
            formatted_q = (
                f"{prompt}Question: {question}\n"
                f"{choices_text}\n"
                "Provide only the letter corresponding to the correct choice (A, B, C, or D).\n"
                "Answer:"
            )
            
            data_list.append({
                'qid': str(i),
                'question': formatted_q,
                'answer': ann.get('answer', ''),
                'image': image,
                'category': ann.get('spurious_type', ann.get('category', '')),
            })
        
        return data_list
    
    elif data_name == "vqa_1k":
        from data.vqav2 import VQADataset_json
        dataset = VQADataset_json(prompt=prompt)
        
        data_list = []
        for i in range(len(dataset)):
            item = dataset[i]
            data_list.append({
                'qid': str(item.get('question_id', i)),
                'question': item['question'],
                'answer': item.get('answer', ''),
                'all_answers': item.get('answers', [item.get('answer', '')]),
                'image': item.get('image', item.get('image_path', '')),
                'category': item.get('question_type', ''),
            })
        return data_list
    
    elif data_name == "vqa_5k":
        from data.vqav2 import VQADataset
        from torch.utils.data import Subset
        
        with open('/home/work/yuna/HPA/data/s1_qids.json', 'r') as file:
            qids = json.load(file)
        
        dataset = VQADataset(prompt=prompt, filter_qids=qids)
        indices = np.random.choice(len(dataset), size=min(5000, len(dataset)), replace=False)
        
        data_list = []
        for idx in indices:
            item = dataset[idx]
            data_list.append({
                'qid': str(item.get('question_id', idx)),
                'question': item['question'],
                'answer': item.get('answer', ''),
                'all_answers': item.get('answers', [item.get('answer', '')]),
                'image': item.get('image', item.get('image_path', '')),
                'category': item.get('question_type', ''),
            })
        return data_list
    
    else:
        raise ValueError(f"Unknown dataset: {data_name}")


# =============================================================================
# Model Loading and Inference
# =============================================================================

def load_model_for_eval(
    model_path: str,
    lora_path: str = None,
    device_map: str = "auto",
):
    """
    Load model for evaluation using SWIFT.
    
    Returns:
        engine, request_config
    """
    from swift.llm import (
        PtEngine, RequestConfig, safe_snapshot_download, 
        get_model_tokenizer, get_template
    )
    from swift.tuners import Swift
    
    print(f"📦 Loading model: {model_path}")
    
    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(model_path, use_hf=True)
    
    # Load LoRA if provided
    if lora_path is not None:
        print(f"🔌 Loading LoRA adapter: {lora_path}")
        lora_checkpoint = safe_snapshot_download(lora_path)
        model = Swift.from_pretrained(model, lora_checkpoint)
    
    model.eval()
    
    # Get template
    template_type = model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=None)
    
    # Create engine
    engine = PtEngine.from_model_template(model, template, max_batch_size=1)
    request_config = RequestConfig(max_tokens=512, temperature=0)
    
    print("✓ Model loaded successfully")
    return engine, request_config


def run_inference(
    engine,
    request_config,
    data_list: List[Dict],
    condition: str = "",
    is_blind: bool = False,
    model_type: str = "vlm",
    dataset_name: str = "mmstar",
) -> List[Dict]:
    """
    Run inference on dataset.
    
    Args:
        engine: SWIFT PtEngine
        request_config: SWIFT RequestConfig
        data_list: List of data items
        condition: Condition string (_blind, _inst_blind, etc.)
        is_blind: Whether to use black images
        model_type: 'vlm' or 'llm'
        dataset_name: Name of dataset
    
    Returns:
        List of results with predictions
    """
    from swift.llm import InferRequest
    
    results = []
    
    for data in tqdm(data_list, desc="Inference"):
        prompt = data['question']
        
        # Handle blind condition
        image_path = data.get('image', BLACK_IMAGE_PATH)
        if is_blind or 'blind' in condition:
            image_path = BLACK_IMAGE_PATH
        
        # Add instruction based on condition
        if condition == '_inst_blind' and 'vqa' not in dataset_name:
            prompt += SYSTEM_MESSAGE
        
        # Add MC prompt for non-VQA datasets
        dataset_type = DATASET_TYPE.get(dataset_name, 'multi-choice')
        if model_type == 'vlm' and dataset_type == 'multi-choice':
            prompt += '\nPlease select the correct answer from the options above.'
        
        # Build messages
        messages = []
        if 'sys_inst' in condition:
            messages.append({'role': 'system', 'content': SYSTEM_MESSAGE})
        
        if model_type == 'vlm':
            messages.append({'role': 'user', 'content': f'<image>{prompt}'})
            infer_request = InferRequest(
                messages=messages,
                images=[image_path] if isinstance(image_path, str) else [image_path]
            )
        else:
            messages.append({'role': 'user', 'content': prompt})
            infer_request = InferRequest(messages=messages)
        
        # Run inference
        try:
            resp_list = engine.infer([infer_request], request_config)
            output_text = resp_list[0].choices[0].message.content
        except Exception as e:
            print(f"⚠ Inference error for qid={data.get('qid')}: {e}")
            output_text = ""
        
        # Clean output
        matches = re.findall(r"Answer\s*:?\s*(.+)", output_text)
        if matches:
            output_text = matches[-1].strip().replace('*', '')
        else:
            output_text = output_text.strip().replace('*', '')
        
        # Store result
        result = {
            'qid': data.get('qid', ''),
            'question': data['question'],
            'ground_truth': data.get('answer', ''),
            'all_answers': data.get('all_answers', [data.get('answer', '')]),
            'prediction': output_text,
            'category': data.get('category', ''),
        }
        
        results.append(result)
    
    return results


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(
    results: List[Dict],
    dataset_type: str = "multi-choice",
    encoder = None,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics.
    
    Args:
        results: List of prediction results
        dataset_type: 'multi-choice' or 'open-ended'
        encoder: SentenceTransformer for embedding similarity
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'num_samples': len(results),
        'accuracy': 0.0,
        'embedding_similarity': 0.0,
        'by_category': {},
    }
    
    if not results:
        return metrics
    
    correct = 0
    similarities = []
    by_category = defaultdict(lambda: {'correct': 0, 'total': 0, 'similarities': []})
    
    for r in results:
        gt = r.get('ground_truth', '')
        pred = r.get('prediction', '')
        all_answers = r.get('all_answers', [gt])
        category = r.get('category', 'Unknown')
        
        # Accuracy
        if dataset_type == 'multi-choice':
            is_correct = mc_accuracy(gt, pred)
        else:
            # VQA accuracy
            if len(all_answers) > 1:
                is_correct = vqa_accuracy(all_answers, pred) >= 0.5
            else:
                is_correct = exact_match(gt, pred)
        
        r['correct'] = is_correct
        correct += int(is_correct)
        by_category[category]['correct'] += int(is_correct)
        by_category[category]['total'] += 1
        
        # Embedding similarity
        if encoder is not None:
            sim = answer_similarity(all_answers if all_answers else [gt], pred, encoder)
            similarities.append(sim)
            by_category[category]['similarities'].append(sim)
    
    metrics['accuracy'] = correct / len(results)
    metrics['num_correct'] = correct
    
    if similarities:
        metrics['embedding_similarity'] = np.mean(similarities)
    
    # Per-category
    for cat, cat_data in by_category.items():
        cat_acc = cat_data['correct'] / cat_data['total'] if cat_data['total'] > 0 else 0
        cat_sim = np.mean(cat_data['similarities']) if cat_data['similarities'] else 0
        metrics['by_category'][cat] = {
            'accuracy': cat_acc,
            'embedding_similarity': cat_sim,
            'num_samples': cat_data['total'],
        }
    
    return metrics


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def run_evaluation(
    model_path: str,
    dataset: str,
    condition: str = "",
    lora_path: str = None,
    output_dir: str = "./eval_results",
    model_type: str = "vlm",
    gpu: str = "0",
    compute_similarity: bool = True,
    max_samples: int = None,
    resume: bool = False,
):
    """
    Run full evaluation pipeline.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("📊 MODEL EVALUATION")
    print("=" * 70)
    print(f"   Model: {model_path}")
    print(f"   LoRA: {lora_path}")
    print(f"   Dataset: {dataset}")
    print(f"   Condition: {condition}")
    print("=" * 70)
    
    # Determine if blind
    is_blind = 'blind' in condition
    
    # Load encoder for similarity
    encoder = None
    if compute_similarity:
        encoder = get_encoder()
    
    # Load model
    engine, request_config = load_model_for_eval(model_path, lora_path)
    
    # Load dataset
    prompt = ""
    if 'blind' in condition and 'inst' in condition and 'sys' not in condition:
        prompt = '\n' + SYSTEM_MESSAGE
    
    data_list = load_dataset(dataset, prompt)
    
    if max_samples:
        data_list = data_list[:max_samples]
    
    print(f"✓ Loaded {len(data_list)} samples")
    
    # Check for resume
    model_name = lora_path.split('/')[-3] if lora_path else model_path.split('/')[-1]
    predictions_path = os.path.join(output_dir, f"{model_name}_{dataset}{condition}.jsonl")
    
    processed_ids = set()
    if resume and os.path.exists(predictions_path):
        with open(predictions_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    processed_ids.add(item.get('qid', ''))
                except:
                    continue
        print(f"   Resuming: {len(processed_ids)} already processed")
        data_list = [d for d in data_list if d.get('qid', '') not in processed_ids]
    
    # Run inference
    results = run_inference(
        engine, request_config, data_list,
        condition=condition,
        is_blind=is_blind,
        model_type=model_type,
        dataset_name=dataset,
    )
    
    # Compute metrics
    dataset_type = DATASET_TYPE.get(dataset, 'multi-choice')
    metrics = compute_metrics(results, dataset_type, encoder)
    
    # Print results
    print("\n" + "=" * 70)
    print("📈 RESULTS")
    print("=" * 70)
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['num_correct']}/{metrics['num_samples']})")
    if metrics['embedding_similarity'] > 0:
        print(f"   Embedding Similarity: {metrics['embedding_similarity']:.4f}")
    print("=" * 70)
    
    # Save predictions (JSONL format matching your existing format)
    write_mode = 'a' if resume and os.path.exists(predictions_path) else 'w'
    with open(predictions_path, write_mode, encoding='utf-8') as f:
        for r in results:
            # Format to match your existing output
            output_item = {
                'qid': r['qid'],
                'question': r['question'],
                'answer': r['ground_truth'],
                'output': r['prediction'],
                'correct': r.get('correct', False),
                'category': r['category'],
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    print(f"✓ Predictions saved: {predictions_path}")
    
    # Save metrics summary
    summary = {
        'model': model_path,
        'lora_path': lora_path,
        'dataset': dataset,
        'condition': condition,
        'is_blind': is_blind,
        'metrics': metrics,
    }
    
    summary_path = os.path.join(output_dir, f"{model_name}_{dataset}{condition}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {summary_path}")
    
    return summary


# =============================================================================
# Batch Evaluation (for ablation study)
# =============================================================================

def run_ablation_evaluation(
    model_path: str,
    checkpoint_dirs: List[str],
    dataset: str,
    conditions: List[str],
    output_dir: str = "./eval_results",
    gpu: str = "0",
):
    """
    Run evaluation for multiple checkpoints and conditions.
    
    Useful for ablation study.
    """
    all_results = {}
    
    for checkpoint in checkpoint_dirs:
        checkpoint_name = Path(checkpoint).name
        
        for condition in conditions:
            print(f"\n{'='*70}")
            print(f"Evaluating: {checkpoint_name} | {condition}")
            print(f"{'='*70}")
            
            result = run_evaluation(
                model_path=model_path,
                dataset=dataset,
                condition=condition,
                lora_path=checkpoint,
                output_dir=os.path.join(output_dir, checkpoint_name),
                gpu=gpu,
            )
            
            all_results[f"{checkpoint_name}_{condition}"] = result
    
    # Save combined results
    combined_path = os.path.join(output_dir, "ablation_results.json")
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ All ablation results saved: {combined_path}")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VQA models on blind and real images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate trained model (blind condition)
  python evaluate_models.py \\
      --model OpenGVLab/InternVL3_5-2B \\
      --lora_path ./output/A5/checkpoint-100 \\
      --dataset mmstar \\
      --condition _inst_blind \\
      --output_dir ./eval_results/A5

  # Evaluate base model (zero-shot)
  python evaluate_models.py \\
      --model OpenGVLab/InternVL3_5-2B \\
      --dataset mmstar \\
      --condition "" \\
      --output_dir ./eval_results/A0
      
  # Evaluate on multiple conditions
  python evaluate_models.py \\
      --model OpenGVLab/InternVL3_5-2B \\
      --dataset mmstar \\
      --condition "_inst_blind,_blind," \\
      --output_dir ./eval_results
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="Base model path")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA checkpoint path")
    parser.add_argument("--dataset", type=str, default="mmstar",
                        choices=["mmstar", "spubench", "vqa_1k", "vqa_5k"],
                        help="Dataset name")
    parser.add_argument("--condition", type=str, default="",
                        help="Condition(s), comma-separated: _inst_blind, _blind, _sys_inst_blind, or empty")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Output directory")
    parser.add_argument("--model_type", type=str, default="vlm",
                        choices=["vlm", "llm"],
                        help="Model type")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU ID")
    parser.add_argument("--no_similarity", action="store_true",
                        help="Skip embedding similarity computation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (for debugging)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing predictions")
    
    args = parser.parse_args()
    
    # Handle multiple conditions
    conditions = [c.strip() for c in args.condition.split(',')]
    
    if len(conditions) > 1:
        # Run for multiple conditions
        for condition in conditions:
            run_evaluation(
                model_path=args.model,
                dataset=args.dataset,
                condition=condition,
                lora_path=args.lora_path,
                output_dir=args.output_dir,
                model_type=args.model_type,
                gpu=args.gpu,
                compute_similarity=not args.no_similarity,
                max_samples=args.max_samples,
                resume=args.resume,
            )
    else:
        # Single condition
        run_evaluation(
            model_path=args.model,
            dataset=args.dataset,
            condition=args.condition,
            lora_path=args.lora_path,
            output_dir=args.output_dir,
            model_type=args.model_type,
            gpu=args.gpu,
            compute_similarity=not args.no_similarity,
            max_samples=args.max_samples,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()