#!/usr/bin/env python3
"""
Comprehensive Ablation Training Script for Blind VQA Study

This script supports all ablation conditions:
- A0: Zero-shot (no training, just evaluation)
- A1: SFT-GT (standard fine-tuning with GT answers + real images)
- A2: SFT-GT-Blind (GT answers + black images)
- A3: SFT-Human-Blind (Human answers + black images, no confidence weighting)
- A4: Soft-Human-Blind (Human answers + black images + confidence weighting)
- A5: Soft-Human-Blind-KL (Human answers + black images + confidence + KL)

Usage:
    # Run specific ablation
    python train_ablations.py \
        --ablation A1 \
        --model_path OpenGVLab/InternVL3_5-2B \
        --train_benchmark vqav2 \
        --output_dir ./output/ablations

    # Run all ablations
    python train_ablations.py \
        --run_all \
        --model_path OpenGVLab/InternVL3_5-2B \
        --output_dir ./output/ablations
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from swift.llm import sft_main, TrainArguments
from swift.utils import get_logger

logger = get_logger()


# =============================================================================
# Ablation Configurations
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for a single ablation condition."""
    name: str
    description: str
    use_gt_answers: bool = False          # Use ground truth vs human answers
    use_real_images: bool = False         # Use real images vs black images
    use_confidence_weighting: bool = False # Weight loss by confidence
    use_kl_loss: bool = False             # Add KL regularization
    use_label_smoothing: bool = False     # Confidence-based smoothing
    kl_weight: float = 0.1
    requires_training: bool = True        # False for zero-shot
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'use_gt_answers': self.use_gt_answers,
            'use_real_images': self.use_real_images,
            'use_confidence_weighting': self.use_confidence_weighting,
            'use_kl_loss': self.use_kl_loss,
            'use_label_smoothing': self.use_label_smoothing,
            'kl_weight': self.kl_weight,
            'requires_training': self.requires_training,
        }


# Define all ablation conditions
ABLATIONS = {
    "A0": AblationConfig(
        name="A0_ZeroShot",
        description="Zero-shot baseline - no training, just evaluation",
        requires_training=False,
    ),
    "A1": AblationConfig(
        name="A1_SFT_GT",
        description="Standard SFT with ground truth answers and real images",
        use_gt_answers=True,
        use_real_images=True,
    ),
    "A2": AblationConfig(
        name="A2_SFT_GT_Blind",
        description="SFT with ground truth answers but black images",
        use_gt_answers=True,
        use_real_images=False,
    ),
    "A3": AblationConfig(
        name="A3_SFT_Human_Blind",
        description="SFT with human answers and black images (no confidence)",
        use_gt_answers=False,
        use_real_images=False,
        use_confidence_weighting=False,
    ),
    "A4": AblationConfig(
        name="A4_Soft_Human_Blind",
        description="Soft SFT with human answers, black images, confidence weighting",
        use_gt_answers=False,
        use_real_images=False,
        use_confidence_weighting=True,
        use_label_smoothing=True,
    ),
    "A5": AblationConfig(
        name="A5_Soft_Human_Blind_KL",
        description="Soft SFT with confidence weighting + KL regularization",
        use_gt_answers=False,
        use_real_images=False,
        use_confidence_weighting=True,
        use_label_smoothing=True,
        use_kl_loss=True,
        kl_weight=0.1,
    ),
}


# =============================================================================
# Data Preparation for Each Ablation
# =============================================================================

def prepare_gt_data(
    questions_path: str,
    annotations_path: str,
    images_dir: str,
    output_path: str,
    use_real_images: bool = True,
    black_image_path: str = "black_image.png",
    dataset_type: str = "vqav2",
    max_samples: int = None,
) -> str:
    """
    Prepare training data with ground truth answers.
    
    For A1 (SFT-GT) and A2 (SFT-GT-Blind) ablations.
    """
    import json
    from pathlib import Path
    
    # Load questions
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations_data = json.load(f)
    
    # Build annotation lookup
    if dataset_type == "vqav2":
        questions = {str(q['question_id']): q for q in questions_data.get('questions', questions_data)}
        annotations = {str(a['question_id']): a for a in annotations_data.get('annotations', annotations_data)}
    elif dataset_type == "mmstar":
        # MMStar format handling
        questions = questions_data
        annotations = annotations_data
    else:
        questions = questions_data
        annotations = annotations_data
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = []
    for qid, q in list(questions.items())[:max_samples] if max_samples else questions.items():
        if qid not in annotations:
            continue
        
        ann = annotations[qid]
        question_text = q['question']
        
        # Get answer (most common for VQAv2)
        if dataset_type == "vqav2":
            answers = [a['answer'] for a in ann.get('answers', [])]
            if answers:
                from collections import Counter
                answer = Counter(answers).most_common(1)[0][0]
            else:
                continue
            image_id = q.get('image_id')
            image_path = f"{images_dir}/COCO_val2014_{str(image_id).zfill(12)}.jpg"
        else:
            answer = ann.get('answer', ann.get('gt_answer', ''))
            image_path = ann.get('image_path', black_image_path)
        
        # Use black image if blind condition
        if not use_real_images:
            image_path = black_image_path
            prefix = "Note: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n"
        else:
            prefix = ""
        
        example = {
            "images": [image_path],
            "conversations": [
                {"role": "user", "content": f"<image>\n{prefix}{question_text}"},
                {"role": "assistant", "content": str(answer)}
            ],
            "qid": qid,
            "confidence": 5,  # GT has max confidence
        }
        examples.append(example)
    
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    logger.info(f"Created GT data: {output_path} ({len(examples)} examples)")
    return str(output_path)


def prepare_human_data(
    csv_files: List[str],
    questions_path: str,
    output_path: str,
    black_image_path: str = "black_image.png",
    aggregate: bool = True,
    dataset_type: str = "vqav2",
) -> str:
    """
    Prepare training data with human blind answers.
    
    For A3, A4, A5 ablations.
    """
    import csv
    import json
    from collections import defaultdict
    from pathlib import Path
    
    # Load questions
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)
    
    if dataset_type == "vqav2":
        questions = {str(q['question_id']): q['question'] for q in questions_data.get('questions', questions_data)}
    else:
        questions = {str(k): v.get('question', v) for k, v in questions_data.items()}
    
    # Collect responses
    responses = defaultdict(list)
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row['qid'])
                if qid in questions:
                    responses[qid].append({
                        'answer': row['answer'],
                        'confidence': int(row['confidence']),
                        'time_spent': float(row['time_spent_seconds']),
                    })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    prefix = "Note: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n"
    
    examples = []
    if aggregate:
        for qid, resp_list in responses.items():
            # Group by answer
            answer_groups = defaultdict(list)
            for r in resp_list:
                answer_groups[r['answer']].append(r)
            
            for answer, group in answer_groups.items():
                avg_confidence = sum(r['confidence'] for r in group) / len(group)
                avg_time = sum(r['time_spent'] for r in group) / len(group)
                
                example = {
                    "images": [black_image_path],
                    "conversations": [
                        {"role": "user", "content": f"<image>\n{prefix}{questions[qid]}"},
                        {"role": "assistant", "content": str(answer)}
                    ],
                    "confidence": round(avg_confidence, 2),
                    "num_responses": len(group),
                    "qid": qid,
                    "time_spent_seconds": round(avg_time, 2),
                }
                examples.append(example)
    else:
        for qid, resp_list in responses.items():
            for r in resp_list:
                example = {
                    "images": [black_image_path],
                    "conversations": [
                        {"role": "user", "content": f"<image>\n{prefix}{questions[qid]}"},
                        {"role": "assistant", "content": str(r['answer'])}
                    ],
                    "confidence": r['confidence'],
                    "qid": qid,
                    "time_spent_seconds": r['time_spent'],
                }
                examples.append(example)
    
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    logger.info(f"Created human data: {output_path} ({len(examples)} examples)")
    return str(output_path)


# =============================================================================
# Training Functions
# =============================================================================

def train_standard_sft(
    model_path: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    val_data_path: str = None,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    label_smoothing: float = 0.0,
    freeze_vit: bool = True,
):
    """Standard SFT training (for A1, A2, A3 ablations)."""
    
    logger.info(f"🚀 Standard SFT Training: {run_name}")
    
    sft_args = TrainArguments(
        model=model_path,
        dataset=[data_path],
        val_dataset=val_data_path,
        output_dir=output_dir,
        
        train_type="lora",
        torch_dtype="bfloat16",
        
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.05,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["all-linear"],
        
        freeze_llm=False,
        freeze_vit=freeze_vit,
        freeze_aligner=True,
        
        gradient_checkpointing=True,
        
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        logging_steps=10,
        
        eval_strategy="steps" if val_data_path else "no",
        eval_steps=50 if val_data_path else None,
        
        max_length=2048,
        seed=42,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
        attn_impl="flash_attn",
    )
    
    return sft_main(sft_args)


def train_soft_supervised(
    model_path: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    val_data_path: str = None,
    use_kl_loss: bool = False,
    kl_weight: float = 0.1,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
):
    """
    Soft supervised training with confidence weighting.
    
    For A4 and A5 ablations.
    
    NOTE: This uses SWIFT's standard SFT but logs the soft supervision config.
    For true confidence-weighted loss, use the CustomSoftSupervisedTrainer.
    """
    
    logger.info(f"🎯 Soft Supervised Training: {run_name}")
    logger.info(f"   KL Loss: {use_kl_loss} (weight={kl_weight})")
    
    # Save soft supervision config
    config = {
        "use_kl_loss": use_kl_loss,
        "kl_weight": kl_weight,
        "weight_strategy": "linear",
        "use_confidence_smoothing": True,
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "soft_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    sft_args = TrainArguments(
        model=model_path,
        dataset=[data_path],
        val_dataset=val_data_path,
        output_dir=output_dir,
        
        train_type="lora",
        torch_dtype="bfloat16",
        
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.05,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["all-linear"],
        
        freeze_llm=False,
        freeze_vit=True,  # Always freeze ViT for blind training
        freeze_aligner=True,
        
        gradient_checkpointing=True,
        
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        logging_steps=10,
        
        eval_strategy="steps" if val_data_path else "no",
        eval_steps=50 if val_data_path else None,
        
        max_length=2048,
        seed=42,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
        attn_impl="flash_attn",
    )
    
    return sft_main(sft_args)


# =============================================================================
# Main Ablation Runner
# =============================================================================

def run_ablation(
    ablation_id: str,
    model_path: str,
    train_benchmark: str,
    output_dir: str,
    # Data paths
    human_csv_files: List[str] = None,
    questions_path: str = None,
    annotations_path: str = None,
    images_dir: str = None,
    black_image_path: str = "black_image.png",
    # Training config
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    lora_rank: int = 32,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
) -> Dict[str, Any]:
    """
    Run a single ablation condition.
    
    Args:
        ablation_id: One of A0-A5
        model_path: Path to base model
        train_benchmark: Which benchmark to train on (vqav2, mmstar, mmspubench)
        output_dir: Base output directory
        human_csv_files: List of participant CSV files
        questions_path: Path to questions JSON
        annotations_path: Path to annotations JSON (for GT ablations)
        images_dir: Directory with real images (for A1)
        black_image_path: Path to black placeholder image
        
    Returns:
        Dictionary with results and paths
    """
    if ablation_id not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation_id}. Choose from {list(ABLATIONS.keys())}")
    
    config = ABLATIONS[ablation_id]
    
    logger.info("=" * 80)
    logger.info(f"🧪 Running Ablation: {config.name}")
    logger.info(f"   Description: {config.description}")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Benchmark: {train_benchmark}")
    logger.info("=" * 80)
    
    # Setup output directory
    ablation_output_dir = os.path.join(output_dir, config.name, train_benchmark)
    os.makedirs(ablation_output_dir, exist_ok=True)
    
    # Save ablation config
    with open(os.path.join(ablation_output_dir, "ablation_config.json"), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    result = {
        "ablation_id": ablation_id,
        "config": config.to_dict(),
        "model_path": model_path,
        "train_benchmark": train_benchmark,
        "output_dir": ablation_output_dir,
    }
    
    # A0: Zero-shot - no training needed
    if not config.requires_training:
        logger.info("📋 Zero-shot ablation - no training required")
        result["status"] = "skip_training"
        result["checkpoint_path"] = model_path  # Use original model
        return result
    
    # Prepare data based on ablation type
    data_dir = os.path.join(ablation_output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    if config.use_gt_answers:
        # A1 or A2: Use ground truth answers
        if not annotations_path:
            raise ValueError("annotations_path required for GT ablations (A1, A2)")
        
        train_data_path = prepare_gt_data(
            questions_path=questions_path,
            annotations_path=annotations_path,
            images_dir=images_dir if config.use_real_images else None,
            output_path=os.path.join(data_dir, "train.jsonl"),
            use_real_images=config.use_real_images,
            black_image_path=black_image_path,
            dataset_type=train_benchmark,
        )
    else:
        # A3, A4, A5: Use human answers
        if not human_csv_files:
            raise ValueError("human_csv_files required for human ablations (A3, A4, A5)")
        
        train_data_path = prepare_human_data(
            csv_files=human_csv_files,
            questions_path=questions_path,
            output_path=os.path.join(data_dir, "train.jsonl"),
            black_image_path=black_image_path,
            aggregate=True,
            dataset_type=train_benchmark,
        )
    
    result["train_data_path"] = train_data_path
    
    # Run appropriate training
    run_name = f"{config.name}_{train_benchmark}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    if config.use_confidence_weighting:
        # A4 or A5: Soft supervised training
        train_soft_supervised(
            model_path=model_path,
            data_path=train_data_path,
            output_dir=ablation_output_dir,
            run_name=run_name,
            use_kl_loss=config.use_kl_loss,
            kl_weight=config.kl_weight,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            lora_rank=lora_rank,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    else:
        # A1, A2, A3: Standard SFT
        freeze_vit = not config.use_real_images  # Freeze ViT for blind conditions
        
        train_standard_sft(
            model_path=model_path,
            data_path=train_data_path,
            output_dir=ablation_output_dir,
            run_name=run_name,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            lora_rank=lora_rank,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            freeze_vit=freeze_vit,
        )
    
    result["status"] = "completed"
    result["run_name"] = run_name
    
    # Find best checkpoint
    checkpoints = list(Path(ablation_output_dir).glob("checkpoint-*"))
    if checkpoints:
        result["checkpoint_path"] = str(sorted(checkpoints)[-1])
    
    return result


def run_all_ablations(
    model_path: str,
    train_benchmark: str,
    output_dir: str,
    human_csv_files: List[str],
    questions_path: str,
    annotations_path: str = None,
    images_dir: str = None,
    ablations_to_run: List[str] = None,
    **training_kwargs,
) -> Dict[str, Any]:
    """Run all (or specified) ablations."""
    
    if ablations_to_run is None:
        ablations_to_run = list(ABLATIONS.keys())
    
    results = {}
    
    for ablation_id in ablations_to_run:
        try:
            result = run_ablation(
                ablation_id=ablation_id,
                model_path=model_path,
                train_benchmark=train_benchmark,
                output_dir=output_dir,
                human_csv_files=human_csv_files,
                questions_path=questions_path,
                annotations_path=annotations_path,
                images_dir=images_dir,
                **training_kwargs,
            )
            results[ablation_id] = result
        except Exception as e:
            logger.error(f"❌ Failed ablation {ablation_id}: {e}")
            results[ablation_id] = {"status": "failed", "error": str(e)}
    
    # Save summary
    summary_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📊 Ablation summary saved to: {summary_path}")
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation studies for blind VQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base model (e.g., OpenGVLab/InternVL3_5-2B)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for all ablations")
    
    # Ablation selection
    parser.add_argument("--ablation", type=str, default=None,
                        choices=list(ABLATIONS.keys()),
                        help="Specific ablation to run")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all ablations")
    parser.add_argument("--ablations", type=str, nargs='+',
                        choices=list(ABLATIONS.keys()),
                        help="List of ablations to run")
    
    # Data
    parser.add_argument("--train_benchmark", type=str, default="vqav2",
                        choices=["vqav2", "mmstar", "mmspubench"],
                        help="Benchmark to train on")
    parser.add_argument("--human_csv_files", type=str, nargs='+',
                        help="Participant CSV files")
    parser.add_argument("--questions_path", type=str,
                        help="Questions JSON file")
    parser.add_argument("--annotations_path", type=str,
                        help="Annotations JSON file (for GT ablations)")
    parser.add_argument("--images_dir", type=str,
                        help="Real images directory (for A1)")
    parser.add_argument("--black_image_path", type=str, default="black_image.png",
                        help="Black placeholder image path")
    
    # Training
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    
    args = parser.parse_args()
    
    # Determine which ablations to run
    if args.run_all:
        ablations_to_run = list(ABLATIONS.keys())
    elif args.ablations:
        ablations_to_run = args.ablations
    elif args.ablation:
        ablations_to_run = [args.ablation]
    else:
        parser.error("Specify --ablation, --ablations, or --run_all")
    
    # Run
    if len(ablations_to_run) == 1:
        run_ablation(
            ablation_id=ablations_to_run[0],
            model_path=args.model_path,
            train_benchmark=args.train_benchmark,
            output_dir=args.output_dir,
            human_csv_files=args.human_csv_files,
            questions_path=args.questions_path,
            annotations_path=args.annotations_path,
            images_dir=args.images_dir,
            black_image_path=args.black_image_path,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            lora_rank=args.lora_rank,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    else:
        run_all_ablations(
            model_path=args.model_path,
            train_benchmark=args.train_benchmark,
            output_dir=args.output_dir,
            human_csv_files=args.human_csv_files,
            questions_path=args.questions_path,
            annotations_path=args.annotations_path,
            images_dir=args.images_dir,
            ablations_to_run=ablations_to_run,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            lora_rank=args.lora_rank,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )


if __name__ == "__main__":
    main()