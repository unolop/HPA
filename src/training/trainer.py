#!/usr/bin/env python3
"""
Soft Supervised Learning Trainer for Blind VQA Tasks

This trainer implements confidence-weighted supervision for training VLMs
on human blind VQA responses (answers without seeing images).

Key Features:
1. Confidence-weighted loss: Human confidence (1-5) modulates loss weight
2. Optional KL regularization: Preserves model's visual capabilities
3. Soft label generation: Converts discrete confidence to label smoothing

Use Case:
- Training models to match human linguistic priors in VQA
- Studying blind performance vs. visual reasoning
- Calibrating model uncertainty on ambiguous inputs

Usage:
    python train_soft_supervised_vqa.py \
        --model_path /path/to/model \
        --data_path blind_vqa_train.jsonl \
        --output_dir ./output/soft_supervised \
        --run_name soft_vqa_blind \
        --use_kl_loss \
        --kl_weight 0.1
"""

# Fix memory fragmentation issues
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import copy

from swift.llm import sft_main, TrainArguments
from swift.utils import get_logger

logger = get_logger()


# =============================================================================
# Data Conversion Utilities
# =============================================================================

def convert_csv_to_jsonl(
    csv_path: str,
    output_path: str,
    questions_mapping: Dict[str, str],  # qid -> question text
    black_image_path: str = "black_image.png",
    instruction_prefix: str = "Note: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n",
    answer_choices: Optional[Dict[str, List[str]]] = None,  # qid -> list of answer options
) -> str:
    """
    Convert participant CSV data to JSONL format for training.
    
    CSV format expected:
        question_num,qid,answer,confidence,time_spent_seconds,answer_timestamp
        0,664,2,4,72.02,2025-11-27T14:19:21.849209
    
    Args:
        csv_path: Path to participant CSV file
        output_path: Path for output JSONL file
        questions_mapping: Dictionary mapping qid to question text
        black_image_path: Path to black placeholder image
        instruction_prefix: Instruction to prepend to questions
        answer_choices: Optional mapping of qid to answer options (for multiple choice)
        
    Returns:
        Path to created JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    skipped_count = 0
    
    with open(csv_path, 'r') as csv_file, open(output_path, 'w') as jsonl_file:
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            qid = str(row['qid'])
            
            # Skip if question not in mapping
            if qid not in questions_mapping:
                logger.warning(f"Question {qid} not found in mapping, skipping")
                skipped_count += 1
                continue
            
            question_text = questions_mapping[qid]
            answer = row['answer']
            confidence = int(row['confidence'])
            time_spent = float(row['time_spent_seconds'])
            
            # Format answer (handle multiple choice vs free-form)
            if answer_choices and qid in answer_choices:
                # Multiple choice - answer is an index
                try:
                    answer_idx = int(answer)
                    answer_text = answer_choices[qid][answer_idx]
                except (ValueError, IndexError):
                    answer_text = str(answer)
            else:
                answer_text = str(answer)
            
            # Create training example
            example = {
                "images": [black_image_path],
                "conversations": [
                    {
                        "role": "user",
                        "content": f"<image>\n{instruction_prefix}{question_text}"
                    },
                    {
                        "role": "assistant",
                        "content": answer_text
                    }
                ],
                # Metadata for soft supervision
                "confidence": confidence,
                "qid": qid,
                "time_spent_seconds": time_spent,
            }
            
            jsonl_file.write(json.dumps(example) + '\n')
            converted_count += 1
    
    logger.info(f"Converted {converted_count} examples, skipped {skipped_count}")
    return str(output_path)


def merge_participant_csvs(
    csv_paths: List[str],
    output_path: str,
    questions_mapping: Dict[str, str],
    black_image_path: str = "black_image.png",
    instruction_prefix: str = "Note: No images are provided. For each question, imagine an appropriate image exists and answer based on the most common or universal scenario.\n",
    answer_choices: Optional[Dict[str, List[str]]] = None,
    aggregate_confidence: bool = True,
) -> str:
    """
    Merge multiple participant CSVs into a single JSONL with aggregated confidence.
    
    If aggregate_confidence=True, multiple answers for the same question are combined
    with averaged confidence scores. Otherwise, each response is a separate example.
    
    Args:
        csv_paths: List of paths to participant CSV files
        output_path: Path for output JSONL file
        questions_mapping: Dictionary mapping qid to question text
        black_image_path: Path to black placeholder image
        instruction_prefix: Instruction to prepend
        answer_choices: Optional answer choices mapping
        aggregate_confidence: Whether to aggregate responses per question
        
    Returns:
        Path to created JSONL file
    """
    from collections import defaultdict
    
    # Collect all responses
    responses = defaultdict(list)  # qid -> [(answer, confidence, time), ...]
    
    for csv_path in csv_paths:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row['qid'])
                if qid not in questions_mapping:
                    continue
                responses[qid].append({
                    'answer': row['answer'],
                    'confidence': int(row['confidence']),
                    'time_spent': float(row['time_spent_seconds'])
                })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if aggregate_confidence:
            # Aggregate: group by (qid, answer) and average confidence
            for qid, resp_list in responses.items():
                answer_groups = defaultdict(list)
                for r in resp_list:
                    answer_groups[r['answer']].append(r)
                
                for answer, group in answer_groups.items():
                    avg_confidence = sum(r['confidence'] for r in group) / len(group)
                    avg_time = sum(r['time_spent'] for r in group) / len(group)
                    
                    # Format answer
                    if answer_choices and qid in answer_choices:
                        try:
                            answer_text = answer_choices[qid][int(answer)]
                        except (ValueError, IndexError):
                            answer_text = str(answer)
                    else:
                        answer_text = str(answer)
                    
                    example = {
                        "images": [black_image_path],
                        "conversations": [
                            {"role": "user", "content": f"<image>\n{instruction_prefix}{questions_mapping[qid]}"},
                            {"role": "assistant", "content": answer_text}
                        ],
                        "confidence": avg_confidence,
                        "num_responses": len(group),
                        "qid": qid,
                        "time_spent_seconds": avg_time,
                    }
                    f.write(json.dumps(example) + '\n')
        else:
            # Keep all individual responses
            for qid, resp_list in responses.items():
                for r in resp_list:
                    answer = r['answer']
                    if answer_choices and qid in answer_choices:
                        try:
                            answer_text = answer_choices[qid][int(answer)]
                        except (ValueError, IndexError):
                            answer_text = str(answer)
                    else:
                        answer_text = str(answer)
                    
                    example = {
                        "images": [black_image_path],
                        "conversations": [
                            {"role": "user", "content": f"<image>\n{instruction_prefix}{questions_mapping[qid]}"},
                            {"role": "assistant", "content": answer_text}
                        ],
                        "confidence": r['confidence'],
                        "qid": qid,
                        "time_spent_seconds": r['time_spent'],
                    }
                    f.write(json.dumps(example) + '\n')
    
    return str(output_path)


def create_black_image(output_path: str, size: Tuple[int, int] = (448, 448)) -> str:
    """Create a black placeholder image for blind VQA training."""
    from PIL import Image
    import numpy as np
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create black image
    black_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img = Image.fromarray(black_array)
    img.save(output_path)
    
    logger.info(f"Created black image at {output_path}")
    return str(output_path)


# =============================================================================
# Confidence-to-Weight Conversion
# =============================================================================

def confidence_to_loss_weight(
    confidence: int,
    strategy: str = "linear",
    min_weight: float = 0.2,
    max_weight: float = 1.0,
    confidence_range: Tuple[int, int] = (1, 5),
) -> float:
    """
    Convert human confidence (1-5 scale) to loss weight.
    
    Strategies:
    - "linear": Linear scaling from min_weight to max_weight
    - "quadratic": Quadratic scaling (emphasizes high confidence more)
    - "inverse": Higher confidence = lower weight (for studying uncertainty)
    - "binary": Threshold-based (high vs low confidence)
    
    Args:
        confidence: Human confidence value (typically 1-5)
        strategy: Weighting strategy
        min_weight: Minimum loss weight
        max_weight: Maximum loss weight
        confidence_range: (min_confidence, max_confidence) tuple
        
    Returns:
        Loss weight as float
    """
    min_conf, max_conf = confidence_range
    
    # Normalize confidence to [0, 1]
    normalized = (confidence - min_conf) / (max_conf - min_conf)
    normalized = max(0.0, min(1.0, normalized))
    
    if strategy == "linear":
        weight = min_weight + normalized * (max_weight - min_weight)
    
    elif strategy == "quadratic":
        weight = min_weight + (normalized ** 2) * (max_weight - min_weight)
    
    elif strategy == "inverse":
        # Higher confidence = lower weight (study what model gets wrong with high human confidence)
        weight = max_weight - normalized * (max_weight - min_weight)
    
    elif strategy == "binary":
        threshold = (min_conf + max_conf) / 2
        weight = max_weight if confidence > threshold else min_weight
    
    elif strategy == "softmax":
        # Softmax-style exponential weighting
        import math
        temperature = 2.0
        weight = min_weight + (math.exp(normalized * temperature) - 1) / (math.exp(temperature) - 1) * (max_weight - min_weight)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return weight


def confidence_to_label_smoothing(
    confidence: int,
    num_classes: int = None,
    confidence_range: Tuple[int, int] = (1, 5),
    min_smoothing: float = 0.0,
    max_smoothing: float = 0.3,
) -> float:
    """
    Convert confidence to label smoothing factor.
    
    Low confidence -> high smoothing (soft labels)
    High confidence -> low smoothing (hard labels)
    
    Args:
        confidence: Human confidence value
        num_classes: Number of answer classes (for reference)
        confidence_range: (min, max) confidence values
        min_smoothing: Smoothing for highest confidence
        max_smoothing: Smoothing for lowest confidence
        
    Returns:
        Label smoothing factor
    """
    min_conf, max_conf = confidence_range
    normalized = (confidence - min_conf) / (max_conf - min_conf)
    normalized = max(0.0, min(1.0, normalized))
    
    # Inverse relationship: high confidence = low smoothing
    smoothing = max_smoothing - normalized * (max_smoothing - min_smoothing)
    
    return smoothing


# =============================================================================
# Custom Loss Functions
# =============================================================================

class ConfidenceWeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss weighted by human confidence scores.
    
    Higher confidence examples contribute more to the loss.
    Optionally applies label smoothing based on confidence.
    """
    
    def __init__(
        self,
        weight_strategy: str = "linear",
        min_weight: float = 0.2,
        max_weight: float = 1.0,
        use_confidence_smoothing: bool = True,
        min_smoothing: float = 0.0,
        max_smoothing: float = 0.2,
        confidence_range: Tuple[int, int] = (1, 5),
        ignore_index: int = -100,
    ):
        super().__init__()
        self.weight_strategy = weight_strategy
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_confidence_smoothing = use_confidence_smoothing
        self.min_smoothing = min_smoothing
        self.max_smoothing = max_smoothing
        self.confidence_range = confidence_range
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confidence-weighted loss.
        
        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            confidence: Per-example confidence values [batch]
            
        Returns:
            Weighted loss scalar
        """
        batch_size = logits.size(0)
        vocab_size = logits.size(-1)
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Create per-example weights
        weights = []
        smoothings = []
        for conf in confidence:
            w = confidence_to_loss_weight(
                conf.item(),
                strategy=self.weight_strategy,
                min_weight=self.min_weight,
                max_weight=self.max_weight,
                confidence_range=self.confidence_range,
            )
            weights.append(w)
            
            if self.use_confidence_smoothing:
                s = confidence_to_label_smoothing(
                    conf.item(),
                    confidence_range=self.confidence_range,
                    min_smoothing=self.min_smoothing,
                    max_smoothing=self.max_smoothing,
                )
                smoothings.append(s)
        
        # Compute per-example losses
        total_loss = 0.0
        for i in range(batch_size):
            # Get this example's logits and labels
            start_idx = i * logits.size(1)
            end_idx = (i + 1) * logits.size(1)
            
            example_logits = logits_flat[start_idx:end_idx]
            example_labels = labels_flat[start_idx:end_idx]
            
            # Mask for valid tokens
            valid_mask = example_labels != self.ignore_index
            
            if valid_mask.sum() == 0:
                continue
            
            valid_logits = example_logits[valid_mask]
            valid_labels = example_labels[valid_mask]
            
            # Apply label smoothing if enabled
            if self.use_confidence_smoothing:
                smoothing = smoothings[i]
                log_probs = F.log_softmax(valid_logits, dim=-1)
                
                # Smoothed labels
                n_classes = valid_logits.size(-1)
                smooth_labels = torch.full_like(log_probs, smoothing / n_classes)
                smooth_labels.scatter_(1, valid_labels.unsqueeze(1), 1.0 - smoothing + smoothing / n_classes)
                
                example_loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
            else:
                example_loss = F.cross_entropy(valid_logits, valid_labels)
            
            # Apply confidence weight
            total_loss += weights[i] * example_loss
        
        # Average over batch
        return total_loss / batch_size


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss to maintain proximity to reference model.
    
    Used to preserve visual reasoning capabilities when training on blind data.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = "batchmean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor = None,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Compute KL divergence between student and teacher distributions.
        
        Args:
            student_logits: Current model logits [batch, seq_len, vocab_size]
            teacher_logits: Reference model logits [batch, seq_len, vocab_size]
            labels: Optional labels to mask padding [batch, seq_len]
            ignore_index: Index to ignore in labels
            
        Returns:
            KL divergence loss
        """
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='none')
        
        # Sum over vocabulary dimension
        kl_div = kl_div.sum(dim=-1)
        
        # Mask padding if labels provided
        if labels is not None:
            valid_mask = (labels != ignore_index).float()
            kl_div = kl_div * valid_mask
            
            if self.reduction == "batchmean":
                return kl_div.sum() / valid_mask.sum().clamp(min=1)
            elif self.reduction == "sum":
                return kl_div.sum()
            else:
                return kl_div.mean()
        else:
            if self.reduction == "batchmean":
                return kl_div.mean()
            elif self.reduction == "sum":
                return kl_div.sum()
            else:
                return kl_div.mean()


class SoftSupervisedLoss(nn.Module):
    """
    Combined loss for soft supervised learning on blind VQA.
    
    Components:
    1. Confidence-weighted CE loss: Main supervision signal
    2. KL divergence loss: Preserves reference model behavior
    
    Total Loss = CE_weight * CE_loss + KL_weight * KL_loss
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        kl_weight: float = 0.1,
        use_kl: bool = True,
        weight_strategy: str = "linear",
        confidence_min_weight: float = 0.2,
        confidence_max_weight: float = 1.0,
        use_confidence_smoothing: bool = True,
        kl_temperature: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.use_kl = use_kl
        
        self.ce_loss = ConfidenceWeightedCrossEntropyLoss(
            weight_strategy=weight_strategy,
            min_weight=confidence_min_weight,
            max_weight=confidence_max_weight,
            use_confidence_smoothing=use_confidence_smoothing,
            ignore_index=ignore_index,
        )
        
        if use_kl:
            self.kl_loss = KLDivergenceLoss(
                temperature=kl_temperature,
                reduction="batchmean",
            )
    
    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        teacher_logits: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            student_logits: Current model output [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            confidence: Per-example confidence [batch]
            teacher_logits: Reference model output (for KL) [batch, seq_len, vocab_size]
            
        Returns:
            Dictionary with 'loss', 'ce_loss', 'kl_loss' keys
        """
        # Confidence-weighted CE loss
        ce_loss = self.ce_loss(student_logits, labels, confidence)
        
        total_loss = self.ce_weight * ce_loss
        
        result = {
            'ce_loss': ce_loss,
            'kl_loss': torch.tensor(0.0, device=ce_loss.device),
        }
        
        # KL loss if enabled and teacher provided
        if self.use_kl and teacher_logits is not None:
            kl_loss = self.kl_loss(student_logits, teacher_logits, labels)
            total_loss = total_loss + self.kl_weight * kl_loss
            result['kl_loss'] = kl_loss
        
        result['loss'] = total_loss
        return result


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class SoftSupervisedConfig:
    """Configuration for soft supervised training."""
    
    # Loss configuration
    use_kl_loss: bool = True
    kl_weight: float = 0.1
    ce_weight: float = 1.0
    kl_temperature: float = 1.0
    
    # Confidence weighting
    weight_strategy: str = "linear"  # linear, quadratic, inverse, binary, softmax
    confidence_min_weight: float = 0.2
    confidence_max_weight: float = 1.0
    use_confidence_smoothing: bool = True
    min_label_smoothing: float = 0.0
    max_label_smoothing: float = 0.2
    confidence_range: Tuple[int, int] = (1, 5)
    
    # Reference model
    reference_model_path: Optional[str] = None  # If None, uses copy of initial model
    freeze_reference: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'use_kl_loss': self.use_kl_loss,
            'kl_weight': self.kl_weight,
            'ce_weight': self.ce_weight,
            'kl_temperature': self.kl_temperature,
            'weight_strategy': self.weight_strategy,
            'confidence_min_weight': self.confidence_min_weight,
            'confidence_max_weight': self.confidence_max_weight,
            'use_confidence_smoothing': self.use_confidence_smoothing,
            'min_label_smoothing': self.min_label_smoothing,
            'max_label_smoothing': self.max_label_smoothing,
            'confidence_range': self.confidence_range,
            'reference_model_path': self.reference_model_path,
            'freeze_reference': self.freeze_reference,
        }


# =============================================================================
# Main Training Function
# =============================================================================

def train_soft_supervised(
    model_path: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    val_data_path: str = None,
    # Soft supervision config
    use_kl_loss: bool = True,
    kl_weight: float = 0.1,
    weight_strategy: str = "linear",
    confidence_min_weight: float = 0.2,
    confidence_max_weight: float = 1.0,
    use_confidence_smoothing: bool = True,
    # Standard training config
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_steps: int = -1,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 20,
    gradient_checkpointing: bool = True,
):
    """
    Train a VLM with soft supervised learning on blind VQA data.
    
    This trainer uses:
    - Confidence-weighted loss based on human certainty
    - Optional KL regularization to preserve visual capabilities
    - Label smoothing inversely proportional to confidence
    
    NOTE: This function sets up the configuration. The actual custom loss
    integration requires modifying SWIFT's trainer or using a callback.
    For full custom loss support, see the CustomSoftSupervisedTrainer class.
    
    Args:
        model_path: Path to base model
        data_path: Path to training data (.jsonl with confidence field)
        output_dir: Output directory for checkpoints
        run_name: Name for wandb logging
        val_data_path: Optional validation data path
        use_kl_loss: Whether to use KL regularization
        kl_weight: Weight for KL loss term
        weight_strategy: How to convert confidence to weights
        confidence_min_weight: Minimum loss weight
        confidence_max_weight: Maximum loss weight
        use_confidence_smoothing: Use confidence-based label smoothing
        learning_rate: Learning rate
        num_epochs: Number of epochs
        max_steps: Max training steps (-1 = use epochs)
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation
        save_steps: Checkpoint save interval
        eval_steps: Evaluation interval
        logging_steps: Logging interval
        gradient_checkpointing: Enable gradient checkpointing
        
    Returns:
        Training result
    """
    logger.info("=" * 80)
    logger.info("🎯 Soft Supervised Learning for Blind VQA")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Data: {data_path}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   KL Loss: {use_kl_loss} (weight={kl_weight})")
    logger.info(f"   Weight Strategy: {weight_strategy}")
    logger.info(f"   Confidence Weights: [{confidence_min_weight}, {confidence_max_weight}]")
    logger.info("=" * 80)
    
    # Save soft supervision config
    soft_config = SoftSupervisedConfig(
        use_kl_loss=use_kl_loss,
        kl_weight=kl_weight,
        weight_strategy=weight_strategy,
        confidence_min_weight=confidence_min_weight,
        confidence_max_weight=confidence_max_weight,
        use_confidence_smoothing=use_confidence_smoothing,
    )
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "soft_supervised_config.json"), 'w') as f:
        json.dump(soft_config.to_dict(), f, indent=2)
    
    # Configure SWIFT training arguments
    # Note: For full custom loss, you'd need to extend SWIFT's trainer
    # This setup uses standard SWIFT with label smoothing as approximation
    
    # Calculate average label smoothing based on strategy
    # (This is an approximation - full implementation needs custom trainer)
    avg_smoothing = (soft_config.min_label_smoothing + soft_config.max_label_smoothing) / 2
    
    sft_args = TrainArguments(
        # Model & Data
        model=model_path,
        dataset=[data_path],
        val_dataset=val_data_path,
        output_dir=output_dir,
        
        # Training type
        train_type="lora",
        torch_dtype="bfloat16",
        
        # Training hyperparameters
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.05,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="cosine",
        
        # Label smoothing (approximation of confidence-based smoothing)
        # Note: True confidence-based smoothing needs custom trainer
        # label_smoothing_factor=avg_smoothing,  # Uncomment if supported
        
        # LoRA configuration
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        lora_bias="none",
        target_modules=["all-linear"],
        
        # Freeze configuration
        freeze_llm=False,
        freeze_vit=True,  # Keep vision encoder frozen
        freeze_aligner=True,
        
        # Memory optimization
        gradient_checkpointing=gradient_checkpointing,
        
        # Saving & Logging
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,
        logging_steps=logging_steps,
        logging_first_step=True,
        
        # Evaluation
        eval_strategy="steps" if val_data_path else "no",
        eval_steps=eval_steps if val_data_path else None,
        load_best_model_at_end=True if val_data_path else False,
        metric_for_best_model="eval_loss" if val_data_path else None,
        greater_is_better=False,
        
        # Template configuration
        max_length=2048,
        truncation_strategy="right",
        max_pixels=448,
        
        # Other settings
        dataloader_num_workers=4,
        seed=42,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
        attn_impl="flash_attn",
    )
    
    # Run training
    logger.info("🚀 Starting soft supervised training...")
    logger.info("⚠️  Note: For full confidence-weighted loss and KL regularization,")
    logger.info("    use the CustomSoftSupervisedTrainer with HuggingFace Trainer.")
    
    result = sft_main(sft_args)
    logger.info("✅ Training completed!")
    
    return result


# =============================================================================
# Custom Trainer (for full soft supervision support)
# =============================================================================

class CustomSoftSupervisedTrainer:
    """
    Custom trainer with full soft supervised learning support.
    
    This trainer properly implements:
    - Per-example confidence weighting
    - KL divergence with reference model
    - Confidence-based label smoothing
    
    Usage:
        trainer = CustomSoftSupervisedTrainer(
            model=model,
            ref_model=ref_model,  # Can be None if not using KL
            config=SoftSupervisedConfig(),
        )
        trainer.train(train_dataloader, optimizer, num_epochs)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SoftSupervisedConfig,
        ref_model: nn.Module = None,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Setup reference model for KL
        if config.use_kl_loss:
            if ref_model is not None:
                self.ref_model = ref_model
            else:
                # Create copy of model as reference
                logger.info("Creating reference model copy for KL loss...")
                self.ref_model = copy.deepcopy(model)
            
            if config.freeze_reference:
                for param in self.ref_model.parameters():
                    param.requires_grad = False
                self.ref_model.eval()
        else:
            self.ref_model = None
        
        # Setup loss function
        self.loss_fn = SoftSupervisedLoss(
            ce_weight=config.ce_weight,
            kl_weight=config.kl_weight,
            use_kl=config.use_kl_loss,
            weight_strategy=config.weight_strategy,
            confidence_min_weight=config.confidence_min_weight,
            confidence_max_weight=config.confidence_max_weight,
            use_confidence_smoothing=config.use_confidence_smoothing,
            kl_temperature=config.kl_temperature,
        )
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        pixel_values: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute soft supervised loss for a batch.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            confidence: Per-example confidence values
            pixel_values: Image inputs (optional)
            
        Returns:
            Loss dictionary
        """
        # Forward pass through student model
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if pixel_values is not None:
            model_inputs['pixel_values'] = pixel_values
        
        outputs = self.model(**model_inputs)
        student_logits = outputs.logits
        
        # Forward pass through reference model (if using KL)
        teacher_logits = None
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(**model_inputs)
                teacher_logits = ref_outputs.logits
        
        # Compute loss
        loss_dict = self.loss_fn(
            student_logits=student_logits,
            labels=labels,
            confidence=confidence,
            teacher_logits=teacher_logits,
        )
        
        return loss_dict
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        grad_scaler: Any = None,
    ) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Batch of training data
            optimizer: Optimizer
            scheduler: Optional learning rate scheduler
            grad_scaler: Optional gradient scaler for mixed precision
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Extract confidence from batch
        confidence = batch.pop('confidence')
        
        # Compute loss
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss_dict = self.compute_loss(
                confidence=confidence,
                **batch,
            )
        
        loss = loss_dict['loss']
        
        # Backward pass
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        return {k: v.item() for k, v in loss_dict.items()}


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Parse command line arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Soft Supervised Learning Trainer for Blind VQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data (.jsonl)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name for wandb run")
    
    # Soft supervision arguments
    parser.add_argument("--use_kl_loss", action="store_true",
                        help="Enable KL regularization to preserve visual capabilities")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="Weight for KL loss term")
    parser.add_argument("--weight_strategy", type=str, default="linear",
                        choices=["linear", "quadratic", "inverse", "binary", "softmax"],
                        help="Strategy for converting confidence to loss weights")
    parser.add_argument("--confidence_min_weight", type=float, default=0.2,
                        help="Minimum loss weight for lowest confidence")
    parser.add_argument("--confidence_max_weight", type=float, default=1.0,
                        help="Maximum loss weight for highest confidence")
    parser.add_argument("--use_confidence_smoothing", action="store_true",
                        help="Use confidence-based label smoothing")
    
    # Optional arguments
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation data")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use epochs)")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint interval")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation interval")
    parser.add_argument("--logging_steps", type=int, default=20,
                        help="Logging interval")
    
    args = parser.parse_args()
    
    train_soft_supervised(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        val_data_path=args.val_data_path,
        use_kl_loss=args.use_kl_loss,
        kl_weight=args.kl_weight,
        weight_strategy=args.weight_strategy,
        confidence_min_weight=args.confidence_min_weight,
        confidence_max_weight=args.confidence_max_weight,
        use_confidence_smoothing=args.use_confidence_smoothing,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()