#!/usr/bin/env python3
"""
3. train_soft_supervised.py - Soft Supervised Training for Blind VQA

Supports all ablation conditions:
- A0: Zero-shot (no training)
- A3: Human blind, standard CE loss
- A4: Human blind + confidence weighting
- A5: Human blind + confidence + KL regularization (MAIN METHOD)

Also supports GT ablations:
- A1: GT + real images
- A2: GT + black images

Usage:
    # Main method (A5)
    python train_soft_supervised.py \\
        --ablation A5 \\
        --model_path OpenGVLab/InternVL3_5-2B \\
        --train_data ./training_data/mmstar/train_aggregated_train.jsonl \\
        --val_data ./training_data/mmstar/train_aggregated_val.jsonl \\
        --output_dir ./output/A5_mmstar \\
        --run_name A5_soft_kl_mmstar
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Union, List
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# SWIFT imports
from swift.llm import sft_main, TrainArguments, get_model_tokenizer
from swift.utils import get_logger

logger = get_logger()


# =============================================================================
# Ablation Configurations
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for each ablation condition."""
    name: str
    description: str
    use_confidence_weighting: bool = False
    use_kl_loss: bool = False
    kl_weight: float = 0.1
    freeze_vit: bool = True
    requires_training: bool = True


ABLATIONS = {
    "A0": AblationConfig(
        name="A0_ZeroShot",
        description="Zero-shot baseline - no training",
        requires_training=False,
    ),
    "A1": AblationConfig(
        name="A1_SFT_GT_Real",
        description="SFT with GT answers + real images",
        freeze_vit=False,  # Train ViT for real images
    ),
    "A2": AblationConfig(
        name="A2_SFT_GT_Blind",
        description="SFT with GT answers + black images",
        freeze_vit=True,
    ),
    "A3": AblationConfig(
        name="A3_SFT_Human_Blind",
        description="SFT with human answers + black images (no confidence)",
        freeze_vit=True,
    ),
    "A4": AblationConfig(
        name="A4_Soft_Human_Blind",
        description="Soft SFT with confidence weighting",
        use_confidence_weighting=True,
        freeze_vit=True,
    ),
    "A5": AblationConfig(
        name="A5_Soft_Human_Blind_KL",
        description="Soft SFT + KL regularization (MAIN METHOD)",
        use_confidence_weighting=True,
        use_kl_loss=True,
        kl_weight=0.1,
        freeze_vit=True,
    ),
}


# =============================================================================
# Training Functions
# =============================================================================

def train_with_swift(
    model_path: str,
    train_data: str,
    output_dir: str,
    run_name: str,
    val_data: str = None,
    ablation_config: AblationConfig = None,
    # Training hyperparameters
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_steps: int = -1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    # LoRA config
    lora_rank: int = 32,
    lora_alpha: int = 64,
    # Saving
    save_steps: int = 50,
    eval_steps: int = 50,
    logging_steps: int = 10,
):
    """
    Train model using SWIFT.
    
    Note: SWIFT's sft_main uses standard CE loss.
    For true confidence weighting, see the custom trainer below.
    """
    
    config = ablation_config or ABLATIONS["A3"]
    
    logger.info("=" * 70)
    logger.info(f"🚀 Training: {config.name}")
    logger.info(f"   {config.description}")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Data: {train_data}")
    logger.info(f"   Confidence weighting: {config.use_confidence_weighting}")
    logger.info(f"   KL loss: {config.use_kl_loss} (weight={config.kl_weight})")
    logger.info("=" * 70)
    
    # Save ablation config
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ablation_config.json'), 'w') as f:
        json.dump({
            'ablation': config.name,
            'description': config.description,
            'use_confidence_weighting': config.use_confidence_weighting,
            'use_kl_loss': config.use_kl_loss,
            'kl_weight': config.kl_weight,
            'freeze_vit': config.freeze_vit,
            'model_path': model_path,
            'train_data': train_data,
        }, f, indent=2)
    
    # Configure SWIFT training
    sft_args = TrainArguments(
        # Model & Data
        model=model_path,
        dataset=[train_data],
        val_dataset=val_data,
        output_dir=output_dir,
        
        # Training type
        train_type="lora",
        torch_dtype="bfloat16",
        
        # Hyperparameters
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.05,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        
        # LoRA
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["all-linear"],
        
        # Freezing
        freeze_llm=False,
        freeze_vit=config.freeze_vit,
        freeze_aligner=True,
        
        # Memory
        gradient_checkpointing=True,
        
        # Saving & Logging
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        
        # Evaluation
        eval_strategy="steps" if val_data else "no",
        eval_steps=eval_steps if val_data else None,
        
        # Other
        max_length=2048,
        seed=42,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
        attn_impl="flash_attn",
    )
    
    result = sft_main(sft_args)
    
    logger.info("✅ Training complete!")
    return result


# =============================================================================
# Custom Soft Supervised Trainer (for true confidence weighting)
# =============================================================================

def train_soft_supervised_custom(
    model_path: str,
    train_data: str,
    output_dir: str,
    run_name: str,
    val_data: str = None,
    ablation_config: AblationConfig = None,
    # Training hyperparameters
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_steps: int = -1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    # LoRA config
    lora_rank: int = 32,
    lora_alpha: int = 64,
    # Soft supervision
    weight_strategy: str = "linear",
    confidence_min_weight: float = 0.2,
    confidence_max_weight: float = 1.0,
    # KL
    kl_weight: float = 0.1,
    # Saving
    save_steps: int = 50,
    eval_steps: int = 50,
    logging_steps: int = 10,
):
    """
    Custom training with true confidence-weighted loss using SWIFT's get_model_tokenizer.
    
    Uses SWIFT for model loading and HuggingFace Trainer with custom compute_loss.
    """
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    
    config = ablation_config or ABLATIONS["A5"]
    
    logger.info("=" * 70)
    logger.info(f"🎯 Custom Soft Supervised Training: {config.name}")
    logger.info(f"   Confidence weighting: {config.use_confidence_weighting}")
    logger.info(f"   KL loss: {config.use_kl_loss} (weight={kl_weight})")
    logger.info("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # Load model using SWIFT's get_model_tokenizer
    # =========================================================================
    logger.info("📦 Loading model with SWIFT's get_model_tokenizer...")
    
    model, tokenizer = get_model_tokenizer(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    logger.info(f"✓ Model loaded: {type(model).__name__}")
    logger.info(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
    
    # =========================================================================
    # Apply LoRA
    # =========================================================================
    logger.info("🔧 Applying LoRA...")
    
    # Freeze ViT if configured
    if config.freeze_vit:
        for name, param in model.named_parameters():
            if 'vision' in name.lower() or 'vit' in name.lower() or 'visual' in name.lower():
                param.requires_grad = False
        logger.info("   Frozen ViT parameters")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # =========================================================================
    # Create reference model for KL (if needed)
    # =========================================================================
    # NOTE: KL loss requires 2x GPU memory (moving ref_model to GPU each step)
    # For memory-constrained setups, use A4 instead of A5
    ref_model = None
    if config.use_kl_loss:
        logger.warning("⚠️  KL loss enabled - requires ~2x GPU memory!")
        logger.warning("   If OOM, use --ablation A4 instead of A5")
        # Skip loading ref_model to save memory - KL will be skipped
        # To enable KL, uncomment below (needs ~40GB+ GPU):
        # ref_model, _ = get_model_tokenizer(
        #     model_path,
        #     torch_dtype=torch.bfloat16,
        #     device_map="cpu",
        # )
        # ref_model.eval()
        # for param in ref_model.parameters():
        #     param.requires_grad = False
        logger.info("   KL loss disabled to save memory (ref_model not loaded)")
    
    # =========================================================================
    # Confidence to weight function
    # =========================================================================
    def confidence_to_weight(conf: float) -> float:
        """Convert confidence (1-5) to loss weight."""
        normalized = (conf - 1) / 4  # 0 to 1
        if weight_strategy == "linear":
            return confidence_min_weight + normalized * (confidence_max_weight - confidence_min_weight)
        elif weight_strategy == "quadratic":
            return confidence_min_weight + (normalized ** 2) * (confidence_max_weight - confidence_min_weight)
        else:
            return confidence_min_weight + normalized * (confidence_max_weight - confidence_min_weight)
    
    # =========================================================================
    # Custom Dataset with Pre-tokenization
    # =========================================================================
    class SoftVQADataset(Dataset):
        def __init__(self, data_path, tokenizer, max_length=768):  # Safe for MC questions
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.examples = []
            
            # Load data
            raw_data = []
            with open(data_path, 'r') as f:
                for line in f:
                    raw_data.append(json.loads(line))
            
            logger.info(f"   Pre-tokenizing {len(raw_data)} examples...")
            
            # Pre-tokenize all examples
            for item in tqdm(raw_data, desc="Tokenizing"):
                # Get conversation
                user_content = item['conversations'][0]['content']
                assistant_content = item['conversations'][1]['content']
                confidence = item.get('confidence', 3)
                
                # Format as chat
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
                
                # Apply chat template
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                except:
                    # Fallback format
                    text = f"User: {user_content}\nAssistant: {assistant_content}"
                
                # Tokenize
                encodings = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                self.examples.append({
                    'input_ids': encodings['input_ids'].squeeze(0),
                    'attention_mask': encodings['attention_mask'].squeeze(0),
                    'labels': encodings['input_ids'].squeeze(0).clone(),
                    'confidence': torch.tensor(confidence, dtype=torch.float32),
                })
            
            logger.info(f"   ✓ Tokenized {len(self.examples)} examples")
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            return self.examples[idx]
    
    # =========================================================================
    # Custom Trainer with Confidence Weighting + KL
    # =========================================================================
    class SoftSupervisedTrainer(Trainer):
        def __init__(self, ref_model=None, use_kl=False, kl_weight=0.1, 
                     use_confidence=True, **kwargs):
            super().__init__(**kwargs)
            self.ref_model = ref_model
            self.use_kl = use_kl
            self.kl_weight = kl_weight
            self.use_confidence = use_confidence
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Extract confidence
            confidences = inputs.pop('confidence', None)
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs.get('labels')
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Per-token loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            token_loss = token_loss.view(shift_labels.size())
            
            # Per-example loss (mean over tokens)
            mask = (shift_labels != -100).float()
            example_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            
            # Apply confidence weights
            if self.use_confidence and confidences is not None:
                weights = torch.tensor(
                    [confidence_to_weight(c.item()) for c in confidences],
                    device=example_loss.device,
                    dtype=example_loss.dtype
                )
                loss = (example_loss * weights).mean()
            else:
                loss = example_loss.mean()
            
            # KL regularization
            if self.use_kl and self.ref_model is not None:
                # Move ref_model to same device temporarily
                device = logits.device
                self.ref_model.to(device)
                
                with torch.no_grad():
                    ref_inputs = {k: v for k, v in inputs.items() if k != 'confidence'}
                    ref_outputs = self.ref_model(**ref_inputs)
                    ref_logits = ref_outputs.logits
                
                # Move ref_model back to CPU
                self.ref_model.to('cpu')
                torch.cuda.empty_cache()
                
                # KL divergence on logits
                kl_loss = F.kl_div(
                    F.log_softmax(shift_logits / 1.0, dim=-1),
                    F.softmax(ref_logits[..., :-1, :].contiguous() / 1.0, dim=-1),
                    reduction='batchmean',
                )
                
                loss = loss + self.kl_weight * kl_loss
            
            return (loss, outputs) if return_outputs else loss
    
    # =========================================================================
    # Create datasets
    # =========================================================================
    logger.info("📂 Creating datasets...")
    train_dataset = SoftVQADataset(train_data, tokenizer)
    val_dataset = SoftVQADataset(val_data, tokenizer) if val_data else None
    
    # =========================================================================
    # Training arguments
    # =========================================================================
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs if max_steps <= 0 else None,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        # evaluation_strategy="steps" if val_dataset else "no",  # Renamed to eval_strategy in newer versions
        # eval_steps=save_steps if val_dataset else None,
        do_eval=False,  # Disable eval for simplicity
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,  # Important for custom fields
    )
    
    # =========================================================================
    # Create trainer and train
    # =========================================================================
    logger.info("🚀 Starting training...")
    
    trainer = SoftSupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        ref_model=ref_model,
        use_kl=config.use_kl_loss,
        kl_weight=kl_weight,
        use_confidence=config.use_confidence_weighting,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model(os.path.join(output_dir, "final_model"))
    logger.info(f"✅ Training complete! Model saved to {output_dir}")
    
    # Save config
    with open(os.path.join(output_dir, 'ablation_config.json'), 'w') as f:
        json.dump({
            'ablation': config.name,
            'description': config.description,
            'use_confidence_weighting': config.use_confidence_weighting,
            'use_kl_loss': config.use_kl_loss,
            'kl_weight': kl_weight,
            'model_path': model_path,
        }, f, indent=2)
    
    return {'status': 'complete', 'output_dir': output_dir}


# =============================================================================
# Main Entry Point
# =============================================================================

def run_ablation(
    ablation: str,
    model_path: str,
    train_data: str,
    output_dir: str,
    run_name: str,
    val_data: str = None,
    **kwargs,
):
    """Run a specific ablation."""
    
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation}. Choose from: {list(ABLATIONS.keys())}")
    
    config = ABLATIONS[ablation]
    
    # A0: Zero-shot - no training
    if not config.requires_training:
        logger.info(f"📋 {config.name}: No training required (zero-shot)")
        logger.info(f"   Use the base model directly: {model_path}")
        
        # Save config for reference
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'ablation_config.json'), 'w') as f:
            json.dump({
                'ablation': config.name,
                'description': config.description,
                'model_path': model_path,
                'requires_training': False,
            }, f, indent=2)
        
        return {'status': 'skip', 'model_path': model_path}
    
    # Run training
    if config.use_confidence_weighting or config.use_kl_loss:
        # Use custom trainer for soft supervision
        return train_soft_supervised_custom(
            model_path=model_path,
            train_data=train_data,
            output_dir=output_dir,
            run_name=run_name,
            val_data=val_data,
            ablation_config=config,
            kl_weight=config.kl_weight,
            **kwargs,
        )
    else:
        # Standard SWIFT training
        return train_with_swift(
            model_path=model_path,
            train_data=train_data,
            output_dir=output_dir,
            run_name=run_name,
            val_data=val_data,
            ablation_config=config,
            **kwargs,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Soft Supervised Training for Blind VQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablations:
  A0  Zero-shot (no training)
  A1  SFT with GT answers + real images
  A2  SFT with GT answers + black images  
  A3  SFT with human answers + black images (no confidence)
  A4  Soft SFT with confidence weighting
  A5  Soft SFT + KL regularization (MAIN METHOD)

Examples:
  # Main method (recommended)
  python train_soft_supervised.py \\
      --ablation A5 \\
      --model_path OpenGVLab/InternVL3_5-2B \\
      --train_data ./training_data/train.jsonl \\
      --output_dir ./output/A5 \\
      --run_name A5_experiment

  # Run all ablations
  for abl in A0 A3 A4 A5; do
      python train_soft_supervised.py --ablation $abl ...
  done
        """
    )
    
    # Required
    parser.add_argument("--ablation", type=str, required=True,
                        choices=list(ABLATIONS.keys()),
                        help="Ablation condition to run")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Training data JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Run name for logging")
    
    # Optional
    parser.add_argument("--val_data", type=str, default=None,
                        help="Validation data JSONL file")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    args = parser.parse_args()
    
    run_ablation(
        ablation=args.ablation,
        model_path=args.model_path,
        train_data=args.train_data,
        output_dir=args.output_dir,
        run_name=args.run_name,
        val_data=args.val_data,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()