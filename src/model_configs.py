#!/usr/bin/env python3
"""
Model-specific training configuration for InternVL and Llava models.

This script provides model-specific configurations that differ from QwenVL.
"""

import os
from typing import Dict, Any

def get_model_config(model_path: str) -> Dict[str, Any]:
    """
    Get model-specific configuration.

    Args:
        model_path: HuggingFace model path

    Returns:
        Dict with model-specific training parameters
    """
    model_name = model_path.lower()

    # Base configuration (works for QwenVL)
    base_config = {
        "freeze_vit": True,
        "freeze_aligner": True,
        "freeze_llm": False,
        "max_pixels": 448,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "target_modules": ["all-linear"],
        "learning_rate": 2e-5,
        "gradient_checkpointing": True,
    }

    # InternVL-specific configuration
    if "internvl" in model_name:
        return {
            **base_config,
            # InternVL uses different vision encoder
            "freeze_vit": True,
            "freeze_aligner": True,  # Keep projector frozen

            # InternVL-specific LoRA targets
            # Target only LLM layers, not vision encoder
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj",     # MLP
            ],

            # InternVL works better with higher resolution
            "max_pixels": 448 * 448,  # or 672*672 for larger models

            # Slightly lower LR for stability
            "learning_rate": 1e-5,

            # InternVL-specific settings
            "lora_rank": 16,  # Increase rank for better capacity
            "lora_alpha": 32,

            # Vision encoder settings
            "vision_encoder_name": "InternViT",  # For reference
            "use_flash_attn": True,  # If available
        }

    # Llava-specific configuration
    elif "llava" in model_name:
        return {
            **base_config,
            # Llava uses CLIP vision encoder
            "freeze_vit": True,
            "freeze_aligner": True,  # mm_projector

            # Llava-specific LoRA targets
            # Only target LLM (Mistral/Vicuna base)
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],

            # Llava typically uses 336x336 or 672x672
            "max_pixels": 336 * 336,

            # Llava-specific settings
            "learning_rate": 2e-5,
            "lora_rank": 8,
            "lora_alpha": 16,

            # Image processing settings
            "image_aspect_ratio": "pad",  # or "square"
            "vision_encoder_name": "CLIP",
        }

    # Default for QwenVL and others
    else:
        return base_config


def get_data_format_config(model_path: str) -> Dict[str, Any]:
    """
    Get data format configuration for different models.

    Some models require different conversation templates or image tokens.
    """
    model_name = model_path.lower()

    if "internvl" in model_name:
        return {
            "image_token": "<image>",
            "conversation_template": "internvl",  # If Swift supports it
            "requires_system_message": False,
        }

    elif "llava" in model_name:
        return {
            "image_token": "<image>",
            "conversation_template": "llava",  # or "vicuna" depending on base
            "requires_system_message": False,
        }

    else:  # QwenVL
        return {
            "image_token": "<image>",
            "conversation_template": "qwen",
            "requires_system_message": False,
        }


def print_model_info(model_path: str):
    """Print model-specific training recommendations."""
    config = get_model_config(model_path)
    data_config = get_data_format_config(model_path)

    print(f"\n{'='*80}")
    print(f"Training Configuration for: {model_path}")
    print(f"{'='*80}\n")

    print("Model Architecture Settings:")
    print(f"  Vision Encoder: {config.get('vision_encoder_name', 'Default')}")
    print(f"  Freeze ViT: {config['freeze_vit']}")
    print(f"  Freeze Aligner: {config['freeze_aligner']}")
    print(f"  Freeze LLM: {config['freeze_llm']}")

    print("\nLoRA Settings:")
    print(f"  Rank: {config['lora_rank']}")
    print(f"  Alpha: {config['lora_alpha']}")
    print(f"  Dropout: {config['lora_dropout']}")
    print(f"  Target modules: {', '.join(config['target_modules'][:3])}...")

    print("\nTraining Hyperparameters:")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Max Pixels: {config['max_pixels']}")
    print(f"  Gradient Checkpointing: {config['gradient_checkpointing']}")

    print("\nData Format:")
    print(f"  Image Token: {data_config['image_token']}")
    print(f"  Conv Template: {data_config.get('conversation_template', 'default')}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                       help="HuggingFace model path")
    args = parser.parse_args()

    print_model_info(args.model_path)

    # Export config as JSON
    import json
    config = get_model_config(args.model_path)

    output_file = f"config_{args.model_path.split('/')[-1]}.json"
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Saved configuration to: {output_file}")
