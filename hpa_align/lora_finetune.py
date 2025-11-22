
"""
LoRA fine-tuning entry for HPA with PEFT (or eval-only mode).

Usage:
python lora_finetune.py --human human.jsonl --model_id meta-llama/Llama-3-8b-instruct --out out_dir --mode peft
python lora_finetune.py --human human.jsonl --model_id meta-llama/Llama-3-8b-instruct --out out_dir --mode eval
"""
import argparse, os
from typing import List
from datasets import build_items, split
from trainer import HPAListwiseScorer, train_epoch
from utils import set_seed, device
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--model_id", required=True, help="HF model id (causal LM)")
    ap.add_argument("--mode", choices=["peft","eval"], default="peft")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--loss_mode", choices=["js","ce"], default="js")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt_template", default="Q: {q}\nA:")
    args = ap.parse_args()

    set_seed(123)
    os.makedirs(args.out, exist_ok=True)

    # Lazy import to avoid heavy deps if user only wants eval
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else None)
    model.to(device())

    # PEFT optional
    if args.mode == "peft":
        from peft import LoraConfig, get_peft_model, TaskType
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            bias="none", task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj","v_proj","k_proj","o_proj"]  # adjust per model
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Build data
    items = build_items(args.human)
    train, valid, test = split(items)

    # Scorer
    scorer = HPAListwiseScorer(tok, model, prompt_template=args.prompt_template)

    # Optimizer
    if args.mode == "peft":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            loss = train_epoch(train, scorer, opt, tok, batch_size=args.batch_size, loss_mode=args.loss_mode)
            print(f"[ep {ep+1}] train loss: {loss:.4f}")
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
    else:
        print("Eval-only mode selected: skipping fine-tuning. You can run inference with run_infer_llm.py")

if __name__ == "__main__":
    main()
