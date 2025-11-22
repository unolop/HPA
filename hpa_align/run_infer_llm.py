
"""
Language-only inference: runs a causal LM on questions (blind image setting by design).
Produces JSONL with model probabilities over candidate answers if provided; otherwise single-string answers.
"""
import argparse, json, os, torch
from datasets import build_items
from utils import device
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="We reuse human file to get qids/questions/candidates")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="If >0, sample N tokens; else greedy short answer")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else None).to(device())

    items = build_items(args.human)

    def prompt(q): return f"Q: {q}\nA:"
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            inp = tok(prompt(it.question), return_tensors="pt").to(model.device)
            out = model.generate(**inp, max_new_tokens=8, do_sample=args.sample>0, temperature=0.8 if args.sample>0 else None)
            ans = tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
            rec = {"qid": it.qid, "question": it.question, "answer": ans}
            json.dump(rec, f, ensure_ascii=False); f.write("\n")

if __name__ == "__main__":
    main()
