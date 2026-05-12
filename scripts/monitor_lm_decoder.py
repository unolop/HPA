#!/usr/bin/env python3
"""
Compare lm_decoder outputs vs blind VLM outputs for a given model.
Checks answer agreement, distribution similarity, and flags anomalies.
Usage: python scripts/monitor_lm_decoder.py <model_short_name> [condition]
  e.g. python scripts/monitor_lm_decoder.py llava-1.5-7b-hf _control_blind
"""
import sys, json, os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
LM_DIR   = ROOT / "evaluation/logits/lm_decoder/pretrained"
VLM_DIR  = ROOT / "evaluation/logits/vlm/pretrained"

def load_jsonl(path):
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows

def top_answers(rows, field="question", n=8):
    answers = [r["generated_answers"].get(field, "?") for r in rows]
    return Counter(answers).most_common(n)

def agreement_rate(rows_a, rows_b, field="question"):
    by_id = {r["question_id"]: r["generated_answers"].get(field, "") for r in rows_b}
    matches = sum(
        1 for r in rows_a
        if r["generated_answers"].get(field, "") == by_id.get(r["question_id"], "__MISSING__")
    )
    return matches / len(rows_a) if rows_a else 0

def avg_answer_len(rows, field="question"):
    lens = [len(r["generated_answers"].get(field, "").split()) for r in rows]
    return sum(lens) / len(lens) if lens else 0

def long_answers(rows, field="question", threshold=5):
    return [r["generated_answers"].get(field, "") for r in rows
            if len(r["generated_answers"].get(field, "").split()) > threshold]

def report(model, condition="_control_blind"):
    lm_path  = LM_DIR  / model / f"vqa_1k{condition}.jsonl"
    vlm_path = VLM_DIR / model / f"vqa_1k{condition}.jsonl"

    lm_rows  = load_jsonl(lm_path)
    vlm_rows = load_jsonl(vlm_path)

    print(f"\n{'='*60}")
    print(f"Model:     {model}")
    print(f"Condition: {condition}")
    print(f"{'='*60}")

    if lm_rows is None:
        print(f"  [MISSING] lm_decoder output not found: {lm_path}")
        return
    if vlm_rows is None:
        print(f"  [MISSING] VLM blind output not found: {vlm_path}")
        return

    print(f"  lm_decoder rows: {len(lm_rows)}  |  VLM rows: {len(vlm_rows)}")

    # Agreement
    agree = agreement_rate(lm_rows, vlm_rows)
    print(f"\n  Answer agreement (lm == vlm):  {agree:.1%}")

    # Avg answer length — flag if lm_decoder is producing long outputs
    lm_len  = avg_answer_len(lm_rows)
    vlm_len = avg_answer_len(vlm_rows)
    flag = "  ⚠ lm_decoder answers are longer than expected" if lm_len > 3 else ""
    print(f"  Avg answer length  lm={lm_len:.2f} words  vlm={vlm_len:.2f} words{flag}")

    # Long answer examples
    long = long_answers(lm_rows)
    if long:
        print(f"\n  ⚠ Long answers in lm_decoder ({len(long)} instances):")
        for a in long[:5]:
            print(f"     • {a!r}")

    # Top answer distributions
    print(f"\n  Top answers — lm_decoder:")
    for ans, cnt in top_answers(lm_rows):
        bar = "█" * int(cnt / len(lm_rows) * 40)
        print(f"    {ans!r:25s} {cnt:4d} ({cnt/len(lm_rows):.1%}) {bar}")

    print(f"\n  Top answers — VLM blind:")
    for ans, cnt in top_answers(vlm_rows):
        bar = "█" * int(cnt / len(vlm_rows) * 40)
        print(f"    {ans!r:25s} {cnt:4d} ({cnt/len(vlm_rows):.1%}) {bar}")

    # Check for abstention tokens
    abstain_tokens = ["none", "nothing", "unknown", "unanswerable", "no image",
                      "cannot", "can't", "unable", "n/a", "not visible", "not shown"]
    lm_abstain  = sum(1 for r in lm_rows  if any(t in r["generated_answers"].get("question","").lower() for t in abstain_tokens))
    vlm_abstain = sum(1 for r in vlm_rows if any(t in r["generated_answers"].get("question","").lower() for t in abstain_tokens))
    print(f"\n  Soft abstentions  lm={lm_abstain} ({lm_abstain/len(lm_rows):.1%})  vlm={vlm_abstain} ({vlm_abstain/len(vlm_rows):.1%})")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else None
    conds = [sys.argv[2]] if len(sys.argv) > 2 else ["_control_blind", "_control_inst_blind"]

    if model:
        for c in conds:
            report(model, c)
    else:
        # Report all models that have lm_decoder output
        for model_dir in sorted(LM_DIR.iterdir()):
            if model_dir.is_dir():
                for c in conds:
                    report(model_dir.name, c)
