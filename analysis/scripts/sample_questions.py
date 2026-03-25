"""
sample_questions.py — Sample VQA questions for human study.

Balances by entity type (ent) and operator (op) with equal group quotas
for between-group statistical comparisons. Prioritizes "interesting" questions
based on model blind-inference signals:

  Priority 1 — Tier A: overconfident wrong in blind (canonical hallucination)
  Priority 2 — Tier B: answer flips across control variants (linguistically sensitive)
  Priority 3 — Tier C: consensus wrong across models (hard for models, possibly easy for humans)
  Priority 4 — high inst_blind flip rate (blind→inst answer change = prior unlocked by instruction)
  Priority 5 — in previous human study (374 questions from analysis/csv/human_vqa.csv)
  Priority 6 — remaining questions, higher blind accuracy first (easier = more human engagement)


Usage:
    python analysis/scripts/sample_questions.py --n 100 --seed 42 \
        --output analysis/csv/human_study_sample.csv

Output CSV columns:
    question_id, image_id, question_text, ent, op, w,
    blind_acc, inst_blind_acc, flip_rate, mean_conf_blind,
    tier_A, tier_B, tier_C, tiers_str,
    is_prev_study, priority, selection_reason
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
SEMANTICS = REPO / "dataset/vqa/vqa1k_semantics.jsonl"
SCORED_DIR = REPO / "evaluation/scored/pretrained"
LOGITS_DIR = REPO / "evaluation/logits/pretrained"
HUMAN_CSV = REPO / "analysis/csv/human_vqa.csv"
TIER_CSV = REPO / "analysis/csv/sampled_control_questions.csv"
DEFAULT_OUT = REPO / "analysis/csv/human_study_sample.csv"

# Models with reliable blind scores for vqa_1k
MODELS = [
    "InternVL3_5-2B",
    "InternVL3_5-4B",
    "InternVL3_5-8B",
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "llava-1.5-7b-hf",
    "llava-v1.6-mistral-7b-hf",
    "llava-v1.6-vicuna-7b-hf",
]

# Op groups for balancing (merge rare ops into broader buckets)
OP_MERGE = {
    "attr": "attr",
    "ident": "ident",
    "count": "count",
    "exist": "exist",
    "spat": "spat",
    "act": "act",
    "text": "text_op",
    "know": "know",
    "comp": "other_op",
    "cause": "other_op",
    "temp": "other_op",
    "other": "other_op",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_semantics():
    records = {}
    with open(SEMANTICS) as f:
        for line in f:
            d = json.loads(line)
            records[d["question_id"]] = d
    return records


def load_scored(condition="vqa_1k_blind"):
    """Return dict[question_id] -> {model: correct_score}."""
    scores = defaultdict(dict)
    for model in MODELS:
        # Two naming conventions observed in the repo
        candidates = [
            SCORED_DIR / model / f"{model}_{condition}.jsonl",
            SCORED_DIR / model / f"{condition}.jsonl",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            print(f"  [warn] missing scored file: {model}/{condition}")
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                qid = d["question_id"]
                scores[qid][model] = float(d.get("correct", 0.0))
    return scores


def load_first_token_logprob(condition="vqa_1k_control_blind"):
    """
    Extract first-token logprob from control logits files (blind condition).
    Returns dict[question_id] -> mean logprob across models.
    Only models that have control blind logits are used.
    """
    logprobs = defaultdict(list)
    logits_models = [
        "Qwen3-VL-8B-Instruct",
        "llava-1.5-7b-hf",
        "llava-v1.6-mistral-7b-hf",
        "llava-v1.6-vicuna-7b-hf",
    ]
    for model in logits_models:
        path = LOGITS_DIR / model / f"{condition}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                qid = d["question_id"]
                # The 'question' key in generated_logits holds original question logits
                gl = d.get("generated_logits", {})
                q_logits = gl.get("question", {})
                content = q_logits.get("content", [])
                if content:
                    lp = content[0]["logprob"]
                    logprobs[qid].append(lp)
    return {qid: sum(v) / len(v) for qid, v in logprobs.items() if v}


def load_tiers():
    """Return dict[question_id] -> {'tier_A': bool, 'tier_B': bool, 'tier_C': bool, 'tiers_str': str}."""
    df = pd.read_csv(TIER_CSV)
    result = {}
    for _, row in df.iterrows():
        qid = int(row["question_id"])
        tiers_str = str(row.get("tiers", ""))
        result[qid] = {
            "tier_A": "A_overconfident" in tiers_str,
            "tier_B": "B_answer_flip" in tiers_str,
            "tier_C": "C_consensus_wrong" in tiers_str,
            "tiers_str": tiers_str,
        }
    return result


def load_prev_study_qids():
    df = pd.read_csv(HUMAN_CSV)
    return set(df["question_id"].unique().tolist())


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------

PRIORITY_LABELS = {
    1: "tier_A_overconfident_wrong",
    2: "tier_B_answer_flip",
    3: "tier_C_consensus_wrong",
    4: "high_inst_flip_rate",
    5: "prev_study_continuity",
    6: "fill_balanced",
}


def compute_priority(qid, tiers, flip_rate, is_prev_study, flip_threshold=0.5):
    if tiers.get("tier_A", False):
        return 1
    if tiers.get("tier_B", False):
        return 2
    if tiers.get("tier_C", False):
        return 3
    if flip_rate >= flip_threshold:
        return 4
    if is_prev_study:
        return 5
    return 6


# ---------------------------------------------------------------------------
# Quota allocation — equal quotas per ent group, remainder distributed
# ---------------------------------------------------------------------------

def allocate_quotas(ent_counts: dict, n: int) -> dict:
    """
    Give each ent group an equal quota where possible. Groups smaller than the
    ideal quota are capped at their pool size; freed slots are redistributed
    iteratively to groups with remaining capacity. Maximizes total while keeping
    uncapped groups as equal as possible.
    """
    groups = sorted(ent_counts.keys())
    quotas = {g: 0 for g in groups}
    remaining_n = n
    uncapped = set(groups)

    while remaining_n > 0 and uncapped:
        base = remaining_n // len(uncapped)
        if base == 0:
            # Distribute 1 extra each to largest remaining uncapped groups
            sorted_unc = sorted(uncapped, key=lambda g: ent_counts[g], reverse=True)
            for g in sorted_unc:
                if remaining_n <= 0:
                    break
                quotas[g] += 1
                remaining_n -= 1
            break

        newly_capped = []
        for g in list(uncapped):
            ideal = base + (1 if remaining_n % len(uncapped) > 0 else 0)  # rough upper bound
            if ent_counts[g] <= base:
                # This group is smaller than ideal — cap it
                freed = base - (ent_counts[g] - quotas[g])
                quotas[g] = ent_counts[g]
                remaining_n -= (ent_counts[g] - sum([quotas[g]]))  # already set above
                newly_capped.append(g)

        if not newly_capped:
            # No more caps — distribute evenly with remainder
            r = remaining_n % len(uncapped)
            sorted_unc = sorted(uncapped, key=lambda g: ent_counts[g], reverse=True)
            for g in sorted_unc:
                quota = base + (1 if r > 0 else 0)
                quota = min(quota, ent_counts[g] - quotas[g])
                quotas[g] += quota
                remaining_n -= quota
                if r > 0:
                    r -= 1
            break
        else:
            # Recompute remaining_n correctly
            remaining_n = n - sum(quotas.values())
            uncapped -= set(newly_capped)

    return quotas


# ---------------------------------------------------------------------------
# Main sampling
# ---------------------------------------------------------------------------

def build_dataframe(semantics, blind_scores, inst_scores, logprobs, tiers, prev_qids):
    rows = []
    for qid, sem in semantics.items():
        blind_per_model = blind_scores.get(qid, {})
        inst_per_model = inst_scores.get(qid, {})

        n_blind = len(blind_per_model)
        blind_acc = sum(blind_per_model.values()) / n_blind if n_blind else float("nan")

        n_inst = len(inst_per_model)
        inst_acc = sum(inst_per_model.values()) / n_inst if n_inst else float("nan")

        # Flip rate: fraction of models that change their answer blind→inst_blind
        # Approximated as |inst_acc - blind_acc| weighted by disagreement direction
        flip_count = 0
        shared = set(blind_per_model) & set(inst_per_model)
        for m in shared:
            if blind_per_model[m] != inst_per_model[m]:
                flip_count += 1
        flip_rate = flip_count / len(shared) if shared else float("nan")

        mean_conf = logprobs.get(qid, float("nan"))

        tier_info = tiers.get(qid, {"tier_A": False, "tier_B": False, "tier_C": False, "tiers_str": ""})
        is_prev = qid in prev_qids

        priority = compute_priority(qid, tier_info, flip_rate if not math.isnan(flip_rate) else 0.0, is_prev)

        rows.append({
            "question_id": qid,
            "image_id": sem["image_id"],
            "question_text": sem["question"],
            "ent": sem["ent"],
            "op": OP_MERGE.get(sem["op"], sem["op"]),
            "op_raw": sem["op"],
            "w": sem.get("w", ""),
            "blind_acc": round(blind_acc, 4) if not math.isnan(blind_acc) else None,
            "inst_blind_acc": round(inst_acc, 4) if not math.isnan(inst_acc) else None,
            "flip_rate": round(flip_rate, 4) if not math.isnan(flip_rate) else None,
            "mean_conf_blind": round(mean_conf, 4) if not math.isnan(mean_conf) else None,
            "tier_A": tier_info["tier_A"],
            "tier_B": tier_info["tier_B"],
            "tier_C": tier_info["tier_C"],
            "tiers_str": tier_info["tiers_str"],
            "is_prev_study": is_prev,
            "priority": priority,
            "selection_reason": PRIORITY_LABELS[priority],
        })

    return pd.DataFrame(rows)


def sample_questions(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)

    ent_groups = df["ent"].unique().tolist()
    ent_counts = df.groupby("ent").size().to_dict()
    quotas = allocate_quotas(ent_counts, n)

    print(f"\nQuota allocation (total={sum(quotas.values())}):")
    for g in sorted(quotas):
        print(f"  {g:10s}: {quotas[g]:3d}  (pool={ent_counts[g]})")

    selected = []

    for ent in ent_groups:
        quota = quotas[ent]
        group = df[df["ent"] == ent].copy()

        # Sort by priority (ascending) then blind_acc (descending for fill)
        group = group.sort_values(
            ["priority", "blind_acc"],
            ascending=[True, False],
            na_position="last",
        )

        # Op balancing: within each priority level, try to cover op types evenly
        picked_ids = _op_balanced_pick(group, quota, rng)
        selected.append(df[df["question_id"].isin(picked_ids)])

    result = pd.concat(selected, ignore_index=True)

    # Final shuffle for presentation
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    return result


def _op_balanced_pick(group: pd.DataFrame, quota: int, rng: random.Random) -> list:
    """
    Pick `quota` question IDs from `group` while:
      1. Respecting priority order (lower = better)
      2. Spreading across op types as evenly as possible
    """
    if len(group) <= quota:
        return group["question_id"].tolist()

    # Process by priority tier: fill op slots in order
    op_types = group["op"].unique().tolist()
    per_op = max(1, quota // len(op_types))
    op_slots = {op: per_op for op in op_types}
    remainder = quota - per_op * len(op_types)

    # Give extra slots to ops with most questions
    op_sizes = group.groupby("op").size().sort_values(ascending=False)
    for op in op_sizes.index:
        if remainder <= 0:
            break
        op_slots[op] += 1
        remainder -= 1

    picked = []

    for priority_level in sorted(group["priority"].unique()):
        plevel = group[group["priority"] == priority_level]
        for op in op_types:
            needed = op_slots.get(op, 0) - sum(
                1 for qid in picked
                if group.loc[group["question_id"] == qid, "op"].values[0] == op
            )
            if needed <= 0:
                continue
            candidates = plevel[plevel["op"] == op]["question_id"].tolist()
            rng.shuffle(candidates)
            take = candidates[:needed]
            picked.extend(take)
            op_slots[op] -= len(take)

        if len(picked) >= quota:
            break

    # Fill any remaining slots from leftover questions
    if len(picked) < quota:
        already = set(picked)
        remaining = group[~group["question_id"].isin(already)]["question_id"].tolist()
        rng.shuffle(remaining)
        picked.extend(remaining[: quota - len(picked)])

    return picked[:quota]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sample VQA questions for human study")
    parser.add_argument("--n", type=int, default=100, help="Total questions to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT),
                        help="Output CSV path")
    parser.add_argument("--flip-threshold", type=float, default=0.5,
                        help="Minimum flip rate for priority 4 (default: 0.5)")
    args = parser.parse_args()

    print("Loading data...")
    semantics = load_semantics()
    print(f"  Semantics: {len(semantics)} questions")

    blind_scores = load_scored("vqa_1k_blind")
    inst_scores = load_scored("vqa_1k_inst_blind")
    print(f"  Blind scores: {len(blind_scores)} questions covered")

    logprobs = load_first_token_logprob("vqa_1k_control_blind")
    print(f"  Logprobs: {len(logprobs)} questions covered")

    tiers = load_tiers()
    print(f"  Tier labels: {len(tiers)} questions")

    prev_qids = load_prev_study_qids()
    print(f"  Previous study: {len(prev_qids)} questions")

    print("\nBuilding feature table...")
    df = build_dataframe(semantics, blind_scores, inst_scores, logprobs, tiers, prev_qids)

    # Summary stats
    print(f"\nPriority distribution (all {len(df)} questions):")
    for p, label in PRIORITY_LABELS.items():
        cnt = (df["priority"] == p).sum()
        print(f"  P{p} ({label}): {cnt}")

    print(f"\nSampling {args.n} questions (seed={args.seed})...")
    sampled = sample_questions(df, args.n, args.seed)

    # Print summary of selected sample
    print(f"\nSelected {len(sampled)} questions:")
    print("\nBy ent:")
    print(sampled.groupby("ent").size().sort_values(ascending=False).to_string())
    print("\nBy op (merged):")
    print(sampled.groupby("op").size().sort_values(ascending=False).to_string())
    print("\nBy priority:")
    print(sampled.groupby("selection_reason").size().sort_values(ascending=False).to_string())
    print(f"\nPrev study overlap: {sampled['is_prev_study'].sum()} questions")
    print(f"Has tier label: {(sampled['tiers_str'] != '').sum()} questions")
    print(f"Mean blind acc: {sampled['blind_acc'].mean():.3f}")
    print(f"Mean flip rate: {sampled['flip_rate'].mean():.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
